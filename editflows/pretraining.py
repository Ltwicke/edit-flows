import logging

import numpy as np
import torch
from hydra.utils import instantiate
from jaxtyping import Int
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch import Tensor, nn

from editflows import CubicKappaSchedule, DataBatch, EditFlow, TokenWiseMixturePath
from editflows.ema_pytorch import EMA

log = logging.getLogger(__name__)


class Plots(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        assert isinstance(pl_module, Pretraining)
        if not trainer.is_global_zero:
            return

        # Log samples to w&b
        logger = next(
            (logger for logger in trainer.loggers if isinstance(logger, WandbLogger)), None
        )
        if logger is None:
            return

        generator = torch.Generator(pl_module.device).manual_seed(89013)
        batch_size = 3
        x_0 = pl_module.p_noise.sample(batch_size, generator=generator)
        x_flows = pl_module.edit_flow.sample_flow_euler(
            pl_module.eval_model, x_0=x_0, n_steps=50, generator=generator
        )

        tokenizer = trainer.datamodule.tokenizer
        data = [
            [i]
            + [tokenizer.decode(x_t.numpy(force=True).tolist()) for x_t in x_ts.split_sequences()]
            for i, x_ts in enumerate(x_flows)
        ]

        if logger is not None:
            logger.log_table(
                "val/samples", ["Step"] + [f"Sample {i + 1}" for i in range(batch_size)], data
            )


def create_ema(model, beta=0.9999, update_after_step=100, update_every=10, **kwargs):
    return EMA(
        model,
        beta=beta,
        update_after_step=update_after_step,
        update_every=update_every,
        include_online_model=False,
        use_foreach=True,
    )


class Pretraining(LightningModule):
    def __init__(
        self,
        *,
        model: dict,
        noise: dict,
        coupling: dict,
        aligner: dict,
        path: dict,
        ema: dict,
        vocab_size: int,
        low_discrepancy_sampling: bool,
        optimizer: dict,
        compile: bool,
        seed_sequence: np.random.SeedSequence,
        lr_scheduler: dict | None = None,
    ):
        super().__init__()

        self.model = None
        self.model_config = OmegaConf.create(model)
        self.noise_config = OmegaConf.create(noise)
        self.coupling_config = OmegaConf.create(coupling)
        self.aligner_config = OmegaConf.create(aligner)
        self.path_config = OmegaConf.create(path)
        self.ema_config = OmegaConf.create(ema)
        self.vocab_size = vocab_size
        self.low_discrepancy_sampling = low_discrepancy_sampling

        self.optimizer_config = OmegaConf.create(optimizer)
        self.lr_scheduler_config = (
            OmegaConf.create(lr_scheduler) if lr_scheduler is not None else None
        )
        self.should_compile = compile

        self.train_seed_sequence, self.val_seed_sequence, self.test_seed_sequence = (
            seed_sequence.spawn(3)
        )

    def configure_model(self):
        datamodule = self.trainer.datamodule
        datamodule.batch_transform = self.create_aligned_pair

        self.p_noise = instantiate(self.noise_config, bos_token=datamodule.bos_token)
        self.coupling = instantiate(self.coupling_config)
        if "ShiftAligner" in self.aligner_config._target_:
            aligner_kwargs = {"bos_token": datamodule.bos_token}
        else:
            aligner_kwargs = {}
        self.aligner = instantiate(self.aligner_config, **aligner_kwargs)

        path = instantiate(self.path_config)
        self.edit_flow = EditFlow(
            probability_path=path,
            bos_token=datamodule.bos_token,
            low_discrepancy_sampling=self.low_discrepancy_sampling,
        )

        if self.model is not None:
            return

        model = instantiate(self.model_config)
        assert isinstance(model, nn.Module)
        self.model = model
        self.ema_model = create_ema(model, **self.ema_config)

        if self.should_compile:
            # Compiling the backwards pass with SDPA and nested jagged tensors
            # is bugged as of July 24, 2026:
            # https://github.com/pytorch/pytorch/issues/155421
            # self.train_model = torch.compile(self.model)
            log.info("Compilation disabled because of an autograd bug with SDPA and NJTs")
            self.train_model = self.model

            self.eval_model = torch.compile(self.ema_model)
        else:
            self.train_model = self.model
            self.eval_model = self.ema_model

    def create_aligned_pair(
        self, x_1: DataBatch, generator: torch.Generator
    ) -> tuple[
        DataBatch,
        DataBatch,
        Int[Tensor, "batch_size aug_seq_len"],
        Int[Tensor, "batch_size aug_seq_len"],
    ]:
        """Create a pair of aligned sequences for training.

        This runs in the data worker on CPU.
        """

        # Draw x_0 and x_1 from the coupling
        x_0 = self.p_noise.sample(x_1.batch_size, generator=generator)
        x_0, x_1 = self.coupling.couple(x_0, x_1, generator=generator)
        # Match x_0 and x_1 by finding representations in augmented space
        z_0, z_1 = self.aligner.align(x_0, x_1)

        return x_0, x_1, z_0, z_1

    def _generator(self, seed_sequence: np.random.SeedSequence) -> torch.Generator:
        """Generate a unique random generator for each process.

        Note that the returned generator is the same every time for the same seed sequence. This
        way, we use the same noise for every validation epoch, improving comparability across
        training steps.
        """
        rank = self.trainer.global_rank
        seed = seed_sequence.generate_state(self.trainer.world_size, np.uint64)[rank]
        return torch.Generator(self.device).manual_seed(int(seed))

    def on_train_start(self):
        self.train_generator = self._generator(self.train_seed_sequence)

    def training_step(self, batch, batch_idx):
        x_0, x_1, z_0, z_1 = batch

        per_sample_loss = self.edit_flow.bregman_divergence(
            self.train_model, z_0=z_0, z_1=z_1, generator=self.train_generator
        )
        loss = per_sample_loss.mean()

        self.log("train/loss", loss, batch_size=x_1.batch_size, prog_bar=True)
        return {"loss": loss}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema_model is not None:
            self.ema_model.update()

    def on_validation_epoch_start(self):
        self.val_generator = self._generator(self.val_seed_sequence)

    def validation_step(self, batch, batch_idx):
        return self.eval_step("val", batch, generator=self.val_generator)

    def on_test_epoch_start(self):
        self.test_generator = self._generator(self.test_seed_sequence)

    def test_step(self, batch, batch_idx):
        return self.eval_step("test", batch, generator=self.test_generator)

    def eval_step(self, stage: str, batch, generator: torch.Generator):
        x_0, x_1, z_0, z_1 = batch

        per_sample_loss = self.edit_flow.bregman_divergence(
            self.eval_model, z_0=z_0, z_1=z_1, generator=generator
        )
        loss = per_sample_loss.mean()

        self.log(f"{stage}/loss", loss, batch_size=x_1.batch_size, sync_dist=True)
        return {"loss": loss}

    def configure_optimizers(self):  # type: ignore
        opt = instantiate(self.optimizer_config, self.model.parameters())

        if self.lr_scheduler_config is None:
            return {"optimizer": opt}
        else:
            config = self.lr_scheduler_config.copy()
            config.pop("name")
            scheduler = instantiate(config, opt)
            lr_scheduler_config = {"interval": "step", "frequency": 1, "scheduler": scheduler}
            return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        assert isinstance(callbacks, list)
        if self.trainer.is_global_zero:
            callbacks = callbacks + [Plots()]
        return callbacks
