"""Training task for the deletion-only edit flow: `Pretraining` plus per-epoch deletion metrics."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from hydra.utils import instantiate
from torch import nn

from editflows import DataBatch
from editflows.coupling import CoupledBatch
from editflows.lr_scheduler import WarmUpCosineGroups
from editflows.metrics import RANK_COLUMNS, Buckets, DeletionCounts, write_metrics
from editflows.pretraining import Pretraining

log = logging.getLogger(__name__)

OPTIM_DEFAULTS = {"decay_matrices_only": False, "encoder_lr_scale": 1.0, "encoder_freeze_steps": 0, "encoder_ramp_steps": 0}


def to_device(batch: DataBatch, device: torch.device) -> DataBatch:
    """Copy of a (coupled) batch on `device`, without the cached index tensors."""
    if isinstance(batch, CoupledBatch):
        return CoupledBatch(seq_lens=batch.seq_lens, x=batch.x.to(device), inserted=batch.inserted.to(device))
    return DataBatch(seq_lens=batch.seq_lens, x=batch.x.to(device))


def initialize(
    model: nn.Module,
    *,
    head_log_rates: list[float] | None = None,
    head_weight_scale: float = 0.0,
    time_embedding_zero: bool = False,
    trunk: str = "default",
    encoder_checkpoint: str | None = None,
    checkpoint: str | None = None,
):
    """Weight initialisation of a freshly built model. Only values change: the modules, their shapes and the
    state-dict keys stay as they are.

    `head_log_rates = [insert, substitute, delete]`: the rate head's last layer gets the bias that produces
    these log-rates, so training starts from chosen rates instead of rates around e^0 = 1 for all three
    operations. Its weights are multiplied by `head_weight_scale`: 0 starts every token and time at exactly
    these rates, 1 keeps the default weights and with them a head that responds to its input from the first
    step. `time_embedding_zero`: the time embedding starts at zero.
    `trunk="scaled"`: the output projections of the encoder's residual branches start smaller by
    1 / sqrt(2 * layers), as in GPT-2. `encoder_checkpoint`: the encoder starts from a checkpoint of
    `editflows.encoder_pretraining` (or of another run of this task). `checkpoint`: the whole model starts
    from the weights another run of this task was evaluated with (its EMA weights), a warm start with a fresh
    optimizer and learning-rate schedule.
    """
    head = getattr(model, "head", model)
    with torch.no_grad():
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            prefix = "ema_model.ema_model." if any(k.startswith("ema_model.ema_model.") for k in state) else "model."
            model.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
        if encoder_checkpoint is not None:
            state = torch.load(encoder_checkpoint, map_location="cpu", weights_only=False)["state_dict"]
            prefix = "encoder." if any(k.startswith("encoder.") for k in state) else "model.encoder."
            model.encoder.load_state_dict({k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)})
        if trunk == "scaled":
            trunk_model = model.encoder.model
            std = trunk_model.config.initializer_range / math.sqrt(2 * len(trunk_model.layers))
            for layer in trunk_model.layers:
                nn.init.normal_(layer.self_attn.o_proj.weight, std=std)
                nn.init.normal_(layer.mlp.down_proj.weight, std=std)
        elif trunk != "default":
            raise ValueError(f"unknown trunk initialisation {trunk!r}")
        if head_log_rates is not None:
            last = head.lambda_head[-1]
            target = torch.tensor(head_log_rates, dtype=last.bias.dtype) / head.max_log_lambda
            assert target.shape == last.bias.shape and bool((target.abs() < 1).all()), head_log_rates
            last.weight.mul_(head_weight_scale)
            last.bias.copy_(torch.atanh(target))
        if time_embedding_zero:
            last = head.t_emb[-1][-1]
            last.weight.zero_()
            last.bias.zero_()


class DeletionTask(Pretraining):
    """Edit-flow training on the extended coupling with the deletion confusion matrix every epoch.

    `eval` settings: `n_sample_seqs` sequences of the validation / test split are sampled end to end with
    `sample_steps` Euler steps; `target_len_bins` and `fraction_bins` define the buckets. Results go to the
    loggers (`<stage>/del_f1`, ...) and to `<run dir>/eval/`.

    `init` is passed to `initialize` once, when the model is first built. `optim` splits the parameters into
    groups (see `OPTIM_DEFAULTS`); with the defaults the optimizer is the plain single-group one.
    """

    def __init__(self, *, eval: dict, init: dict | None = None, optim: dict | None = None, **kwargs):
        super().__init__(**kwargs)

        self.eval_config = dict(eval)
        self.init_config = {k: v for k, v in dict(init or {}).items() if v not in (None, False, "default")}
        self.optim_config = {**OPTIM_DEFAULTS, **dict(optim or {})}
        self.buckets = Buckets(self.eval_config["target_len_bins"], self.eval_config["fraction_bins"])
        self._rank_counts: DeletionCounts | None = None
        self._grad_norm_max = 0.0

    def configure_model(self):
        fresh = self.model is None
        super().configure_model()
        if fresh and self.init_config:
            initialize(self.model, **self.init_config)
            self.ema_model.copy_params_from_model_to_ema()

    def configure_optimizers(self):  # type: ignore
        # Epoch-based training leaves trainer.max_steps at -1; schedules that need the total number of
        # steps get Lightning's estimate instead.
        config = self.lr_scheduler_config
        if config is not None and "max_steps" in config and (config.max_steps is None or config.max_steps <= 0):
            config.max_steps = int(self.trainer.estimated_stepping_batches)
        if self.optim_config == OPTIM_DEFAULTS:
            return super().configure_optimizers()
        return self._grouped_optimizers()

    def _grouped_optimizers(self):
        """Parameter groups by (encoder or not) x (weight decay or not), with a delayed start for the encoder."""
        options, base = self.optim_config, self.optimizer_config
        groups: dict[tuple[bool, bool], list] = {}
        for name, p in self.model.named_parameters():
            decays = p.ndim >= 2 or not options["decay_matrices_only"]
            groups.setdefault((name.startswith("encoder."), decays), []).append(p)
        # The first group is the reference of the schedule: not the encoder, with weight decay
        keys = sorted(groups, key=lambda k: (k[0], not k[1]))
        param_groups = [
            {
                "params": groups[k],
                "lr": base.lr * (options["encoder_lr_scale"] if k[0] else 1.0),
                "weight_decay": base.get("weight_decay", 0.0) if k[1] else 0.0,
            }
            for k in keys
        ]
        opt = instantiate(base, param_groups)

        config = self.lr_scheduler_config
        assert config is not None and config._target_.endswith("WarmUpCosineAnnealing"), (
            "parameter groups are implemented for the warm-up + cosine schedule"
        )
        scheduler = WarmUpCosineGroups(
            opt,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            start_lr=config.start_lr,
            end_lr=config.end_lr,
            delays=[options["encoder_freeze_steps"] if k[0] else 0 for k in keys],
            ramps=[options["encoder_ramp_steps"] if k[0] else 0 for k in keys],
        )
        return {"optimizer": opt, "lr_scheduler": {"interval": "step", "frequency": 1, "scheduler": scheduler}}

    def on_before_optimizer_step(self, optimizer):
        # Gradient norm before clipping: how far the configured clip value is from the gradients it clips,
        # and how much of the gradient reaches the encoder.
        encoder = torch.zeros((), device=self.device)
        rest = torch.zeros((), device=self.device)
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                squared = p.grad.detach().float().pow(2).sum()
                if name.startswith("encoder."):
                    encoder = encoder + squared
                else:
                    rest = rest + squared
        norm = float((encoder + rest).sqrt())
        self._grad_norm_max = max(self._grad_norm_max, norm)
        self.log("train/grad_norm", norm)
        self.log("train/grad_norm_max", self._grad_norm_max)
        self.log("train/grad_encoder_share", float(encoder / (encoder + rest).clamp(min=1e-30)))
        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self._grad_norm_max = 0.0

    # ---- validation / test epochs --------------------------------------------------------------

    def _eval_epoch_start(self):
        self.trainer.datamodule.use_eval_generator()
        self._rank_counts = DeletionCounts(self.buckets, device=self.device)

    def _eval_step(self, stage: str, batch, generator: torch.Generator):
        out = self.eval_step(stage, batch, generator=generator)
        if self._rank_counts is not None:
            x_0, x_1, _, _ = batch
            rate = self.eval_model(x_0, torch.zeros(x_0.batch_size, device=self.device))
            self._rank_counts.update_ranking(rate.log_lambda_del, x_0, self.buckets.index_pair(x_0, x_1))
        return out

    def _eval_epoch_end(self, stage: str, dataset, generator: torch.Generator):
        self.trainer.datamodule.restore_train_generator()
        if self.trainer.sanity_checking:
            return
        counts = self._sampling_eval(dataset, generator)
        counts.all_reduce()
        if self._rank_counts is not None:
            self._rank_counts.all_reduce()
            counts.counts[:, list(RANK_COLUMNS)] = self._rank_counts.counts[:, list(RANK_COLUMNS)]
        self._report(stage, counts)

    def _sampling_eval(self, dataset, generator: torch.Generator) -> DeletionCounts:
        """Sample from the sources of a fixed subset of `dataset` (split across processes) with provenance."""
        config = self.eval_config
        n = min(config["n_sample_seqs"], len(dataset))
        indices = list(range(n))[self.trainer.global_rank :: self.trainer.world_size]
        batch_size = self.trainer.datamodule.eval_batch_size
        # Same couplings every epoch: fixed CPU generator, one stream per process
        coupling_generator = torch.Generator().manual_seed(config.get("seed", 1234) + self.trainer.global_rank)

        counts = DeletionCounts(self.buckets, device=self.device)
        for start in range(0, len(indices), batch_size):
            x_1 = DataBatch.from_sequences([dataset[i] for i in indices[start : start + batch_size]])
            x_0, x_1, _, _ = self.create_aligned_pair(x_1, coupling_generator)
            x_0, x_1 = to_device(x_0, self.device), to_device(x_1, self.device)
            x_0.origin = torch.arange(len(x_0.x), device=self.device)
            x_hat = self.edit_flow.sample_euler(
                self.eval_model, x_0=x_0, n_steps=config["sample_steps"], generator=generator
            )
            counts.update_sampling(x_0, x_1, x_hat, self.buckets.index_pair(x_0, x_1))
        return counts

    def _report(self, stage: str, counts: DeletionCounts):
        summary = counts.summary()
        for name, value in summary.items():
            self.log(f"{stage}/{name}", value, prog_bar=name in ("del_precision", "del_recall", "del_f1"))
        for (target, fraction), bucket in counts.per_bucket().items():
            for name in ("del_precision", "del_recall", "del_f1"):
                self.log(f"{stage}/len{target}/frac{fraction}/{name}", bucket[name])

        if self.trainer.is_global_zero:
            epoch, step = self.current_epoch, self.global_step
            write_metrics(Path(self.trainer.default_root_dir) / "eval", counts, stage=stage, epoch=epoch, step=step)
            log.info(
                f"{stage} epoch {epoch} step {step}: "
                + ", ".join(f"{k}={v:.3f}" for k, v in summary.items())
            )

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._eval_epoch_start()

    def validation_step(self, batch, batch_idx):
        return self._eval_step("val", batch, self.val_generator)

    def on_validation_epoch_end(self):
        self._eval_epoch_end("val", self.trainer.datamodule.val_data, self.val_generator)

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self._eval_epoch_start()

    def test_step(self, batch, batch_idx):
        return self._eval_step("test", batch, self.test_generator)

    def on_test_epoch_end(self):
        self._eval_epoch_end("test", self.trainer.datamodule.test_data, self.test_generator)
