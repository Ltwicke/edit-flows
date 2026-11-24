"""Pretraining of the sequence encoder alone: masked-letter prediction on the target sequences.

The deletion task assumes that a sequence encoder is given. This task produces one for the placeholder encoder
the way BERT and the protein language models do: a share of the letters of every target is hidden and the
encoder predicts them from their context. Only the encoder is used afterwards (`task.init.encoder_checkpoint`
of the deletion task); the linear prediction layer is discarded.

The encoder class and its vocabulary do not change. The BOS id doubles as the mask symbol: it never occurs
inside a sequence, so the encoder already has an embedding for it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning import LightningModule
from omegaconf import OmegaConf
from torch import nn

from editflows import DataBatch


class MaskedLetterPretraining(LightningModule):
    """`mask_fraction` of the letters are selected; of those 80 % become the mask symbol, 10 % a random letter and
    10 % stay, and the encoder predicts the original letter at every selected position."""

    def __init__(
        self,
        *,
        model: dict,
        vocab_size: int,
        optimizer: dict,
        seed_sequence: np.random.SeedSequence,
        lr_scheduler: dict | None = None,
        mask_fraction: float = 0.15,
        **unused,  # keys of the deletion task that an experiment file sets as well
    ):
        super().__init__()

        self.encoder = None
        self.encoder_config = OmegaConf.create(model).encoder
        self.optimizer_config = OmegaConf.create(optimizer)
        self.lr_scheduler_config = OmegaConf.create(lr_scheduler) if lr_scheduler is not None else None
        self.mask_fraction = mask_fraction
        self.bos_token = vocab_size - 1
        self.train_seed, self.val_seed = (int(s.generate_state(1)[0]) for s in seed_sequence.spawn(2))

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = instantiate(self.encoder_config)
        self.decoder = nn.Linear(self.encoder.hidden_size, self.bos_token)

    def on_train_start(self):
        self.train_generator = torch.Generator(self.device).manual_seed(self.train_seed)

    def on_validation_epoch_start(self):
        self.val_generator = torch.Generator(self.device).manual_seed(self.val_seed)  # the same masks every epoch

    def _step(self, x_1: DataBatch, generator: torch.Generator):
        tokens = x_1.x
        selected = (torch.rand(len(tokens), device=self.device, generator=generator) < self.mask_fraction) & (
            tokens != self.bos_token
        )
        how = torch.rand(len(tokens), device=self.device, generator=generator)
        random_letters = torch.randint(0, self.bos_token, (len(tokens),), device=self.device, generator=generator)
        corrupted = torch.where(selected & (how < 0.8), torch.full_like(tokens, self.bos_token), tokens)
        corrupted = torch.where(selected & (how >= 0.9), random_letters.to(tokens.dtype), corrupted)

        hidden = self.encoder(DataBatch(seq_lens=x_1.seq_lens, x=corrupted))
        logits = self.decoder(hidden[selected].float())
        target = tokens[selected].long()
        return F.cross_entropy(logits, target), (logits.argmax(dim=-1) == target).float().mean()

    def training_step(self, batch: DataBatch, batch_idx):
        loss, accuracy = self._step(batch, self.train_generator)
        self.log("train/loss", loss, batch_size=batch.batch_size, prog_bar=True)
        self.log("train/accuracy", accuracy, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch: DataBatch, batch_idx):
        loss, accuracy = self._step(batch, self.val_generator)
        self.log("val/loss", loss, batch_size=batch.batch_size, sync_dist=True)
        self.log("val/accuracy", accuracy, batch_size=batch.batch_size, sync_dist=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore
        opt = instantiate(self.optimizer_config, self.parameters())
        if self.lr_scheduler_config is None:
            return {"optimizer": opt}
        config = self.lr_scheduler_config.copy()
        config.pop("name")
        if "max_steps" in config and (config.max_steps is None or config.max_steps <= 0):
            config.max_steps = int(self.trainer.estimated_stepping_batches)
        return {"optimizer": opt, "lr_scheduler": {"interval": "step", "frequency": 1, "scheduler": instantiate(config, opt)}}
