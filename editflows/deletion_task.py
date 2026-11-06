"""Training task for the deletion-only edit flow: `Pretraining` plus per-epoch deletion metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from editflows import DataBatch
from editflows.coupling import CoupledBatch
from editflows.metrics import RANK_COLUMNS, Buckets, DeletionCounts, write_metrics
from editflows.pretraining import Pretraining

log = logging.getLogger(__name__)


def to_device(batch: DataBatch, device: torch.device) -> DataBatch:
    """Copy of a (coupled) batch on `device`, without the cached index tensors."""
    if isinstance(batch, CoupledBatch):
        return CoupledBatch(seq_lens=batch.seq_lens, x=batch.x.to(device), inserted=batch.inserted.to(device))
    return DataBatch(seq_lens=batch.seq_lens, x=batch.x.to(device))


class DeletionTask(Pretraining):
    """Edit-flow training on the extended coupling with the deletion confusion matrix every epoch.

    `eval` settings: `n_sample_seqs` sequences of the validation / test split are sampled end to end with
    `sample_steps` Euler steps; `target_len_bins` and `fraction_bins` define the buckets. Results go to the
    loggers (`<stage>/del_f1`, ...) and to `<run dir>/eval/`.
    """

    def __init__(self, *, eval: dict, **kwargs):
        super().__init__(**kwargs)

        self.eval_config = dict(eval)
        self.buckets = Buckets(self.eval_config["target_len_bins"], self.eval_config["fraction_bins"])
        self._rank_counts: DeletionCounts | None = None

    def configure_optimizers(self):  # type: ignore
        # Epoch-based training leaves trainer.max_steps at -1; schedules that need the total number of
        # steps get Lightning's estimate instead.
        config = self.lr_scheduler_config
        if config is not None and "max_steps" in config and (config.max_steps is None or config.max_steps <= 0):
            config.max_steps = int(self.trainer.estimated_stepping_batches)
        return super().configure_optimizers()

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
