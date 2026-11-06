#!/usr/bin/env python

import faulthandler
import logging
import os
import socket
import warnings

import sys
from datetime import timedelta
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from submitit import JobEnvironment

from editflows.utils import (
    filter_device_available,
    log_hyperparameters,
    print_config,
    print_exceptions,
    set_seed,
)
from editflows.utils.callbacks import ConfigInCheckpoint, TokenizerInCheckpoint

# Log to traceback to stderr on segfault
faulthandler.enable(all_threads=False)

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings(
    "ignore", r"The '\w+_dataloader' does not have many workers", module="lightning"
)
warnings.filterwarnings(
    "ignore", "The `srun` command is available on your system but is not used", module="lightning"
)
logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(filter_device_available)

# Let's us do arithmetic (and anything else) in configurations
OmegaConf.register_new_resolver("eval", eval)

log = logging.getLogger("train")


def store_job_info(config: DictConfig):
    host = socket.gethostname()
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    process_id = os.getpid()

    with open_dict(config):
        config.host = host
        config.process_id = process_id
        if array_job_id is not None and array_task_id is not None:
            config.slurm_job_id = f"{array_job_id}_{array_task_id}"
        elif job_id is not None:
            config.slurm_job_id = job_id


def wandb_config(config: DictConfig) -> DictConfig | None:
    """Return the WandbLogger config node within config.logging."""

    logging = config.logging
    if "_target_" in logging:
        return logging if logging._target_.endswith("WandbLogger") else None
    for cfg in logging.values():
        if "_target_" in cfg and cfg._target_.endswith("WandbLogger"):
            return cfg
    return None


def store_wandb_run_id(run_id: str):
    """Store the w&b run ID on disk, so that we can resume the run after a timeout."""

    if "SUBMITIT_FOLDER" not in os.environ:
        return

    (JobEnvironment().paths.folder / "wandb_run_id").write_text(run_id)


def restore_resumed_wandb_run_id(config: DictConfig) -> bool:
    """If this is requeued/resumed run, restore the w&b run ID in the config."""

    if "SUBMITIT_FOLDER" not in os.environ:
        return False

    wandb_cfg = wandb_config(config)
    if wandb_cfg is None:
        return False

    run_id_file = JobEnvironment().paths.folder / "wandb_run_id"
    if not run_id_file.is_file():
        return False

    with open_dict(wandb_cfg):
        wandb_cfg.id = run_id_file.read_text()

    return True


def resume_path(config: DictConfig, ckpt_dir: Path) -> str | None:
    """An explicit `from_ckpt` wins (`none` forces a fresh start); otherwise a run continues from the
    newest checkpoint in its own run directory."""

    if config.from_ckpt is not None:
        return None if str(config.from_ckpt).lower() == "none" else str(config.from_ckpt)
    candidates = [p for p in (ckpt_dir / "last.ckpt", ckpt_dir / "periodic.ckpt") if p.is_file()]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def get_callbacks(config, ckpt_dir):
    monitor = {"monitor": config.checkpoint.monitor, "mode": config.checkpoint.mode}
    callbacks = [
        # Best model by the monitored metric plus last.ckpt, written after every validation epoch
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            filename="best-{epoch}-{step}",
            **monitor,
        ),
        # periodic.ckpt every few minutes of training, so that a job killed by a time limit loses little
        ModelCheckpoint(
            dirpath=ckpt_dir,
            train_time_interval=timedelta(minutes=config.checkpoint.time_interval_min),
            save_top_k=1,
            filename="periodic",
            enable_version_counter=False,
        ),
        TQDMProgressBar(refresh_rate=1 if sys.stdout.isatty() else 50),
        LearningRateMonitor(),
        ConfigInCheckpoint(config),
        TokenizerInCheckpoint(),
    ]
    return callbacks


@hydra.main(config_path="config", config_name="train", version_base=None)
@print_exceptions
def main(config: DictConfig):
    assert config.seed is not None, "Specify a seed to ensure correctness in multi-GPU training"

    seed_sequence = set_seed(config)

    # Log host and slurm job ID
    store_job_info(config)

    # Log to same W&B ID if this is a requeued run after a timeout
    restore_resumed_wandb_run_id(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)

    # Everything a run writes goes to its run directory
    run_dir = Path(config.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    if rank_zero_only.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, run_dir / "config.yaml")
        handler = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(handler)

    if "_target_" in config.logging:
        loggers = [instantiate(config.logging)]
    else:
        loggers = list(instantiate(config.logging).values())
    wandb_logger = next((l for l in loggers if isinstance(l, WandbLogger)), None)

    # Initialize the loggers (e.g. w&b) here to log outputs from setting up the
    # datamodule etc.
    for logger in loggers:
        logger.experiment

    # Store the W&B run ID on disk so that we can resume the run after a timeout
    if wandb_logger is not None and wandb_logger.version is not None:
        store_wandb_run_id(wandb_logger.version)

    torch.set_float32_matmul_precision(config.matmul_precision)

    log.info("Loading data")
    datamodule = instantiate(config.data, seed_sequence=seed_sequence.spawn(1)[0])

    log.info("Instantiating task")
    task = instantiate(config.task, seed_sequence=seed_sequence.spawn(1)[0])

    log_hyperparameters(loggers, config, task)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, ckpt_dir)
    # Under SLURM, lightning's auto-requeue would write a checkpoint from inside a signal handler. Jobs
    # are resumed from the periodic checkpoints by resubmitting them instead.
    plugins = [SLURMEnvironment(auto_requeue=False)] if SLURMEnvironment.detect() else []
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(run_dir),
        plugins=plugins,
    )

    log.info("Starting training!")
    ckpt_path = resume_path(config, ckpt_dir)
    if ckpt_path is not None:
        log.info(f"Resuming from {ckpt_path}")
    trainer.fit(task, datamodule=datamodule, ckpt_path=ckpt_path)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    if wandb_logger is not None:
        import wandb

        wandb.finish()

    checkpoint = trainer.checkpoint_callback
    assert isinstance(checkpoint, ModelCheckpoint)
    log.info(f"Best checkpoint path:\n{checkpoint.best_model_path}")

    best_score = checkpoint.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
