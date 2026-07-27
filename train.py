#!/usr/bin/env python

import faulthandler
import logging
import os
import socket
import warnings

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
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


def get_callbacks(config, ckpt_dir):
    monitor = {"monitor": "val/loss", "mode": "min"}
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_last=True,
            save_top_k=1,
            every_n_epochs=1,
            **monitor,
        ),
        TQDMProgressBar(refresh_rate=1),
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
    resumed = restore_resumed_wandb_run_id(config)

    # Resolve interpolations to work around a bug:
    # https://github.com/omry/omegaconf/issues/862
    OmegaConf.resolve(config)
    print_config(config)

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

    primary_logger = wandb_logger if wandb_logger is not None else loggers[0]
    ckpt_dir = f"checkpoints/{primary_logger.name}/{primary_logger.version}"

    torch.set_float32_matmul_precision(config.matmul_precision)

    log.info("Loading data")
    datamodule = instantiate(config.data, seed_sequence=seed_sequence.spawn(1)[0])

    log.info("Instantiating task")
    task = instantiate(config.task, seed_sequence=seed_sequence.spawn(1)[0])

    log_hyperparameters(loggers, config, task)

    log.info("Instantiating trainer")
    callbacks = get_callbacks(config, ckpt_dir)
    trainer = Trainer(
        **config.trainer,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=ckpt_dir,
    )

    log.info("Starting training!")
    fit_kwargs = {}
    if resumed:
        fit_kwargs["ckpt_path"] = "hpc"
    elif config.from_ckpt is not None:
        fit_kwargs["ckpt_path"] = config.from_ckpt
    trainer.fit(task, datamodule=datamodule, **fit_kwargs)

    if config.eval_testset:
        log.info("Starting testing!")
        trainer.test(ckpt_path="best", datamodule=datamodule)

    wandb.finish()

    checkpoint = trainer.checkpoint_callback
    assert isinstance(checkpoint, ModelCheckpoint)
    log.info(f"Best checkpoint path:\n{checkpoint.best_model_path}")

    best_score = checkpoint.best_model_score
    return float(best_score) if best_score is not None else None


if __name__ == "__main__":
    main()
