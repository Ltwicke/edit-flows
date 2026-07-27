from typing import Any

from lightning import Callback
from omegaconf import DictConfig, OmegaConf


class ConfigInCheckpoint(Callback):
    """Save the config in the checkpoint."""

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]):
        checkpoint["config"] = OmegaConf.to_container(self.config, resolve=True)


class TokenizerInCheckpoint(Callback):
    """Save the tokenizer in the checkpoint.

    A SentencePiece tokenizer is result of model training and not deterministic.
    Therefore, it becomes part of the model state and we embed it in the
    checkpoint.
    """

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]):
        tokenizer = getattr(trainer.datamodule, "tokenizer", None)
        if tokenizer is not None:
            checkpoint["tokenizer"] = tokenizer.serialized_model_proto()
