import hashlib
import json
import logging
import os
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.seed import pl_worker_init_function
from loky import get_reusable_executor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from editflows import DataBatch

from .sampler import DistributedNonPaddingSampler, InfiniteRandomSampler

log = logging.getLogger(__name__)


class SubsequenceBOSDataset(Dataset):
    """A dataset of fixed-length sub-sequences of a single long sequence.

    Prepends a BOS token to each sequence.
    """

    def __init__(self, seq: torch.Tensor, n: int, *, bos_token: int | None):
        super().__init__()

        self.seq = seq
        if bos_token is None:
            self.n = n
            self.bos_token = None
        else:
            self.n = n - 1
            self.bos_token = seq.new_tensor([bos_token])

    def __len__(self):
        return len(self.seq) - self.n + 1

    def __getitem__(self, idx: int):
        if self.bos_token is None:
            return self.seq[idx : idx + self.n]
        else:
            return torch.cat((self.bos_token, self.seq[idx : idx + self.n]))


def hash_dict(config: dict) -> str:
    """Compute a deterministic hash from a config dict."""
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.shake_128(config_str.encode("utf-8")).hexdigest(10)


class Text8DataModule(LightningDataModule):
    """The text8 dataset from [1]

    [1] https://mattmahoney.net/dc/textdata.html
    """

    def __init__(
        self,
        root: os.PathLike,
        *,
        seq_len: int,
        vocab_size: int,
        batch_size: int,
        eval_batch_size: int | None,
        num_workers: int,
        pin_memory: bool | None = None,
        seed_sequence: np.random.SeedSequence,
        **kwargs,
    ):
        super().__init__()

        self.root = Path(root)
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or self.batch_size
        self.num_workers = num_workers
        self.seed_sequence = seed_sequence

        self.split_params = {"seq_len": seq_len}
        assert vocab_size <= 2**16, "Tokens have to fit in a uint16"
        self.spm_params = {
            "vocab_size": vocab_size,
            "character_coverage": 1.0,
            "model_type": "bpe",
            "unk_id": 0,
            "bos_id": 1,
            # Disable EOS tokens, because they don't matter for edit flows
            "eos_id": -1,
        }

        self._tokenizer = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_samples_seed = 2271217414
        (
            self.train_worker_seed,
            self.val_worker_seed,
            self.test_worker_seed,
            batch_transform_seed,
        ) = seed_sequence.spawn(4)

        self.batch_transform: Callable[[DataBatch, torch.Generator], Any] | None = None
        self.batch_transform_generator: torch.Generator = torch.Generator().manual_seed(
            int(batch_transform_seed.generate_state(1)[0])
        )

        if pin_memory is None:
            self.pin_memory = torch.cuda.is_available() and self.num_workers > 0
        else:
            self.pin_memory = pin_memory

        if self.pin_memory and self.num_workers == 0:
            log.warning(
                "Memory pinning without data workers creates unnecessary copies. Disabling it."
            )
            self.pin_memory = False

    def prepare_data(self):
        log.info("Creating data directory")
        self.root.mkdir(parents=True, exist_ok=True)

        # Download dataset
        zip_path = self.root / "text8.zip"
        if not zip_path.exists():
            log.info("Downloading text8 dataset")
            urllib.request.urlretrieve("https://mattmahoney.net/dc/text8.zip", zip_path)
        else:
            log.info("text8.zip already exists, skipping download")

        # Unzip
        text_path = self.root / "text8"
        if not text_path.exists():
            log.info("Extracting text8.zip")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extract("text8", self.root)
        else:
            log.info("text8 already extracted, skipping extraction")

        # Split the dataset into train/val/test
        split_paths = self.split_paths
        train_path = split_paths["train"]
        val_path = split_paths["val"]
        test_path = split_paths["test"]
        if not all(p.exists() for p in [train_path, val_path, test_path]):
            log.info("Splitting dataset into train/val/test")
            text = text_path.read_bytes()

            # sentencepiece does not accept sentences/lines longer than 4192, so we break the data
            # into these chunks
            text = [text[i : i + 4192] for i in range(0, len(text), 4192)]

            # Split: 94% train, 1% val, 5% test
            total_len = len(text)
            train_end = int(0.94 * total_len)
            val_end = int(0.95 * total_len)

            train_path.write_bytes(b"\n".join(text[:train_end]))
            val_path.write_bytes(b"\n".join(text[train_end:val_end]))
            test_path.write_bytes(b"\n".join(text[val_end:]))
            log.info("Dataset split completed")
        else:
            log.info("Train/val/test splits already exist, skipping split")

        # Fit a sentencepiece tokenizer to the train set
        tokenizer_prefix = self._tokenizer_prefix
        if not tokenizer_prefix.with_suffix(".model").exists():
            log.info("Training SentencePiece tokenizer")
            spm.SentencePieceTrainer.train(
                input=str(train_path),
                model_prefix=str(tokenizer_prefix),
                **self.spm_params,
            )
            log.info("Tokenizer training completed")
        else:
            log.info("Tokenizer already exists, skipping training")

        for stage in ["train", "val", "test"]:
            token_path = self.token_paths[stage]
            if token_path.is_file():
                log.info(f"{token_path.name} exists, skip tokenization")
            else:
                split_path = self.split_paths[stage]
                log.info(f"Tokenize {split_path.name} into {token_path.name}")
                sentences = split_path.read_text().splitlines()
                pool = get_reusable_executor()
                tokens = np.concat(
                    [
                        ids
                        for ids in tqdm(
                            pool.map(lambda s: self.tokenizer.encode_as_ids(s), sentences),
                            total=len(sentences),
                        )
                    ]
                )
                np.save(token_path, tokens)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = self._dataset("train")
        if stage in ("fit", "validate"):
            self.val_data = self._dataset("val")
        if stage == "test":
            self.test_data = self._dataset("test")

    def _dataset(self, split: str):
        data = np.load(self.token_paths[split])
        return SubsequenceBOSDataset(
            torch.asarray(data, dtype=torch.uint16), self.seq_len, bos_token=self.bos_token
        )

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.train_samples_seed)
        data = self.train_data
        assert data is not None, "Call `setup(stage='fit')` before creating the dataloader"
        return self._dataloader(
            data,
            batch_size=self.batch_size,
            sampler=InfiniteRandomSampler(data, generator=generator),
            worker_generator=torch.Generator().manual_seed(
                int(self.train_worker_seed.generate_state(1)[0])
            ),
        )

    def val_dataloader(self):
        data = self.val_data
        assert data is not None, "Call `setup(stage='validate')` before creating the dataloader"
        return self._dataloader(
            data,
            batch_size=self.eval_batch_size,
            sampler=DistributedNonPaddingSampler(data),
            worker_generator=torch.Generator().manual_seed(
                int(self.val_worker_seed.generate_state(1)[0])
            ),
        )

    def test_dataloader(self):
        data = self.test_data
        assert data is not None, "Call `setup(stage='test')` before creating the dataloader"
        return self._dataloader(
            data,
            batch_size=self.eval_batch_size,
            sampler=DistributedNonPaddingSampler(data),
            worker_generator=torch.Generator().manual_seed(
                int(self.test_worker_seed.generate_state(1)[0])
            ),
        )

    def _dataloader(self, dataset, *, batch_size: int, sampler, worker_generator: torch.Generator):
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate,
            generator=worker_generator,
            worker_init_fn=self._worker_init_fn,
        )

    def _worker_init_fn(self, worker_id: int):
        # Let lightning derive a high-quality seed from the worker ID
        pl_worker_init_function(worker_id)

        # Initialize the batch generator with the derived seed
        self.batch_transform_generator = torch.Generator().manual_seed(torch.initial_seed())

    def collate(self, samples: list[torch.Tensor]) -> DataBatch | Any:
        batch = DataBatch.from_equilength_batch(torch.stack(samples, dim=0).to(dtype=torch.int32))
        if self.batch_transform is None:
            return batch
        else:
            return self.batch_transform(batch, self.batch_transform_generator)

    @property
    def _tokenizer_prefix(self):
        config_hash = hash_dict({**self.split_params, **self.spm_params})
        return self.root / f"tokenizer-{config_hash}"

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            model_path = self._tokenizer_prefix.with_suffix(".model")
            if not model_path.is_file():
                raise RuntimeError(f"SentencePiece tokenizer {model_path} does not exist")
            self._tokenizer = spm.SentencePieceProcessor(model_file=str(model_path))

        return self._tokenizer

    @property
    def bos_token(self) -> int:
        return self.tokenizer.bos_id()

    @property
    def split_paths(self):
        config_hash = hash_dict(self.split_params)
        return {
            "train": self.root / f"train-{config_hash}.txt",
            "val": self.root / f"val-{config_hash}.txt",
            "test": self.root / f"test-{config_hash}.txt",
        }

    @property
    def token_paths(self):
        config_hash = hash_dict({**self.split_params, **self.spm_params})
        return {
            "train": self.root / f"train-{config_hash}.npy",
            "val": self.root / f"val-{config_hash}.npy",
            "test": self.root / f"test-{config_hash}.npy",
        }
