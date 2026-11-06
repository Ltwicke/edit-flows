"""Target sequences read from a file with one sequence of letters per row (.xlsx, .csv or .txt)."""

import csv
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.fabric.utilities.seed import pl_worker_init_function
from torch import distributed
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from editflows import DataBatch

from .sampler import DistributedNonPaddingSampler

log = logging.getLogger(__name__)

ALPHABET = "ABCDEFGHIJKLMNOPQRST"


def hash_dict(config: dict) -> str:
    """Compute a deterministic hash from a config dict (same scheme as the text8 datamodule)."""
    config_str = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.shake_128(config_str.encode("utf-8")).hexdigest(10)


def read_sequences(path: os.PathLike, *, sheet: int = 0, column: int = 0) -> list[str]:
    """Read the raw sequence strings, one per row, from the first column of an .xlsx sheet, a .csv or a text file."""
    path = Path(path)
    if path.suffix == ".xlsx":
        from openpyxl import load_workbook

        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = workbook.worksheets[sheet].iter_rows(values_only=True)
            seqs = [str(row[column]).strip() for row in rows if len(row) > column and row[column] is not None]
        finally:
            workbook.close()
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            seqs = [row[column].strip() for row in csv.reader(f) if len(row) > column]
    else:
        seqs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [s for s in seqs if s]


def encode_sequences(seqs: list[str], alphabet: str) -> tuple[np.ndarray, np.ndarray]:
    """Map letters to ids 0..len(alphabet)-1.

    Returns the concatenated ids (uint8) and the offsets of the sequences (int64, length n + 1). A first row that
    is not a valid sequence is treated as a header; any other invalid row raises.
    """
    table = np.full(256, -1, dtype=np.int16)
    table[np.frombuffer(alphabet.encode("ascii"), dtype=np.uint8)] = np.arange(len(alphabet))

    encoded = []
    for i, s in enumerate(seqs):
        try:
            ids = table[np.frombuffer(s.encode("ascii"), dtype=np.uint8)]
        except UnicodeEncodeError:
            ids = np.array([-1])
        if (ids < 0).any():
            if i == 0:
                continue  # header row
            raise ValueError(f"row {i + 1} contains letters outside the alphabet {alphabet!r}: {s[:40]!r}")
        encoded.append(ids.astype(np.uint8))

    lengths = np.array([len(e) for e in encoded], dtype=np.int64)
    offsets = np.concatenate([[0], lengths.cumsum()])
    tokens = np.concatenate(encoded) if encoded else np.zeros(0, dtype=np.uint8)
    return tokens, offsets


class SequenceDataset(Dataset):
    """Sequences `indices` of a token store, each prefixed with a BOS token."""

    def __init__(self, tokens: np.ndarray, offsets: np.ndarray, indices: np.ndarray, *, bos_token: int | None):
        super().__init__()

        self.tokens = tokens
        self.offsets = offsets
        self.indices = indices
        self.bos_token = bos_token

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        j = self.indices[idx]
        seq = torch.from_numpy(self.tokens[self.offsets[j] : self.offsets[j + 1]].astype(np.int32))
        if self.bos_token is None:
            return seq
        return torch.cat((seq.new_tensor([self.bos_token]), seq))


def _world() -> tuple[int, int]:
    if distributed.is_available() and distributed.is_initialized():
        return distributed.get_world_size(), distributed.get_rank()
    return 1, 0


class SequenceFileDataModule(LightningDataModule):
    """Known target sequences from a file, split into train/val/test by a fixed permutation.

    Letters are ids 0..n_letters-1, BOS is `n_letters`. The file is parsed once and cached as .npz next to it.
    Training is epoch based (finite loaders), so "epoch" in the logs means one pass over the training split.
    """

    def __init__(
        self,
        path: os.PathLike,
        *,
        alphabet: str = ALPHABET,
        sheet: int = 0,
        column: int = 0,
        max_len: int = 500,
        val_fraction: float = 0.05,
        test_fraction: float = 0.05,
        split_seed: int = 0,
        batch_size: int,
        eval_batch_size: int | None = None,
        num_workers: int,
        pin_memory: bool | None = None,
        seed_sequence: np.random.SeedSequence,
        **kwargs,
    ):
        super().__init__()

        self.path = Path(path)
        self.alphabet = alphabet
        self.sheet = sheet
        self.column = column
        self.max_len = max_len
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.split_seed = split_seed
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.seed_sequence = seed_sequence

        # Other config nodes read these through interpolation (e.g. `${data.vocab_size}`), keep them consistent
        if "vocab_size" in kwargs:
            assert kwargs["vocab_size"] == self.vocab_size, "data.vocab_size must be len(alphabet) + 1 (BOS)"

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
        self._saved_generator_state = None

        if pin_memory is None:
            self.pin_memory = torch.cuda.is_available() and self.num_workers > 0
        else:
            self.pin_memory = pin_memory

    # ---- tokens ------------------------------------------------------------------------------

    @property
    def n_letters(self) -> int:
        return len(self.alphabet)

    @property
    def bos_token(self) -> int:
        return len(self.alphabet)

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet) + 1

    def decode(self, ids: Iterable[int]) -> str:
        """Letters of a token sequence; the BOS token is dropped."""
        return "".join(self.alphabet[i] for i in ids if i < self.n_letters)

    # ---- data ----------------------------------------------------------------------------------

    @property
    def cache_path(self) -> Path:
        stat = self.path.stat()
        key = hash_dict({"size": stat.st_size, "mtime": stat.st_mtime, "alphabet": self.alphabet,
                         "sheet": self.sheet, "column": self.column})
        return self.path.with_name(f"{self.path.name}.{key}.npz")

    def prepare_data(self):
        cache = self.cache_path
        if cache.is_file():
            log.info(f"{cache.name} exists, skip parsing {self.path.name}")
            return
        log.info(f"Parsing {self.path}")
        tokens, offsets = encode_sequences(read_sequences(self.path, sheet=self.sheet, column=self.column),
                                           self.alphabet)
        np.savez(cache, tokens=tokens, offsets=offsets)
        log.info(f"Cached {len(offsets) - 1} sequences in {cache.name}")

    def setup(self, stage: str):
        if self.train_data is not None:
            return
        data = np.load(self.cache_path)
        tokens, offsets = data["tokens"], data["offsets"]
        lengths = np.diff(offsets)
        keep = np.flatnonzero((lengths > 0) & (lengths <= self.max_len))
        if len(keep) < len(lengths):
            log.info(f"Dropping {len(lengths) - len(keep)} sequences longer than {self.max_len} tokens")

        perm = keep[np.random.default_rng(self.split_seed).permutation(len(keep))]
        n_val = round(len(perm) * self.val_fraction)
        n_test = round(len(perm) * self.test_fraction)
        splits = {"val": perm[:n_val], "test": perm[n_val : n_val + n_test], "train": perm[n_val + n_test :]}
        self.train_data, self.val_data, self.test_data = (
            SequenceDataset(tokens, offsets, np.sort(splits[s]), bos_token=self.bos_token)
            for s in ("train", "val", "test")
        )
        log.info(f"Sequences: train {len(self.train_data)}, val {len(self.val_data)}, test {len(self.test_data)}")

    # ---- loaders -------------------------------------------------------------------------------

    def train_dataloader(self):
        data = self.train_data
        assert data is not None, "Call `setup(stage='fit')` before creating the dataloader"
        world_size, rank = _world()
        sampler = DistributedSampler(
            data, num_replicas=world_size, rank=rank, shuffle=True, seed=self.train_samples_seed
        )
        return self._dataloader(
            data,
            batch_size=self.batch_size,
            sampler=sampler,
            worker_generator=torch.Generator().manual_seed(
                int(self.train_worker_seed.generate_state(1)[0])
            ),
            persistent_workers=True,
        )

    def val_dataloader(self):
        data = self.val_data
        assert data is not None, "Call `setup(stage='fit')` before creating the dataloader"
        return self._dataloader(
            data,
            batch_size=self.eval_batch_size,
            sampler=DistributedNonPaddingSampler(data),
            worker_generator=torch.Generator().manual_seed(
                int(self.val_worker_seed.generate_state(1)[0])
            ),
            persistent_workers=False,
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
            persistent_workers=False,
        )

    def _dataloader(self, dataset, *, batch_size, sampler, worker_generator, persistent_workers):
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate,
            generator=worker_generator,
            worker_init_fn=self._worker_init_fn,
            persistent_workers=persistent_workers and self.num_workers > 0,
        )

    def _worker_init_fn(self, worker_id: int):
        # Let lightning derive a high-quality seed from the worker ID
        pl_worker_init_function(worker_id)

        # Initialize the batch generator with the derived seed
        self.batch_transform_generator = torch.Generator().manual_seed(torch.initial_seed())

    def collate(self, samples: list[torch.Tensor]) -> DataBatch | Any:
        batch = DataBatch.from_sequences(samples)
        if self.batch_transform is None:
            return batch
        else:
            return self.batch_transform(batch, self.batch_transform_generator)

    # ---- evaluation without data workers -------------------------------------------------------
    # With workers, every validation epoch re-seeds them identically, so the coupling noise of the validation
    # set is the same every epoch. Without workers the main-process generator is shared with training, so the
    # task switches it to a fixed state for the duration of an evaluation epoch.

    def use_eval_generator(self):
        if self.num_workers == 0:
            self._saved_generator_state = self.batch_transform_generator.get_state()
            self.batch_transform_generator.manual_seed(int(self.val_worker_seed.generate_state(1)[0]))

    def restore_train_generator(self):
        if self._saved_generator_state is not None:
            self.batch_transform_generator.set_state(self._saved_generator_state)
            self._saved_generator_state = None
