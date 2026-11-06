import numpy as np
import pytest
import torch

from editflows import DataBatch
from editflows.data.sequences import (
    ALPHABET,
    SequenceFileDataModule,
    encode_sequences,
    read_sequences,
)


@pytest.fixture
def sequences():
    rng = np.random.default_rng(0)
    return ["".join(rng.choice(list(ALPHABET), size=rng.integers(5, 20))) for _ in range(40)]


def write_file(path, seqs, header=True):
    rows = (["sequence"] if header else []) + seqs
    if path.suffix == ".xlsx":
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        for r in rows:
            ws.append([r])
        wb.save(path)
    else:
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize("suffix", [".xlsx", ".csv", ".txt"])
def test_read_formats(tmp_path, sequences, suffix):
    path = tmp_path / f"targets{suffix}"
    write_file(path, sequences)
    assert read_sequences(path) == ["sequence"] + sequences


@pytest.mark.parametrize("header", [True, False])
def test_encode_detects_header(sequences, header):
    rows = (["sequence"] if header else []) + sequences
    tokens, offsets = encode_sequences(rows, ALPHABET)
    assert len(offsets) == len(sequences) + 1
    for i, s in enumerate(sequences):
        ids = tokens[offsets[i] : offsets[i + 1]]
        assert "".join(ALPHABET[j] for j in ids) == s


def test_encode_rejects_invalid_letters(sequences):
    with pytest.raises(ValueError, match="row 3"):
        encode_sequences(sequences[:2] + ["ABCX"] + sequences[2:], ALPHABET)


def make_datamodule(path, seed=0, **kwargs):
    kwargs = {"batch_size": 4, "num_workers": 0, "val_fraction": 0.2, "test_fraction": 0.1, **kwargs}
    dm = SequenceFileDataModule(path, seed_sequence=np.random.SeedSequence(seed), **kwargs)
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")
    return dm


def test_datamodule_split_cache_and_batches(tmp_path, sequences):
    path = tmp_path / "targets.xlsx"
    write_file(path, sequences)
    dm = make_datamodule(path)

    assert dm.bos_token == 20 and dm.vocab_size == 21
    assert len(dm.val_data) == 8 and len(dm.test_data) == 4 and len(dm.train_data) == 28
    assert dm.cache_path.is_file()

    # The split is a function of the file and split_seed only
    other = make_datamodule(path, seed=123)
    assert np.array_equal(dm.train_data.indices, other.train_data.indices)
    assert not np.array_equal(dm.train_data.indices, make_datamodule(path, split_seed=1).train_data.indices)

    # A second datamodule reuses the cache instead of parsing again
    mtime = dm.cache_path.stat().st_mtime_ns
    make_datamodule(path)
    assert dm.cache_path.stat().st_mtime_ns == mtime

    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch, DataBatch)
    assert batch.batch_size == 4
    for seq in batch.split_sequences():
        assert seq[0] == dm.bos_token
        assert (seq[1:] < dm.n_letters).all()
    assert dm.decode(batch.split_sequences()[0].tolist()) in sequences

    # Validation covers every sequence exactly once
    n_val = sum(b.batch_size for b in dm.val_dataloader())
    assert n_val == len(dm.val_data)


def test_batch_transform_and_eval_generator(tmp_path, sequences):
    path = tmp_path / "targets.csv"
    write_file(path, sequences)
    dm = make_datamodule(path)
    dm.batch_transform = lambda batch, generator: (batch, torch.rand((), generator=generator).item())

    first = next(iter(dm.val_dataloader()))[1]
    dm.use_eval_generator()
    a = next(iter(dm.val_dataloader()))[1]
    dm.restore_train_generator()
    dm.use_eval_generator()
    b = next(iter(dm.val_dataloader()))[1]
    dm.restore_train_generator()
    assert a == b, "the evaluation generator is reset to the same state every time"
    assert first != a, "training state is not the evaluation state"


def test_max_len_filter(tmp_path, sequences):
    path = tmp_path / "targets.txt"
    write_file(path, sequences, header=False)
    dm = make_datamodule(path, max_len=10)
    total = len(dm.train_data) + len(dm.val_data) + len(dm.test_data)
    assert total == sum(len(s) <= 10 for s in sequences)
