import pytest
import torch

from editflows import DataBatch, EditFlow, OptimalAligner
from editflows.coupling import CoupledBatch, ExtendedCoupling, KnownAligner, NullSource
from editflows.editflows import rm_blanks

BOS = 20
EPS = EditFlow.EPSILON_TOKEN


def targets(n_seqs=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    seqs = []
    for _ in range(n_seqs):
        n = int(torch.randint(20, 60, (), generator=g))
        seqs.append(torch.cat((torch.tensor([BOS]), torch.randint(0, 20, (n,), generator=g))))
    return DataBatch.from_sequences(seqs)


def couple(x_1, original_fraction=(0.4, 0.9), seed=0):
    coupling = ExtendedCoupling(original_fraction=original_fraction, n_letters=20, bos_token=BOS)
    noise = NullSource(bos_token=BOS).sample(x_1.batch_size, generator=torch.Generator())
    assert noise.batch_size == x_1.batch_size and len(noise.x) == 0
    return coupling.couple(noise, x_1, generator=torch.Generator().manual_seed(seed))


def test_source_structure():
    x_1 = targets()
    x_0, x_1_out = couple(x_1)
    assert x_1_out is x_1
    assert isinstance(x_0, CoupledBatch) and x_0.inserted is not None

    for src, ins, tgt in zip(x_0.split_sequences(), torch.split(x_0.inserted, list(x_0.seq_lens)), x_1.split_sequences()):
        assert src[0] == BOS and not ins[0], "BOS stays first and is never an insertion"
        assert torch.equal(src[~ins], tgt), "the original letters survive in order"
        assert (src[ins] < 20).all(), "inserted tokens are letters, never BOS"
        n_1 = len(tgt) - 1
        k = int(ins.sum())
        f = n_1 / (n_1 + k)
        assert 0.4 - 0.02 <= f <= 0.9 + 0.02


def test_fixed_fraction_and_reproducibility():
    x_1 = targets()
    x_0, _ = couple(x_1, original_fraction=(0.5, 0.5))
    for src, tgt in zip(x_0.split_sequences(), x_1.split_sequences()):
        n_1 = len(tgt) - 1
        assert len(src) == 1 + n_1 + round(n_1 * (1 / 0.5 - 1))

    again, _ = couple(x_1, original_fraction=(0.5, 0.5))
    assert torch.equal(again.x, x_0.x) and torch.equal(again.inserted, x_0.inserted)
    other, _ = couple(x_1, original_fraction=(0.5, 0.5), seed=1)
    assert not torch.equal(other.x, x_0.x)


def test_no_insertions_at_fraction_one():
    x_1 = targets()
    x_0, _ = couple(x_1, original_fraction=(1.0, 1.0))
    assert torch.equal(x_0.x, x_1.x) and not x_0.inserted.any()


def test_known_alignment_matches_rmblanks():
    x_1 = targets()
    x_0, _ = couple(x_1)
    z_0, z_1 = KnownAligner().align(x_0, x_1)

    assert torch.equal(rm_blanks(z_0).x, x_0.x)
    assert torch.equal(rm_blanks(z_1).x, x_1.x)
    assert tuple(rm_blanks(z_1).seq_lens) == x_1.seq_lens
    assert int(((z_0 != EPS) & (z_1 == EPS)).sum()) == int(x_0.inserted.sum())
    assert not ((z_0 == EPS) & (z_1 != EPS)).any(), "no insertions"
    assert not ((z_0 != EPS) & (z_1 != EPS) & (z_0 != z_1)).any(), "no substitutions"


def test_known_alignment_agrees_with_needleman_wunsch():
    x_1 = targets(n_seqs=4, seed=3)
    x_0, _ = couple(x_1, seed=3)
    z_0, z_1 = OptimalAligner().align(x_0, x_1)

    deletions = ((z_0 != EPS) & (z_1 == EPS)).sum(dim=1)
    insertions = ((z_0 == EPS) & (z_1 != EPS)).sum(dim=1)
    substitutions = ((z_0 != EPS) & (z_1 != EPS) & (z_0 != z_1)).sum(dim=1)
    k = torch.tensor([int(m.sum()) for m in torch.split(x_0.inserted, list(x_0.seq_lens))])
    assert torch.equal(deletions, k)
    assert not insertions.any() and not substitutions.any()


def test_known_aligner_rejects_plain_batches():
    x_1 = targets()
    with pytest.raises(AssertionError):
        KnownAligner().align(x_1, x_1)
