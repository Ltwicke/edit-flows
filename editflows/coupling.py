"""The extended coupling: the source is the target with random letters inserted.

Under this coupling the only edits that lead from the source $x_0$ to the target $x_1$ are deletions of the
inserted letters, and the alignment in the augmented space is known at construction time, so no alignment
search is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Int
from torch import Tensor

from editflows import DataBatch, EditFlow


@dataclass
class CoupledBatch(DataBatch):
    """A source batch that remembers which of its tokens were inserted."""

    inserted: Bool[Tensor, "n"] | None = None


class NullSource:
    """Stands in for the noise distribution when the coupling derives the source from the target."""

    def __init__(self, *, bos_token: int | None = None):
        pass

    def sample(self, n_seqs: int, *, generator: torch.Generator) -> DataBatch:
        return DataBatch(seq_lens=(0,) * n_seqs, x=torch.zeros(0, dtype=torch.int32))


class ExtendedCoupling:
    r"""Insert uniformly random letters at uniformly random positions of the target.

    For a target of $n_1$ letters, the share of the source that is original, $f \sim U(f_\min, f_\max)$, is drawn
    per sequence and $K = \operatorname{round}(n_1 (1 / f - 1))$ letters are inserted, so that $n_1 / n_0 \approx f$.
    A BOS token at the start of the target stays the first token of the source.
    """

    def __init__(
        self, *, original_fraction: tuple[float, float], n_letters: int, bos_token: int | None = None
    ):
        lo, hi = original_fraction
        assert 0 < lo <= hi <= 1, "original_fraction must be a range within (0, 1]"
        self.original_fraction = (float(lo), float(hi))
        self.n_letters = n_letters
        self.bos_token = bos_token

    def couple(
        self, x_0: DataBatch, x_1: DataBatch, *, generator: torch.Generator
    ) -> tuple[CoupledBatch, DataBatch]:
        """Build the source from `x_1`; `x_0` is ignored."""
        lo, hi = self.original_fraction
        seqs, inserted_masks = [], []
        for seq in x_1.split_sequences():
            has_bos = self.bos_token is not None and len(seq) > 0 and seq[0] == self.bos_token
            body = seq[1:] if has_bos else seq
            n_1 = len(body)

            f = lo + (hi - lo) * torch.rand((), generator=generator).item()
            k = round(n_1 * (1 / f - 1))
            # gap g means "before body[g]"; the i-th inserted token (sorted by gap) lands at position gap_i + i
            gaps = torch.randint(0, n_1 + 1, (k,), generator=generator).sort().values
            positions = gaps + torch.arange(k)
            letters = torch.randint(0, self.n_letters, (k,), generator=generator, dtype=seq.dtype)

            inserted = torch.zeros(n_1 + k, dtype=torch.bool)
            inserted[positions] = True
            source = torch.empty(n_1 + k, dtype=seq.dtype)
            source[positions] = letters
            source[~inserted] = body
            if has_bos:
                source = torch.cat((seq[:1], source))
                inserted = torch.cat((inserted.new_zeros(1), inserted))
            seqs.append(source)
            inserted_masks.append(inserted)

        source_batch = CoupledBatch(
            seq_lens=tuple(len(s) for s in seqs), x=torch.cat(seqs), inserted=torch.cat(inserted_masks)
        )
        return source_batch, x_1


class KnownAligner:
    r"""Alignment for sources built by `ExtendedCoupling`: $z_0 = x_0$ and $z_1$ is $x_0$ with the inserted
    tokens blanked, so $\operatorname{rmblanks}(z_1) = x_1$."""

    def align(
        self, x_0: DataBatch, x_1: DataBatch
    ) -> tuple[Int[Tensor, "batch_size aug_seq_len"], Int[Tensor, "batch_size aug_seq_len"]]:
        assert isinstance(x_0, CoupledBatch) and x_0.inserted is not None, (
            "KnownAligner needs the source of ExtendedCoupling"
        )
        max_len = max(x_0.seq_lens)
        z_0 = x_0.x.new_full((x_0.batch_size, max_len), EditFlow.EPSILON_TOKEN)
        z_0[x_0.seq_idx, x_0.token_pos] = x_0.x
        z_1 = z_0.clone()
        inserted = x_0.inserted
        z_1[x_0.seq_idx[inserted], x_0.token_pos[inserted]] = EditFlow.EPSILON_TOKEN
        return z_0, z_1
