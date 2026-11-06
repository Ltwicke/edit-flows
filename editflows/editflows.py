"""A well-documented and typed PyTorch implementation of Edit Flows by Marten Lienen."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import einops as eo
import numba
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


@dataclass
class DataBatch:
    """A batch of samples in the space of variable length sequences."""

    # Lengths of the sequences in the batch
    seq_lens: tuple[int, ...]

    # Tokens
    x: Int[Tensor, "n"]

    # Cache for on-device tensor of sequence lengths
    _seq_lens_tensor: Int[Tensor, "batch_size"] | None = None

    # Cache for sequence index of each token
    _seq_idx: Int[Tensor, "n"] | None = None

    # Cache for token positions
    _token_pos: Int[Tensor, "n"] | None = None

    # Optional provenance: index of the token in the sequence this batch was sampled from, -1 for
    # tokens that were inserted along the way. Carried through `Operations.apply`.
    origin: Int[Tensor, "n"] | None = None

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    @property
    def seq_lens_tensor(self) -> Int[Tensor, "batch_size"]:
        seq_lens = self._seq_lens_tensor
        if seq_lens is None:
            seq_lens = torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device)
            self._seq_lens_tensor = seq_lens
        return seq_lens

    @property
    def seq_idx(self) -> Int[Tensor, "n"]:
        """The sequence index of each token in the batch."""
        seq_idx = self._seq_idx
        if seq_idx is None:
            seq_idx = torch.repeat_interleave(
                torch.arange(self.batch_size, device=self.device),
                self.seq_lens_tensor,
                output_size=sum(self.seq_lens),
            )
            self._seq_idx = seq_idx
        return seq_idx

    @property
    def token_pos(self) -> Int[Tensor, "n"]:
        """The position of each token within their sequence.

        So for `seq_lens = (3, 1, 4)`, the `token_pos` would be `[0, 1, 2, 0, 0, 1, 2, 3]`.
        """
        pos = self._token_pos
        if pos is None:
            shifts = F.pad(self.seq_lens_tensor[:-1].cumsum(dim=0), (1, 0))
            pos = (
                torch.arange(len(self.x), device=self.device, dtype=shifts.dtype)
                - shifts[self.seq_idx]
            )
            self._token_pos = pos
        return pos

    def split_sequences(self) -> tuple[Int[Tensor, "seq_len"], ...]:
        """Split the batch into a list of sequences."""
        return torch.split(self.x, list(self.seq_lens), dim=0)

    @staticmethod
    def from_sequences(seqs: Iterable[Any]) -> DataBatch:
        """Create a `DataBatch` from a sequence of token sequences."""
        seqs = [torch.asarray(s, dtype=torch.int32) for s in seqs]
        x = torch.cat(seqs, dim=0)
        assert x.ndim == 1, "Token sequences must be one-dimensional."
        return DataBatch(seq_lens=tuple(len(s) for s in seqs), x=x)

    @staticmethod
    def from_equilength_batch(x: Int[Tensor, "batch_size seq_len"]) -> DataBatch:
        """Convert a rectangular batch matrix of tokens into a `DataBatch`."""
        batch_size, seq_len = x.shape
        return DataBatch(seq_lens=tuple([seq_len] * batch_size), x=x.flatten())


def rm_blanks(z_t: Int[Tensor, "batch_size aug_seq_len"]) -> DataBatch:
    """Map a batch of sequences from augmented space to data space, i.e. drop all epsilon tokens."""
    token_exists = z_t != EditFlow.EPSILON_TOKEN
    return DataBatch(seq_lens=tuple(token_exists.sum(dim=1).tolist()), x=z_t[token_exists])


def rm_blanks_idx(
    z_t: Int[Tensor, "batch_size aug_seq_len"],
) -> Int[Tensor, "batch_size aug_seq_len"]:
    """Compute the index in `x_t` of each non-epsilon token in `z_t` where `x_t =
    rm_blanks(z_t)`.

    The indices of epsilon tokens point to the position where the previous non-blank token ended up.
    If an epsilon token had no predecessor non-epsilon token in its sequence, its index will be -1.
    """
    # First, we compute the indices within each sequence in the batch.
    non_blanks_mask = z_t != EditFlow.EPSILON_TOKEN
    within_seq_idx = non_blanks_mask.cumsum(dim=-1) - 1
    # Then we shift the indices of each token by the number of tokens in earlier sequences, since
    # x_t is one long sequence.
    token_counts = non_blanks_mask.sum(dim=-1)
    global_seq_idx = (
        within_seq_idx + token_counts.roll(1).cumsum(dim=-1)[:, None] - token_counts[-1]
    )
    # Finally, we restore the -1 marks of blank tokens at the beginning of the sequence.
    return torch.where(within_seq_idx == -1, within_seq_idx, global_seq_idx)


class SequenceDistribution(Protocol):
    """A probability distribution over token sequences."""

    def sample(self, n_seqs: int, *, generator: torch.Generator) -> DataBatch:
        """Sample a batch of `n_seqs` sequences."""
        ...


class EquiLengthUniformTokenDistribution(SequenceDistribution):
    """A uniform distribution over token sequences of equal length."""

    def __init__(self, *, vocab_size: int, seq_len: int, bos_token: int | None):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.bos_token = bos_token

    def sample(self, n_seqs: int, *, generator: torch.Generator) -> DataBatch:
        """Sample a batch of `n_seqs` sequences."""
        tokens = torch.randint(
            0, self.vocab_size, (n_seqs, self.seq_len), generator=generator, device=generator.device
        )
        if self.bos_token is not None:
            tokens[:, 0] = self.bos_token
        return DataBatch.from_equilength_batch(tokens)


class Coupling(Protocol):
    r"""A coupling between two sequence distributions.

    All practical couplings $\pi(x_0, x_1)$ are implemented as minibatch approximations to take a
    batch of samples from the left and right marginal and match them up, so a coupling in our case
    only has to implement the matching up.
    """

    def couple(
        self, x_0: DataBatch, x_1: DataBatch, *, generator: torch.Generator
    ) -> tuple[DataBatch, DataBatch]:
        """Given samples `x_0` and `x_1` from the left and right marginal, return a coupled pair."""
        ...


class IndependentCoupling:
    r"""The independent coupling $\pi(x_0, x_1) = p(x_0) \cdot p(x_1)$."""

    def couple(
        self, x_0: DataBatch, x_1: DataBatch, *, generator: torch.Generator
    ) -> tuple[DataBatch, DataBatch]:
        return x_0, x_1


class SequenceAligner(Protocol):
    """Aligns sequences in the augmented space."""

    def align(
        self, x_0: DataBatch, x_1: DataBatch
    ) -> tuple[Int[Tensor, "batch_size aug_seq_len"], Int[Tensor, "batch_size aug_seq_len"]]:
        r"""Compute sequences $z_0$ and $z_1$ in the augmented space.

        For the aligned sequences, it shall hold that $x_i = \operatorname{rmblanks}(z_i)$.
        """
        ...


class ShiftAligner(SequenceAligner):
    """Aligns two sequences by shifting the second one by `k` palces.

    This means that the first $k$ tokens of $x_0$ are deleted, the remaining are substituted and
    whatever remains of $x_1$ is inserted.

    This simple alignment scheme is described in the paper in Section 5.
    """

    def __init__(self, *, k: int, bos_token: int | None = None):
        """
        Args:
            k: The number of tokens to delete from the beginning of $x_0$.
            bos_token: BOS tokens will always be aligned to the left, regardless of `k`
        """
        self.k = k
        self.bos_token = bos_token

    def align(
        self, x_0: DataBatch, x_1: DataBatch
    ) -> tuple[Int[Tensor, "batch_size aug_seq_len"], Int[Tensor, "batch_size aug_seq_len"]]:
        max_len = max(max(x_0.seq_lens), self.k + max(x_1.seq_lens))
        z_0 = x_0.x.new_full((x_0.batch_size, max_len), EditFlow.EPSILON_TOKEN)
        z_1 = x_1.x.new_full((x_1.batch_size, max_len), EditFlow.EPSILON_TOKEN)
        z_0[x_0.seq_idx, x_0.token_pos] = x_0.x
        z_1[x_1.seq_idx, self.k + x_1.token_pos] = x_1.x
        if self.bos_token is not None:
            z_1_bos = z_1 == self.bos_token
            z_1[z_1_bos] = EditFlow.EPSILON_TOKEN
            z_1[z_1.any(dim=1), 0] = self.bos_token
        return z_0, z_1


@numba.jit
def needleman_wunsch(
    x: np.ndarray, y: np.ndarray, *, epsilon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Align two sequences using the Needleman-Wunsch algorithm.

    The scores for a match are fixed to 1 and to -1 for everything else.
    """

    c_match = 1
    c_sub = -1
    c_ins = -1
    c_del = -1

    n, m = len(x), len(y)
    # Compute the matrix of prefix distances
    D = np.empty((n + 1, m + 1), dtype=np.int32)
    D[0, 0] = 0
    D[0, 1:] = np.full(m, c_ins, dtype=np.int32).cumsum()
    D[1:, 0] = np.full(n, c_del, dtype=np.int32).cumsum()
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = max(
                D[i - 1, j - 1] + (c_match if x[i - 1] == y[j - 1] else c_sub),
                D[i - 1, j] + c_del,
                D[i, j - 1] + c_ins,
            )

    # Extract the optimal alignment
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        prev_candidates = np.array(
            [
                D[i - 1, j - 1] + (c_match if x[i - 1] == y[j - 1] else c_sub),
                D[i - 1, j] + c_del,
                D[i, j - 1] + c_ins,
            ]
        )
        match prev_candidates.argmax():
            case 0:
                path.append(0)
                i = i - 1
                j = j - 1
            case 1:
                path.append(1)
                i = i - 1
            case 2:
                path.append(2)
                j = j - 1
    # Delete or insert any remaining characters
    for _ in range(i):
        path.append(1)
    for _ in range(j):
        path.append(2)
    d = len(path)
    x_aligned = np.empty(d, dtype=x.dtype)
    y_aligned = np.empty(d, dtype=y.dtype)
    i, j = 0, 0
    for k, op in enumerate(path[::-1]):
        match op:
            case 0:
                x_aligned[k] = x[i]
                i = i + 1
                y_aligned[k] = y[j]
                j = j + 1
            case 1:
                x_aligned[k] = x[i]
                i = i + 1
                y_aligned[k] = epsilon
            case 2:
                x_aligned[k] = epsilon
                y_aligned[k] = y[j]
                j = j + 1
    assert i == n
    assert j == m
    return x_aligned, y_aligned


class OptimalAligner(SequenceAligner):
    """Aligns sequences optimally, i.e. with the minimal Levenshtein distance.

    This uses the Needleman-Wunsch algorithm.
    """

    def align(
        self, x_0: DataBatch, x_1: DataBatch
    ) -> tuple[Int[Tensor, "batch_size aug_seq_len"], Int[Tensor, "batch_size aug_seq_len"]]:
        z_0s = []
        z_1s = []
        for x_0_seq, x_1_seq in zip(x_0.split_sequences(), x_1.split_sequences()):
            # We compute the alignments sequentially for ease of implementation, but since this runs
            # in the data workers anyway, this should not slow down training.
            z_0_seq, z_1_seq = needleman_wunsch(
                x_0_seq.numpy(force=True), x_1_seq.numpy(force=True), epsilon=EditFlow.EPSILON_TOKEN
            )
            z_0s.append(x_0.x.new_tensor(z_0_seq))
            z_1s.append(x_1.x.new_tensor(z_1_seq))

        max_len = max(max([len(s) for s in z_0s]), max([len(s) for s in z_1s]))
        z_0 = torch.stack(
            [F.pad(s, (0, max_len - len(s)), value=EditFlow.EPSILON_TOKEN) for s in z_0s]
        )
        z_1 = torch.stack(
            [F.pad(s, (0, max_len - len(s)), value=EditFlow.EPSILON_TOKEN) for s in z_1s]
        )
        return z_0, z_1


class ProbabilityPath(Protocol):
    """A probability path between the two distributions.

    This choice determines the exact form of the sum over log-rates in the Bregman divergence in
    Equation (28).
    """

    def sample(
        self,
        *,
        t: Float[Tensor, "batch_size"],
        z_0: Int[Tensor, "batch_size aug_seq_len"],
        z_1: Int[Tensor, "batch_size aug_seq_len"],
        generator: torch.Generator,
    ) -> tuple[Int[Tensor, "batch_size aug_seq_len"], Any]:
        r"""Sample from $p_t(z_t \mid z_0, z_1)$.

        The second return value can contain any extra latent variables that are needed to compute
        the log-rates term in the Bregman divergence.
        """
        ...

    def sum_log_rates(
        self,
        *,
        u_theta: LogRate,
        t: Float[Tensor, "batch_size"],
        z_t: Int[Tensor, "batch_size aug_seq_len"],
        z_1: Int[Tensor, "batch_size aug_seq_len"],
        latents: Any,
    ) -> Float[Tensor, "batch_size"]:
        r"""Compute the sum of log-rates for the given inputs.

        Specifically the double sum

        -E[u_t^\theta(x_t \mid x_t) +
            \sum_z\sum_{x\ne x_t} u_t(x, z\mid z_t, z_0, z_1)\log u_t^\theta(x\mid x_t)]
            ===========================================================================

        in the Bregman divergence in Equation (28) that depends on the choice of probability path.
        """
        ...


class KappaSchedule(nn.Module):
    """A kappa schedule for the mixture probability in token-wise mixture paths."""

    def forward(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        r"""Compute $\kappa_t$."""
        ...

    def loss_coefficient(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        r"""Compute $\dot{\kappa_t} / (1 - \kappa_t)$ term in the Edit Flows loss."""
        ...


class LinearKappaSchedule(KappaSchedule):
    r"""The linear schedule for $\kappa_t = t$."""

    def forward(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return t

    def loss_coefficient(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 1 / (1 - t).clamp(min=1e-3)


class CubicKappaSchedule(KappaSchedule):
    r"""The cubic schedule for $\kappa_t = t^3$."""

    def forward(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return t**3

    def loss_coefficient(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return 3 * t**2 / (1 - t**3).clamp(min=1e-3)


class TokenWiseMixturePath(nn.Module, ProbabilityPath):
    """The token-wise mixture path described in the paper."""

    def __init__(self, kappa: KappaSchedule):
        super().__init__()

        self.kappa = kappa

    def sample(
        self,
        *,
        t: Float[Tensor, "batch_size"],
        z_0: Int[Tensor, "batch_size aug_seq_len"],
        z_1: Int[Tensor, "batch_size aug_seq_len"],
        generator: torch.Generator,
    ) -> tuple[Int[Tensor, "batch_size aug_seq_len"], Any]:
        zt_mask = (
            torch.rand(z_1.shape, device=t.device, generator=generator) <= self.kappa(t)[..., None]
        )
        return torch.where(zt_mask, z_1, z_0), None

    def sum_log_rates(
        self,
        *,
        u_theta: LogRate,
        t: Float[Tensor, "batch_size"],
        z_t: Int[Tensor, "batch_size aug_seq_len"],
        z_1: Int[Tensor, "batch_size aug_seq_len"],
        latents: Any,
    ) -> Float[Tensor, "batch_size"]:
        z_1_is_epsilon = z_1 == EditFlow.EPSILON_TOKEN
        z_t_is_epsilon = z_t == EditFlow.EPSILON_TOKEN
        z_t_idx = rm_blanks_idx(z_t)
        per_token_rate_loss = ((z_t_idx != -1) & (z_1 != z_t)) * (
            z_1_is_epsilon * u_theta.log_lambda_del[z_t_idx]
            + z_t_is_epsilon
            * (
                u_theta.log_lambda_ins[z_t_idx]
                + torch.nn.functional.log_softmax(u_theta.Q_ins_logits, dim=-1)[z_t_idx, z_1]
            )
            + ~(z_1_is_epsilon | z_t_is_epsilon)
            * (
                u_theta.log_lambda_sub[z_t_idx]
                + torch.nn.functional.log_softmax(u_theta.Q_sub_logits, dim=-1)[z_t_idx, z_1]
            )
        )
        return self.kappa.loss_coefficient(t) * per_token_rate_loss.sum(dim=-1)


@dataclass
class LogRate:
    r"""A rate function evaluation $u_t^\theta(x \mid x_t)$.

    Note that this representation contains the logs of the rates and the logits for the token
    probabilities for greater numerical stability.
    """

    log_lambdas: Float[Tensor, "3 data_seq_len"]
    Q_logits: Float[Tensor, "2 data_seq_len vocab_size"]

    @property
    def log_lambda_ins(self) -> Float[Tensor, "data_seq_len"]:
        return self.log_lambdas[0]

    @property
    def log_lambda_sub(self) -> Float[Tensor, "data_seq_len"]:
        return self.log_lambdas[1]

    @property
    def log_lambda_del(self) -> Float[Tensor, "data_seq_len"]:
        return self.log_lambdas[2]

    @property
    def Q_ins_logits(self) -> Float[Tensor, "data_seq_len vocab_size"]:
        return self.Q_logits[0]

    @property
    def Q_sub_logits(self) -> Float[Tensor, "data_seq_len vocab_size"]:
        return self.Q_logits[1]

    def sample_euler_ops(self, h: float, *, generator: torch.Generator) -> Operations:
        """Sample edit operations using the Euler step approximation to."""

        # Pretty crude approximation but it works for small h
        p_ops = h * torch.exp(self.log_lambdas)
        op_masks = (
            torch.rand(p_ops.shape, dtype=p_ops.dtype, device=p_ops.device, generator=generator)
            < p_ops
        )
        p_tokens = torch.softmax(self.Q_logits, dim=-1)
        op_tokens = torch.multinomial(p_tokens.flatten(end_dim=1), 1, generator=generator)[
            :, 0
        ].unflatten(dim=0, sizes=(2, -1))
        return Operations(op_masks=op_masks, op_tokens=op_tokens)


def bincount_known_size(input: Int[Tensor, "n"], size: int) -> Int[Tensor, "size"]:
    """Count occurences of values in `input` with a known maximum size.

    `size` must be 1 larger than the largest value in `input`. In contrast to `torch.bincount`, this
    does not require a CPU-GPU sync, because the output shape is already known.
    """
    return torch.index_add(
        torch.zeros(size, dtype=torch.int64, device=input.device), 0, input, torch.ones_like(input)
    )


@dataclass
class Operations:
    """A set of edit operations resulting from one sampling step."""

    op_masks: Bool[Tensor, "3 data_seq_len"]
    op_tokens: Int[Tensor, "2 data_seq_len"]

    def apply(self, x_t: DataBatch, *, bos_token: int | None = None) -> DataBatch:
        """Apply the edit operations to `x_t`.

        If all tokens of a sequence would be deleted, no deletions on that sequence are performed.

        BOS tokens can neither be inserted, deleted, substituted nor substituted in. However, they
        can be inserted after.
        """

        op_ins, op_sub, op_del = self.op_masks
        tok_ins, tok_sub = self.op_tokens

        if bos_token is not None:
            op_ins = op_ins & (tok_ins != bos_token)
            op_sub = op_sub & (tok_sub != bos_token) & (x_t.x != bos_token)
            op_del = op_del & (x_t.x != bos_token)

        x = torch.where(op_sub, tok_sub, x_t.x)

        n_pre = len(x)
        n_ins = int(op_ins.sum())

        ins_idx = torch.argwhere(op_ins)[:, 0] + torch.arange(1, n_ins + 1, device=x.device)
        ins_mask = torch.zeros(n_pre + n_ins, device=x.device, dtype=torch.bool)
        ins_mask[ins_idx] = True
        keep_mask = ~ins_mask

        xt_ins = x.new_empty(n_pre + n_ins)
        xt_ins[keep_mask] = x
        xt_ins[ins_mask] = tok_ins[op_ins]

        seq_idx_ins = x_t.seq_idx.new_empty(n_pre + n_ins)
        seq_idx_ins[keep_mask] = x_t.seq_idx
        seq_idx_ins[ins_mask] = x_t.seq_idx[op_ins]

        op_del_ins = op_del.new_empty(n_pre + n_ins)
        op_del_ins[keep_mask] = op_del
        op_del_ins[ins_mask] = False

        # Prevent the deletion of all tokens of a sample
        sample_lengths = bincount_known_size(seq_idx_ins, x_t.batch_size)
        per_sample_del_ops = bincount_known_size(seq_idx_ins[op_del_ins], x_t.batch_size)
        op_del_ins[op_del_ins.clone()] ^= (sample_lengths == per_sample_del_ops)[
            seq_idx_ins[op_del_ins]
        ]

        x = xt_ins[~op_del_ins]
        seq_lens = bincount_known_size(seq_idx_ins[~op_del_ins], x_t.batch_size)

        origin = None
        if x_t.origin is not None:
            origin_ins = x_t.origin.new_empty(n_pre + n_ins)
            origin_ins[keep_mask] = x_t.origin
            origin_ins[ins_mask] = -1
            origin = origin_ins[~op_del_ins]

        return DataBatch(seq_lens=tuple(seq_lens.tolist()), x=x, origin=origin)


class EditFlowModel(Protocol):
    def __call__(self, xt: DataBatch, t: Float[Tensor, "batch_size"]) -> LogRate: ...


class EditFlow(nn.Module):
    """Implementation of Edit Flows [1].

    [1] Havasi, Karrer, Gar, Chen. "Edit Flows: Flow Matching with Edit Operations",
        https://arxiv.org/abs/2506.09018
    """

    # Special token to represent blank positions in the augmented space. Since the model never sees
    # data representations in the augmented space, it does not really matter what this is as long as
    # it does not conflict with any actual token value.
    EPSILON_TOKEN = -1

    def __init__(
        self,
        *,
        probability_path: ProbabilityPath,
        bos_token: int | None = None,
        low_discrepancy_sampling: bool = True,
    ):
        super().__init__()

        self.probability_path = probability_path
        self.bos_token = bos_token
        self.low_discrepancy_sampling = low_discrepancy_sampling

    def bregman_divergence(
        self,
        model: EditFlowModel,
        *,
        z_0: Int[Tensor, "batch_size aug_seq_len"],
        z_1: Int[Tensor, "batch_size aug_seq_len"],
        generator: torch.Generator,
    ) -> Float[Tensor, "batch_size"]:
        """Compute a Monte Carlo estimate of the Bregman divergence."""

        batch_size = z_0.shape[0]
        t = self._sample_t(batch_size=batch_size, device=z_0.device, generator=generator)
        z_t, latents_t = self.probability_path.sample(t=t, z_0=z_0, z_1=z_1, generator=generator)

        x_t = rm_blanks(z_t)
        u_theta = model(x_t, t)

        rate_penalty = torch.zeros(batch_size, device=z_0.device, dtype=u_theta.log_lambdas.dtype)
        if self.bos_token is None:
            rate_penalty = torch.scatter_add(
                rate_penalty, dim=0, index=x_t.seq_idx, src=u_theta.log_lambdas.exp().sum(dim=0)
            )
        else:
            # Substituting or deleting the BOS token is ignored, so we also ignore it in the rate
            # penalty
            rate_penalty = torch.scatter_add(
                rate_penalty, dim=0, index=x_t.seq_idx, src=u_theta.log_lambda_ins.exp()
            )
            x_t_is_not_bos = x_t.x != self.bos_token
            seq_idx_not_bos = x_t.seq_idx[x_t_is_not_bos]
            rate_penalty = torch.scatter_add(
                rate_penalty,
                dim=0,
                index=seq_idx_not_bos,
                src=u_theta.log_lambda_sub[x_t_is_not_bos].exp(),
            )
            rate_penalty = torch.scatter_add(
                rate_penalty,
                dim=0,
                index=seq_idx_not_bos,
                src=u_theta.log_lambda_del[x_t_is_not_bos].exp(),
            )

        return rate_penalty - self.probability_path.sum_log_rates(
            u_theta=u_theta, t=t, z_t=z_t, z_1=z_1, latents=latents_t
        )

    def _sample_t(
        self, *, batch_size: int, device: torch.device, generator: torch.Generator
    ) -> Float[Tensor, "{batch_size}"]:
        if self.low_discrepancy_sampling:
            # Low-discrepancy sampling as in the VDM paper [1].
            #
            # [1] https://arxiv.org/abs/2107.00630
            offset = torch.rand((), device=device, generator=generator)
            grid = torch.arange(batch_size, device=device) / batch_size
            return torch.remainder(grid + offset, 1)
        else:
            return torch.rand(batch_size, generator=generator, device=device)

    @torch.inference_mode()
    def sample_euler(
        self, model: EditFlowModel, *, x_0: DataBatch, n_steps: int, generator: torch.Generator
    ) -> DataBatch:
        """Sample from the model with the Euler approximation to the learned CTMC."""

        x_t = x_0
        h = 1 / n_steps
        for t in torch.linspace(0.0, 1.0, n_steps + 1, device=x_0.device)[:-1]:
            rate = model(x_t, eo.repeat(t, "-> b", b=x_0.batch_size))
            ops = rate.sample_euler_ops(h, generator=generator)
            x_t = ops.apply(x_t, bos_token=self.bos_token)

        return x_t

    @torch.inference_mode()
    def sample_flow_euler(
        self, model: EditFlowModel, *, x_0: DataBatch, n_steps: int, generator: torch.Generator
    ) -> list[DataBatch]:
        """Sample from the model with the Euler approximation to the learned CTMC and return the
        intermediate states of each sample.
        """

        x_t = x_0
        h = 1 / n_steps
        samples = [x_0]
        for t in torch.linspace(0.0, 1.0, n_steps + 1, device=x_0.device)[:-1]:
            rate = model(x_t, eo.repeat(t, "-> b", b=x_0.batch_size))
            ops = rate.sample_euler_ops(h, generator=generator)
            x_t = ops.apply(x_t, bos_token=self.bos_token)
            samples.append(x_t)

        return samples
