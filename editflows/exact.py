r"""Exact references on the GPU: the posterior, joint posterior samples and an exact-rate Euler sampler.

`editflows.grammar` holds the numpy reference of the hidden-Markov posterior
$P(\text{letter is an insertion} \mid \text{sequence})$. Here the same model is batched in torch (float64), which
makes it cheap enough to drive a sampler with it:

* `posterior` — forward-backward, equal to `grammar.insertion_posterior`;
* `posterior_sample` — forward filtering, backward sampling: one exact *joint* sample of "which letters are
  insertions" per sequence, without any time axis;
* `exact_euler` — the Euler sampler of `EditFlow.sample_euler` driven by exact rates
  (hazard times posterior), with a prior a model could have: the expected share of leftover insertions at time t.

The hidden state is "the last two original letters": none yet, one letter `a`, or a context `(a, b)`. Every token is
an insertion with probability `rho` (a uniform letter, the state stays) or the next letter of the chain.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def pad(seqs, device) -> tuple[Tensor, Tensor]:
    """Letter ids `(N, L)` and the mask of real positions for sequences of different lengths."""
    lengths = torch.tensor([len(s) for s in seqs], device=device)
    letters = torch.zeros((len(seqs), int(lengths.max()) if len(seqs) else 0), dtype=torch.int64, device=device)
    for i, s in enumerate(seqs):
        letters[i, : len(s)] = torch.as_tensor(np.asarray(s), dtype=torch.int64, device=device)
    return letters, torch.arange(letters.shape[1], device=device)[None] < lengths[:, None]


def _weights(valid: Tensor, rho: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """Per position: weight of "insertion" and of "original"; padding is a step that changes nothing."""
    rho = rho.to(torch.float64)[:, None]
    one, zero = torch.ones((), dtype=torch.float64, device=valid.device), torch.zeros((), dtype=torch.float64, device=valid.device)
    return torch.where(valid, rho / n, one), torch.where(valid, 1 - rho, zero)


def _forward(S: Tensor, ins: Tensor, org: Tensor, P: Tensor, store: bool = True):
    """Filtered state before every token, `(F0 (L, N), F1 (L, N, n), FA (L, N, n, n))` (None without `store`), the
    final state and the log-likelihood of every sequence."""
    N, L = S.shape
    n = P.shape[0]
    Pc_all = P.permute(2, 0, 1)  # [c, a, b]
    s0 = torch.ones(N, dtype=torch.float64, device=S.device)
    s1 = torch.zeros((N, n), dtype=torch.float64, device=S.device)
    A = torch.zeros((N, n, n), dtype=torch.float64, device=S.device)
    F0 = F1 = FA = None
    if store:
        F0, F1, FA = S.new_empty((L, N), dtype=torch.float64), S.new_empty((L, N, n), dtype=torch.float64), S.new_empty((L, N, n, n), dtype=torch.float64)
    loglik = torch.zeros(N, dtype=torch.float64, device=S.device)
    for i in range(L):
        c, w_ins, w_org = S[:, i], ins[:, i], org[:, i]
        if store:
            F0[i], F1[i], FA[i] = s0, s1, A
        n0 = s0 * w_ins
        n1 = (s1 * w_ins[:, None]).scatter_add(1, c[:, None], (s0 * w_org / n)[:, None])
        step = ((A * Pc_all[c]).sum(1) + s1 / n) * w_org[:, None]  # into the state (b, c), for every b
        nA = (A * w_ins[:, None, None]).scatter_add(2, c[:, None, None].expand(-1, n, 1), step[:, :, None])
        z = n0 + n1.sum(1) + nA.sum((1, 2))
        s0, s1, A = n0 / z, n1 / z[:, None], nA / z[:, None, None]
        loglik += torch.log(z)
    return (F0, F1, FA), (s0, s1, A), loglik


def _chunks(lengths: Tensor, n: int, budget: float):
    """Index chunks of sequences of similar length whose forward messages fit into `budget` floats."""
    order = torch.argsort(lengths, stable=True)
    states = 1 + n + n * n
    start = 0
    while start < len(order):
        rows = max(1, int(budget // (max(int(lengths[order[start]]), 1) * states)))
        longest = int(lengths[order[min(start + rows, len(order)) - 1]])
        rows = max(1, min(rows, int(budget // (max(longest, 1) * states))))
        yield order[start : start + rows]
        start += rows


@torch.no_grad()
def posterior(S: Tensor, valid: Tensor, rho: Tensor, P: Tensor, *, budget: float = 6e7) -> Tensor:
    """P(insertion | sequence) for every position of the padded batch `S`; 0 at padding."""
    N, L = S.shape
    n = P.shape[0]
    out = torch.zeros((N, L), dtype=torch.float64, device=S.device)
    lengths = valid.sum(1)
    for idx in _chunks(lengths, n, budget):
        length = int(lengths[idx].max())
        if length == 0:
            continue
        s, v = S[idx, :length], valid[idx, :length]
        ins, org = _weights(v, rho[idx], n)
        (F0, F1, FA), _, _ = _forward(s, ins, org, P)
        Pc_all = P.permute(2, 0, 1)
        M = len(idx)
        b0 = torch.ones(M, dtype=torch.float64, device=S.device)
        b1 = torch.ones((M, n), dtype=torch.float64, device=S.device)
        B = torch.ones((M, n, n), dtype=torch.float64, device=S.device)
        post = torch.empty((M, length), dtype=torch.float64, device=S.device)
        for i in range(length - 1, -1, -1):
            c, w_ins, w_org = s[:, i], ins[:, i], org[:, i]
            p0, p1, pA = F0[i], F1[i], FA[i]
            Bc = B.gather(2, c[:, None, None].expand(-1, n, 1)).squeeze(2)  # beta of the state (x, c), every x
            nb0 = w_ins * b0 + w_org / n * b1.gather(1, c[:, None]).squeeze(1)
            nb1 = w_ins[:, None] * b1 + w_org[:, None] / n * Bc
            nB = w_ins[:, None, None] * B + w_org[:, None, None] * Pc_all[c] * Bc[:, None, :]
            stay = w_ins * (p0 * b0 + (p1 * b1).sum(1) + (pA * B).sum((1, 2)))
            post[:, i] = stay / (p0 * nb0 + (p1 * nb1).sum(1) + (pA * nB).sum((1, 2)))
            m = torch.maximum(torch.maximum(nb0, nb1.amax(1)), nB.amax((1, 2)))
            b0, b1, B = nb0 / m, nb1 / m[:, None], nB / m[:, None, None]
        out[idx, :length] = post * v
        del F0, F1, FA  # free the messages before the next chunk allocates its own
    return out


@torch.no_grad()
def log_likelihood(S: Tensor, valid: Tensor, rho: Tensor, P: Tensor, *, budget: float = 6e7) -> Tensor:
    """log P(sequence | insertion rate `rho`) of every sequence."""
    out = torch.zeros(S.shape[0], dtype=torch.float64, device=S.device)
    lengths = valid.sum(1)
    for idx in _chunks(lengths, P.shape[0], budget):
        length = max(int(lengths[idx].max()), 1)
        ins, org = _weights(valid[idx, :length], rho[idx], P.shape[0])
        out[idx] = _forward(S[idx, :length], ins, org, P, store=False)[2]
    return out


@torch.no_grad()
def estimate_rho(S: Tensor, valid: Tensor, P: Tensor, *, grid=None) -> Tensor:
    """The insertion rate of every sequence by maximum likelihood over a grid: what a model can know about the
    noise level from the source alone."""
    grid = torch.linspace(0.005, 0.6, 24, dtype=torch.float64, device=S.device) if grid is None else grid
    ll = torch.stack([log_likelihood(S, valid, torch.full((S.shape[0],), float(r), dtype=torch.float64, device=S.device), P) for r in grid])
    return grid[ll.argmax(0)]


@torch.no_grad()
def posterior_sample(S: Tensor, valid: Tensor, rho: Tensor, P: Tensor, *, generator: torch.Generator, budget: float = 6e7) -> Tensor:
    """One exact joint sample of the insertion mask of every sequence (forward filtering, backward sampling)."""
    N, L = S.shape
    n = P.shape[0]
    out = torch.zeros((N, L), dtype=torch.bool, device=S.device)
    lengths = valid.sum(1)
    for idx in _chunks(lengths, n, budget):
        length = int(lengths[idx].max())
        if length == 0:
            continue
        s, v = S[idx, :length], valid[idx, :length]
        ins, org = _weights(v, rho[idx], n)
        (F0, F1, FA), (s0, s1, A), _ = _forward(s, ins, org, P)
        M = len(idx)
        rows = torch.arange(M, device=S.device)
        # the final state: kind 0 = no original letter, 1 = one letter b, 2 = context (a, b)
        final = torch.multinomial(torch.cat([s0[:, None], s1, A.flatten(1)], 1), 1, generator=generator).squeeze(1)
        kind = (final > 0).long() + (final > n).long()
        b = torch.where(kind == 1, final - 1, torch.where(kind == 2, (final - 1 - n) % n, torch.zeros_like(final)))
        a = torch.where(kind == 2, (final - 1 - n) // n, torch.zeros_like(final))
        sample = torch.zeros((M, length), dtype=torch.bool, device=S.device)
        for i in range(length - 1, -1, -1):
            c, w_ins, w_org = s[:, i], ins[:, i], org[:, i]
            p0, p1, pA = F0[i], F1[i], FA[i]
            here = torch.where(kind == 0, p0, torch.where(kind == 1, p1[rows, b], pA[rows, a, b]))
            match = ((b == c) & (kind > 0)).to(torch.float64)
            simple = torch.where(kind == 1, p0, p1[rows, a]) * w_org / n * match * (kind > 0)
            chain = pA[rows, :, a] * P[:, a, c].T * (w_org * match * (kind == 2))[:, None]  # previous context (a', a)
            W = torch.cat([(w_ins * here)[:, None], simple[:, None], chain], 1)
            W[W.sum(1) == 0, 0] = 1.0  # numerically empty: call it an insertion
            choice = torch.multinomial(W, 1, generator=generator).squeeze(1)
            sample[:, i] = (choice == 0) & v[:, i]
            first, deeper = choice == 1, choice >= 2
            # an original letter was read: step back to the state before it
            new_kind = torch.where(first, kind - 1, kind)
            new_b = torch.where(first | deeper, a, b)
            new_a = torch.where(deeper, choice - 2, a)
            kind, a, b = new_kind, new_a, new_b
        out[idx, :length] = sample
        del F0, F1, FA
    return out


def leftover_share(rho0: Tensor, kappa_t: float) -> Tensor:
    """Expected share of insertions among the letters still present when a share `kappa_t` of them is gone."""
    left = rho0 * (1 - kappa_t)
    return left / (1 - rho0 + left)


@torch.no_grad()
def exact_euler(
    letters: Tensor,
    valid: Tensor,
    rho0: Tensor,
    P: Tensor,
    kappa,
    *,
    n_steps: int = 100,
    seed: int = 0,
    start_step: int = 0,
    alive: Tensor | None = None,
    exclusion: int = 0,
    leak: float = 0.0,
    budget: float = 6e7,
) -> Tensor:
    """The Euler sampler of `EditFlow.sample_euler` with exact rates instead of a model's.

    At step k (time t = k / n_steps) every letter still present is deleted with probability
    `h * hazard(t) * q`, where q is its exact posterior of being a leftover insertion given the current sequence
    and the prior `leftover_share(rho0, kappa(t))` - the share a model can expect at time t, not the realised one.
    The uniform random numbers are keyed by (step, source position), so runs that differ in one intervention share
    all others. `exclusion = w > 0` allows at most one deletion per window of +-w letters per step (the most
    urgent one). `leak > 0` is a deliberate defect: no letter's q is taken to be smaller than `leak`, which
    makes the otherwise exact sampler delete letters without reason at a known rate. Returns the step at which
    every source letter was deleted, -1 if it survived (and at padding).
    """
    N, L0 = letters.shape
    device = letters.device
    alive = valid.clone() if alive is None else alive.clone()
    del_step = torch.full((N, L0), -1, dtype=torch.int16, device=device)
    h = 1.0 / n_steps
    position = torch.arange(L0, device=device)[None].expand(N, -1)
    for k in range(start_step, n_steps):
        t = torch.tensor(k * h, dtype=torch.float64)
        hazard, kappa_t = float(kappa.loss_coefficient(t)), float(kappa(t))
        order = torch.argsort((~alive).to(torch.int8), dim=1, stable=True)  # letters still present first
        length = int(alive.sum(1).max())
        if length == 0:
            break
        order = order[:, :length]
        present = position[:, :length] < alive.sum(1)[:, None]
        q = posterior(letters.gather(1, order), present, leftover_share(rho0, kappa_t), P, budget=budget)
        u = torch.rand((N, L0), generator=torch.Generator(device).manual_seed(seed * 1_000_003 + k), device=device, dtype=torch.float64)
        p = (h * hazard * q.clamp(min=leak)).clamp(max=1.0)
        urgency = torch.where(present, p - u.gather(1, order), torch.full_like(p, -1.0))
        delete = urgency > 0
        if exclusion > 0:
            best = torch.nn.functional.max_pool1d(urgency[:, None], 2 * exclusion + 1, stride=1, padding=exclusion)[:, 0]
            delete &= urgency >= best
        rows = torch.arange(N, device=device)[:, None].expand(-1, length)[delete]
        cols = order[delete]
        alive[rows, cols] = False
        del_step[rows, cols] = k
    return del_step
