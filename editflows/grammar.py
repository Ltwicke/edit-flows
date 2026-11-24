r"""Exact references for the synthetic targets: what a model that knows the transition table can achieve.

The placeholder targets are an order-2 Markov chain with a table `P[a, b, c]` = probability of letter `c` after the
letters `a, b` (`scripts/make_synthetic_targets.py`). A source is such a target with uniform letters inserted, so
"which letters are insertions" is inference in a small hidden Markov model with a known answer. Everything here
takes the table as an argument; nothing depends on a model.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def insertion_posterior(seqs, rho, P: np.ndarray, budget: float = 2e7) -> list[np.ndarray]:
    r"""$P(\text{letter is an insertion} \mid \text{whole sequence})$ for every letter of every sequence.

    `seqs` are letter ids `0 .. n_letters - 1` without BOS, `rho` is the insertion probability per letter (one
    number or one per sequence). Exact forward-backward over the hidden state "the last two original letters"
    (no letter yet, one letter `a`, context `(a, b)`): each token is an insertion with probability `rho` (a uniform
    letter, the state stays) or the next letter of the chain. Sequences are padded; padding is a no-op step.
    """
    seqs = [np.asarray(s, dtype=np.int64) for s in seqs]
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (len(seqs),))
    n = P.shape[0]
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out: list = [None] * len(seqs)
    start = 0
    while start < len(order):  # chunks of similar length, about `budget` floats of memory
        states = 1 + n + n * n
        rows = max(1, int(budget // (max(len(seqs[order[start]]), 1) * states)))
        longest = len(seqs[order[min(start + rows, len(order)) - 1]])
        rows = max(1, min(rows, int(budget // (max(longest, 1) * states))))
        idx = order[start : start + rows]
        length = max(len(seqs[i]) for i in idx)
        S = np.zeros((len(idx), length), dtype=np.int64)
        valid = np.zeros((len(idx), length), dtype=bool)
        for r, i in enumerate(idx):
            S[r, : len(seqs[i])] = seqs[i]
            valid[r, : len(seqs[i])] = True
        q = _forward_backward(S, np.where(valid, rho[idx, None] / n, 1.0), np.where(valid, 1 - rho[idx, None], 0.0), P)
        for r, i in enumerate(idx):
            out[i] = q[r, : len(seqs[i])]
        start += len(idx)
    return out


def _forward_backward(S: np.ndarray, INS: np.ndarray, ORG: np.ndarray, P: np.ndarray) -> np.ndarray:
    N, L = S.shape
    n = P.shape[0]
    rows = np.arange(N)
    s0, s1, A = np.ones(N), np.zeros((N, n)), np.zeros((N, n, n))
    fwd = []
    for i in range(L):  # forward, normalised every step
        c, ins, org = S[:, i], INS[:, i], ORG[:, i]
        fwd.append((s0, s1, A))
        n0 = s0 * ins
        n1 = s1 * ins[:, None]
        n1[rows, c] += s0 * org / n
        nA = A * ins[:, None, None]
        nA[rows, :, c] += ((A * P[:, :, c].transpose(2, 0, 1)).sum(1) + s1 / n) * org[:, None]
        z = n0 + n1.sum(1) + nA.sum((1, 2))
        s0, s1, A = n0 / z, n1 / z[:, None], nA / z[:, None, None]
    b0, b1, B = np.ones(N), np.ones((N, n)), np.ones((N, n, n))
    post = np.empty((N, L))
    for i in range(L - 1, -1, -1):  # backward
        c, ins, org = S[:, i], INS[:, i], ORG[:, i]
        p0, p1, pA = fwd[i]
        Bc = B[rows, :, c]  # beta of the state (x, c), for every x
        nb0 = ins * b0 + org / n * b1[rows, c]
        nb1 = ins[:, None] * b1 + org[:, None] / n * Bc
        nB = ins[:, None, None] * B + org[:, None, None] * P[:, :, c].transpose(2, 0, 1) * Bc[:, None, :]
        stay = ins * (p0 * b0 + (p1 * b1).sum(1) + (pA * B).sum((1, 2)))
        post[:, i] = stay / (p0 * nb0 + (p1 * nb1).sum(1) + (pA * nB).sum((1, 2)))
        m = np.maximum(np.maximum(nb0, nb1.max(1)), nB.max((1, 2)))
        b0, b1, B = nb0 / m, nb1 / m[:, None], nB / m[:, None, None]
    return post


def brute_force_posterior(seq, rho: float, P: np.ndarray) -> np.ndarray:
    """The same posterior by summing the likelihood over all `2^len(seq)` insertion patterns (tests only)."""
    seq = np.asarray(seq, dtype=np.int64)
    n = P.shape[0]

    def likelihood(mask):
        original = seq[~np.array(mask, dtype=bool)]
        p = (rho / n) ** sum(mask) * (1 - rho) ** len(original) * (1 / n) ** min(2, len(original))
        for i in range(2, len(original)):
            p *= P[original[i - 2], original[i - 1], original[i]]
        return p

    lik = {m: likelihood(m) for m in product((0, 1), repeat=len(seq))}
    total = sum(lik.values())
    return np.array([sum(v for m, v in lik.items() if m[i]) for i in range(len(seq))]) / total


def auroc(score, positive) -> float:
    """P(score of a random positive > score of a random negative); ties count half. NaN if a class is empty."""
    score, positive = np.asarray(score, dtype=float), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mean_auroc(scores, masks) -> float:
    """Mean AUROC per sequence (sequences with both classes): a ranking score that no per-sequence offset of
    the scores, such as the time dependence of a rate, can change."""
    values = [auroc(s, m) for s, m in zip(scores, masks)]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def ranking_scores(scores, masks) -> dict[str, float]:
    """`rank_p_at_k` of the scores (as in `editflows.metrics`) and the value of a random ranking."""
    hits = sum(int(m[np.argsort(-np.asarray(s), kind="stable")[: int(m.sum())]].sum()) for s, m in zip(scores, masks))
    k = np.array([int(m.sum()) for m in masks], dtype=float)
    n = np.array([len(m) for m in masks], dtype=float)
    return {"rank_p_at_k": hits / max(k.sum(), 1), "rank_p_at_k_random": float((k * k / n).sum() / max(k.sum(), 1))}


def ideal_scores(masks, q) -> dict[str, float]:
    """What the exact posteriors `q` achieve on sources with the inserted masks `masks`: top-k precision
    (compare `rank_p_at_k`), precision / recall when every token is deleted with its posterior probability
    (compare the sampled `del_precision` / `del_recall`), F1 of deleting exactly the tokens with q > 1/2."""
    K = sum(int(m.sum()) for m in masks)
    ins, p = np.concatenate(masks), np.concatenate(q)
    tp, fp = int((ins & (p > 0.5)).sum()), int((~ins & (p > 0.5)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(K, 1)
    sp, sr = p[ins].sum() / max(p.sum(), 1e-12), p[ins].sum() / max(K, 1)
    return {
        "chance": float(ins.mean()),
        "top_k_precision": ranking_scores(q, masks)["rank_p_at_k"],
        "sampling_precision": float(sp),
        "sampling_recall": float(sr),
        "sampling_f1": float(2 * sp * sr / max(sp + sr, 1e-12)),
        "map_f1": float(2 * prec * rec / max(prec + rec, 1e-12)),
        "auroc": mean_auroc(q, masks),
    }


def rule_probes(tokens: np.ndarray, offsets: np.ndarray, indices: np.ndarray, *, n_letters: int = 20,
                repeats: int = 2, left: int = 120, seed: int = 0, subsample: float = 0.2):
    """Probe sequences for every rule: `left` letters of a target that end in a context `(a, b)`, followed by
    a probe letter `c` as the last letter - for all `n_letters^2` contexts and every `c`, `repeats` times.

    If `c` may follow `(a, b)` the probe is a valid target; otherwise it contains exactly one violated window and
    nothing after `c` can give it away. Returns `(sequences, context id, probe letter)`; the probe letter sits at
    position `left` of every row. Only a random share `subsample` of the candidate positions is searched.
    """
    rng = np.random.default_rng(seed)
    tok = tokens.astype(np.int64)
    seq_of = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    selected = np.zeros(len(offsets) - 1, dtype=bool)
    selected[indices] = True
    end = np.arange(left - 1, len(tok))  # position of b, the last letter of the context
    end = end[rng.random(len(end)) < subsample]
    end = end[(seq_of[end - left + 1] == seq_of[end]) & selected[seq_of[end]]]
    ctx = tok[end - 1] * n_letters + tok[end]
    order = np.lexsort((rng.random(len(end)), ctx))  # by context, random within
    end, ctx = end[order], ctx[order]
    first = np.searchsorted(ctx, np.arange(n_letters**2))
    assert (np.diff(np.append(first, len(ctx))) >= repeats).all(), "a context with fewer than `repeats` occurrences"
    picks = (first[:, None] + np.arange(repeats)).ravel()
    prefix = np.repeat(tok[end[picks, None] + np.arange(-left + 1, 1)], n_letters, axis=0)
    letter = np.tile(np.arange(n_letters), n_letters**2 * repeats)
    context = np.repeat(np.arange(n_letters**2), repeats * n_letters)
    return np.concatenate([prefix, letter[:, None]], axis=1), context, letter


def rule_auroc(score: np.ndarray, context: np.ndarray, eligible: np.ndarray, n_contexts: int = 400) -> np.ndarray:
    """Per context: AUROC of the scores of the probe letter between ineligible and eligible letters
    (0.5 = the rule is unknown, 1 = known)."""
    return np.array([auroc(score[context == k], ~eligible[context == k]) for k in range(n_contexts)])


def ideal_flow(seqs, masks, P: np.ndarray, hazard, *, n_steps: int = 100, max_rate: float = np.exp(10.0), seed: int = 0):
    """Run the Euler sampler with the exact rates instead of a model's.

    As in `EditFlow.sample_euler`, at each of the `n_steps` steps every remaining letter is deleted with
    probability `h * rate`; the rate is `hazard(t)` times the exact probability that the letter is a leftover
    insertion given the current sequence (with the true share of leftovers as the prior). `hazard` is the
    schedule's `loss_coefficient`. Returns, per source, the mask of the deleted letters: what a perfect model
    achieves with this sampler and this schedule.
    """
    rng = np.random.default_rng(seed)
    seqs = [np.asarray(s, dtype=np.int64) for s in seqs]
    alive = [np.ones(len(s), dtype=bool) for s in seqs]
    h = 1 / n_steps
    for step in range(n_steps):
        rate = hazard(step * h)
        open_ = [i for i, (m, a) in enumerate(zip(masks, alive)) if m[a].any()]  # sources with leftovers
        if not open_:
            break
        posterior = insertion_posterior(
            [seqs[i][alive[i]] for i in open_], [float(masks[i][alive[i]].mean()) for i in open_], P
        )
        for i, q in zip(open_, posterior):
            remaining = np.flatnonzero(alive[i])
            alive[i][remaining[rng.random(len(remaining)) < h * np.minimum(rate * q, max_rate)]] = False
    return [~a for a in alive]


def deletion_scores(deleted, masks) -> dict[str, float]:
    """Precision, recall, F1 and mean absolute length error of deleted-letter masks against the inserted masks."""
    d, m = np.concatenate(deleted), np.concatenate(masks)
    tp, fp, fn = int((d & m).sum()), int((d & ~m).sum()), int((~d & m).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "len_err": float(np.mean([abs(int(x.sum()) - int(y.sum())) for x, y in zip(deleted, masks)])),
        "exact_match": float(np.mean([bool((x == y).all()) for x, y in zip(deleted, masks)])),
    }
