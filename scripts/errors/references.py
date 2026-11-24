#!/usr/bin/env python
"""References and error budget (hypothesis C): which errors would a perfectly informed sampler make too?

    python scripts/errors/references.py --run runs/lap-01

The ladder of samplers, all on the sources of `runs/error-study/sources.npz`:

    G      exact joint posterior sample under the prior of the coupling (built here, see `gap_*`)
    FFBS   exact joint posterior sample under independent insertion marks   runs/error-study/ffbs.npz
    I2     Euler sampler with exact rates, true noise level                  runs/error-study/i2-100.npz
    I3     ... noise level estimated from the source                         runs/error-study/i3-100.npz
    M      the trained model                                                 runs/<run>/errors/m100.npz

plus the deterministic decisions "delete iff the probability is above 1/2" for the three posteriors (MAP-G,
MAP-exact, MAP-model). The script uses what is there, says what is missing, and writes tables, `summary.json`
and figures to `runs/<run>/errors/results/references/`.

Sections: 1 reliability of the reference posteriors, 2 the 2q(1-q) law, 3 the error budget by kind of cluster
(every comparison on the largest set of sources that all its samplers cover), 4 the label-only class, 5 sizes of
holes, 6 the step-size sweep.

Why G exists: `editflows.exact` assumes independent insertion marks, the coupling throws a fixed number of letters
into the gaps between the originals. Section 1 shows that the first posterior is measurably miscalibrated for the
sources and the second is not, so the second one is the reference for "perfectly informed".

Outputs: budget.csv (rungs x metrics x kinds x sets of sources, with intervals), reliability.csv,
reliability_by_left_neighbour.csv, law.csv, label_only.csv, holes.csv, holes_by_noise.csv, steps.csv,
confounders.csv, summary.json, fig1 ... fig8; cached tables in scratch/. Intervals: bootstrap over sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import textwrap
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import Sources, canonical_counts, clusters  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
KINDS = ["Z", "V", "R", "U"]
KIND_TEXT = {"Z": "Z  label-only (same string)", "V": "V  valid alternative", "R": "R  repaired hole",
             "U": "U  unresolved (ungrammatical)"}
KIND_COLOR = dict(zip(KINDS, COLORS))
LADDER = ["G", "FFBS", "I2", "I3", "M"]
RUNG_COLOR = {"FFBS": COLORS[0], "I2": COLORS[1], "I3": COLORS[2], "M": COLORS[3], "G": COLORS[4]}
RUNG_MARKER = {"FFBS": "s", "I2": "^", "I3": "v", "M": "o", "G": "D"}
RUNG_TEXT = {
    "G": "exact sample, prior of the coupling (G)",
    "FFBS": "exact sample, independent marks (FFBS)",
    "I2": "exact rates, Euler steps (I2)",
    "I3": "exact rates, noise level estimated (I3)",
    "M": "trained model (M)",
    "MAP-G": "most probable label, prior of the coupling",
    "MAP-exact": "most probable label, independent marks",
    "MAP-model": "most probable label, the model's own q",
}
RUNG_TICK = {
    "MAP-G": "most probable\nlabel\n(coupling prior)", "G": "exact sample\n(coupling prior)\nG",
    "MAP-exact": "most probable\nlabel\n(indep. marks)", "FFBS": "exact sample\n(indep. marks)\nFFBS",
    "I2": "exact rates,\nEuler steps\nI2", "I3": "+ noise level\nestimated\nI3", "M": "trained\nmodel\nM",
    "MAP-model": "most probable\nlabel\n(model's q)",
}
RUNG_SHORT = {
    "MAP-G": "most\nprobable\nlabel,\ncoupling\nprior", "G": "exact\nsample,\ncoupling\nprior\n(G)",
    "MAP-exact": "most\nprobable\nlabel,\nindep.\nmarks", "FFBS": "exact\nsample,\nindep.\nmarks\n(FFBS)",
    "I2": "exact\nrates,\nEuler\nsteps\n(I2)", "I3": "noise\nlevel\nestimated\n(I3)", "M": "trained\nmodel\n(M)",
    "MAP-model": "most\nprobable\nlabel,\nmodel's\nown q",
}
STEPS = [("FFBS", "G"), ("I2", "FFBS"), ("I3", "I2"), ("M", "I3")]
STEP_TEXT = {("FFBS", "G"): "reference prior ignores\nthe gap counts\n(FFBS − G)", ("I2", "FFBS"): "Euler dynamics\n(I2 − FFBS)",
             ("I3", "I2"): "noise level\nnot known\n(I3 − I2)", ("M", "I3"): "learning\n(M − I3)"}
EXTRA_STEPS = [("M", "G"), ("M", "FFBS"), ("M", "I2"), ("I2", "G"), ("MAP-model", "MAP-G"), ("MAP-exact", "MAP-G")]
NOISE_EDGES = [0.0, 0.1, 0.2, 0.35, 1.0]
NOISE_TEXT = ["under 10 %", "10–20 %", "20–35 %", "35–50 %"]
Q_EDGES = np.array([0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.999, 0.9999, 1.0000001])
REGIONS = {"certainly original (p < 1 %)": (0, 3), "ambiguous (1 % – 99 %)": (3, 15), "certainly inserted (p > 99 %)": (15, 18)}
RHO_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 1.0])  # finer noise bins for the holes
V_WINDOW = 5 / 20  # chance that one window of three random letters is allowed
GAP_CAP = 12  # insertions per gap that the coupling prior tracks exactly

BASE = ["letters", "inserted", "deleted", "tp", "s_wrong", "s_missed", "equal"]
COLS = BASE + [f"{what}_{k}" for k in KINDS for what in ("clusters", "x", "miss")]
IDX = {name: i for i, name in enumerate(COLS)}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E4E4E4",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8.5,
    "lines.linewidth": 1.6, "lines.markersize": 5, "figure.titlesize": 11.5, "figure.titleweight": "bold",
})


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ---- the posterior under the prior of the coupling ---------------------------------------------------------------
#
# `editflows.exact` assumes that every source letter is an insertion independently with probability rho, so the
# number of insertions in one gap between two originals is geometric. The coupling throws K letters into the
# n1 + 1 gaps uniformly, so that number is Binomial(K, 1 / (n1 + 1)): after an insertion a further one is less
# likely than after an original. The hidden state below therefore also counts the insertions of the current gap
# (capped at GAP_CAP); with a constant hazard it is the model of `editflows.exact` (checked in `gap_reference`).


def gap_hazards(n_inserted: int, n_original: int, cap: int = GAP_CAP) -> np.ndarray:
    """h[j] = P(N > j | N >= j) for N ~ Binomial(K, 1 / (n1 + 1)), the number of insertions in one gap."""
    p = 1.0 / (n_original + 1)
    pmf = np.zeros(cap + 2)
    pmf[0] = (1 - p) ** n_inserted
    for j in range(1, cap + 2):
        pmf[j] = pmf[j - 1] * (n_inserted - j + 1) / j * p / (1 - p) if n_inserted - j + 1 > 0 else 0.0
    at_least = np.clip(1 - np.concatenate([[0.0], np.cumsum(pmf)]), 0.0, 1.0)  # P(N >= j)
    h = np.where(at_least[:-1] > 1e-300, at_least[1:] / np.maximum(at_least[:-1], 1e-300), 0.0)
    return np.clip(h[: cap + 1], 0.0, 1.0 - 1e-12)


@njit(cache=False)
def _gap_forward(seq, P, h):
    """Filtered state before every token; the state is (last two originals, insertions so far in this gap)."""
    L, n, J = len(seq), P.shape[0], len(h) - 1
    F0 = np.zeros((L + 1, J + 1))
    F1 = np.zeros((L + 1, n, J + 1))
    FA = np.zeros((L + 1, n, n, J + 1))
    F0[0, 0] = 1.0
    for i in range(L):
        c = seq[i]
        f0, f1, fA = F0[i], F1[i], FA[i]
        n0, n1, nA = F0[i + 1], F1[i + 1], FA[i + 1]
        for j in range(J + 1):
            jj = min(j + 1, J)
            w = h[j] / n  # one more insertion (a uniform letter)
            u = 1.0 - h[j]  # the gap closes: the token is the next original
            if f0[j] > 0:
                n0[jj] += f0[j] * w
                n1[c, 0] += f0[j] * u / n
            for a in range(n):
                if f1[a, j] > 0:
                    n1[a, jj] += f1[a, j] * w
                    nA[a, c, 0] += f1[a, j] * u / n
                for b in range(n):
                    v = fA[a, b, j]
                    if v > 0:
                        nA[a, b, jj] += v * w
                        p = P[a, b, c]
                        if p > 0:
                            nA[b, c, 0] += v * u * p
        z = n0.sum() + n1.sum() + nA.sum()
        n0 /= z
        n1 /= z
        nA /= z
    return F0, F1, FA


@njit(cache=False)
def _gap_backward(seq, P, h, F0, F1, FA, end_factor):
    """P(token is an insertion | sequence) from the forward messages."""
    L, n, J = len(seq), P.shape[0], len(h) - 1
    b0 = np.ones(J + 1)
    b1 = np.ones((n, J + 1))
    bA = np.ones((n, n, J + 1))
    if end_factor:  # the end of the sequence closes the last gap
        for j in range(J + 1):
            b0[j] = 1.0 - h[j]
            b1[:, j] = 1.0 - h[j]
            bA[:, :, j] = 1.0 - h[j]
    post = np.zeros(L)
    nb0 = np.zeros(J + 1)
    nb1 = np.zeros((n, J + 1))
    nbA = np.zeros((n, n, J + 1))
    for i in range(L - 1, -1, -1):
        c = seq[i]
        f0, f1, fA = F0[i], F1[i], FA[i]
        stay, total, m = 0.0, 0.0, 0.0
        for j in range(J + 1):
            jj = min(j + 1, J)
            w = h[j] / n
            u = 1.0 - h[j]
            s = w * b0[jj]
            nb0[j] = s + u / n * b1[c, 0]
            stay += f0[j] * s
            total += f0[j] * nb0[j]
            m = max(m, nb0[j])
            for a in range(n):
                s = w * b1[a, jj]
                nb1[a, j] = s + u / n * bA[a, c, 0]
                stay += f1[a, j] * s
                total += f1[a, j] * nb1[a, j]
                m = max(m, nb1[a, j])
                for b in range(n):
                    s = w * bA[a, b, jj]
                    nbA[a, b, j] = s + u * P[a, b, c] * bA[b, c, 0]
                    v = fA[a, b, j]
                    if v > 0:
                        stay += v * s
                        total += v * nbA[a, b, j]
                    m = max(m, nbA[a, b, j])
        post[i] = stay / total
        for j in range(J + 1):
            b0[j] = nb0[j] / m
            for a in range(n):
                b1[a, j] = nb1[a, j] / m
                for b in range(n):
                    bA[a, b, j] = nbA[a, b, j] / m
    return post


@njit(cache=False)
def _pick(weights, k, u):
    total = 0.0
    for i in range(k):
        total += weights[i]
    target = u * total
    acc = 0.0
    for i in range(k):
        acc += weights[i]
        if acc >= target and weights[i] > 0:
            return i
    for i in range(k - 1, -1, -1):
        if weights[i] > 0:
            return i
    return 0


@njit(cache=False)
def _gap_draw(seq, P, h, F0, F1, FA, uniforms):
    """One joint sample of the insertion mask (backward sampling from the forward messages)."""
    L, n, J = len(seq), P.shape[0], len(h) - 1
    out = np.zeros(L, dtype=np.bool_)
    block = 1 + n + n * n
    w = np.zeros((J + 1) * block)
    at = 0
    for j in range(J + 1):
        u = 1.0 - h[j]
        w[at] = F0[L, j] * u
        at += 1
        for a in range(n):
            w[at] = F1[L, a, j] * u
            at += 1
            for b in range(n):
                w[at] = FA[L, a, b, j] * u
                at += 1
    pick = _pick(w, len(w), uniforms[L])
    j = pick // block
    r = pick % block
    kind, a, b = 0, 0, 0
    if r > 0:
        r -= 1
        a = r // (n + 1)
        if r % (n + 1) == 0:
            kind = 1
        else:
            kind, b = 2, r % (n + 1) - 1
    cand = np.zeros((n + 1) * (J + 1) + 2)
    for i in range(L - 1, -1, -1):
        c = seq[i]
        if j > 0:  # the token was an insertion: the count was j - 1 (or stayed at the cap)
            if j == J:
                for k, jp in enumerate((J - 1, J)):
                    v = F0[i, jp] if kind == 0 else (F1[i, a, jp] if kind == 1 else FA[i, a, b, jp])
                    cand[k] = v * h[jp]
                j = J - 1 if _pick(cand, 2, uniforms[i]) == 0 else J
            else:
                j -= 1
            out[i] = True
        elif kind == 1:  # the first original
            for jp in range(J + 1):
                cand[jp] = F0[i, jp] * (1.0 - h[jp])
            j = _pick(cand, J + 1, uniforms[i])
            kind = 0
        elif kind == 2:  # an original after the context (a', a), or the second original
            k = 0
            for jp in range(J + 1):
                cand[k] = F1[i, a, jp] * (1.0 - h[jp]) / n
                k += 1
            for ap in range(n):
                p = P[ap, a, c]
                for jp in range(J + 1):
                    cand[k] = FA[i, ap, a, jp] * (1.0 - h[jp]) * p
                    k += 1
            which = _pick(cand, k, uniforms[i])
            if which <= J:
                kind, j = 1, which
            else:
                which -= J + 1
                b = a
                a = which // (J + 1)
                j = which % (J + 1)
    return out


@njit(cache=False)
def _gap_both(seq, P, h, uniforms):
    F0, F1, FA = _gap_forward(seq, P, h)
    return _gap_backward(seq, P, h, F0, F1, FA, True), _gap_draw(seq, P, h, F0, F1, FA, uniforms)


@njit(cache=False)
def _gap_posterior_only(seq, P, h, end_factor):
    F0, F1, FA = _gap_forward(seq, P, h)
    return _gap_backward(seq, P, h, F0, F1, FA, end_factor)


def gap_reference(sources: Sources, P: np.ndarray, q_true: np.ndarray, scratch: Path, refresh: bool) -> dict:
    """Posterior and one joint sample under the prior of the coupling, for all sources (cached)."""
    path = scratch / "gap-reference.npz"
    key = hashlib.sha1(sources.letters.tobytes() + sources.inserted.tobytes()).hexdigest()[:16]
    if path.exists() and not refresh:
        z = np.load(path)
        if str(z["key"]) == key:
            return {"p_gap": z["p_gap"], "sample": z["sample"], "check": float(z["check"])}
    start = time.time()
    rho = sources.rho
    worst = 0.0
    for i in range(min(10, len(sources))):  # with a constant hazard this is the model of editflows.exact
        post = _gap_posterior_only(sources.seq(i), P, np.full(GAP_CAP + 1, rho[i]), False)
        worst = max(worst, float(np.abs(post - q_true[sources.span(i)]).max()))
    if worst > 1e-5:
        raise RuntimeError(f"the gap model with a constant hazard does not reproduce q_true (max difference {worst:.2e})")
    rng = np.random.default_rng(0)
    p_gap = np.zeros(len(sources.letters))
    sample = np.zeros(len(sources.letters), dtype=bool)
    for i in range(len(sources)):
        seq = sources.seq(i)
        k = int(sources.mask(i).sum())
        p_gap[sources.span(i)], sample[sources.span(i)] = _gap_both(seq, P, gap_hazards(k, len(seq) - k), rng.random(len(seq) + 1))
        if i % 1000 == 999:
            log(f"  coupling-prior reference: {i + 1} sources, {time.time() - start:.0f} s")
    np.savez_compressed(path, key=key, p_gap=p_gap, sample=sample, check=worst)
    return {"p_gap": p_gap, "sample": sample, "check": worst}


# ---- helpers -----------------------------------------------------------------------------------------------------


def load_npz(path: Path) -> dict | None:
    """All arrays of a file, None if it is missing or (after one retry) unreadable."""
    if not path.exists():
        return None
    for attempt in range(2):
        try:
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        except Exception:  # noqa: BLE001 - a file that is still being written
            if attempt == 0:
                time.sleep(8)
    return None


def boot_weights(m: int, n_boot: int, key: str = "") -> np.ndarray:
    """How often every one of `m` sequences is drawn in each of `n_boot` resamples (the same for the same key)."""
    rng = np.random.default_rng([m, n_boot, zlib.crc32(key.encode())])
    return rng.multinomial(m, np.full(m, 1.0 / m), size=n_boot).astype(np.float64)


def interval(samples: np.ndarray):
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return lo, hi


def per_sequence(values: np.ndarray, seq_of: np.ndarray, n_seq: int, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return np.bincount(seq_of, weights=values, minlength=n_seq)
    return np.bincount(seq_of[mask], weights=None if values is None else values[mask], minlength=n_seq).astype(np.float64)


def x_runs(pattern: str) -> list[int]:
    """Lengths of the runs of deleted originals of a cluster, in target coordinates."""
    return [len(r) for r in pattern.replace("+", "").replace("_", "").split(".") if r]


def analyse(tag: str, sources: Sources, deleted: np.ndarray, first: int, eligible: np.ndarray, scratch: Path, refresh: bool):
    """Per-sequence counts (columns `COLS`) and the table of clusters of one set of deletions (cached)."""
    n = int(sources.offsets[first])
    deleted = np.ascontiguousarray(deleted[:n].astype(bool))
    key = hashlib.sha1(deleted.tobytes()).hexdigest()[:16]
    npz, csv = scratch / f"{tag}.npz", scratch / f"{tag}-clusters.csv"
    if npz.exists() and csv.exists() and not refresh:
        z = np.load(npz)
        if str(z["key"]) == key:
            return z["X"], pd.read_csv(csv, dtype={"pattern": str, "kind": str}, keep_default_na=False)
    inserted = sources.inserted[:n]
    off = sources.offsets[: first + 1]
    X = np.zeros((first, len(COLS)))
    X[:, IDX["letters"]] = np.diff(off)
    X[:, IDX["inserted"]] = np.add.reduceat(inserted.astype(np.int64), off[:-1])
    X[:, IDX["deleted"]] = np.add.reduceat(deleted.astype(np.int64), off[:-1])
    X[:, IDX["tp"]] = np.add.reduceat((inserted & deleted).astype(np.int64), off[:-1])
    tables = []
    for i in range(first):
        span = slice(int(off[i]), int(off[i + 1]))
        letters, mask, gone = sources.letters[span].astype(np.int64), inserted[span], deleted[span]
        counts = canonical_counts(letters, mask, gone)
        X[i, IDX["s_wrong"]], X[i, IDX["s_missed"]], X[i, IDX["equal"]] = counts["wrong"], counts["missed"], counts["equal"]
        table = clusters(letters, mask, gone, eligible, gap=2)
        if len(table):
            table.insert(0, "source", i)
            tables.append(table)
    if tables:
        table = pd.concat(tables, ignore_index=True)
    else:
        table = pd.DataFrame(columns=["source", "cluster", "start", "end", "n_x", "n_miss", "n_plus", "n_dot", "violated", "kind", "pattern"])
    table["longest"] = [max(x_runs(p), default=0) for p in table.pattern]
    for k in KINDS:
        sel = table[table.kind == k]
        src = sel.source.to_numpy(dtype=np.int64)
        X[:, IDX[f"clusters_{k}"]] = np.bincount(src, minlength=first)
        X[:, IDX[f"x_{k}"]] = np.bincount(src, weights=sel.n_x.to_numpy(dtype=float), minlength=first)
        X[:, IDX[f"miss_{k}"]] = np.bincount(src, weights=sel.n_miss.to_numpy(dtype=float), minlength=first)
    S = X.sum(0)
    assert abs(sum(S[IDX[f"x_{k}"]] for k in KINDS) - (S[IDX["deleted"]] - S[IDX["tp"]])) < 0.5, "clusters do not add up (x)"
    assert abs(sum(S[IDX[f"miss_{k}"]] for k in KINDS) - (S[IDX["inserted"]] - S[IDX["tp"]])) < 0.5, "clusters do not add up (_)"
    np.savez_compressed(npz, key=key, X=X)
    table.to_csv(csv, index=False)
    return X, table


def metrics(S: np.ndarray, m: int) -> dict:
    """All budget metrics from column sums `S` (the last axis runs over `COLS`) of `m` sequences."""
    c = lambda name: S[..., IDX[name]]  # noqa: E731
    letters, ins, dele, tp = c("letters"), c("inserted"), c("deleted"), c("tp")
    tps = dele - c("s_wrong")  # deletions that are right on the string
    per = 1000.0 / letters
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            ("precision", "all"): tp / dele, ("recall", "all"): tp / ins, ("f1", "all"): 2 * tp / (dele + ins),
            ("string_precision", "all"): tps / dele, ("string_recall", "all"): tps / ins,
            ("string_f1", "all"): 2 * tps / (dele + ins), ("exact_match", "all"): c("equal") / m,
            ("errors_per_1000", "all"): (dele + ins - 2 * tp) * per, ("x_per_1000", "all"): (dele - tp) * per,
            ("miss_per_1000", "all"): (ins - tp) * per,
            ("string_errors_per_1000", "all"): (c("s_wrong") + c("s_missed")) * per,
            ("label_only_per_1000", "all"): (dele + ins - 2 * tp - c("s_wrong") - c("s_missed")) * per,
            ("clusters_per_1000", "all"): sum(c(f"clusters_{k}") for k in KINDS) * per,
        }
        for k in KINDS:
            out[("errors_per_1000", k)] = (c(f"x_{k}") + c(f"miss_{k}")) * per
            out[("clusters_per_1000", k)] = c(f"clusters_{k}") * per
            out[("x_per_1000", k)] = c(f"x_{k}") * per
            out[("miss_per_1000", k)] = c(f"miss_{k}") * per
    return out


def budget_rows(subset: str, ids: np.ndarray, rungs: dict, pairs: list, n_boot: int) -> list[dict]:
    """Every metric of every rung on the sources `ids`, and paired differences, with bootstrap intervals."""
    m = len(ids)
    if m == 0:
        return []
    W = boot_weights(m, n_boot, subset)
    point, boot, rows = {}, {}, []
    for name, X in rungs.items():
        if X.shape[0] <= ids.max():
            continue
        Xs = X[ids]
        point[name], boot[name] = metrics(Xs.sum(0), m), metrics(W @ Xs, m)
        letters = int(Xs[:, 0].sum())
        for key, value in point[name].items():
            lo, hi = interval(boot[name][key])
            rows.append({"subset": subset, "rung": name, "metric": key[0], "kind": key[1], "value": float(value),
                         "lo": float(lo), "hi": float(hi), "n_sources": m, "n_letters": letters})
    for a, b in pairs:
        if a in point and b in point:
            letters = int(rungs[a][ids][:, 0].sum())
            for key in point[a]:
                lo, hi = interval(boot[a][key] - boot[b][key])
                rows.append({"subset": subset, "rung": f"{a} - {b}", "metric": key[0], "kind": key[1],
                             "value": float(point[a][key] - point[b][key]), "lo": float(lo), "hi": float(hi),
                             "n_sources": m, "n_letters": letters})
    return rows


class Budget:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.index = {(r.subset, r.rung, r.metric, r.kind): (r.value, r.lo, r.hi) for r in frame.itertuples()}

    def get(self, subset, rung, metric, kind="all"):
        return self.index.get((subset, rung, metric, kind), (np.nan, np.nan, np.nan))

    def has(self, subset, rung):
        return (subset, rung, "f1", "all") in self.index

    def text(self, subset, rung, metric, kind="all", digits=1):
        v, lo, hi = self.get(subset, rung, metric, kind)
        return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


# ---- 1. reliability of the reference posteriors -----------------------------------------------------------------


def reliability(sources: Sources, posteriors: dict, noise_bin: np.ndarray, n_boot: int):
    """Reliability in 20 bins (overall and by noise level) and the test that a correct posterior must also pass
    after conditioning on the previous letter."""
    n_seq, seq_of, ins = len(sources), sources.seq_of, sources.inserted.astype(np.float64)
    subsets = [("all", np.arange(n_seq))] + [(f"noise {NOISE_TEXT[b]}", np.flatnonzero(noise_bin == b)) for b in range(len(NOISE_TEXT))]
    rows, context_rows, summary = [], [], {}
    rng = np.random.default_rng(1)
    first_letter = np.zeros(len(ins), dtype=bool)
    first_letter[sources.offsets[:-1]] = True
    for name, q in posteriors.items():
        q = q.astype(np.float64)
        b = np.minimum((q * 20).astype(np.int64), 19)
        idx = seq_of * 20 + b
        shape = (n_seq, 20)
        cnt = np.bincount(idx, minlength=n_seq * 20).reshape(shape).astype(np.float64)
        sq = np.bincount(idx, weights=q, minlength=n_seq * 20).reshape(shape)
        si = np.bincount(idx, weights=ins, minlength=n_seq * 20).reshape(shape)
        null = []  # what the calibration error would be if the labels were independent draws from q
        for _ in range(10):
            null.append(np.bincount(idx, weights=(rng.random(len(q)) < q).astype(np.float64), minlength=n_seq * 20).reshape(shape))
        loss = -np.log(np.clip(np.where(ins > 0, q, 1 - q), 1e-12, 1.0))
        loss_seq = per_sequence(loss, seq_of, n_seq)
        letters_seq = np.diff(sources.offsets).astype(np.float64)
        summary[name] = {}
        for subset, ids in subsets:
            if len(ids) == 0:
                continue
            W = boot_weights(len(ids), n_boot, "reliability " + subset)
            C, Q, I = cnt[ids].sum(0), sq[ids].sum(0), si[ids].sum(0)
            Cb, Qb, Ib = W @ cnt[ids], W @ sq[ids], W @ si[ids]
            with np.errstate(divide="ignore", invalid="ignore"):
                dev, devb = (I - Q) / C, (Ib - Qb) / Cb
            lo, hi = interval(devb)
            for k in range(20):
                rows.append({"posterior": name, "subset": subset, "bin": k, "n_letters": int(C[k]),
                             "stated": Q[k] / C[k] if C[k] else np.nan, "share_inserted": I[k] / C[k] if C[k] else np.nan,
                             "deviation": dev[k], "lo": lo[k], "hi": hi[k], "n_sources": len(ids)})
            ece = np.abs(I - Q).sum() / C.sum()
            ece_lo, ece_hi = interval(np.abs(Ib - Qb).sum(1) / Cb.sum(1))
            ece_null = float(np.mean([np.abs(z[ids].sum(0) - Q).sum() / C.sum() for z in null]))
            big = C >= 1000
            worst = int(np.flatnonzero(big)[np.argmax(np.abs(dev[big]))])
            bias_lo, bias_hi = interval((Ib.sum(1) - Qb.sum(1)) / Cb.sum(1))
            ll_lo, ll_hi = interval((W @ loss_seq[ids]) / (W @ letters_seq[ids]))
            summary[name][subset] = {
                "n_sources": len(ids), "n_letters": int(C.sum()),
                "calibration_error": float(ece), "calibration_error_ci": [float(ece_lo), float(ece_hi)],
                "calibration_error_if_exact": ece_null,
                "largest_deviation": float(dev[worst]), "largest_deviation_ci": [float(lo[worst]), float(hi[worst])],
                "largest_deviation_at_stated": float(Q[worst] / C[worst]),
                "mean_bias": float((I.sum() - Q.sum()) / C.sum()), "mean_bias_ci": [float(bias_lo), float(bias_hi)],
                "log_loss": float(loss_seq[ids].sum() / letters_seq[ids].sum()), "log_loss_ci": [float(ll_lo), float(ll_hi)],
            }
        # conditional on what the previous letter evidently is: a function of the source only, so a correct
        # posterior is calibrated in both groups
        previous = np.concatenate([[0.5], q[:-1]])
        for context, mask in (("after an evident original", (previous < 0.02) & ~first_letter),
                              ("after an evident insertion", (previous > 0.98) & ~first_letter)):
            c_seq = per_sequence(None, seq_of, n_seq, mask)
            q_seq = per_sequence(q, seq_of, n_seq, mask)
            i_seq = per_sequence(ins, seq_of, n_seq, mask)
            for subset, ids in subsets:
                if len(ids) == 0 or c_seq[ids].sum() == 0:
                    continue
                W = boot_weights(len(ids), n_boot, "reliability " + subset)
                C, Q, I = c_seq[ids].sum(), q_seq[ids].sum(), i_seq[ids].sum()
                lo, hi = interval((W @ i_seq[ids] - W @ q_seq[ids]) / (W @ c_seq[ids]))
                context_rows.append({"posterior": name, "subset": subset, "context": context, "n_letters": int(C),
                                     "stated": Q / C, "share_inserted": I / C, "deviation": (I - Q) / C, "lo": float(lo),
                                     "hi": float(hi), "n_sources": len(ids)})
    return pd.DataFrame(rows), pd.DataFrame(context_rows), summary


# ---- 2. the 2q(1-q) law -------------------------------------------------------------------------------------------


def law_table(reference: str, q: np.ndarray, rung: str, deleted: np.ndarray, first: int, sources: Sources, n_boot: int,
              truth: np.ndarray | None = None):
    """Observed errors of a rung in bins of a posterior `q`, against 2q(1-q) (wrong deletions and misses: q(1-q) each).

    `truth`: a better posterior p of the labels; then the expectation q(1-p) + p(1-q) for a sampler that draws
    from q is given too."""
    n = int(sources.offsets[first])
    q = q[:n].astype(np.float64)
    ins, dele = sources.inserted[:n], deleted[:n].astype(bool)
    seq_of = sources.seq_of[:n]
    nb = len(Q_EDGES) - 1
    b = np.clip(np.digitize(q, Q_EDGES) - 1, 0, nb - 1)
    idx = seq_of * nb + b

    def table(values):
        return np.bincount(idx, weights=values, minlength=first * nb).reshape(first, nb)

    T = {"n": table(np.ones(n)), "q": table(q), "law": table(2 * q * (1 - q)), "x": table((dele & ~ins).astype(np.float64)),
         "miss": table((~dele & ins).astype(np.float64))}
    if truth is not None:
        p = truth[:n].astype(np.float64)
        T["cross"] = table(q * (1 - p) + p * (1 - q))
    W = boot_weights(first, n_boot, f"law {first}")
    S = {k: v.sum(0) for k, v in T.items()}
    Sb = {k: W @ v for k, v in T.items()}
    rows = []
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_lo, rate_hi = interval((Sb["x"] + Sb["miss"]) / Sb["n"])
        ratio_lo, ratio_hi = interval((Sb["x"] + Sb["miss"]) / Sb["law"])
        for k in range(nb):
            if S["n"][k] == 0:
                continue
            rows.append({"reference": reference, "rung": rung, "bin": k, "q_from": Q_EDGES[k], "q_to": min(Q_EDGES[k + 1], 1.0),
                         "n_letters": int(S["n"][k]), "mean_q": S["q"][k] / S["n"][k], "wrong_rate": S["x"][k] / S["n"][k],
                         "missed_rate": S["miss"][k] / S["n"][k], "error_rate": (S["x"][k] + S["miss"][k]) / S["n"][k],
                         "error_rate_lo": rate_lo[k], "error_rate_hi": rate_hi[k], "law_rate": S["law"][k] / S["n"][k],
                         "observed_over_law": (S["x"][k] + S["miss"][k]) / S["law"][k], "ratio_lo": ratio_lo[k], "ratio_hi": ratio_hi[k],
                         "cross_rate": S["cross"][k] / S["n"][k] if truth is not None else np.nan, "n_sources": first})
    letters = S["n"].sum()
    errors, errors_b = S["x"].sum() + S["miss"].sum(), Sb["x"].sum(1) + Sb["miss"].sum(1)
    lo, hi = interval(errors_b / Sb["law"].sum(1))
    total = {"n_sources": first, "n_letters": int(letters), "errors": int(errors), "law_sum": float(S["law"].sum()),
             "observed_over_law": float(errors / S["law"].sum()), "observed_over_law_ci": [float(lo), float(hi)],
             "wrong": int(S["x"].sum()), "missed": int(S["miss"].sum()), "law_each": float(S["law"].sum() / 2)}
    if truth is not None:
        lo, hi = interval(errors_b / Sb["cross"].sum(1))
        total.update(cross_sum=float(S["cross"].sum()), observed_over_cross=float(errors / S["cross"].sum()),
                     observed_over_cross_ci=[float(lo), float(hi)])
    total["excess_per_1000"] = {}
    for region, (a, z) in REGIONS.items():
        entry = {}
        for what, obs, obs_b, share in (("errors", S["x"] + S["miss"], Sb["x"] + Sb["miss"], 1.0), ("wrong", S["x"], Sb["x"], 0.5),
                                        ("missed", S["miss"], Sb["miss"], 0.5)):
            value = 1000 * (obs[a:z].sum() - share * S["law"][a:z].sum()) / letters
            lo, hi = interval(1000 * (obs_b[:, a:z].sum(1) - share * Sb["law"][:, a:z].sum(1)) / Sb["n"].sum(1))
            entry[what] = [float(value), float(lo), float(hi)]
        entry["observed_per_1000"] = float(1000 * (S["x"][a:z].sum() + S["miss"][a:z].sum()) / letters)
        entry["law_per_1000"] = float(1000 * S["law"][a:z].sum() / letters)
        entry["share_of_letters"] = float(S["n"][a:z].sum() / letters)
        total["excess_per_1000"][region] = entry
    return rows, total


def map_check(q: np.ndarray, deleted: np.ndarray, first: int, sources: Sources, n_boot: int) -> dict:
    """Errors of a decision against the sum of min(q, 1 - q), the errors the best decision under q expects."""
    n = int(sources.offsets[first])
    q = q[:n].astype(np.float64)
    seq_of = sources.seq_of[:n]
    err = per_sequence((deleted[:n].astype(bool) != sources.inserted[:n]).astype(np.float64), seq_of, first)
    best = per_sequence(np.minimum(q, 1 - q), seq_of, first)
    W = boot_weights(first, n_boot, f"law {first}")
    lo, hi = interval((W @ err) / (W @ best))
    return {"n_sources": first, "errors": int(err.sum()), "sum_min": float(best.sum()), "observed_over_expected": float(err.sum() / best.sum()),
            "observed_over_expected_ci": [float(lo), float(hi)]}


# ---- 4. the label-only class --------------------------------------------------------------------------------------


def twin_flags(sources: Sources):
    """For every letter: does it equal an adjacent source letter / the nearest original on either side?"""
    letters, ins = sources.letters.astype(np.int64), sources.inserted
    n = len(letters)
    pos = np.arange(n)
    seq_of = sources.seq_of
    start, stop = sources.offsets[:-1][seq_of], sources.offsets[1:][seq_of]
    left = np.maximum.accumulate(np.where(~ins, pos, -1))
    left = np.where(ins, left, np.concatenate([[-1], left[:-1]]))  # for an original: the original before it
    right = np.minimum.accumulate(np.where(~ins, pos, n)[::-1])[::-1]
    right = np.where(ins, right, np.concatenate([right[1:], [n]]))
    eq_left = (left >= start) & (letters[np.clip(left, 0, n - 1)] == letters)
    eq_right = (right < stop) & (letters[np.clip(right, 0, n - 1)] == letters)
    adj_left = (pos > start) & (letters[np.maximum(pos - 1, 0)] == letters)
    adj_right = (pos < stop - 1) & (letters[np.minimum(pos + 1, n - 1)] == letters)
    adj_original = (adj_left & ~ins[np.maximum(pos - 1, 0)]) | (adj_right & ~ins[np.minimum(pos + 1, n - 1)])
    return adj_left | adj_right, eq_left | eq_right, eq_left & eq_right, adj_original


def label_only(sources: Sources, rungs: dict, deleted_of: dict, n_boot: int):
    """Prediction P1: an inserted letter that equals a neighbouring original can be swapped with it for free.

    "Twin" below: the letter equals the nearest original on its left or right. Other insertions may lie between the
    two; they are deleted in either reading, so the two readings are still the same string."""
    adjacent, twin, both, adj_original = twin_flags(sources)
    ins = sources.inserted
    n_seq, seq_of = len(sources), sources.seq_of
    groups = {"twin of an adjacent original": adj_original, "twin of an original behind other insertions": twin & ~adj_original,
              "identical only to an adjacent insertion": adjacent & ~twin, "no identical neighbour": ~adjacent & ~twin}
    summary = {
        "n_inserted": int(ins.sum()),
        "share_equal_to_an_adjacent_source_letter": float(adjacent[ins].mean()),
        "share_equal_to_an_adjacent_original": float(adj_original[ins].mean()),
        "share_of_identical_neighbours_that_are_originals": float(adj_original[ins].sum() / adjacent[ins].sum()),
        "share_equal_to_the_nearest_original_on_a_side": float(twin[ins].mean()),
        "share_equal_on_both_sides": float(both[ins].mean()),
        "theory_one_minus_19_20_squared": 1 - (19 / 20) ** 2,
        "groups_share_of_insertions": {g: float(m[ins].mean()) for g, m in groups.items()},
    }
    twin_seq = per_sequence(None, seq_of, n_seq, ins & twin)
    rows = []
    for name, X in rungs.items():
        first = X.shape[0]
        n = int(sources.offsets[first])
        dele = deleted_of[name][:n].astype(bool)
        m_ins, m_twin = ins[:n], twin[:n]
        W = boot_weights(first, n_boot, f"twins {first}")
        c = lambda col: X[:, IDX[col]]  # noqa: E731
        letters = c("letters")
        predicted = twin_seq[:first]  # one wrong deletion + one miss with probability 1/2 per such letter
        z_letters = c("x_Z") + c("miss_Z")
        label_minus_string = c("deleted") + c("inserted") - 2 * c("tp") - c("s_wrong") - c("s_missed")
        tp, dele_n, ins_n = c("tp"), c("deleted"), c("inserted")
        tps = dele_n - c("s_wrong")

        def stat(num, den, scale=1.0):
            lo, hi = interval(scale * (W @ num) / (W @ den))
            return float(scale * num.sum() / den.sum()), float(lo), float(hi)

        miss_twin = per_sequence(None, seq_of[:n], first, m_ins & m_twin & ~dele)
        n_twin = per_sequence(None, seq_of[:n], first, m_ins & m_twin)
        miss_other = per_sequence(None, seq_of[:n], first, m_ins & ~m_twin & ~dele)
        n_other = per_sequence(None, seq_of[:n], first, m_ins & ~m_twin)
        entries = {
            "predicted label-only letters per 1000": stat(predicted, letters, 1000),
            "class Z letters per 1000": stat(z_letters, letters, 1000),
            "label errors that are not string errors per 1000": stat(label_minus_string, letters, 1000),
            "class Z over prediction": stat(z_letters, predicted),
            "label f1": stat(2 * tp, dele_n + ins_n),
            "string f1 predicted (label f1 + twins)": stat(2 * tp + predicted, dele_n + ins_n),
            "string f1 measured": stat(2 * tps, dele_n + ins_n),
            "share of twin insertions missed": stat(miss_twin, n_twin),
            "share of other insertions missed": stat(miss_other, n_other),
        }
        for group, mask in groups.items():
            g = m_ins & mask[:n]
            entries[f"share missed: {group}"] = stat(per_sequence(None, seq_of[:n], first, g & ~dele), per_sequence(None, seq_of[:n], first, g))
        for quantity, (value, lo, hi) in entries.items():
            rows.append({"rung": name, "quantity": quantity, "value": value, "lo": lo, "hi": hi, "n_sources": first})
    return pd.DataFrame(rows), summary


# ---- 5. holes -----------------------------------------------------------------------------------------------------


def law_survival(law: str, L: np.ndarray, r: float = 0.5) -> np.ndarray:
    """P(size > L | size >= 2) for L >= 1 under the three descriptions of a hole."""
    L = np.asarray(L, dtype=np.float64)
    keep = 1 - V_WINDOW**2
    if law == "junction":
        return keep ** (L * (L + 1) / 2 - 1)
    if law == "greedy":
        return keep ** (L - 1)
    return r ** (L - 1)


def law_loglik(law: str, sizes: np.ndarray, r: float = 0.5) -> float:
    """Mean log-probability of the sizes (>= 2) under a law."""
    sizes = np.asarray(sizes, dtype=np.float64)
    pmf = law_survival(law, sizes - 1, r) - law_survival(law, sizes, r)
    return float(np.log(np.maximum(pmf, 1e-300)).mean())


def geometric_ratio(sizes: np.ndarray) -> float:
    m = float(np.mean(sizes) - 2)
    return m / (1 + m) if m > 0 else 0.0


def holes(tables: dict, firsts: dict, rho: np.ndarray, noise_bin: np.ndarray, n_boot: int, top: int = 40):
    """Sizes of the clusters of kind R: wrongly deleted originals, and the longest contiguous run of them."""
    rows, summary = [], {}
    L = np.arange(1, top)
    for name, table in tables.items():
        first = firsts[name]
        R = table[table.kind == "R"]
        summary[name] = {"n_sources": first, "n_holes": int(len(R))}
        if len(R) < 20:
            continue
        src = R.source.to_numpy(dtype=np.int64)
        W = boot_weights(first, n_boot, f"holes {first}")
        for measure, sizes in (("wrongly deleted originals", R.n_x.to_numpy()), ("longest contiguous run", R.longest.to_numpy())):
            H = np.zeros((first, top + 1))
            np.add.at(H, (src, np.minimum(sizes, top)), 1.0)
            if measure == "longest contiguous run":
                H[:, :2] = 0  # the laws describe a contiguous hole of at least two letters
            counts, counts_b = H.sum(0), W @ H
            total, total_b = counts.sum(), counts_b.sum(1)
            if total < 20:
                continue
            surv = 1 - np.cumsum(counts)[L] / total
            lo, hi = interval(1 - np.cumsum(counts_b, 1)[:, L] / total_b[:, None])
            sizes2 = sizes[sizes >= 2]
            grid = np.arange(top + 1)
            mean_b = (counts_b * grid).sum(1) / total_b
            r_b = (mean_b - 2) / (mean_b - 1)
            half = src[sizes >= 2] < first // 2  # ratio fitted on the first half of the sources, laws scored on the second
            r_fit = geometric_ratio(sizes2[half]) if half.sum() > 5 else float("nan")
            held = sizes2[~half]
            entry = {
                "n": int(total), "mean": float(sizes2.mean()), "mean_ci": [float(v) for v in interval(mean_b)],
                "median": float(np.median(sizes2)),
                "share_at_least_5": float((sizes2 >= 5).mean()), "share_at_least_5_ci": [float(v) for v in interval(counts_b[:, 5:].sum(1) / total_b)],
                "share_at_least_10": float((sizes2 >= 10).mean()),
                "largest": int(sizes2.max()), "geometric_ratio": geometric_ratio(sizes2), "geometric_ratio_ci": [float(v) for v in interval(r_b)],
                "loglik_per_hole_second_half": {"n": int(len(held)), "junction repair": law_loglik("junction", held) if len(held) else None,
                                                "greedy deletion": law_loglik("greedy", held) if len(held) else None,
                                                "geometric (ratio from first half)": law_loglik("geometric", held, r_fit) if len(held) and r_fit > 0 else None,
                                                "ratio_first_half": r_fit},
                "largest_gap_to_law": {law: float(np.abs(surv - law_survival(law, L, geometric_ratio(sizes2))).max())
                                       for law in ("junction", "greedy", "geometric")},
            }
            by_noise = {}
            for bin_id, text in enumerate(NOISE_TEXT):
                sel = (noise_bin[src] == bin_id) & (sizes >= 2)
                if sel.sum() >= 20:
                    rho_mean = float(rho[src[sel]].mean())
                    by_noise[text] = {"n": int(sel.sum()), "mean": float(sizes[sel].mean()), "geometric_ratio": geometric_ratio(sizes[sel]),
                                      "share_inserted_of_these_sources": rho_mean, "static_ratio_rho_over_4_1_minus_rho": rho_mean / (4 * (1 - rho_mean))}
            entry["by_noise"] = by_noise
            summary[name][measure] = entry
            for k, size in enumerate(L):
                rows.append({"rung": name, "measure": measure, "size": int(size), "survival": surv[k], "lo": lo[k], "hi": hi[k],
                             "n_holes": int(total), "n_sources": first})
    for law in ("junction", "greedy"):
        for size in L:
            rows.append({"rung": f"law: {law}", "measure": "law", "size": int(size), "survival": float(law_survival(law, size)),
                         "lo": np.nan, "hi": np.nan, "n_holes": 0, "n_sources": 0})
    return pd.DataFrame(rows), summary


def holes_by_noise(tables: dict, firsts: dict, sources: Sources, n_boot: int):
    """How the holes depend on the noise level r = rho / (1 - rho).

    A static alternative reading with one more deleted original costs one more factor r (times letter probabilities
    that do not depend on the noise), so both the factor per further letter and the frequency of holes should rise
    with r; a hole that grows by a cascade after one wrong deletion needs no further insertion."""
    rho = sources.rho
    lengths = np.diff(sources.offsets).astype(np.float64)
    nb = len(RHO_EDGES) - 1
    rho_bin = np.clip(np.digitize(rho, RHO_EDGES) - 1, 0, nb - 1)
    onehot = (rho_bin[:, None] == np.arange(nb)[None]).astype(np.float64)
    rows, summary = [], {}
    for name, table in tables.items():
        first = firsts[name]
        R = table[(table.kind == "R") & (table.n_x >= 2)]
        if len(R) < 50:
            continue
        src = R.source.to_numpy(dtype=np.int64)
        size = R.n_x.to_numpy(dtype=np.float64)
        W = boot_weights(first, n_boot, f"holes {first}")
        hot = onehot[:first]
        per_seq = {"holes": np.bincount(src, minlength=first).astype(np.float64), "extra": np.bincount(src, weights=size - 2, minlength=first),
                   "letters": lengths[:first], "rho_letters": rho[:first] * lengths[:first]}
        for s, label in ((2, "size 2"), (3, "size 3"), (4, "size 4 or more")):
            sel = size == s if s < 4 else size >= 4
            per_seq[label] = np.bincount(src[sel], minlength=first).astype(np.float64)
        point = {k: (v[:, None] * hot).sum(0) for k, v in per_seq.items()}
        boot = {k: W @ (v[:, None] * hot) for k, v in per_seq.items()}

        def derived(S):
            with np.errstate(divide="ignore", invalid="ignore"):
                mean_extra = S["extra"] / S["holes"]
                out = {"ratio": mean_extra / (1 + mean_extra), "holes_per_1000": 1000 * S["holes"] / S["letters"]}
                for label in ("size 2", "size 3", "size 4 or more"):
                    out[f"{label} per 1000"] = 1000 * S[label] / S["letters"]
                rho_mean = S["rho_letters"] / S["letters"]
                out["log_r"] = np.log(rho_mean / (1 - rho_mean))
            return out

        d, db = derived(point), derived(boot)
        use = point["holes"] >= 30

        def slope(y, x):
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(y[..., use])
            x = x[..., use]
            xc = x - x.mean(-1, keepdims=True)
            with np.errstate(invalid="ignore"):  # a resample without holes in a bin gives no slope
                return (xc * (y - y.mean(-1, keepdims=True))).sum(-1) / (xc**2).sum(-1)

        entry = {"n_holes": int(len(R)), "noise_bins_used": int(use.sum())}
        if use.sum() >= 3:
            for key in ("ratio", "holes_per_1000", "size 2 per 1000", "size 3 per 1000", "size 4 or more per 1000"):
                lo, hi = interval(slope(db[key], db["log_r"]))
                entry[f"slope of log({key}) in log r"] = [float(slope(d[key], d["log_r"])), float(lo), float(hi)]
        summary[name] = entry
        for b in range(nb):
            if point["holes"][b] == 0:
                continue
            row = {"rung": name, "rho_from": RHO_EDGES[b], "rho_to": min(RHO_EDGES[b + 1], 0.5), "mean_rho": point["rho_letters"][b] / point["letters"][b],
                   "n_holes": int(point["holes"][b]), "n_sources": int(hot[:, b].sum())}
            for key in ("ratio", "holes_per_1000", "size 2 per 1000", "size 3 per 1000", "size 4 or more per 1000"):
                lo, hi = interval(db[key][:, b])
                row[key], row[f"{key} lo"], row[f"{key} hi"] = float(d[key][b]), float(lo), float(hi)
            rows.append(row)
    return pd.DataFrame(rows), summary


# ---- 6. step-size sweep -------------------------------------------------------------------------------------------


def sweep(families: dict, references: dict, n_boot: int):
    """Errors against the step size h = 1 / n, a straight line in h, and its value at h = 0 against the exact samples.

    `families[name][n]` and `references[name]` are per-sequence count tables on the same sources."""
    rows, summary = [], {}
    sizes = [X.shape[0] for fam in families.values() for X in fam.values()]
    if not sizes:
        return pd.DataFrame(rows), summary
    m = min(sizes)
    W = boot_weights(m, n_boot, f"sweep {m}")
    ref_point = {name: metrics(X[:m].sum(0), m) for name, X in references.items() if X.shape[0] >= m}
    ref_boot = {name: metrics(W @ X[:m], m) for name, X in references.items() if X.shape[0] >= m}
    keys = [("errors_per_1000", "all")] + [("errors_per_1000", k) for k in KINDS] + [("clusters_per_1000", "all"), ("string_errors_per_1000", "all"), ("f1", "all")]
    for name, value in ref_point.items():
        for key in keys:
            lo, hi = interval(ref_boot[name][key])
            rows.append({"family": name, "n_steps": 0, "h": 0.0, "metric": key[0], "kind": key[1], "value": float(value[key]),
                         "lo": float(lo), "hi": float(hi), "what": "reference", "n_sources": m})
    for family, found in families.items():
        steps = sorted(found)
        summary[family] = {"n_sources": m, "steps": steps}
        if not steps:
            continue
        h = np.array([1.0 / n for n in steps])
        point = [metrics(found[n][:m].sum(0), m) for n in steps]
        boot = [metrics(W @ found[n][:m], m) for n in steps]
        for key in keys:
            y = np.array([p[key] for p in point])
            yb = np.stack([b[key] for b in boot], 1)
            lo, hi = interval(yb)
            for k, n in enumerate(steps):
                rows.append({"family": family, "n_steps": n, "h": h[k], "metric": key[0], "kind": key[1], "value": float(y[k]),
                             "lo": float(lo[k]), "hi": float(hi[k]), "what": "measured", "n_sources": m})
            if len(steps) < 3:
                continue
            hc = h - h.mean()
            slope, slope_b = (hc * y).sum() / (hc**2).sum(), (yb * hc).sum(1) / (hc**2).sum()
            icpt, icpt_b = y.mean() - slope * h.mean(), yb.mean(1) - slope_b * h.mean()
            resid = y - (icpt + slope * h)
            entry = {"limit": float(icpt), "limit_ci": [float(v) for v in interval(icpt_b)], "slope_per_unit_h": float(slope),
                     "slope_ci": [float(v) for v in interval(slope_b)], "largest_residual": float(np.abs(resid).max()),
                     "richardson_two_finest": float(2 * y[-1] - y[-2])}
            lo, hi = interval(icpt_b)
            rows.append({"family": family, "n_steps": -1, "h": 0.0, "metric": key[0], "kind": key[1], "value": float(icpt),
                         "lo": float(lo), "hi": float(hi), "what": "limit h -> 0 (line)", "n_sources": m})
            for ref in ref_point:
                lo, hi = interval(icpt_b - ref_boot[ref][key])
                entry[f"limit_minus_{ref}"] = [float(icpt - ref_point[ref][key]), float(lo), float(hi)]
                entry[ref] = float(ref_point[ref][key])
            summary[family]["/".join(key)] = entry
    return pd.DataFrame(rows), summary


# ---- 3b. confounders ----------------------------------------------------------------------------------------------


def confounders(sources: Sources, deleted_of: dict, first_of: dict, noise_bin: np.ndarray, n_boot: int) -> pd.DataFrame:
    """Error letters per 1 000 letters by position in the source and by length of the target within a noise level."""
    off, seq_of = sources.offsets, sources.seq_of
    lengths = np.diff(off)
    pos = np.arange(len(sources.letters)) - off[:-1][seq_of]
    zone = np.where(pos < 5, 0, np.where(pos >= lengths[seq_of] - 5, 2, 1))
    target_len = lengths - np.add.reduceat(sources.inserted.astype(np.int64), off[:-1])
    long = target_len >= np.median(target_len)
    rows = []
    for name, deleted in deleted_of.items():
        first = first_of[name]
        n = int(off[first])
        err = deleted[:n].astype(bool) != sources.inserted[:n]
        W = boot_weights(first, n_boot, f"confounders {first}")
        for z, text in enumerate(("first 5 source letters", "letters in between", "last 5 source letters")):
            mask = zone[:n] == z
            e, c = per_sequence(None, seq_of[:n], first, mask & err), per_sequence(None, seq_of[:n], first, mask)
            lo, hi = interval(1000 * (W @ e) / (W @ c))
            rows.append({"rung": name, "grouping": "position", "group": text, "errors_per_1000": 1000 * e.sum() / c.sum(), "lo": float(lo),
                         "hi": float(hi), "n_sources": first, "n_letters": int(c.sum())})
        e, c = per_sequence(err.astype(np.float64), seq_of[:n], first), lengths[:first].astype(np.float64)
        for b, text in enumerate(NOISE_TEXT):
            for flag, label in ((False, "shorter targets"), (True, "longer targets")):
                ids = np.flatnonzero((noise_bin[:first] == b) & (long[:first] == flag))
                if len(ids) < 5:
                    continue
                Ws = boot_weights(len(ids), n_boot, f"confounders {text} {label}")
                lo, hi = interval(1000 * (Ws @ e[ids]) / (Ws @ c[ids]))
                rows.append({"rung": name, "grouping": "target length within noise level", "group": f"noise {text}, {label}",
                             "errors_per_1000": 1000 * e[ids].sum() / c[ids].sum(), "lo": float(lo), "hi": float(hi), "n_sources": len(ids),
                             "n_letters": int(c[ids].sum())})
    return pd.DataFrame(rows)


# ---- figures ------------------------------------------------------------------------------------------------------


def titled(fig, text: str, width: int = 120):
    fig.suptitle(textwrap.fill(text, width), x=0.01, ha="left")


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {path.name}")


def legend_below(ax, offset: float = 0.2, **kw):
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -offset), **kw)


def figure_reliability(rel: pd.DataFrame, ctx: pd.DataFrame, summary: dict, out: Path):
    names = {"q_true": ("independent marks, true noise level (used by FFBS, I2)", RUNG_COLOR["FFBS"], "s"),
             "q_est": ("independent marks, noise level estimated (used by I3)", RUNG_COLOR["I3"], "v"),
             "p_gap": ("prior of the coupling: insertions per gap counted (used by G)", RUNG_COLOR["G"], "D")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    ax = axes[0]
    ax.axhline(0, color=COLORS[6], lw=1, label="perfectly calibrated")
    for name, (text, color, marker) in names.items():
        d = rel[(rel.posterior == name) & (rel.subset == "all")]
        if d.empty:
            continue
        ax.errorbar(100 * d.stated, 100 * d.deviation, yerr=[100 * (d.deviation - d.lo), 100 * (d.hi - d.deviation)], color=color,
                    marker=marker, capsize=2, label=text)
    ax.set_xlabel("stated probability that a letter is inserted (%), 20 bins")
    ax.set_ylabel("share actually inserted − stated (percentage points)")
    ax.set_title("All letters: posterior computed with the prior ...")
    legend_below(ax)
    ax = axes[1]
    ax.axhline(0, color=COLORS[6], lw=1)
    d = rel[(rel.posterior == "q_true") & rel.subset.str.startswith("noise")]
    for k, text in enumerate(NOISE_TEXT):
        e = d[(d.subset == f"noise {text}") & (d.n_letters >= 2000)]
        if len(e):
            ax.plot(100 * e.stated, 100 * e.deviation, color=COLORS[k], marker="osD^"[k], label=f"{text} of the source inserted")
    ax.set_xlabel("stated probability that a letter is inserted (%)")
    ax.set_ylabel("share actually inserted − stated (points)")
    ax.set_title("Independent-marks posterior by noise level\n(bins with at least 2 000 letters)")
    legend_below(ax, ncol=2)
    ax = axes[2]
    ax.axhline(0, color=COLORS[6], lw=1)
    x = np.arange(len(NOISE_TEXT))
    styles = {"after an evident original": "-", "after an evident insertion": "--"}
    for name in ("q_true", "p_gap"):
        if name not in set(ctx.posterior):
            continue
        text, color, marker = names[name]
        for context, style in styles.items():
            e = ctx[(ctx.posterior == name) & (ctx.context == context) & ctx.subset.str.startswith("noise")]
            e = e.set_index("subset").reindex([f"noise {t}" for t in NOISE_TEXT])
            ax.errorbar(x, 100 * e.deviation, yerr=[100 * (e.deviation - e.lo), 100 * (e.hi - e.deviation)], color=color, ls=style,
                        marker=marker, capsize=2, label=f"{text.split(',')[0].split(':')[0]}: letters {context}")
    ax.set_xticks(x, NOISE_TEXT)
    ax.set_xlabel("share of the source that is inserted")
    ax.set_ylabel("share actually inserted − stated (points)")
    ax.set_title("The cause: after an insertion a further one is rarer\nthan independent marks assume")
    legend_below(ax)
    a, g = summary["q_true"]["all"], summary.get("p_gap", {}).get("all")
    text = (f"The reference posterior with independent insertion marks is not exact for this coupling: calibration error "
            f"{100 * a['calibration_error']:.2f} points (an exact posterior would show {100 * a['calibration_error_if_exact']:.2f} by chance), "
            f"largest deviation {100 * a['largest_deviation']:+.1f} points at a stated {100 * a['largest_deviation_at_stated']:.0f} %")
    if g:
        text += (f". With the coupling's own prior (the number of insertions in a gap is binomial) the calibration error is "
                 f"{100 * g['calibration_error']:.2f} points (by chance: {100 * g['calibration_error_if_exact']:.2f})")
    titled(fig, text + f". n = {a['n_letters']:,} letters of {a['n_sources']:,} sources; whiskers: 95 % bootstrap intervals over sources.", 175)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / "fig1_reliability.png")


def logit_axis(ax):
    ax.set_xscale("logit")
    ax.set_xticks([1e-4, 1e-2, 0.5, 0.99, 0.9999], ["0.01 %", "1 %", "50 %", "99 %", "99.99 %"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(2e-6, 1 - 2e-6)


def figure_law(law: pd.DataFrame, totals: dict, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    grid = 1 / (1 + np.exp(-np.linspace(-13, 13, 400)))
    reference = "p_gap" if any(k[0] == "p_gap" for k in totals) else "q_true"
    ref_text = "coupling-prior posterior" if reference == "p_gap" else "independent-marks posterior"
    panels = [(axes[0], [("p_gap", "G"), ("q_true", "FFBS")], "Exact samples in bins of the posterior they were drawn from"),
              (axes[1], [(reference, r) for r in LADDER], f"All samplers in bins of the {ref_text}")]
    for ax, series, title in panels:
        ax.plot(grid, 2 * grid * (1 - grid), color=COLORS[6], lw=1.2, label="law 2q(1 − q): truth and sample are two draws from the same posterior")
        for ref, rung in series:
            d = law[(law.reference == ref) & (law.rung == rung) & (law.n_letters >= 200)]
            if d.empty:
                continue
            y = d.error_rate.where(d.error_rate > 0)
            ax.errorbar(d.mean_q, y, yerr=[np.clip(d.error_rate - d.error_rate_lo, 0, None), np.clip(d.error_rate_hi - d.error_rate, 0, None)],
                        color=RUNG_COLOR[rung], marker=RUNG_MARKER[rung], ls="none", capsize=2, label=RUNG_TEXT[rung])
        logit_axis(ax)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.5)
        ax.set_xlabel("posterior probability that the letter is inserted (logit scale)")
        ax.set_ylabel("share of letters labelled wrongly")
        ax.set_title(title)
        legend_below(ax, fontsize=8)
    ax = axes[2]
    rungs = [r for r in LADDER if (reference, r) in totals]
    width = 0.8 / max(len(rungs), 1)
    for k, rung in enumerate(rungs):
        e = totals[(reference, rung)]["excess_per_1000"]
        values = np.array([e[region]["errors"] for region in REGIONS])
        x = np.arange(len(REGIONS)) + (k - (len(rungs) - 1) / 2) * width
        ax.bar(x, values[:, 0], width * 0.92, color=RUNG_COLOR[rung], label=f"{RUNG_TEXT[rung]}, n = {totals[(reference, rung)]['n_sources']:,} sources",
               yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]], error_kw={"lw": 0.9, "capsize": 2}, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color=COLORS[6], lw=1)
    ax.set_xticks(np.arange(len(REGIONS)), [r.replace(" (", "\n(") for r in REGIONS])
    ax.set_xlabel(f"what the {ref_text} says about the letter")
    ax.set_ylabel("error letters per 1 000 letters above the law")
    ax.set_title("Where the errors beyond the law sit")
    ax.grid(axis="x", visible=False)
    legend_below(ax, 0.34, fontsize=8)
    parts = []
    if ("p_gap", "G") in totals:
        t = totals[("p_gap", "G")]
        parts.append(f"the exact sample under the coupling prior makes {t['observed_over_law']:.3f} × the errors the law predicts "
                     f"[{t['observed_over_law_ci'][0]:.3f}, {t['observed_over_law_ci'][1]:.3f}]")
    if ("q_true", "FFBS") in totals:
        t = totals[("q_true", "FFBS")]
        parts.append(f"FFBS {t['observed_over_law']:.3f} × [{t['observed_over_law_ci'][0]:.3f}, {t['observed_over_law_ci'][1]:.3f}] of its own law")
    if (reference, "M") in totals:
        e = totals[(reference, "M")]["excess_per_1000"]
        parts.append("the trained model exceeds the law by " + ", ".join(f"{e[r]['errors'][0]:+.1f} on letters that are {r.split(' (')[0]}" for r in REGIONS)
                     + " (error letters per 1 000 letters)")
    titled(fig, "The 2q(1 − q) law: " + "; ".join(parts), 175)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / "fig2_law.png")


def figure_budget(budget: Budget, subset: str, out: Path):
    order = [r for r in ("MAP-G", "G", "MAP-exact", "FFBS", "I2", "I3", "M", "MAP-model") if budget.has(subset, r)]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, metric, label in ((axes[0], "errors_per_1000", "wrongly labelled letters per 1 000 source letters"),
                              (axes[1], "clusters_per_1000", "error clusters per 1 000 source letters")):
        bottom = np.zeros(len(order))
        top = max(budget.get(subset, r, metric)[2] for r in order)
        for kind in KINDS:
            values = np.array([budget.get(subset, r, metric, kind)[0] for r in order])
            ax.bar(np.arange(len(order)), values, 0.62, bottom=bottom, color=KIND_COLOR[kind], edgecolor="white", linewidth=1.2, label=KIND_TEXT[kind])
            for x, (v, b) in enumerate(zip(values, bottom)):
                if v > 0.035 * top:
                    ax.text(x, b + v / 2, f"{v:.1f}", ha="center", va="center", fontsize=8, color="white")
            bottom += values
        for x, r in enumerate(order):
            v, lo, hi = budget.get(subset, r, metric)
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=COLORS[6], capsize=3, lw=1)
            ax.text(x, hi + 0.012 * top, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.set_xticks(np.arange(len(order)), [RUNG_TICK[r] for r in order], fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(0, top * 1.1)
        ax.grid(axis="x", visible=False)
    axes[0].set_title("Error letters (wrong deletions + missed insertions)")
    axes[1].set_title("Error clusters (errors at most two kept originals apart)")
    axes[0].legend(title="kind of cluster", loc="upper left")
    n = int(budget.frame[(budget.frame.subset == subset)].n_sources.iloc[0])
    floor = "G" if budget.has(subset, "G") else "FFBS"
    text = (f"Error budget on the {n:,} sources that every sampler was run on: the exact sample already makes "
            f"{budget.get(subset, floor, 'errors_per_1000')[0]:.1f} error letters per 1 000")
    if budget.has(subset, "M"):
        text += f", the trained model {budget.get(subset, 'M', 'errors_per_1000')[0]:.1f}"
        gain = {k: budget.get(subset, "M", "errors_per_1000", k)[0] - budget.get(subset, floor, "errors_per_1000", k)[0] for k in KINDS}
        text += "; its excess by kind: " + ", ".join(f"{k} {v:+.1f}" for k, v in gain.items())
    titled(fig, text + ". Black whiskers: 95 % bootstrap intervals of the totals.", 160)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out / "fig3_budget.png")


def figure_steps(budget: Budget, step_subset: dict, out: Path):
    steps = [s for s in STEPS if s in step_subset and budget.has(step_subset[s], f"{s[0]} - {s[1]}")]
    if not steps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    n_of = {s: int(budget.frame[budget.frame.subset == step_subset[s]].n_sources.iloc[0]) for s in steps}
    for ax, metric, label in ((axes[0], "errors_per_1000", "change in error letters per 1 000 letters"),
                              (axes[1], "clusters_per_1000", "change in error clusters per 1 000 letters")):
        width = 0.8 / len(KINDS)
        for k, kind in enumerate(KINDS):
            values = np.array([budget.get(step_subset[s], f"{s[0]} - {s[1]}", metric, kind) for s in steps])
            x = np.arange(len(steps)) + (k - 1.5) * width
            ax.bar(x, values[:, 0], width * 0.92, color=KIND_COLOR[kind], label=KIND_TEXT[kind], edgecolor="white", linewidth=0.8,
                   yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]], error_kw={"lw": 0.9, "capsize": 2})
        ax.axhline(0, color=COLORS[6], lw=1)
        ax.set_xticks(np.arange(len(steps)), [f"{STEP_TEXT[s]}\nn = {n_of[s]:,} sources" for s in steps], fontsize=8.5)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].legend(title="kind of cluster")
    axes[0].set_title("Error letters added by each step of the ladder")
    axes[1].set_title("Error clusters added by each step")
    text = "What each step of the ladder adds (error letters per 1 000, all kinds): " + "; ".join(
        f"{STEP_TEXT[s].splitlines()[-1].strip('()')} {budget.get(step_subset[s], f'{s[0]} - {s[1]}', 'errors_per_1000')[0]:+.1f} "
        f"[{budget.get(step_subset[s], f'{s[0]} - {s[1]}', 'errors_per_1000')[1]:+.1f}, {budget.get(step_subset[s], f'{s[0]} - {s[1]}', 'errors_per_1000')[2]:+.1f}]"
        for s in steps)
    titled(fig, text + ". Every step is paired on the largest set of sources both samplers were run on; whiskers: 95 % paired bootstrap intervals.", 160)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, out / "fig4_steps.png")


def figure_noise(budget: Budget, prefix: str, out: Path):
    rungs = [r for r in LADDER if budget.has(f"{prefix} noise {NOISE_TEXT[-1]}", r)]
    if not rungs:
        return
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.9))
    x = np.arange(len(NOISE_TEXT))
    short = [t.replace(" %", "").replace("under ", "< ") for t in NOISE_TEXT]
    for ax, kind in zip(axes, ["all"] + KINDS):
        for rung in rungs:
            values = np.array([budget.get(f"{prefix} noise {t}", rung, "errors_per_1000", kind) for t in NOISE_TEXT])
            ax.errorbar(x, values[:, 0], yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]], color=RUNG_COLOR[rung],
                        marker=RUNG_MARKER[rung], capsize=2, label=RUNG_TEXT[rung])
        ax.set_xticks(x, short)
        ax.set_title("all kinds" if kind == "all" else KIND_TEXT[kind])
        ax.set_xlabel("inserted share of the source (%)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("error letters per 1 000 source letters")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(rungs), fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    top, low = NOISE_TEXT[-1], NOISE_TEXT[0]
    floor = "G" if "G" in rungs else rungs[0]
    n = int(budget.frame[budget.frame.subset == prefix].n_sources.iloc[0])
    text = (f"Errors by noise level ({n:,} sources): the exact sample makes {budget.get(f'{prefix} noise {low}', floor, 'errors_per_1000')[0]:.1f} "
            f"error letters per 1 000 when {low} of the source is inserted and {budget.get(f'{prefix} noise {top}', floor, 'errors_per_1000')[0]:.1f} at {top}")
    if "M" in rungs:
        text += (f"; the trained model {budget.get(f'{prefix} noise {low}', 'M', 'errors_per_1000')[0]:.1f} and "
                 f"{budget.get(f'{prefix} noise {top}', 'M', 'errors_per_1000')[0]:.1f}")
    titled(fig, text + ". Whiskers: 95 % bootstrap intervals.", 170)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save(fig, out / "fig5_noise.png")


def figure_holes(table: pd.DataFrame, by_noise: pd.DataFrame, summary: dict, noise_summary: dict, out: Path):
    if table.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.9))
    L = np.arange(1, 40)
    for ax, measure, xlabel in ((axes[0], "wrongly deleted originals", "L = wrongly deleted originals in the cluster"),
                                (axes[1], "longest contiguous run", "L = longest run of deleted originals that are neighbours in the target")):
        ax.plot(L, law_survival("junction", L), color=COLORS[6], ls="--", lw=1.3, label="law: junction repair, (1 − v²)^(L(L+1)/2), v = 1/4")
        ax.plot(L, law_survival("greedy", L), color=COLORS[7], ls=":", lw=1.6, label="law: one-sided greedy deletion, (1 − v²)^L")
        for rung in LADDER:
            d = table[(table.rung == rung) & (table.measure == measure) & (table.survival > 0)]
            if d.empty:
                continue
            ax.plot(d["size"], d.survival, color=RUNG_COLOR[rung], marker=RUNG_MARKER[rung], ms=4,
                    label=f"{RUNG_TEXT[rung]}: {int(d.n_holes.iloc[0]):,} holes in {int(d.n_sources.iloc[0]):,} sources")
            ax.fill_between(d["size"], np.maximum(d.lo, 1e-5), d.hi, color=RUNG_COLOR[rung], alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 1.3)
        ax.set_xlim(1, 30)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("share of holes larger than L (holes of at least 2)")
        legend_below(ax, 0.15, fontsize=8)
    axes[0].set_title("Repaired holes (kind R): wrongly deleted originals")
    axes[1].set_title("Repaired holes: longest contiguous run")
    ax = axes[2]
    for rung in LADDER:
        d = by_noise[(by_noise.rung == rung) & (by_noise.n_holes >= 30)] if len(by_noise) else by_noise
        if len(d) == 0:
            continue
        r = d.mean_rho / (1 - d.mean_rho)
        fit = noise_summary.get(rung, {}).get("slope of log(ratio) in log r")
        label = RUNG_TEXT[rung] + (f": exponent {fit[0]:.2f} [{fit[1]:.2f}, {fit[2]:.2f}]" if fit else "")
        ax.errorbar(r, d.ratio, yerr=[d.ratio - d["ratio lo"], d["ratio hi"] - d.ratio], color=RUNG_COLOR[rung], marker=RUNG_MARKER[rung], capsize=2, label=label)
    guide = np.array([0.05, 1.0])
    ax.plot(guide, 0.5 * guide, color=COLORS[6], ls="--", lw=1.2, label="static alternative reading: proportional to r (exponent 1)")
    ax.axhline(1 - V_WINDOW**2, color=COLORS[7], ls=":", lw=1.6, label="greedy deletion: 1 − v² at every noise level (exponent 0)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise level r = inserted / original letters of the source")
    ax.set_ylabel("factor per further deleted original")
    ax.set_title("Dependence of the hole size on the noise level")
    legend_below(ax, 0.15, fontsize=8)
    bits = []
    for rung in LADDER:
        e = summary.get(rung, {}).get("wrongly deleted originals")
        if e:
            bits.append(f"{rung}: mean {e['mean']:.2f} wrongly deleted originals, factor per further letter {e['geometric_ratio']:.2f}, "
                        f"{100 * e['share_at_least_10']:.1f} % with 10 or more")
    best = []
    for rung in LADDER:
        ll = summary.get(rung, {}).get("wrongly deleted originals", {}).get("loglik_per_hole_second_half", {})
        scores = {k: v for k, v in ll.items() if k not in ("n", "ratio_first_half") and v is not None}
        if scores:
            best.append(f"{rung} {max(scores, key=scores.get).split(' (')[0]}")
    titled(fig, "Sizes of repaired holes. " + "; ".join(bits) + ". Law with the highest likelihood on held-out sources: " + ", ".join(best)
           + ". Bands: 95 % bootstrap intervals.", 175)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out / "fig6_holes.png")


def figure_sweep(table: pd.DataFrame, summary: dict, out: Path):
    if table.empty or not (table.what == "measured").any():
        return
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.9))
    fam_style = {"I2": (RUNG_COLOR["I2"], RUNG_MARKER["I2"], "exact rates (I2)"), "M": (RUNG_COLOR["M"], RUNG_MARKER["M"], "trained model (M)")}
    for ax, kind in zip(axes, ["all"] + KINDS):
        d = table[(table.metric == "errors_per_1000") & (table.kind == kind)]
        for family, (color, marker, text) in fam_style.items():
            e = d[(d.family == family) & (d.what == "measured")].sort_values("h")
            if e.empty:
                continue
            ax.errorbar(e.h, e.value, yerr=[e.value - e.lo, e.hi - e.value], color=color, marker=marker, ls="none", capsize=2, label=text)
            fit = summary.get(family, {}).get(f"errors_per_1000/{kind}")
            if fit:
                hs = np.array([0, e.h.max()])
                ax.plot(hs, fit["limit"] + fit["slope_per_unit_h"] * hs, color=color, lw=1.2)
                ax.errorbar([0], [fit["limit"]], yerr=[[fit["limit"] - fit["limit_ci"][0]], [fit["limit_ci"][1] - fit["limit"]]], color=color,
                            marker=marker, mfc="white", capsize=3, label=f"{text}: straight line, value at h = 0")
        for ref, text in (("FFBS", "exact sample, independent marks (FFBS)"), ("G", "exact sample, coupling prior (G)")):
            e = d[(d.family == ref) & (d.what == "reference")]
            if len(e):
                ax.axhline(e.value.iloc[0], color=RUNG_COLOR[ref], ls="--", lw=1.1, label=text)
                ax.axhspan(e.lo.iloc[0], e.hi.iloc[0], color=RUNG_COLOR[ref], alpha=0.12, lw=0)
        steps = sorted(set(d[d.what == "measured"].n_steps))
        ax.set_xticks([0.0] + [1 / n for n in steps], ["∞"] + [str(n) for n in steps], fontsize=7.5)
        ax.set_title("all kinds" if kind == "all" else KIND_TEXT[kind])
        ax.set_xlabel("Euler steps n (axis: step size 1/n)")
        ax.set_xlim(left=-0.0012)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("error letters per 1 000 source letters")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    bits = []
    for family in ("I2", "M"):
        fit = summary.get(family, {}).get("errors_per_1000/all")
        if fit:
            text = f"{family}: value at h = 0 is {fit['limit']:.1f} [{fit['limit_ci'][0]:.1f}, {fit['limit_ci'][1]:.1f}]"
            if "limit_minus_FFBS" in fit:
                v = fit["limit_minus_FFBS"]
                text += f", that is {v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}] against FFBS"
            bits.append(text)
    n = max((v.get("n_sources", 0) for v in summary.values()), default=0)
    titled(fig, f"Errors against the step size on {n:,} sources (error letters per 1 000). " + ("; ".join(bits) + ". " if bits else
           "Too few step counts for a line yet. ") + "Whiskers and bands: 95 % bootstrap intervals.", 170)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save(fig, out / "fig7_stepsize.png")


def figure_label_only(table: pd.DataFrame, summary: dict, out: Path):
    order = [r for r in ("MAP-G", "G", "MAP-exact", "FFBS", "I2", "I3", "M", "MAP-model") if r in set(table.rung)]
    if not order:
        return
    pick = lambda q: table[table.quantity == q].set_index("rung").reindex(order)  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.6))
    x = np.arange(len(order))
    ticks = [RUNG_SHORT[r] for r in order]
    ax = axes[0]
    d = pick("predicted label-only letters per 1000")
    ax.axhline(d.value.iloc[0], color=COLORS[6], ls="--", lw=1.2, label="predicted for a coin flip: one error letter per twin insertion")
    for k, (quantity, text) in enumerate((("class Z letters per 1000", "measured: letters in clusters of kind Z"),
                                          ("label errors that are not string errors per 1000", "measured: label errors that are not errors of the string"))):
        d = pick(quantity)
        ax.bar(x + (k - 0.5) * 0.36, d.value, 0.34, color=COLORS[k], label=text, yerr=[d.value - d.lo, d.hi - d.value],
               error_kw={"lw": 0.9, "capsize": 2}, edgecolor="white", linewidth=0.8)
    ax.set_ylabel("error letters per 1 000 source letters")
    ax.set_title("Label-only errors: prediction and measurement")
    ax = axes[1]
    ax.axhline(50, color=COLORS[6], ls="--", lw=1.2, label="coin flip between two identical letters")
    series = (("share missed: twin of an adjacent original", "the original next to it is the same letter"),
              ("share missed: twin of an original behind other insertions", "the same letter as the nearest original, other insertions between"),
              ("share missed: identical only to an adjacent insertion", "the same letter only as a neighbouring insertion"),
              ("share missed: no identical neighbour", "no identical neighbour"))
    for k, (quantity, text) in enumerate(series):
        d = pick(quantity)
        share = summary["groups_share_of_insertions"][quantity.split(": ")[1]]
        ax.bar(x + (k - 1.5) * 0.2, 100 * d.value, 0.19, color=COLORS[k], label=f"{text} ({100 * share:.1f} % of insertions)",
               yerr=[100 * (d.value - d.lo), 100 * (d.hi - d.value)], error_kw={"lw": 0.9, "capsize": 1.5}, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("share of these insertions that is kept (%)")
    ax.set_title("How often an insertion is missed, by its neighbours")
    ax = axes[2]
    for k, (quantity, text) in enumerate((("label f1", "F1 on the labels"), ("string f1 predicted (label f1 + twins)", "F1 on the string, predicted: labels + twins"),
                                          ("string f1 measured", "F1 on the string, measured"))):
        d = pick(quantity)
        ax.bar(x + (k - 1) * 0.27, d.value, 0.25, color=COLORS[k], label=text, yerr=[d.value - d.lo, d.hi - d.value],
               error_kw={"lw": 0.9, "capsize": 2}, edgecolor="white", linewidth=0.8)
    low = float(pick("label f1").value.min())
    ax.set_ylim(max(0.0, low - 0.03), 1.0)
    ax.set_ylabel("F1 of the deletions")
    ax.set_title("F1 on the labels and on the string")
    for ax in axes:
        ax.set_xticks(x, ticks, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        legend_below(ax, 0.3, fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.82, bottom=0.36, wspace=0.2)
    ref = "FFBS" if "FFBS" in order else order[0]
    get = lambda q: table[(table.quantity == q) & (table.rung == ref)].value.iloc[0]  # noqa: E731
    titled(fig, f"{100 * summary['share_equal_to_the_nearest_original_on_a_side']:.2f} % of the inserted letters equal the nearest original beside them "
                f"(theory 1 − (19/20)² = 9.75 %; for {100 * summary['share_equal_to_an_adjacent_original']:.1f} % it is the direct neighbour). Swapping the two "
                f"costs two label errors and no string error. Predicted for a coin flip: {get('predicted label-only letters per 1000'):.1f} letters per 1 000; "
                f"measured for {ref}: {get('label errors that are not string errors per 1000'):.1f}, of which {get('class Z letters per 1000'):.1f} in pure "
                f"label-only clusters. String-level F1 of {ref}: predicted {get('string f1 predicted (label f1 + twins)'):.3f}, "
                f"measured {get('string f1 measured'):.3f}.", 180)
    save(fig, out / "fig8_label_only.png")


# ---- main ---------------------------------------------------------------------------------------------------------


def clean(obj):
    """JSON-ready copy: numpy scalars to Python, NaN to None, tuple keys to strings."""
    if isinstance(obj, dict):
        return {(" / ".join(map(str, k)) if isinstance(k, tuple) else str(k)): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("runs/lap-01"))
    parser.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (over sequences)")
    parser.add_argument("--refresh", action="store_true", help="recompute the cached tables in scratch/")
    parser.add_argument("--no-gap", action="store_true", help="skip the reference with the prior of the coupling")
    args = parser.parse_args()
    started = time.time()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    study = run.parent / "error-study"
    out = run / "errors" / "results" / "references"
    scratch = out / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    sources = Sources.load(study / "sources.npz")
    P = transition_table(np.random.default_rng(0), 20, 5, 1.0)
    eligible = P > 0
    posterior = load_npz(study / "posterior.npz")
    n_seq = len(sources)
    rho = sources.rho
    noise_bin = np.clip(np.digitize(rho, NOISE_EDGES) - 1, 0, len(NOISE_TEXT) - 1)
    missing: list[str] = []
    summary: dict = {"run": run.name, "n_sources": n_seq, "n_letters": int(len(sources.letters)), "n_inserted": int(sources.inserted.sum()),
                     "bootstrap_resamples": args.boot,
                     "grammar": {"allowed_per_context": int(eligible.sum(2).min()), "smallest_allowed_probability": float(P[eligible].min()),
                                 "largest_allowed_probability": float(P[eligible].max()),
                                 "note": "the allowed letters of a context are not equally likely (Dirichlet weights)"}}
    log(f"{n_seq} sources, {len(sources.letters):,} letters; results -> {out}")

    # ---- the rungs
    deleted_of: dict[str, np.ndarray] = {}
    first_of: dict[str, int] = {}
    files = {"FFBS": study / "ffbs.npz", "I2": study / "i2-100.npz", "I3": study / "i3-100.npz", "M": run / "errors" / "m100.npz"}
    q0 = None
    for name, path in files.items():
        z = load_npz(path)
        if z is None:
            missing.append(str(path.relative_to(run.parent)))
            continue
        first_of[name] = int(z["first"])
        deleted_of[name] = z["del_step"] >= 0
        if name == "M":
            q0 = z["q0"].astype(np.float64)
            summary["model"] = {"epoch": int(z["epoch"]), "steps": int(z["steps"]), "seed": int(z["seed"])}
    posteriors = {}
    if posterior is not None:
        posteriors["q_true"], posteriors["q_est"] = posterior["q_true"].astype(np.float64), posterior["q_est"].astype(np.float64)
        deleted_of["MAP-exact"], first_of["MAP-exact"] = posteriors["q_true"] > 0.5, n_seq
    else:
        missing.append("error-study/posterior.npz")
    if q0 is not None:
        deleted_of["MAP-model"], first_of["MAP-model"] = q0 > 0.5, first_of["M"]
    if not args.no_gap and posterior is not None:
        log("reference with the prior of the coupling (forward-backward with gap counts)")
        gap = gap_reference(sources, P, posteriors["q_true"], scratch, args.refresh)
        posteriors["p_gap"] = gap["p_gap"]
        deleted_of["G"], first_of["G"] = gap["sample"], n_seq
        deleted_of["MAP-G"], first_of["MAP-G"] = gap["p_gap"] > 0.5, n_seq
        summary["coupling_prior_reference"] = {
            "what": "hidden-Markov posterior whose prior counts the insertions of the current gap: Binomial(K, 1/(n1+1)) per gap, "
                    "as the coupling draws them, instead of independent marks (geometric)",
            "check_constant_hazard_reproduces_q_true_max_abs_difference": gap["check"], "cap_on_tracked_insertions_per_gap": GAP_CAP}
    if missing:
        log("missing: " + ", ".join(missing))

    log("cluster tables of the rungs")
    X_of, table_of = {}, {}
    for name in deleted_of:
        X_of[name], table_of[name] = analyse(name, sources, deleted_of[name], first_of[name], eligible, scratch, args.refresh)
        S = X_of[name].sum(0)
        log(f"  {name:10s} {first_of[name]:5d} sources  precision {S[IDX['tp']] / S[IDX['deleted']]:.4f}  recall {S[IDX['tp']] / S[IDX['inserted']]:.4f}  "
            f"clusters {len(table_of[name]):,}")

    # ---- 1. reliability
    if posteriors:
        log("1. reliability of the reference posteriors")
        named = dict(posteriors)
        if q0 is not None and first_of["M"] == n_seq:
            named["model_q0"] = q0
        rel, ctx, rel_summary = reliability(sources, named, noise_bin, args.boot)
        rel.to_csv(out / "reliability.csv", index=False)
        ctx.to_csv(out / "reliability_by_left_neighbour.csv", index=False)
        summary["1_reliability"] = rel_summary
        summary["1_reliability_by_left_neighbour"] = {
            f"{r.posterior} / {r.subset} / {r.context}": {"n_letters": int(r.n_letters), "stated": r.stated, "share_inserted": r.share_inserted,
                                                         "deviation": r.deviation, "ci": [r.lo, r.hi]} for r in ctx.itertuples()}
        figure_reliability(rel, ctx, rel_summary, out)

    # ---- 2. the law
    log("2. the 2q(1-q) law")
    law_rows, law_totals = [], {}
    for reference, q in posteriors.items():
        for name in LADDER:
            if name in deleted_of:
                truth = posteriors.get("p_gap") if (reference == "q_true" and name == "FFBS") else None
                rows, total = law_table(reference, q, name, deleted_of[name], first_of[name], sources, args.boot, truth)
                law_rows += rows
                law_totals[(reference, name)] = total
    if q0 is not None:
        rows, total = law_table("model_q0", q0, "M", deleted_of["M"], first_of["M"], sources, args.boot)
        law_rows += rows
        law_totals[("model_q0", "M")] = total
    law = pd.DataFrame(law_rows)
    law.to_csv(out / "law.csv", index=False)
    map_totals = {}
    for decision, q_name in (("MAP-exact", "q_true"), ("MAP-G", "p_gap"), ("MAP-model", "q_true"), ("MAP-model", "p_gap")):
        if decision in deleted_of and q_name in posteriors:
            map_totals[f"{decision} against sum of min under {q_name}"] = map_check(posteriors[q_name], deleted_of[decision], first_of[decision], sources, args.boot)
    if q0 is not None:
        map_totals["MAP-model against sum of min under model_q0"] = map_check(q0, deleted_of["MAP-model"], first_of["M"], sources, args.boot)
    summary["2_law"] = {"samplers": law_totals, "decisions": map_totals}
    if len(law):
        figure_law(law, law_totals, out)

    # ---- 3. budget: every comparison on the largest set of sources that all its rungs cover
    log("3. error budget")
    ladder_present = [r for r in LADDER if r in X_of]
    tiers = sorted({first_of[r] for r in X_of})
    n_common = min(first_of[r] for r in ladder_present)
    pairs = STEPS + EXTRA_STEPS
    rows = []
    for c in tiers:
        covered = {r: X for r, X in X_of.items() if first_of[r] >= c}
        rows += budget_rows(f"first {c}", np.arange(c), covered, pairs, args.boot)
        for b, text in enumerate(NOISE_TEXT):
            rows += budget_rows(f"first {c} noise {text}", np.flatnonzero(noise_bin[:c] == b), covered, pairs, args.boot)
    top = tiers[-1]
    full = {r: X for r, X in X_of.items() if first_of[r] == top}
    rows += budget_rows("first half", np.arange(0, top // 2), full, pairs, args.boot)
    rows += budget_rows("second half", np.arange(top // 2, top), full, pairs, args.boot)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "budget.csv", index=False)
    budget = Budget(frame)
    main_subset = f"first {n_common}"
    step_subset = {(a, b): f"first {min(first_of[a], first_of[b])}" for a, b in pairs if a in first_of and b in first_of}
    summary["3_budget"] = {"sets": {f"first {c}": sorted(r for r in X_of if first_of[r] >= c) for c in tiers},
                           "set_with_every_rung": main_subset,
                           "largest_common_set_of_each_step": {f"{a} - {b}": v for (a, b), v in step_subset.items()}}
    wanted = ("precision", "recall", "f1", "string_precision", "string_recall", "string_f1", "exact_match", "errors_per_1000", "x_per_1000",
              "miss_per_1000", "string_errors_per_1000", "label_only_per_1000", "clusters_per_1000")

    def block_of(subset, rung):
        entry = {m: list(budget.get(subset, rung, m)) for m in wanted}
        for kind in KINDS:
            for m in ("errors_per_1000", "clusters_per_1000", "x_per_1000", "miss_per_1000"):
                entry[f"{m} {kind}"] = list(budget.get(subset, rung, m, kind))
        entry["n_sources"] = int(frame[(frame.subset == subset) & (frame.rung == rung)].n_sources.iloc[0])
        return entry

    for subset in [f"first {c}" for c in tiers] + ["first half", "second half"]:
        summary["3_budget"][subset] = {rung: block_of(subset, rung) for rung in X_of if budget.has(subset, rung)}
    summary["3_budget"]["steps_on_their_largest_common_set"] = {
        f"{a} - {b}": dict(block_of(subset, f"{a} - {b}"), set=subset) for (a, b), subset in step_subset.items() if budget.has(subset, f"{a} - {b}")}
    summary["3_budget"]["steps_by_noise_on_their_largest_common_set"] = {
        f"{a} - {b}": {text: {k: list(budget.get(f"{subset} noise {text}", f"{a} - {b}", "errors_per_1000", k)) for k in ["all"] + KINDS}
                       for text in NOISE_TEXT} for (a, b), subset in step_subset.items() if budget.has(subset, f"{a} - {b}")}
    floor = "G" if "G" in X_of else "FFBS"
    floor_set = f"first {first_of[floor]}"
    verdict = {"unit": "error letters per 1000 source letters [value, lo, hi]",
               "irreducible": {"rung": floor, "set": floor_set, **{k: list(budget.get(floor_set, floor, "errors_per_1000", k)) for k in ["all"] + KINDS},
                               "string_errors": list(budget.get(floor_set, floor, "string_errors_per_1000"))}}
    for (a, b), cause in ((("FFBS", "G"), "reference prior ignores the gap counts (not a property of the model)"), (("I2", "FFBS"), "Euler dynamics"),
                          (("I3", "I2"), "noise level not known"), (("M", "I3"), "learning"), (("M", "G"), "trained model against the irreducible floor")):
        if (a, b) in step_subset and budget.has(step_subset[(a, b)], f"{a} - {b}"):
            verdict[f"{a} - {b}"] = {"cause": cause, "set": step_subset[(a, b)],
                                     **{k: list(budget.get(step_subset[(a, b)], f"{a} - {b}", "errors_per_1000", k)) for k in ["all"] + KINDS},
                                     "string_errors": list(budget.get(step_subset[(a, b)], f"{a} - {b}", "string_errors_per_1000"))}
    if "M" in X_of:
        m_set = f"first {min(first_of['M'], first_of[floor])}"
        verdict["trained model"] = {"set": m_set, **{k: list(budget.get(m_set, "M", "errors_per_1000", k)) for k in ["all"] + KINDS},
                                    "string_errors": list(budget.get(m_set, "M", "string_errors_per_1000"))}
        verdict["share_of_the_model_errors_that_the_exact_sample_makes_too"] = {
            "labels": budget.get(m_set, floor, "errors_per_1000")[0] / budget.get(m_set, "M", "errors_per_1000")[0],
            "string": budget.get(m_set, floor, "string_errors_per_1000")[0] / budget.get(m_set, "M", "string_errors_per_1000")[0]}
    summary["verdict_C"] = verdict
    figure_budget(budget, main_subset, out)
    figure_steps(budget, step_subset, out)
    figure_noise(budget, main_subset, out)

    log("3b. confounders: position in the sequence, length of the target")
    conf = confounders(sources, {r: deleted_of[r] for r in ladder_present}, first_of, noise_bin, args.boot)
    conf.to_csv(out / "confounders.csv", index=False)
    summary["3b_confounders"] = {f"{r.rung} / {r.grouping} / {r.group}": [r.errors_per_1000, r.lo, r.hi, int(r.n_sources)] for r in conf.itertuples()}

    # ---- 4. label-only class
    log("4. the label-only class")
    lo_table, lo_summary = label_only(sources, X_of, deleted_of, args.boot)
    lo_table.to_csv(out / "label_only.csv", index=False)
    lo_summary["by_rung"] = {name: {r.quantity: [r.value, r.lo, r.hi] for r in g.itertuples()} for name, g in lo_table.groupby("rung")}
    summary["4_label_only"] = lo_summary
    figure_label_only(lo_table, lo_summary, out)

    # ---- 5. holes
    log("5. holes")
    hole_table, hole_summary = holes({r: table_of[r] for r in ladder_present}, first_of, rho, noise_bin, args.boot)
    hole_table.to_csv(out / "holes.csv", index=False)
    noise_table, noise_summary = holes_by_noise({r: table_of[r] for r in ladder_present}, first_of, sources, args.boot)
    noise_table.to_csv(out / "holes_by_noise.csv", index=False)
    hole_summary["dependence_on_noise_level"] = noise_summary
    patterns = {}
    for name in ladder_present:
        R = table_of[name][table_of[name].kind == "R"]
        patterns[name] = {p: int(c) for p, c in R.pattern.value_counts().head(12).items()}
        contiguous = float((R.longest == R.n_x).mean()) if len(R) else float("nan")
        hole_summary[name]["share_one_contiguous_run"] = contiguous
        hole_summary[name]["share_with_a_missed_insertion"] = float((R.n_miss > 0).mean()) if len(R) else float("nan")
    hole_summary["most_frequent_patterns"] = patterns
    summary["5_holes"] = hole_summary
    figure_holes(hole_table, noise_table, hole_summary, noise_summary, out)

    # ---- 6. step-size sweep
    log("6. step-size sweep")
    families: dict[str, dict] = {"I2": {}, "M": {}}
    for family, pattern in (("I2", str(study / "i2-{n}.npz")), ("M", str(run / "errors" / "m{n}.npz"))):
        for n in (50, 100, 200, 400):
            path = Path(pattern.format(n=n))
            z = load_npz(path)
            if z is None:
                missing.append(str(path.relative_to(run.parent)))
                continue
            first = int(z["first"])
            families[family][n], _ = analyse(f"{family.lower()}-{n}", sources, z["del_step"] >= 0, first, eligible, scratch, args.refresh)
    sweep_table, sweep_summary = sweep(families, {r: X_of[r] for r in ("FFBS", "G") if r in X_of}, args.boot)
    sweep_table.to_csv(out / "steps.csv", index=False)
    summary["6_step_size"] = sweep_summary
    figure_sweep(sweep_table, sweep_summary, out)

    summary["missing_files"] = sorted(set(missing))
    summary["seconds"] = round(time.time() - started)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(clean(summary), f, indent=1, ensure_ascii=False)
    log(f"done in {time.time() - started:.0f} s; missing: {sorted(set(missing)) or 'nothing'}")

    # ---- headline on the console
    print()
    for c in tiers:
        subset = f"first {c}"
        print(f"--- first {c} sources: error letters per 1 000 letters [95 % interval]")
        for rung in ["MAP-G", "G", "MAP-exact", "FFBS", "I2", "I3", "M", "MAP-model"] + [f"{a} - {b}" for a, b in pairs]:
            if budget.has(subset, rung):
                print(f"{rung:22s} all {budget.text(subset, rung, 'errors_per_1000'):22s} " + "  ".join(
                    f"{k} {budget.text(subset, rung, 'errors_per_1000', k):20s}" for k in KINDS)
                    + f"  F1 {budget.get(subset, rung, 'f1')[0]:+.4f}  string F1 {budget.get(subset, rung, 'string_f1')[0]:+.4f}")


if __name__ == "__main__":
    main()
