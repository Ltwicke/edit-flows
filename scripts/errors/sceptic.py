#!/usr/bin/env python
r"""The sceptic of the error study: recompute the headline numbers from the raw corpora with independent code and
try to break the conclusions drawn from them.

    python scripts/errors/sceptic.py --run runs/lap-01

Nothing here calls the other scripts of this folder. Clusters, kinds, the string-level counts, the smallest repair,
the Mantel-Haenszel odds ratio and the event study are re-implemented below (numba) and compared, on a sample, with
the tested primitives of `editflows/errors.py`. The only things taken over are the corpora themselves, the exact
posterior of `editflows/exact.py` (needed to re-evaluate the rates of the leak sampler) and, for the intervention
experiment, the cached sampled runs (the GPU part cannot be repeated here).

Parts: C1 - C12 recompute the reported numbers (`claims.csv`). Four attacks on the conclusions drawn from them:

(a) the share of the perfect sampler in the model's errors and "the surplus is holes": other cluster gaps, noise
    level, source length, and definitions that need no clusters (string errors, runs of adjacent originals);
(b) the event study by break size: strata of time, noise and prior doubt, a contrast within cells, a placebo in
    time, and the randomised version (forced removals, one forward pass, cached by the intervention study);
(c) first reading against cascade: the account in label errors, one independent draw from the first reading, the
    neighbourhoods of wrong deletions, and where the model's belief changes along the logged run;
(d) the leak experiment: the letters the leak itself deleted, counted from the exact q of the sampler in the step
    of every wrong deletion (about two minutes on the CPU, cached in `scratch/`).

Outputs in `runs/<run>/errors/results/sceptic/`: `claims.csv` (claim, quantity, reported, recomputed, interval, n,
verdict, note), `summary.json` (every part, `conclusions` with the headline numbers and checks of the four
attacks), tables `attack_*.csv`, figures `fig*.png`; scratch files in `scratch/`. CPU only; about half a minute
when the exact q of the leak runs is cached, about three minutes otherwise.

The numbers in `REPORTED` are what the other analyses reported for one checkpoint; for any other run the column
`reported` stays empty and the verdict is `recomputed`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # a training run owns the GPU

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import Sources, canonical_counts, clusters, minimal_repair  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
KINDS = "ZVRU"
NOISE_EDGES = np.array([0.0, 0.1, 0.2, 0.35, 0.51])
NOISE_TEXT = ["noise under 10 %", "noise 10-20 %", "noise 20-35 %", "noise 35-50 %"]
Q_EDGES = np.array([0, 1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.999, 1.0001])
QBINS = np.array([0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  0.97, 1.01])  # bins of q for matching controls in the event study

# What the other analyses reported (checkpoint lap-01); see the brief of the sceptic.
REPORTED = {"lap-01": {
    "C1 errors": 42.3, "C1 Z": 21.3, "C1 V": 15.4, "C1 R": 5.6, "C1 U": 0.0, "C1 law": 42.3, "C1 logloss gap": 0.0664,
    "C1 logloss true": 0.0738,
    "C2 errors": 72.9, "C2 wrong": 45.8, "C2 missed": 27.2, "C2 Z": 18.9, "C2 V": 13.7, "C2 R": 22.5, "C2 U": 17.9,
    "C2 f1": 0.875, "C2 string f1": 0.919, "C2 f1 perfect": 0.925, "C2 string f1 perfect": 0.969,
    "C3 i2-ffbs": 0.8, "C3 i3-i2": 0.2, "C3 limit": 0.0,
    "C4 twin": 9.7, "C4 adjacent": 7.5, "C4 half": 0.5, "C4 identical": 89.5,
    "C5 model": 4.84, "C5 exact": 1.40, "C5 perfect": 0.50, "C5 surplus": 95.0,
    "C6 model": 0.76, "C6 perfect": 0.0, "C6 median wrong": 51.0, "C6 median correct": 20.0,
    "C7 first reading": 36.3, "C7 exact": 24.9, "C7 observed": 45.8,
    "C8 b0": 7.8, "C8 b1": 40.0, "C8 b2": 52.2, "C8 hole": 4.1,
    "C9 cost": 3.36,
    "C10 model": 9.4, "C10 exact": 4.8, "C10 perfect": 2.8,
    "C11 b2": 0.22, "C11 b1": 0.10, "C11 b0": 0.0,
    "C12 model b0": 0.03, "C12 model b1": 2.56, "C12 model b2": 3.02, "C12 exact b1": 3.45, "C12 exact b2": 3.55,
    "C12 gram exact": 99.5, "C12 gram model": 58.0,
}}


# ---- numba kernels ---------------------------------------------------------------------------------------------------


@numba.njit(cache=True)
def nb_clusters(letters, inserted, deleted, offsets, gap, eligible):
    """Error clusters of all sources (at most `gap` kept originals between two errors of a cluster) and their kind:
    0 Z output = target, 1 V other grammatical reading, 2 R grammatical hole, 3 U a forbidden window touches it.
    Returns per cluster: source, start, end, wrong deletions, missed insertions, correct deletions, kept originals,
    kind."""
    n = offsets[-1]
    n_err = 0
    longest = 0
    for s in range(len(offsets) - 1):
        longest = max(longest, offsets[s + 1] - offsets[s])
    for i in range(n):
        if inserted[i] != deleted[i]:
            n_err += 1
    out = np.zeros((n_err, 8), dtype=np.int64)
    kept = np.empty(longest, dtype=np.int64)
    f_first = np.empty(longest, dtype=np.int64)
    f_last = np.empty(longest, dtype=np.int64)
    c = 0
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        nk = 0
        for i in range(a, b):
            if not deleted[i]:
                kept[nk] = i
                nk += 1
        nf = 0
        for j in range(2, nk):
            if not eligible[letters[kept[j - 2]], letters[kept[j - 1]], letters[kept[j]]]:
                f_first[nf] = kept[j - 2]
                f_last[nf] = kept[j]
                nf += 1
        c0 = c
        start, last, dots = -1, -1, 0
        for i in range(a, b):
            if inserted[i] != deleted[i]:
                if start < 0 or dots > gap:
                    if start >= 0:
                        out[c, 0], out[c, 1], out[c, 2] = s, start, last
                        c += 1
                    start = i
                last = i
                dots = 0
            elif (not inserted[i]) and (not deleted[i]):
                dots += 1
        if start >= 0:
            out[c, 0], out[c, 1], out[c, 2] = s, start, last
            c += 1
        for k in range(c0, c):
            st, en = out[k, 1], out[k, 2]
            for i in range(st, en + 1):
                if inserted[i]:
                    if deleted[i]:
                        out[k, 5] += 1
                    else:
                        out[k, 4] += 1
                elif deleted[i]:
                    out[k, 3] += 1
                else:
                    out[k, 6] += 1
            io, it = st, st
            same = True
            while True:
                while io <= en and deleted[io]:
                    io += 1
                while it <= en and inserted[it]:
                    it += 1
                if io > en or it > en:
                    same = io > en and it > en
                    break
                if letters[io] != letters[it]:
                    same = False
                    break
                io += 1
                it += 1
            violated = False
            for f in range(nf):
                if f_first[f] <= en and f_last[f] >= st:
                    violated = True
                    break
            if same:
                out[k, 7] = 0
            elif violated:
                out[k, 7] = 3
            elif out[k, 3] >= 2 and out[k, 3] > out[k, 4]:
                out[k, 7] = 2
            else:
                out[k, 7] = 1
    return out[:c]


@numba.njit(cache=True)
def nb_lcs(letters, inserted, deleted, offsets):
    """Length of the longest common subsequence of output and target of every source (plain dynamic programme)."""
    S = len(offsets) - 1
    out = np.zeros(S, dtype=np.int64)
    longest = 0
    for s in range(S):
        longest = max(longest, offsets[s + 1] - offsets[s])
    o = np.empty(longest, dtype=np.int64)
    t = np.empty(longest, dtype=np.int64)
    prev = np.zeros(longest + 1, dtype=np.int64)
    cur = np.zeros(longest + 1, dtype=np.int64)
    for s in range(S):
        no = nt = 0
        for i in range(offsets[s], offsets[s + 1]):
            if not deleted[i]:
                o[no] = letters[i]
                no += 1
            if not inserted[i]:
                t[nt] = letters[i]
                nt += 1
        for j in range(nt + 1):
            prev[j] = 0
        for i in range(no):
            cur[0] = 0
            oi = o[i]
            for j in range(nt):
                if oi == t[j]:
                    cur[j + 1] = prev[j] + 1
                else:
                    cur[j + 1] = max(prev[j + 1], cur[j])
            for j in range(nt + 1):
                prev[j] = cur[j]
        out[s] = prev[nt]
    return out


@numba.njit(cache=True)
def nb_runs(inserted, deleted, offsets):
    """For every wrongly deleted original: the length of the run of consecutive wrongly deleted originals it lies
    in (target coordinates: inserted letters are skipped). 0 elsewhere."""
    out = np.zeros(len(inserted), dtype=np.int64)
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        i = a
        while i < b:
            if inserted[i] or not deleted[i]:
                i += 1
                continue
            j, length, last = i, 0, i
            while j < b and (inserted[j] or deleted[j]):
                if not inserted[j]:
                    length += 1
                    last = j
                j += 1
            for k in range(i, last + 1):
                if not inserted[k]:
                    out[k] = length
            i = j
    return out


# ---- context and small tools -------------------------------------------------------------------------------------------


def load_npz(path: Path):
    """The arrays of an npz file, None if it is missing; one retry in case it is still being written."""
    if not path.exists():
        return None
    for attempt in range(2):
        try:
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        except Exception:  # noqa: BLE001
            if attempt == 0:
                time.sleep(8)
    return None


class Context:
    def __init__(self, args):
        self.args = args
        self.run = Path(args.run)
        self.errors = self.run / "errors"
        self.study = self.run.parent / "error-study"
        self.out = self.errors / "results" / "sceptic"
        (self.out / "scratch").mkdir(parents=True, exist_ok=True)
        self.sources = Sources.load(self.study / "sources.npz")
        s = self.sources
        self.S = len(s)
        self.offsets = s.offsets.astype(np.int64)
        self.lengths = np.diff(self.offsets).astype(np.float64)
        self.letters = s.letters.astype(np.int64)
        self.ins = s.inserted.astype(np.bool_)
        self.seq_of = s.seq_of
        self.rho = s.rho
        self.P = transition_table(np.random.default_rng(0), 20, 5, 1.0)
        self.eligible = self.P > 0
        post = load_npz(self.study / "posterior.npz") or {}
        self.q_gap, self.q_true = post.get("q_gap"), post.get("q_true")
        self.reported = REPORTED.get(self.run.name, {})
        self.claims: list[dict] = []
        self.summary: dict = {"run": self.run.name, "n_sources": self.S, "n_letters": int(self.offsets[-1]),
                              "bootstrap_resamples": args.boot}
        self.missing: list[str] = []
        self.tables: dict = {}
        self._W: dict = {}
        self._corpus: dict = {}
        self._clusters: dict = {}
        self.hazard, self.kappa_t, self.kappa = None, None, None

    # -- corpora
    def corpus(self, path: Path):
        """(arrays, number of sources covered) of a corpus file, (None, 0) if it is missing."""
        if path not in self._corpus:
            z = load_npz(path)
            if z is None:
                self.missing.append(str(path))
                self._corpus[path] = (None, 0)
            else:
                first = int(z["first"])
                assert len(z["del_step"]) == int(self.offsets[first]), f"{path}: arrays do not end at a source boundary"
                self._corpus[path] = (z, first)
        return self._corpus[path]

    def deleted(self, path: Path):
        z, first = self.corpus(path)
        return (None, 0) if z is None else (z["del_step"] >= 0, first)

    # -- per-source sums and the bootstrap over sources
    def ps(self, values, first: int | None = None) -> np.ndarray:
        """Sum of a per-letter array over the letters of every source (the first `first` sources)."""
        first = self.S if first is None else first
        n = int(self.offsets[first])
        return np.bincount(self.seq_of[:n], weights=np.asarray(values[:n], dtype=np.float64), minlength=first)

    def W(self, m: int) -> np.ndarray:
        if m not in self._W:
            rng = np.random.default_rng([self.args.seed, m])
            self._W[m] = rng.multinomial(m, np.full(m, 1.0 / m), size=self.args.boot).astype(np.float64)
        return self._W[m]

    def boot(self, fn, *arrays, ids=None):
        """`fn(sum_1, sum_2, ...)` of per-source arrays: the value and its 95 % bootstrap interval over sources.
        `ids` restricts to a subset of sources (boolean mask or indices)."""
        arrays = [np.asarray(a, dtype=np.float64) if ids is None else np.asarray(a, dtype=np.float64)[ids] for a in arrays]
        W = self.W(len(arrays[0]))
        with np.errstate(all="ignore"):
            point = fn(*[a.sum(0) for a in arrays])
            rep = fn(*[W @ a for a in arrays])
            lo, hi = np.nanpercentile(rep, [2.5, 97.5], axis=0)
        return point, lo, hi

    def ratio(self, num, den, scale: float = 1.0, ids=None):
        return self.boot(lambda a, b: scale * a / b, num, den, ids=ids)

    # -- clusters of a corpus
    def cluster_table(self, key, deleted: np.ndarray, first: int, gap: int = 2) -> np.ndarray:
        if (key, gap) not in self._clusters:
            n = int(self.offsets[first])
            self._clusters[key, gap] = nb_clusters(self.letters[:n], self.ins[:n], deleted[:n], self.offsets[: first + 1], gap, self.eligible)
        return self._clusters[key, gap]

    def kind_sums(self, key, deleted: np.ndarray, first: int, gap: int = 2) -> dict:
        """Per source: wrong deletions and missed insertions by kind of cluster, hole runs (two or more wrong
        deletions more than missed insertions) and the wrong deletions inside them."""
        t = self.cluster_table(key, deleted, first, gap)
        src = t[:, 0]
        out = {"clusters": np.bincount(src, minlength=first).astype(np.float64)}
        for k, name in enumerate(KINDS):
            sel = t[:, 7] == k
            out[f"x_{name}"] = np.bincount(src[sel], weights=t[sel, 3], minlength=first)
            out[f"miss_{name}"] = np.bincount(src[sel], weights=t[sel, 4], minlength=first)
            out[f"n_{name}"] = np.bincount(src[sel], minlength=first).astype(np.float64)
        hole = t[:, 3] - t[:, 4] >= 2
        out["hole_runs"] = np.bincount(src[hole], minlength=first).astype(np.float64)
        out["x_in_hole_runs"] = np.bincount(src[hole], weights=t[hole, 3], minlength=first)
        out["err_in_hole_runs"] = np.bincount(src[hole], weights=t[hole, 3] + t[hole, 4], minlength=first)
        return out

    def target_break(self, gpos: np.ndarray) -> np.ndarray:
        """Break size of removing the original at the flat position `gpos`, among the originals alone (the inserted
        letters ignored): 0, 1 or 2 forbidden windows across the gap, -1 within two letters of an end of the target."""
        if not hasattr(self, "_target"):
            orig = ~self.ins
            t_index = np.cumsum(orig) - 1
            t_off = np.concatenate([[0], np.cumsum(self.ps(orig).astype(np.int64))])
            self._target = (self.letters[orig], t_index, t_off)
        T, t_index, t_off = self._target
        gpos = np.asarray(gpos)
        i = t_index[gpos]
        s = self.seq_of[gpos]
        ok = (i - 2 >= t_off[s]) & (i + 2 < t_off[s + 1])
        j = np.where(ok, i, 2)
        E = self.eligible
        b = (~E[T[j - 2], T[j - 1], T[j + 1]]).astype(np.int64) + (~E[T[j - 1], T[j + 1], T[j + 2]])
        return np.where(ok, b, -1)

    # -- claims
    def claim(self, cid: str, what: str, key: str | None, value, interval=None, tol: float | None = None, note: str = "",
              verdict: str | None = None, n: str = ""):
        """Register a recomputed number. `key` looks up the reported number; `tol` is the largest difference that
        still counts as reproduced (default: a little more than half a unit of the last reported digit)."""
        reported = self.reported.get(key) if key else None
        value = None if value is None or not np.isfinite(value) else float(value)
        if verdict is None:
            if value is None:
                verdict = "not computed"
            elif reported is None:
                verdict = "recomputed"
            else:
                if tol is None:
                    text = f"{reported:.10g}"
                    digits = len(text.split(".")[1]) if "." in text else 0
                    tol = 0.6 * 10.0 ** (-digits)
                verdict = "reproduced" if abs(value - reported) <= tol else "not reproduced"
        self.claims.append({"claim": cid, "quantity": what, "reported": reported, "recomputed": value,
                            "interval": "" if interval is None else f"[{interval[0]:.4g}, {interval[1]:.4g}]", "n": n,
                            "verdict": verdict, "note": note})
        shown = "-" if value is None else f"{value:.4g}"
        print(f"  {cid:4s} {what[:74]:74s} reported {'-' if reported is None else reported!s:>7}  recomputed {shown:>8} "
              f"{self.claims[-1]['interval']:>18}  {verdict}", flush=True)


def r4(x):
    """Rounded plain numbers for json."""
    if isinstance(x, dict):
        return {str(k): r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r4(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(f"{float(x):.4g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def triple(est) -> list:
    return [float(est[0]), float(est[1]), float(est[2])]


def schedule(C: Context, n_steps: int):
    """Hazard and kappa at the step boundaries t = k / n, from the config of the run."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    kappa = instantiate(OmegaConf.load(C.run / "config.yaml").task.path.kappa)
    t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    return kappa.loss_coefficient(t).numpy(), kappa(t).numpy(), kappa


def basic(C: Context, deleted: np.ndarray, first: int) -> dict:
    """Per source: letters, inserted, deleted, correct, wrong, missed and the string-level counts."""
    n = int(C.offsets[first])
    ins = C.ins[:n]
    d = deleted[:n]
    out = {"letters": C.lengths[:first], "inserted": C.ps(ins, first), "deleted": C.ps(d, first), "tp": C.ps(d & ins, first),
           "wrong": C.ps(d & ~ins, first), "missed": C.ps(~d & ins, first)}
    lcs = nb_lcs(C.letters[:n], ins, d, C.offsets[: first + 1]).astype(np.float64)
    out["s_wrong"] = (out["letters"] - out["inserted"]) - lcs  # target letters missing from the output
    out["s_missed"] = (out["letters"] - out["deleted"]) - lcs  # letters the output has too many
    return out


def self_checks(C: Context):
    """The re-implementations against the tested primitives on a sample of sources."""
    z, first = C.corpus(C.errors / "m100.npz")
    tag = "m100"
    if z is None:
        z, first = C.corpus(C.study / "g.npz")
        tag = "g"
    if z is None:
        return
    deleted = z["del_step"] >= 0
    rng = np.random.default_rng(C.args.seed)
    sample = np.sort(rng.choice(first, size=min(150, first), replace=False))
    b = basic(C, deleted, first)
    bad_lcs = bad_clusters = n_clusters = 0
    for gap in (1, 2, 4):
        t = C.cluster_table(tag, deleted, first, gap)
        for i in sample:
            span = C.sources.span(i)
            ref = clusters(C.sources.seq(i), C.sources.mask(i), deleted[span], C.eligible, gap=gap)
            mine = t[t[:, 0] == i]
            n_clusters += len(ref)
            if len(ref) != len(mine):
                bad_clusters += max(len(ref), len(mine))
                continue
            if len(ref):
                same = ((ref.start.to_numpy() + span.start == mine[:, 1]) & (ref.end.to_numpy() + span.start == mine[:, 2])
                        & (ref.n_x.to_numpy() == mine[:, 3]) & (ref.n_miss.to_numpy() == mine[:, 4])
                        & (ref.kind.to_numpy() == np.array(list(KINDS))[mine[:, 7]]))
                bad_clusters += int((~same).sum())
    for i in sample:
        span = C.sources.span(i)
        counts = canonical_counts(C.sources.seq(i), C.sources.mask(i), deleted[span])
        bad_lcs += int(counts["wrong"] != b["s_wrong"][i] or counts["missed"] != b["s_missed"][i])
    C.summary["self_checks"] = {"corpus": tag, "sources_compared": int(len(sample)), "clusters_compared (gaps 1, 2, 4)": n_clusters,
                                "clusters_that_differ_from_editflows.errors.clusters": bad_clusters,
                                "sources_whose_string_counts_differ_from_canonical_counts": bad_lcs}
    print(f"self-check on {len(sample)} sources of {tag}: {bad_clusters} of {n_clusters} clusters differ, {bad_lcs} string counts differ")
    assert bad_clusters == 0 and bad_lcs == 0, "the re-implementation disagrees with editflows.errors"


# ---- C1: the perfect sampler -----------------------------------------------------------------------------------------


def reliability(C: Context, q: np.ndarray, labels: np.ndarray):
    """Calibration of q against 0/1 labels: mean bias, expected calibration error over fixed bins, log loss."""
    y = labels.astype(np.float64)
    q = q.astype(np.float64)
    b = np.clip(np.digitize(q, Q_EDGES) - 1, 0, len(Q_EDGES) - 2)
    nb = len(Q_EDGES) - 1
    key = C.seq_of * nb + b
    Y = np.bincount(key, weights=y, minlength=C.S * nb).reshape(C.S, nb)
    Q = np.bincount(key, weights=q, minlength=C.S * nb).reshape(C.S, nb)
    loss = C.ps(-np.log(np.clip(np.where(labels, q, 1 - q), 1e-12, 1.0)))
    ece = C.boot(lambda a, c, L: np.abs(a - c).sum(-1) / L, Y, Q, C.lengths)
    bias = C.boot(lambda a, c, L: (a - c).sum(-1) / L, Y, Q, C.lengths)
    return {"mean_bias (share inserted minus stated)": triple(bias), "calibration_error": triple(ece),
            "log_loss": triple(C.ratio(loss, C.lengths)),
            "by_bin": [{"q_from": float(Q_EDGES[k]), "q_to": float(Q_EDGES[k + 1]), "stated": float(Q[:, k].sum() / max(1, (b == k).sum())),
                        "observed": float(Y[:, k].sum() / max(1, (b == k).sum())), "n": int((b == k).sum())} for k in range(nb)]}


def claim1(C: Context):
    print("C1 perfect sampler")
    d, first = C.deleted(C.study / "g.npz")
    if d is None or C.q_gap is None:
        C.claim("C1", "perfect sampler: files missing", None, None)
        return
    b = basic(C, d, first)
    k = C.kind_sums("g", d, first)
    n = f"{first} sources, {int(b['letters'].sum())} letters"
    est = C.boot(lambda w, m, L: 1000 * (w + m) / L, b["wrong"], b["missed"], b["letters"])
    C.claim("C1", "perfect sampler g: errors per 1000 letters", "C1 errors", est[0], est[1:], n=n)
    for name in KINDS:
        e = C.boot(lambda w, m, L: 1000 * (w + m) / L, k[f"x_{name}"], k[f"miss_{name}"], b["letters"])
        C.claim("C1", f"perfect sampler g: errors per 1000 in clusters of kind {name}", f"C1 {name}", e[0], e[1:], n=n)
    q = C.q_gap.astype(np.float64)
    law = C.ps(2 * q * (1 - q))
    e = C.ratio(law, C.lengths, 1000)
    C.claim("C1", "law: sum of 2 q (1 - q) with q_gap, per 1000 letters", "C1 law", e[0], e[1:], n=n)
    diff = C.boot(lambda w, m, a, L: 1000 * (w + m - a) / L, b["wrong"], b["missed"], law[:first], b["letters"])
    C.claim("C1", "observed minus law (per 1000; 0 if g is a sample of q_gap)", None, diff[0], diff[1:], n=n,
            verdict="reproduced" if diff[1] <= 0 <= diff[2] else "not reproduced", note="the interval must contain 0")
    rel = {name: reliability(C, getattr(C, name), C.ins) for name in ("q_gap", "q_true") if getattr(C, name) is not None}
    C.claim("C1", "log loss of q_gap on the insertion labels", "C1 logloss gap", rel["q_gap"]["log_loss"][0], rel["q_gap"]["log_loss"][1:], n=n)
    if "q_true" in rel:
        C.claim("C1", "log loss of q_true on the insertion labels", "C1 logloss true", rel["q_true"]["log_loss"][0], rel["q_true"]["log_loss"][1:], n=n)
    # calibration: against the truth, and against a joint sample drawn from the same model (what an exactly calibrated q gives)
    null = reliability(C, C.q_gap, d)
    bias, ece = rel["q_gap"]["mean_bias (share inserted minus stated)"], rel["q_gap"]["calibration_error"]
    calibrated = bias[1] <= 0 <= bias[2] and ece[0] <= 2 * null["calibration_error"][0]
    C.claim("C1", "q_gap calibrated: mean bias (share inserted minus stated q)", None, bias[0], bias[1:], n=n,
            verdict="reproduced" if calibrated else "not reproduced",
            note=f"calibration error {ece[0]:.5f} against {null['calibration_error'][0]:.5f} for labels drawn from q_gap itself (g.npz)"
                 + (f"; q_true: bias {rel['q_true']['mean_bias (share inserted minus stated)'][0]:+.5f}, calibration error "
                    f"{rel['q_true']['calibration_error'][0]:.5f}" if "q_true" in rel else ""))
    # the replicates: is every letter drawn with its own probability?
    reps = [C.deleted(C.study / f"g-rep{r}.npz") for r in range(64) if (C.study / f"g-rep{r}.npz").exists()]
    extra = {}
    if len(reps) >= 4:
        m = min(f for _, f in reps)
        nn = int(C.offsets[m])
        D = np.stack([x[:nn] for x, _ in reps])
        freq, qq = D.mean(0), q[:nn]
        ok = (qq > 0.02) & (qq < 0.98)
        z2 = (freq[ok] - qq[ok]) ** 2 / (qq[ok] * (1 - qq[ok]) / len(reps))
        extra = {"replicates": len(reps), "sources": m, "letters with 0.02 < q_gap < 0.98": int(ok.sum()),
                 "mean squared z-score of the per-letter frequency (1 if every letter is drawn with probability q_gap)": float(z2.mean()),
                 "deleted per 1000 letters, mean over replicates": float(1000 * D.mean()), "sum of q_gap per 1000 letters": float(1000 * qq.mean())}
    C.summary["C1"] = {"reliability": rel, "q_gap against a sample of itself (null)": null, "g_replicates_against_q_gap": extra,
                       "f1": triple(C.boot(lambda tp, dd, i: 2 * tp / (dd + i), b["tp"], b["deleted"], b["inserted"])),
                       "string_f1": triple(C.boot(lambda dd, sw, i: 2 * (dd - sw) / (dd + i), b["deleted"], b["s_wrong"], b["inserted"]))}


# ---- C2: the trained model -------------------------------------------------------------------------------------------


def claim2(C: Context):
    print("C2 trained model")
    d, first = C.deleted(C.errors / "m100.npz")
    if d is None:
        C.claim("C2", "trained model: m100.npz missing", None, None)
        return
    b = basic(C, d, first)
    k = C.kind_sums("m100", d, first)
    n = f"{first} sources, {int(b['letters'].sum())} letters"
    for what, key, num in (("errors", "C2 errors", b["wrong"] + b["missed"]), ("wrong deletions", "C2 wrong", b["wrong"]),
                           ("missed insertions", "C2 missed", b["missed"])):
        e = C.ratio(num, b["letters"], 1000)
        C.claim("C2", f"model m100: {what} per 1000 letters", key, e[0], e[1:], n=n)
    for name in KINDS:
        e = C.ratio(k[f"x_{name}"] + k[f"miss_{name}"], b["letters"], 1000)
        C.claim("C2", f"model m100: errors per 1000 in clusters of kind {name}", f"C2 {name}", e[0], e[1:], n=n)
    f1 = C.boot(lambda tp, dd, i: 2 * tp / (dd + i), b["tp"], b["deleted"], b["inserted"])
    sf1 = C.boot(lambda dd, sw, i: 2 * (dd - sw) / (dd + i), b["deleted"], b["s_wrong"], b["inserted"])
    C.claim("C2", "model m100: F1 on the labels", "C2 f1", f1[0], f1[1:], n=n)
    C.claim("C2", "model m100: F1 on the strings (LCS)", "C2 string f1", sf1[0], sf1[1:], n=n)
    C.summary["C2"] = {"precision": triple(C.ratio(b["tp"], b["deleted"])), "recall": triple(C.ratio(b["tp"], b["inserted"])),
                       "string_errors_per_1000": triple(C.ratio(b["s_wrong"] + b["s_missed"], b["letters"], 1000)),
                       "deleted_minus_inserted_per_1000": triple(C.boot(lambda dd, i, L: 1000 * (dd - i) / L, b["deleted"], b["inserted"], b["letters"]))}
    g = C.summary.get("C1")
    if g:
        C.claim("C2", "perfect sampler g: F1 on the labels", "C2 f1 perfect", g["f1"][0], g["f1"][1:])
        C.claim("C2", "perfect sampler g: F1 on the strings (LCS)", "C2 string f1 perfect", g["string_f1"][0], g["string_f1"][1:])


# ---- C3: exact rates in the Euler sampler ----------------------------------------------------------------------------


def claim3(C: Context):
    print("C3 exact rates")
    ffbs, f_first = C.deleted(C.study / "ffbs.npz")
    i2, i2_first = C.deleted(C.study / "i2-100.npz")
    out = {}
    if ffbs is None or i2 is None:
        C.claim("C3", "exact rates: files missing", None, None)
        return
    err = lambda d: C.ps(d ^ C.ins[: len(d)], min(len(C.offsets) - 1, int(np.searchsorted(C.offsets, len(d)))))  # noqa: E731
    m = min(f_first, i2_first)
    e = C.boot(lambda a, b_, L: 1000 * (a - b_) / L, err(i2)[:m], err(ffbs)[:m], C.lengths[:m])
    C.claim("C3", "i2-100 minus ffbs, errors per 1000 (same sources)", "C3 i2-ffbs", e[0], e[1:], n=f"{m} sources", tol=0.06)
    i3, i3_first = C.deleted(C.study / "i3-100.npz")
    if i3 is not None:
        m3 = min(i3_first, i2_first)
        e = C.boot(lambda a, b_, L: 1000 * (a - b_) / L, err(i3)[:m3], err(i2)[:m3], C.lengths[:m3])
        C.claim("C3", "i3-100 minus i2-100, errors per 1000 (same sources)", "C3 i3-i2", e[0], e[1:], n=f"{m3} sources", tol=0.06)
    # step size: errors against h = 1 / n on the common sources, a straight line in h, its value at h = 0 against ffbs
    sweep = {}
    for path in sorted(C.study.glob("i2-*.npz")):
        tail = path.stem[3:]
        if tail.isdigit():
            d, first = C.deleted(path)
            if d is not None:
                sweep[int(tail)] = (d, first)
    if len(sweep) >= 3:
        m = min([f for _, f in sweep.values()] + [f_first])
        steps = np.array(sorted(sweep))
        h = 1.0 / steps
        E = np.stack([err(sweep[s][0])[:m] for s in steps], 1)  # sources x step counts
        R = np.stack([C.ps(nb_runs(C.ins[: len(sweep[s][0])], sweep[s][0], C.offsets[: sweep[s][1] + 1]) >= 2, sweep[s][1])[:m] for s in steps], 1)
        ref, L = err(ffbs)[:m], C.lengths[:m]
        hc = h - h.mean()

        def line(e_sum, ref_sum, l_sum):
            y = 1000 * e_sum / l_sum[..., None]
            slope = (y * hc).sum(-1) / (hc**2).sum()
            return np.stack([y.mean(-1) - slope * h.mean() - 1000 * ref_sum / l_sum, slope], -1)

        est = C.boot(line, E, ref, L)
        for j, s in enumerate(steps):
            e = C.boot(lambda a, b_, l_: 1000 * (a - b_) / l_, E[:, j], ref, L)
            out[f"i2-{s} minus ffbs, errors per 1000"] = triple(e)
            out[f"i2-{s}: wrong deletions in runs of 2 or more, per 1000"] = triple(C.ratio(R[:, j], L, 1000))
        ref_runs = C.ps(nb_runs(C.ins, ffbs, C.offsets) >= 2)[:m]
        out["ffbs: wrong deletions in runs of 2 or more, per 1000"] = triple(C.ratio(ref_runs, L, 1000))
        run_line = C.boot(line, R, ref_runs, L)
        out["wrong deletions in runs: limit h -> 0 minus ffbs, and slope per unit h"] = [triple([run_line[0][i], run_line[1][i], run_line[2][i]]) for i in range(2)]
        C.claim("C3", "exact rates, limit h -> 0 minus ffbs, errors per 1000", "C3 limit", est[0][0], (est[1][0], est[2][0]),
                n=f"{m} sources, steps {list(map(int, steps))}", verdict="reproduced" if est[1][0] <= 0 <= est[2][0] else "not reproduced",
                note=f"slope {est[0][1]:.0f} [{est[1][1]:.0f}, {est[2][1]:.0f}] errors per 1000 per unit h: "
                     + ("the interval of the slope contains 0, so these files do not show a dependence on h at all"
                        if est[1][1] <= 0 <= est[2][1] else "errors fall with h")
                     + f"; holes (wrong deletions in runs of 2+) limit minus ffbs {run_line[0][0]:+.2f} [{run_line[1][0]:+.2f}, {run_line[2][0]:+.2f}], "
                       f"slope {run_line[0][1]:.0f} [{run_line[1][1]:.0f}, {run_line[2][1]:.0f}]")
    C.summary["C3"] = out


# ---- C4: twins -------------------------------------------------------------------------------------------------------


def nearest_originals(C: Context):
    """For every letter the position of the nearest original on its left and right within its source (-1: none)."""
    n = len(C.ins)
    pos = np.arange(n)
    start, stop = C.offsets[:-1][C.seq_of], C.offsets[1:][C.seq_of]
    left = np.maximum.accumulate(np.where(~C.ins, pos, -1))
    left = np.concatenate([[-1], left[:-1]])  # strictly to the left
    right = np.minimum.accumulate(np.where(~C.ins, pos, n)[::-1])[::-1]
    right = np.concatenate([right[1:], [n]])
    left = np.where(left >= start, left, -1)
    right = np.where(right < stop, right, -1)
    return left, right


def claim4(C: Context):
    print("C4 twins")
    left, right = nearest_originals(C)
    L = C.letters
    eq_l = (left >= 0) & (L[np.maximum(left, 0)] == L)
    eq_r = (right >= 0) & (L[np.maximum(right, 0)] == L)
    pos = np.arange(len(L))
    adj = (eq_l & (left == pos - 1)) | (eq_r & (right == pos + 1))
    ins = C.ins
    n_ins = C.ps(ins)
    e = C.ratio(C.ps(ins & (eq_l | eq_r)), n_ins, 100)
    C.claim("C4", "% of insertions equal to the nearest original on a side", "C4 twin", e[0], e[1:], n=f"{int(ins.sum())} insertions")
    e = C.ratio(C.ps(ins & adj), n_ins, 100)
    C.claim("C4", "% of insertions equal to a directly adjacent original", "C4 adjacent", e[0], e[1:], n=f"{int(ins.sum())} insertions")
    out = {"theory 1 - (19/20)^2": 1 - (19 / 20) ** 2}
    if C.q_gap is not None:
        q = C.q_gap.astype(np.float64)
        one = ins & (eq_l ^ eq_r)  # exactly one twin original
        partner = np.where(eq_l, left, right)
        tw = np.flatnonzero(one)
        e_i = C.ratio(C.ps(np.where(one, q, 0)), C.ps(one))
        qp = np.zeros(len(q))
        qp[tw] = q[partner[tw]]
        e_o = C.ratio(C.ps(qp), C.ps(one))
        adj_one = one & adj
        e_ia = C.ratio(C.ps(np.where(adj_one, q, 0)), C.ps(adj_one))
        s = q[tw] + qp[tw]
        near_half = float(((q[tw] > 0.4) & (q[tw] < 0.6) & (qp[tw] > 0.4) & (qp[tw] < 0.6)).mean())
        spread = (f"medians {np.median(q[tw]):.2f} and {np.median(qp[tw]):.2f}, but only {100 * near_half:.0f} % of the pairs have both values between 0.4 and 0.6 "
                  f"(quartiles {np.quantile(q[tw], 0.25):.2f}-{np.quantile(q[tw], 0.75):.2f} and {np.quantile(qp[tw], 0.25):.2f}-{np.quantile(qp[tw], 0.75):.2f})")
        C.claim("C4", "mean exact q_gap of an insertion that has a twin original", "C4 half", e_i[0], e_i[1:], tol=0.06, n=f"{len(tw)} insertions", note=spread)
        C.claim("C4", "mean exact q_gap of that twin original", "C4 half", e_o[0], e_o[1:], tol=0.06, n=f"{len(tw)} originals", note=spread)
        out.update({"q_gap of the insertion, adjacent twins only": triple(e_ia),
                    "share of twin pairs whose two q_gap are both within 0.4-0.6": float(((q[tw] > 0.4) & (q[tw] < 0.6) & (qp[tw] > 0.4) & (qp[tw] < 0.6)).mean()),
                    "quartiles of q_gap of the insertion": [float(v) for v in np.quantile(q[tw], [0.25, 0.5, 0.75])],
                    "quartiles of q_gap of the twin original": [float(v) for v in np.quantile(qp[tw], [0.25, 0.5, 0.75])],
                    "quartiles of the sum of both": [float(v) for v in np.quantile(s, [0.25, 0.5, 0.75])]})
    d, first = C.deleted(C.errors / "m100.npz")
    if d is not None:
        t = C.cluster_table("m100", d, first)
        pair = (t[:, 2] - t[:, 1] == 1) & (t[:, 3] == 1) & (t[:, 4] == 1)  # exactly `x_` or `_x`
        same = L[t[:, 1]] == L[t[:, 2]]
        for label, ids in (("sources of the reporting half", t[:, 0] >= C.S // 2), ("all sources", np.ones(len(t), bool))):
            sel = pair & ids
            e = C.ratio(np.bincount(t[sel & same, 0], minlength=first), np.bincount(t[sel, 0], minlength=first), 100)
            if label.startswith("sources"):
                C.claim("C4", "% of the model's `x_` / `_x` clusters with two identical letters", "C4 identical", e[0], e[1:],
                        n=f"{int(sel.sum())} clusters, sources {C.S // 2}-{first - 1}")
            out[f"identical letters in x_ / _x clusters, {label}"] = triple(e)
    C.summary["C4"] = out


# ---- C5: hole runs ---------------------------------------------------------------------------------------------------


def claim5(C: Context):
    print("C5 hole runs")
    half = np.arange(C.S) >= C.S // 2
    out = {}
    m, m_first = C.deleted(C.errors / "m100.npz")
    g, g_first = C.deleted(C.study / "g.npz")
    i2, i2_first = C.deleted(C.study / "i2-100.npz")
    km = kg = None
    if m is not None:
        km = C.kind_sums("m100", m, m_first)
        e = C.ratio(km["hole_runs"], C.lengths[:m_first], 1000, ids=half[:m_first])
        C.claim("C5", "hole runs per 1000 letters, model", "C5 model", e[0], e[1:], n=f"sources {C.S // 2}-{m_first - 1}")
    if g is not None:
        kg = C.kind_sums("g", g, g_first)
        e = C.ratio(kg["hole_runs"], C.lengths[:g_first], 1000, ids=half[:g_first])
        C.claim("C5", "hole runs per 1000 letters, perfect sampler", "C5 perfect", e[0], e[1:], n=f"sources {C.S // 2}-{g_first - 1}")
    if i2 is not None:
        ki = C.kind_sums("i2-100", i2, i2_first)
        e = C.ratio(ki["hole_runs"], C.lengths[:i2_first], 1000)
        C.claim("C5", "hole runs per 1000 letters, exact rates (i2-100)", "C5 exact", e[0], e[1:], n=f"sources 0-{i2_first - 1}")
        if m is not None:
            out["model on the sources of i2-100"] = triple(C.ratio(km["hole_runs"][:i2_first], C.lengths[:i2_first], 1000))
    if km is not None and kg is not None:
        f = min(m_first, g_first)
        xm, xg = C.ps(m & ~C.ins[: len(m)], m_first)[:f], C.ps(g & ~C.ins[: len(g)], g_first)[:f]
        e = C.boot(lambda hm, hg, a, b_: 100 * (hm - hg) / (a - b_), km["x_in_hole_runs"][:f], kg["x_in_hole_runs"][:f], xm, xg, ids=half[:f])
        C.claim("C5", "% of the model's surplus of wrong deletions that lies in hole runs", "C5 surplus", e[0], e[1:],
                n=f"sources {C.S // 2}-{f - 1}", tol=1.0)
    C.summary["C5"] = out


# ---- C6: certain originals -------------------------------------------------------------------------------------------


def claim6(C: Context):
    print("C6 certain originals")
    z, first = C.corpus(C.errors / "m100.npz")
    g, g_first = C.deleted(C.study / "g.npz")
    ffbs, f_first = C.deleted(C.study / "ffbs.npz")
    out = {}
    if z is None or C.q_gap is None:
        C.claim("C6", "certain originals: files missing", None, None)
        return
    d = z["del_step"] >= 0
    n = len(d)
    orig = ~C.ins[:n]
    for name, q in (("q_gap", C.q_gap), ("q_true", C.q_true)):
        if q is None:
            continue
        sure = orig & (q[:n] < 1e-4)
        e = C.ratio(C.ps(sure & d, first), C.ps(sure, first), 100)
        C.claim("C6", f"% of originals with {name} < 1e-4 that the model deletes", "C6 model", e[0], e[1:], n=f"{int(sure.sum())} originals",
                note="q_gap: the reference posterior (prior of the coupling)" if name == "q_gap" else "q_true: the simpler prior of independent insertions")
        out[f"model, originals with {name} < 1e-4: % deleted"] = triple(e)
        out[f"model, originals with {name} < 1e-4: share of all originals"] = float(sure.sum() / orig.sum())
        out[f"model, originals with {name} < 1e-4: share of all wrong deletions"] = float((sure & d).sum() / (orig & d).sum())
        for tag, ref, r_first in (("g", g, g_first), ("ffbs", ffbs, f_first)):
            if ref is not None:
                s2 = ~C.ins[: len(ref)] & (q[: len(ref)] < 1e-4)
                e = C.ratio(C.ps(s2 & ref, r_first), C.ps(s2, r_first), 100)
                out[f"{tag}, originals with {name} < 1e-4: % deleted"] = triple(e)
                if tag == "g" and name == "q_gap":
                    C.claim("C6", "% of originals with q_gap < 1e-4 that the perfect sampler deletes", "C6 perfect", e[0], e[1:], tol=0.01,
                            n=f"{int(s2.sum())} originals")
    step, q0 = z["del_step"], z["q0"]
    late = orig & d & (q0 < 0.01)
    C.claim("C6", "median step of wrong deletions of originals with model q0 < 0.01", "C6 median wrong", float(np.median(step[late])), tol=1.5,
            n=f"{int(late.sum())} wrong deletions")
    C.claim("C6", "median step of correct deletions", "C6 median correct", float(np.median(step[d & ~orig])), tol=1.5, n=f"{int((d & ~orig).sum())} deletions")
    if C.hazard is not None:
        # when would a letter with a small constant q be deleted? the reference without which "late" means nothing
        mass = C.hazard[:-1] / C.hazard[:-1].sum()
        out["median step of a deletion with a small constant q (hazard-weighted)"] = int(np.searchsorted(np.cumsum(mass), 0.5))
        out["median step of ALL wrong deletions"] = float(np.median(step[orig & d]))
        out["median step of wrong deletions with q0 >= 0.1"] = float(np.median(step[orig & d & (q0 >= 0.1)]))
    C.summary["C6"] = out


# ---- C7: first reading -----------------------------------------------------------------------------------------------


def claim7(C: Context):
    print("C7 first reading")
    z, first = C.corpus(C.errors / "m100.npz")
    if z is None or C.kappa_t is None:
        C.claim("C7", "first reading: m100.npz or the schedule of the run missing", None, None)
        return
    d, q0 = z["del_step"] >= 0, z["q0"].astype(np.float64)
    n = len(d)
    ins, orig = C.ins[:n], ~C.ins[:n]
    L = C.lengths[:first]
    k1 = float(C.kappa_t[-1])
    nn = f"{first} sources"
    e = C.ratio(k1 * C.ps(np.where(orig, q0, 0), first), L, 1000)
    C.claim("C7", "kappa(1) x sum of model q0 over originals, per 1000 letters", "C7 first reading", e[0], e[1:], n=nn)
    out = {"kappa(1)": k1}
    for name, q in (("q_true", C.q_true), ("q_gap", C.q_gap)):
        if q is not None:
            e = C.ratio(C.ps(np.where(orig, q[:n], 0), first), L, 1000)
            out[f"sum of {name} over originals per 1000"] = triple(e)
            if name == "q_true":
                C.claim("C7", "sum of q_true over originals, per 1000 letters", "C7 exact", e[0], e[1:], n=nn)
            else:
                C.claim("C7", "sum of q_gap over originals, per 1000 letters (the reference prior)", None, e[0], e[1:], n=nn)
    e = C.ratio(C.ps(orig & d, first), L, 1000)
    C.claim("C7", "observed wrong deletions per 1000 letters", "C7 observed", e[0], e[1:], n=nn)
    # the same account on the other side: what the first reading asks for the inserted letters
    out["first reading asks for missed insertions per 1000: sum over insertions of (1 - kappa(1) q0)"] = triple(
        C.ratio(C.ps(np.where(ins, 1 - k1 * q0, 0), first), L, 1000))
    out["observed missed insertions per 1000"] = triple(C.ratio(C.ps(ins & ~d, first), L, 1000))
    out["first reading asks for deletions per 1000: kappa(1) x sum of q0 over all letters"] = triple(C.ratio(k1 * C.ps(q0, first), L, 1000))
    out["observed deletions per 1000"] = triple(C.ratio(C.ps(d, first), L, 1000))
    out["inserted letters per 1000"] = triple(C.ratio(C.ps(ins, first), L, 1000))
    C.summary["C7"] = out


# ---- C8: the grammar alone -------------------------------------------------------------------------------------------


@numba.njit(cache=True)
def nb_repair(target, sites, window, eligible):
    """Fewest further deletions among the `window` letters on each side of a removed letter that leave no forbidden
    window (the two letters beyond the stretch on each side stay); -1 if no choice works. Dynamic programme over
    the last two kept letters."""
    n_letters = eligible.shape[0]
    big = 1 << 30
    out = np.empty(len(sites), dtype=np.int64)
    cost = np.empty((n_letters, n_letters), dtype=np.int64)
    new = np.empty((n_letters, n_letters), dtype=np.int64)
    for e in range(len(sites)):
        s = sites[e]
        cost[:, :] = big
        cost[target[s - window - 2], target[s - window - 1]] = 0
        for i in range(s - window, s + window + 1):
            if i == s:
                continue  # the removed letter
            c = target[i]
            for a in range(n_letters):
                for b in range(n_letters):
                    new[a, b] = cost[a, b] + 1 if cost[a, b] < big else big
            for a in range(n_letters):
                for b in range(n_letters):
                    if cost[a, b] < big and eligible[a, b, c] and cost[a, b] < new[b, c]:
                        new[b, c] = cost[a, b]
            cost, new = new, cost
        r1, r2 = target[s + window + 1], target[s + window + 2]
        best = big
        for a in range(n_letters):
            for b in range(n_letters):
                if cost[a, b] < best and eligible[a, b, r1] and eligible[b, r1, r2]:
                    best = cost[a, b]
        out[e] = best if best < big else -1
    return out


@numba.njit(cache=True)
def nb_repair_brute(target, sites, window, eligible, most):
    """The same by trying every set of at most `most` further deletions (a check of the dynamic programme);
    `most + 1` if none works."""
    out = np.empty(len(sites), dtype=np.int64)
    m = 2 * window
    idx = np.empty(m, dtype=np.int64)
    seq = np.empty(m + 4, dtype=np.int64)
    for e in range(len(sites)):
        s = sites[e]
        j = 0
        for i in range(s - window, s + window + 1):
            if i != s:
                idx[j] = i
                j += 1
        found = most + 1
        for size in range(most + 1):
            if found <= most:
                break
            # all subsets of `size` of the m letters, as ascending index tuples
            pick = np.arange(size)
            while True:
                n = 0
                seq[0], seq[1] = target[s - window - 2], target[s - window - 1]
                n = 2
                p = 0
                for j in range(m):
                    if p < size and pick[p] == j:
                        p += 1
                    else:
                        seq[n] = target[idx[j]]
                        n += 1
                seq[n], seq[n + 1] = target[s + window + 1], target[s + window + 2]
                n += 2
                ok = True
                for j in range(2, n):
                    if not eligible[seq[j - 2], seq[j - 1], seq[j]]:
                        ok = False
                        break
                if ok:
                    found = size
                    break
                # next combination
                k = size - 1
                while k >= 0 and pick[k] == m - size + k:
                    k -= 1
                if k < 0:
                    break
                pick[k] += 1
                for j in range(k + 1, size):
                    pick[j] = pick[j - 1] + 1
        out[e] = found
    return out


def claim8(C: Context, window: int = 12, per_target: int = 20):
    print("C8 grammar alone")
    orig = ~C.ins
    T = C.letters[orig]
    seq_t = C.seq_of[orig]
    t_off = np.concatenate([[0], np.cumsum(C.ps(orig).astype(np.int64))])
    pos = np.arange(len(T)) - t_off[seq_t]
    t_len = np.diff(t_off)[seq_t]
    E = C.eligible
    # break size of every letter that has `window + 2` letters on each side (the sites of the reported analysis are a sample of these)
    inner = np.flatnonzero((pos >= window + 2) & (pos < t_len - window - 3))
    b = (~E[T[inner - 2], T[inner - 1], T[inner + 1]]).astype(np.int64) + (~E[T[inner - 1], T[inner + 1], T[inner + 2]])
    n_inner = np.bincount(seq_t[inner], minlength=C.S)
    for size, key in ((0, "C8 b0"), (1, "C8 b1"), (2, "C8 b2")):
        e = C.ratio(np.bincount(seq_t[inner][b == size], minlength=C.S), n_inner, 100)
        C.claim("C8", f"% of removed letters that leave {size} forbidden windows across the gap", key, e[0], e[1:], tol=0.3,
                n=f"all {len(inner)} letters with {window + 2} neighbours on each side",
                note="reported from a sample of 120 000 letters (sampling error about 0.15); here every letter")
    v = float(E.mean())
    rng = np.random.default_rng([C.args.seed, 8])
    chosen = []
    for i in np.flatnonzero(np.diff(t_off) >= 2 * window + 8):
        chosen.append(t_off[i] + rng.choice(np.arange(window + 2, t_off[i + 1] - t_off[i] - window - 3), size=per_target, replace=False))
    sites = np.sort(np.concatenate(chosen))
    repair = nb_repair(T, sites, window, E)
    check = sites[rng.choice(len(sites), size=min(300, len(sites)), replace=False)]
    brute = nb_repair_brute(T, check, window, E, 3)
    dp = nb_repair(T, check, window, E)
    agree_brute = int(((dp == brute) | ((brute == 4) & ((dp > 3) | (dp < 0)))).sum())
    ref = np.array([minimal_repair(np.delete(T[s - window - 2: s + window + 3], window + 2), 2, 2 + 2 * window, E) for s in check[:150]])
    solved = repair >= 0
    src = seq_t[sites]
    hole = C.ratio(np.bincount(src[solved], weights=repair[solved] + 1.0, minlength=C.S), np.bincount(src[solved], minlength=C.S))
    C.claim("C8", f"smallest repair: mean hole in letters, the removed letter included (+-{window})", "C8 hole", hole[0], hole[1:],
            n=f"{int(solved.sum())} sites on {len(chosen)} targets")
    bs = (~E[T[sites - 2], T[sites - 1], T[sites + 1]]).astype(np.int64) + (~E[T[sites - 1], T[sites + 1], T[sites + 2]])
    C.summary["C8"] = {
        "share of allowed windows v": v, "break sizes predicted from v (v^2, 2 v (1 - v), (1 - v)^2)": [v**2, 2 * v * (1 - v), (1 - v) ** 2],
        "sites of the repair": int(len(sites)), "share without a repair inside the window": float((~solved).mean()),
        "median hole": float(np.median(repair[solved] + 1)),
        "mean hole by break size": {int(k): float((repair[solved & (bs == k)] + 1).mean()) for k in range(3)},
        "share of holes of size 1, 2, 3, 4, 5+": [float(((repair[solved] + 1) == k).mean()) for k in (1, 2, 3, 4)] + [float(((repair[solved] + 1) >= 5).mean())],
        "check: sites where the dynamic programme equals trying all sets of up to 3 deletions": f"{agree_brute} of {len(check)}",
        "check: sites where it equals editflows.errors.minimal_repair": f"{int((ref == dp[:150]).sum())} of {len(ref)}"}
    assert agree_brute == len(check) and (ref == dp[:150]).all(), "the repair programme disagrees with its checks"


# ---- C9 and attack (d): the leak experiment ---------------------------------------------------------------------------


def exact_q(C: Context, del_step: np.ndarray, pairs: np.ndarray, batch: int = 384) -> dict:
    """The exact posterior the exact-rate sampler works with: for every pair (source, step) the probability that a
    letter is a leftover insertion, given the letters present before that step and the prior of that step (the
    share of leftovers expected at time t). Returns (source, step) -> (flat positions of the letters present, q)."""
    import torch

    from editflows import exact

    torch.set_num_threads(2)
    P = torch.tensor(C.P, dtype=torch.float64)
    states = []
    for i, k in pairs:
        a, b = int(C.offsets[i]), int(C.offsets[i + 1])
        ds = del_step[a:b]
        states.append(a + np.flatnonzero((ds < 0) | (ds >= k)))
    order = np.argsort([len(x) for x in states], kind="stable")
    out = {}
    for at in range(0, len(order), batch):
        chunk = order[at: at + batch]
        letters, valid = exact.pad([C.letters[states[j]] for j in chunk], "cpu")
        prior = exact.leftover_share(torch.tensor(C.rho[pairs[chunk, 0]], dtype=torch.float64), torch.tensor(C.kappa_t[pairs[chunk, 1]], dtype=torch.float64))
        q = exact.posterior(letters, valid, prior, P).numpy()
        for r, j in enumerate(chunk):
            out[int(pairs[j, 0]), int(pairs[j, 1])] = (states[j], q[r, : len(states[j])])
    return out


def leak_seeds(C: Context, tag: str, leak: float, del_step: np.ndarray, first: int, grid: int = 10):
    """What the leak itself did in one run, from the exact q of the sampler (cached in the scratch folder).

    * `q_del`: for every wrongly deleted original its exact q in the step of its deletion. Below the leak the
      letter was deleted at the rate of the leak; the share (leak - q) / leak of these deletions is the leak's.
    * `q_grid`: the exact q of every letter in the states before steps 0, `grid`, 2 `grid`, ... (NaN once gone),
      for the expected number of leak deletions and for the letters the leak cannot reach (q above the leak)."""
    n = int(C.offsets[first])
    path = C.study / f"{tag}.npz"
    cache = C.out / "scratch" / f"leak-{tag}.npz"
    key = f"{path.stat().st_mtime_ns}-{path.stat().st_size}-{first}-{grid}"
    if cache.exists():
        z = load_npz(cache)
        if z is not None and str(z["key"]) == key:
            return z["q_del"], z["q_grid"]
    orig = ~C.ins[:n]
    wrong = np.flatnonzero(orig & (del_step[:n] >= 0))
    pairs = np.unique(np.stack([C.seq_of[wrong], del_step[wrong].astype(np.int64)], 1), axis=0)
    start = time.time()
    states = exact_q(C, del_step, pairs)
    q_del = np.full(n, np.nan)
    for (i, k), (present, q) in states.items():
        hit = del_step[present] == k
        q_del[present[hit]] = q[hit]
    steps = np.arange(0, C.kappa_t.shape[0] - 1, grid)
    q_grid = np.full((n, len(steps)), np.nan, dtype=np.float32)
    for g, k in enumerate(steps):
        for (i, _), (present, q) in exact_q(C, del_step, np.stack([np.arange(first), np.full(first, k)], 1)).items():
            q_grid[present, g] = q
    print(f"    exact q along {tag}: {len(pairs)} states with a wrong deletion, {len(steps)} full passes, {time.time() - start:.0f} s", flush=True)
    np.savez_compressed(cache, key=key, q_del=q_del, q_grid=q_grid)
    return q_del, q_grid


def claim9(C: Context):
    print("C9 leak experiment, attack (d)")
    tags = sorted((p_.stem for p_ in C.study.glob("i2-leak*.npz")), key=lambda t: float(t[7:]))
    base, b_first = C.deleted(C.study / "i2-100.npz")
    if not tags or base is None or C.hazard is None:
        C.claim("C9", "leak experiment: files missing", None, None)
        return
    runs = {t: C.corpus(C.study / f"{t}.npz") for t in tags}
    n_src = min([f for _, f in runs.values()] + [b_first])
    n = int(C.offsets[n_src])
    ins, orig = C.ins[:n], ~C.ins[:n]
    L, originals = C.lengths[:n_src], C.ps(orig, n_src)
    n_steps = len(C.hazard) - 1
    p_step = C.hazard[:n_steps] / n_steps
    lam = float(-np.log1p(-C.kappa_t[-1]))  # the cumulative hazard of the schedule
    base_wrong, base_missed = C.ps(base[:n] & orig, n_src), C.ps(~base[:n] & ins, n_src)
    kb = C.kind_sums("i2-100", base, b_first)
    # more baselines: every exact-rate run with the same number of steps on these sources
    more = [base_wrong]
    for k in range(256):
        path = C.study / f"i2-rep{k}.npz"
        if path.exists():
            d_, f_ = C.deleted(path)
            if d_ is not None and f_ >= n_src:
                more.append(C.ps(d_[:n] & orig, n_src))
    pooled_base = np.mean(more, 0)
    out = {"sources": n_src, "cumulative hazard": lam, "baselines pooled (i2-100 and i2-rep*)": len(more),
           "baseline wrong deletions per 1000: i2-100 alone, pooled": [float(1000 * base_wrong.sum() / L.sum()), float(1000 * pooled_base.sum() / L.sum())]}
    rows, per_leak, seed_steps = [], {}, {}
    naive_all, extra_all, seeds_all, extra_pooled_all = [], [], [], []
    for tag in tags:
        leak = float(tag[7:])
        z, _ = runs[tag]
        step = z["del_step"][:n].astype(np.int64)
        d = step >= 0
        wrong, missed = C.ps(d & orig, n_src), C.ps(~d & ins, n_src)
        naive = (1 - np.exp(-leak * lam)) * originals  # the reported seeds: the leak acts on every original for the whole path
        k_ = C.kind_sums(tag, d, n_src)
        cost_naive = C.boot(lambda w, b_, s_: (w - b_) / s_, wrong, base_wrong, naive)
        # --- own estimate of the seeds
        q_del, q_grid = leak_seeds(C, tag, leak, z["del_step"], n_src)
        wd = orig & d
        weight = np.where(wd, np.clip(leak - np.nan_to_num(q_del, nan=1.0), 0, None) / leak, 0.0)  # share of the deletion that is the leak's
        seeds_obs = C.ps(weight, n_src)
        seed_steps[tag] = (step[weight > 0], weight[weight > 0])
        at_leak_rate = C.ps(wd & (q_del < leak), n_src)
        # expected leak deletions from the exact q on a grid of steps: h x hazard x (leak - q)+ while the letter is present
        grid = (n_steps + q_grid.shape[1] - 1) // q_grid.shape[1]
        expected = np.zeros(n)
        reach = np.zeros((q_grid.shape[1], 2))
        for g in range(q_grid.shape[1]):
            qg = q_grid[:, g].astype(np.float64)
            below = np.clip(leak - qg, 0, None)  # NaN (gone) -> NaN
            for k in range(g * grid, min((g + 1) * grid, n_steps)):
                present = ~d | (step >= k)
                expected += np.where(present & np.isfinite(below), p_step[k] * np.nan_to_num(below), 0.0)
            there = np.isfinite(qg)
            reach[g] = [float((qg[there & orig] < leak).mean()), float((qg[there & ins] < leak).mean())]
        seeds_exp, leaked_ins = C.ps(np.where(orig, expected, 0), n_src), C.ps(np.where(ins, expected, 0), n_src)
        cost_obs = C.boot(lambda w, b_, s_: (w - b_) / s_, wrong, pooled_base, seeds_obs)
        cost_obs_single = C.boot(lambda w, b_, s_: (w - b_) / s_, wrong, base_wrong, seeds_obs)
        cost_exp = C.boot(lambda w, b_, s_: (w - b_) / s_, wrong, pooled_base, seeds_exp)
        ratio = C.boot(lambda a, b_: a / b_, seeds_obs, naive)
        per_leak[tag] = {
            "leak": leak, "seeds per 1000 letters as reported: (1 - exp(-leak x cumulative hazard)) per original": float(1000 * naive.sum() / L.sum()),
            "seeds per 1000 letters, counted: wrong deletions made at the rate of the leak, weighted with (leak - q) / leak": triple(C.ratio(seeds_obs, L, 1000)),
            "seeds per 1000 letters, expected from the exact q on a grid of steps": triple(C.ratio(seeds_exp, L, 1000)),
            "counted seeds as a share of the reported seeds": triple(ratio),
            "wrong deletions made while the exact q was below the leak, per 1000": triple(C.ratio(at_leak_rate, L, 1000)),
            "extra wrong deletions per 1000 against i2-100": triple(C.boot(lambda w, b_, l_: 1000 * (w - b_) / l_, wrong, base_wrong, L)),
            "extra wrong deletions per 1000 against the pooled baseline": triple(C.boot(lambda w, b_, l_: 1000 * (w - b_) / l_, wrong, pooled_base, L)),
            "extra missed insertions per 1000 against i2-100": triple(C.boot(lambda w, b_, l_: 1000 * (w - b_) / l_, missed, base_missed, L)),
            "insertions the leak deletes (expected), per 1000": triple(C.ratio(leaked_ins, L, 1000)),
            "wrong deletions per seed, as reported (seeds by the formula, baseline i2-100)": triple(cost_naive),
            "wrong deletions per seed, counted seeds, pooled baseline": triple(cost_obs),
            "wrong deletions per seed, counted seeds, baseline i2-100": triple(cost_obs_single),
            "wrong deletions per seed, expected seeds, pooled baseline": triple(cost_exp),
            "share of the originals present whose exact q is below the leak, at steps 0, middle, last grid step": [float(reach[0, 0]), float(reach[len(reach) // 2, 0]), float(reach[-1, 0])],
            "share of the insertions present whose exact q is below the leak, at steps 0, middle, last grid step": [float(reach[0, 1]), float(reach[len(reach) // 2, 1]), float(reach[-1, 1])],
            "extra errors per 1000 by kind of cluster (Z, V, R, U) against i2-100": [
                triple(C.boot(lambda a, b_, l_: 1000 * (a - b_) / l_, k_[f"x_{kind}"] + k_[f"miss_{kind}"], (kb[f"x_{kind}"] + kb[f"miss_{kind}"])[:n_src], L)) for kind in KINDS],
            "median step of the counted seeds": float(np.median(step[wd & (q_del < leak / 2)])) if (wd & (q_del < leak / 2)).any() else None,
        }
        rows.append({"leak": leak, "n_sources": n_src, "seeds_reported_per_1000": 1000 * naive.sum() / L.sum(), "seeds_counted_per_1000": 1000 * seeds_obs.sum() / L.sum(),
                     "seeds_expected_per_1000": 1000 * seeds_exp.sum() / L.sum(), "extra_wrong_per_1000_vs_i2-100": 1000 * (wrong - base_wrong).sum() / L.sum(),
                     "extra_wrong_per_1000_vs_pooled": 1000 * (wrong - pooled_base).sum() / L.sum(),
                     "cost_reported": cost_naive[0], "cost_reported_lo": cost_naive[1], "cost_reported_hi": cost_naive[2],
                     "cost_counted_seeds": cost_obs[0], "cost_counted_lo": cost_obs[1], "cost_counted_hi": cost_obs[2],
                     "cost_expected_seeds": cost_exp[0], "cost_expected_lo": cost_exp[1], "cost_expected_hi": cost_exp[2]})
        naive_all.append(naive); extra_all.append(wrong - base_wrong); seeds_all.append(seeds_obs); extra_pooled_all.append(wrong - pooled_base)
        C.claim("C9", f"leak {leak:g}: wrong deletions per leak deletion (seeds by the formula)", "C9 cost", cost_naive[0], cost_naive[1:], tol=0.45,
                n=f"{n_src} sources", note="reported: the same at all three leak sizes")
    pooled = C.boot(lambda e, s_: e / s_, np.sum(extra_all, 0), np.sum(naive_all, 0))
    C.claim("C9", "all leaks: wrong deletions per leak deletion (seeds by the formula)", "C9 cost", pooled[0], pooled[1:], n=f"{n_src} sources")
    mine = C.boot(lambda e, s_: e / s_, np.sum(extra_pooled_all, 0), np.sum(seeds_all, 0))
    C.claim("C9", "all leaks: the same with the seeds counted from the exact q (attack d)", None, mine[0], mine[1:], n=f"{n_src} sources",
            verdict="recomputed", note="seeds = wrong deletions made at the rate of the leak; baseline pooled over all exact-rate runs")
    for i, kind in enumerate(KINDS):
        v = [per_leak[t]["extra errors per 1000 by kind of cluster (Z, V, R, U) against i2-100"][i] for t in tags]
        C.claim("C9", f"largest leak: extra errors per 1000 in clusters of kind {kind}", None, v[-1][0], v[-1][1:], n=f"{n_src} sources",
                verdict="reproduced" if (kind in "RU") == (v[-1][1] > 0) else "not reproduced", note="reported: the additional errors are of kind R and U")
    exp4 = C.errors / "results" / "interventions" / "exp4_model_vs_exact.csv"
    if exp4.exists():
        e4 = pd.read_csv(exp4)
        e4 = e4[e4.family.str.startswith("f")]
        times = np.array(sorted(e4.t0.unique()))
        orig_all = ~C.ins
        tb = C.target_break(np.flatnonzero(orig_all))
        mix = np.array([(tb == v).sum() for v in range(3)], dtype=float)
        mix /= mix.sum()
        further = np.array([sum(mix[v] * float(e4[(e4.t0 == t0) & (e4.break_size == v)].extra_wrong_exact.iloc[0]) for v in range(3)) for t0 in times])
        grid_t, grid_f = np.concatenate([times, [1.0]]), np.concatenate([further, [0.0]])  # a removal at the very end has no time to spread
        predicted = {}
        for tag in tags:
            st_, wt_ = seed_steps[tag]
            predicted[tag] = float(1 + (wt_ * np.interp(st_ / n_steps, grid_t, grid_f)).sum() / wt_.sum())
        out["cross-check with the forced removals under exact rates (table exp4 of the intervention study, not recomputed for t0 > 0)"] = {
            "further wrong deletions after a forced removal at t0, break sizes mixed as in the targets": {f"t0 = {t0:g}": float(v) for t0, v in zip(times, further)},
            "predicted wrong deletions per leak deletion (seed included), given when the seeds fell": predicted}
    # do the runs share their random numbers? with the same seed and the same batch a deletion falls in the same step
    def same_step(a, b):
        gone = a[:n] >= 0
        return float((a[:n][gone] == b[:n][gone]).mean())

    steps_of = {t: runs[t][0]["del_step"] for t in tags}
    shared = {f"{tags[i]} and {tags[i + 1]}": same_step(steps_of[tags[i]], steps_of[tags[i + 1]]) for i in range(len(tags) - 1)}
    shared[f"{tags[0]} and i2-100 (the baseline)"] = same_step(steps_of[tags[0]], C.corpus(C.study / "i2-100.npz")[0]["del_step"])
    out["share of the deletions of one run that the other run makes in the same step (high: the two runs share their random numbers)"] = shared
    out["by leak"] = per_leak
    out["all leaks pooled: wrong deletions per seed as reported"] = triple(pooled)
    out["all leaks pooled: wrong deletions per counted seed"] = triple(mine)
    pd.DataFrame(rows).to_csv(C.out / "attack_d_leak.csv", index=False)
    C.summary["C9_attack_d"] = out
    C.tables["leak"] = pd.DataFrame(rows)


# ---- C10: replicates -------------------------------------------------------------------------------------------------


def replicates(C: Context, pattern: str, k_max: int = 256):
    """Deleted masks (replicates x letters) of the replicate files that exist, cut to the sources all of them cover."""
    runs, firsts = [], []
    for k in range(k_max):
        path = Path(pattern.format(k=k))
        if path.exists():
            d, first = C.deleted(path)
            if d is not None:
                runs.append(d)
                firsts.append(first)
    if not runs:
        return None, 0
    first = min(firsts)
    return np.stack([r[: int(C.offsets[first])] for r in runs]), first


def mh_adjacent(C: Context, wrong: np.ndarray, seq_o: np.ndarray, n_sources: int, keep=None, distance: int = 1):
    """Mantel-Haenszel odds ratio of (wrongly deleted, wrongly deleted) for two originals `distance` apart in the
    target, one 2x2 table per pair across the replicates. `wrong`: replicates x originals."""
    R = wrong.shape[0]
    Wf = wrong.astype(np.float64)
    m = Wf.sum(0)
    d = distance
    pair = seq_o[:-d] == seq_o[d:]
    if keep is not None:
        pair &= keep[:-d] & keep[d:]
    a = (Wf[:, :-d] * Wf[:, d:]).sum(0)
    mi, mj = m[:-d], m[d:]
    num = a * (R - mi - mj + a) / R  # both wrong x neither wrong
    den = (mi - a) * (mj - a) / R  # only the first x only the second
    s = seq_o[:-d][pair]
    return (C.ratio(np.bincount(s, weights=num[pair], minlength=n_sources), np.bincount(s, weights=den[pair], minlength=n_sources)),
            int(((num + den)[pair] > 0).sum()))


def claim10(C: Context):
    print("C10 replicates")
    files = {"model": str(C.errors / "rep{k}.npz"), "exact": str(C.study / "i2-rep{k}.npz"), "perfect": str(C.study / "g-rep{k}.npz")}
    D = {}
    for name, pattern in files.items():
        d, first = replicates(C, pattern)
        if d is not None and len(d) >= 4:
            D[name] = (d, first)
        else:
            C.missing.append(f"{pattern}: fewer than 4 replicates")
    if not D:
        C.claim("C10", "replicates missing", None, None)
        return
    S = min(f for _, f in D.values())
    n = int(C.offsets[S])
    orig = ~C.ins[:n]
    seq_o = C.seq_of[:n][orig]
    r_common = min(len(d) for d, _ in D.values())
    clear = (C.q_gap[:n][orig] < 0.01) if C.q_gap is not None else None
    out = {"sources": S, "replicates": {k: int(len(d)) for k, (d, _) in D.items()}, "replicates used for the claim": r_common}
    for name, (d, _) in D.items():
        wrong = d[:, :n][:, orig]
        est, strata = mh_adjacent(C, wrong[:r_common], seq_o, S)
        C.claim("C10", f"odds ratio of two adjacent wrong deletions (Mantel-Haenszel): {name}", f"C10 {name}", est[0], est[1:],
                n=f"{S} sources, {r_common} replicates, {strata} informative pairs")
        out[name] = {"first replicates": triple(est), "all replicates": triple(mh_adjacent(C, wrong, seq_o, S)[0]),
                     "distance 2": triple(mh_adjacent(C, wrong[:r_common], seq_o, S, distance=2)[0]),
                     "distance 5": triple(mh_adjacent(C, wrong[:r_common], seq_o, S, distance=5)[0])}
        rng = np.random.default_rng([C.args.seed, 10])
        out[name]["letters scrambled across replicates (must be 1)"] = triple(mh_adjacent(C, rng.permuted(wrong[:r_common], axis=0), seq_o, S)[0])
        if clear is not None:
            e, strata = mh_adjacent(C, wrong, seq_o, S, keep=clear)
            out[name]["both originals clear in the source (q_gap < 0.01), all replicates"] = triple(e) + [strata]
            e, strata = mh_adjacent(C, wrong, seq_o, S, keep=~clear)
            out[name]["both originals doubtful in the source (q_gap >= 0.01), all replicates"] = triple(e) + [strata]
    C.summary["C10"] = out


# ---- attack (a): the perfect sampler's share and the kinds -----------------------------------------------------------


def attack_a(C: Context):
    print("attack (a): share of the perfect sampler, other gaps, noise, length, other definitions")
    m, m_first = C.deleted(C.errors / "m100.npz")
    g, g_first = C.deleted(C.study / "g.npz")
    if m is None or g is None:
        C.summary["attack_a"] = "m100.npz or g.npz missing"
        return
    f = min(m_first, g_first)
    n = int(C.offsets[f])
    L = C.lengths[:f]
    bm, bg = basic(C, m, f), basic(C, g, f)
    em, eg = bm["wrong"] + bm["missed"], bg["wrong"] + bg["missed"]
    rows = []

    def add(test, group, quantity, est, n_sources=f):
        rows.append({"test": test, "group": group, "quantity": quantity, "value": est[0], "lo": est[1], "hi": est[2], "n_sources": int(n_sources)})

    share = lambda a, b_: 100 * a / b_  # noqa: E731
    add("all", "all sources", "perfect sampler's label errors as % of the model's", C.boot(share, eg, em))
    add("all", "all sources", "perfect sampler's STRING errors as % of the model's",
        C.boot(share, bg["s_wrong"] + bg["s_missed"], bm["s_wrong"] + bm["s_missed"]))
    add("all", "all sources", "perfect sampler's wrong deletions as % of the model's", C.boot(share, bg["wrong"], bm["wrong"]))
    add("all", "all sources", "perfect sampler's missed insertions as % of the model's", C.boot(share, bg["missed"], bm["missed"]))
    # 1. other cluster gaps
    for gap in (1, 2, 3, 4):
        km, kg = C.kind_sums("m100", m, m_first, gap), C.kind_sums("g", g, g_first, gap)
        for name in KINDS:
            xm, xg = (km[f"x_{name}"] + km[f"miss_{name}"])[:f], (kg[f"x_{name}"] + kg[f"miss_{name}"])[:f]
            add(f"gap {gap}", "all sources", f"model, errors per 1000 in kind {name}", C.ratio(xm, L, 1000))
            add(f"gap {gap}", "all sources", f"perfect sampler, errors per 1000 in kind {name}", C.ratio(xg, L, 1000))
            add(f"gap {gap}", "all sources", f"surplus of the model in kind {name} as % of its whole surplus",
                C.boot(lambda a, b_, c, d: 100 * (a - b_) / (c - d), xm, xg, em, eg))
        ru_m = sum((km[f"x_{k}"] + km[f"miss_{k}"])[:f] for k in "RU")
        ru_g = sum((kg[f"x_{k}"] + kg[f"miss_{k}"])[:f] for k in "RU")
        add(f"gap {gap}", "all sources", "surplus of the model in kinds R + U as % of its whole surplus",
            C.boot(lambda a, b_, c, d: 100 * (a - b_) / (c - d), ru_m, ru_g, em, eg))
        add(f"gap {gap}", "all sources", "surplus of the model in hole runs (2+ more x than _) as % of its whole surplus",
            C.boot(lambda a, b_, c, d: 100 * (a - b_) / (c - d), km["err_in_hole_runs"][:f], kg["err_in_hole_runs"][:f], em, eg))
    # 2. definitions that need no clusters: label-only against string errors, runs of adjacent originals
    lab_m, lab_g = em - bm["s_wrong"] - bm["s_missed"], eg - bg["s_wrong"] - bg["s_missed"]
    str_m, str_g = bm["s_wrong"] + bm["s_missed"], bg["s_wrong"] + bg["s_missed"]
    run_m, run_g = nb_runs(C.ins[:n], m[:n], C.offsets[: f + 1]), nb_runs(C.ins[:n], g[:n], C.offsets[: f + 1])
    r2m, r2g = C.ps(run_m >= 2, f), C.ps(run_g >= 2, f)
    r1m, r1g = C.ps(run_m == 1, f), C.ps(run_g == 1, f)
    km, kg = C.kind_sums("m100", m, m_first, 2), C.kind_sums("g", g, g_first, 2)
    groups = {"all sources": np.ones(f, bool)}
    noise = np.clip(np.digitize(C.rho[:f], NOISE_EDGES) - 1, 0, len(NOISE_TEXT) - 1)
    for k, text in enumerate(NOISE_TEXT):
        groups[text] = noise == k
    cut = np.quantile(L, [1 / 3, 2 / 3])
    for k, text in enumerate(("short sources", "medium sources", "long sources")):
        groups[f"{text} ({'up to' if k == 0 else 'from'} {int(cut[0] if k == 0 else cut[k - 1])} letters)"] = np.digitize(L, cut) == k
    # the length of a source and its noise level are tied (more insertions make it longer): length within noise level
    for k, text in enumerate(NOISE_TEXT):
        med = np.median(L[noise == k])
        groups[f"{text}, shorter half"] = (noise == k) & (L <= med)
        groups[f"{text}, longer half"] = (noise == k) & (L > med)
    for group, ids in groups.items():
        test = "no clusters" if group == "all sources" else "by noise / length"
        ns = int(ids.sum())
        add(test, group, "model, errors per 1000", C.ratio(em, L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler, errors per 1000", C.ratio(eg, L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler's label errors as % of the model's", C.boot(share, eg, em, ids=ids), ns)
        add(test, group, "perfect sampler's STRING errors as % of the model's", C.boot(share, str_g, str_m, ids=ids), ns)
        add(test, group, "model, label-only errors per 1000 (label errors that are no string errors)", C.ratio(lab_m, L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler, label-only errors per 1000", C.ratio(lab_g, L, 1000, ids=ids), ns)
        add(test, group, "model, string errors per 1000", C.ratio(str_m, L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler, string errors per 1000", C.ratio(str_g, L, 1000, ids=ids), ns)
        add(test, group, "surplus of the model in string errors as % of its surplus in label errors",
            C.boot(lambda a, b_, c, d: 100 * (a - b_) / (c - d), str_m, str_g, em, eg, ids=ids), ns)
        add(test, group, "model, wrong deletions in runs of 2+ adjacent originals per 1000", C.ratio(r2m, L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler, wrong deletions in runs of 2+ adjacent originals per 1000", C.ratio(r2g, L, 1000, ids=ids), ns)
        add(test, group, "model, single wrong deletions (run of 1) per 1000", C.ratio(r1m, L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler, single wrong deletions (run of 1) per 1000", C.ratio(r1g, L, 1000, ids=ids), ns)
        add(test, group, "surplus of the model's wrong deletions that lies in runs of 2+ (%)",
            C.boot(lambda a, b_, c, d: 100 * (a - b_) / (c - d), r2m, r2g, bm["wrong"], bg["wrong"], ids=ids), ns)
        ru_m = sum((km[f"x_{k}"] + km[f"miss_{k}"])[:f] for k in "RU")
        ru_g = sum((kg[f"x_{k}"] + kg[f"miss_{k}"])[:f] for k in "RU")
        add(test, group, "surplus of the model in kinds R + U as % of its whole surplus (gap 2)",
            C.boot(lambda a, b_, c, d: 100 * (a - b_) / (c - d), ru_m, ru_g, em, eg, ids=ids), ns)
        add(test, group, "model, missed insertions per 1000", C.ratio(bm["missed"], L, 1000, ids=ids), ns)
        add(test, group, "perfect sampler, missed insertions per 1000", C.ratio(bg["missed"], L, 1000, ids=ids), ns)
    table = pd.DataFrame(rows)
    table.to_csv(C.out / "attack_a_share_and_kinds.csv", index=False)
    pick = lambda test, group, quantity: [float(x) for x in table[(table.test == test) & (table.group == group) & (table.quantity == quantity)][["value", "lo", "hi"]].iloc[0]]  # noqa: E731
    out = {"perfect sampler's label errors as % of the model's": pick("all", "all sources", "perfect sampler's label errors as % of the model's"),
           "perfect sampler's STRING errors as % of the model's": pick("all", "all sources", "perfect sampler's STRING errors as % of the model's"),
           "perfect sampler's wrong deletions as % of the model's": pick("all", "all sources", "perfect sampler's wrong deletions as % of the model's"),
           "perfect sampler's missed insertions as % of the model's": pick("all", "all sources", "perfect sampler's missed insertions as % of the model's"),
           "surplus in R + U as % of the whole surplus, by gap": {gap: pick(f"gap {gap}", "all sources", "surplus of the model in kinds R + U as % of its whole surplus") for gap in (1, 2, 3, 4)},
           "surplus in hole runs as % of the whole surplus, by gap": {gap: pick(f"gap {gap}", "all sources", "surplus of the model in hole runs (2+ more x than _) as % of its whole surplus") for gap in (1, 2, 3, 4)},
           "errors per 1000 in Z, V, R, U by gap, model": {gap: [pick(f"gap {gap}", "all sources", f"model, errors per 1000 in kind {k}")[0] for k in KINDS] for gap in (1, 2, 3, 4)},
           "errors per 1000 in Z, V, R, U by gap, perfect sampler": {gap: [pick(f"gap {gap}", "all sources", f"perfect sampler, errors per 1000 in kind {k}")[0] for k in KINDS] for gap in (1, 2, 3, 4)},
           "no clusters": {q: pick("no clusters", "all sources", q) for q in table[table.test == "no clusters"].quantity},
           "by noise / length": {group: {q: pick("by noise / length", group, q) for q in (
               "model, errors per 1000", "perfect sampler, errors per 1000", "perfect sampler's label errors as % of the model's",
               "perfect sampler's STRING errors as % of the model's", "surplus of the model in kinds R + U as % of its whole surplus (gap 2)",
               "surplus of the model's wrong deletions that lies in runs of 2+ (%)", "model, missed insertions per 1000",
               "perfect sampler, missed insertions per 1000")} for group in groups if group != "all sources"}}
    C.summary["attack_a"] = out
    C.tables["attack_a"] = (table, groups)


# ---- C12: the cached intervention runs -------------------------------------------------------------------------------


def neighbourhood_grammatical(C: Context, seq: np.ndarray, present: np.ndarray, lo: int, hi: int) -> bool:
    """No forbidden window of the letters that are present touches the positions lo .. hi."""
    kept = np.flatnonzero(present)
    s = seq[kept]
    bad = np.flatnonzero(~C.eligible[s[:-2], s[1:-1], s[2:]]) + 2  # index (in kept) of the last letter of a forbidden window
    return not bool(((kept[bad - 2] <= hi) & (kept[bad] >= lo)).any())


def claim12(C: Context, reach: int = 10):
    print("C12 interventions (cached runs, family t0 = 0)")
    scratch = C.errors / "results" / "interventions" / "scratch"
    if not (scratch / "sites.csv").exists():
        C.missing.append(str(scratch / "sites.csv"))
        C.claim("C12", "interventions: no cached runs", None, None)
        return
    sites = pd.read_csv(scratch / "sites.csv")
    sites = sites[(sites.fam == "f00") & (sites.exp == "run")].set_index("site")
    pool = json.loads((scratch / "pool.json").read_text())["f00"] if (scratch / "pool.json").exists() else {}
    runs = {"model": {}, "exact": {}}
    control, q_control, null_diff = {}, {}, []
    for path in sorted(scratch.glob("calls-f00-*.npz")):
        z = load_npz(path)
        bounds = np.concatenate([[0], np.cumsum(z["length"])])
        if int(z["null_diff"]) >= 0:
            null_diff.append(int(z["null_diff"]))
        for j, (i, sid) in enumerate(zip(z["source"], z["site"])):
            seg = slice(bounds[j], bounds[j + 1])
            if sid < 0:
                control[int(i)], q_control[int(i)] = z["del_step"][seg], z["q0"][seg].astype(np.float32)
            elif int(sid) in sites.index:
                runs["model"][int(sid)] = (int(i), z["del_step"][seg], z["q0"][seg].astype(np.float32))
    for path in sorted(scratch.glob("exact-f00-*.npz")):
        z = load_npz(path)
        off = np.concatenate([[0], np.cumsum([C.offsets[i + 1] - C.offsets[i] for i in z["sources"]])])
        for c, r in zip(*np.nonzero(z["table"] >= 0)):
            sid = int(z["table"][c, r])
            if sid in sites.index:
                runs["exact"][sid] = (int(z["sources"][r]), z["del_step"][1 + c][off[r]: off[r + 1]], z["del_step"][0][off[r]: off[r + 1]])
    rows = []
    for engine, table in runs.items():
        for sid, item in table.items():
            i, d_int = item[0], item[1]
            d_ctl = control.get(i) if engine == "model" else item[2]
            if d_ctl is None:
                continue
            row = sites.loc[sid]
            pos = int(row.pos)
            seq, ins = C.sources.seq(i), C.sources.mask(i)
            nb = np.concatenate([np.arange(pos - reach, pos), np.arange(pos + 1, pos + reach + 1)])
            orig = ~ins[nb]
            gone_c, gone_i = d_ctl[nb] >= 0, d_int[nb] >= 0
            new, rescued = int((orig & gone_i & ~gone_c).sum()), int((orig & gone_c & ~gone_i).sum())
            present = d_int < 0
            present[pos] = False
            away = np.abs(np.arange(len(seq)) - pos)
            differ = (d_ctl >= 0) != (d_int >= 0)
            differ[pos] = False
            whole = int(((d_int >= 0) & ~ins).sum() - ((d_ctl >= 0) & ~ins).sum())  # all originals; the forced letter is in neither
            rows.append({"engine": engine, "site": sid, "source": i, "b": int(row.b), "stratum": row.stratum, "paired": bool(row.exact),
                         "b_target": int(C.target_break(np.array([C.offsets[i] + pos]))[0]),
                         "extra_wrong": new - rescued, "new": new, "rescued": rescued,
                         "gram": neighbourhood_grammatical(C, seq, present, nb[0], nb[-1]),
                         "forced_letter_marked_deleted": bool(d_int[pos] >= 0), "control_deletes_the_letter": bool(d_ctl[pos] >= 0),
                         "extra_wrong_whole_source": whole, "differ_beyond_10": int(differ[away > reach].sum()),
                         "differ_beyond_30": int(differ[away > 30].sum()), "letters_beyond_30": int((away > 30).sum())})
    if not rows:
        C.claim("C12", "interventions: no cached runs of the family t0 = 0", None, None)
        return
    r = pd.DataFrame(rows)
    S = C.S

    def mean(frame, column):
        if not len(frame):
            return (np.nan, np.nan, np.nan)
        src = frame.source.to_numpy()
        return C.ratio(np.bincount(src, weights=frame[column].to_numpy(dtype=float), minlength=S), np.bincount(src, minlength=S))

    out = {}
    model_clean = r[(r.engine == "model") & (r.stratum == "clean")]
    for b in range(3):
        f = model_clean[model_clean.b == b]
        e = mean(f, "extra_wrong")
        C.claim("C12", f"model, break {b}: further wrong deletions after a forced removal at t0 = 0", f"C12 model b{b}", e[0], e[1:],
                n=f"{len(f)} clean sites")
    exact = r[r.engine == "exact"]
    for b in (1, 2):
        f = exact[exact.b == b]
        e = mean(f, "extra_wrong")
        C.claim("C12", f"exact rates, break {b}: further wrong deletions after a forced removal", f"C12 exact b{b}", e[0], e[1:], n=f"{len(f)} sites")
    paired_model = r[(r.engine == "model") & r.paired & (r.stratum == "clean")]
    e = mean(exact[exact.b > 0], "gram")
    C.claim("C12", "exact rates, break 1 or 2: % grammatical at the end", "C12 gram exact", 100 * e[0], (100 * e[1], 100 * e[2]),
            tol=0.6, n=f"{int((exact.b > 0).sum())} sites")
    e = mean(paired_model[paired_model.b > 0], "gram")
    C.claim("C12", "model, break 1 or 2: % grammatical at the end (same sites)", "C12 gram model", 100 * e[0], (100 * e[1], 100 * e[2]),
            tol=3.5, n=f"{int((paired_model.b > 0).sum())} sites")
    for engine, frame in (("model, clean sites", model_clean), ("model, sites sampled with both engines", paired_model), ("exact rates", exact),
                          ("model, dirty sites (a leftover insertion within 3 letters)", r[(r.engine == "model") & (r.stratum == "dirty")])):
        out[engine] = {f"break {b}": {"n": int((frame.b == b).sum()), "further wrong deletions": triple(mean(frame[frame.b == b], "extra_wrong")),
                                      "new": float(frame[frame.b == b].new.mean()), "rescued": float(frame[frame.b == b].rescued.mean()),
                                      "grammatical at the end": triple(mean(frame[frame.b == b], "gram")),
                                      "further wrong deletions counted over the whole source": triple(mean(frame[frame.b == b], "extra_wrong_whole_source"))}
                       for b in range(3) if (frame.b == b).any()}
    dirty = r[(r.engine == "model") & (r.stratum == "dirty")]
    out["model, dirty sites by the break size among the originals alone (insertions ignored)"] = {
        f"break {b} in the sequence, break {t} among the originals": {"n": int(((dirty.b == b) & (dirty.b_target == t)).sum()),
                                                                     "further wrong deletions": triple(mean(dirty[(dirty.b == b) & (dirty.b_target == t)], "extra_wrong"))}
        for b in range(3) for t in range(3) if ((dirty.b == b) & (dirty.b_target == t)).sum() >= 5}
    out["model, all sites by the break size among the originals alone"] = {
        f"break {t} among the originals": {"n": int((model_all.b_target == t).sum()), "further wrong deletions": triple(mean(model_all[model_all.b_target == t], "extra_wrong"))}
        for model_all in [r[r.engine == "model"]] for t in range(3)}
    # break size 0 on all sites of the source, clean and dirty with their shares among all originals
    if pool and (r[(r.engine == "model") & (r.stratum == "dirty")].b == 0).any():
        mix = {}
        for b in range(3):
            wc, wd = pool.get(f"b{b}_clean", 0), pool.get(f"b{b}_dirty", 0)
            src_c = model_clean[model_clean.b == b]
            src_d = r[(r.engine == "model") & (r.stratum == "dirty") & (r.b == b)]
            if len(src_c) and len(src_d):
                mix[f"break {b}"] = {"share of clean sites among all sites of this break size": wc / (wc + wd),
                                     "further wrong deletions, clean and dirty sites mixed with their natural shares":
                                         float((wc * src_c.extra_wrong.mean() + wd * src_d.extra_wrong.mean()) / (wc + wd))}
        out["natural mix of clean and dirty sites (model)"] = mix
    # bookkeeping
    model = r[r.engine == "model"]
    q_diff = []
    z, first = C.corpus(C.errors / "m100.npz")
    if z is not None:
        for i, q in q_control.items():
            if i < first:
                q_diff.append(float(np.abs(q - z["q0"][C.sources.span(i)]).max()))
    ctl_wrong = sum(int(((d >= 0) & ~C.sources.mask(i)).sum()) for i, d in control.items())
    ctl_letters = sum(len(d) for d in control.values())
    out["bookkeeping"] = {
        "runs found": {"model": int(len(model)), "exact": int(len(exact)), "control sources (model)": len(control)},
        "sites where the forced letter is marked as deleted by the sampler (must be 0: it is never counted)": int(r.forced_letter_marked_deleted.sum()),
        "sites whose control run deletes the same letter on its own": int(r.control_deletes_the_letter.sum()),
        "null test stored with the runs (differences between two compositions of the same call)": null_diff,
        "largest difference between q0 of a control run and q0 of m100 for the same letters": max(q_diff) if q_diff else None,
        "control runs: wrong deletions per 1000 letters": 1000 * ctl_wrong / max(ctl_letters, 1),
        "share of sites with any difference beyond 10 letters, model / exact": [float((model.differ_beyond_10 > 0).mean()), float((exact.differ_beyond_10 > 0).mean())],
        "share of sites with any difference beyond 30 letters, model / exact": [float((model.differ_beyond_30 > 0).mean()), float((exact.differ_beyond_30 > 0).mean())],
        "differing letters beyond 30 letters per 1000 such letters, model / exact": [float(1000 * model.differ_beyond_30.sum() / model.letters_beyond_30.sum()),
                                                                                    float(1000 * exact.differ_beyond_30.sum() / exact.letters_beyond_30.sum())],
        "the same after a placebo removal (break 0, clean), model": float(1000 * model_clean[model_clean.b == 0].differ_beyond_30.sum()
                                                                          / max(model_clean[model_clean.b == 0].letters_beyond_30.sum(), 1)),
    }
    C.summary["C12"] = out
    C.tables["C12"] = r


# ---- C11 and attack (b): the event study on the logged q -------------------------------------------------------------


@numba.njit(cache=True)
def nb_event_pass(del_step, q, offsets, inserted, letters, eligible, qedges, iso, n_steps):
    """One pass over the sampled paths of all sources. Returns

    * `events`, one row per wrong deletion: letter, step, isolated (no other deletion of the step within `iso`
      letters), first (no earlier wrong deletion between the `iso`-th letters on each side), break size in the
      sequence the model sees, break size among the originals alone, clean (no insertion within 3 letters), the
      letters next to it on the left and right (-1: none);
    * the controls: sum and count of the one-step change of q of original letters with no deletion within `iso`
      letters in that step, by step and bin of q.

    Distances are counted in the sequence as it is at the step (letters that are gone do not count)."""
    n = offsets[-1]
    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    near = np.full(n, -1, dtype=np.int64)
    lonely = np.zeros(n, dtype=np.bool_)
    next_at = np.full(n, -1, dtype=np.int64)
    first_at = np.full(n_steps, -1, dtype=np.int64)
    n_wrong = 0
    for i in range(n):
        if del_step[i] >= 0 and not inserted[i]:
            n_wrong += 1
    events = np.full((n_wrong, 9), -1, dtype=np.int64)
    nb = len(qedges) - 1
    c_sum = np.zeros((n_steps, nb))
    c_cnt = np.zeros((n_steps, nb))
    e = 0
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        for k in range(n_steps):
            first_at[k] = -1
        for i in range(b - 1, a - 1, -1):
            prev[i] = i - 1 if i > a else -1
            nxt[i] = i + 1 if i < b - 1 else -1
            k = del_step[i]
            if k >= 0:
                next_at[i] = first_at[k]
                first_at[k] = i
        head = a
        for k in range(n_steps):
            d = first_at[k]
            while d >= 0:  # mark the surroundings of every deletion of this step
                near[d] = k
                alone = True
                for side in range(2):
                    j = prev[d] if side == 0 else nxt[d]
                    c = 0
                    while j >= 0 and c < iso:
                        near[j] = k
                        if del_step[j] == k:
                            alone = False
                        j = prev[j] if side == 0 else nxt[j]
                        c += 1
                lonely[d] = alone
                d = next_at[d]
            if k + 1 < n_steps:
                j = head
                while j >= 0:  # controls
                    if near[j] != k and not inserted[j]:
                        x = q[j, k]
                        bin_ = np.searchsorted(qedges, x, side="right") - 1
                        if bin_ >= nb:
                            bin_ = nb - 1
                        c_sum[k, bin_] += q[j, k + 1] - x
                        c_cnt[k, bin_] += 1
                    j = nxt[j]
            d = first_at[k]
            while d >= 0:  # the wrong deletions of this step
                if not inserted[d]:
                    l1, r1 = prev[d], nxt[d]
                    l2 = prev[l1] if l1 >= 0 else -1
                    r2 = nxt[r1] if r1 >= 0 else -1
                    if l2 >= 0 and r2 >= 0:
                        events[e, 4] = (0 if eligible[letters[l2], letters[l1], letters[r1]] else 1) + (0 if eligible[letters[l1], letters[r1], letters[r2]] else 1)
                    # the same among the originals alone
                    t = np.full(4, -1, dtype=np.int64)
                    j, c = prev[d], 1
                    while j >= 0 and c >= 0:
                        if not inserted[j]:
                            t[c] = j
                            c -= 1
                        j = prev[j]
                    j, c = nxt[d], 2
                    while j >= 0 and c <= 3:
                        if not inserted[j]:
                            t[c] = j
                            c += 1
                        j = nxt[j]
                    if t[0] >= 0 and t[3] >= 0:
                        events[e, 5] = (0 if eligible[letters[t[0]], letters[t[1]], letters[t[2]]] else 1) + (0 if eligible[letters[t[1]], letters[t[2]], letters[t[3]]] else 1)
                    clean = True
                    lo, hi = d, d
                    for side in range(2):
                        j = prev[d] if side == 0 else nxt[d]
                        c = 0
                        while j >= 0 and c < iso:
                            if c < 3 and inserted[j]:
                                clean = False
                            if side == 0:
                                lo = j
                            else:
                                hi = j
                            j = prev[j] if side == 0 else nxt[j]
                            c += 1
                        if j < 0:  # fewer than `iso` letters on this side: up to the end of the source
                            if side == 0:
                                lo = a
                            else:
                                hi = b - 1
                    first = True
                    for i in range(lo, hi + 1):
                        if not inserted[i] and del_step[i] >= 0 and del_step[i] < k:
                            first = False
                            break
                    events[e, 0], events[e, 1], events[e, 2], events[e, 3] = d, k, 1 if lonely[d] else 0, 1 if first else 0
                    events[e, 6], events[e, 7], events[e, 8] = 1 if clean else 0, l1, r1
                    e += 1
                d = next_at[d]
            d = first_at[k]
            while d >= 0:  # the letters leave the sequence
                p, x = prev[d], nxt[d]
                if p >= 0:
                    nxt[p] = x
                else:
                    head = x
                if x >= 0:
                    prev[x] = p
                d = next_at[d]
    return events[:e], c_sum, c_cnt


def load_qlog(C: Context):
    """The logged run: deletion steps, q (letters x steps, float32), sources covered, steps."""
    z, first = C.corpus(C.errors / "qlog.npz")
    if z is None or ("qlog" not in z and "q32" not in z):
        return None
    if "q32" not in z:
        z["q32"] = np.ascontiguousarray(z["qlog"], dtype=np.float32)
        del z["qlog"]
    return z["del_step"].astype(np.int64), z["q32"], first, int(z["steps"])


def claim11(C: Context, iso: int = 12, min_controls: int = 50):
    print("C11 event study, attack (b)")
    data = load_qlog(C)
    if data is None:
        C.claim("C11", "event study: qlog.npz missing", None, None)
        return
    step, q, first, n_steps = data
    n = len(step)
    ins, seq_of = C.ins[:n], C.seq_of[:n]
    events, c_sum, c_cnt = nb_event_pass(step, q, C.offsets[: first + 1], ins, C.letters[:n], C.eligible, QBINS, iso, n_steps)
    with np.errstate(invalid="ignore", divide="ignore"):
        c_mean = np.where(c_cnt >= min_controls, c_sum / c_cnt, np.nan)
    d, k, lonely, is_first, b, bt, clean, l1, r1 = events.T
    out = {"sources": first, "wrong deletions": int(len(events)), "isolated": int(lonely.sum()), "isolated and first": int((lonely & is_first).sum()),
           "isolated first, break 0 / 1 / 2 in the sequence the model sees": [int(((lonely == 1) & (is_first == 1) & (b == v)).sum()) for v in range(3)],
           "share of control cells (step x bin of q) with fewer than 50 controls": float((c_cnt < min_controls).mean())}
    # one record per (isolated wrong deletion, original letter next to it); the last step has no next q
    use = (lonely == 1) & (b >= 0) & (k + 1 < n_steps)
    rec = {}
    for side, nbr in (("left", l1), ("right", r1)):
        ok = use & (nbr >= 0)
        ok[ok] &= ~ins[nbr[ok]]
        rec[side] = (np.flatnonzero(ok), nbr[ok])
    e_idx = np.concatenate([rec["left"][0], rec["right"][0]])
    nbr = np.concatenate([rec["left"][1], rec["right"][1]])
    kk = k[e_idx]
    q_before = q[nbr, kk].astype(np.float64)
    dq = q[nbr, kk + 1].astype(np.float64) - q_before
    bins = np.clip(np.searchsorted(QBINS, q_before, side="right") - 1, 0, len(QBINS) - 2)
    adj = dq - c_mean[kk, bins]
    kp = np.maximum(kk - 1, 0)
    q_prev = q[nbr, kp].astype(np.float64)
    pre = (q_before - q_prev) - c_mean[kp, np.clip(np.searchsorted(QBINS, q_prev, side="right") - 1, 0, len(QBINS) - 2)]
    pre[kk == 0] = np.nan
    src = seq_of[d[e_idx]]
    valid = np.isfinite(adj)
    out["share of neighbours without matched controls"] = float(1 - valid.mean())
    R = pd.DataFrame({"source": src, "step": kk, "b": b[e_idx], "b_target": bt[e_idx], "first": is_first[e_idx] == 1, "clean": clean[e_idx] == 1,
                      "q_deleted": q[d[e_idx], kk].astype(np.float64), "q_neighbour": q_before, "dq": dq, "adj": adj, "pre": pre,
                      "rho": C.rho[src], "length": C.lengths[src]})
    if C.q_gap is not None:
        R["exact_deleted"], R["exact_neighbour"] = C.q_gap[d[e_idx]].astype(np.float64), C.q_gap[nbr].astype(np.float64)
    R = R[valid]

    def mean(frame, column="adj"):
        v = frame[column].to_numpy()
        okv = np.isfinite(v)
        s_ = frame.source.to_numpy()[okv]
        if okv.sum() == 0:
            return (np.nan, np.nan, np.nan)
        return C.ratio(np.bincount(s_, weights=v[okv], minlength=first), np.bincount(s_, minlength=first))

    F = R[R["first"]]
    for size, key in ((2, "C11 b2"), (1, "C11 b1"), (0, "C11 b0")):
        f = F[F.b == size]
        est = mean(f)
        C.claim("C11", f"break {size}: rise of q of the adjacent original after an isolated first wrong deletion", key,
                est[0], est[1:], tol=0.02 if size == 0 else None, n=f"{len(f)} neighbours, {first} sources")
    # --- attack (b): strata
    strata = {"all isolated first wrong deletions": lambda f: [("all", np.ones(len(f), bool))],
              "one step BEFORE the event (placebo in time)": None,
              "time of the event": lambda f: [("steps 0-9", f.step < 10), ("steps 10-19", (f.step >= 10) & (f.step < 20)),
                                              ("steps 20-39", (f.step >= 20) & (f.step < 40)), (f"steps 40-{n_steps - 1}", f.step >= 40)],
              "noise level of the source": lambda f: [(t, np.clip(np.digitize(f.rho, NOISE_EDGES) - 1, 0, 3) == j) for j, t in enumerate(NOISE_TEXT)],
              "q of the neighbour before the event": lambda f: [("below 0.01", f.q_neighbour < 0.01), ("0.01 to 0.1", (f.q_neighbour >= 0.01) & (f.q_neighbour < 0.1)),
                                                              ("0.1 or more", f.q_neighbour >= 0.1)],
              "q of the deleted original itself, just before": lambda f: [("below 0.03 (an accident)", f.q_deleted < 0.03), ("0.03 to 0.3", (f.q_deleted >= 0.03) & (f.q_deleted < 0.3)),
                                                                        ("0.3 or more (the model took it for an insertion)", f.q_deleted >= 0.3)],
              "insertions within 3 letters at the time": lambda f: [("none (clean)", f.clean), ("at least one (dirty)", ~f.clean)],
              "break size among the originals alone (insertions ignored)": lambda f: [(f"target break {v}", f.b_target == v) for v in range(3)],
              "not only first: also later wrong deletions of a neighbourhood": None}
    if "exact_deleted" in R:
        strata["exact posterior of the deleted original (in the source)"] = lambda f: [("clear, below 0.01", f.exact_deleted < 0.01), ("0.01 to 0.1", (f.exact_deleted >= 0.01) & (f.exact_deleted < 0.1)),
                                                                                    ("ambiguous, 0.1 or more", f.exact_deleted >= 0.1)]
        strata["exact posterior of the neighbour (in the source)"] = lambda f: [("clear, below 0.01", f.exact_neighbour < 0.01), ("doubtful, 0.01 or more", f.exact_neighbour >= 0.01)]
    rows = []
    for name, levels in strata.items():
        if name.startswith("one step BEFORE"):
            parts, column, frame = [("all", np.ones(len(F), bool))], "pre", F
        elif name.startswith("not only first"):
            parts, column, frame = [("later wrong deletions only", ~R["first"].to_numpy())], "adj", R
        else:
            parts, column, frame = levels(F), "adj", F
        for level, mask in parts:
            mask = np.asarray(mask)
            for size in range(3):
                f = frame[mask & (frame.b == size).to_numpy()]
                est = mean(f, column)
                rows.append({"stratum": name, "level": level, "break_size": size, "n_neighbours": len(f), "change_of_q": est[0], "lo": est[1], "hi": est[2]})
    table = pd.DataFrame(rows)
    # who are the events? the break sizes are not the same kind of event
    who = []
    for size in range(3):
        f = F[F.b == size]
        who.append({"break_size": size, "n_neighbours": len(f), "median step": float(f.step.median()), "mean q of the deleted original": float(f.q_deleted.mean()),
                    "share deleted original with q >= 0.3": float((f.q_deleted >= 0.3).mean()), "mean q of the neighbour before": float(f.q_neighbour.mean()),
                    "share clean": float(f.clean.mean()), "mean noise level": float(f.rho.mean()),
                    "share target break 0": float((f.b_target == 0).mean()),
                    **({"share deleted original clear in the source (q_gap < 0.01)": float((f.exact_deleted < 0.01).mean())} if "exact_deleted" in f else {})})
    out["who the events are, by break size"] = who
    # contrast within cells of time x q of the deleted letter x q of the neighbour x clean, weighted as the broken events are
    cell = (np.digitize(F.step, [10, 20, 40]) * 18 + np.digitize(F.q_deleted, [0.03, 0.3]) * 6 + np.digitize(F.q_neighbour, [0.01, 0.1]) * 2 + F.clean.to_numpy().astype(int))
    n_cells = 72
    sums = np.zeros((3, first, n_cells))
    counts = np.zeros((3, first, n_cells))
    for size in range(3):
        m_ = (F.b == size).to_numpy()
        np.add.at(sums[size], (F.source.to_numpy()[m_], cell[m_]), F.adj.to_numpy()[m_])
        np.add.at(counts[size], (F.source.to_numpy()[m_], cell[m_]), 1.0)

    def standardised(s0, c0, s1, c1, s2, c2):
        res = []
        for s_b, c_b in ((s1, c1), (s2, c2)):
            okc = (c0 >= 5) & (c_b >= 5)
            w = np.where(okc, c_b, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                diff = np.where(okc, s_b / c_b - s0 / c0, 0.0)
                res.append((w * diff).sum(-1) / w.sum(-1))
                res.append(w.sum(-1) / c_b.sum(-1))
        return np.stack(res, -1)

    est = C.boot(standardised, sums[0], counts[0], sums[1], counts[1], sums[2], counts[2])
    raw = [mean(F[F.b == v]) for v in range(3)]
    out["break 1 minus break 0, raw"] = float(raw[1][0] - raw[0][0])
    out["break 2 minus break 0, raw"] = float(raw[2][0] - raw[0][0])
    out["break 1 minus break 0 within cells (step, q of the deleted letter, q of the neighbour, clean)"] = [float(est[0][0]), float(est[1][0]), float(est[2][0])]
    out["... share of the break-1 events in cells that also hold break-0 events"] = float(est[0][1])
    out["break 2 minus break 0 within cells"] = [float(est[0][2]), float(est[1][2]), float(est[2][2])]
    out["... share of the break-2 events in cells that also hold break-0 events"] = float(est[0][3])
    # the randomised version: one forward pass with and without a forced removal (cached by the intervention study)
    scratch = C.errors / "results" / "interventions" / "scratch"
    if (scratch / "sites.csv").exists():
        sites = pd.read_csv(scratch / "sites.csv")
        forced = {}
        for fam, engine in (("f00", "model"), ("f00", "exact"), ("tgt09", "model"), ("tgt09", "exact")):
            z = load_npz(scratch / f"onepass-{engine}-{fam}.npz")
            if z is None:
                continue
            s_ = sites.set_index("site").loc[z["site"]]
            change = z["qi"] - z["qc"]
            centre = change.shape[1] // 2
            if fam == "f00":  # the state is the source: the letters beside the site may be insertions
                gpos = C.offsets[s_.source.to_numpy()] + s_.pos.to_numpy()
                keep_l, keep_r = ~C.ins[gpos - 1], ~C.ins[gpos + 1]
            else:
                keep_l = keep_r = np.ones(len(s_), bool)
            where = "source at t = 0" if fam == "f00" else "clean target at t = 0.9"
            cells = [(f"{stratum} sites, break {size}", ((s_.b == size) & (s_.stratum == stratum)).to_numpy())
                     for stratum in sorted(s_.stratum.unique()) for size in range(3)]
            if fam == "f00":  # the break size the originals alone would show, for the sites with an insertion nearby
                bt_ = C.target_break(gpos)
                cells += [(f"dirty sites, break {size} in the sequence, break {t_} among the originals alone", ((s_.b == size) & (s_.stratum == "dirty")).to_numpy() & (bt_ == t_))
                          for size in range(3) for t_ in range(3)]
            for label, m_ in cells:
                v = np.concatenate([change[m_ & keep_l, centre - 1], change[m_ & keep_r, centre + 1]]).astype(np.float64)
                so = np.concatenate([s_.source.to_numpy()[m_ & keep_l], s_.source.to_numpy()[m_ & keep_r]])
                if len(v) >= 10:
                    est_ = C.ratio(np.bincount(so, weights=v, minlength=C.S), np.bincount(so, minlength=C.S))
                    forced[f"{engine}, {where}, {label}"] = triple(est_) + [int(len(v))]
        out["forced removal of a random original, one forward pass: change of q of the adjacent original"] = forced
    table.to_csv(C.out / "attack_b_event_study.csv", index=False)
    out["strata"] = {f"{r_.stratum} | {r_.level} | break {r_.break_size}": [r_.change_of_q, r_.lo, r_.hi, r_.n_neighbours] for r_ in table.itertuples()}
    C.summary["C11_attack_b"] = out
    C.tables["attack_b"] = table


# ---- attack (c): the accounting of first reading and cascade ----------------------------------------------------------


@numba.njit(cache=True)
def nb_belief_pass(del_step, q, offsets, inserted, krem, widths, n_steps, threshold):
    """The belief of the model that a letter will be deleted by the end, m = q x (share of the leftover insertions
    that the rest of the schedule deletes), followed step by step. Its change over one step is booked by what
    happened in that step: 0 the letter itself was deleted, 1 another letter within `width` letters was wrongly
    deleted, 2 (else) one was correctly deleted, 3 no deletion that near. Summed over all steps the changes equal
    (deleted or not) minus the first belief. Returns the sums per source, kind of letter (0 original with a first
    q below `threshold`, 1 inserted, 2 original with a first q of at least `threshold`), width and category, and
    for every original the largest one-step rise of q and its category (first width)."""
    n = offsets[-1]
    S = len(offsets) - 1
    nw = len(widths)
    wmax = widths.max()
    prev = np.full(n, -1, dtype=np.int64)
    nxt = np.full(n, -1, dtype=np.int64)
    near_w = np.full((nw, n), -1, dtype=np.int64)
    near_c = np.full((nw, n), -1, dtype=np.int64)
    next_at = np.full(n, -1, dtype=np.int64)
    first_at = np.full(n_steps, -1, dtype=np.int64)
    sums = np.zeros((S, 3, nw, 4))
    best = np.zeros(n)
    ctx = np.zeros(n, dtype=np.int8)
    for s in range(S):
        a, b = offsets[s], offsets[s + 1]
        for k in range(n_steps):
            first_at[k] = -1
        for i in range(b - 1, a - 1, -1):
            prev[i] = i - 1 if i > a else -1
            nxt[i] = i + 1 if i < b - 1 else -1
            k = del_step[i]
            if k >= 0:
                next_at[i] = first_at[k]
                first_at[k] = i
        head = a
        for k in range(n_steps):
            d = first_at[k]
            while d >= 0:
                for side in range(2):
                    j = prev[d] if side == 0 else nxt[d]
                    c = 1
                    while j >= 0 and c <= wmax:
                        for w in range(nw):
                            if c <= widths[w]:
                                if inserted[d]:
                                    near_c[w, j] = k
                                else:
                                    near_w[w, j] = k
                        j = prev[j] if side == 0 else nxt[j]
                        c += 1
                d = next_at[d]
            j = head
            while j >= 0:
                kind = 1 if inserted[j] else (2 if q[j, 0] >= threshold else 0)
                m = q[j, k] * krem[k]
                if del_step[j] == k:
                    for w in range(nw):
                        sums[s, kind, w, 0] += 1.0 - m
                else:
                    change = (q[j, k + 1] * krem[k + 1] if k + 1 < n_steps else 0.0) - m
                    for w in range(nw):
                        cat = 1 if near_w[w, j] == k else (2 if near_c[w, j] == k else 3)
                        sums[s, kind, w, cat] += change
                    if k + 1 < n_steps and kind != 1:
                        rise = q[j, k + 1] - q[j, k]
                        if rise > best[j]:
                            best[j] = rise
                            ctx[j] = 1 if near_w[0, j] == k else (2 if near_c[0, j] == k else 3)
                j = nxt[j]
            d = first_at[k]
            while d >= 0:
                p, x = prev[d], nxt[d]
                if p >= 0:
                    nxt[p] = x
                else:
                    head = x
                if x >= 0:
                    prev[x] = p
                d = next_at[d]
    return sums, best, ctx


def attack_c(C: Context, threshold: float = 0.1):
    print("attack (c): first reading against cascade")
    out = {}
    z, first = C.corpus(C.errors / "m100.npz")
    if z is None or C.kappa_t is None:
        C.summary["attack_c"] = "m100.npz or the schedule of the run missing"
        return
    d, q0 = z["del_step"] >= 0, z["q0"].astype(np.float64)
    n = len(d)
    ins, orig = C.ins[:n], ~C.ins[:n]
    L = C.lengths[:first]
    k1 = float(C.kappa_t[-1])
    per = lambda v: triple(C.ratio(C.ps(v, first), L, 1000))  # noqa: E731
    # 1. the account in label errors per 1000 letters
    account = {"model, first reading (kappa(1) q0)": {"wrong deletions": per(np.where(orig, k1 * q0, 0)), "missed insertions": per(np.where(ins, 1 - k1 * q0, 0)),
                                                       "label errors": per(np.where(orig, k1 * q0, 1 - k1 * q0))},
               "model, observed after sampling": {"wrong deletions": per(orig & d), "missed insertions": per(ins & ~d), "label errors": per(d ^ ins)}}
    if C.q_gap is not None:
        qg = C.q_gap[:n].astype(np.float64)
        account["perfect sampler (sum of q_gap)"] = {"wrong deletions": per(np.where(orig, qg, 0)), "missed insertions": per(np.where(ins, 1 - qg, 0)),
                                                     "label errors": per(np.where(orig, qg, 1 - qg))}
    out["account per 1000 letters"] = account
    shift = C.boot(lambda w, a, L_: 1000 * (w - a) / L_, C.ps(orig & d, first), C.ps(np.where(orig, k1 * q0, 0), first), L)
    out["sampling adds wrong deletions to the first reading, per 1000"] = triple(shift)
    out["sampling removes missed insertions from the first reading, per 1000"] = triple(
        C.boot(lambda w, a, L_: 1000 * (a - w) / L_, C.ps(ins & ~d, first), C.ps(np.where(ins, 1 - k1 * q0, 0), first), L))
    out["label errors, observed minus first reading, per 1000"] = triple(
        C.boot(lambda w, a, L_: 1000 * (w - a) / L_, C.ps(d ^ ins, first), C.ps(np.where(orig, k1 * q0, 1 - k1 * q0), first), L))
    if C.q_gap is not None:
        surplus = lambda wm, am, g_: np.stack([100 * (am - g_) / (wm - g_), 100 * (wm - am) / (wm - g_)], -1)  # noqa: E731
        est = C.boot(surplus, C.ps(orig & d, first), C.ps(np.where(orig, k1 * q0, 0), first), C.ps(np.where(orig, qg, 0), first))
        out["surplus of wrong deletions over the perfect sampler: % already in the first reading, % added by sampling"] = [
            [float(est[0][i]), float(est[1][i]), float(est[2][i])] for i in range(2)]
    # by noise level
    noise = np.clip(np.digitize(C.rho[:first], NOISE_EDGES) - 1, 0, 3)
    out["sampling adds wrong deletions / removes missed insertions per 1000, by noise level"] = {
        text: [triple(C.boot(lambda w, a, L_: 1000 * (w - a) / L_, C.ps(orig & d, first), C.ps(np.where(orig, k1 * q0, 0), first), L, ids=noise == j)),
               triple(C.boot(lambda w, a, L_: 1000 * (a - w) / L_, C.ps(ins & ~d, first), C.ps(np.where(ins, 1 - k1 * q0, 0), first), L, ids=noise == j))]
        for j, text in enumerate(NOISE_TEXT)}
    # 2. one draw from the first reading alone: every letter deleted independently with probability kappa(1) q0
    rng = np.random.default_rng([C.args.seed, 3])
    one = rng.random(n) < k1 * q0
    bo, bm = basic(C, one, first), basic(C, d, first)
    runs_o, runs_m = nb_runs(ins, one, C.offsets[: first + 1]), nb_runs(ins, d, C.offsets[: first + 1])
    ko, km = C.kind_sums("oneshot", one, first), C.kind_sums("m100", d, first)
    table = {}
    for name, bb, runs, kk in (("one independent draw from the first reading", bo, runs_o, ko), ("the sampler (m100)", bm, runs_m, km)):
        table[name] = {"label errors per 1000": triple(C.ratio(bb["wrong"] + bb["missed"], L, 1000)),
                       "wrong deletions per 1000": triple(C.ratio(bb["wrong"], L, 1000)),
                       "string errors per 1000": triple(C.ratio(bb["s_wrong"] + bb["s_missed"], L, 1000)),
                       "wrong deletions in runs of 2+ adjacent originals per 1000": triple(C.ratio(C.ps(runs >= 2, first), L, 1000)),
                       "wrong deletions in runs of 4+ per 1000": triple(C.ratio(C.ps(runs >= 4, first), L, 1000)),
                       "hole runs (2+ more x than _) per 1000": triple(C.ratio(kk["hole_runs"], L, 1000)),
                       "errors per 1000 in kind U (a forbidden window is left)": triple(C.ratio(kk["x_U"] + kk["miss_U"], L, 1000)),
                       "string F1": triple(C.boot(lambda dd, sw, i: 2 * (dd - sw) / (dd + i), bb["deleted"], bb["s_wrong"], bb["inserted"]))}
    out["what the first reading alone would give"] = table

    # groups of wrong deletions that lie less than 13 originals apart: how many groups, how many wrong deletions beyond the first of a group
    def neighbourhoods(deleted, f):
        nn = int(C.offsets[f])
        o = ~C.ins[:nn]
        w = np.flatnonzero(deleted[:nn] & o)
        t_index = np.cumsum(o) - 1
        src_w = C.seq_of[w]
        new_group = np.ones(len(w), bool)
        new_group[1:] = (src_w[1:] != src_w[:-1]) | (np.diff(t_index[w]) > 12)
        return np.bincount(src_w[new_group], minlength=f).astype(np.float64), np.bincount(src_w, minlength=f).astype(np.float64)

    groups = {}
    corpora = [("the sampler (m100)", d, first), ("one independent draw from the first reading", one, first)]
    for tag, path in (("perfect sampler (g)", C.study / "g.npz"), ("exact rates, same Euler steps (i2-100)", C.study / "i2-100.npz")):
        dd, ff = C.deleted(path)
        if dd is not None:
            corpora.append((tag, dd, ff))
            if ff < first:
                corpora.append((f"the sampler (m100) on the {ff} sources of {path.stem}", d, ff))
    for tag, dd, ff in corpora:
        g_, w_ = neighbourhoods(dd, ff)
        groups[tag] = {"wrong deletions per 1000": triple(C.ratio(w_, C.lengths[:ff], 1000)),
                       "neighbourhoods with a wrong deletion per 1000": triple(C.ratio(g_, C.lengths[:ff], 1000)),
                       "further wrong deletions per neighbourhood": triple(C.boot(lambda a, b_: a / b_ - 1, w_, g_)),
                       "wrong deletions beyond the first of a neighbourhood per 1000": triple(C.boot(lambda a, b_, l_: 1000 * (a - b_) / l_, w_, g_, C.lengths[:ff]))}
    out["neighbourhoods (wrong deletions less than 13 originals apart)"] = groups
    # 3. the logged run: where does the belief of the model change?
    data = load_qlog(C)
    if data is not None and C.hazard is not None:
        step, q, qfirst, n_steps = data
        nq = len(step)
        hazard = C.hazard if n_steps == len(C.hazard) - 1 else schedule(C, n_steps)[0]
        p_step = np.minimum(hazard[:n_steps] / n_steps, 1.0)
        krem = np.concatenate([1 - np.cumprod((1 - p_step)[::-1])[::-1], [0.0]])
        widths = np.array([2, 10], dtype=np.int64)
        sums, best, ctx = nb_belief_pass(step, q, C.offsets[: qfirst + 1], C.ins[:nq], krem, widths, n_steps, threshold)
        Lq = C.lengths[:qfirst]
        o_q, d_q = ~C.ins[:nq], step >= 0
        wrong = C.ps(o_q & d_q, qfirst)
        firstread = C.ps(np.where(o_q, q[:, 0] * krem[0], 0), qfirst)
        both = sums[:, 0] + sums[:, 2]  # all originals
        check = float(np.abs(both[:, 0].sum(1) - (wrong - firstread)).max())
        names = ("own deletion", "after a wrong deletion that near", "after a correct deletion that near", "no deletion that near")
        book = {"sources": qfirst, "check: largest per-source gap between the booked changes and (deleted - first belief)": check,
                "observed wrong deletions per 1000": triple(C.ratio(wrong, Lq, 1000)), "first belief per 1000": triple(C.ratio(firstread, Lq, 1000))}
        for w_i, w in enumerate(widths):
            for values, label in ((both, "all originals"), (sums[:, 0], f"originals with q0 < {threshold}"), (sums[:, 2], f"originals with q0 >= {threshold}"),
                                  (sums[:, 1], "inserted letters")):
                book[f"{label}, within {w} letters"] = {name: triple(C.ratio(values[:, w_i, c], Lq, 1000)) for c, name in enumerate(names)}
                book[f"{label}, within {w} letters"]["total change (deleted minus first belief)"] = triple(C.ratio(values[:, w_i].sum(1), Lq, 1000))
            book[f"all originals, within {w} letters: belief added right after a wrong deletion, as % of all wrong deletions"] = triple(
                C.ratio(both[:, w_i, 1], wrong, 100))
        out["belief account on the logged run (per 1000 letters)"] = book
        C.tables["belief"] = (both, sums, Lq, wrong, firstread, widths, names)
        # the reported definition: not suspicious at t = 0, suspicious when deleted, the largest rise right after a wrong deletion within 2 letters
        wd = np.flatnonzero(o_q & d_q)
        q_start, q_end = q[wd, 0], q[wd, step[wd]]
        cls_b = (q_start < threshold) & (q_end >= threshold)
        src = C.seq_of[wd]
        share = lambda mask: triple(C.ratio(np.bincount(src[mask], minlength=qfirst), wrong, 100))  # noqa: E731
        out["classes of wrong deletions as reported (threshold 0.1), % of all wrong deletions"] = {
            "a suspicious from the start (q0 >= 0.1)": share(q_start >= threshold),
            "b became suspicious during sampling": share(cls_b),
            "b, largest rise right after a wrong deletion within 2 letters (the reported 'one in eight')": share(cls_b & (ctx[wd] == 1)),
            "b, largest rise right after a correct deletion within 2 letters": share(cls_b & (ctx[wd] == 2)),
            "c never suspicious": share((q_start < threshold) & (q_end < threshold)),
            "a, whose q rose by 0.2 or more in one step right after a wrong deletion within 2 letters": share((q_start >= threshold) & (ctx[wd] == 1) & (best[wd] >= 0.2)),
            "any class: largest rise (0.1 or more) right after a wrong deletion within 2 letters": share((ctx[wd] == 1) & (best[wd] >= 0.1))}
        # how many of class a would a self-consistent sampler delete? their first belief
        a_mask = q_start >= threshold
        out["class a: wrong deletions observed against the sum of their first beliefs (all originals with q0 >= 0.1), per 1000"] = [
            triple(C.ratio(np.bincount(src[a_mask], minlength=qfirst), Lq, 1000)),
            triple(C.ratio(C.ps(np.where(o_q & (q[:, 0] >= threshold), q[:, 0] * krem[0], 0), qfirst), Lq, 1000))]
        out["originals with q0 < 0.1: wrong deletions observed against the sum of their first beliefs, per 1000"] = [
            triple(C.ratio(np.bincount(src[~a_mask], minlength=qfirst), Lq, 1000)),
            triple(C.ratio(C.ps(np.where(o_q & (q[:, 0] < threshold), q[:, 0] * krem[0], 0), qfirst), Lq, 1000))]
    else:
        C.missing.append("qlog.npz or the schedule: no belief account")
    C.summary["attack_c"] = out


# ---- the four conclusions and the figures ----------------------------------------------------------------------------


def conclusions(C: Context):
    """The headline numbers of the four attacks, collected from the parts above, and what they decide."""
    S = C.summary
    out = {}
    a = S.get("attack_a")
    if isinstance(a, dict):
        by = a["by noise / length"]
        label = {g: by[g]["perfect sampler's label errors as % of the model's"] for g in NOISE_TEXT if g in by}
        runs = {g: by[g]["surplus of the model's wrong deletions that lies in runs of 2+ (%)"] for g in NOISE_TEXT if g in by}
        ru = a["surplus in R + U as % of the whole surplus, by gap"]
        out["a"] = {"perfect sampler's label errors as % of the model's": a["perfect sampler's label errors as % of the model's"],
                    "the same by noise level": label,
                    "the same on the strings (LCS)": a["perfect sampler's STRING errors as % of the model's"],
                    "surplus of the model in kinds R + U as % of its whole surplus, gaps 1 to 4": [ru[g][0] for g in (1, 2, 3, 4)],
                    "without clusters: % of the surplus in label errors that are string errors": a["no clusters"]["surplus of the model in string errors as % of its surplus in label errors"],
                    "without clusters: % of the surplus of wrong deletions in runs of 2+ adjacent originals": a["no clusters"]["surplus of the model's wrong deletions that lies in runs of 2+ (%)"],
                    "the same by noise level (runs of 2+)": runs,
                    "checks": {"R + U hold more than 90 % of the surplus at every gap": bool(min(ru[g][0] for g in (1, 2, 3, 4)) > 90),
                               "string errors hold more than 90 % of the surplus (no clusters)": bool(a["no clusters"]["surplus of the model in string errors as % of its surplus in label errors"][0] > 90),
                               "the share of the perfect sampler is the same at every noise level (within 10 points)": bool(max(v[0] for v in label.values()) - min(v[0] for v in label.values()) <= 10)}}
    b = S.get("C11_attack_b")
    if isinstance(b, dict):
        st = b["strata"]
        forced = b.get("forced removal of a random original, one forward pass: change of q of the adjacent original", {})
        cells = [b["break 1 minus break 0 within cells (step, q of the deleted letter, q of the neighbour, clean)"], b["break 2 minus break 0 within cells"]]
        placebo = [st[f"one step BEFORE the event (placebo in time) | all | break {v}"][:3] for v in range(3)]
        dirty0 = forced.get("model, source at t = 0, dirty sites, break 0")
        out["b"] = {"break 1 / 2 minus break 0, raw": [b["break 1 minus break 0, raw"], b["break 2 minus break 0, raw"]],
                    "the same within cells of step, q of the deleted letter, q of the neighbour, insertions nearby": cells,
                    "one step before the event (must be 0), break 0 / 1 / 2": placebo,
                    "when the deleted original was not suspected (its q below 0.03), break 0 / 1 / 2": [
                        st[f"q of the deleted original itself, just before | below 0.03 (an accident) | break {v}"][:3] for v in range(3)],
                    "forced removal, model, clean sites, break 0 / 1 / 2": [forced.get(f"model, source at t = 0, clean sites, break {v}") for v in range(3)],
                    "forced removal, exact posterior, clean sites, break 0 / 1 / 2": [forced.get(f"exact, source at t = 0, clean sites, break {v}") for v in range(3)],
                    "forced removal, model, sites with an insertion nearby, break 0 in the sequence": dirty0,
                    "... of which break 0 / 1 / 2 among the originals alone": [
                        forced.get(f"model, source at t = 0, dirty sites, break 0 in the sequence, break {v} among the originals alone") for v in range(3)],
                    "checks": {"the contrast survives within cells (both intervals above 0)": bool(cells[0][1] > 0 and cells[1][1] > 0),
                               "no change one step before the event (all intervals within +-0.005)": bool(all(abs(v[1]) < 0.005 and abs(v[2]) < 0.005 for v in placebo)),
                               "a removal with visible break 0 next to an insertion leaves the neighbour alone (interval contains 0)":
                                   None if not dirty0 else bool(dirty0[1] <= 0 <= dirty0[2])}}
    c = S.get("attack_c")
    if isinstance(c, dict):
        book = c.get("belief account on the logged run (per 1000 letters)", {})
        first = c["what the first reading alone would give"]
        acc = c["account per 1000 letters"]
        obs, fr = acc["model, observed after sampling"]["wrong deletions"][0], acc["model, first reading (kappa(1) q0)"]["wrong deletions"][0]
        eighth = c.get("classes of wrong deletions as reported (threshold 0.1), % of all wrong deletions", {}).get(
            "b, largest rise right after a wrong deletion within 2 letters (the reported 'one in eight')")
        out["c"] = {"wrong deletions per 1000: perfect sampler, first reading, observed": [
                        acc.get("perfect sampler (sum of q_gap)", {}).get("wrong deletions"), acc["model, first reading (kappa(1) q0)"]["wrong deletions"],
                        acc["model, observed after sampling"]["wrong deletions"]],
                    "surplus over the perfect sampler: % in the first reading, % added by sampling": c.get(
                        "surplus of wrong deletions over the perfect sampler: % already in the first reading, % added by sampling"),
                    "sampling adds wrong deletions per 1000": c["sampling adds wrong deletions to the first reading, per 1000"],
                    "... as % of all wrong deletions": 100 * (obs - fr) / obs,
                    "sampling removes missed insertions per 1000": c["sampling removes missed insertions from the first reading, per 1000"],
                    "label errors, observed minus first reading, per 1000": c["label errors, observed minus first reading, per 1000"],
                    "belief added right after a wrong deletion within 2 / 10 letters, % of all wrong deletions": [
                        book.get(f"all originals, within {w} letters: belief added right after a wrong deletion, as % of all wrong deletions") for w in (2, 10)],
                    "the reported share with its own definition, % of all wrong deletions": eighth,
                    "wrong deletions in runs of 4+ per 1000: first reading alone, sampler": [
                        first["one independent draw from the first reading"]["wrong deletions in runs of 4+ per 1000"],
                        first["the sampler (m100)"]["wrong deletions in runs of 4+ per 1000"]],
                    "neighbourhoods": c.get("neighbourhoods (wrong deletions less than 13 originals apart)"),
                    "checks": {"the first reading holds more than half of the surplus of wrong deletions over the perfect sampler":
                                   None if not c.get("surplus of wrong deletions over the perfect sampler: % already in the first reading, % added by sampling")
                                   else bool(c["surplus of wrong deletions over the perfect sampler: % already in the first reading, % added by sampling"][0][0] > 50),
                               "sampling changes the number of label errors (interval excludes 0)": bool(
                                   not c["label errors, observed minus first reading, per 1000"][1] <= 0 <= c["label errors, observed minus first reading, per 1000"][2]),
                               "what sampling adds is no more than the reported share of wrong deletions that follow a neighbour":
                                   None if eighth is None else bool(100 * (obs - fr) / obs <= eighth[2])}}
    d = S.get("C9_attack_d")
    if isinstance(d, dict):
        cross = d.get("cross-check with the forced removals under exact rates (table exp4 of the intervention study, not recomputed for t0 > 0)", {})
        rep_, mine = d["all leaks pooled: wrong deletions per seed as reported"], d["all leaks pooled: wrong deletions per counted seed"]
        out["d"] = {"wrong deletions per seed as reported (pooled)": rep_, "wrong deletions per counted seed (pooled)": mine,
                    "counted seeds as a share of the reported seeds, by leak": {t: v["counted seeds as a share of the reported seeds"] for t, v in d["by leak"].items()},
                    "per leak: reported, counted": {t: [v["wrong deletions per seed, as reported (seeds by the formula, baseline i2-100)"],
                                                        v["wrong deletions per seed, counted seeds, pooled baseline"]] for t, v in d["by leak"].items()},
                    "predicted from the forced removals under exact rates": cross.get("predicted wrong deletions per leak deletion (seed included), given when the seeds fell"),
                    "checks": {"the cost with counted seeds lies inside the interval of the reported cost": bool(rep_[1] <= mine[0] <= rep_[2])}}
    S["conclusions"] = out


def title(ax, text: str, width: int = 62):
    """The message of a panel as its title, wrapped to the width of the panel."""
    import textwrap

    ax.set_title(textwrap.fill(" ".join(text.split()), width))


def style():
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "figure.constrained_layout.use": True})


def err(ax, x, est, color, label=None, marker="o"):
    est = np.asarray(est, dtype=float).reshape(-1, 3)
    ax.errorbar(x, est[:, 0], yerr=[est[:, 0] - est[:, 1], est[:, 2] - est[:, 0]], fmt=marker, color=color, ms=5, capsize=2.5, lw=1.2, label=label)


def figures(C: Context):
    style()
    S = C.summary
    # 1. reported against recomputed (numbers reported as 0 have no relative deviation and are left out)
    claims = pd.DataFrame(C.claims)
    c = claims[claims.reported.notna() & claims.recomputed.notna()].reset_index(drop=True)
    if len(c):
        zero = (c.reported == 0).to_numpy()
        ok_all = (c.verdict == "reproduced").to_numpy()
        c = c[~zero].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9.4, 0.19 * len(c) + 1.8))
        dev = 100 * (c.recomputed.to_numpy(dtype=float) - c.reported.to_numpy(dtype=float)) / np.abs(c.reported.to_numpy(dtype=float))
        ok = (c.verdict == "reproduced").to_numpy()
        y = np.arange(len(c))[::-1]
        ax.axvline(0, color=COLORS[7], lw=1, label="recomputed = reported")
        ax.plot(np.clip(dev[ok], -60, 130), y[ok], "o", color=COLORS[0], ms=4.5, label="reproduced")
        ax.plot(np.clip(dev[~ok], -60, 130), y[~ok], "s", color=COLORS[1], ms=6, label="not reproduced")
        ax.set_yticks(y, [f"{a} {b[:74]}" for a, b in zip(c.claim, c.quantity)], fontsize=6.3)
        ax.set_xlabel("recomputed minus reported, in % of the reported number")
        ax.set_xlim(-60, 135)
        ax.set_ylim(-1, len(c))
        ax.legend(loc="lower right", fontsize=8)
        fig.suptitle(f"{int(ok_all.sum())} of {len(ok_all)} reported numbers are reproduced by independent code"
                     + (f"; not reproduced: {', '.join(sorted(set(claims[claims.verdict == 'not reproduced'].claim)))}" if (~ok_all).any() else "")
                     + (f"\n({int(zero.sum())} numbers reported as 0 are not drawn; all in claims.csv)" if zero.any() else ""), fontsize=10, fontweight="bold")
        fig.savefig(C.out / "fig1_reported_against_recomputed.png")
        plt.close(fig)
    # 2. attack (a)
    if "attack_a" in C.tables:
        table, groups = C.tables["attack_a"]
        get = lambda test, group, quantity: table[(table.test == test) & (table.group == group) & (table.quantity == quantity)][["value", "lo", "hi"]].to_numpy()[0]  # noqa: E731
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        ax = axes[0]
        x = np.arange(len(NOISE_TEXT))
        err(ax, x - 0.06, [get("by noise / length", g, "perfect sampler's label errors as % of the model's") for g in NOISE_TEXT], COLORS[0], "label errors (wrong deletions + missed insertions)")
        err(ax, x + 0.06, [get("by noise / length", g, "perfect sampler's STRING errors as % of the model's") for g in NOISE_TEXT], COLORS[1], "string errors (output compared with target)", "s")
        overall = get("all", "all sources", "perfect sampler's label errors as % of the model's")[0]
        overall_s = get("all", "all sources", "perfect sampler's STRING errors as % of the model's")[0]
        ax.axhline(overall, color=COLORS[0], ls="--", lw=1)
        ax.text(len(x) - 0.55, overall + 1, f"all sources: {overall:.0f} %", color=COLORS[0], ha="right", fontsize=8)
        ax.axhline(overall_s, color=COLORS[1], ls="--", lw=1)
        ax.text(len(x) - 0.55, overall_s + 1, f"all sources: {overall_s:.0f} %", color=COLORS[1], ha="right", fontsize=8)
        ax.set_xticks(x, [g.replace("noise ", "") for g in NOISE_TEXT])
        ax.set_xlabel("share of inserted letters in the source")
        ax.set_ylabel("errors of the perfect sampler\nin % of the errors of the model")
        ax.set_ylim(0, 100)
        lab = [get("by noise / length", g, "perfect sampler's label errors as % of the model's")[0] for g in NOISE_TEXT]
        title(ax, f"The perfect sampler makes {overall:.0f} % of the model's label errors over all sources:\n{lab[0]:.0f} % at the lowest noise level, "
                     f"{lab[-1]:.0f} % at the highest; on the strings {overall_s:.0f} %")
        ax.legend(fontsize=8, loc="lower left")
        ax = axes[1]
        gaps = np.array([1, 2, 3, 4])
        err(ax, gaps - 0.08, [get(f"gap {g}", "all sources", "surplus of the model in kinds R + U as % of its whole surplus") for g in gaps], COLORS[0], "clusters of kind R or U (holes)")
        err(ax, gaps + 0.08, [get(f"gap {g}", "all sources", "surplus of the model in hole runs (2+ more x than _) as % of its whole surplus") for g in gaps], COLORS[1],
            "hole runs: 2+ more wrong deletions than missed insertions", "s")
        free = get("no clusters", "all sources", "surplus of the model in string errors as % of its surplus in label errors")
        ax.axhline(free[0], color=COLORS[2], ls="--", lw=1.2, label=f"no clusters at all: string errors, {free[0]:.0f} % of the surplus")
        ax.axhline(100, color=COLORS[7], lw=1, ls=":", label="100 % = the whole surplus")
        ax.set_xticks(gaps)
        ax.set_xlabel("kept originals allowed between two errors of one cluster (reported: 2)")
        ax.set_ylabel("share of the model's surplus of errors\nover the perfect sampler, in %")
        ax.set_ylim(60, 130)
        ru = [get(f"gap {g}", "all sources", "surplus of the model in kinds R + U as % of its whole surplus")[0] for g in gaps]
        title(ax, f"Kinds R + U hold {min(ru):.0f} to {max(ru):.0f} % of the model's surplus, whatever the cluster rule\n(above 100 %: the other kinds hold fewer errors than in the perfect sampler)")
        ax.legend(fontsize=8, loc="lower right")
        fig.savefig(C.out / "fig2_attack_a_share_and_kinds.png")
        plt.close(fig)
    # 3. attack (b)
    b = S.get("C11_attack_b")
    if isinstance(b, dict):
        st = b["strata"]
        forced = b.get("forced removal of a random original, one forward pass: change of q of the adjacent original", {})
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        ax = axes[0]
        x = np.arange(3)
        series = [("observed: all first wrong deletions (reported)", [st[f"all isolated first wrong deletions | all | break {v}"][:3] for v in x], COLORS[0], "o"),
                  ("observed: the deleted original was not suspected (its q below 0.03)", [st[f"q of the deleted original itself, just before | below 0.03 (an accident) | break {v}"][:3] for v in x], COLORS[1], "s")]
        if forced.get("model, source at t = 0, clean sites, break 0"):
            series.append(("forced removal of a random original, model", [forced[f"model, source at t = 0, clean sites, break {v}"][:3] for v in x], COLORS[2], "D"))
            series.append(("forced removal, exact posterior (an ideal model)", [forced[f"exact, source at t = 0, clean sites, break {v}"][:3] for v in x], COLORS[3], "^"))
        for j, (label, est, color, marker) in enumerate(series):
            err(ax, x + 0.07 * (j - 1.5), est, color, label, marker)
        ax.axhline(0, color=COLORS[7], lw=1)
        ax.set_xticks(x, ["0 (gap stays allowed)", "1", "2"])
        ax.set_xlabel("forbidden windows across the gap after the deletion")
        ax.set_ylabel("rise of q of the original next to the gap, one step later")
        obs, acc_ = series[0][1], series[1][1]
        text = f"Rise of the neighbour's q for break 0 / 1 / 2: observed {obs[0][0]:+.2f} / {obs[1][0]:+.2f} / {obs[2][0]:+.2f};\nwhen the deleted letter was not suspected {acc_[0][0]:+.2f} / {acc_[1][0]:+.2f} / {acc_[2][0]:+.2f}"
        if len(series) > 2:
            text += f"; forced removal {series[2][1][0][0]:+.2f} / {series[2][1][1][0]:+.2f} / {series[2][1][2][0]:+.2f}"
        title(ax, text)
        ax.legend(fontsize=7.5, loc="upper left")
        ax = axes[1]
        drawn = []
        for t_ in range(3):
            est = [forced.get(f"model, source at t = 0, dirty sites, break {v} in the sequence, break {t_} among the originals alone") for v in range(3)]
            have = [v for v in range(3) if est[v]]
            if have:
                err(ax, np.array(have) + 0.08 * (t_ - 1), [est[v][:3] for v in have], COLORS[t_], f"the originals alone: {t_} forbidden window{'s' if t_ != 1 else ''} across the gap",
                    "osD"[t_])
                drawn.append([est[v][0] for v in have])
        if drawn:
            ax.axhline(0, color=COLORS[7], lw=1)
            ax.set_xticks(range(3), ["0", "1", "2"])
            ax.set_xlabel("forbidden windows across the gap as the model sees it (insertions included)")
            ax.set_ylabel("rise of q of the original next to the gap")
            ax.set_ylim(top=1.45 * max(max(v) for v in drawn))
            title(ax, f"Forced removal next to an insertion: if the originals alone still fit together the neighbour's q rises by "
                      f"{min(drawn[0]):+.2f} to {max(drawn[0]):+.2f}, whatever break is visible; otherwise by {min(min(v) for v in drawn[1:]):+.2f} to {max(max(v) for v in drawn[1:]):+.2f}")
            ax.legend(fontsize=7.5, loc="upper left")
        fig.savefig(C.out / "fig3_attack_b_break_size.png")
        plt.close(fig)
    # 4. attack (c)
    c_ = S.get("attack_c")
    if isinstance(c_, dict):
        acc = c_["account per 1000 letters"]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        ax = axes[0]
        names = [("perfect sampler (sum of q_gap)", "perfect sampler"), ("model, first reading (kappa(1) q0)", "model: first reading\n(q at t = 0)"),
                 ("model, observed after sampling", "model: after\n100 steps")]
        names = [(k, t) for k, t in names if k in acc]
        x = np.arange(len(names))
        w = [acc[k]["wrong deletions"][0] for k, _ in names]
        m = [acc[k]["missed insertions"][0] for k, _ in names]
        ax.bar(x, w, 0.6, color=COLORS[1], label="wrong deletions")
        ax.bar(x, m, 0.6, bottom=w, color=COLORS[5], label="missed insertions")
        for j in range(len(x)):
            ax.text(x[j], w[j] / 2, f"{w[j]:.1f}", ha="center", va="center", color="white", fontsize=9)
            ax.text(x[j], w[j] + m[j] / 2, f"{m[j]:.1f}", ha="center", va="center", fontsize=9)
            ax.text(x[j], w[j] + m[j] + 1, f"{w[j] + m[j]:.1f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x, [t for _, t in names])
        ax.set_ylabel("wrongly labelled letters per 1 000 source letters")
        ax.set_ylim(0, max(a_ + b_ for a_, b_ in zip(w, m)) * 1.15)
        title(ax, f"Label errors per 1 000 letters: {w[-2] + m[-2]:.1f} in the model's first reading, {w[-1] + m[-1]:.1f} after sampling;\n"
                     f"sampling turns {w[-1] - w[-2]:.1f} missed insertions into wrong deletions")
        ax.legend(fontsize=8, loc="upper left")
        ax = axes[1]
        first = c_["what the first reading alone would give"]
        groups_ = c_.get("neighbourhoods (wrong deletions less than 13 originals apart)", {})
        rows_ = [("perfect sampler (g)", "perfect sampler"), ("one independent draw from the first reading", "model: first reading,\nletters drawn\nindependently"),
                 ("the sampler (m100)", "model: after\n100 steps")]
        rows_ = [(k, t) for k, t in rows_ if k in groups_]
        x = np.arange(len(rows_))
        firsts = [groups_[k]["neighbourhoods with a wrong deletion per 1000"][0] for k, _ in rows_]
        later = [groups_[k]["wrong deletions beyond the first of a neighbourhood per 1000"][0] for k, _ in rows_]
        ax.bar(x, firsts, 0.6, color=COLORS[0], label="the first wrong deletion of a neighbourhood")
        ax.bar(x, later, 0.6, bottom=firsts, color=COLORS[4], label="further wrong deletions within 12 originals")
        for j in range(len(x)):
            ax.text(x[j], firsts[j] / 2, f"{firsts[j]:.1f}", ha="center", va="center", color="white", fontsize=9)
            ax.text(x[j], firsts[j] + later[j] / 2, f"{later[j]:.1f}", ha="center", va="center", fontsize=9)
        r4plus = [first[k]["wrong deletions in runs of 4+ per 1000"][0] for k in ("one independent draw from the first reading", "the sampler (m100)")]
        ax.set_xticks(x, [t for _, t in rows_])
        ax.set_ylabel("wrong deletions per 1 000 source letters")
        title(ax, f"Places with a wrong deletion per 1 000 letters: {min(firsts):.1f} to {max(firsts):.1f}; what differs is how many letters go per place.\n"
                     f"Runs of 4+ adjacent originals: {r4plus[0]:.1f} per 1 000 in the first reading, {r4plus[1]:.1f} after sampling")
        ax.legend(fontsize=8, loc="upper left")
        fig.savefig(C.out / "fig4_attack_c_first_reading.png")
        plt.close(fig)
    # 5. attack (d)
    if "leak" in C.tables:
        t = C.tables["leak"]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        ax = axes[0]
        x = np.arange(len(t))
        ax.bar(x - 0.2, t.seeds_reported_per_1000, 0.38, color=COLORS[0], label="reported: every original, the whole path")
        ax.bar(x + 0.2, t.seeds_counted_per_1000, 0.38, color=COLORS[1], label="counted: wrong deletions made at the rate of the leak")
        ax.plot(x + 0.2, t.seeds_expected_per_1000, "D", color=COLORS[6], ms=5, label="expected from the exact q (grid of steps)")
        ax.set_xticks(x, [f"leak {v:g}" for v in t.leak])
        ax.set_ylabel("letters deleted by the leak itself, per 1 000 letters")
        title(ax, f"The leak itself deleted {100 * t.seeds_counted_per_1000.sum() / t.seeds_reported_per_1000.sum():.0f} % of the letters the formula assumes\n"
                     "(it cannot act where the exact q is above it, nor on letters already gone)")
        ax.legend(fontsize=8, loc="upper left")
        ax = axes[1]
        ax.errorbar(x - 0.08, t.cost_reported, yerr=[t.cost_reported - t.cost_reported_lo, t.cost_reported_hi - t.cost_reported], fmt="o", color=COLORS[0], capsize=2.5,
                    label="reported seeds, baseline i2-100")
        ax.errorbar(x + 0.08, t.cost_counted_seeds, yerr=[t.cost_counted_seeds - t.cost_counted_lo, t.cost_counted_hi - t.cost_counted_seeds], fmt="s", color=COLORS[1], capsize=2.5,
                    label="counted seeds, baseline pooled over all exact-rate runs")
        hole = next((r["recomputed"] for r in C.claims if r["claim"] == "C8" and "smallest repair" in r["quantity"]), None)
        if hole:
            ax.axhline(hole, color=COLORS[2], ls="--", lw=1.2, label=f"grammar alone: smallest repair, {hole:.1f} letters")
        cross = S.get("C9_attack_d", {}).get("cross-check with the forced removals under exact rates (table exp4 of the intervention study, not recomputed for t0 > 0)")
        if cross:
            v_ = list(cross["predicted wrong deletions per leak deletion (seed included), given when the seeds fell"].values())
            ax.plot(x, v_, "D", color=COLORS[3], ms=6, label="predicted from forced removals with exact rates")
        ax.set_xticks(x, [f"leak {v:g}" for v in t.leak])
        ax.set_ylabel("wrong deletions per letter deleted by the leak\n(that letter included)")
        ax.set_ylim(0, None)
        d_ = S.get("C9_attack_d", {})
        title(ax, f"Wrong deletions per letter deleted by the leak, all leaks pooled:\n{d_['all leaks pooled: wrong deletions per seed as reported'][0]:.2f} with the seeds of the formula, "
                     f"{d_['all leaks pooled: wrong deletions per counted seed'][0]:.2f} with the seeds counted")
        ax.legend(fontsize=8, loc="lower right")
        fig.savefig(C.out / "fig5_attack_d_leak.png")
        plt.close(fig)


# ---- main ------------------------------------------------------------------------------------------------------------

PARTS = {"1": claim1, "2": claim2, "3": claim3, "4": claim4, "5": claim5, "6": claim6, "7": claim7, "8": claim8, "9": claim9, "10": claim10, "12": claim12,
         "11": claim11, "a": attack_a, "c": attack_c}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=ROOT / "runs" / "lap-01")
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--parts", default="", help="comma-separated subset of the parts (for development)")
    args = p.parse_args()
    args.run = args.run if args.run.is_absolute() else ROOT / args.run
    start = time.time()
    C = Context(args)
    try:
        C.hazard, C.kappa_t, C.kappa = schedule(C, 100)
    except Exception as error:  # noqa: BLE001
        C.missing.append(f"schedule of {C.run / 'config.yaml'}: {type(error).__name__}: {error}")
    self_checks(C)
    for key, part in PARTS.items():
        if not args.parts or key in args.parts.split(","):
            t0 = time.time()
            part(C)
            print(f"  ({time.time() - t0:.0f} s)")
    conclusions(C)
    C.summary["missing"] = sorted(set(C.missing))
    counts = pd.Series([c["verdict"] for c in C.claims]).value_counts().to_dict()
    C.summary["verdicts"] = counts
    C.summary["not reproduced"] = [f"{c['claim']} {c['quantity']}: reported {c['reported']}, recomputed {c['recomputed']:.4g}. {c['note']}"
                                   for c in C.claims if c["verdict"] == "not reproduced"]
    if not args.parts:
        figures(C)
        C.summary["run_time_seconds"] = round(time.time() - start)
        pd.DataFrame(C.claims).to_csv(C.out / "claims.csv", index=False)
        (C.out / "summary.json").write_text(json.dumps(r4(C.summary), indent=1))
    else:
        (C.out / "scratch" / "summary_parts.json").write_text(json.dumps(r4(C.summary), indent=1))
    print(f"verdicts: {counts}")
    print(f"done in {time.time() - start:.0f} s; missing: {C.summary['missing']}")


if __name__ == "__main__":
    main()
