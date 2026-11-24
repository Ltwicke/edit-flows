#!/usr/bin/env python
r"""Bottom-up census of the error patterns in the verdict strings (topic `patterns` of the error study).

    python scripts/errors/patterns.py --run runs/lap-01

Steps 1-6 use only the verdicts (`.` kept original, `x` wrong deletion, `_` missed insertion, `+` correct
deletion), the letters, the insertion labels, the deletion steps and the model's `q0`: no grammar. The grammar
enters in step 7 only (`--blind` stops before it): the bottom-up categories are cross-tabulated with the types
`Z`, `V`, `R`, `U` of `editflows.errors.clusters`, and the references get the exact posterior as their belief at t = 0.

Corpora: M = `<run>/errors/m100.npz` (the trained model), the perfect sampler = `runs/error-study/g.npz` (exact
joint posterior sample with the prior of the coupling, K letters thrown into gaps; no time axis) or, if that file
is missing, `runs/error-study/ffbs.npz` (the same with independent insertions as prior), I2 =
`runs/error-study/i2-100.npz` (exact rates in the same Euler sampler).
Corpora that cover all sources are analysed on `--sources report` (the second half of the sources). A corpus that
covers only the first N sources (or, if `i2-100.npz` is missing, a replicate `i2-rep<k>.npz`) is analysed together
with the others in a second pass restricted to these N sources; its outputs go to a subfolder `first<N>_with_<key>`.
Missing files are named in `summary.json`.

Chosen on the exploring half (first half of the sources, model only): the 40 patterns of the census and the
dictionary of elemental shapes. The eight bottom-up categories (`CATEGORIES`) were fixed there too, before step 7
was looked at. Every number comes with a bootstrap interval over sequences. Outputs go to
`<run>/errors/results/patterns/`: `clusters_<corpus>.csv` (the cluster table), `patterns.csv` (census),
`categories.csv`, `dictionary.csv`, `clustering.csv` (null model), `timing.csv`, `crosstab_kinds.csv`,
`summary.json`, `examples.md` and the figures.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import CORRECT, DOT, MISSED, WRONG, Sources, cluster_ids, verdict_codes  # noqa: E402

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
ALPHABET = np.frombuffer(b"ABCDEFGHIJKLMNOPQRST", dtype="S1")
SYM = np.frombuffer(b".x_+", dtype="S1")
D_MAX = 20  # largest distance of the pair correlation
SIZE_CAP = 30  # cluster sizes (error letters) are binned 1 .. 30+
RUN_CAP = 20  # run lengths 1 .. 20+
T_MAX = 10  # largest distance of the timing statistics
NOISE_BINS = [("low", 0.0, 1 / 6), ("mid", 1 / 6, 1 / 3), ("high", 1 / 3, 1.01)]
NOISE_TEXT = {"all": "all sources", "low": "under 17 % inserted", "mid": "17-33 % inserted", "high": "over 33 % inserted"}
SIZE_CLASSES = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-9", 6, 9), ("10+", 10, 10**6)]

# The bottom-up categories (defined from counts and letters alone, fixed before the grammar was looked at)
CATEGORIES = {
    "L": "lone missed insertion: `_` alone",
    "X": "lone wrong deletion: `x` alone",
    "S=": "relabelling: as many wrong deletions as missed insertions and the output string equals the target string",
    "S1": "exchange 1:1: one wrong deletion, one missed insertion, the strings differ",
    "Sk": "exchange k:k (k >= 2): as many wrong deletions as missed insertions, the strings differ",
    "H1": "one net hole with company: one wrong deletion more than missed insertions, at least one missed insertion",
    "HH": "hole run: at least two wrong deletions more than missed insertions",
    "LL": "net leftover with company: more missed insertions than wrong deletions, at least two errors",
}


# ---- loading ---------------------------------------------------------------------------------------------------


def load_npz(path: Path):
    """The arrays of an npz file, None if it is missing; one retry in case the file is still being written."""
    if not path.exists():
        return None
    for attempt in range(2):
        try:
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        except Exception:  # noqa: BLE001  (a half-written zip raises several things)
            if attempt == 0:
                time.sleep(10)
    return None


def load_sources(study: Path) -> SimpleNamespace:
    s = Sources.load(study / "sources.npz")
    n_seq = len(s)
    length = np.diff(s.offsets)
    seq_of = np.repeat(np.arange(n_seq), length)
    g = SimpleNamespace(letters=s.letters.astype(np.int64), inserted=s.inserted.astype(bool), offsets=s.offsets,
                        n_seq=n_seq, length=length, seq_of=seq_of, n=len(s.letters))
    g.start_of, g.end_of = s.offsets[seq_of], s.offsets[seq_of + 1]
    g.n_ins = np.bincount(seq_of, weights=g.inserted, minlength=n_seq)
    g.n_orig = length - g.n_ins
    g.noise = g.n_ins / length
    g.orig = np.flatnonzero(~g.inserted)  # target coordinates: the originals in order
    g.orig_seq = seq_of[g.orig]
    # number of position pairs at source distance d by label: both original, both inserted, one of each
    g.c_oo, g.c_ii, g.c_oi = (np.zeros((n_seq, D_MAX)) for _ in range(3))
    for d in range(1, D_MAX + 1):
        same = seq_of[:-d] == seq_of[d:]
        a, b, s0 = g.inserted[:-d], g.inserted[d:], seq_of[:-d]
        g.c_ii[:, d - 1] = np.bincount(s0[same & a & b], minlength=n_seq)
        g.c_oo[:, d - 1] = np.bincount(s0[same & ~a & ~b], minlength=n_seq)
        g.c_oi[:, d - 1] = np.bincount(s0[same & (a ^ b)], minlength=n_seq)
    return g


def make_corpus(key: str, label: str, z: dict, g: SimpleNamespace) -> SimpleNamespace:
    """A corpus as flat arrays over all letters; letters of sources it does not cover count as kept."""
    covered = min(int(z["first"]), g.n_seq)
    n = int(g.offsets[covered])
    step = np.full(g.n, -1, dtype=np.int64)
    step[:n] = z["del_step"][:n]
    q0 = None
    if "q0" in z:
        q0 = np.zeros(g.n)
        q0[:n] = z["q0"][:n]
    c = SimpleNamespace(key=key, label=label, step=step, deleted=step >= 0, n_steps=int(z["steps"]), q0=q0, covered=covered)
    c.codes = verdict_codes(g.inserted, c.deleted)
    c.timed = c.n_steps > 0
    return c


# ---- bootstrap over sequences ------------------------------------------------------------------------------------


class Boot:
    """Ratios of per-sequence sums on the selected sequences with percentile intervals (resampling sequences)."""

    def __init__(self, sel: np.ndarray, n_boot: int, seed: int):
        self.sel = sel
        n = len(sel)
        self.w = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot).astype(np.float64)

    def sums(self, a):
        a = np.asarray(a, dtype=np.float64)[self.sel]
        return a.sum(0), self.w @ a

    def ratio(self, num, den, scale: float = 1.0):
        """(estimate, lo, hi) of scale * sum(num) / sum(den); `num` may have a second axis."""
        tn, bn = self.sums(num)
        td, bd = self.sums(den)
        if bn.ndim == 2 and bd.ndim == 1:
            bd = bd[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            est = scale * tn / td
            lo, hi = np.nanpercentile(scale * bn / bd, [2.5, 97.5], axis=0)
        return est, lo, hi

    def stat(self, num, den, scale: float = 1.0, n=None, digits: int = 4) -> dict:
        est, lo, hi = self.ratio(num, den, scale)
        out = {"value": round(float(est), digits), "lo": round(float(lo), digits), "hi": round(float(hi), digits)}
        if n is not None:
            out["n"] = int(n)
        return out

    def ratio_of_ratios(self, n1, d1, n2, d2, digits: int = 3) -> dict:
        """(sum n1 / sum d1) / (sum n2 / sum d2) with an interval."""
        (tn1, bn1), (td1, bd1), (tn2, bn2), (td2, bd2) = self.sums(n1), self.sums(d1), self.sums(n2), self.sums(d2)
        with np.errstate(divide="ignore", invalid="ignore"):
            est = (tn1 / td1) / (tn2 / td2)
            lo, hi = np.nanpercentile((bn1 / bd1) / (bn2 / bd2), [2.5, 97.5])
        return {"value": round(float(est), digits), "lo": round(float(lo), digits), "hi": round(float(hi), digits)}

    def quantiles(self, hist, qs):
        """Quantiles of a per-sequence histogram (sequences x bins): estimate and interval per quantile."""
        t, b = self.sums(hist)

        def quant(h):
            cdf = np.cumsum(h, axis=-1) / h.sum(-1, keepdims=True)
            return np.stack([(cdf < q).sum(-1) for q in qs], axis=-1)

        est = quant(t[None])[0]
        lo, hi = np.percentile(quant(b), [2.5, 97.5], axis=0)
        return est, lo, hi


def per_seq(g, index, weights=None):
    """Sum over items (given by their sequence index) per sequence."""
    return np.bincount(index, weights=weights, minlength=g.n_seq).astype(np.float64)


# ---- clusters ----------------------------------------------------------------------------------------------------


def find_clusters(codes: np.ndarray, seq_of: np.ndarray, gap: int):
    """`cluster_ids` for all sources at once: flat index of the first and the last error of every cluster, the
    flat index of every error and its cluster."""
    err = np.flatnonzero((codes == WRONG) | (codes == MISSED))
    kept = np.cumsum(codes == DOT)
    new = np.ones(len(err), dtype=bool)
    new[1:] = (np.diff(kept[err]) > gap) | (np.diff(seq_of[err]) != 0)
    starts = np.flatnonzero(new)
    return err[starts], err[np.r_[starts[1:] - 1, len(err) - 1]], err, np.cumsum(new) - 1


def expand(first: np.ndarray, last: np.ndarray):
    """Flat index of every letter of the spans first..last and the number of the span it belongs to."""
    lens = last - first + 1
    owner = np.repeat(np.arange(len(first)), lens)
    return np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(first, lens), owner


def cluster_table(g, c, gap: int = 2, strings: bool = True) -> pd.DataFrame:
    """One row per cluster: where, how many of each symbol, whether output and target are the same string."""
    first, last, _, _ = find_clusters(c.codes, g.seq_of, gap)
    prefix = np.zeros((4, g.n + 1), dtype=np.int64)
    for code in range(4):
        prefix[code, 1:] = np.cumsum(c.codes == code)
    cnt = prefix[:, last + 1] - prefix[:, first]
    src = g.seq_of[first]
    df = pd.DataFrame({"source": src, "start": first - g.offsets[src], "end": last - g.offsets[src], "n_x": cnt[WRONG],
                       "n_miss": cnt[MISSED], "n_plus": cnt[CORRECT], "n_dot": cnt[DOT]})
    df["size"] = df.n_x + df.n_miss
    df["balance"] = df.n_x - df.n_miss
    df["noise"] = np.round(g.noise[src], 4)
    # output == target inside the span (needs as many kept as original letters)
    bal = np.flatnonzero(cnt[WRONG] == cnt[MISSED])
    idx, owner = expand(first[bal], last[bal])
    kept, original = ~c.deleted[idx], ~g.inserted[idx]
    differs = np.bincount(owner[kept], weights=g.letters[idx[kept]] != g.letters[idx[original]], minlength=len(bal)) > 0
    equal = np.zeros(len(first), dtype=bool)
    equal[bal] = ~differs
    df["equal"] = equal
    df["category"] = categorise(df)
    df["first"], df["last"] = first, last  # flat indices (not written to the csv)
    if strings:
        sym, abc = SYM[c.codes], ALPHABET[g.letters]
        df["pattern"] = [sym[a : b + 1].tobytes().decode() for a, b in zip(first, last)]
        df["letters"] = [abc[a : b + 1].tobytes().decode() for a, b in zip(first, last)]
        if c.timed:
            names = np.array(["-"] + [str(k) for k in range(c.n_steps)])
            df["steps"] = [" ".join(names[c.step[a : b + 1] + 1]) for a, b in zip(first, last)]
        df["normal_form"] = df.pattern.str.replace(r"[+.]", "", regex=True)
    return df


def categorise(df: pd.DataFrame) -> np.ndarray:
    nx, nm, bal, eq = df.n_x.to_numpy(), df.n_miss.to_numpy(), df.balance.to_numpy(), df.equal.to_numpy()
    cat = np.full(len(df), "??", dtype=object)
    cat[(nx == 0) & (nm == 1)] = "L"
    cat[(nx == 1) & (nm == 0)] = "X"
    cat[(bal == 0) & eq] = "S="
    cat[(bal == 0) & ~eq & (nx == 1)] = "S1"
    cat[(bal == 0) & ~eq & (nx >= 2)] = "Sk"
    cat[(bal == 1) & (nm >= 1)] = "H1"
    cat[bal >= 2] = "HH"
    cat[(bal <= -1) & (nx + nm >= 2)] = "LL"
    assert not (cat == "??").any()
    return cat


def self_test(g, c, gap: int = 2, n_check: int = 300):
    """The vectorised clustering must reproduce `editflows.errors.cluster_ids` source by source."""
    first, last, err, cid = find_clusters(c.codes, g.seq_of, gap)
    for i in np.linspace(0, g.n_seq - 1, n_check).astype(int):
        span = slice(int(g.offsets[i]), int(g.offsets[i + 1]))
        ref = cluster_ids(c.codes[span], gap)
        mine = np.full(span.stop - span.start, -1)
        inside = (err >= span.start) & (err < span.stop)
        if inside.any():
            mine[err[inside] - span.start] = cid[inside] - cid[inside][0]
        assert (ref == mine).all(), f"clustering differs from cluster_ids in source {i}"


# ---- steps 1 and 2: cluster table, sensitivity to the gap, census ---------------------------------------------------


def gap_sensitivity(g, corpora, boot) -> pd.DataFrame:
    rows = []
    for c in corpora:
        for gap in (1, 2, 3, 4):
            df = cluster_table(g, c, gap, strings=False)
            src, size = df.source.to_numpy(), df["size"].to_numpy()
            one = np.ones(len(df))
            n_err = per_seq(g, src, size)
            row = {"corpus": c.key, "gap": gap, "n_clusters": int(np.isin(src, boot.sel).sum())}
            for name, num, den, scale in [
                ("clusters_per_1000_letters", per_seq(g, src, one), g.length, 1000),
                ("error_letters_per_cluster", n_err, per_seq(g, src, one), 1),
                ("share_clusters_single_error", per_seq(g, src, size == 1), per_seq(g, src, one), 1),
                ("share_errors_in_clusters_of_3plus", per_seq(g, src, size * (size >= 3)), n_err, 1),
                ("share_errors_in_clusters_of_6plus", per_seq(g, src, size * (size >= 6)), n_err, 1),
                ("share_clusters_output_equals_target", per_seq(g, src, df.equal), per_seq(g, src, one), 1),
                ("share_errors_in_hole_runs", per_seq(g, src, size * (df.balance >= 2)), n_err, 1),
                ("share_wrong_deletions_in_hole_runs", per_seq(g, src, df.n_x * (df.balance >= 2)), per_seq(g, src, df.n_x), 1),
            ]:
                est, lo, hi = boot.ratio(num, den, scale)
                row[name], row[name + "_lo"], row[name + "_hi"] = round(float(est), 4), round(float(lo), 4), round(float(hi), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def noise_masks(g):
    masks = {"all": np.ones(g.n_seq, dtype=bool)}
    for name, lo, hi in NOISE_BINS:
        masks[name] = (g.noise >= lo) & (g.noise < hi)
    return masks


def census(g, tables: dict, patterns: list[str], column: str, boot) -> pd.DataFrame:
    """Frequency per 1 000 source letters and share of the error letters of the given patterns, per corpus and
    noise level."""
    rows = []
    position = {p: k for k, p in enumerate(patterns)}
    for key, df in tables.items():
        which = df[column].map(position).fillna(-1).astype(int).to_numpy()
        hit = which >= 0
        src, size = df.source.to_numpy(), df["size"].to_numpy()
        counts = np.bincount(src[hit] * len(patterns) + which[hit], minlength=g.n_seq * len(patterns)).reshape(g.n_seq, -1)
        errors = np.bincount(src[hit] * len(patterns) + which[hit], weights=size[hit],
                             minlength=g.n_seq * len(patterns)).reshape(g.n_seq, -1)
        all_clusters, all_errors = per_seq(g, src), per_seq(g, src, size)
        for noise, mask in noise_masks(g).items():
            m = mask.astype(float)
            freq = boot.ratio(counts * m[:, None], g.length * m, 1000)
            share_c = boot.ratio(counts * m[:, None], all_clusters * m)
            share_e = boot.ratio(errors * m[:, None], all_errors * m)
            n_here = (counts * m[:, None])[boot.sel].sum(0)
            for k, p in enumerate(patterns):
                rows.append({"pattern": p, "errors_in_pattern": p.count("x") + p.count("_"), "corpus": key, "noise": noise,
                             "n_clusters": int(n_here[k]), "per_1000_letters": round(freq[0][k], 4),
                             "per_1000_lo": round(freq[1][k], 4), "per_1000_hi": round(freq[2][k], 4),
                             "share_of_clusters": round(share_c[0][k], 4), "share_of_error_letters": round(share_e[0][k], 4),
                             "share_err_lo": round(share_e[1][k], 4), "share_err_hi": round(share_e[2][k], 4),
                             "n_sequences": int(mask[boot.sel].sum()), "n_letters": int((g.length * m)[boot.sel].sum())})
    return pd.DataFrame(rows)


def size_table(g, tables: dict, boot) -> pd.DataFrame:
    """Clusters by number of error letters: frequency, share of clusters, share of error letters, per noise level."""
    rows = []
    for key, df in tables.items():
        src, size = df.source.to_numpy(), df["size"].to_numpy()
        for noise, mask in noise_masks(g).items():
            m = mask.astype(float)
            for name, lo, hi in SIZE_CLASSES:
                inside = (size >= lo) & (size <= hi)
                a = boot.ratio(per_seq(g, src, inside) * m, g.length * m, 1000)
                b = boot.ratio(per_seq(g, src, inside) * m, per_seq(g, src) * m)
                e = boot.ratio(per_seq(g, src, size * inside) * m, per_seq(g, src, size) * m)
                rows.append({"corpus": key, "noise": noise, "errors_in_cluster": name,
                             "n_clusters": int((per_seq(g, src, inside) * m)[boot.sel].sum()),
                             "per_1000_letters": round(float(a[0]), 4), "per_1000_lo": round(float(a[1]), 4),
                             "per_1000_hi": round(float(a[2]), 4), "share_of_clusters": round(float(b[0]), 4),
                             "share_of_error_letters": round(float(e[0]), 4), "share_err_lo": round(float(e[1]), 4),
                             "share_err_hi": round(float(e[2]), 4)})
    return pd.DataFrame(rows)


OWNER_PATTERNS = {
    "1 `x_` exactly": lambda p: p == "x_",
    "1' `x_` with correct deletions or kept originals between (x+_, x._, ...)": lambda p: re.fullmatch(r"x[+.]*_", p) is not None,
    "2 `_x` exactly": lambda p: p == "_x",
    "2' `_x` with correct deletions or kept originals between": lambda p: re.fullmatch(r"_[+.]*x", p) is not None,
    "3 contains `xxx` (3 or more adjacent source letters wrongly deleted)": lambda p: "xxx" in p,
    "3' 3 or more consecutive originals deleted (target coordinates)": lambda p: "xxx" in p.replace("+", "").replace("_", ""),
    "4 contains `x+x+x` (3 or more x separated only by +)": lambda p: re.search(r"x(?:\++x){2,}", p) is not None,
    "4' contains `x+x` (2 or more x separated only by +)": lambda p: re.search(r"x\++x", p) is not None,
    "5 mixed: contains x, _ and +": lambda p: "x" in p and "_" in p and "+" in p,
}


def owner_patterns(g, tables: dict, boot) -> pd.DataFrame:
    rows = []
    for key, df in tables.items():
        src, size = df.source.to_numpy(), df["size"].to_numpy()
        unique = df.pattern.unique()
        for name, test in OWNER_PATTERNS.items():
            hit = df.pattern.map({p: test(p) for p in unique}).to_numpy().astype(bool)
            a = boot.ratio(per_seq(g, src, hit), g.length, 1000)
            b = boot.ratio(per_seq(g, src, hit), per_seq(g, src))
            e = boot.ratio(per_seq(g, src, size * hit), per_seq(g, src, size))
            x = boot.ratio(per_seq(g, src, df.n_x * hit), per_seq(g, src, df.n_x))
            rows.append({"pattern": name, "corpus": key, "n_clusters": int(per_seq(g, src, hit)[boot.sel].sum()),
                         "per_1000_letters": round(float(a[0]), 4), "per_1000_lo": round(float(a[1]), 4),
                         "per_1000_hi": round(float(a[2]), 4), "share_of_clusters": round(float(b[0]), 4),
                         "share_of_error_letters": round(float(e[0]), 4), "share_err_lo": round(float(e[1]), 4),
                         "share_err_hi": round(float(e[2]), 4), "share_of_wrong_deletions": round(float(x[0]), 4)})
    return pd.DataFrame(rows)


# ---- step 3: normal forms and a dictionary of elemental shapes -------------------------------------------------------


def min_tiles(word: str, elements) -> int:
    """Fewest elements whose concatenation is `word`."""
    best = [0] + [10**6] * len(word)
    for i in range(1, len(word) + 1):
        b = 10**6
        for e in elements:
            j = i - len(e)
            if j >= 0 and best[j] + 1 < b and word.startswith(e, j):
                b = best[j] + 1
        best[i] = b
    return best[-1]


def min_tiles_runs(word: str) -> int:
    """Fewest elements if the shapes are `_`, `x_`, `_x` and a run of x of any length."""
    best = [0] + [10**6] * len(word)
    for i in range(1, len(word) + 1):
        b = best[i - 1] + 1  # `_` or a single x
        if i >= 2 and word[i - 2 : i] in ("x_", "_x"):
            b = min(b, best[i - 2] + 1)
        j = i
        while j > 0 and word[j - 1] == "x":  # a run of x ending here
            j -= 1
            b = min(b, best[j] + 1)
        best[i] = b
    return best[-1]


def greedy_dictionary(words: Counter, n_elements: int, max_len: int = 4) -> list[str]:
    """Start from the two letters and add, one at a time, the word over {x, _} (up to `max_len` letters) that
    lowers the mean number of elements per cluster the most."""
    candidates = [""]
    for _ in range(max_len):
        candidates = [w + s for w in candidates for s in "x_"] + [w for w in candidates if w]
    candidates = sorted({w for w in candidates if len(w) >= 2}, key=lambda w: (len(w), w))
    dictionary = ["x", "_"]
    items = list(words.items())
    while len(dictionary) < n_elements:
        base = {w: min_tiles(w, dictionary) for w, _ in items}
        scores = {}
        for cand in candidates:
            if cand in dictionary:
                continue
            trial = dictionary + [cand]
            scores[cand] = sum(n * (min_tiles(w, trial) if cand in w else base[w]) for w, n in items)
        dictionary.append(min(scores, key=scores.get))
    return dictionary


def dictionary_coverage(g, tables: dict, dictionary: list[str], boot) -> pd.DataFrame:
    """Share of clusters (and of error letters) that are one element, or at most two / three, per dictionary size."""
    rows = []
    for key, df in tables.items():
        src, size = df.source.to_numpy(), df["size"].to_numpy()
        unique = df.normal_form.unique()
        settings = [(f"{k}", dictionary[:k]) for k in range(2, len(dictionary) + 1)] + [("runs", None)]
        for name, elements in settings:
            cost = {w: (min_tiles_runs(w) if elements is None else min_tiles(w, elements)) for w in unique}
            tiles = df.normal_form.map(cost).to_numpy()
            row = {"corpus": key, "dictionary_size": name,
                   "elements": "_ | x_ | _x | run of x of any length" if elements is None else " | ".join(elements),
                   "added": "" if elements is None else elements[-1],
                   "mean_elements_per_cluster": round(float(boot.ratio(per_seq(g, src, tiles), per_seq(g, src))[0]), 4)}
            for most in (1, 2, 3):
                a = boot.ratio(per_seq(g, src, tiles <= most), per_seq(g, src))
                e = boot.ratio(per_seq(g, src, size * (tiles <= most)), per_seq(g, src, size))
                row[f"clusters_upto_{most}"], row[f"clusters_upto_{most}_lo"], row[f"clusters_upto_{most}_hi"] = (round(float(v), 4) for v in a)
                row[f"errors_upto_{most}"], row[f"errors_upto_{most}_lo"], row[f"errors_upto_{most}_hi"] = (round(float(v), 4) for v in e)
            rows.append(row)
    return pd.DataFrame(rows)


def normal_forms(g, tables: dict, boot) -> dict:
    """How many distinct descriptions and how much the most common ones cover, per normal form."""
    out = {}
    for key, df in tables.items():
        df = df[np.isin(df.source, boot.sel)]
        forms = {"raw": df.pattern, "without +": df.pattern.str.replace("+", "", regex=False),
                 "without + and .": df.normal_form, "counts (n_x, n_miss)": df.n_x.astype(str) + "," + df.n_miss.astype(str),
                 "balance n_x - n_miss": df.balance.astype(str)}
        out[key] = {}
        for name, col in forms.items():
            counts = col.value_counts()
            cum = np.cumsum(counts.to_numpy()) / len(col)
            out[key][name] = {"distinct": int(len(counts)), "n_clusters": int(len(col)),
                              **{f"top_{k}": round(float(cum[min(k, len(cum)) - 1]), 4) for k in (4, 10, 20, 40)}}
    return out


def target_runs(g, deleted):
    """Maximal runs of consecutive deleted originals of a sequence (target coordinates): first and last member as
    indices into `g.orig`, and the length."""
    dx, so = deleted[g.orig], g.orig_seq
    brk = so[1:] != so[:-1]
    a = np.flatnonzero(dx & np.r_[True, ~dx[:-1] | brk])
    b = np.flatnonzero(dx & np.r_[~dx[1:] | brk, True])
    return a, b, b - a + 1


def runs_and_insertions(g, c, boot) -> dict:
    """Is `x+x+x` the same object as `xxx`? Insertions inside and next to runs of deleted originals."""
    a, b, k = target_runs(g, c.deleted)
    fa, fb = g.orig[a], g.orig[b]
    src = g.seq_of[fa]
    inner = (fb - fa + 1) - k  # insertions between the first and the last member
    per_gap = g.n_ins / (g.n_orig + 1)  # insertions per slot between originals in this source
    plus = np.zeros(g.n + 1)
    plus[1:] = np.cumsum(c.codes == CORRECT)
    inner_plus = plus[fb + 1] - plus[fa]
    # the two slots that flank the run
    before = fa - np.where(a > 0, np.where(g.orig_seq[np.maximum(a - 1, 0)] == src, g.orig[np.maximum(a - 1, 0)] + 1, g.start_of[fa]),
                           g.start_of[fa])
    nxt = np.minimum(b + 1, len(g.orig) - 1)
    after = np.where((b + 1 < len(g.orig)) & (g.orig_seq[nxt] == src), g.orig[nxt] - 1, g.end_of[fb] - 1) - fb
    out = {}
    for name, lo, hi in [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 10**6), ("2+", 2, 10**6), ("3+", 3, 10**6)]:
        m = (k >= lo) & (k <= hi)
        row = {"n_runs": int(np.isin(src[m], boot.sel).sum()),
               "insertions_per_flanking_slot": boot.stat(per_seq(g, src[m], (before + after)[m]), per_seq(g, src[m], np.full(m.sum(), 2.0))),
               "flank_density_vs_source_average": boot.stat(per_seq(g, src[m], (before + after)[m]), per_seq(g, src[m], 2 * per_gap[src[m]]))}
        if lo >= 2:
            gaps = (k - 1)[m]
            row["insertions_per_inner_slot"] = boot.stat(per_seq(g, src[m], inner[m]), per_seq(g, src[m], gaps))
            row["inner_density_vs_source_average"] = boot.stat(per_seq(g, src[m], inner[m]), per_seq(g, src[m], gaps * per_gap[src[m]]))
            row["share_of_inner_insertions_deleted"] = boot.stat(per_seq(g, src[m], inner_plus[m]), per_seq(g, src[m], inner[m]))
            row["share_runs_without_insertion_inside (xxx)"] = boot.stat(per_seq(g, src[m], inner[m] == 0), per_seq(g, src[m]))
            row["share_runs_all_inner_insertions_deleted (x+x+x or xxx)"] = boot.stat(per_seq(g, src[m], inner_plus[m] == inner[m]), per_seq(g, src[m]))
        out[name] = row
    out["recall_overall"] = boot.stat(per_seq(g, g.seq_of, c.codes == CORRECT), g.n_ins)
    return out


# ---- step 4: letters ---------------------------------------------------------------------------------------------------


def neighbour_same(g, i, which=None):
    """Does the source letter next to letter i (left or right, same source) carry the same letter (and `which`)?"""
    which = np.ones(g.n, dtype=bool) if which is None else which
    left, right = i - 1 >= g.start_of[i], i + 1 < g.end_of[i]
    out = np.zeros(len(i), dtype=bool)
    out[left] |= which[i[left] - 1] & (g.letters[i[left] - 1] == g.letters[i[left]])
    out[right] |= which[i[right] + 1] & (g.letters[i[right] + 1] == g.letters[i[right]])
    return out


def at_edge(g, i, width: int = 2):
    return ((i - g.start_of[i]) < width) | ((g.end_of[i] - 1 - i) < width)


def letter_level(g, c, df, boot) -> dict:
    src, size = df.source.to_numpy(), df["size"].to_numpy()
    first, last = df["first"].to_numpy(), df["last"].to_numpy()
    out = {"output_equals_target": {
        "share_of_clusters": boot.stat(per_seq(g, src, df.equal), per_seq(g, src), n=np.isin(src, boot.sel).sum()),
        "share_of_error_letters": boot.stat(per_seq(g, src, size * df.equal), per_seq(g, src, size)),
        "share_of_wrong_deletions": boot.stat(per_seq(g, src, df.n_x * df.equal), per_seq(g, src, df.n_x)),
        "share_of_missed_insertions": boot.stat(per_seq(g, src, df.n_miss * df.equal), per_seq(g, src, df.n_miss))}}
    # one wrong deletion and one missed insertion: same letter? same string?
    pairs = {}
    one_one = ((df.n_x == 1) & (df.n_miss == 1)).to_numpy()
    same = g.letters[first] == g.letters[last]
    groups = {"all clusters with one x and one _": one_one, "... nothing kept between them (x_, x+_, ...)": one_one & (df.n_dot == 0).to_numpy(),
              "... a kept original between them (x._, ...)": one_one & (df.n_dot > 0).to_numpy()}
    for p in ["x_", "_x", "x+_", "_+x", "x++_", "_++x", "x._", "_.x"]:
        groups[p] = (df.pattern == p).to_numpy()
    for name, m in groups.items():
        pairs[name] = {"n": int(np.isin(src[m], boot.sel).sum()),
                       "same_letter": boot.stat(per_seq(g, src[m], same[m]), per_seq(g, src[m])),
                       "output_equals_target": boot.stat(per_seq(g, src[m], df.equal.to_numpy()[m]), per_seq(g, src[m]))}
    out["pairs"] = pairs
    out["pairs"]["chance_same_letter"] = round(1 / (g.letters.max() + 1), 4)
    for k in (2, 3, 4):
        m = ((df.n_x == k) & (df.n_miss == k)).to_numpy()
        out["pairs"][f"balanced {k}+{k}"] = {"n": int(np.isin(src[m], boot.sel).sum()),
                                           "output_equals_target": boot.stat(per_seq(g, src[m], df.equal.to_numpy()[m]), per_seq(g, src[m]))}
    # lone errors against the letters of the same kind
    groups = {"lone _": first[(df.category == "L").to_numpy()], "all insertions": np.flatnonzero(g.inserted),
              "correctly deleted insertions": np.flatnonzero(c.codes == CORRECT), "all missed insertions": np.flatnonzero(c.codes == MISSED),
              "lone x": first[(df.category == "X").to_numpy()], "all originals": g.orig, "kept originals": np.flatnonzero(c.codes == DOT),
              "all wrong deletions": np.flatnonzero(c.codes == WRONG)}
    hole = (df.category == "HH").to_numpy()
    idx, _ = expand(first[hole], last[hole])
    groups["x in hole runs"] = idx[c.codes[idx] == WRONG]
    groups["_ in hole runs"] = idx[c.codes[idx] == MISSED]
    rel = one_one & df.equal.to_numpy()
    x_first = c.codes[first] == WRONG
    groups["x of relabelled pairs"] = np.where(x_first, first, last)[rel]
    groups["_ of relabelled pairs"] = np.where(x_first, last, first)[rel]
    lone = {}
    for name, i in groups.items():
        s = g.seq_of[i]
        row = {"n": int(np.isin(s, boot.sel).sum()),
               "within_2_letters_of_an_end": boot.stat(per_seq(g, s, at_edge(g, i)), per_seq(g, s)),
               "same_letter_as_a_source_neighbour": boot.stat(per_seq(g, s, neighbour_same(g, i)), per_seq(g, s)),
               "next_to_an_insertion": boot.stat(per_seq(g, s, _next_to(g, i, g.inserted)), per_seq(g, s))}
        if c.q0 is not None:
            q = c.q0[i]
            hist = np.bincount(s * 1001 + np.minimum((q * 1000).astype(int), 1000), minlength=g.n_seq * 1001).reshape(g.n_seq, 1001)
            est, lo, hi = boot.quantiles(hist, [0.25, 0.5, 0.75])
            row["q0_quartiles"] = [round(v / 1000, 3) for v in est]
            row["q0_median_interval"] = [round(lo[1] / 1000, 3), round(hi[1] / 1000, 3)]
            row["q0_below_0.01"] = boot.stat(per_seq(g, s, q < 0.01), per_seq(g, s))
            row["q0_below_0.1"] = boot.stat(per_seq(g, s, q < 0.1), per_seq(g, s))
            row["q0_above_0.5"] = boot.stat(per_seq(g, s, q > 0.5), per_seq(g, s))
        lone[name] = row
    out["letters_by_group"] = lone
    if c.q0 is not None:  # wrong deletions of originals the model was sure about at t = 0: in which categories do they sit?
        idx, owner = expand(first, last)
        x_idx = idx[c.codes[idx] == WRONG]
        x_cat = df.category.to_numpy()[owner[c.codes[idx] == WRONG]]
        sure = c.q0[x_idx] < 0.01
        xs = g.seq_of[x_idx]
        out["wrong_deletions_with_q0_below_0.01"] = {
            "n": int(np.isin(xs[sure], boot.sel).sum()), "share_of_all_wrong_deletions": boot.stat(per_seq(g, xs, sure), per_seq(g, xs)),
            "by_category": {cat: {"share_of_these": boot.stat(per_seq(g, xs, sure & (x_cat == cat)), per_seq(g, xs, sure)),
                                  "share_of_all_wrong_deletions_in_this_category": boot.stat(per_seq(g, xs, x_cat == cat), per_seq(g, xs))}
                            for cat in CATEGORIES if cat != "L"}}
    # twins: an insertion directly next to an original with the same letter; deleting either gives the same string
    same = (g.seq_of[:-1] == g.seq_of[1:]) & (g.letters[:-1] == g.letters[1:]) & (g.inserted[:-1] ^ g.inserted[1:])
    left = np.flatnonzero(same)
    ins_i = np.where(g.inserted[left], left, left + 1)
    org_i = np.where(g.inserted[left], left + 1, left)
    s = g.seq_of[left]
    fate = {"insertion deleted, original kept (correct)": c.deleted[ins_i] & ~c.deleted[org_i],
            "original deleted, insertion kept (relabelling)": ~c.deleted[ins_i] & c.deleted[org_i],
            "both kept": ~c.deleted[ins_i] & ~c.deleted[org_i], "both deleted": c.deleted[ins_i] & c.deleted[org_i]}
    out["twins"] = {"what": "pairs (insertion, original) that are neighbours in the source and carry the same letter",
                    "n_pairs": int(np.isin(s, boot.sel).sum()), "pairs_per_1000_source_letters": boot.stat(per_seq(g, s), g.length, 1000),
                    "share_of_insertions_with_an_identical_original_neighbour": boot.stat(per_seq(g, g.seq_of[np.unique(ins_i)]), g.n_ins),
                    **{name: boot.stat(per_seq(g, s, hit), per_seq(g, s)) for name, hit in fate.items()},
                    "relabelled_twins_per_1000_source_letters": boot.stat(per_seq(g, s, fate["original deleted, insertion kept (relabelling)"]), g.length, 1000),
                    "half_of_the_twins_per_1000_source_letters (a fair coin)": boot.stat(0.5 * per_seq(g, s), g.length, 1000)}
    return out


def _next_to(g, i, which):
    left, right = i - 1 >= g.start_of[i], i + 1 < g.end_of[i]
    out = np.zeros(len(i), dtype=bool)
    out[left] |= which[i[left] - 1]
    out[right] |= which[i[right] + 1]
    return out


# ---- step 5: clustering against a null model ---------------------------------------------------------------------------------


def permuted_flags(g, deleted, rng):
    """Permute the deleted flags within every source among the originals and, separately, among the insertions."""
    out = np.empty_like(deleted)
    for idx in (g.orig, np.flatnonzero(g.inserted)):
        order = np.argsort(g.seq_of[idx] + rng.random(len(idx)))
        out[idx] = deleted[idx[order]]
    return out


def static_hists(g, deleted, gap: int):
    """Per sequence: clusters by size (count and error letters), runs of deleted originals by length (count and x)."""
    codes = verdict_codes(g.inserted, deleted)
    first, _, _, cid = find_clusters(codes, g.seq_of, gap)
    size = np.bincount(cid)
    key = g.seq_of[first] * SIZE_CAP + np.minimum(size, SIZE_CAP) - 1
    shape = (g.n_seq, SIZE_CAP)
    a, _, k = target_runs(g, deleted)
    rkey = g.orig_seq[a] * RUN_CAP + np.minimum(k, RUN_CAP) - 1
    rshape = (g.n_seq, RUN_CAP)
    return {"cluster_n": np.bincount(key, minlength=shape[0] * shape[1]).reshape(shape).astype(float),
            "cluster_e": np.bincount(key, weights=size, minlength=shape[0] * shape[1]).reshape(shape),
            "run_n": np.bincount(rkey, minlength=rshape[0] * rshape[1]).reshape(rshape).astype(float),
            "run_x": np.bincount(rkey, weights=k, minlength=rshape[0] * rshape[1]).reshape(rshape)}


def pair_counts(g, c):
    """Observed pairs of errors at distance d and their expectation under the permutation null (exact), per sequence."""
    n_seq = g.n_seq
    dx, so = c.deleted[g.orig], g.orig_seq
    m_x = np.bincount(so, weights=dx, minlength=n_seq)
    m_m = g.n_ins - np.bincount(g.seq_of, weights=c.codes == CORRECT, minlength=n_seq)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_xx = np.where(g.n_orig > 1, m_x * (m_x - 1) / (g.n_orig * (g.n_orig - 1)), 0.0)
        p_mm = np.where(g.n_ins > 1, m_m * (m_m - 1) / (g.n_ins * (g.n_ins - 1)), 0.0)
        p_xm = np.where(g.n_ins > 0, m_x * m_m / (g.n_orig * g.n_ins), 0.0)
    x, miss = c.codes == WRONG, c.codes == MISSED
    err = x | miss
    out = {k: np.zeros((n_seq, D_MAX)) for k in ("xx_target", "slots_target", "ee_source", "xm_source", "slots_source")}
    for d in range(1, D_MAX + 1):
        both = dx[:-d] & dx[d:] & (so[:-d] == so[d:])
        out["xx_target"][:, d - 1] = np.bincount(so[:-d][both], minlength=n_seq)
        out["slots_target"][:, d - 1] = np.maximum(g.n_orig - d, 0)
        same = g.seq_of[:-d] == g.seq_of[d:]
        out["ee_source"][:, d - 1] = np.bincount(g.seq_of[:-d][same & err[:-d] & err[d:]], minlength=n_seq)
        cross = ((x[:-d] & miss[d:]) | (miss[:-d] & x[d:])) & same
        out["xm_source"][:, d - 1] = np.bincount(g.seq_of[:-d][cross], minlength=n_seq)
        out["slots_source"][:, d - 1] = np.maximum(g.length - d, 0)
    out["xx_target_null"] = out["slots_target"] * p_xx[:, None]
    out["ee_source_null"] = g.c_oo * p_xx[:, None] + g.c_ii * p_mm[:, None] + g.c_oi * p_xm[:, None]
    out["xm_source_null"] = g.c_oi * p_xm[:, None]
    out["m_x"], out["m_err"] = m_x, m_x + m_m
    return out


def clustering(g, c, boot, gap: int, n_null: int, seed: int):
    """Cluster sizes, pair correlations and run lengths against the permutation null. Returns rows and headlines."""
    rng = np.random.default_rng(seed)
    data = static_hists(g, c.deleted, gap)
    null = {k: np.zeros_like(v) for k, v in data.items()}
    for _ in range(n_null):
        for k, v in static_hists(g, permuted_flags(g, c.deleted, rng), gap).items():
            null[k] += v / n_null
    rows, head = [], {}

    def add(stat, x, obs, exp, den, scale=1.0, note=""):
        """obs / den and exp / den (scaled), and obs / exp, each with an interval, for every column."""
        o, e, r = boot.ratio(obs, den, scale), boot.ratio(exp, den, scale), boot.ratio(obs, exp)
        n = np.atleast_1d(np.asarray(obs, dtype=float)[boot.sel].sum(0))
        for j, xv in enumerate(x):
            pick = (lambda t: float(np.atleast_1d(t)[j]))
            rows.append({"corpus": c.key, "statistic": stat, "x": xv, "n_observed": int(round(n[j])), "data": pick(o[0]), "data_lo": pick(o[1]),
                         "data_hi": pick(o[2]), "null": pick(e[0]), "null_lo": pick(e[1]), "null_hi": pick(e[2]),
                         "data_over_null": pick(r[0]), "ratio_lo": pick(r[1]), "ratio_hi": pick(r[2]), "note": note})

    sizes = [str(s) for s in range(1, SIZE_CAP)] + [f"{SIZE_CAP}+"]
    all_err = data["cluster_e"].sum(1)
    add("clusters of this size per 1000 source letters", sizes, data["cluster_n"], null["cluster_n"], g.length, 1000)
    add("share of error letters in clusters of this size", sizes, data["cluster_e"], null["cluster_e"], all_err)
    lengths = [str(s) for s in range(1, RUN_CAP)] + [f"{RUN_CAP}+"]
    all_x = data["run_x"].sum(1)
    add("runs of deleted originals of this length per 1000 originals (target coordinates)", lengths, data["run_n"], null["run_n"], g.n_orig, 1000)
    add("share of wrong deletions in runs of this length (target coordinates)", lengths, data["run_x"], null["run_x"], all_x)
    for k in (2, 3, 6):
        add(f"share of error letters in clusters of {k}+ errors", ["all"], data["cluster_e"][:, k - 1 :].sum(1), null["cluster_e"][:, k - 1 :].sum(1), all_err)
        add(f"share of wrong deletions in runs of {k}+ (target coordinates)", ["all"], data["run_x"][:, k - 1 :].sum(1), null["run_x"][:, k - 1 :].sum(1), all_x)
    for k in range(1, 11):  # how often a run of at least k deleted originals goes on
        add("probability that a run of at least k deleted originals is longer than k (target coordinates)", [k],
            data["run_n"][:, k:].sum(1), null["run_n"][:, k:].sum(1), data["run_n"][:, k - 1 :].sum(1))
        rows[-1].update(dict(zip(("null", "null_lo", "null_hi"), boot.ratio(null["run_n"][:, k:].sum(1), null["run_n"][:, k - 1 :].sum(1)))))
        rows[-1]["data_over_null"] = rows[-1]["data"] / rows[-1]["null"] if rows[-1]["null"] > 0 else np.nan
        rows[-1]["ratio_lo"] = rows[-1]["ratio_hi"] = np.nan
    add("clusters per 1000 source letters", ["all"], data["cluster_n"].sum(1), null["cluster_n"].sum(1), g.length, 1000)
    add("error letters per cluster", ["all"], all_err, all_err, data["cluster_n"].sum(1))
    rows[-1].update(dict(zip(("null", "null_lo", "null_hi"), boot.ratio(all_err, null["cluster_n"].sum(1)))))
    rows[-1].update(dict(zip(("data_over_null", "ratio_lo", "ratio_hi"), boot.ratio(null["cluster_n"].sum(1), data["cluster_n"].sum(1)))))

    pc = pair_counts(g, c)
    dist = list(range(1, D_MAX + 1))
    add("pairs of wrong deletions at target distance d, per 1000 pairs of originals", dist, pc["xx_target"], pc["xx_target_null"], pc["slots_target"], 1000)
    for noise, mask in noise_masks(g).items():
        if noise != "all":
            m = mask.astype(float)[:, None]
            add(f"pairs of wrong deletions at target distance d, per 1000 pairs of originals [{noise} noise]", dist, pc["xx_target"] * m,
                pc["xx_target_null"] * m, pc["slots_target"] * m, 1000, note=NOISE_TEXT[noise])
    add("pairs of errors at source distance d, per 1000 pairs of letters", dist, pc["ee_source"], pc["ee_source_null"], pc["slots_source"], 1000)
    add("pairs (wrong deletion, missed insertion) at source distance d, per 1000 pairs of letters", dist, pc["xm_source"], pc["xm_source_null"], pc["slots_source"], 1000)
    # the plain pair correlation g(d) = P(error at distance d from an error) / P(error), pooled over sequences
    for name, obs, slots, m, n_all in [("g(d) of wrong deletions in target coordinates (pooled, not corrected for noise level)", "xx_target", "slots_target", pc["m_x"], g.n_orig),
                                       ("g(d) of errors in source coordinates (pooled, not corrected for noise level)", "ee_source", "slots_source", pc["m_err"], g.length)]:
        p = np.asarray(m)[boot.sel].sum() / n_all[boot.sel].sum()
        add(name, dist, pc[obs], pc[slots] * p * p, pc[slots] * p * p, 1.0, note="data = g(d); null = 1")

    table = pd.DataFrame(rows)

    def pick(stat, x):
        r = table[(table.statistic == stat) & (table.x.astype(str) == str(x))].iloc[0]
        return {"data": round(r["data"], 4), "data_interval": [round(r.data_lo, 4), round(r.data_hi, 4)], "null": round(r["null"], 4),
                "data_over_null": round(r.data_over_null, 3), "ratio_interval": [round(r.ratio_lo, 3), round(r.ratio_hi, 3)], "n_observed": int(r.n_observed)}

    for stat, x in [("clusters per 1000 source letters", "all"), ("error letters per cluster", "all"),
                    ("share of error letters in clusters of 3+ errors", "all"), ("share of error letters in clusters of 6+ errors", "all"),
                    ("share of wrong deletions in runs of 2+ (target coordinates)", "all"),
                    ("share of wrong deletions in runs of 3+ (target coordinates)", "all"),
                    ("share of wrong deletions in runs of 6+ (target coordinates)", "all"),
                    ("pairs of wrong deletions at target distance d, per 1000 pairs of originals", 1),
                    ("pairs of wrong deletions at target distance d, per 1000 pairs of originals", 2),
                    ("pairs of wrong deletions at target distance d, per 1000 pairs of originals", 5),
                    ("pairs of wrong deletions at target distance d, per 1000 pairs of originals", 10),
                    ("pairs of wrong deletions at target distance d, per 1000 pairs of originals", 20),
                    ("pairs of errors at source distance d, per 1000 pairs of letters", 1),
                    ("pairs of errors at source distance d, per 1000 pairs of letters", 5),
                    ("pairs of errors at source distance d, per 1000 pairs of letters", 20),
                    ("pairs (wrong deletion, missed insertion) at source distance d, per 1000 pairs of letters", 1),
                    ("pairs (wrong deletion, missed insertion) at source distance d, per 1000 pairs of letters", 2)]:
        head[f"{stat} [{x}]"] = pick(stat, x)
    # wrong deletions in runs beyond chance, per 1000 originals (to compare corpora in absolute terms)
    for k in (2, 3):
        obs, exp = data["run_x"][:, k - 1 :].sum(1), null["run_x"][:, k - 1 :].sum(1)
        head[f"wrong deletions in runs of {k}+ beyond chance, per 1000 originals"] = boot.stat(obs - exp, g.n_orig, 1000)
    head["excess pairs of wrong deletions at target distance 1, per 1000 originals"] = boot.stat(
        pc["xx_target"][:, 0] - pc["xx_target_null"][:, 0], g.n_orig, 1000)
    for k in (1, 2, 3, 4, 6, 8):
        head[f"probability that a run of at least {k} deleted originals is longer"] = pick(
            "probability that a run of at least k deleted originals is longer than k (target coordinates)", k)
    # the same headline numbers within a noise level (the noise level drives the error rate)
    head["by_noise_level"] = {}
    for noise, mask in noise_masks(g).items():
        if noise == "all":
            continue
        m = mask.astype(float)
        obs3, exp3 = data["run_x"][:, 2:].sum(1) * m, null["run_x"][:, 2:].sum(1) * m
        o, e, r = boot.ratio(obs3, all_x * m), boot.ratio(exp3, all_x * m), boot.ratio(obs3, exp3)
        g1 = boot.ratio(pc["xx_target"][:, 0] * m, pc["xx_target_null"][:, 0] * m)
        size = boot.ratio(all_err * m, data["cluster_n"].sum(1) * m)
        size0 = boot.ratio(all_err * m, null["cluster_n"].sum(1) * m)
        head["by_noise_level"][NOISE_TEXT[noise]] = {
            "n_sequences": int(mask[boot.sel].sum()),
            "share of wrong deletions in runs of 3+": {"data": round(float(o[0]), 4), "data_interval": [round(float(o[1]), 4), round(float(o[2]), 4)],
                                                       "null": round(float(e[0]), 4), "data_over_null": round(float(r[0]), 2)},
            "pairs of wrong deletions at target distance 1: data over null": {"value": round(float(g1[0]), 3), "lo": round(float(g1[1]), 3), "hi": round(float(g1[2]), 3)},
            "error letters per cluster": {"data": round(float(size[0]), 3), "data_interval": [round(float(size[1]), 3), round(float(size[2]), 3)],
                                          "null": round(float(size0[0]), 3)}}
    return table, head, pc


# ---- step 6: timing ----------------------------------------------------------------------------------------------------------


def operational_time(run: Path, n_steps: int):
    """Hazard of the schedule at the Euler steps, operational time at mid-step and at the end; the schedule comes
    from the run's config."""
    try:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        kappa = instantiate(OmegaConf.load(run / "config.yaml").task.path.kappa)
        hazard = np.array([float(kappa.loss_coefficient(torch.tensor(k / n_steps, dtype=torch.float64))) for k in range(n_steps)])
        origin = "schedule of the run's config.yaml"
    except Exception as error:  # noqa: BLE001
        hazard = 3.0 * np.exp(1.5 * np.arange(n_steps) / n_steps)
        origin = f"fallback 3 exp(1.5 t) ({type(error).__name__})"
    h = 1.0 / n_steps
    return hazard, h * (np.cumsum(hazard) - 0.5 * hazard), h * hazard.sum(), origin


def permuted_steps(g, c, rng, within=None):
    """Shuffle the deletion steps among the wrong deletions and, separately, among the correct deletions of the
    same source (or of the same `within` group for the wrong deletions)."""
    out = c.step.copy()
    for code in (WRONG, CORRECT):
        idx = np.flatnonzero(c.codes == code)
        group = g.seq_of[idx] if within is None or code == CORRECT else within[idx]
        # both orders list the letters group by group: the first in random order, the second as they stand
        out[idx[np.argsort(group, kind="stable")]] = c.step[idx[np.argsort(group + rng.random(len(idx)))]]
    return out


def timing(g, c, df, boot, run: Path, n_null: int, seed: int):
    rng = np.random.default_rng(seed)
    hazard, lam, lam_end, origin = operational_time(run, c.n_steps)
    n_steps, n_seq = c.n_steps, g.n_seq
    out = {"schedule": origin, "operational_time_at_the_end": round(float(lam_end), 4), "n_steps": n_steps}
    rows = []
    first, last = df["first"].to_numpy(), df["last"].to_numpy()
    idx, owner = expand(first, last)
    cat_of = np.full(g.n, "", dtype=object)
    cat_of[idx] = df.category.to_numpy()[owner]
    cluster_of = np.full(g.n, -1, dtype=np.int64)
    cluster_of[idx] = owner

    # (a) when are letters deleted: step and operational time, by kind of deletion
    groups = {"correct deletions (+)": c.codes == CORRECT, "wrong deletions (x)": c.codes == WRONG}
    for cat in ("X", "S=", "S1", "Sk", "H1", "HH"):
        groups[f"x in {cat}"] = (c.codes == WRONG) & (cat_of == cat)
    reference = {"letter that is certainly an insertion (q = 1)": 1 - np.exp(-(lam + 0.5 * hazard / n_steps)),
                 "letter with a small constant q": (lam + 0.5 * hazard / n_steps) / lam_end}
    when = {}
    for name, mask in groups.items():
        i = np.flatnonzero(mask)
        s = g.seq_of[i]
        hist = np.bincount(s * n_steps + c.step[i], minlength=n_seq * n_steps).reshape(n_seq, n_steps).astype(float)
        est, lo, hi = boot.quantiles(hist, [0.25, 0.5, 0.75])
        total, bsum = boot.sums(hist)
        cdf, bcdf = np.cumsum(total) / total.sum(), np.cumsum(bsum, 1) / bsum.sum(1, keepdims=True)
        blo, bhi = np.percentile(bcdf, [2.5, 97.5], axis=0)
        for k in range(n_steps):
            rows.append({"corpus": c.key, "table": "cdf of the deletion step", "group": name, "x": k, "operational_time": lam[k],
                         "value": cdf[k], "lo": blo[k], "hi": bhi[k], "n": int(total.sum())})
        when[name] = {"n": int(total.sum()), "step_quartiles": [int(v) for v in est], "median_step_interval": [int(lo[1]), int(hi[1])],
                      "mean_operational_time_over_end": boot.stat(per_seq(g, s, lam[c.step[i]]), per_seq(g, s) * lam_end),
                      "share_in_last_fifth_of_the_steps": boot.stat(per_seq(g, s, c.step[i] >= 0.8 * n_steps), per_seq(g, s))}
    for name, cdf in reference.items():
        for k in range(n_steps):
            rows.append({"corpus": c.key, "table": "cdf of the deletion step", "group": "reference: " + name, "x": k,
                         "operational_time": lam[k], "value": cdf[k], "lo": np.nan, "hi": np.nan, "n": 0})
    when["references"] = {"q = 1: mean operational time over end": round(float((1 - (1 + lam_end) * math.exp(-lam_end)) / (1 - math.exp(-lam_end)) / lam_end), 4),
                          "small constant q: mean operational time over end": 0.5,
                          "small constant q: share in last fifth of the steps": round(float(1 - lam[int(0.8 * n_steps)] / lam_end), 4)}
    out["when"] = when

    # (b) which deletion of a cluster comes first
    def first_is_wrong(step, rows_of):
        is_x, is_p = c.codes[idx] == WRONG, c.codes[idx] == CORRECT
        mx, mp = np.full(len(df), 10**9), np.full(len(df), 10**9)
        np.minimum.at(mx, owner[is_x], step[idx[is_x]])
        np.minimum.at(mp, owner[is_p], step[idx[is_p]])
        return ((mx < mp) + 0.5 * (mx == mp))[rows_of]

    src = df.source.to_numpy()
    order = {}
    for name, m in [("clusters with a wrong and a correct deletion", ((df.n_x >= 1) & (df.n_plus >= 1)).to_numpy()),
                    ("hole runs (HH) with a correct deletion inside", ((df.category == "HH") & (df.n_plus >= 1)).to_numpy())]:
        rows_of = np.flatnonzero(m)
        obs = first_is_wrong(c.step, rows_of)
        null = np.zeros(len(rows_of))
        for _ in range(n_null):
            null += first_is_wrong(permuted_steps(g, c, rng), rows_of) / n_null
        share_x = (df.n_x / (df.n_x + df.n_plus)).to_numpy()[rows_of]
        order[name] = {"n": int(np.isin(src[rows_of], boot.sel).sum()),
                       "first_deletion_is_wrong": boot.stat(per_seq(g, src[rows_of], obs), per_seq(g, src[rows_of])),
                       "same_if_steps_shuffled_within_kind_and_source": boot.stat(per_seq(g, src[rows_of], null), per_seq(g, src[rows_of])),
                       "share_of_wrong_among_the_deletions": boot.stat(per_seq(g, src[rows_of], share_x), per_seq(g, src[rows_of]))}
    out["first_event"] = order

    # (c) two deletions at distance d: same step? how far apart in time?
    sx, so = c.step[g.orig], g.orig_seq

    def pair_times(step_orig, step_all):
        res = {k: np.zeros((n_seq, T_MAX)) for k in ("xx_n", "xx_same", "xx_abs", "xx_near", "xx_lam", "pp_n", "pp_same", "pp_abs")}
        plus = c.codes == CORRECT
        for d in range(1, T_MAX + 1):
            both = (step_orig[:-d] >= 0) & (step_orig[d:] >= 0) & (so[:-d] == so[d:])
            s0, diff = so[:-d][both], np.abs(step_orig[:-d][both] - step_orig[d:][both])
            res["xx_n"][:, d - 1] = np.bincount(s0, minlength=n_seq)
            res["xx_same"][:, d - 1] = np.bincount(s0, weights=diff == 0, minlength=n_seq)
            res["xx_near"][:, d - 1] = np.bincount(s0, weights=diff <= 0.05 * n_steps, minlength=n_seq)
            res["xx_abs"][:, d - 1] = np.bincount(s0, weights=diff, minlength=n_seq)
            res["xx_lam"][:, d - 1] = np.bincount(s0, weights=np.abs(lam[step_orig[:-d][both]] - lam[step_orig[d:][both]]), minlength=n_seq)
            both = plus[:-d] & plus[d:] & (g.seq_of[:-d] == g.seq_of[d:])
            s0, diff = g.seq_of[:-d][both], np.abs(step_all[:-d][both] - step_all[d:][both])
            res["pp_n"][:, d - 1] = np.bincount(s0, minlength=n_seq)
            res["pp_same"][:, d - 1] = np.bincount(s0, weights=diff == 0, minlength=n_seq)
            res["pp_abs"][:, d - 1] = np.bincount(s0, weights=diff, minlength=n_seq)
        return res

    obs = pair_times(sx, c.step)
    null = {k: np.zeros_like(v) for k, v in obs.items()}
    for _ in range(n_null):
        shuffled = permuted_steps(g, c, rng)
        for k, v in pair_times(shuffled[g.orig], shuffled).items():
            null[k] += v / n_null
    pairs = {}
    for kind, label in (("xx", "two wrong deletions at target distance d"), ("pp", "two correct deletions at source distance d")):
        for stat, text in (("same", "share deleted in the same step"), ("abs", "mean distance in steps"), ("near", "share within 5 % of the steps"),
                           ("lam", "mean distance in operational time")):
            if f"{kind}_{stat}" not in obs:
                continue
            o, e, r = (boot.ratio(obs[f"{kind}_{stat}"], obs[f"{kind}_n"]), boot.ratio(null[f"{kind}_{stat}"], null[f"{kind}_n"]),
                       boot.ratio(obs[f"{kind}_{stat}"], null[f"{kind}_{stat}"]))
            n = obs[f"{kind}_n"][boot.sel].sum(0)
            for d in range(T_MAX):
                rows.append({"corpus": c.key, "table": f"{label}: {text}", "group": "data", "x": d + 1, "value": o[0][d], "lo": o[1][d], "hi": o[2][d], "n": int(n[d])})
                rows.append({"corpus": c.key, "table": f"{label}: {text}", "group": "steps shuffled within kind and source", "x": d + 1,
                             "value": e[0][d], "lo": e[1][d], "hi": e[2][d], "n": int(n[d])})
                rows.append({"corpus": c.key, "table": f"{label}: {text}", "group": "data over shuffled", "x": d + 1, "value": r[0][d], "lo": r[1][d], "hi": r[2][d], "n": int(n[d])})
            pairs[f"{label}: {text}"] = {f"d={d + 1}": {"data": round(float(o[0][d]), 4), "data_interval": [round(float(o[1][d]), 4), round(float(o[2][d]), 4)],
                                                      "shuffled": round(float(e[0][d]), 4), "data_over_shuffled": round(float(r[0][d]), 3),
                                                      "ratio_interval": [round(float(r[1][d]), 3), round(float(r[2][d]), 3)], "n_pairs": int(n[d])}
                                         for d in (0, 1, 2, 4, 9)}
    out["pairs_in_time"] = pairs

    # (d) order inside runs of deleted originals: a follower is deleted strictly after a neighbour in the run
    a, b, k = target_runs(g, c.deleted)
    members, run_of = expand(a, b)  # indices into g.orig
    run_src = so[a]

    def followers(step_orig):
        s = step_orig[members].astype(float)
        left = np.where(np.r_[False, run_of[1:] == run_of[:-1]], np.r_[np.inf, s[:-1]], np.inf)
        right = np.where(np.r_[run_of[:-1] == run_of[1:], False], np.r_[s[1:], np.inf], np.inf)
        return np.bincount(run_of, weights=np.minimum(left, right) < s, minlength=len(a))

    run_id = np.full(len(g.orig), -1, dtype=np.int64)
    run_id[members] = run_of
    within = np.full(g.n, -1, dtype=np.int64)
    within[g.orig] = run_id
    obs_f = followers(sx)
    null_f = np.zeros(len(a))
    for _ in range(n_null):
        null_f += followers(permuted_steps(g, c, rng, within=within)[g.orig]) / n_null
    spread = {}
    for name, lo, hi in [("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6+", 6, 10**6), ("2+", 2, 10**6)]:
        m = (k >= lo) & (k <= hi)
        if not m.any():
            continue
        spread[f"runs of {name}"] = {
            "n_runs": int(np.isin(run_src[m], boot.sel).sum()),
            "share_of_members_deleted_after_a_neighbour": boot.stat(per_seq(g, run_src[m], obs_f[m]), per_seq(g, run_src[m], k[m])),
            "same_if_steps_shuffled_within_the_run": boot.stat(per_seq(g, run_src[m], null_f[m]), per_seq(g, run_src[m], k[m])),
            "data_over_shuffled": boot.stat(per_seq(g, run_src[m], obs_f[m]), per_seq(g, run_src[m], null_f[m])),
            "if_one_seed_spreads_outward": round(float(((k[m] - 1).sum()) / k[m].sum()), 4)}
    out["order_inside_runs"] = spread

    # (e) step offset against the distance from the first wrong deletion of a cluster
    xs = np.flatnonzero((c.codes == WRONG) & (cluster_of >= 0))
    many = df.n_x.to_numpy()[cluster_of[xs]] >= 2
    xs = xs[many]
    cl = cluster_of[xs]
    tpos = np.cumsum(~g.inserted) - 1  # target coordinate of every source letter (of the original at or before it)
    edges = [1, 2, 3, 4, 5, 10, 10**6]
    names = ["1", "2", "3", "4", "5-9", "10+"]

    def offsets(step):
        s = step[xs]
        order = np.lexsort((rng.random(len(xs)), s, cl))
        firsts = order[np.r_[True, cl[order][1:] != cl[order][:-1]]]
        seed_of = np.full(len(df), -1)
        seed_of[cl[firsts]] = firsts
        seed = seed_of[cl]
        other = seed != np.arange(len(xs))
        dist = np.abs(tpos[xs] - tpos[xs[seed]])[other]
        off = (s - s[seed])[other]
        bin_ = np.searchsorted(edges, dist, side="right") - 1
        key = g.seq_of[xs][other] * len(names) + bin_
        shape = (n_seq, len(names))
        return (np.bincount(key, minlength=n_seq * len(names)).reshape(shape).astype(float),
                np.bincount(key, weights=off, minlength=n_seq * len(names)).reshape(shape),
                np.bincount(key, weights=off == 0, minlength=n_seq * len(names)).reshape(shape))

    o_n, o_off, o_same = offsets(c.step)
    e_n, e_off, e_same = (np.zeros_like(o_n) for _ in range(3))
    within_cluster = np.where(c.codes == WRONG, cluster_of, -1)
    for _ in range(n_null):
        for acc, v in zip((e_n, e_off, e_same), offsets(permuted_steps(g, c, rng, within=within_cluster))):
            acc += v / n_null
    seed_offsets = {}
    o, e = boot.ratio(o_off, o_n), boot.ratio(e_off, e_n)
    o2, e2 = boot.ratio(o_same, o_n), boot.ratio(e_same, e_n)
    for j, name in enumerate(names):
        n = int(o_n[boot.sel].sum(0)[j])
        seed_offsets[f"target distance {name}"] = {
            "n": n, "mean_steps_after_the_first": {"value": round(float(o[0][j]), 3), "lo": round(float(o[1][j]), 3), "hi": round(float(o[2][j]), 3)},
            "same_if_steps_shuffled_within_the_cluster": {"value": round(float(e[0][j]), 3), "lo": round(float(e[1][j]), 3), "hi": round(float(e[2][j]), 3)},
            "share_in_the_same_step_as_the_first": round(float(o2[0][j]), 4), "same_step_if_shuffled": round(float(e2[0][j]), 4)}
        for group, val in (("data", o), ("steps shuffled within the cluster", e)):
            rows.append({"corpus": c.key, "table": "mean steps after the first wrong deletion of the cluster, by target distance from it", "group": group,
                         "x": name, "value": val[0][j], "lo": val[1][j], "hi": val[2][j], "n": n})
    out["offset_from_first_wrong_deletion"] = seed_offsets

    # (f) the model's belief at t = 0 against what happened, and the belief of the members of a run by order of deletion
    if c.q0 is not None:
        frozen = {}
        # q0 of the members of a run by the order of deletion
        s = sx[members].astype(float)
        rank_key = np.lexsort((rng.random(len(members)), s, run_of))
        rank = np.empty(len(members), dtype=np.int64)
        rank[rank_key] = np.arange(len(members)) - np.repeat(np.cumsum(k) - k, k)
        q_m, src_m, k_m = c.q0[g.orig[members]], so[members], k[run_of]
        by_rank = {}
        for name, m in [("lone wrong deletions (run of 1)", k_m == 1), ("first deleted of a run of 3+", (k_m >= 3) & (rank == 0)),
                        ("second deleted of a run of 3+", (k_m >= 3) & (rank == 1)), ("later deleted of a run of 3+", (k_m >= 3) & (rank >= 2))]:
            hist = np.bincount(src_m[m] * 1001 + np.minimum((q_m[m] * 1000).astype(int), 1000), minlength=n_seq * 1001).reshape(n_seq, 1001)
            est, lo, hi = boot.quantiles(hist, [0.25, 0.5, 0.75])
            by_rank[name] = {"n": int(np.isin(src_m[m], boot.sel).sum()), "q0_quartiles": [round(v / 1000, 3) for v in est],
                             "q0_median_interval": [round(lo[1] / 1000, 3), round(hi[1] / 1000, 3)],
                             "q0_below_0.05": boot.stat(per_seq(g, src_m[m], q_m[m] < 0.05), per_seq(g, src_m[m]))}
        frozen["q0_by_order_of_deletion"] = by_rank
        out["belief_at_t0"] = frozen
    return pd.DataFrame(rows), out


def deletion_probability(q, hazard, n_steps: int):
    """Probability that the Euler sampler deletes a letter whose q stays at this value for the whole path."""
    grid = np.linspace(0, 1, 4001)
    survive = np.ones_like(grid)
    for hz in hazard:
        survive *= 1 - np.minimum(hz / n_steps * grid, 1.0)
    return np.interp(q, grid, 1 - survive)


def belief_analysis(g, c, belief, p_del, boot, what: str) -> dict:
    """What happened against the belief at t = 0 (`belief` = probability that a letter is an insertion, `p_del` =
    the probability of deleting it that follows from this belief alone), and the fate of an original the belief
    was sure about depending on the fate of an ambiguous neighbouring original."""
    so = g.orig_seq
    po, dxo, q_orig = p_del[g.orig], c.deleted[g.orig], belief[g.orig]
    adj = so[:-1] == so[1:]
    out = {"what": what,
           "wrong_deletions_observed_over_expected": boot.stat(per_seq(g, so, dxo), per_seq(g, so, po), n=np.isin(so[dxo], boot.sel).sum()),
           "missed_insertions_observed_over_expected": boot.stat(per_seq(g, g.seq_of, c.codes == MISSED), per_seq(g, g.seq_of, g.inserted * (1 - p_del)),
                                                                 n=np.isin(g.seq_of[c.codes == MISSED], boot.sel).sum()),
           "neighbouring_pairs_of_wrong_deletions_observed_over_independent": boot.stat(
               per_seq(g, so[:-1][adj], (dxo[:-1] & dxo[1:])[adj]), per_seq(g, so[:-1][adj], (po[:-1] * po[1:])[adj]),
               n=np.isin(so[:-1][adj & dxo[:-1] & dxo[1:]], boot.sel).sum())}
    levels = [0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0001]
    out["originals_by_belief"] = {}
    for lo, hi in zip(levels[:-1], levels[1:]):
        m = (q_orig >= lo) & (q_orig < hi)
        out["originals_by_belief"][f"{lo:g} <= q < {min(hi, 1):g}"] = {
            "n_originals": int(np.isin(so[m], boot.sel).sum()), "share_of_wrong_deletions": boot.stat(per_seq(g, so[m], dxo[m]), per_seq(g, so, dxo)),
            "deleted": boot.stat(per_seq(g, so[m], dxo[m]), per_seq(g, so[m])), "expected_from_belief": boot.stat(per_seq(g, so[m], po[m]), per_seq(g, so[m]))}
    # an original the belief is sure about (j) next to an ambiguous original (n): is j deleted more often when n is?
    effect = {}
    for d in (1, 2):
        left = np.flatnonzero(so[:-d] == so[d:])
        j, n = np.r_[left, left + d], np.r_[left + d, left]
        for name, jlo, jhi in (("q < 0.001", 0, 0.001), ("0.001 <= q < 0.01", 0.001, 0.01), ("0.01 <= q < 0.1", 0.01, 0.1)):
            for nlo, nhi in ((0.1, 0.3), (0.3, 0.6), (0.6, 1.0001)):
                m = (q_orig[j] >= jlo) & (q_orig[j] < jhi) & (q_orig[n] >= nlo) & (q_orig[n] < nhi)
                jj, gone = j[m], dxo[n[m]]
                num1, den1 = per_seq(g, so[jj][gone], dxo[jj][gone]), per_seq(g, so[jj][gone])
                num0, den0 = per_seq(g, so[jj][~gone], dxo[jj][~gone]), per_seq(g, so[jj][~gone])
                row = {"n_pairs_neighbour_deleted": int(den1[boot.sel].sum()), "n_pairs_neighbour_kept": int(den0[boot.sel].sum()),
                       "deleted_if_neighbour_deleted": boot.stat(num1, den1), "deleted_if_neighbour_kept": boot.stat(num0, den0),
                       "risk_ratio": boot.ratio_of_ratios(num1, den1, num0, den0)}
                if c.timed:
                    both = gone & dxo[jj]
                    later = c.step[g.orig[jj[both]]] > c.step[g.orig[n[m][both]]]
                    row["if_both_deleted_share_where_the_sure_one_goes_later"] = boot.stat(per_seq(g, so[jj][both], later), per_seq(g, so[jj][both]))
                effect[f"target distance {d}: original with {name} next to an original with {nlo:g} <= q < {min(nhi, 1):g}"] = row
    out["sure_original_next_to_ambiguous_original"] = effect
    return out


# ---- categories, examples, step 7 -------------------------------------------------------------------------------------------------


def category_table(g, tables: dict, boot) -> pd.DataFrame:
    rows = []
    for key, df in tables.items():
        src, size = df.source.to_numpy(), df["size"].to_numpy()
        for (noise, mask), (cat, text) in ((nm, ct) for nm in noise_masks(g).items() for ct in CATEGORIES.items()):
            m = (df.category == cat).to_numpy()
            w = mask.astype(float)
            a = boot.ratio(per_seq(g, src, m) * w, g.length * w, 1000)
            b = boot.ratio(per_seq(g, src, m) * w, per_seq(g, src) * w)
            e = boot.ratio(per_seq(g, src, size * m) * w, per_seq(g, src, size) * w)
            x = boot.ratio(per_seq(g, src, df.n_x * m) * w, g.length * w, 1000)
            u = boot.ratio(per_seq(g, src, df.n_miss * m) * w, g.length * w, 1000)
            rows.append({"category": cat, "meaning": text, "corpus": key, "noise": noise, "n_clusters": int((per_seq(g, src, m) * w)[boot.sel].sum()),
                         "clusters_per_1000_letters": round(float(a[0]), 4), "clusters_lo": round(float(a[1]), 4), "clusters_hi": round(float(a[2]), 4),
                         "share_of_clusters": round(float(b[0]), 4), "share_of_error_letters": round(float(e[0]), 4),
                         "share_err_lo": round(float(e[1]), 4), "share_err_hi": round(float(e[2]), 4),
                         "wrong_deletions_per_1000_letters": round(float(x[0]), 4), "wrong_lo": round(float(x[1]), 4), "wrong_hi": round(float(x[2]), 4),
                         "missed_per_1000_letters": round(float(u[0]), 4), "missed_lo": round(float(u[1]), 4), "missed_hi": round(float(u[2]), 4)})
    return pd.DataFrame(rows)


def surplus_in_hole_runs(g, tables: dict, corpora, boot) -> dict:
    """Where does the main corpus make more errors than a reference? Share of its surplus that sits in hole runs,
    and the hole runs by the length of the source (a check that they are not an effect of long sequences)."""
    main, out = tables[corpora[0].key], {}

    def sums(df, column, only_holes):
        w = df[column].to_numpy() * ((df.category == "HH").to_numpy() if only_holes else 1)
        return per_seq(g, df.source.to_numpy(), w)

    for c in corpora[1:]:
        ref = tables[c.key]
        row = {}
        for name, column in (("wrong deletions", "n_x"), ("missed insertions", "n_miss"), ("error letters", "size")):
            row[f"surplus of {name} per 1000 source letters"] = boot.stat(sums(main, column, False) - sums(ref, column, False), g.length, 1000)
            row[f"share of that surplus in hole runs ({name})"] = boot.stat(sums(main, column, True) - sums(ref, column, True),
                                                                           sums(main, column, False) - sums(ref, column, False))
        out[f"{corpora[0].key} minus {c.key}"] = row
    edges = np.quantile(g.length[boot.sel], [0, 1 / 3, 2 / 3, 1])
    out["hole runs per 1000 source letters by length of the source"] = {}
    for c in corpora:
        df = tables[c.key]
        holes = per_seq(g, df.source.to_numpy(), (df.category == "HH").to_numpy())
        out["hole runs per 1000 source letters by length of the source"][c.key] = {
            f"{int(lo)}-{int(hi)} letters": boot.stat(holes * ((g.length >= lo) & (g.length <= hi)), g.length * ((g.length >= lo) & (g.length <= hi)), 1000)
            for lo, hi in zip(edges[:-1], edges[1:])}
    return out


def balance_table(g, tables: dict, boot, lo: int = -3, hi: int = 8) -> pd.DataFrame:
    """Errors by the balance of their cluster (wrong deletions minus missed insertions)."""
    rows = []
    for key, df in tables.items():
        src, size = df.source.to_numpy(), df["size"].to_numpy()
        bal = np.clip(df.balance.to_numpy(), lo, hi)
        for v in range(lo, hi + 1):
            m = bal == v
            a = boot.ratio(per_seq(g, src, m), g.length, 1000)
            e = boot.ratio(per_seq(g, src, size * m), g.length, 1000)
            rows.append({"corpus": key, "balance": v, "label": f"{v}" if lo < v < hi else (f"{v} or less" if v == lo else f"{v} or more"),
                         "n_clusters": int(per_seq(g, src, m)[boot.sel].sum()), "clusters_per_1000_letters": round(float(a[0]), 4),
                         "clusters_lo": round(float(a[1]), 4), "clusters_hi": round(float(a[2]), 4),
                         "error_letters_per_1000_letters": round(float(e[0]), 4), "errors_lo": round(float(e[1]), 4), "errors_hi": round(float(e[2]), 4)})
    return pd.DataFrame(rows)


def write_examples(path: Path, g, c, df, boot, seed: int, per_category: int = 5, context: int = 6):
    rng = np.random.default_rng(seed)
    sym, abc = SYM[c.codes], ALPHABET[g.letters]
    lines = [f"# Examples of every bottom-up category ({c.label})", "",
             "Letters `A`-`T` of the source with the cluster in brackets, the verdict of every letter (`.` kept original, "
             "`x` wrong deletion, `_` missed insertion, `+` correct deletion) and, for the letters of the cluster, the Euler "
             "step of the deletion (`-` = kept). Positions count source letters from 0. Examples are drawn at random "
             "(clusters of at most 40 letters) from the sources used for the reported numbers.", ""]
    chosen = df[np.isin(df.source, boot.sel) & (df.end - df.start < 40)]
    for cat, text in CATEGORIES.items():
        pool = chosen[chosen.category == cat]
        lines += [f"## {cat} - {text}", "", f"{int((df[np.isin(df.source, boot.sel)].category == cat).sum()):,} clusters of this category in the reported sources.", ""]
        for _, r in pool.sample(min(per_category, len(pool)), random_state=int(rng.integers(1 << 31))).sort_values("source").iterrows():
            a, b = int(r["first"]), int(r["last"])
            lo, hi = max(a - context, int(g.start_of[a])), min(b + context, int(g.end_of[a]) - 1)

            def show(arr):
                return f"{arr[lo:a].tobytes().decode()}[{arr[a:b + 1].tobytes().decode()}]{arr[b + 1:hi + 1].tobytes().decode()}"

            lines += [f"source {int(r.source)}, noise level {r.noise:.2f}, letters {int(r.start)}-{int(r.end)} of {int(g.length[int(r.source)])}; "
                      f"{int(r.n_x)} x, {int(r.n_miss)} _, {int(r.n_plus)} +" + (", output string = target string" if r.equal else ""), "",
                      "```", f"letters  {show(abc)}", f"verdict  {show(sym)}", f"steps    {r.steps}" if c.timed else "steps    (no time axis)", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def grammar_kinds(g, c, df, sel) -> np.ndarray:
    """Step 7: the type Z / V / R / U of `editflows.errors.clusters` for every cluster of the selected sources."""
    from editflows.errors import clusters
    from make_synthetic_targets import transition_table

    eligible = transition_table(np.random.default_rng(0), 20, 5, 1.0) > 0
    kind = np.full(len(df), "", dtype=object)
    rows_of = {s: rows for s, rows in df.groupby("source").indices.items()}
    for i in sel:
        if i not in rows_of:
            continue
        span = slice(int(g.offsets[i]), int(g.offsets[i + 1]))
        ref = clusters(g.letters[span], g.inserted[span], c.deleted[span], eligible)
        rows = rows_of[i]
        assert (ref.start.to_numpy() == df.start.to_numpy()[rows]).all() and (ref.end.to_numpy() == df.end.to_numpy()[rows]).all()
        kind[rows] = ref.kind.to_numpy()
    return kind


def crosstab(g, tables: dict, kinds: dict, boot) -> pd.DataFrame:
    rows = []
    for key, df in tables.items():
        src, size, kind = df.source.to_numpy(), df["size"].to_numpy(), kinds[key]
        for cat in CATEGORIES:
            in_cat = (df.category == cat).to_numpy()
            for k in "ZVRU":
                m = in_cat & (kind == k)
                a = boot.ratio(per_seq(g, src, m), per_seq(g, src, in_cat))
                e = boot.ratio(per_seq(g, src, size * m), per_seq(g, src, size))
                rows.append({"corpus": key, "category": cat, "kind": k, "n_clusters": int(per_seq(g, src, m)[boot.sel].sum()),
                             "share_of_category": round(float(a[0]), 4), "share_lo": round(float(a[1]), 4), "share_hi": round(float(a[2]), 4),
                             "share_of_all_error_letters": round(float(e[0]), 4)})
    return pd.DataFrame(rows)


# ---- figures ---------------------------------------------------------------------------------------------------------------------


def perfect_key(corpora):
    """Key of the corpus that plays the perfect sampler (None if there is none)."""
    return next((c.key for c in corpora if c.role == "perfect"), None)


def style():
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6,
                         "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 150,
                         "savefig.bbox": "tight", "axes.titlelocation": "left"})


def save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)


def fig_census(path, pat: pd.DataFrame, corpora, colour, n_show: int = 16):
    d = pat[pat.noise == "all"]
    order = d[d.corpus == corpora[0].key].sort_values("per_1000_letters", ascending=False).pattern.tolist()[:n_show]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    height = 0.8 / len(corpora)
    for j, c in enumerate(corpora):
        r = d[d.corpus == c.key].set_index("pattern").loc[order]
        y = np.arange(len(order)) + (j - (len(corpora) - 1) / 2) * height
        ax.barh(y, r.per_1000_letters, height=height * 0.9, color=colour[c.key], label=c.label,
                xerr=[r.per_1000_letters - r.per_1000_lo, r.per_1000_hi - r.per_1000_letters], error_kw={"lw": 0.8, "ecolor": "#333333"})
    ax.set_yticks(np.arange(len(order)), [" ".join(p) for p in order], family="monospace", fontsize=11)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("clusters with exactly this verdict pattern per 1 000 source letters (log scale; bars: 95 % interval)")
    ax.set_ylabel("verdict pattern from the first to the last error of a cluster")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="x wrong deletion, _ missed insertion,\n+ correct deletion, . kept original")
    top4 = d[(d.corpus == corpora[0].key) & d.pattern.isin(order[:4])].share_of_clusters.sum()
    names = "  ".join("'" + " ".join(p) + "'" for p in order[:4])
    ax.set_title(f"Four patterns ({names}) make up {100 * top4:.0f} % of the model's error clusters;\n"
                 "the same patterns lead in the perfect sampler, the model adds patterns with several x")
    save(fig, path)


def fig_dictionary(path, cov: pd.DataFrame, corpora, colour):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for c in corpora:
        r = cov[(cov.corpus == c.key) & (cov.dictionary_size != "runs")]
        x = r.dictionary_size.astype(int)
        ax.plot(x, 100 * r.clusters_upto_2, color=colour[c.key], lw=2, marker="o", ms=4, label=f"{c.label}: clusters")
        ax.plot(x, 100 * r.errors_upto_2, color=colour[c.key], lw=2, ls="--", marker="s", ms=4, label=f"{c.label}: error letters")
    r = cov[cov.corpus == corpora[0].key]
    r = r[r.dictionary_size != "runs"]
    for xv, added, yv in zip(r.dictionary_size.astype(int), r.added, 100 * r.clusters_upto_2):
        ax.annotate(added, (xv, yv), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, family="monospace", color="#333333")
    ax.set_ylim(0, 105)
    ax.set_xlabel("number of elemental shapes in the dictionary (the shape added at each size is written above the curve)")
    ax.set_ylabel("% that are one shape or two shapes in a row")
    ax.legend(ncol=len(corpora), fontsize=8, loc="lower right")
    m = cov[(cov.corpus == corpora[0].key) & (cov.dictionary_size == "4")].iloc[0]
    ax.set_title(f"Four shapes ({m.elements.replace(' | ', ', ')}) describe {100 * m.clusters_upto_2:.0f} % of the model's clusters\n"
                 f"as one or two shapes, but only {100 * m.errors_upto_2:.0f} % of its error letters: the rest sits in long clusters")
    save(fig, path)


def fig_paircorr(path, clus: pd.DataFrame, corpora, colour):
    fig, axes = plt.subplots(1, len(NOISE_BINS), figsize=(11.5, 4.6), sharey=True)
    first = {}
    for ax, (noise, _, _) in zip(axes, NOISE_BINS):
        stat = f"pairs of wrong deletions at target distance d, per 1000 pairs of originals [{noise} noise]"
        for j, c in enumerate(corpora):
            r = clus[(clus.corpus == c.key) & (clus.statistic == stat)]
            x = r.x.astype(int)
            ax.plot(x, r.data_over_null, color=colour[c.key], lw=2, marker="o", ms=3, label=c.label)
            ax.fill_between(x, r.ratio_lo, r.ratio_hi, color=colour[c.key], alpha=0.2, lw=0)
            ax.annotate(f"{r.data_over_null.iloc[0]:.1f}", (1, r.data_over_null.iloc[0]), textcoords="offset points", xytext=(7 + 24 * j, 4), va="center",
                        fontsize=9, color=colour[c.key], fontweight="bold")
            first[(noise, c.key)] = r.data_over_null.iloc[0]
        ax.axhline(1, color=COLORS[7], lw=1.2, ls="--", label="chance: wrong deletions shuffled\namong the originals of a source")
        ax.set_yscale("log")
        ax.set_xticks(range(2, D_MAX + 1, 2))
        ax.set_title(f"sources with {NOISE_TEXT[noise]}", fontweight="normal", fontsize=10)
        ax.set_xlabel("distance between two originals (in originals)")
    axes[0].set_ylabel("pairs of wrong deletions: observed / chance (log scale)")
    axes[-1].legend(fontsize=8)
    main = corpora[0].key
    title = (f"A wrong deletion of the model makes a wrong deletion of the neighbouring original {first[('low', main)]:.0f} (low noise) to "
             f"{first[('high', main)]:.0f} (high noise) times as likely as chance")
    perfect = perfect_key(corpora)
    if perfect is not None:
        title += f";\nthe perfect sampler: {first[('low', perfect)]:.0f} to {first[('high', perfect)]:.0f} times. The attraction fades within about ten letters"
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def fig_runs(path, clus: pd.DataFrame, corpora, colour, show: int = 14):
    stat = "runs of deleted originals of this length per 1000 originals (target coordinates)"
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for c in corpora:
        r = clus[(clus.corpus == c.key) & (clus.statistic == stat)].iloc[:show]
        x = np.arange(1, len(r) + 1)
        ok = r["data"] > 0
        ax.plot(x[ok], r["data"][ok], color=colour[c.key], lw=2, marker="o", ms=4, label=f"{c.label}: observed")
        ok = r["null"] > 0
        ax.plot(x[ok], r["null"][ok], color=colour[c.key], lw=1.2, ls=":", label=f"{c.label}: by chance")
    ax.set_yscale("log")
    ax.set_xticks(range(1, show + 1))
    ax.set_xlabel("length of a run of consecutive original letters that were all deleted (target coordinates)")
    ax.set_ylabel("runs per 1 000 original letters (log scale)")
    ax.legend(ncol=len(corpora), fontsize=8)
    share = "share of wrong deletions in runs of 3+ (target coordinates)"
    m = clus[(clus.corpus == corpora[0].key) & (clus.statistic == share)].iloc[0]
    title = f"{100 * m['data']:.0f} % of the model's wrong deletions lie in runs of three or more consecutive originals\n(by chance: {100 * m['null']:.0f} %"
    if perfect_key(corpora) is not None:
        f = clus[(clus.corpus == perfect_key(corpora)) & (clus.statistic == share)].iloc[0]
        title += f"; perfect sampler: {100 * f['data']:.0f} %"
    ax.set_title(title + "); dotted: wrong deletions shuffled within the source")
    save(fig, path)


def fig_when(path, tim: pd.DataFrame, corpora, colour):
    timed = [c for c in corpora if c.timed]
    fig, axes = plt.subplots(1, len(timed), figsize=(max(5.2 * len(timed), 9.5), 4.8), sharey=True, squeeze=False)
    # kinds of deletion, in colours that do not stand for a sampler elsewhere
    groups = [("correct deletions (+)", COLORS[6], "-", "correct deletions"), ("x in X", COLORS[4], "-", "lone wrong deletions"),
              ("x in S=", COLORS[3], "-", "wrong deletions of relabellings"), ("x in HH", COLORS[5], "-", "wrong deletions of hole runs")]
    for ax, c in zip(axes[0], timed):
        d = tim[(tim.corpus == c.key) & (tim.table == "cdf of the deletion step")]
        for name, col, ls, text in groups:
            r = d[d.group == name]
            if len(r):
                ax.plot(r.x.astype(int) + 1, r.value, color=col, lw=2, ls=ls, label=text)
        for name, ls, text in [("reference: letter that is certainly an insertion (q = 1)", "--", "reference: certain insertion (q = 1)"),
                               ("reference: letter with a small constant q", ":", "reference: small constant q")]:
            r = d[d.group == name]
            ax.plot(r.x.astype(int) + 1, r.value, color=COLORS[7], lw=1.2, ls=ls, label=text)
        ax.set_xlabel("Euler step (of %d)" % c.n_steps)
        ax.set_title(c.label, fontweight="normal")
    axes[0][0].set_ylabel("share of these deletions that has happened by this step")
    axes[0][-1].legend(fontsize=8, loc="lower right")
    d = tim[(tim.corpus == timed[0].key) & (tim.table == "cdf of the deletion step")]

    def median(group):
        r = d[d.group == group]
        return int(r.x[r.value >= 0.5].min()) + 1

    fig.suptitle(f"Wrong deletions are not late accidents ({timed[0].label}): half of the correct deletions are done by step {median('correct deletions (+)')},\n"
                 f"half of the wrong deletions of hole runs by step {median('x in HH')}; a letter with a small constant deletion probability "
                 f"would need until step {median('reference: letter with a small constant q')}",
                 x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def fig_spread(path, tim: pd.DataFrame, corpora, colour):
    timed = [c for c in corpora if c.timed]
    table = "mean steps after the first wrong deletion of the cluster, by target distance from it"
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for c in timed:
        d = tim[(tim.corpus == c.key) & (tim.table == table)]
        r = d[d.group == "data"]
        x = np.arange(len(r))
        ax.errorbar(x, r.value, yerr=[r.value - r.lo, r.hi - r.value], color=colour[c.key], lw=2, marker="o", ms=5, capsize=2, label=f"{c.label}: observed")
        e = d[d.group == "steps shuffled within the cluster"]
        ax.plot(x, e.value, color=colour[c.key], lw=1.2, ls=":", marker="s", ms=3, label=f"{c.label}: steps shuffled among the wrong deletions of the cluster")
    ax.set_xticks(x, r.x.astype(str))
    ax.set_ylim(0, None)
    ax.set_xlabel("distance from the first wrong deletion of the cluster, counted in original letters")
    ax.set_ylabel("Euler steps after the first wrong deletion (mean, 95 % interval)")
    ax.legend(fontsize=8, loc="lower right")
    m = tim[(tim.corpus == timed[0].key) & (tim.table == table)]
    o, e = m[m.group == "data"].set_index("x").value, m[m.group == "steps shuffled within the cluster"].set_index("x").value
    title = (f"Clusters grow outward, but slowly ({timed[0].label}): the neighbour of the first wrong deletion\n"
             f"follows {o['1']:.0f} steps later, a letter 5-9 originals away {o['5-9']:.0f} steps later (shuffled: {e['1']:.0f} and {e['5-9']:.0f})")
    if len(timed) > 1:
        m = tim[(tim.corpus == timed[1].key) & (tim.table == table)]
        o2, e2 = m[m.group == "data"].set_index("x").value, m[m.group == "steps shuffled within the cluster"].set_index("x").value
        title += f";\n{timed[1].label}: {o2['1']:.0f} and {o2['5-9']:.0f} (shuffled: {e2['1']:.0f} and {e2['5-9']:.0f})"
    ax.set_title(title)
    save(fig, path)


def fig_crosstab(path, cross: pd.DataFrame, cats: pd.DataFrame, corpus):
    d = cross[cross.corpus == corpus.key]
    table = d.pivot(index="category", columns="kind", values="share_of_category").loc[list(CATEGORIES), list("ZVRU")]
    n = d.pivot(index="category", columns="kind", values="n_clusters").loc[list(CATEGORIES), list("ZVRU")]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.imshow(table.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.to_numpy()[i, j]
            ax.text(j, i, f"{100 * v:.0f} %\n({int(n.to_numpy()[i, j]):,})", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "#222222")
    share = cats[(cats.corpus == corpus.key) & (cats.noise == "all")].set_index("category").share_of_error_letters
    short = {"L": "L  lone _", "X": "X  lone x", "S=": "S= relabelling (same string)", "S1": "S1 exchange 1 : 1", "Sk": "Sk exchange k : k",
             "H1": "H1 one x more than _", "HH": "HH hole run: 2+ x more than _", "LL": "LL more _ than x"}
    ax.set_yticks(range(len(CATEGORIES)), [f"{short[k]}  [{100 * share[k]:.0f} % of errors]" for k in CATEGORIES], family="monospace", fontsize=8)
    ax.set_xticks(range(4), ["Z\noutput =\ntarget", "V\nother valid\nalternative", "R\ngrammatical\nhole", "U\nforbidden\nwindow left"])
    ax.grid(False)
    ax.set_xlabel("type from the grammar (editflows.errors.clusters)")
    ax.set_ylabel("bottom-up category (from verdicts and letters only)")
    total = n.to_numpy().sum()
    fig.suptitle(f"{corpus.label}: relabellings are exactly type Z and net holes are R or U. What the verdicts cannot see is a forbidden window\n"
                 f"left behind (U): {100 * n['U'].sum() / total:.0f} % of all clusters, from {100 * table['U'].drop('S=').min():.0f} % of a category "
                 f"up to {100 * table.loc['HH', 'U']:.0f} % of the hole runs",
                 x=0.02, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, path)


def fig_categories(path, cats: pd.DataFrame, corpora, colour):
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    width = 0.8 / len(corpora)
    for j, c in enumerate(corpora):
        r = cats[(cats.corpus == c.key) & (cats.noise == "all")].set_index("category").loc[list(CATEGORIES)]
        x = np.arange(len(r)) + (j - (len(corpora) - 1) / 2) * width
        ax.bar(x, r.wrong_deletions_per_1000_letters, width=width * 0.9, color=colour[c.key], label=f"{c.label}: wrong deletions")
        ax.bar(x, r.missed_per_1000_letters, bottom=r.wrong_deletions_per_1000_letters, width=width * 0.9, color=colour[c.key], alpha=0.45,
               label=f"{c.label}: missed insertions")
    names = {"L": "L\nlone _", "X": "X\nlone x", "S=": "S=\nrelabelling\n(same string)", "S1": "S1\nexchange\n1 : 1", "Sk": "Sk\nexchange\nk : k",
             "H1": "H1\none x more\nthan _", "HH": "HH\nhole run: 2+\nx more than _", "LL": "LL\nmore _\nthan x"}
    ax.set_xticks(range(len(CATEGORIES)), [names[k] for k in CATEGORIES], fontsize=8)
    ax.set_ylabel("error letters per 1 000 source letters")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.05 * ax.get_ylim()[1])
    ax.legend(fontsize=8, ncol=1, loc="upper left")
    cats = cats[cats.noise == "all"]
    m = cats[cats.corpus == corpora[0].key].set_index("category")
    title = f"Hole runs hold {100 * m.share_of_error_letters['HH']:.0f} % of the model's error letters"
    if perfect_key(corpora) is not None:
        f = cats[cats.corpus == perfect_key(corpora)].set_index("category")
        other = [f"{100 * f.share_of_error_letters['HH']:.0f} % in the perfect sampler"]
        if "i2-100" in set(cats.corpus):
            other.append(f"{100 * cats[cats.corpus == 'i2-100'].set_index('category').share_of_error_letters['HH']:.0f} % with exact rates")
        ratio = (m.clusters_per_1000_letters / f.clusters_per_1000_letters).drop("HH")
        title += f" ({', '.join(other)});\nthe other categories are {ratio.min():.1f} to {ratio.max():.1f} times as frequent as in the perfect sampler"
    ax.set_title(title)
    save(fig, path)


# ---- main ------------------------------------------------------------------------------------------------------------------------


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def analyse(g, corpora, sel, out: Path, args, run: Path, study: Path, choices: dict, colour: dict, summary: dict, main_pass: bool):
    """All steps for the corpora on the sequences `sel`; tables, figures and summary.json go to `out`."""
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    half = g.n_seq // 2
    boot = Boot(sel, args.boot, args.seed)
    main_c = corpora[0]
    covered = min(c.covered for c in corpora)
    summary = {**summary, "sources_used": f"sequences {int(sel[0])}-{int(sel[-1])}", "n_sequences": int(len(sel)),
               "n_source_letters": int(g.length[sel].sum()), "n_originals": int(g.n_orig[sel].sum()), "n_insertions": int(g.n_ins[sel].sum()),
               "corpora": {c.key: c.label for c in corpora}}
    print(f"corpora: {[c.key for c in corpora]}; sequences {sel[0]}-{sel[-1]} -> {out}", flush=True)

    # step 1: the cluster table
    tables = {}
    for c in corpora:
        self_test(g, c, args.gap)
        df = cluster_table(g, c, args.gap)
        df = df[df.source < covered].reset_index(drop=True)
        df.insert(0, "set", np.where(df.source < half, "explore", "report"))
        tables[c.key] = df
        if main_pass or c.covered < g.n_seq:
            df.drop(columns=["first", "last"]).to_csv(out / f"clusters_{c.key}.csv", index=False)
        inside = np.isin(g.seq_of, sel)
        tp = (c.codes[inside] == CORRECT).sum()
        summary.setdefault("totals", {})[c.key] = {
            "precision": round(float(tp / c.deleted[inside].sum()), 4), "recall": round(float(tp / g.inserted[inside].sum()), 4),
            "wrong_deletions": int((c.codes[inside] == WRONG).sum()), "missed_insertions": int((c.codes[inside] == MISSED).sum()),
            "clusters": int(np.isin(df.source, sel).sum()),
            "wrong_deletions_per_1000_letters": boot.stat(per_seq(g, g.seq_of, c.codes == WRONG), g.length, 1000),
            "missed_insertions_per_1000_letters": boot.stat(per_seq(g, g.seq_of, c.codes == MISSED), g.length, 1000)}
    gaps = gap_sensitivity(g, corpora, boot)
    gaps.to_csv(out / "gap_sensitivity.csv", index=False)
    print(f"[{time.time() - started:.0f} s] cluster tables", flush=True)

    # step 2: census (the patterns were chosen on the exploring half of the main corpus)
    top = choices["patterns"]
    pat = census(g, tables, top, "pattern", boot)
    pat.to_csv(out / "patterns.csv", index=False)
    census(g, tables, choices["normal_forms"], "normal_form", boot).to_csv(out / "patterns_normal_form.csv", index=False)
    size_table(g, tables, boot).to_csv(out / "cluster_sizes.csv", index=False)
    owner = owner_patterns(g, tables, boot)
    owner.to_csv(out / "owner_patterns.csv", index=False)
    summary["owner_patterns"] = {name: {key: {k: r[k] for k in ("n_clusters", "per_1000_letters", "per_1000_lo", "per_1000_hi", "share_of_clusters",
                                                                 "share_of_error_letters", "share_err_lo", "share_err_hi", "share_of_wrong_deletions")}
                                        for key, r in owner[owner.pattern == name].set_index("corpus").iterrows()} for name in OWNER_PATTERNS}
    d = pat[pat.noise == "all"]
    summary["census"] = {key: {"top_4_share_of_clusters": round(float(d[(d.corpus == key) & d.pattern.isin(top[:4])].share_of_clusters.sum()), 4),
                               "top_4_share_of_error_letters": round(float(d[(d.corpus == key) & d.pattern.isin(top[:4])].share_of_error_letters.sum()), 4),
                               "top_40_share_of_clusters": round(float(d[d.corpus == key].share_of_clusters.sum()), 4),
                               "top_40_share_of_error_letters": round(float(d[d.corpus == key].share_of_error_letters.sum()), 4)} for key in tables}
    summary["census"]["top_40_patterns_chosen_on_exploring_half"] = top
    print(f"[{time.time() - started:.0f} s] census", flush=True)

    # step 3: normal forms and dictionary (chosen on the exploring half of the main corpus)
    summary["normal_forms"] = normal_forms(g, tables, boot)
    dictionary = choices["dictionary"]
    cov = dictionary_coverage(g, tables, dictionary, boot)
    cov.to_csv(out / "dictionary.csv", index=False)
    keep = ["mean_elements_per_cluster", "clusters_upto_1", "clusters_upto_2", "clusters_upto_2_lo", "clusters_upto_2_hi", "clusters_upto_3",
            "errors_upto_1", "errors_upto_2", "errors_upto_2_lo", "errors_upto_2_hi", "errors_upto_3"]
    summary["dictionary"] = {"greedy_order": dictionary, "coverage": {key: {r.dictionary_size: {"elements": r.elements, **{k: r[k] for k in keep}}
                                                                            for _, r in cov[cov.corpus == key].iterrows() if r.dictionary_size in ("2", "4", "6", "8", "12", "runs")}
                                                                      for key in tables}}
    summary["runs_and_insertions"] = {c.key: runs_and_insertions(g, c, boot) for c in corpora}
    cats = category_table(g, tables, boot)
    cats.to_csv(out / "categories.csv", index=False)
    balance_table(g, tables, boot).to_csv(out / "balance.csv", index=False)
    keep = ("n_clusters", "clusters_per_1000_letters", "clusters_lo", "clusters_hi", "share_of_clusters", "share_of_error_letters", "share_err_lo",
            "share_err_hi", "wrong_deletions_per_1000_letters", "wrong_lo", "wrong_hi", "missed_per_1000_letters")
    summary["categories_table"] = {key: {r.category: {k: r[k] for k in keep} for _, r in cats[(cats.corpus == key) & (cats.noise == "all")].iterrows()}
                                   for key in tables}
    summary["hole_runs_by_noise_level"] = {key: {NOISE_TEXT[r.noise]: {k: r[k] for k in keep} for _, r in cats[(cats.corpus == key) & (cats.category == "HH")].iterrows()}
                                           for key in tables}
    summary["surplus"] = surplus_in_hole_runs(g, tables, corpora, boot)
    print(f"[{time.time() - started:.0f} s] dictionary", flush=True)

    # step 4: letters
    summary["letters"] = {c.key: letter_level(g, c, tables[c.key], boot) for c in corpora}
    summary["gap_sensitivity"] = {f"{r.corpus} gap {r.gap}": {k: r[k] for k in gaps.columns if k not in ("corpus", "gap") and not k.endswith(("_lo", "_hi"))}
                                  for _, r in gaps.iterrows()}
    print(f"[{time.time() - started:.0f} s] letters", flush=True)

    # step 5: clustering against chance
    parts, summary["clustering"] = [], {}
    for c in corpora:
        table, head, _ = clustering(g, c, boot, args.gap, args.null, args.seed + 1)
        parts.append(table)
        summary["clustering"][c.key] = head
        print(f"[{time.time() - started:.0f} s] clustering {c.key}", flush=True)
    clus = pd.concat(parts, ignore_index=True)
    clus.to_csv(out / "clustering.csv", index=False)

    # step 6: timing
    parts, summary["timing"] = [], {}
    for c in corpora:
        if c.timed:
            table, head = timing(g, c, tables[c.key], boot, run, args.null, args.seed + 2)
            parts.append(table)
            summary["timing"][c.key] = head
            print(f"[{time.time() - started:.0f} s] timing {c.key}", flush=True)
    tim = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    summary["belief_at_t0"] = {}
    for c in corpora:
        if c.q0 is not None:
            hazard = operational_time(run, c.n_steps)[0]
            summary["belief_at_t0"][c.key] = belief_analysis(
                g, c, c.q0, deletion_probability(c.q0, hazard, c.n_steps), boot,
                "belief = the model's q at t = 0; expected = deletion probability of the sampler if q stayed at that value")
    if len(tim):
        tim.to_csv(out / "timing.csv", index=False)
    if main_pass:
        write_examples(out / "examples.md", g, main_c, tables[main_c.key], boot, args.seed)
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")  # steps 1-6 are on disk before the grammar is read

    # figures of steps 1-6
    fig_census(out / "fig1_census.png", pat, corpora, colour)
    fig_categories(out / "fig2_categories.png", cats, corpora, colour)
    fig_dictionary(out / "fig3_dictionary.png", cov, corpora, colour)
    fig_paircorr(out / "fig4_pair_correlation.png", clus, corpora, colour)
    fig_runs(out / "fig5_run_lengths.png", clus, corpora, colour)
    if len(tim):
        fig_when(out / "fig6_when.png", tim, corpora, colour)
        fig_spread(out / "fig7_spread.png", tim, corpora, colour)
    print(f"[{time.time() - started:.0f} s] figures of steps 1-6", flush=True)
    if args.blind:
        return

    # step 7: the grammar's types against the bottom-up categories
    kinds = {c.key: grammar_kinds(g, c, tables[c.key], sel) for c in corpora}
    cross = crosstab(g, tables, kinds, boot)
    cross.to_csv(out / "crosstab_kinds.csv", index=False)
    summary["crosstab_with_grammar_kinds"] = {
        key: {cat: {k: {"n": int(r.n_clusters), "share_of_category": r.share_of_category, "interval": [r.share_lo, r.share_hi]}
                    for k, r in cross[(cross.corpus == key) & (cross.category == cat)].set_index("kind").iterrows()} for cat in CATEGORIES} for key in tables}
    for c in corpora:
        df, kind = tables[c.key], kinds[c.key]
        used = np.isin(df.source, sel)
        summary["crosstab_with_grammar_kinds"][c.key]["kind_shares_of_clusters"] = {k: round(float((kind[used] == k).mean()), 4) for k in "ZVRU"}
        summary["crosstab_with_grammar_kinds"][c.key]["kind_shares_of_error_letters"] = {
            k: round(float(df["size"][used][kind[used] == k].sum() / df["size"][used].sum()), 4) for k in "ZVRU"}
    fig_crosstab(out / "fig8_crosstab_kinds.png", cross, cats, main_c)
    # the references have no q0: their belief at t = 0 is the exact posterior, which comes from the grammar
    posterior = load_npz(study / "posterior.npz")
    if posterior is not None:
        for c in corpora:
            if c.q0 is None:
                # g.npz samples the posterior with the prior of the coupling (q_gap), the other references the one with independent insertions
                name = "q_gap" if c.key == "g" and "q_gap" in posterior else "q_true"
                belief = posterior[name].astype(np.float64)
                p_del = deletion_probability(belief, operational_time(run, c.n_steps)[0], c.n_steps) if c.timed else belief
                summary["belief_at_t0"][c.key] = belief_analysis(
                    g, c, belief, p_del, boot, f"belief = exact posterior at t = 0 (`{name}` of posterior.npz); expected = "
                    + ("deletion probability of the sampler if q stayed at that value" if c.timed else "the posterior itself (no dynamics)"))
    summary["run_time_seconds"] = round(time.time() - started)
    (out / "summary.json").write_text(json.dumps(jsonable(summary), indent=1), encoding="utf-8")
    print(f"[{time.time() - started:.0f} s] done -> {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/lap-01"))
    p.add_argument("--sources", choices=["report", "explore", "all"], default="report",
                   help="sequences the numbers are computed on: second half (default), first half, all")
    p.add_argument("--gap", type=int, default=2)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--null", type=int, default=20, help="permutations of the null models")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="output folder (default <run>/errors/results/patterns)")
    p.add_argument("--blind", action="store_true", help="stop before step 7 (the only step that reads the grammar)")
    args = p.parse_args()
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")  # empty cells of the bootstrap
    run = args.run if args.run.is_absolute() else ROOT / args.run
    study = run.parent / "error-study"
    out = args.out or run / "errors" / "results" / "patterns"
    (out / "scratch").mkdir(parents=True, exist_ok=True)
    style()

    g = load_sources(study)
    half = g.n_seq // 2
    sel = {"report": np.arange(half, g.n_seq), "explore": np.arange(half), "all": np.arange(g.n_seq)}[args.sources]
    exact_label = "exact rates in the same Euler sampler"
    missing = []
    perfect = study / "g.npz"  # exact posterior sample with the prior of the coupling; before it existed: ffbs.npz
    if load_npz(perfect) is None:
        missing.append(f"{perfect.relative_to(ROOT).as_posix()} (missing): ffbs.npz is the perfect sampler")
        perfect = study / "ffbs.npz"
    files = [("m100", "model", "trained model", run / "errors" / "m100.npz"),
             (perfect.stem, "perfect", "perfect sampler (exact posterior sample)", perfect),
             ("i2-100", "exact", exact_label, study / "i2-100.npz")]
    by_role = {"model": COLORS[0], "exact": COLORS[1], "perfect": COLORS[2]}  # the same colour for a sampler in all figures of the study
    colour = {key: by_role[role] for key, role, _, _ in files}
    corpora, partial = [], []
    for key, role, label, path in files:
        z = load_npz(path)
        if z is None:
            missing.append(f"{path.relative_to(ROOT).as_posix()} (missing)")
            continue
        c = make_corpus(key, label, z, g)
        c.role = role
        if c.covered < g.n_seq:
            missing.append(f"{path.relative_to(ROOT).as_posix()} (covers only the first {c.covered} sources)")
            partial.append(c)
        else:
            corpora.append(c)
    assert corpora and corpora[0].key == "m100", "the corpus of the model (<run>/errors/m100.npz, all sources) is needed"
    main_c = corpora[0]
    if not any(c.key == "i2-100" for c in corpora + partial):
        # no i2-100 at all: a replicate of the exact sampler on the first sources stands in, if it uses the same number of steps
        for path in sorted(study.glob("i2-rep*.npz"), key=lambda f: int(re.sub(r"\D", "", f.stem[2:]) or 0)):
            z = load_npz(path)
            if z is not None and int(z["steps"]) == main_c.n_steps:
                c = make_corpus("i2-100", exact_label, z, g)
                c.role = "exact"
                c.label += f" ({path.name})"
                partial.append(c)
                missing.append(f"in its place {path.relative_to(ROOT).as_posix()} (first {c.covered} sources) is used in a separate pass")
                break
    # what is chosen on the exploring half of the model's clusters: the 40 patterns of the census and the dictionary
    explore = cluster_table(g, main_c, args.gap)
    explore = explore[explore.source < half]
    choices = {"patterns": explore.pattern.value_counts().index[:40].tolist(),
               "normal_forms": explore.normal_form.value_counts().index[:40].tolist(),
               "dictionary": greedy_dictionary(Counter(explore.normal_form), 12)}
    summary = {"run": run.name, "perfect_sampler_file": perfect.relative_to(ROOT).as_posix(), "gap": args.gap, "bootstrap_resamples": args.boot, "null_permutations": args.null, "missing_files": missing,
               "intervals": "95 % percentile bootstrap over sequences", "categories": CATEGORIES,
               "chosen_on_exploring_half": f"sequences 0-{half - 1} of the model: the 40 patterns of the census, the dictionary of shapes"}
    analyse(g, corpora, sel, out, args, run, study, choices, colour, summary, main_pass=True)
    for c in partial:
        if c.covered >= 100:
            note = (f"{c.label} covers only the first {c.covered} sources. They belong to the exploring half, on which the census patterns and the "
                    "dictionary were chosen; nothing else was tuned. All corpora are restricted to these sources here.")
            analyse(g, corpora + [c], np.arange(c.covered), out / f"first{c.covered}_with_{c.key}", args, run, study, choices, colour,
                    {**summary, "note": note}, main_pass=False)


if __name__ == "__main__":
    main()
