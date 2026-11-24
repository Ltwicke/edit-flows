#!/usr/bin/env python
"""Solutions that need no grammar: how to count, bound, diagnose and reduce the errors of the deletion flow with
nothing but sources, insertion labels, the model's outputs and samples.

    python scripts/errors/solutions.py --run runs/lap-01            # analysis (CPU) -> runs/lap-01/errors/results/solutions/
    python scripts/errors/solutions.py --run runs/lap-01 --stage sample --variant anchor0.2,sharp2 --sources 0:500   # GPU

The analysis uses whatever corpora exist (it lists what is missing) and has five parts:

1. counting: label-level against string-level precision / recall / F1 (any deletion set that yields the same
   output string is the same answer); validated with the clusters of kind Z;
2. bracket for the error no sampler can remove: upper bounds from the calibrated first reading of the model
   (Brier score) and from two samples per source (2 x error - disagreement), lower bounds from the ambiguity of the
   labels themselves (identical neighbours; all embeddings of the target in the source); validated with the
   reference posterior and the perfect sampler;
3. self-consistency: deletion frequency of a letter against kappa(1) * q0, its first reading; a sample against
   another sample and against the truth;
4. sampler variants: step counts (`m<n>.npz`), exclusion, sharpening, floor, anchoring (GPU corpora made by
   `scripts/build_error_corpus.py ... --tag v-<name>` or by `--stage sample` here), decisions from one forward pass
   and the consensus of replicates (CPU). Variants are compared on the sources all of them cover (a prefix of the
   tuning half) and, if they were also run there, on the held-out sources 3000-5999 with paired intervals;
5. a detector for single outputs: deleted although the first reading said "original".

The grammar is used for validation only (cluster kinds, reference posterior, perfect sampler). `--stage sample`
runs sampler variants (comma-separated) for a range of sources and writes
`results/solutions/scratch/s-<variant>-<a>-<b>.npz`; a variant is a `+`-joined list of `anchor<c>` (letters whose
first reading is below c are protected for the whole run), `sharp<beta>`, `floor<c>`, `excl<w>`, `steps<n>`, or
`base`. It needs the GPU: at most `--max-tokens` letters per forward pass, `--gpu-fraction` of the memory.
"""

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import Sources, canonical_counts, clusters, lcs_length  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

BLUE, ORANGE, GREEN, PINK, YELLOW, SKY, BLACK, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F")
KINDS = "ZVRU"
COLS = ["n", "letters", "inserted", "deleted", "wrong", "missed", "wrong_s", "missed_s", "exact_l", "exact_s"] + [
    f"{a}_{k}" for k in KINDS for a in ("cl", "x", "m")]
TUNE, REPORT = (0, 3000), (3000, 6000)  # sources for exploring / tuning and for reporting what was tuned
NOISE_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5001])
NOTES: list[str] = []  # what was missing or skipped, printed and stored in summary.json
PERFECT, EXACT = "perfect sampler", "exact rates, Euler steps"  # the reference samplers (they need the grammar)


def note(text: str):
    NOTES.append(text)
    print("NOTE:", text)


# ---- loading ---------------------------------------------------------------------------------------------------


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
    note(f"{path.name} could not be read")
    return None


class Corpus:
    """Deletions of one sampler for a set of sources: `deleted` is flat over all letters (False where the sampler
    was not run), `covered` says for which sources it was run."""

    def __init__(self, name: str, src: Sources, passes: float):
        self.name, self.passes = name, passes
        self.deleted = np.zeros(len(src.letters), dtype=bool)
        self.covered = np.zeros(len(src), dtype=bool)
        self.q0 = None
        self.exposure = None

    def add_prefix(self, src: Sources, z: dict):
        first = int(z["first"])
        n = int(src.offsets[first])
        self.deleted[:n] = z["del_step"][:n] >= 0
        self.covered[:first] = True
        for key in ("q0", "exposure"):
            if key in z:
                if getattr(self, key) is None:
                    setattr(self, key, np.full(len(src.letters), np.nan, dtype=np.float32))
                getattr(self, key)[:n] = z[key][:n]
        return self

    def add_range(self, src: Sources, z: dict):
        a, b = int(z["start"]), int(z["stop"])
        lo, hi = int(src.offsets[a]), int(src.offsets[b])
        self.deleted[lo:hi] = z["del_step"] >= 0
        self.covered[a:b] = True
        return self

    @classmethod
    def from_mask(cls, name: str, src: Sources, deleted: np.ndarray, covered: np.ndarray, passes: float):
        c = cls(name, src, passes)
        c.deleted[: len(deleted)] = deleted
        c.covered[:] = covered
        return c


# ---- per-source counts (cached) and bootstrap ------------------------------------------------------------------


def source_stats(src: Sources, corpus: Corpus, eligible, cache: Path, with_clusters: bool = True) -> dict:
    """Counts per source for the sources the corpus covers (zeros elsewhere): label level, string level
    (`canonical_counts`) and, for validation, the error letters by cluster kind (`editflows.errors.clusters`).
    `kind` is flat over the letters: the cluster kind (index into "ZVRU") of every error letter, -1 elsewhere."""
    ids = np.flatnonzero(corpus.covered)
    spans = np.concatenate([np.arange(src.offsets[i], src.offsets[i + 1]) for i in ids]) if len(ids) else np.zeros(0, int)
    key = zlib.crc32(np.packbits(corpus.deleted[spans]).tobytes() + ids.tobytes())
    file = cache / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', corpus.name)}-{key:08x}-{int(with_clusters)}.npz"
    if file.exists():
        z = load_npz(file)
        if z is not None:
            return z
    out = {c: np.zeros(len(src)) for c in COLS}
    kind = np.full(len(src.letters), -1, dtype=np.int8)
    for i in ids:
        span = src.span(i)
        letters, inserted, deleted = src.letters[span].astype(np.int64), src.inserted[span], corpus.deleted[span]
        cc = canonical_counts(letters, inserted, deleted)
        wrong, missed = int((deleted & ~inserted).sum()), int((~deleted & inserted).sum())
        row = {"n": 1, "letters": len(letters), "inserted": cc["inserted"], "deleted": cc["deleted"], "wrong": wrong,
               "missed": missed, "wrong_s": cc["wrong"], "missed_s": cc["missed"], "exact_l": int(wrong + missed == 0),
               "exact_s": cc["equal"]}
        for c, v in row.items():
            out[c][i] = v
        if with_clusters and wrong + missed:
            table = clusters(letters, inserted, deleted, eligible)
            error = (deleted != inserted)
            for k, name in enumerate(KINDS):
                rows = table[table.kind == name]
                out[f"cl_{name}"][i] = len(rows)
                out[f"x_{name}"][i] = rows.n_x.sum()
                out[f"m_{name}"][i] = rows.n_miss.sum()
                for start, end in zip(rows.start.values, rows.end.values):
                    part = kind[span.start + start : span.start + end + 1]
                    part[error[start : end + 1]] = k
    out["kind"] = kind
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file, **out)
    return out


def rates(s: dict) -> dict:
    """Scores from sums over sources (scalars or one value per bootstrap resample)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        tp, tps = s["deleted"] - s["wrong"], s["deleted"] - s["wrong_s"]
        both = s["deleted"] + s["inserted"]
        per = 1000.0 / s["letters"]
        return {
            "precision_label": tp / s["deleted"], "recall_label": tp / s["inserted"], "f1_label": 2 * tp / both,
            "precision_string": tps / s["deleted"], "recall_string": tps / s["inserted"], "f1_string": 2 * tps / both,
            "exact_label": s["exact_l"] / s["n"], "exact_string": s["exact_s"] / s["n"],
            "wrong_label_per1000": s["wrong"] * per, "missed_label_per1000": s["missed"] * per,
            "wrong_string_per1000": s["wrong_s"] * per, "missed_string_per1000": s["missed_s"] * per,
            "errors_label_per1000": (s["wrong"] + s["missed"]) * per,
            "errors_string_per1000": (s["wrong_s"] + s["missed_s"]) * per,
            "clusters_R_per1000": s["cl_R"] * per, "clusters_U_per1000": s["cl_U"] * per,
            "wrong_in_R_per1000": s["x_R"] * per, "wrong_in_U_per1000": s["x_U"] * per,
            "errors_in_RU_per1000": (s["x_R"] + s["x_U"] + s["m_R"] + s["m_U"]) * per,
            "wrong_in_Z_per1000": s["x_Z"] * per, "missed_in_Z_per1000": s["m_Z"] * per,
            "wrong_in_V_per1000": s["x_V"] * per,
        }


class Booter:
    """Bootstrap over sequences: the same resamples for every table with the same number of sources, so that
    differences between two samplers on the same sources are paired."""

    def __init__(self, n_resamples: int, seed: int = 0):
        self.B, self.seed, self._w = n_resamples, seed, {}

    def weights(self, n: int) -> np.ndarray:
        if n not in self._w:
            rng = np.random.default_rng([self.seed, n])
            self._w[n] = rng.multinomial(n, np.full(n, 1.0 / n), size=self.B).astype(np.float64)
        return self._w[n]

    def raw(self, cols: dict, sel, fn):
        names = [c for c in cols if c != "kind"]
        X = np.stack([np.asarray(cols[c], dtype=np.float64)[sel] for c in names], 1)
        point = fn(dict(zip(names, X.sum(0))))
        boots = fn(dict(zip(names, (self.weights(len(X)) @ X).T)))
        return point, boots

    def ci(self, cols: dict, sel, fn=rates) -> dict:
        point, boots = self.raw(cols, sel, fn)
        return {k: interval(point[k], boots[k]) for k in point}

    def paired(self, a: dict, b: dict, sel, fn=rates) -> dict:
        """a minus b on the same sources."""
        pa, ba = self.raw(a, sel, fn)
        pb, bb = self.raw(b, sel, fn)
        return {k: interval(pa[k] - pb[k], ba[k] - bb[k]) for k in pa}


def interval(point, boots) -> list:
    lo, hi = np.nanpercentile(np.asarray(boots, dtype=np.float64), [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


def flat(prefix: str, d: dict) -> dict:
    """{metric: [value, lo, hi]} -> columns `metric`, `metric_lo`, `metric_hi`."""
    out = {}
    for k, v in d.items():
        out[f"{prefix}{k}"], out[f"{prefix}{k}_lo"], out[f"{prefix}{k}_hi"] = v
    return out


def seq_sum(values: np.ndarray, src: Sources) -> np.ndarray:
    return np.add.reduceat(np.asarray(values, dtype=np.float64), src.offsets[:-1])


def region(src: Sources, lo: int, hi: int) -> np.ndarray:
    m = np.zeros(len(src), dtype=bool)
    m[lo:hi] = True
    return m


def letters_of(src: Sources, sources_mask: np.ndarray) -> np.ndarray:
    return np.repeat(sources_mask, np.diff(src.offsets))


# ---- label ambiguity that needs no grammar ----------------------------------------------------------------------


def swap_bound(src: Sources) -> np.ndarray:
    """Per letter: expected label errors of a sampler that knows how many letters of every run of identical
    letters are insertions but cannot know which ones (they give the same string): a run of r letters with m
    insertions costs 2 m (r - m) / r errors, spread evenly over its letters."""
    letters = src.letters
    new = np.ones(len(letters), dtype=bool)
    new[1:] = letters[1:] != letters[:-1]
    new[src.offsets[:-1]] = True
    run = np.cumsum(new) - 1
    r = np.bincount(run).astype(np.float64)
    m = np.bincount(run, weights=src.inserted.astype(np.float64))
    return (2 * m * (r - m) / r**2)[run]


@numba.njit(cache=True)
def _embedding_marginals(S, T, g):
    """P(letter of S is an insertion | S, T) when every way of reading T as a subsequence of S has the weight
    prod g[c] over the runs of c consecutive skipped letters (g = 1: all embeddings alike; g = 1 / c!: the
    extended coupling, which throws its insertions independently into the gaps of the target)."""
    n, n1 = len(S), len(T)
    m = n - n1
    out = np.ones(n)
    if n1 == 0:
        return out
    if m == 0:
        return np.zeros(n)
    f = np.zeros((n1 + 1, m + 1))  # f[j, d]: T[:j] read, its last letter at source position j + d (1-based)
    b = np.zeros((n1 + 1, m + 1))
    f[0, 0] = 1.0
    for j in range(1, n1 + 1):
        t, top = T[j - 1], 0.0
        for d in range(m + 1):
            if S[j + d - 1] != t:
                continue
            s = 0.0
            for c in range(d + 1):
                v = f[j - 1, d - c]
                if v != 0.0:
                    s += g[c] * v
            f[j, d] = s
            if s > top:
                top = s
        if top > 0:
            for d in range(m + 1):
                f[j, d] /= top
    top = 0.0
    for d in range(m + 1):
        if S[n1 + d - 1] == T[n1 - 1]:
            b[n1, d] = g[m - d]
            top = max(top, b[n1, d])
    if top > 0:
        for d in range(m + 1):
            b[n1, d] /= top
    for j in range(n1 - 1, 0, -1):
        t, top = T[j - 1], 0.0
        for d in range(m + 1):
            if S[j + d - 1] != t:
                continue
            s = 0.0
            for c in range(m - d + 1):
                v = b[j + 1, d + c]
                if v != 0.0:
                    s += g[c] * v
            b[j, d] = s
            if s > top:
                top = s
        if top > 0:
            for d in range(m + 1):
                b[j, d] /= top
    for j in range(1, n1 + 1):
        z = 0.0
        for d in range(m + 1):
            z += f[j, d] * b[j, d]
        if z > 0:
            for d in range(m + 1):
                v = f[j, d] * b[j, d]
                if v != 0.0:
                    out[j + d - 1] -= v / z
    for i in range(n):
        if out[i] < 0:
            out[i] = 0.0
    return out


def embedding_marginals(src: Sources, cache: Path, weights: str) -> np.ndarray:
    """Soft labels: the share of the equally right answers (deletion sets that give the target) in which a letter
    is deleted. `weights` = "coupling" (insertions fall independently into the gaps) or "uniform"."""
    file = cache / f"embedding-{weights}-{zlib.crc32(src.letters.tobytes() + src.inserted.tobytes()):08x}.npy"
    if file.exists():
        return np.load(file)
    longest = int(np.diff(src.offsets).max()) + 1
    g = np.ones(longest)
    if weights == "coupling":
        g = np.exp(-np.cumsum(np.log(np.maximum(np.arange(longest), 1))))  # 1 / c!
    out = np.zeros(len(src.letters), dtype=np.float32)
    for i in range(len(src)):
        span = src.span(i)
        letters, inserted = src.letters[span].astype(np.int64), src.inserted[span]
        out[span] = _embedding_marginals(letters, letters[~inserted], g)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(file, out)
    return out


# ---- calibration (isotonic, numpy only) -------------------------------------------------------------------------


def isotonic_fit(x: np.ndarray, y: np.ndarray, block: int = 200):
    """Monotone map x -> E[y | x]: means over blocks of `block` sorted points, pooled until they increase."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order].astype(np.float64), y[order].astype(np.float64)
    edges = np.arange(0, len(xs), block)
    count = np.diff(np.append(edges, len(xs))).astype(np.float64)
    bx, by = np.add.reduceat(xs, edges) / count, np.add.reduceat(ys, edges) / count
    sx, sy, sw = [], [], []
    for x_, y_, w_ in zip(bx, by, count):
        sx.append(x_ * w_); sy.append(y_ * w_); sw.append(w_)
        while len(sw) > 1 and sy[-2] / sw[-2] >= sy[-1] / sw[-1]:
            x2, y2, w2 = sx.pop(), sy.pop(), sw.pop()
            sx[-1] += x2; sy[-1] += y2; sw[-1] += w2
    sw = np.array(sw)
    return np.array(sx) / sw, np.array(sy) / sw


def isotonic_apply(model, x: np.ndarray) -> np.ndarray:
    return np.interp(x, model[0], model[1])


# ---- figures ----------------------------------------------------------------------------------------------------


def style(ax, grid: str = "y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9A9A9A")
    ax.tick_params(colors="#444444", labelsize=9)
    if grid:
        ax.grid(axis=grid, color="#E3E3E3", linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  figure", path.name)


def place_labels(fig, ax, points, labels, fontsize: float = 7.5):
    """Write a label next to every point: the first of several positions around the point that covers neither
    another point nor a label already placed (the last position if none is free)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72
    boxes = [(x - 6 * scale, y - 6 * scale, x + 6 * scale, y + 6 * scale) for x, y in ax.transData.transform(np.asarray(points, dtype=np.float64))]
    places = [(dx, dy, ha) for dy in (5, -12, 17, -24, 29, -36) for dx, ha in ((8, "left"), (-8, "right"))]
    for point, text in zip(points, labels):
        for k, (dx, dy, ha) in enumerate(places):
            label = ax.annotate(text, tuple(point), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=fontsize, color="#222222")
            e = label.get_window_extent(renderer)
            box = (e.x0, e.y0, e.x1, e.y1)
            if k == len(places) - 1 or not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in boxes):
                break
            label.remove()
        boxes.append(box)


def err(v) -> np.ndarray:
    """[value, lo, hi] rows -> the (2, n) array of matplotlib's error bars."""
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    return np.abs(np.stack([v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]]))


# ---- 1. counting -------------------------------------------------------------------------------------------------


def step_counting(src, stats: dict, boot: Booter, out: Path) -> dict:
    """`stats`: display name -> (per-source counts, covered sources)."""
    print("1. label level against string level")
    rows, summary = [], {}
    for name, (st, covered) in stats.items():
        for label, sel in (("all", covered), (f"sources {REPORT[0]}-{REPORT[1] - 1}", covered & region(src, *REPORT))):
            if not sel.any():
                continue
            ci = boot.ci(st, sel)
            total = {c: st[c][sel].sum() for c in COLS}
            vanished_w, vanished_m = total["wrong"] - total["wrong_s"], total["missed"] - total["missed_s"]
            row = {"sampler": name, "sources": label, "n_sources": int(sel.sum()), "n_letters": int(total["letters"])}
            row.update(flat("", {k: ci[k] for k in (
                "precision_label", "recall_label", "f1_label", "precision_string", "recall_string", "f1_string",
                "exact_label", "exact_string", "wrong_label_per1000", "missed_label_per1000", "wrong_string_per1000",
                "missed_string_per1000", "wrong_in_Z_per1000", "missed_in_Z_per1000")}))
            # validation with the grammar: the label errors that disappear on the strings are the clusters of kind Z
            row["label_errors_that_vanish_share"] = (vanished_w + vanished_m) / max(total["wrong"] + total["missed"], 1)
            row["vanished_explained_by_Z"] = (total["x_Z"] + total["m_Z"]) / max(vanished_w + vanished_m, 1)
            row["sources_where_vanished_equals_Z"] = float(np.mean(
                (st["wrong"] - st["wrong_s"] == st["x_Z"])[sel] & (st["missed"] - st["missed_s"] == st["m_Z"])[sel]))
            row["sources_where_vanished_is_less_than_Z"] = float(np.mean((st["wrong"] - st["wrong_s"] < st["x_Z"])[sel]))  # must be 0
            rows.append(row)
            if label == "all":
                summary[name] = {"n_sources": row["n_sources"], **{k: ci[k] for k in (
                    "precision_label", "recall_label", "f1_label", "precision_string", "recall_string", "f1_string",
                    "exact_label", "exact_string", "errors_label_per1000", "errors_string_per1000")},
                    "label_errors_that_vanish_share": row["label_errors_that_vanish_share"],
                    "vanished_explained_by_Z": row["vanished_explained_by_Z"]}
    table = pd.DataFrame(rows)
    table.to_csv(out / "counting.csv", index=False)

    shown = [n for n in stats if (table.sampler == n).any()][:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1.25]})
    groups = [("precision", "precision"), ("recall", "recall"), ("f1", "F1")]
    width = 0.2
    for ax, items, title in ((axes[0], groups, "Share of correct deletions"),
                             (axes[1], [("exact", "all letters of a\nsequence right")], "Share of sequences")):
        for s, name in enumerate(shown):
            row = table[(table.sampler == name) & (table.sources == "all")].iloc[0]
            for l, (level, alpha) in enumerate((("label", 0.45), ("string", 1.0))):
                x = np.arange(len(items)) + (2 * s + l - 1.5) * width
                vals = np.array([[row[f"{m}_{level}"], row[f"{m}_{level}_lo"], row[f"{m}_{level}_hi"]] for m, _ in items])
                ax.bar(x, vals[:, 0], width * 0.92, color=(BLUE, GREEN)[s], alpha=alpha, yerr=err(vals),
                       error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
                       label=f"{name}: {'labels (which copy was deleted)' if level == 'label' else 'strings (what is left)'}")
                for xi, v in zip(x, vals[:, 0]):
                    ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="#222222")
        ax.set_xticks(np.arange(len(items)), [t for _, t in items])
        ax.set_title(title, fontsize=10, loc="left", color="#333333")
        style(ax)
    lo = min(table[[f"{m}_label" for m, _ in groups]].min().min(), 1.0)
    axes[0].set_ylim(max(0.0, lo - 0.08), 1.0)
    axes[1].set_ylim(0, 1.0)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left", ncol=2, bbox_to_anchor=(0.0, -0.32))
    if len(shown) == 2:
        a = table[(table.sampler == shown[0]) & (table.sources == "all")].iloc[0]
        b = table[(table.sampler == shown[1]) & (table.sources == "all")].iloc[0]
        fig.suptitle(f"Counted on strings instead of labels, {a.label_errors_that_vanish_share:.0%} of the model's errors disappear "
                     f"(F1 {a.f1_label:.3f} -> {a.f1_string:.3f}) and {b.label_errors_that_vanish_share:.0%} of the perfect sampler's\n"
                     f"({b.f1_label:.3f} -> {b.f1_string:.3f}): the sampler deleted another copy of the same letter and left the same string "
                     f"(n = {int(a.n_sources)} sequences)", fontsize=11.5, x=0.02, ha="left", y=1.04)
    save(fig, out / "fig1_label_vs_string.png")
    return summary


# ---- 2. the bracket ------------------------------------------------------------------------------------------------


def step_bracket(src, model: Corpus, st_model, st_perfect, q_ref, posteriors: dict, boot: Booter, out: Path, cache: Path):
    """`q_ref`: the reference posterior (the truth of the bracket); `posteriors`: every reference posterior there is."""
    print("2. bracket for the error no sampler can remove")
    y = src.inserted.astype(np.float64)
    q0 = model.q0.astype(np.float64)
    tune_l, report_l = letters_of(src, region(src, *TUNE) & model.covered), letters_of(src, region(src, *REPORT) & model.covered)
    cal = isotonic_fit(q0[tune_l], y[tune_l])
    qc = isotonic_apply(cal, q0)
    p = embedding_marginals(src, cache, "coupling").astype(np.float64)
    p_uniform = embedding_marginals(src, cache, "uniform").astype(np.float64)
    cols = {
        "letters": np.diff(src.offsets).astype(np.float64),
        "model_errors": st_model["wrong"] + st_model["missed"],
        "model_errors_string": st_model["wrong_s"] + st_model["missed_s"],
        "upper_raw": seq_sum(2 * (q0 - y) ** 2, src),            # Brier score of the first reading
        "upper": seq_sum(2 * (qc - y) ** 2, src),                # ... of the calibrated first reading: the upper bound
        "upper_sharpness": seq_sum(2 * qc * (1 - qc), src),      # the same if the calibration holds
        "lower_swap": seq_sum(swap_bound(src), src),
        "lower_embedding": seq_sum(2 * p * (1 - p), src),
        "lower_embedding_uniform": seq_sum(2 * p_uniform * (1 - p_uniform), src),
        "upper_real": seq_sum(2 * (qc - p) ** 2, src),           # upper bound on the part that changes the string
        "model_first_reading": seq_sum(2 * q0 * (1 - q0), src),
    }
    if q_ref is not None:
        qt = q_ref.astype(np.float64)
        cols.update({
            "truth": seq_sum(2 * qt * (1 - qt), src),            # expected errors of the perfect sampler
            "truth_brier": seq_sum(2 * (qt - y) ** 2, src),      # equal to `truth` if the reference is calibrated
            "truth_real": seq_sum(2 * (qt - p) ** 2, src),
            "model_gap": seq_sum(2 * (qc - qt) ** 2, src),
            "model_gap_raw": seq_sum(2 * (q0 - qt) ** 2, src),
        })
    if st_perfect is not None:
        cols["perfect_errors"] = st_perfect["wrong"] + st_perfect["missed"]
        cols["perfect_errors_string"] = st_perfect["wrong_s"] + st_perfect["missed_s"]

    def per1000(s):
        return {k: 1000 * v / s["letters"] for k, v in s.items() if k != "letters"}

    rho = src.rho
    rows = []
    sets = [(f"sources {REPORT[0]}-{REPORT[1] - 1} (held out)", region(src, *REPORT) & model.covered),
            (f"sources {TUNE[0]}-{TUNE[1] - 1} (calibration fitted here)", region(src, *TUNE) & model.covered)]
    for a, b in zip(NOISE_EDGES[:-1], NOISE_EDGES[1:]):
        sets.append((f"held out, {a:.0%}-{min(b, 0.5):.0%} inserted", region(src, *REPORT) & model.covered & (rho >= a) & (rho < b)))
    for label, sel in sets:
        if sel.sum() < 2:
            continue
        ci = boot.ci(cols, sel, per1000)
        rows.append({"sources": label, "n_sources": int(sel.sum()), "n_letters": int(cols["letters"][sel].sum()), **flat("", ci)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "bracket.csv", index=False)

    # calibration curves of the first reading and of the reference posteriors (held-out sources)
    edges = np.array([0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.997, 1.0001])
    cal_rows = []
    for name, q in (("model first reading q0", q0), ("model first reading, calibrated", qc)) + tuple(
            (f"reference posterior {key}", q.astype(np.float64)) for key, q in posteriors.items()):
        which = np.digitize(q[report_l], edges) - 1
        n = np.bincount(which, minlength=len(edges) - 1)[: len(edges) - 1]
        cal_rows += [{"predictor": name, "bin_lo": edges[k], "bin_hi": min(edges[k + 1], 1.0), "n_letters": int(n[k]),
                      "mean_prediction": float(q[report_l][which == k].mean()) if n[k] else np.nan,
                      "share_inserted": float(y[report_l][which == k].mean()) if n[k] else np.nan} for k in range(len(edges) - 1)]
    pd.DataFrame(cal_rows).to_csv(out / "calibration.csv", index=False)

    head = table.iloc[0]
    v = lambda k: [head[k], head[f"{k}_lo"], head[f"{k}_hi"]]  # noqa: E731

    summary = {"n_sources_held_out": int(head.n_sources), "n_letters_held_out": int(head.n_letters),
               "per_1000_letters_held_out": {k: v(k) for k in cols if k != "letters"},
               "calibration_knots": int(len(cal[0]))}
    if "truth" in cols:
        summary["upper_over_truth"] = float(head.upper / head.truth)
        summary["lower_embedding_over_truth"] = float(head.lower_embedding / head.truth)
        all_l = letters_of(src, model.covered)
        qt = q_ref.astype(np.float64)
        summary["brier_identity_all_sources"] = {
            "E(q0-y)^2": float(((q0 - y) ** 2)[all_l].mean()), "E(q0-q)^2": float(((q0 - qt) ** 2)[all_l].mean()),
            "E q(1-q)": float((qt * (1 - qt))[all_l].mean()), "E(q-y)^2": float(((qt - y) ** 2)[all_l].mean()),
            "cross_term_2E(q0-q)(q-y)": float((2 * (q0 - qt) * (qt - y))[all_l].mean())}
        summary["reference_posterior_check"] = reference_check(src, posteriors, boot, out)
    return summary, table


def figure_bracket(table: pd.DataFrame, agreement: dict | None, out: Path):
    """The bracket on the held-out sources and by noise level. `agreement`: the model's row of the sample-agreement
    table (two-sample upper bound on the sources that have replicates), if there are replicates."""
    head = table.iloc[0]
    v = lambda k: [head[k], head[f"{k}_lo"], head[f"{k}_hi"]]  # noqa: E731
    truth, perfect = "truth" in table.columns, "perfect_errors" in table.columns
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    bars = [("lower bound: identical neighbours", v("lower_swap"), SKY), ("lower bound: all equally right deletion sets", v("lower_embedding"), SKY)]
    if truth:
        bars.append(("truth: perfect sampler, expected (reference posterior)", v("truth"), GREEN))
    if perfect:
        bars.append(("truth: perfect sampler, realised", v("perfect_errors"), GREEN))
    if agreement is not None:
        bars.append((f"upper bound from samples: 2 x error - disagreement of two samples\n(sources 0-{agreement['n_sources'] - 1}, {agreement['replicates']} seeds)",
                     agreement["upper_bound_irreducible_2b_minus_a_label"], YELLOW))
    bars += [("upper bound: calibrated first reading of the model", v("upper"), YELLOW), ("the model's sampler, realised", v("model_errors"), BLUE)]
    ypos = np.arange(len(bars))[::-1]
    for yy, (label, val, colour) in zip(ypos, bars):
        ax.barh(yy, val[0], 0.62, color=colour, xerr=err(val), error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
        ax.text(val[2] + 0.8, yy, f"{val[0]:.1f}", va="center", fontsize=9, color="#222222")
    ax.set_yticks(ypos, [b[0] for b in bars], fontsize=8.5)
    ax.set_xlabel("label errors (wrong + missed deletions) per 1 000 letters")
    ax.set_title(f"Held-out {head.sources.split(' (')[0]} (n = {int(head.n_sources)} sequences)", fontsize=10, loc="left", color="#333333")
    style(ax, "x")
    ax = axes[1]
    noise = table[table.sources.str.startswith("held out,")]
    x = np.arange(len(noise))
    for label, key, colour, marker in [("lower bound (equally right deletion sets)", "lower_embedding", SKY, "o")] + (
            [("truth: perfect sampler (reference posterior)", "truth", GREEN, "s")] if truth else []) + [
            ("upper bound (calibrated first reading)", "upper", YELLOW, "^"), ("the model's sampler", "model_errors", BLUE, "D")]:
        vals = noise[[key, f"{key}_lo", f"{key}_hi"]].values
        ax.errorbar(x, vals[:, 0], yerr=err(vals), color=colour, marker=marker, markersize=6, linewidth=2, capsize=2, label=label)
    ax.set_xticks(x, [s.replace("held out, ", "").replace(" inserted", "") for s in noise.sources])
    ax.set_xlabel("noise level of the source: share of inserted letters")
    ax.set_ylabel("label errors per 1 000 letters")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.set_ylim(bottom=0)
    style(ax)
    best = min(head.upper, agreement["upper_bound_irreducible_2b_minus_a_label"][0]) if agreement is not None else head.upper
    title = (f"What no sampler can remove lies between {head.lower_embedding:.0f} and {best:.0f} label errors per 1 000 letters"
             + (f" (truth: {head.truth:.0f})" if truth else "") + f"; the model's sampler makes {head.model_errors:.0f}.\n"
             + "The upper bounds are as loose as the model is far from the exact posterior: tight at low noise, loose at high noise")
    fig.suptitle(title, fontsize=11.5, x=0.02, ha="left", y=1.05)
    save(fig, out / "fig2_bracket.png")


def reference_check(src: Sources, posteriors: dict, boot: Booter, out: Path) -> dict:
    """Are the reference posteriors calibrated? A calibrated posterior expects of itself the errors it makes: the sum
    of 2 q (1 - q) equals twice its Brier score. `q_true` treats every letter as an insertion independently (the
    number of insertions between two originals is then geometric: variance / mean = 1 + mean); the coupling throws
    its insertions independently into the gaps (variance / mean = 1), which is the prior of `q_gap`."""
    y = src.inserted
    lengths = np.diff(src.offsets).astype(np.float64)
    square = np.zeros(len(src))  # sum over gaps of (insertions in the gap)^2
    for i in range(len(src)):
        mask = src.mask(i)
        bounds = np.concatenate([[-1], np.flatnonzero(~mask), [len(mask)]])
        square[i] = float(((np.diff(bounds) - 1) ** 2).sum())

    def fn(s):
        mean = s["ins"] / s["gaps"]
        return {"insertions_per_gap": mean, "variance_over_mean": (s["square"] / s["gaps"] - mean**2) / mean,
                "geometric_would_give": 1 + mean, "expected_errors_2q(1-q)_per1000": 2000 * s["sharp"] / s["letters"],
                "brier_x2_per1000": 2000 * s["brier"] / s["letters"], "expected_minus_brier_x2_per1000": 2000 * (s["sharp"] - s["brier"]) / s["letters"]}

    rho, rows, summary = src.rho, [], {}
    sets = [("all sources", np.ones(len(src), dtype=bool))] + [
        (f"{a:.0%}-{min(b, 0.5):.0%} inserted", (rho >= a) & (rho < b)) for a, b in zip(NOISE_EDGES[:-1], NOISE_EDGES[1:])]
    for key, q in posteriors.items():
        q = q.astype(np.float64)
        cols = {"gaps": lengths - seq_sum(y, src) + 1, "ins": seq_sum(y, src), "square": square, "letters": lengths,
                "brier": seq_sum((q - y) ** 2, src), "sharp": seq_sum(q * (1 - q), src)}
        for label, sel in sets:
            ci = boot.ci(cols, sel, fn)
            rows.append({"posterior": key, "sources": label, "n_sources": int(sel.sum()), **flat("", ci)})
            if label == "all sources":
                summary[key] = {"n_sources": int(sel.sum()), **ci}
    pd.DataFrame(rows).to_csv(out / "reference_posterior_check.csv", index=False)
    return summary


# ---- 3. self-consistency ----------------------------------------------------------------------------------------


Q_EDGES = np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97, 0.99, 0.999, 1.0001])


def binned_curve(src, ids_mask, q, f, boot: Booter, scale: float):
    """Per bin of the first reading q: letters, mean q, mean deletion frequency f (with intervals) and the excess
    f - scale * q in letters per 1 000 letters, also split by the label."""
    letters = letters_of(src, ids_mask)
    which = np.where(letters, np.digitize(q, Q_EDGES) - 1, -1)
    y = src.inserted
    nseq = int(ids_mask.sum())
    total = np.diff(src.offsets).astype(np.float64)[ids_mask]
    rows = []
    for k in range(len(Q_EDGES) - 1):
        m = which == k
        cols = {"n": seq_sum(m, src)[ids_mask], "q": seq_sum(np.where(m, q, 0), src)[ids_mask],
                "f": seq_sum(np.where(m, f, 0), src)[ids_mask],
                "f_orig": seq_sum(np.where(m & ~y, f, 0), src)[ids_mask], "q_orig": seq_sum(np.where(m & ~y, q, 0), src)[ids_mask],
                "f_ins": seq_sum(np.where(m & y, f, 0), src)[ids_mask], "q_ins": seq_sum(np.where(m & y, q, 0), src)[ids_mask],
                "letters": total}
        if cols["n"].sum() == 0:
            continue

        def fn(s):
            with np.errstate(divide="ignore", invalid="ignore"):
                return {"mean_q": s["q"] / s["n"], "frequency": s["f"] / s["n"],
                        "excess_per1000": 1000 * (s["f"] - scale * s["q"]) / s["letters"],
                        "excess_originals_per1000": 1000 * (s["f_orig"] - scale * s["q_orig"]) / s["letters"],
                        "excess_inserted_per1000": 1000 * (s["f_ins"] - scale * s["q_ins"]) / s["letters"]}

        rows.append({"bin_lo": Q_EDGES[k], "bin_hi": min(Q_EDGES[k + 1], 1.0), "n_letters": int(cols["n"].sum()),
                     "n_sources": nseq, **flat("", boot.ci(cols, slice(None), fn))})
    return pd.DataFrame(rows)


def excess_summary(src, ids_mask, q, f, boot: Booter, scale: float) -> dict:
    """Deletions beyond the first reading, per 1 000 letters: letters the first reading called original (q < 0.5)
    and the others, in total and by label."""
    letters = letters_of(src, ids_mask)
    y = src.inserted
    low = letters & (q < 0.5)
    high = letters & (q >= 0.5)
    d = f - scale * q
    cols = {"letters": np.diff(src.offsets).astype(np.float64)[ids_mask]}
    for name, m in (("low", low), ("high", high), ("low_originals", low & ~y), ("low_inserted", low & y),
                    ("high_originals", high & ~y), ("high_inserted", high & y), ("originals", letters & ~y), ("inserted", letters & y)):
        cols[name] = seq_sum(np.where(m, d, 0), src)[ids_mask]
    cols["wrong"] = seq_sum(np.where(letters & ~y, f, 0), src)[ids_mask]
    cols["missed"] = seq_sum(np.where(letters & y, 1 - f, 0), src)[ids_mask]
    cols["wrong_predicted"] = seq_sum(np.where(letters & ~y, scale * q, 0), src)[ids_mask]
    cols["missed_predicted"] = seq_sum(np.where(letters & y, 1 - scale * q, 0), src)[ids_mask]
    return boot.ci(cols, slice(None), lambda s: {k: 1000 * v / s["letters"] for k, v in s.items() if k != "letters"})


def sample_agreement(src, R: np.ndarray, first: int, q: np.ndarray, st_mean: dict, boot: Booter) -> dict:
    """Per 1 000 letters: (a) disagreement between two samples of one source, (b) error of a sample against the
    labels, both counted on labels and on strings, and (d) what the first reading q promises, 2 q (1 - q). For a
    perfect sampler the truth is one more sample, so (a) = (b) = (d). In general, with f the deletion frequency of a
    letter and q* the true posterior, E(b) = (a) / 2 + (irreducible) / 2 + sum (f - q*)^2, so 2 (b) - (a) is an upper
    bound on the irreducible error (it is twice the Brier score of f)."""
    K, n = R.shape
    ids = region(src, 0, first)
    y, f = src.inserted[:n], R.mean(0)
    pad = lambda v: np.concatenate([v, np.zeros(len(src.letters) - n)])  # noqa: E731
    pairs = [(k, k + 1) for k in range(0, K - 1, 2)]
    between = np.zeros(len(src))
    for i in range(first):
        span = src.span(i)
        letters = src.letters[span]
        for a, b in pairs:
            o1, o2 = letters[~R[a, span]], letters[~R[b, span]]
            between[i] += (len(o1) + len(o2) - 2 * lcs_length(o1, o2)) / len(pairs)
    cols = {"letters": np.diff(src.offsets).astype(np.float64)[ids],
            "a": seq_sum(pad(2 * K / (K - 1) * f * (1 - f)), src)[ids], "b": seq_sum(pad(np.where(y, 1 - f, f)), src)[ids],
            "d": seq_sum(pad(2 * q[:n] * (1 - q[:n])), src)[ids], "a_s": between[ids],
            "b_s": (st_mean["wrong_s"] + st_mean["missed_s"])[ids], "u": seq_sum(pad(2 * (q[:n] - y) ** 2), src)[ids]}

    def fn(s):
        per = 1000 / s["letters"]
        return {"a_between_two_samples_label": s["a"] * per, "b_sample_against_truth_label": s["b"] * per,
                "d_first_reading_2q(1-q)": s["d"] * per, "a_between_two_samples_string": s["a_s"] * per,
                "b_sample_against_truth_string": s["b_s"] * per, "b_minus_a_label": (s["b"] - s["a"]) * per,
                "a_minus_d_label": (s["a"] - s["d"]) * per, "b_minus_a_string": (s["b_s"] - s["a_s"]) * per,
                "upper_bound_irreducible_2b_minus_a_label": (2 * s["b"] - s["a"]) * per,
                "upper_bound_irreducible_first_reading_label": s["u"] * per}

    return boot.ci(cols, slice(None), fn)


def replicate_stack(src, folder: Path, stem: str):
    """(replicates, letters) deletions of the sources all replicates `<stem><k>.npz` of the folder cover."""
    files = sorted((f for f in folder.glob(f"{stem}*.npz") if re.fullmatch(stem + r"\d+", f.stem)), key=lambda f: int(f.stem[len(stem):]))
    data = [z for z in (load_npz(f) for f in files) if z is not None]
    if not data:
        return None, 0, None
    first = min(int(z["first"]) for z in data)
    n = int(src.offsets[first])
    return np.stack([z["del_step"][:n] >= 0 for z in data]), first, data


def step_consistency(src, run_dir: Path, study: Path, model: Corpus, own: dict, stems: dict, kappa1: float, eligible,
                     boot: Booter, out: Path, cache: Path):
    """`own`: reference sampler -> the posterior it is built on (its own first reading); `stems`: reference sampler ->
    file stem of its replicates."""
    print("3. self-consistency of the sampler")
    summary, reps = {}, {}
    D, first, data = replicate_stack(src, run_dir, "rep")
    if D is None:
        note("no replicates rep<k>.npz: the self-consistency curve uses the summed deletion probability of m100 only")
    else:
        print(f"  {len(D)} model replicates of the first {first} sources")
        n = D.shape[1]
        ids = region(src, 0, first)
        f = np.zeros(len(src.letters)); f[:n] = D.mean(0)
        q = np.zeros(len(src.letters)); q[:n] = data[0]["q0"][:n]
        summary["first_reading_identical_across_replicates"] = bool(max(float(np.abs(z["q0"][:n] - data[0]["q0"][:n]).max()) for z in data) < 1e-3)
        curve = binned_curve(src, ids, q, f, boot, kappa1)
        curve.insert(0, "sampler", f"model, {len(D)} seed{'s' * (len(D) > 1)}"); curves = [curve]
        summary["model"] = {"n_sources": int(first), "n_letters": n, "replicates": int(len(D)), **excess_summary(src, ids, q, f, boot, kappa1)}
        reps["model"] = (D, first)
    # the same curve from one run: the summed deletion probability of a letter has the same expectation as its deletion
    if model.exposure is not None:
        ids = model.covered
        curve = binned_curve(src, ids, np.nan_to_num(model.q0.astype(np.float64)), np.nan_to_num(model.exposure.astype(np.float64)), boot, kappa1)
        curve.insert(0, "sampler", "model, one run: summed deletion probability")
        curves = (curves if D is not None else []) + [curve]
        summary["model_one_run_exposure"] = {"n_sources": int(ids.sum()), **excess_summary(
            src, ids, np.nan_to_num(model.q0.astype(np.float64)), np.nan_to_num(model.exposure.astype(np.float64)), boot, kappa1)}
        summary["model_one_run_deleted"] = {"n_sources": int(ids.sum()), **excess_summary(
            src, ids, np.nan_to_num(model.q0.astype(np.float64)), model.deleted.astype(np.float64), boot, kappa1)}
    for name, scale in ((PERFECT, 1.0), (EXACT, kappa1)):
        stem, q_own = stems.get(name), own.get(name)
        if stem is None or q_own is None:
            continue
        q_own = q_own.astype(np.float64)
        R, first_r, _ = replicate_stack(src, study, stem)
        if R is None:
            note(f"no replicates {stem}<k>.npz: reference curve '{name}' left out")
            continue
        n = R.shape[1]
        ids = region(src, 0, first_r)
        f = np.zeros(len(src.letters)); f[:n] = R.mean(0)
        curve = binned_curve(src, ids, q_own, f, boot, scale)
        curve.insert(0, "sampler", f"{name}, {len(R)} seeds"); curves.append(curve)
        summary[name] = {"n_sources": int(first_r), "replicates": int(len(R)), **excess_summary(src, ids, q_own, f, boot, scale)}
        reps[name] = (R, first_r)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out / "self_consistency.csv", index=False)

    # validation with the truth: surplus of wrong deletions over the perfect sampler, mass of the clusters R and U
    rep_stats, by_seed = {}, []
    for name, (R, first_r) in reps.items():
        acc = None
        for k in range(len(R)):
            st = source_stats(src, Corpus.from_mask(f"rep-{name}-{k}", src, R[k], region(src, 0, first_r), 0), eligible, cache)
            acc = {c: st[c].copy() for c in COLS} if acc is None else {c: acc[c] + st[c] for c in COLS}
            r = rates({c: st[c][:first_r].sum() for c in COLS})
            by_seed.append({"sampler": name, "replicate": k, "n_sources": int(first_r), **{m: float(r[m]) for m in (
                "f1_label", "f1_string", "precision_string", "recall_string", "exact_string", "wrong_label_per1000", "missed_label_per1000")}})
        rep_stats[name] = ({c: v / len(R) for c, v in acc.items()}, first_r)
    if by_seed:  # how much one seed of a sampler differs from another on the same sources
        by_seed = pd.DataFrame(by_seed)
        by_seed.to_csv(out / "scores_by_seed.csv", index=False)
        summary["f1_string_across_seeds"] = {name: {"replicates": int(len(part)), "mean": float(part.f1_string.mean()), "min": float(part.f1_string.min()),
                                                    "max": float(part.f1_string.max())} for name, part in by_seed.groupby("sampler", sort=False)}
    # a sample against another sample and against the truth: for a perfect sampler the two have the same statistics
    rows = []
    for name, (R, first_r) in reps.items():
        if len(R) < 2:
            continue
        q = np.nan_to_num(model.q0.astype(np.float64)) if name == "model" else own[name].astype(np.float64)
        ci = sample_agreement(src, R, first_r, q, rep_stats[name][0], boot)
        rows.append({"sampler": name, "n_sources": int(first_r), "replicates": int(len(R)), **flat("", ci)})
        summary.setdefault("sample_agreement", {})[name] = {"n_sources": int(first_r), "replicates": int(len(R)), **ci}
    if rows:
        pd.DataFrame(rows).to_csv(out / "sample_agreement.csv", index=False)
    if "model" in rep_stats:
        st_m, first_m = rep_stats["model"]
        sel = region(src, 0, first_m)
        summary["model"]["scores_mean_over_replicates"] = {k: v for k, v in boot.ci(st_m, sel).items() if "per1000" in k}
        if PERFECT in rep_stats:
            st_f, first_f = rep_stats[PERFECT]
            sel = region(src, 0, min(first_m, first_f))
            summary[PERFECT]["scores_mean_over_replicates"] = {k: v for k, v in boot.ci(st_f, sel).items() if "per1000" in k}
            diff = boot.paired(st_m, st_f, sel)
            summary["model_minus_perfect_sampler"] = {"n_sources": int(sel.sum()), **{k: diff[k] for k in (
                "wrong_label_per1000", "missed_label_per1000", "wrong_string_per1000", "missed_string_per1000",
                "wrong_in_R_per1000", "wrong_in_U_per1000", "errors_in_RU_per1000", "wrong_in_Z_per1000", "wrong_in_V_per1000")}}

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    grid = np.logspace(-4.6, 0, 50)
    ax.plot(grid, kappa1 * grid, color=BLACK, linewidth=1, linestyle="--")
    ax.text(2.6e-5, 0.75, "dashed line: a consistent flow,\nfrequency = kappa(1) x first reading", fontsize=8.5, color="#222222", va="top")
    for name, part in curves.groupby("sampler", sort=False):
        colour, marker = (BLUE, "s" if "summed" in name else "o") if name.startswith("model") else ((GREEN, "D") if name.startswith(PERFECT) else (ORANGE, "^"))
        ok = part.frequency > 0
        ax.errorbar(part.mean_q[ok], part.frequency[ok], yerr=err(part.loc[ok, ["frequency", "frequency_lo", "frequency_hi"]].values),
                    color=colour, marker=marker, markersize=5 if "summed" in name else 5.5, linewidth=1.2 if "summed" in name else 1.8, capsize=2,
                    linestyle="--" if "summed" in name else "-", markerfacecolor="white" if "summed" in name else colour,
                    label=f"{name} (n = {int(part.n_sources.iloc[0])})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2e-5, 1.3); ax.set_ylim(2e-5, 1.3)
    ax.set_xlabel("probability 'is an insertion' at the first reading, t = 0\n(model: its own reading; exact samplers: the posterior they are built on)")
    ax.set_ylabel("share of sampler seeds in which the letter was deleted")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.2))
    style(ax, "both")
    ax = axes[1]
    part = curves[curves.sampler.str.startswith("model") & ~curves.sampler.str.contains("summed")] if D is not None else curves[curves.sampler.str.contains("summed")]
    x = np.arange(len(part))
    ax.bar(x - 0.2, part.excess_originals_per1000, 0.4, color=PINK, yerr=err(part[["excess_originals_per1000", "excess_originals_per1000_lo", "excess_originals_per1000_hi"]].values),
           error_kw={"elinewidth": 0.8, "ecolor": "#333333"}, label="original letters: extra wrong deletions")
    ax.bar(x + 0.2, part.excess_inserted_per1000, 0.4, color=SKY, yerr=err(part[["excess_inserted_per1000", "excess_inserted_per1000_lo", "excess_inserted_per1000_hi"]].values),
           error_kw={"elinewidth": 0.8, "ecolor": "#333333"}, label="inserted letters: extra correct deletions")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, [f"{a:g}-{b:g}" for a, b in zip(part.bin_lo, part.bin_hi)], rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("probability 'is an insertion' at the first reading (bins)")
    ax.set_ylabel("deletions beyond kappa(1) x first reading,\nper 1 000 letters of the sources")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    style(ax)
    key = "model" if "model" in summary else "model_one_run_exposure"
    s = summary[key]
    others = "".join(f"; {name.split(',')[0]}: {summary[name]['low_originals'][0]:+.1f}" for name in reps if name != "model")
    fig.suptitle(f"The sampler deletes letters its own first reading called original: among letters read as 'original' (first reading < 0.5) it makes "
                 f"{s['low'][0]:+.1f} deletions\nper 1 000 letters beyond what the first reading asked for; {s['low_originals'][0]:+.1f} of them are wrong deletions"
                 + (f" (same count for the {others[2:]})" if others else ""), fontsize=11.5, x=0.02, ha="left", y=1.04)
    save(fig, out / "fig3_self_consistency.png")
    return summary, reps


# ---- 4. sampler variants -------------------------------------------------------------------------------------------

REFERENCES = ("reference", "perfect", "exact")  # tags of what needs the grammar
SCORES = ["precision_label", "recall_label", "f1_label", "precision_string", "recall_string", "f1_string", "exact_label",
          "exact_string", "wrong_label_per1000", "missed_label_per1000", "wrong_string_per1000", "missed_string_per1000",
          "errors_label_per1000", "errors_string_per1000", "clusters_R_per1000", "clusters_U_per1000",
          "wrong_in_R_per1000", "wrong_in_U_per1000", "errors_in_RU_per1000"]
PAIRED = ["f1_label", "f1_string", "precision_string", "recall_string", "exact_string", "wrong_label_per1000",
          "missed_label_per1000", "errors_string_per1000", "errors_in_RU_per1000"]
GIVES_UP = {
    "steps": "nothing but compute (same sampler, finer or coarser time grid)",
    "excl": "independent deletions within a step (still a sampler)",
    "sharp": "posterior sampling: the rates are pushed towards the most likely answer, less diversity",
    "floor": "posterior sampling: letters whose current reading is below the floor are never deleted",
    "anchor": "recall on insertions the first reading overlooked (they can never be deleted); diversity only among the uncertain letters",
    "one-shot": "sampling: one deterministic answer per source, letters decided separately (no diversity, no joint consistency)",
    "one-shot-runs": "sampling: one deterministic answer per source (no diversity); consistent only within runs of identical letters",
    "independent": "joint consistency: every letter is drawn separately from the first reading",
    "veto": "recall on insertions the first reading overlooked (as anchoring); deletions that followed a vetoed one stay",
    "consensus": "diversity (one answer per source) and compute (all replicates are needed)",
    "reference": "needs the grammar (validation only)",
}


def describe(name: str) -> tuple[str, str, str]:
    """Readable name of a variant tag, a short name for figures and what the variant gives up."""
    words = {"anchor": ("anchoring: letters with first reading < {} are never deleted", "anchor {}"),
             "sharp": ("sharpened rates, q -> q^b / (q^b + (1 - q)^b) with b = {}", "sharpen {}"),
             "floor": ("floor: no deletion while the current reading is below {}", "floor {}"),
             "excl": ("exclusion: one deletion per +-{} letters and step", "exclusion {}"), "steps": ("{} Euler steps", "{} steps")}
    parts, short, lost = [], [], []
    for part in name.split("+"):
        m = re.fullmatch(r"(anchor|sharp|floor|excl|steps)([0-9.]+)", part)
        if m is None:
            return name, name, ""
        parts.append(words[m.group(1)][0].format(m.group(2)))
        short.append(words[m.group(1)][1].format(m.group(2)))
        lost.append(GIVES_UP[m.group(1)])
    return "; ".join(parts), " + ".join(short), "; ".join(lost)


def decide(src, q: np.ndarray, thr: float, together: bool) -> np.ndarray:
    """Delete iff q > thr. With `together`, a run of identical neighbours is decided as one unit: its letters are
    interchangeable (deleting any of them leaves the same string), so the number to delete is the rounded sum of
    their q, ceil(sum q - thr), taken from the largest q. For a single letter both rules agree."""
    if not together:
        return q > thr
    letters = src.letters
    new = np.ones(len(letters), dtype=bool)
    new[1:] = letters[1:] != letters[:-1]
    new[src.offsets[:-1]] = True
    run = np.cumsum(new) - 1
    k = np.ceil(np.bincount(run, weights=q) - thr - 1e-9)
    order = np.lexsort((-q, run))  # runs are contiguous, so a run keeps its index range after sorting
    rank = np.empty(len(q), dtype=np.int64)
    rank[order] = np.arange(len(q)) - np.flatnonzero(new)[run[order]]
    return rank < k[run]


def best_threshold(src, q, covered, sel, eligible, cache, grid, name, together) -> tuple[float, pd.DataFrame]:
    """Sweep of the decision `decide(q, threshold)` on the sources `sel`; the threshold with the best F1 on strings."""
    rows = []
    for thr in grid:
        c = Corpus.from_mask(f"{name}-{int(together)}-{thr:.3f}", src, decide(src, q, thr, together), covered & sel, 1)
        st = source_stats(src, c, eligible, cache, with_clusters=False)
        r = rates({k: st[k][sel].sum() for k in COLS})
        rows.append({"runs_decided_together": together, "threshold": thr, "n_sources": int(sel.sum()), **{k: float(r[k]) for k in SCORES[:14]}})
    table = pd.DataFrame(rows)
    return float(table.threshold[table.f1_string.idxmax()]), table


def leading(covered: np.ndarray) -> int:
    """Number of sources from the first one on that a corpus covers."""
    return int(np.argmin(covered)) if not covered.all() else len(covered)


def step_variants(src, errors: Path, model: Corpus, st_model, refs: dict, st_refs: dict, q_ref, reps: dict, kappa1,
                  eligible, boot: Booter, out: Path, cache: Path) -> dict:
    print("4. sampler variants")
    scratch = out / "scratch"
    variants: dict[str, Corpus] = {}
    for file in sorted(errors.glob("m*.npz"), key=lambda f: (len(f.stem), f.stem)):  # other step counts
        m = re.fullmatch(r"m(\d+)", file.stem)
        z = load_npz(file) if m is not None and file.stem != "m100" else None
        if z is not None and int(z["steps"]) != model.passes:
            variants[f"steps{int(z['steps'])}"] = Corpus(f"steps{int(z['steps'])}", src, int(z["steps"])).add_prefix(src, z)
    if not variants:
        note("no other step counts (m50 / m200 / m400.npz) were there")
    for file in sorted(errors.glob("v-*.npz")):
        z = load_npz(file)
        if z is not None:
            variants[file.stem[2:]] = Corpus(file.stem[2:], src, int(z["steps"])).add_prefix(src, z)
    for file in sorted(scratch.glob("s-*.npz")):
        m, z = re.fullmatch(r"s-(.+)-(\d+)-(\d+)", file.stem), load_npz(file)
        if m is not None and z is not None:
            variants.setdefault(m.group(1), Corpus(m.group(1), src, int(z["steps"]))).add_range(src, z)
    if not any(re.match(r"anchor|sharp|floor|excl", name) for name in variants):
        note("no GPU sampler variants (v-*.npz, s-*.npz) were there: only step counts and the CPU variants are compared")

    entries = [{"variant": f"model: standard sampler ({model.passes} Euler steps)", "short": "standard sampler", "corpus": model,
                "stats": st_model, "tag": "baseline", "gives_up": ""}]

    def add(variant, short, corpus, tag, lost):
        entries.append({"variant": variant, "short": short, "corpus": corpus, "stats": source_stats(src, corpus, eligible, cache),
                        "tag": tag, "gives_up": lost})

    for name, corpus in variants.items():
        label, short, lost = describe(name)
        add(label, short, corpus, name, lost)

    q0 = np.nan_to_num(model.q0.astype(np.float64))
    tune, report = region(src, *TUNE) & model.covered, region(src, *REPORT) & model.covered
    grid = np.round(np.arange(0.3, 0.701, 0.05), 3)
    sweeps, chosen = [], {}
    modes = ((False, "every letter separately", "each letter"), (True, "identical neighbours together", "neighbours together"))
    for together, how, brief in modes:
        thr, sweep = best_threshold(src, q0, model.covered, tune, eligible, cache, grid, "oneshot", together)
        sweep.insert(0, "decision", "first reading of the model")
        sweeps.append(sweep)
        chosen[how] = thr
        add(f"one forward pass, {how}: delete iff first reading > {thr:g}", f"one pass, {brief} (> {thr:g})",
            Corpus.from_mask(f"oneshot-{int(together)}", src, decide(src, q0, thr, together), model.covered, 1),
            "one-shot" + "-runs" * together, GIVES_UP["one-shot" + "-runs" * together])
    u = np.random.default_rng(12345).random(len(q0))
    add("one forward pass: independent draw from the first reading", "one pass, independent draw",
        Corpus.from_mask("independent", src, u < kappa1 * q0, model.covered, 1), "independent", GIVES_UP["independent"])
    # veto: take the standard sample and undo the deletions of letters whose first reading was below c. The same
    # letters as anchoring protects, but afterwards: what anchoring gains beyond the veto is damage that a wrong
    # deletion did to its neighbourhood while the letter was missing.
    rows = []
    for c in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
        st = source_stats(src, Corpus.from_mask(f"veto-{c}", src, model.deleted & (q0 >= c), model.covered & tune, model.passes), eligible, cache, with_clusters=False)
        r = rates({k: st[k][tune].sum() for k in COLS})
        rows.append({"undo_deletions_with_first_reading_below": c, "n_sources": int(tune.sum()), **{k: float(r[k]) for k in SCORES[:14]}})
    veto_sweep = pd.DataFrame(rows)
    veto_sweep.to_csv(out / "veto_sweep.csv", index=False)
    chosen["veto"] = float(veto_sweep.undo_deletions_with_first_reading_below[veto_sweep.f1_string.idxmax()])
    anchors = {parse_variant(name)["anchor"] for name in variants if re.fullmatch(r"anchor[0-9.]+", name)}
    for c in sorted(anchors | {chosen["veto"]}):
        add(f"veto: standard sample, deletions of letters with first reading < {c:g} undone", f"veto {c:g}",
            Corpus.from_mask(f"veto-{c}", src, model.deleted & (q0 >= c), model.covered, model.passes), "veto", GIVES_UP["veto"])
    consensus_theta, rep_n = None, 0
    if "model" in reps and len(reps["model"][0]) >= 4:
        D, rep_n = reps["model"]
        covered = region(src, 0, rep_n)
        full = lambda share: np.concatenate([share, np.zeros(len(q0) - len(share))])  # noqa: E731
        consensus_theta, sweep = best_threshold(src, full(D.mean(0)), covered, region(src, 0, rep_n // 2), eligible, cache, grid, f"consensus{len(D)}", True)
        sweep.to_csv(out / "consensus_sweep.csv", index=False)
        add(f"consensus of {len(D)} samples, every letter separately: deleted in more than half", f"consensus of {len(D)}, each letter",
            Corpus.from_mask(f"consensus-{len(D)}-0", src, decide(src, full(D.mean(0)), 0.5, False), covered, len(D) * model.passes), "consensus", GIVES_UP["consensus"])
        for k in sorted({4, 8, 16, len(D)}):
            if k <= len(D):
                add(f"consensus of {k} samples, identical neighbours together: deleted in more than half", f"consensus of {k}, neighbours together",
                    Corpus.from_mask(f"consensus-{k}-1", src, decide(src, full(D[:k].mean(0)), 0.5, True), covered, k * model.passes), "consensus", GIVES_UP["consensus"])
    else:
        note("fewer than 4 model replicates: the consensus variant is left out")
    if q_ref is not None:  # validation only: the same decisions with the reference posterior
        for together, how, brief in modes:
            thr, sweep = best_threshold(src, q_ref.astype(np.float64), model.covered, tune, eligible, cache, grid, "oneshot-true", together)
            sweep.insert(0, "decision", "reference posterior (validation)")
            sweeps.append(sweep)
            add(f"[needs grammar] reference posterior, {how}: delete iff > {thr:g}", f"reference posterior, {brief} *",
                Corpus.from_mask(f"oneshot-true-{int(together)}", src, decide(src, q_ref.astype(np.float64), thr, together), model.covered, 0),
                "reference", GIVES_UP["reference"])
    pd.concat(sweeps, ignore_index=True).to_csv(out / "oneshot_sweep.csv", index=False)
    for name, corpus in refs.items():
        entries.append({"variant": f"[needs grammar] {name}", "short": f"{name} *", "corpus": corpus, "stats": st_refs[name],
                        "tag": "perfect" if name == PERFECT else "exact", "gives_up": GIVES_UP["reference"]})

    def table_for(sets) -> pd.DataFrame:
        """Scores of every variant on a set of sources and its paired difference to the standard sampler there.
        `tags`: only these kinds of variants; `partial`: a variant that covers only part of the set is compared on
        that part (its `sources` column says which)."""
        rows = []
        for set_name, sel_set, tags, partial in sets:
            for e in entries:
                sel = sel_set & e["corpus"].covered
                if (tags is not None and e["tag"] not in tags) or sel.sum() < 2 or (sel.sum() < sel_set.sum() and not partial):
                    continue
                ids = np.flatnonzero(sel)
                ci, diff = boot.ci(e["stats"], sel), boot.paired(e["stats"], st_model, sel)
                rows.append({"sources": set_name, "first_source": int(ids[0]), "last_source": int(ids[-1]), "variant": e["variant"],
                             "short": e["short"], "tag": e["tag"], "n_sources": int(sel.sum()), "n_letters": int(e["stats"]["letters"][sel].sum()),
                             "forward_passes": e["corpus"].passes, **flat("", {k: ci[k] for k in SCORES}),
                             **flat("minus_standard_", {k: diff[k] for k in PAIRED}), "gives_up": e["gives_up"]})
        return pd.DataFrame(rows)

    # the sources every sampler variant was run on (a prefix of the tuning half); the replicates may cover another prefix
    sampled = [leading(c.covered) for c in variants.values() if leading(c.covered) > 0]
    tune_n = min(min(sampled) if sampled else 1000, TUNE[1], leading(model.covered))
    sets = [(f"sources 0-{tune_n - 1}", region(src, 0, tune_n), None, False)]
    if rep_n and rep_n != tune_n:
        sets.append((f"sources 0-{rep_n - 1}", region(src, 0, rep_n), {"consensus", "baseline", "one-shot", "one-shot-runs", *REFERENCES}, False))
    tuning = table_for(sets)
    tuning.to_csv(out / "variants_tuning.csv", index=False)
    held = table_for([(f"sources {REPORT[0]}-{REPORT[1] - 1}", report, None, True)])
    held.to_csv(out / "variants_heldout.csv", index=False)
    # confounder: does a gain hold at every noise level of the source?
    rho = src.rho
    by_noise = table_for([(f"held out, {a:.0%}-{min(b, 0.5):.0%} inserted", report & (rho >= a) & (rho < b), None, True)
                          for a, b in zip(NOISE_EDGES[:-1], NOISE_EDGES[1:])])
    by_noise.to_csv(out / "variants_heldout_by_noise.csv", index=False)
    lengths = np.diff(src.offsets)
    cuts = np.quantile(lengths[report], [0, 1 / 3, 2 / 3, 1]) if report.any() else np.zeros(4)
    by_length = table_for([(f"held out, {int(a)}-{int(b)} letters", report & (lengths >= a) & ((lengths < b) | (b == cuts[-1])), None, True)
                           for a, b in zip(cuts[:-1], cuts[1:])])
    by_length.to_csv(out / "variants_heldout_by_length.csv", index=False)
    # protecting a letter during the run against undoing its deletion afterwards, on the same sources
    rows = []
    by_tag = {e["tag"] if e["tag"] != "veto" else e["short"]: e for e in entries}
    for c in sorted(anchors):
        a, v = by_tag.get(f"anchor{c:g}"), by_tag.get(f"veto {c:g}")
        if a is None or v is None:
            continue
        for set_name, sel_set in ((sets[0][0], sets[0][1]), (f"sources {REPORT[0]}-{REPORT[1] - 1}", report)):
            sel = sel_set & a["corpus"].covered & v["corpus"].covered
            if sel.sum() > 1:
                diff = boot.paired(a["stats"], v["stats"], sel)
                rows.append({"threshold": c, "sources": set_name, "n_sources": int(sel.sum()), **flat("anchor_minus_veto_", {k: diff[k] for k in PAIRED + [
                    "wrong_in_R_per1000", "wrong_in_U_per1000"]})})
    anchor_vs_veto = pd.DataFrame(rows)
    anchor_vs_veto.to_csv(out / "anchor_vs_veto.csv", index=False)

    # ---- figures
    groups = {"baseline": ("trained model, standard sampler", BLUE, "s"), "perfect": ("perfect sampler *", GREEN, "s"),
              "exact": ("exact rates *", ORANGE, "s"), "reference": ("decisions with the reference posterior *", GREY, "s"),
              "one-shot": ("one forward pass", YELLOW, "o"), "one-shot-runs": ("one forward pass", YELLOW, "o"),
              "independent": ("one forward pass", YELLOW, "o"), "consensus": ("consensus of samples", BLACK, "D"),
              "veto": ("standard sample, edited afterwards", SKY, "v")}

    def group(tag: str):
        return groups.get(tag, ("other step counts", PINK, "^") if re.fullmatch(r"steps\d+", tag) else ("changed sampler", PINK, "o"))

    part = tuning[tuning.sources == sets[0][0]]
    base = part[part.tag == "baseline"].iloc[0]
    changed = part[~part.tag.isin(["baseline", "one-shot", "one-shot-runs", "independent", "consensus", "veto", *REFERENCES]) & ~part.tag.str.fullmatch(r"steps\d+")]
    single = part[part.tag.isin(["one-shot", "one-shot-runs"])]
    top = changed.loc[changed.f1_string.idxmax()] if len(changed) else None
    top1 = single.loc[single.f1_string.idxmax()]
    message = (f"best change of the sampler: {top.short} (F1 on strings {base.f1_string:.3f} -> {top.f1_string:.3f}); " if top is not None else "") + \
        f"best single forward pass: {top1.short} ({top1.f1_string:.3f})"

    fig, ax = plt.subplots(figsize=(12, 8.2))
    lo = min(part.precision_string.min(), part.recall_string.min()) - 0.012
    for f1 in np.arange(0.80, 0.99, 0.02):
        pr = np.linspace(0.4, 1.0, 600)
        with np.errstate(divide="ignore", invalid="ignore"):
            rc = f1 * pr / (2 * pr - f1)
        ok = (rc > lo) & (rc <= 1) & (pr > lo)
        if ok.sum() > 2:
            ax.plot(rc[ok], pr[ok], color="#D5D5D5", linewidth=0.8, zorder=0)
            if f1 / (2 - f1) > lo:
                ax.text(0.9995, f1 / (2 - f1) + 0.001, f"F1 {f1:.2f}", fontsize=7.5, color="#8A8A8A", ha="right", va="bottom")
    seen = set()
    for j, (_, row) in enumerate(part.iterrows()):
        name, colour, marker = group(row.tag)
        ax.errorbar(row.recall_string, row.precision_string, xerr=err([[row.recall_string, row.recall_string_lo, row.recall_string_hi]]),
                    yerr=err([[row.precision_string, row.precision_string_lo, row.precision_string_hi]]), color=colour, marker=marker,
                    markersize=7.5, linewidth=0, elinewidth=0.9, markeredgecolor="white", markeredgewidth=0.8, label=None if name in seen else name, zorder=3)
        seen.add(name)
    ax.set_xlabel("recall: share of the inserted letters that are gone (counted on strings)")
    ax.set_ylabel("precision: share of the deletions that were right (counted on strings)")
    ax.set_xlim(lo, 1.0); ax.set_ylim(lo, 1.0)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    style(ax, "both")
    place_labels(fig, ax, part[["recall_string", "precision_string"]].values, list(part.short))
    ax.set_title(f"Sampler variants on {sets[0][0]} (n = {int(base.n_sources)} sequences, 95 % intervals):\n{message}", fontsize=11, loc="left")
    save(fig, out / "fig4_variants_precision_recall.png")

    show = (held if len(held) else part).sort_values("f1_string")
    fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(show) + 1.9))
    ypos = np.arange(len(show))
    for level, alpha, off in (("label", 0.4, 0.19), ("string", 1.0, -0.19)):
        vals = show[[f"f1_{level}", f"f1_{level}_lo", f"f1_{level}_hi"]].values
        ax.barh(ypos + off, vals[:, 0], 0.36, color=[group(t)[1] for t in show.tag], alpha=alpha, xerr=err(vals), error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
        for yy, v in zip(ypos + off, vals):
            ax.text(v[2] + 0.002, yy, f"{v[0]:.3f}", va="center", fontsize=7.5, color="#222222")
    same_n = show.n_sources.nunique() == 1
    ax.set_yticks(ypos, [f"{s}  ({int(p) if p else 'no'} passes{'' if same_n else f', n = {n}'})" for s, p, n in zip(show.short, show.forward_passes, show.n_sources)], fontsize=8.5)
    ax.set_xlim(max(0.5, show.f1_label.min() - 0.04), 1.0)
    ax.set_xlabel("F1 of the deletions. Pale bar: counted on labels; full bar: counted on strings (95 % intervals, bootstrap over sequences)")
    style(ax, "x")
    b = show[show.tag == "baseline"].iloc[0]
    cand = show[~show.tag.isin(["baseline", *REFERENCES])]
    t = cand.loc[cand.f1_string.idxmax()]
    ax.set_title(f"{'Held-out ' + held.sources.iloc[0] if len(held) else sets[0][0]} (n = {int(b.n_sources)}): '{t.short}' lifts F1 on strings from {b.f1_string:.3f} to {t.f1_string:.3f}\n"
                 f"(paired difference {t.minus_standard_f1_string:+.3f}, interval {t.minus_standard_f1_string_lo:+.3f} to {t.minus_standard_f1_string_hi:+.3f}"
                 f"{'' if t.n_sources == b.n_sources else f', on its n = {int(t.n_sources)} sequences'}); * = needs the grammar", fontsize=11, loc="left")
    save(fig, out / "fig5_variants_f1.png")

    order = part.sort_values("wrong_in_R_per1000")
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(order) + 1.9))
    ypos = np.arange(len(order))
    ax.barh(ypos, order.wrong_in_R_per1000, 0.62, color=PINK, label="in repaired holes (kind R: several originals gone, the output is grammatical again)")
    ax.barh(ypos, order.wrong_in_U_per1000, 0.62, left=order.wrong_in_R_per1000, color=YELLOW, label="in unresolved clusters (kind U: a forbidden window is left)")
    for yy, (_, row) in zip(ypos, order.iterrows()):
        ax.text(row.wrong_in_R_per1000 + row.wrong_in_U_per1000 + 0.15, yy, f"{row.wrong_in_R_per1000:.1f} + {row.wrong_in_U_per1000:.1f}", va="center", fontsize=8, color="#222222")
    ax.set_yticks(ypos, list(order.short), fontsize=8.5)
    ax.set_xlabel("wrong deletions per 1 000 letters that sit in clusters of kind R or U (typed with the grammar, validation only)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.55 / (0.42 * len(order))))
    style(ax, "x")
    ax.set_title(f"Cascade mass, {sets[0][0]} (n = {int(base.n_sources)}): wrong deletions in repaired holes per 1 000 letters: standard sampler {base.wrong_in_R_per1000:.1f}"
                 + (f", '{top.short}' {top.wrong_in_R_per1000:.1f}" if top is not None else "") + f";\na single forward pass digs no holes ({top1.wrong_in_R_per1000:.1f}) "
                 f"but leaves {top1.wrong_in_U_per1000:.1f} in unresolved clusters; * = needs the grammar", fontsize=11, loc="left")
    save(fig, out / "fig6_cascade_mass.png")

    summary = {"thresholds_tuned_on_sources_0_2999": chosen, "consensus_threshold_tuned_on_first_half": consensus_theta,
               "variants_found": sorted(variants), "tuning_sources": sets[0][0], "tuning": {}, "held_out": {},
               "anchor_minus_veto": anchor_vs_veto.to_dict(orient="records")}
    for table, key in ((tuning, "tuning"), (held, "held_out")):
        for _, row in table.iterrows():
            summary[key][f"{row.variant} | {row.sources}"] = {
                "n_sources": int(row.n_sources), "forward_passes": float(row.forward_passes),
                **{k: [row[k], row[f"{k}_lo"], row[f"{k}_hi"]] for k in ("f1_label", "f1_string", "precision_string", "recall_string", "exact_string",
                                                                       "wrong_label_per1000", "missed_label_per1000", "wrong_in_R_per1000", "wrong_in_U_per1000")},
                **{f"minus_standard_{k}": [row[f"minus_standard_{k}"], row[f"minus_standard_{k}_lo"], row[f"minus_standard_{k}_hi"]] for k in PAIRED}}
    return summary


# ---- 5. a detector for single outputs --------------------------------------------------------------------------


def step_detector(src, model: Corpus, st_model, boot: Booter, out: Path) -> dict:
    """Flag deleted letters whose first reading was low; how well does that find the wrong deletions?"""
    print("5. detector: deleted although the first reading said 'original'")
    q0 = np.nan_to_num(model.q0.astype(np.float64))
    kind = st_model["kind"]
    deleted, y = model.deleted, src.inserted
    targets = {"wrong deletion": deleted & ~y, "wrong deletion in a repaired hole (R)": deleted & ~y & (kind == KINDS.index("R")),
               "wrong deletion in a hole or unresolved cluster (R or U)": deleted & ~y & ((kind == KINDS.index("R")) | (kind == KINDS.index("U"))),
               "wrong deletion that changes the string (not Z)": deleted & ~y & (kind != KINDS.index("Z"))}
    tune, report = region(src, *TUNE) & model.covered, region(src, *REPORT) & model.covered
    rows, summary = [], {}
    grid = [0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for name, target in targets.items():
        best, best_f1 = None, -1.0
        for c in grid:
            flag = deleted & (q0 < c)
            cols = {"flag": seq_sum(flag, src), "hit": seq_sum(flag & target, src), "target": seq_sum(target, src), "deleted": seq_sum(deleted, src)}

            def fn(s):
                with np.errstate(divide="ignore", invalid="ignore"):
                    return {"precision": s["hit"] / s["flag"], "recall": s["hit"] / s["target"], "f1": 2 * s["hit"] / (s["flag"] + s["target"]),
                            "flagged_share_of_deletions": s["flag"] / s["deleted"], "base_rate_among_deletions": s["target"] / s["deleted"]}

            f1_tune = fn({k: v[tune].sum() for k, v in cols.items()})["f1"]
            if f1_tune > best_f1:
                best, best_f1 = c, f1_tune
            rows.append({"target": name, "flag_if_first_reading_below": c, "n_sources": int(report.sum()), "f1_on_tuning_sources": float(f1_tune),
                         **flat("", boot.ci(cols, report, fn))})
        chosen = [r for r in rows if r["target"] == name and r["flag_if_first_reading_below"] == best][0]
        summary[name] = {"threshold_chosen_on_sources_0_2999": best, "n_sources": int(report.sum()),
                         **{k: [chosen[k], chosen[f"{k}_lo"], chosen[f"{k}_hi"]] for k in ("precision", "recall", "f1", "flagged_share_of_deletions", "base_rate_among_deletions")}}
    table = pd.DataFrame(rows)
    table.to_csv(out / "detector.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for (name, part), colour, marker in zip(table.groupby("target", sort=False), (BLACK, PINK, YELLOW, SKY), "so^D"):
        ax.plot(part.recall, part.precision, color=colour, marker=marker, markersize=5.5, linewidth=1.8, label=f"{name} (base rate {part.base_rate_among_deletions.iloc[0]:.1%} of the deletions)")
    part = table[table.target == "wrong deletion"]
    marked = part[part.flag_if_first_reading_below.isin([0.01, 0.1, 0.3, 0.5])]
    ax.set_xlabel("recall: share of these wrong deletions that are flagged")
    ax.set_ylabel("precision: share of the flagged deletions\nthat are such a wrong deletion")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.14))
    style(ax, "both")
    place_labels(fig, ax, marked[["recall", "precision"]].values, [f"flag if first reading < {c:g}" for c in marked.flag_if_first_reading_below], fontsize=8)
    s = summary["wrong deletion"]
    low = part[part.flag_if_first_reading_below == 0.1].iloc[0]
    ax.set_title(f"'Deleted although the first reading said original' finds wrong deletions without a grammar: {low.precision:.0%} of the\n"
                 f"deletions with first reading below 0.1 are wrong ({low.recall:.0%} of all wrong deletions); below {s['threshold_chosen_on_sources_0_2999']:g}: "
                 f"precision {s['precision'][0]:.2f}, recall {s['recall'][0]:.2f}\n(one sample per source, held-out sources, n = {s['n_sources']})",
                 fontsize=11, loc="left")
    save(fig, out / "fig7_detector.png")
    return summary


# ---- sampler variants on the GPU ----------------------------------------------------------------------------------


def parse_variant(name: str) -> dict:
    """`anchor0.2+sharp2` -> {"anchor": 0.2, "sharp": 2.0, ...}; `base` is the standard sampler."""
    spec = {"anchor": None, "sharp": 1.0, "floor": 0.0, "excl": 0, "steps": 100}
    for part in name.split("+"):
        if part == "base":
            continue
        m = re.fullmatch(r"(anchor|sharp|floor|excl|steps)([0-9.]+)", part)
        if m is None:
            raise SystemExit(f"unknown variant part '{part}'")
        spec[m.group(1)] = int(m.group(2)) if m.group(1) in ("excl", "steps") else float(m.group(2))
    return spec


def sample(args):
    """Sampler variants (comma-separated) for the sources a:b -> results/solutions/scratch/s-<variant>-<a>-<b>.npz."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from editflows.errors import model_euler
    from evaluate import load_model

    run_dir = (ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    scratch = run_dir / "errors" / "results" / "solutions" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    src = Sources.load(ROOT / "runs" / "error-study" / "sources.npz")
    a, b = (int(v) for v in args.sources.split(":"))
    b = min(b, len(src))
    todo = [v for v in args.variant.split(",") if not (scratch / f"s-{v}-{a}-{b}.npz").exists()]
    if not todo:
        print("nothing to do: all outputs exist")
        return
    checkpoint = torch.load(run_dir / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    model = load_model(checkpoint, config, ema=True).to(device)
    kappa = instantiate(config.task.path).kappa
    bos = config.data.vocab_size - 1
    lengths = np.diff(src.offsets)[a:b]
    chunks, current, total = [], [], 0  # sources of similar length, at most `max_tokens` letters per forward pass
    for i in np.argsort(lengths, kind="stable"):
        if current and total + lengths[i] > args.max_tokens:
            chunks.append(current)
            current, total = [], 0
        current.append(a + int(i))
        total += int(lengths[i])
    chunks += [current] if current else []
    base = int(src.offsets[a])
    n = int(src.offsets[b]) - base
    for variant in todo:
        spec = parse_variant(variant)
        del_step, q0 = np.full(n, -1, dtype=np.int16), np.zeros(n, dtype=np.float32)
        start = time.time()
        for ids in chunks:
            seqs = [src.seq(i) for i in ids]
            common = dict(device=device, n_steps=spec["steps"], seed=args.seed)
            keep = None
            if spec["anchor"] is not None:  # the first reading decides which letters are protected for the whole run
                keep = model_euler(model, seqs, kappa, bos, end_step=1, **common)["q0"] < spec["anchor"]
            result = model_euler(model, seqs, kappa, bos, keep=keep, exclusion=spec["excl"], sharpen=spec["sharp"],
                                 floor=spec["floor"], **common)
            at = 0
            for i in ids:
                span = src.span(i)
                length = span.stop - span.start
                del_step[span.start - base : span.stop - base] = result["del_step"][at : at + length]
                q0[span.start - base : span.stop - base] = result["q0"][at : at + length]
                at += length
        np.savez_compressed(scratch / f"s-{variant}-{a}-{b}.npz", start=a, stop=b, steps=spec["steps"], seed=args.seed,
                            epoch=int(checkpoint["epoch"]), del_step=del_step, q0=q0)
        deleted, inserted = del_step >= 0, src.inserted[base : base + n]
        tp = int((deleted & inserted).sum())
        print(f"{run_dir.name} {variant} sources {a}:{b} in {time.time() - start:.0f} s; precision "
              f"{tp / max(deleted.sum(), 1):.4f} recall {tp / max(inserted.sum(), 1):.4f}", flush=True)


# ---- main -------------------------------------------------------------------------------------------------------


def kappa_end(run_dir: Path) -> float:
    """kappa(1) of the run's schedule: the share of the needed deletions a consistent flow has made at t = 1."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(run_dir / "config.yaml")
    return float(instantiate(config.task.path.kappa)(torch.tensor(1.0, dtype=torch.float64)))


def analyse(args):
    run_dir = (ROOT / args.run).resolve() if not Path(args.run).is_absolute() else Path(args.run)
    errors, study = run_dir / "errors", ROOT / "runs" / "error-study"
    out = errors / "results" / "solutions"
    cache = out / "scratch" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    src = Sources.load(study / "sources.npz")
    eligible = transition_table(np.random.default_rng(0), 20, 5, 1.0) > 0  # validation only
    boot = Booter(args.boot)
    kappa1 = kappa_end(run_dir)
    summary = {"run": run_dir.name, "kappa_1": kappa1, "bootstrap_resamples": args.boot}

    z = load_npz(errors / "m100.npz")
    if z is None:
        raise SystemExit(f"{errors / 'm100.npz'} is missing: nothing to analyse")
    model = Corpus("model", src, int(z["steps"])).add_prefix(src, z)
    summary["epoch"] = int(z["epoch"])
    # The truth: the posterior under the prior of the coupling (K letters thrown into the gaps: `q_gap`, joint samples
    # `g*.npz`); if that is not there, the one that takes every letter as an insertion independently (`q_true`,
    # `ffbs*.npz`). The exact-rate sampler `i2-*` stays as it is; its own first reading is `q_true`.
    posterior = load_npz(study / "posterior.npz") or {}
    posteriors = {key: posterior[key] for key in ("q_gap", "q_true") if key in posterior}
    q_ref = next(iter(posteriors.values()), None)
    perfect = next((stem for stem in ("g", "ffbs") if (study / f"{stem}.npz").exists()), None)
    summary["reference"] = {"posterior": next(iter(posteriors), None), "perfect_sampler": perfect}
    print("reference:", summary["reference"])
    if "q_gap" not in posteriors or perfect != "g":
        note(f"the reference with the prior of the coupling is incomplete (posterior: {list(posteriors)}, perfect sampler: {perfect}.npz): fallbacks used")
    own = {PERFECT: posterior.get("q_gap" if perfect == "g" else "q_true", q_ref), EXACT: posterior.get("q_true", q_ref)}
    stems = {PERFECT: f"{perfect}-rep" if perfect else None, EXACT: "i2-rep"}
    refs = {}
    for name, file in ((PERFECT, f"{perfect}.npz"), (EXACT, "i2-100.npz")):
        r = load_npz(study / file)
        if r is None:
            note(f"{file} is missing: reference '{name}' left out")
        else:
            refs[name] = Corpus(name, src, int(r["steps"])).add_prefix(src, r)

    t0 = time.time()
    st_model = source_stats(src, model, eligible, cache)
    st_refs = {name: source_stats(src, c, eligible, cache) for name, c in refs.items()}
    print(f"per-source counts: {time.time() - t0:.0f} s")
    stats = {f"model ({model.passes} Euler steps)": (st_model, model.covered)}
    stats.update({name: (st_refs[name], refs[name].covered) for name in refs})
    summary["counting"] = step_counting(src, stats, boot, out)
    summary["bracket"], bracket = step_bracket(src, model, st_model, st_refs.get(PERFECT), q_ref, posteriors, boot, out, cache)
    summary["self_consistency"], reps = step_consistency(src, errors, study, model, own, stems, kappa1, eligible, boot, out, cache)
    figure_bracket(bracket, summary["self_consistency"].get("sample_agreement", {}).get("model"), out)
    summary["variants"] = step_variants(src, errors, model, st_model, refs, st_refs, q_ref, reps, kappa1, eligible, boot, out, cache)
    summary["detector"] = step_detector(src, model, st_model, boot, out)
    summary["missing_or_skipped"] = NOTES
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("written to", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/lap-01")
    p.add_argument("--stage", choices=["analyse", "sample"], default="analyse")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (over sequences)")
    p.add_argument("--variant", default="base", help="sample: e.g. anchor0.2, sharp2, floor0.1, excl2, anchor0.2+sharp2")
    p.add_argument("--sources", default="0:1000", help="sample: range of sources a:b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=20_000, help="sample: letters per forward pass")
    p.add_argument("--gpu-fraction", type=float, default=0.1, help="sample: share of the GPU memory this process may use")
    args = p.parse_args()
    {"analyse": analyse, "sample": sample}[args.stage](args)


if __name__ == "__main__":
    main()
