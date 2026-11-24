#!/usr/bin/env python
r"""Dependence between errors and their order in time (hypothesis A of the error study, observational evidence).

Errors of a sampled deletion run come in groups. A group can have three reasons, and this script separates them:

(i)   heterogeneity fixed by the source: ambiguous letters sit next to each other;
(ii)  dependence a perfect sampler has too: coherent alternative readings (an inserted letter swapped with an
      identical neighbour, ...);
(iii) dependence created by the dynamics of sampling: one wrong deletion makes the next one more likely.

    python scripts/errors/dependence.py --run runs/lap-01

Parts (every part uses the files that exist and says which ones were missing):

1. `pairs`    replicates of the same sources with different sampler seeds: the Mantel-Haenszel odds ratio of
              (error at letter i, error at letter i + d) with one stratum per (source, i, j), for the trained model
              (M), the exact-rate Euler sampler (I2) and the perfect sampler: exact joint posterior samples
              (`g-rep*.npz`, the prior of the coupling; `ffbs-rep*.npz`, independent insertions, if those are
              missing). Everything fixed by the source cancels inside a stratum; the perfect sampler shows (ii);
              M / perfect shows (iii). The samplers err on
              different letters, so the odds ratios are also given for pairs of originals of the same ambiguity
              (exact posterior of being an insertion, `posterior.npz`). Plain numbers: another wrong deletion
              nearby in the same run against other runs of the same source, with and without the order in time.
2. `events`   event study on the logged q: what happens to the neighbours' q after a deletion, against matched
              letter-steps without a deletion nearby, by kind of deletion and by break size (how many of the two
              windows across the gap become forbidden; break 0 is a near-placebo: the new windows are allowed, but
              the allowed letters are not equally likely, so they can still be improbable). Without matching: the
              wrong deletions that follow a first one, model against exact rates.
3. `origin`   every wrong deletion classified by the model's q at t = 0 and at the step of its deletion, what
              happened in the step in which its q rose, and how ambiguous the original was in the source itself.
4. `same step` how clusters with several wrong deletions start: two deletions in one step or one after another.
5. `chains`   clusters with three or more wrong deletions: their order in space and time.

Outputs: `runs/<run>/errors/results/dependence/` (CSV tables, `summary.json`, PNG figures). CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # a training run owns the GPU; nothing here needs it

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numba  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import Sources, break_size  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
GAP = 2  # kept originals that may lie between two errors of one cluster (the convention of editflows.errors)
DMAX = 20  # largest distance of a pair of letters
WINDOWS = (1, 2, 3, 5, 10, 20)  # "another wrong deletion within +-w originals"
FOLLOW_WINDOWS = (2, 5, 12)  # "an earlier wrong deletion within +-w originals"
TAUS = np.array([-2, -1, 1, 2, 5, 10, 20])  # steps after (before) a deletion at which the neighbours are read
HORIZONS = np.array([1, 2, 5, 10, 20, 50, 10_000])  # "deleted within so many steps after the event" (last = ever)
NEIGHBOURS = 10  # neighbours on each side that are followed
ISOLATION = 12  # an event is isolated / a letter-step is a control if no other deletion of that step is this near
QEDGES = np.array([0, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                   0.8, 0.9, 0.97, 1.01])  # bins of q for matching controls
MIN_CONTROLS = 50
# A letter whose q stays at 0.1 is deleted with probability one half by the end of this schedule, so 0.1 is the
# natural line between "suspicious" and "not suspicious". Not tuned; 0.03 and 0.3 are reported for sensitivity.
Q_HIGH = 0.1
Q_SENSITIVITY = (0.03, 0.1, 0.3)
Q0_BINS = np.array([0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])
# How ambiguous an original is in the source itself: the exact posterior probability that it is an insertion
# (runs/error-study/posterior.npz: q_gap, the prior of the coupling, or q_true, independent insertions, if q_gap
# is missing). The perfect sampler deletes it wrongly with exactly this probability.
EXACT_EDGES = (0.01, Q_HIGH)
EXACT_CLASSES = ("clear in the source (exact posterior below 0.01)", "doubtful (0.01 to 0.1)", "ambiguous in the source (0.1 or more)")
SEED_REACH = 3  # a wrong deletion further than this (in originals) from every earlier one of its chain is a new seed
PERFECT = "perfect"  # key of the perfect sampler in tables and summary
NAMES = {"M": "trained model", "I2": "exact rates, same Euler steps", PERFECT: "perfect sampler (exact posterior)"}
# In every figure of the study a sampler has its colour, and these three colours mean nothing else:
SAMPLER_COLOR = {"M": COLORS[0], "I2": COLORS[1], PERFECT: COLORS[2]}
OTHER_COLORS = [COLORS[5], COLORS[4], COLORS[3], COLORS[7], COLORS[6]]  # for series that are not samplers

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E3E3E3", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "lines.linewidth": 1.8, "lines.markersize": 4.5, "figure.constrained_layout.use": True,
})


# ---- small tools ------------------------------------------------------------------------------------------------


def load_npz(path: Path):
    """The arrays of an npz file, or None if it is not there; one retry in case it is still being written."""
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


class Boot:
    """Bootstrap over sources: every statistic here is a ratio of sums over sources."""

    def __init__(self, n_sources: int, n_boot: int, seed: int):
        rng = np.random.default_rng(seed)
        self.n = n_sources
        self.W = rng.multinomial(n_sources, np.full(n_sources, 1.0 / n_sources), size=n_boot).astype(np.float64)

    def sums(self, per_source: np.ndarray) -> np.ndarray:
        return np.tensordot(self.W, per_source, axes=(1, 0))

    def ratio(self, num: np.ndarray, den: np.ndarray):
        """Point estimate and bootstrap replicates of sum(num) / sum(den); arrays have the source axis first."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return num.sum(0) / den.sum(0), self.sums(num) / self.sums(den)


def ci(boot: np.ndarray):
    with np.errstate(all="ignore"):
        return np.nanpercentile(boot, 2.5, axis=0), np.nanpercentile(boot, 97.5, axis=0)


def p_normal(point, boot, log=True):
    """Two-sided p-value of "no effect" from the bootstrap standard error (on the log scale for ratios)."""
    with np.errstate(all="ignore"):
        x, b = (np.log(point), np.log(boot)) if log else (point, boot)
        b = np.where(np.isfinite(b), b, np.nan)
        z = x / np.nanstd(b, axis=0)
    return np.array([math.erfc(abs(v) / math.sqrt(2)) if np.isfinite(v) else np.nan for v in np.atleast_1d(z)])


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok):
        order = ok[np.argsort(p[ok])]
        q = p[order] * len(ok) / np.arange(1, len(ok) + 1)
        out[order] = np.clip(np.minimum.accumulate(q[::-1])[::-1], 0, 1)
    return out


def boot_quantiles(values: np.ndarray, source: np.ndarray, boot: Boot, qs) -> np.ndarray:
    """Bootstrap replicates (n_boot, len(qs)) of the quantiles of `values`, resampling their sources."""
    order = np.argsort(values, kind="stable")
    v, s = values[order], source[order]
    out = np.full((len(boot.W), len(qs)), np.nan)
    for b, w in enumerate(boot.W):
        c = np.cumsum(w[s])
        if c[-1] > 0:
            out[b] = v[np.minimum(np.searchsorted(c, np.asarray(qs) * c[-1], side="left"), len(v) - 1)]
    return out


def schedule(run: Path, n_steps: int):
    """Hazard at every step and the operational time Lambda at the step boundaries, from the config of the run."""
    try:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        kappa = instantiate(OmegaConf.load(run / "config.yaml").task.path.kappa)
        t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
        hazard = kappa.loss_coefficient(t).numpy()[:n_steps]
        with np.errstate(divide="ignore"):
            lam = -np.log1p(-kappa(t).numpy())
        return hazard, lam, "config.yaml of the run"
    except Exception as error:  # noqa: BLE001
        print(f"  schedule: could not read {run / 'config.yaml'} ({type(error).__name__}); using 3 exp(1.5 t)")
        t = np.arange(n_steps + 1) / n_steps
        return 3 * np.exp(1.5 * t)[:n_steps], 2 * (np.exp(1.5 * t) - 1), "fallback 3 exp(1.5 t)"


def n_covered(data, sources) -> int:
    """Number of sources a corpus file covers: its `first`, checked against the length of its arrays."""
    n = len(data["del_step"])
    first = int(data["first"]) if "first" in data else int(np.searchsorted(sources.offsets, n, side="right") - 1)
    assert int(sources.offsets[first]) == n, "the arrays of a corpus file do not end at a source boundary"
    return first


def source_sets(sources, first: int, also=None) -> dict:
    """The sets of sources, label -> (from, to), on which a file that covers the first `first` sources is summarised.

    Convention of the study: the first half of all sources is for exploring, the second half for reporting. A file
    that covers fewer sources is summarised on all it covers. `also` adds the first `also` sources, for a
    like-for-like comparison with a file that covers fewer. The first set is the one to quote."""
    total = len(sources)
    sets = {}
    if first == total:
        sets[f"sources {total // 2}-{total - 1} (reporting half)"] = (total // 2, total)
        sets[f"sources 0-{total // 2 - 1} (exploring half)"] = (0, total // 2)
    else:
        sets[f"sources 0-{first - 1}"] = (0, first)
    if also is not None and also < first and (0, also) not in sets.values():
        sets[f"sources 0-{also - 1}"] = (0, also)
    return sets


def perfect_sampler_files(study: Path):
    """Replicates of the perfect sampler: exact joint posterior samples with the prior of the coupling (`g-rep`,
    the inserted letters thrown into gaps) if at least 4 are there, else with independent insertions (`ffbs-rep`)."""
    for stem, note in (("g-rep", "g-rep*.npz: exact joint posterior samples, prior of the coupling (letters thrown into gaps)"),
                       ("ffbs-rep", "ffbs-rep*.npz: exact joint posterior samples, every letter an insertion independently")):
        if sum((study / f"{stem}{k}.npz").exists() for k in range(256)) >= 4:
            return study / (stem + "{k}.npz"), note
    return study / "g-rep{k}.npz", "none found"


def exact_posterior(study: Path):
    """The exact posterior that a letter is an insertion (t = 0) for all letters and the name of the array:
    `q_gap` (prior of the coupling) if `posterior.npz` has it, else `q_true` (independent insertions)."""
    post = load_npz(study / "posterior.npz")
    for key in ("q_gap", "q_true"):
        if post is not None and key in post:
            return post[key], key
    return None, None


def save_fig(fig, path: Path):
    fig.savefig(path, dpi=150)
    plt.close(fig)


def headline(fig, text: str):
    """The message of the figure as its title, wrapped to the width of the figure."""
    fig.suptitle(textwrap.fill(text, int(fig.get_figwidth() * 10.8)), fontsize=10.5, fontweight="bold", x=0.01, ha="left")


def plain_log_axis(axes):
    """Give the log y-axes of a row of panels the same range and tick labels written as plain numbers."""
    axes = list(np.atleast_1d(axes))
    lo, hi = min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes)
    ticks = []
    for candidates in ((0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000),
                       (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000), (0.1, 1, 10, 100, 1000, 10000)):
        ticks = [c for c in candidates if lo <= c <= hi]
        if len(ticks) <= 8:
            break
    for i, ax in enumerate(axes):
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{c:g}" for c in ticks] if i == 0 or len(axes) == 1 else [])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def legend_below(fig, ax, ncols: int = 2):
    """One legend for the whole figure, below the panels (so that it never covers data)."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside lower center", ncols=ncols, fontsize=8.5)


def r3(x):
    """Rounded plain floats for json."""
    if isinstance(x, dict):
        return {k: r3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r3(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(f"{float(x):.4g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# ---- numba kernels ------------------------------------------------------------------------------------------------


@numba.njit
def nb_cluster_ids(deleted, inserted, offsets, gap):
    """Global cluster number of every error letter (-1 elsewhere): `editflows.errors.cluster_ids` for all sources."""
    cid = np.full(len(deleted), -1, np.int64)
    c = -1
    for s in range(len(offsets) - 1):
        kept, last, opened = 0, 0, False
        for i in range(offsets[s], offsets[s + 1]):
            if inserted[i] == deleted[i]:  # kept original or correct deletion
                if not inserted[i]:
                    kept += 1
                continue
            if (not opened) or kept - last > gap:
                c += 1
                opened = True
            last = kept
            cid[i] = c
    return cid, c + 1


@numba.njit
def nb_cluster_stats(deleted, inserted, offsets, gap, out):
    """Adds per source: clusters with a wrong deletion, their wrong deletions, those with >= 2, with >= 3."""
    for s in range(len(offsets) - 1):
        kept, last, opened, nx = 0, 0, False, 0
        for i in range(offsets[s], offsets[s + 1] + 1):
            end = i == offsets[s + 1]
            if not end and inserted[i] == deleted[i]:
                if not inserted[i]:
                    kept += 1
                continue
            if end or (opened and kept - last > gap):
                if opened and nx > 0:
                    out[s, 0] += 1
                    out[s, 1] += nx
                    out[s, 2] += nx >= 2
                    out[s, 3] += nx >= 3
                nx = 0
            if end:
                break
            opened = True
            last = kept
            if not inserted[i]:
                nx += 1


@numba.njit
def nb_context(del_step, offsets, inserted, iso, n_steps):
    """For every deleted letter, in the sequence present at its step: the two neighbours on each side (source
    index, -1 if absent), the distance to the nearest other deletion of the same step, and whether there was no
    earlier wrong deletion among the `iso` present letters on each side ("first")."""
    n = len(del_step)
    ctx = np.full((n, 4), -1, np.int64)
    near = np.full(n, 1_000_000, np.int64)
    first = np.zeros(n, np.bool_)
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        cur = np.empty(b - a, np.int64)
        for k in range(n_steps):
            m, any_deletion = 0, False
            for i in range(a, b):
                if del_step[i] < 0 or del_step[i] >= k:
                    cur[m] = i
                    m += 1
                    any_deletion |= del_step[i] == k
            if not any_deletion:
                continue
            last = -1_000_000
            for p in range(m):
                i = cur[p]
                if del_step[i] == k:
                    near[i] = p - last
                    last = p
            last = 2_000_000
            for p in range(m - 1, -1, -1):
                i = cur[p]
                if del_step[i] == k:
                    if last - p < near[i]:
                        near[i] = last - p
                    last = p
                    if p >= 2:
                        ctx[i, 0] = cur[p - 2]
                    if p >= 1:
                        ctx[i, 1] = cur[p - 1]
                    if p + 1 < m:
                        ctx[i, 2] = cur[p + 1]
                    if p + 2 < m:
                        ctx[i, 3] = cur[p + 2]
                    clean = True
                    for j in range(cur[max(p - iso, 0)], cur[min(p + iso, m - 1)] + 1):
                        if del_step[j] >= 0 and del_step[j] < k and not inserted[j]:
                            clean = False
                            break
                    first[i] = clean
    return ctx, near, first


@numba.njit
def nb_controls(del_step, q, offsets, inserted, taus, horizons, qedges, iso, c_sum, c_lsum, c_jump, c_cnt, c_lcnt,
                c_del, c_n, base):
    """Letter-steps with no deletion within `iso` present letters in that step: sums of the later change of q per
    cell (step, kind of letter, bin of q), among survivors and with the last value carried forward, and how many
    of them are deleted within each horizon. `base[kind, r]`: letter-steps with a deletion within r letters."""
    n_steps = q.shape[1]
    radii = np.array([1, 2, 5, 10])
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        cur = np.empty(b - a, np.int64)
        near = np.empty(b - a, np.int64)
        for k in range(n_steps):
            m = 0
            for i in range(a, b):
                if del_step[i] < 0 or del_step[i] >= k:
                    cur[m] = i
                    m += 1
            last = -1_000_000
            for p in range(m):
                near[p] = p - last
                if del_step[cur[p]] == k:
                    last = p
            last = 2_000_000
            for p in range(m - 1, -1, -1):
                if last - p < near[p]:
                    near[p] = last - p
                if del_step[cur[p]] == k:
                    last = p
            for p in range(m):
                j = cur[p]
                dj = del_step[j]
                if dj == k:
                    continue
                kind = 1 if inserted[j] else 0
                base[kind, 0] += 1
                for r in range(4):
                    if near[p] <= radii[r]:
                        base[kind, r + 1] += 1
                if near[p] <= iso:
                    continue
                qp = np.float64(q[j, k])
                cell = np.searchsorted(qedges, qp, side="right") - 1
                c_n[k, kind, cell] += 1
                for h in range(len(horizons)):
                    if dj > k and dj - k <= horizons[h]:
                        c_del[k, kind, cell, h] += 1
                for t in range(len(taus)):
                    kk = k + taus[t]
                    if kk < 0 or kk >= n_steps:
                        continue
                    gone = dj >= 0 and dj < kk
                    value = np.float64(q[j, dj if gone else kk]) - qp
                    c_lsum[k, kind, cell, t] += value
                    c_lcnt[k, kind, cell, t] += 1
                    if not gone:
                        c_sum[k, kind, cell, t] += value
                        c_cnt[k, kind, cell, t] += 1
                        if value > 0.1:
                            c_jump[k, kind, cell, t] += 1


@numba.njit
def nb_events(group, del_step, q, offsets, inserted, taus, horizons, qedges, n_side, m_dq, m_ldq, m_jump, m_del,
              e_n, e_raw, e_adj, e_ladj, e_jadj, e_cnt, e_lcnt, e_pre, e_side, e_side_n, e_del, e_dadj, e_dn):
    """For every event (deleted letter with `group` >= 0) the surviving neighbours up to `n_side` present letters
    away: change of q after tau steps, raw and minus the mean of the matched controls (`m_*`, NaN = no match), and
    whether the neighbour is deleted within each horizon, minus the matched rate. Everything summed per
    (source, group, kind of neighbour, distance, ...)."""
    n_steps = q.shape[1]
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        cur = np.empty(b - a, np.int64)
        for k in range(n_steps):
            m, any_event = 0, False
            for i in range(a, b):
                if del_step[i] < 0 or del_step[i] >= k:
                    cur[m] = i
                    m += 1
                    any_event |= del_step[i] == k and group[i] >= 0
            if not any_event:
                continue
            for p in range(m):
                i = cur[p]
                if del_step[i] != k or group[i] < 0:
                    continue
                g = group[i]
                e_n[s, g] += 1
                for side in range(2):
                    for d in range(1, n_side + 1):
                        pp = p + d if side == 1 else p - d
                        if pp < 0 or pp >= m:
                            break
                        j = cur[pp]
                        dj = del_step[j]
                        if dj == k:
                            continue
                        kind = 1 if inserted[j] else 0
                        qp = np.float64(q[j, k])
                        cell = np.searchsorted(qedges, qp, side="right") - 1
                        if not np.isnan(m_del[k, kind, cell, 0]):
                            e_dn[s, g, kind, d - 1] += 1
                            for h in range(len(horizons)):
                                y = 1.0 if (dj > k and dj - k <= horizons[h]) else 0.0
                                e_del[s, g, kind, d - 1, h] += y
                                e_dadj[s, g, kind, d - 1, h] += y - m_del[k, kind, cell, h]
                        for t in range(len(taus)):
                            kk = k + taus[t]
                            if kk < 0 or kk >= n_steps:
                                continue
                            gone = dj >= 0 and dj < kk
                            value = np.float64(q[j, dj if gone else kk]) - qp
                            if not np.isnan(m_ldq[k, kind, cell, t]):
                                e_ladj[s, g, kind, d - 1, t] += value - m_ldq[k, kind, cell, t]
                                e_lcnt[s, g, kind, d - 1, t] += 1
                            if gone or np.isnan(m_dq[k, kind, cell, t]):
                                continue
                            e_raw[s, g, kind, d - 1, t] += value
                            e_adj[s, g, kind, d - 1, t] += value - m_dq[k, kind, cell, t]
                            e_jadj[s, g, kind, d - 1, t] += (1.0 if value > 0.1 else 0.0) - m_jump[k, kind, cell, t]
                            e_cnt[s, g, kind, d - 1, t] += 1
                            e_pre[s, g, kind, d - 1, t] += qp
                            if taus[t] == 1:
                                e_side[s, g, side, kind, d - 1] += value - m_dq[k, kind, cell, t]
                                e_side_n[s, g, side, kind, d - 1] += 1


@numba.njit
def nb_triggers(idx, del_step, q, start, end, inserted, reach):
    """For the deleted letters `idx`: the step s of the largest one-step rise of their q (from s to s + 1), that
    rise, the nearest deletion made in step s on either side (distance in present letters up to `reach`;
    kind 0 none, 1 correct, 2 wrong, 3 one of each at the same distance), and the distance to the nearest wrong
    deletion made before step s (1 = the gap right next to the letter; -1 = none within `reach`)."""
    n = len(idx)
    jump_step = np.full(n, -1, np.int64)
    rise = np.zeros(n)
    dist = np.full(n, -1, np.int64)
    kind = np.zeros(n, np.int64)
    earlier = np.full(n, -1, np.int64)
    for e in range(n):
        i = idx[e]
        k = del_step[i]
        best, s_best = -1.0, -1
        for s in range(k):
            r = np.float64(q[i, s + 1]) - np.float64(q[i, s])
            if r > best:
                best, s_best = r, s
        if s_best < 0:
            continue
        jump_step[e], rise[e] = s_best, best
        found = np.array([1_000_000, 1_000_000])
        kinds = np.zeros(2, np.int64)
        for side in range(2):
            c = 0
            j = i - 1 if side == 0 else i + 1
            while c < reach and j >= start[i] and j < end[i]:
                dj = del_step[j]
                if dj < 0 or dj >= s_best:
                    c += 1
                    if dj == s_best and found[side] == 1_000_000:
                        found[side] = c
                        kinds[side] = 1 if inserted[j] else 2
                elif not inserted[j] and (earlier[e] < 0 or c + 1 < earlier[e]):
                    earlier[e] = c + 1
                j += -1 if side == 0 else 1
        if found[0] < found[1]:
            dist[e], kind[e] = found[0], kinds[0]
        elif found[1] < found[0]:
            dist[e], kind[e] = found[1], kinds[1]
        elif found[0] < 1_000_000:
            dist[e] = found[0]
            kind[e] = kinds[0] if kinds[0] == kinds[1] else 3
    return jump_step, rise, dist, kind, earlier


@numba.njit
def nb_followups(group, del_step, offsets, inserted, n_steps, n_side, out):
    """Adds per (source, group of the event): [0] events, [1] wrong deletions made later among the originals within
    `n_side` present letters on each side, events followed by at least one such wrong deletion within [2] 2,
    [3] 5, [4] `n_side` letters, [5] originals among those neighbours."""
    for s in range(len(offsets) - 1):
        a, b = offsets[s], offsets[s + 1]
        cur = np.empty(b - a, np.int64)
        for k in range(n_steps):
            m, any_event = 0, False
            for i in range(a, b):
                if del_step[i] < 0 or del_step[i] >= k:
                    cur[m] = i
                    m += 1
                    any_event |= del_step[i] == k and group[i] >= 0
            if not any_event:
                continue
            for p in range(m):
                i = cur[p]
                if del_step[i] != k or group[i] < 0:
                    continue
                later, near2, near5, at_risk = 0, 0, 0, 0
                for side in range(2):
                    for d in range(1, n_side + 1):
                        pp = p + d if side == 1 else p - d
                        if pp < 0 or pp >= m:
                            break
                        j = cur[pp]
                        if inserted[j] or del_step[j] == k:
                            continue
                        at_risk += 1
                        if del_step[j] > k:
                            later += 1
                            if d <= 2:
                                near2 += 1
                            if d <= 5:
                                near5 += 1
                g = group[i]
                out[s, g, 0] += 1
                out[s, g, 1] += later
                out[s, g, 2] += near2 > 0
                out[s, g, 3] += near5 > 0
                out[s, g, 4] += later > 0
                out[s, g, 5] += at_risk


@numba.njit
def nb_same_step(idx, del_step, inserted, letters, start, end, q, has_q, p_scale, reach):
    """For the deleted letters `idx`: was another letter among the `reach` present letters on each side deleted in
    the same step (a correct deletion / a wrong one / the adjacent letter with the same symbol), and the
    probability of at least one such deletion if the letters are drawn independently at the logged rates."""
    n = len(idx)
    with_correct = np.zeros(n, np.bool_)
    with_wrong = np.zeros(n, np.bool_)
    twin = np.zeros(n, np.bool_)
    predicted = np.full(n, np.nan)
    for e in range(n):
        i = idx[e]
        k = del_step[i]
        none = 1.0
        for side in range(2):
            c = 0
            j = i - 1 if side == 0 else i + 1
            while c < reach and j >= start[i] and j < end[i]:
                if del_step[j] < 0 or del_step[j] >= k:
                    c += 1
                    if has_q:
                        none *= 1.0 - min(1.0, p_scale[k] * np.float64(q[j, k]))
                    if del_step[j] == k:
                        if inserted[j]:
                            with_correct[e] = True
                        else:
                            with_wrong[e] = True
                        if c == 1 and letters[j] == letters[i]:
                            twin[e] = True
                j += -1 if side == 0 else 1
        if has_q:
            predicted[e] = 1.0 - none
    return with_correct, with_wrong, twin, predicted


@numba.njit
def nb_followers(T, src, start, end, w, out):
    """T (R, n): step at which every original was wrongly deleted in every replicate (-1 = kept). Adds per source:
    [0] wrong deletions, [1] those with an earlier wrong deletion within +-w originals in the same run, [2] the same
    with the neighbourhood read from the other runs of the same source (average over them)."""
    R, n = T.shape
    earliest = np.empty(R, np.int64)
    for o in range(n):
        wrong_somewhere = False
        for r in range(R):
            if T[r, o] >= 0:
                wrong_somewhere = True
                break
        if not wrong_somewhere:
            continue
        lo, hi = max(o - w, start[o]), min(o + w, end[o] - 1)
        for r in range(R):
            m = 1 << 30
            for j in range(lo, hi + 1):
                if j != o and T[r, j] >= 0 and T[r, j] < m:
                    m = T[r, j]
            earliest[r] = m
        for a in range(R):
            k = T[a, o]
            if k < 0:
                continue
            out[src[o], 0] += 1
            if earliest[a] < k:
                out[src[o], 1] += 1
            c = 0
            for b in range(R):
                if b != a and earliest[b] < k:
                    c += 1
            out[src[o], 2] += c / (R - 1)


@numba.njit
def nb_chain_class(pos, reach):
    """A chain in the order of time: number of seeds (wrong deletions further than `reach` originals from every
    earlier one) and where the later ones lie relative to the first (0 all right, 1 all left, 2 both sides)."""
    seeds, left, right = 1, False, False
    for m in range(1, len(pos)):
        nearest = 1_000_000
        for j in range(m):
            nearest = min(nearest, abs(pos[m] - pos[j]))
        if nearest > reach:
            seeds += 1
        if pos[m] > pos[0]:
            right = True
        else:
            left = True
    return seeds, (2 if (left and right) else (1 if left else 0))


@numba.njit
def nb_chains(starts, pos, reach, n_perm, seed):
    """Class of every chain (0 grows right, 1 grows left, 2 grows on both sides, 3 separate seeds merge), its
    number of seeds, and the same under `n_perm` random orders of time: mean seeds and class frequencies."""
    np.random.seed(seed)
    n = len(starts) - 1
    cls = np.zeros(n, np.int64)
    seeds = np.zeros(n, np.int64)
    null_cls = np.zeros((n, 4))
    null_seeds = np.zeros(n)
    for c in range(n):
        p = pos[starts[c]:starts[c + 1]].copy()
        sd, side = nb_chain_class(p, reach)
        seeds[c] = sd
        cls[c] = 3 if sd > 1 else side
        for _ in range(n_perm):
            np.random.shuffle(p)
            sd, side = nb_chain_class(p, reach)
            null_seeds[c] += sd / n_perm
            null_cls[c, 3 if sd > 1 else side] += 1.0 / n_perm
    return cls, seeds, null_cls, null_seeds


# ---- part 1: dependence within a run, the source held fixed -------------------------------------------------------


def load_replicates(pattern, sources: Sources, k_max: int = 256):
    """Stack `del_step` of the replicate files that exist; every file says in `first` how many sources it covers,
    the stack is cut to the smallest. Returns the steps (R, n) (-1 = kept), the number of sources, the tags and
    whether the sampler has a time axis (the exact posterior samples have none: `steps` = 0)."""
    runs, firsts, tags, timed = [], [], [], True
    for k in range(k_max):
        z = load_npz(Path(str(pattern).format(k=k)))
        if z is None:
            continue
        first = int(z["first"]) if "first" in z else int(np.searchsorted(sources.offsets, len(z["del_step"]), side="right") - 1)
        if len(z["del_step"]) != int(sources.offsets[first]):
            print(f"  {Path(str(pattern).format(k=k)).name}: length does not match first = {first}, skipped")
            continue
        runs.append(z["del_step"])
        firsts.append(first)
        tags.append(k)
        timed &= int(z["steps"]) > 0 if "steps" in z else False
    if not runs:
        return None, 0, [], False
    first = min(firsts)
    return np.stack([r[: int(sources.offsets[first])] for r in runs]), first, tags, timed


def mh_sums(E: np.ndarray, seq: np.ndarray, n_sources: int, labels=None, n_labels: int = 1, keep=None):
    """Mantel-Haenszel sums per (label, source, distance) of the 2x2 tables (error at i, error at i + d) across
    the replicates: [0] sum a d / R, [1] sum b c / R, [2] informative strata, [3] replicate-pairs with both errors.
    `keep(d)` restricts the pairs at distance d."""
    R = E.shape[0]
    Ef = E.astype(np.float32)
    m = Ef.sum(0).astype(np.float64)
    out = np.zeros((4, n_labels, n_sources, DMAX))
    for d in range(1, DMAX + 1):
        same = seq[:-d] == seq[d:]
        if keep is not None:
            same = same & keep(d)
        a = np.einsum("ri,ri->i", Ef[:, :-d], Ef[:, d:]).astype(np.float64)
        mi, mj = m[:-d], m[d:]
        num = a * (R - mi - mj + a) / R
        den = (mi - a) * (mj - a) / R
        key = (seq[:-d] if labels is None else labels(d) * n_sources + seq[:-d])[same]
        for v, values in enumerate((num, den, ((num + den) > 0).astype(np.float64), a)):
            out[v, :, :, d - 1] = np.bincount(key, weights=values[same], minlength=n_labels * n_sources).reshape(n_labels, n_sources)
    return out


def near_wrong_sums(wrong: np.ndarray, seq_o: np.ndarray, n_sources: int, cls=None, n_cls: int = 1):
    """Per window w, class of the original and source: wrong deletions; those with another wrong deletion within
    +-w originals in the same run; the same with the neighbourhood read from the other runs (averaged); and the
    two counts of others. Shape (windows, 5, classes, sources)."""
    R, n = wrong.shape
    W = wrong.astype(np.int32)
    cs = np.concatenate([np.zeros((R, 1), np.int32), np.cumsum(W, 1, dtype=np.int32)], 1)
    first = np.flatnonzero(np.r_[True, seq_o[1:] != seq_o[:-1]])
    lengths = np.diff(np.r_[first, n])
    lo_s, hi_s = np.repeat(first, lengths), np.repeat(first + lengths - 1, lengths)
    A = W.sum(0).astype(np.float64)
    out = np.zeros((len(WINDOWS), 5, n_cls, n_sources))
    key = seq_o if cls is None else cls * n_sources + seq_o
    for w_i, w in enumerate(WINDOWS):
        lo, hi = np.maximum(np.arange(n) - w, lo_s), np.minimum(np.arange(n) + w, hi_s)
        others = cs[:, hi + 1] - cs[:, lo] - W
        some = others > 0
        same_any = (wrong & some).sum(0).astype(np.float64)
        same_cnt = (W * others).sum(0).astype(np.float64)
        cross_any = (A * some.sum(0) - same_any) / (R - 1)
        cross_cnt = (A * others.sum(0) - same_cnt) / (R - 1)
        for v, values in enumerate((A, same_any, cross_any, same_cnt, cross_cnt)):
            out[w_i, v] = np.bincount(key, weights=values, minlength=n_cls * n_sources).reshape(n_cls, n_sources)
    return out


def part_pairs(args, sources: Sources, out: Path, summary: dict, missing: list):
    print("part 1: dependence within a run (replicates)")
    perfect_files, perfect_note = perfect_sampler_files(args.study)
    files = {"M": args.errors / "rep{k}.npz", PERFECT: perfect_files, "I2": args.study / "i2-rep{k}.npz"}
    deleted, covered, steps_of = {}, {}, {}
    for name, pattern in files.items():
        D, first, tags, timed = load_replicates(pattern, sources)
        if D is None or len(tags) < 4:
            missing.append(f"{pattern} (replicates of {name}): {len(tags)} found, at least 4 needed")
            continue
        deleted[name], covered[name] = D >= 0, first
        if timed:
            steps_of[name] = D
        print(f"  {name}: {len(tags)} replicates of the first {first} sources")
    if "M" not in deleted and PERFECT not in deleted:
        return
    S = min(covered.values())
    n = int(sources.offsets[S])
    offsets = sources.offsets[: S + 1]
    ins = sources.inserted[:n]
    letters = sources.letters[:n].astype(np.int64)
    seq = np.repeat(np.arange(S), np.diff(offsets))
    boot = Boot(S, args.boot, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    deleted = {name: D[:, :n] for name, D in deleted.items()}
    errors = {name: D ^ ins[None] for name, D in deleted.items()}
    reps = {name: E.shape[0] for name, E in errors.items()}
    r_common = min(reps.values())
    ALL, EQUAL, SCRAMBLED = "all replicates", "same number of replicates for every sampler", "letters scrambled across replicates"
    # the Mantel-Haenszel estimate is consistent for any number of replicates but its noise is not the same, so
    # the samplers are also compared with the same number; scrambling must give an odds ratio of 1
    variants = {}
    for name, E in errors.items():
        variants[name, ALL] = E
        variants[name, EQUAL] = E[:r_common]
        variants[name, SCRAMBLED] = rng.permuted(E, axis=0)

    def source_labels(d):  # 0 (x,x), 1 (x,_) different letters, 2 (_,_), 3 (x,_) the same letter
        t = ins[:-d].astype(np.int64) + ins[d:]
        return np.where((t == 1) & (letters[:-d] == letters[d:]), 3, t)

    types = {"any pair": [0, 1, 2, 3], "x,x": [0], "x,_": [1, 3], "_,_": [2], "x,_ identical letters": [3],
             "x,_ different letters": [1]}
    # how ambiguous every original is in the source itself (exact posterior at t = 0), if the file is there
    q_all, q_name = exact_posterior(args.study)
    exact_class = np.digitize(q_all[:n][~ins], EXACT_EDGES) if q_all is not None else None
    if exact_class is None:
        missing.append(f"{args.study / 'posterior.npz'}: no split by the exact posterior in part 1")
    PAIR_CLASSES = ("x,x both ambiguous in the source", "x,x one of them ambiguous", "x,x neither ambiguous")
    pair_class = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 0]])  # ambiguous = last class of EXACT_CLASSES
    sums = {}  # (sampler, variant, coordinates, type) -> array (4, S, DMAX)
    for (name, variant), E in variants.items():
        if variant == EQUAL and reps[name] == r_common:
            for key in [k for k in sums if k[:2] == (name, ALL)]:
                sums[(name, EQUAL) + key[2:]] = sums[key]
            continue
        by_label = mh_sums(E, seq, S, source_labels, 4)
        for label, members in types.items():
            sums[name, variant, "source", label] = by_label[:, members].sum(1)
        sums[name, variant, "target", "x,x"] = mh_sums(E[:, ~ins], seq[~ins], S)[:, 0]
        if exact_class is not None and variant != SCRAMBLED:
            by_class = mh_sums(E[:, ~ins], seq[~ins], S, lambda d: pair_class[exact_class[:-d], exact_class[d:]], 3)
            for c, label in enumerate(PAIR_CLASSES):
                sums[name, variant, "target", label] = by_class[:, c]
    # composition: the samplers err on different letters. The same comparison on the pairs of originals of which
    # both letters are sometimes (not always) wrongly deleted in every sampler
    COMMON = "pairs that are informative in every sampler"
    wrong_o = {name: E[:r_common][:, ~ins] for name, E in errors.items()}
    open_all = np.logical_and.reduce([(W.sum(0) > 0) & (W.sum(0) < r_common) for W in wrong_o.values()])
    for name, W in wrong_o.items():
        variants[name, COMMON] = errors[name][:r_common]
        sums[name, COMMON, "target", "x,x"] = mh_sums(W, seq[~ins], S, keep=lambda d: open_all[:-d] & open_all[d:])[:, 0]
    rows, boots = [], {}
    for (name, variant, coord, label), v in sums.items():
        point, b = boot.ratio(v[0], v[1])
        lo, hi = ci(b)
        p = p_normal(point, b)
        qv = benjamini_hochberg(p)
        boots[name, variant, coord, label] = (point, b)
        for d in range(DMAX):
            rows.append({"sampler": name, "variant": variant, "replicates": variants[name, variant].shape[0],
                         "coordinates": coord, "pair": label,
                         "distance": d + 1, "odds_ratio": point[d], "lo": lo[d], "hi": hi[d], "p": p[d], "q_bh": qv[d],
                         "n_sources": S, "n_informative_strata": int(v[2][:, d].sum()),
                         "n_both_errors": int(v[3][:, d].sum())})
    table = pd.DataFrame(rows)
    table.to_csv(out / "pairs_odds_ratio.csv", index=False)

    rows = []
    for variant in (EQUAL, ALL, COMMON):
        for top, bottom in (("M", PERFECT), ("I2", PERFECT), ("M", "I2")):
            for (name, var, coord, label) in list(boots):
                if name != top or var != variant or (bottom, variant, coord, label) not in boots:
                    continue
                (pt, bt), (pb, bb) = boots[top, variant, coord, label], boots[bottom, variant, coord, label]
                point, b = pt / pb, bt / bb
                lo, hi = ci(b)
                p = p_normal(point, b)
                qv = benjamini_hochberg(p)
                for d in range(DMAX):
                    rows.append({"ratio": f"{top} / {bottom}", "variant": variant,
                                 "replicates": f"{variants[top, variant].shape[0]} / {variants[bottom, variant].shape[0]}",
                                 "coordinates": coord, "pair": label, "distance": d + 1,
                                 "ratio_of_odds_ratios": point[d], "lo": lo[d], "hi": hi[d], "p": p[d], "q_bh": qv[d],
                                 "n_sources": S})
    ratios = pd.DataFrame(rows, columns=["ratio", "variant", "replicates", "coordinates", "pair", "distance",
                                         "ratio_of_odds_ratios", "lo", "hi", "p", "q_bh", "n_sources"])
    ratios.to_csv(out / "pairs_ratio.csv", index=False)

    # the same for sources of low / medium / high noise (share of inserted letters)
    rho = sources.rho[:S]
    tercile = np.digitize(rho, np.quantile(rho, [1 / 3, 2 / 3]))
    rows = []
    for name in errors:
        v = sums[name, EQUAL, "target", "x,x"]
        for t, label in enumerate(("low noise", "medium noise", "high noise")):
            mask = (tercile == t)[:, None]
            point, b = boot.ratio(v[0] * mask, v[1] * mask)
            lo, hi = ci(b)
            for d in (0, 1, 2, 4, 9):
                rows.append({"sampler": name, "sources": label, "n_sources": int(mask.sum()), "distance_in_target": d + 1,
                             "odds_ratio": point[d], "lo": lo[d], "hi": hi[d]})
    pd.DataFrame(rows).to_csv(out / "pairs_by_noise.csv", index=False)

    # plain numbers: another wrong deletion nearby, same run against other runs of the same source
    seq_o = seq[~ins]
    near, rows = {}, []
    n_cls = len(EXACT_CLASSES) if exact_class is not None else 1
    expected_perfect = None  # what the perfect sampler does by definition: wrong deletions = sum of the exact posterior
    if exact_class is not None:
        q_exact = q_all[:n][~ins].astype(np.float64)
        expected_perfect = np.stack([np.bincount(seq_o[exact_class == c], weights=q_exact[exact_class == c], minlength=S)
                                     for c in range(n_cls)])
    by_class_rows = []
    for name, D in deleted.items():
        wrong = D[:, :n][:, ~ins]
        full = near_wrong_sums(wrong, seq_o, S, exact_class, n_cls)
        v = full.sum(2)
        near[name] = v
        subsets = [("all originals", v)] + ([(EXACT_CLASSES[c], full[:, :, c]) for c in range(n_cls)] if n_cls > 1 else [])
        for subset, u in subsets:
            for w_i, w in enumerate(WINDOWS):
                A = u[w_i, 0][:, None]
                stats = np.stack([u[w_i, 1], u[w_i, 2], u[w_i, 3], u[w_i, 4], u[w_i, 3] - u[w_i, 4]], 1)
                point, b = boot.ratio(stats, A)
                lo, hi = ci(b)
                rr, rr_b = point[0] / point[1], b[:, 0] / b[:, 1]
                per_source, ps_b = boot.ratio(u[w_i, 0] / D.shape[0], np.ones(S))
                rows.append({"sampler": name, "wrongly_deleted_original": subset, "window_originals": w,
                             "n_wrong_deletions": int(A.sum()), "n_sources": S, "replicates": D.shape[0],
                             "wrong_deletions_per_source_and_run": per_source, "per_source_lo": ci(ps_b)[0], "per_source_hi": ci(ps_b)[1],
                             "p_another_same_run": point[0], "p_same_lo": lo[0], "p_same_hi": hi[0],
                             "p_another_other_run": point[1], "p_other_lo": lo[1], "p_other_hi": hi[1],
                             "risk_ratio": rr, "rr_lo": np.nanpercentile(rr_b, 2.5), "rr_hi": np.nanpercentile(rr_b, 97.5),
                             "others_same_run": point[2], "others_other_run": point[3],
                             "excess_others": point[4], "excess_lo": lo[4], "excess_hi": hi[4]})
        if n_cls > 1:
            w5 = WINDOWS.index(5)
            for c in range(n_cls):
                u = full[w5, :, c]
                per_source, ps_b = boot.ratio(np.stack([u[0] / D.shape[0], u[0] / D.shape[0] - expected_perfect[c]], 1), np.ones((S, 1)))
                lo, hi = ci(ps_b)
                p_near, pn_b = boot.ratio(np.stack([u[1], u[2]], 1), u[0][:, None])
                by_class_rows.append({"sampler": name, "original": EXACT_CLASSES[c], "n_sources": S, "replicates": D.shape[0],
                                      "originals_per_source": float((exact_class == c).sum() / S),
                                      "wrong_deletions_per_source_and_run": per_source[0], "lo": lo[0], "hi": hi[0],
                                      "perfect_sampler_expects": float(expected_perfect[c].mean()),
                                      "surplus_over_the_perfect_sampler": per_source[1], "surplus_lo": lo[1], "surplus_hi": hi[1],
                                      "p_another_wrong_deletion_within_5_same_run": p_near[0], "p_same_lo": ci(pn_b)[0][0], "p_same_hi": ci(pn_b)[1][0],
                                      "p_another_wrong_deletion_within_5_other_run": p_near[1], "p_other_lo": ci(pn_b)[0][1], "p_other_hi": ci(pn_b)[1][1]})
    near_table = pd.DataFrame(rows)
    near_table.to_csv(out / "near_wrong_deletion.csv", index=False)
    by_class_table = pd.DataFrame(by_class_rows)
    if len(by_class_table):
        by_class_table.to_csv(out / "wrong_deletions_by_exact_posterior.csv", index=False)

    # with the order in time: wrong deletions that come after another wrong deletion nearby, same run against the
    # other runs of the same source (what the marginal rates and their timing alone would give)
    first_o = np.flatnonzero(np.r_[True, seq_o[1:] != seq_o[:-1]])
    len_o = np.diff(np.r_[first_o, len(seq_o)])
    start_o, end_o = np.repeat(first_o, len_o), np.repeat(first_o + len_o, len_o)
    followers, rows = {}, []
    for name, T in steps_of.items():
        To = np.ascontiguousarray(T[:, :n][:, ~ins].astype(np.int64))
        for w in FOLLOW_WINDOWS:
            v = np.zeros((S, 3))
            nb_followers(To, seq_o, start_o, end_o, w, v)
            runs_per_source = np.full((S, 1), float(To.shape[0]))
            per_run = np.stack([v[:, 0], v[:, 0] - v[:, 1], v[:, 1], v[:, 2], v[:, 1] - v[:, 2]], 1)
            point, b = boot.ratio(per_run, runs_per_source)
            followers[name, w] = (point, b)
            lo, hi = ci(b)
            share, share_b = boot.ratio(v[:, 1:], v[:, :1])
            s_lo, s_hi = ci(share_b)
            rows.append({"sampler": name, "window_originals": w, "n_sources": S, "replicates": To.shape[0],
                         "n_wrong_deletions": int(v[:, 0].sum()),
                         "wrong_deletions_per_source": point[0], "wd_lo": lo[0], "wd_hi": hi[0],
                         "first_of_their_neighbourhood_per_source": point[1], "first_lo": lo[1], "first_hi": hi[1],
                         "after_an_earlier_one_nearby_per_source": point[2], "after_lo": lo[2], "after_hi": hi[2],
                         "after_an_earlier_one_in_another_run_per_source": point[3], "chance_lo": lo[3], "chance_hi": hi[3],
                         "more_than_chance_per_source": point[4], "excess_lo": lo[4], "excess_hi": hi[4],
                         "p_after_an_earlier_one_same_run": share[0], "p_same_lo": s_lo[0], "p_same_hi": s_hi[0],
                         "p_after_an_earlier_one_other_run": share[1], "p_other_lo": s_lo[1], "p_other_hi": s_hi[1]})
    follower_table = pd.DataFrame(rows)
    if len(follower_table):
        follower_table.to_csv(out / "followers_in_time.csv", index=False)

    # size of the clusters, real runs against runs whose letters come from independent replicates
    rows, cluster_boot = [], {}
    for name, D in deleted.items():
        for kind, runs in (("real", D[:, :n]), ("letters scrambled", rng.permuted(D[:, :n], axis=0))):
            stats = np.zeros((S, 4))
            for r in range(runs.shape[0]):
                nb_cluster_stats(np.ascontiguousarray(runs[r]), ins, offsets, GAP, stats)
            point, b = boot.ratio(stats[:, 1:], stats[:, :1])
            lo, hi = ci(b)
            cluster_boot[name, kind] = (point[0], b[:, 0])
            rows.append({"sampler": name, "runs": kind, "n_clusters_with_wrong_deletion": int(stats[:, 0].sum()),
                         "n_sources": S, "replicates": runs.shape[0],
                         "wrong_deletions_per_cluster": point[0], "lo": lo[0], "hi": hi[0],
                         "share_with_2_or_more": point[1], "share2_lo": lo[1], "share2_hi": hi[1],
                         "share_with_3_or_more": point[2], "share3_lo": lo[2], "share3_hi": hi[2],
                         "wrong_deletions_per_source": stats[:, 1].sum() / S / runs.shape[0]})
    cluster_table = pd.DataFrame(rows)
    cluster_table.to_csv(out / "cluster_size_replicates.csv", index=False)

    # ---- summary ----
    def pick(frame, **conditions):
        mask = np.ones(len(frame), bool)
        for key, value in conditions.items():
            mask &= (frame[key] == value).to_numpy()
        return frame[mask]

    s = {"n_sources": S, "replicates available": reps, "replicates in the comparison with equal numbers": r_common,
         "bootstrap": args.boot, "perfect sampler": perfect_note,
         "exact posterior used for the ambiguity of an original": f"{q_name} of posterior.npz"}
    shown = (1, 2, 3, 4, 5, 7, 10, 20)
    for name in errors:
        for variant, tag in ((EQUAL, f"{r_common} replicates"), (ALL, f"all {reps[name]} replicates")):
            rowset = pick(table, sampler=name, variant=variant, coordinates="target", pair="x,x")
            s[f"OR two originals by target distance, {name}, {tag}"] = {
                int(r.distance): [r.odds_ratio, r.lo, r.hi] for r in rowset.itertuples() if r.distance in shown}
        rowset = pick(table, sampler=name, variant=ALL, coordinates="target", pair="x,x")
        s[f"OR two originals, {name}: replicate-pairs with both wrong at distance 1"] = int(rowset.n_both_errors.iloc[0])
        sig = (rowset.q_bh.to_numpy() < 0.05) & (rowset.lo.to_numpy() > 1)
        s[f"OR two originals, {name}: target distances with OR > 1 (BH 5 %)"] = [int(v) for v in np.flatnonzero(sig) + 1]
        for label in ("x,_ identical letters", "x,_ different letters", "_,_", "x,x"):
            r = pick(table, sampler=name, variant=ALL, coordinates="source", pair=label, distance=1).iloc[0]
            s[f"OR adjacent source letters, {label}, {name}"] = [r.odds_ratio, r.lo, r.hi, int(r.n_both_errors)]
        r = pick(table, sampler=name, variant=SCRAMBLED, coordinates="target", pair="x,x", distance=1).iloc[0]
        s[f"OR two originals at distance 1 after scrambling (must be 1), {name}"] = [r.odds_ratio, r.lo, r.hi]
    for label in (f"M / {PERFECT}", f"I2 / {PERFECT}", "M / I2"):
        for variant, tag in ((EQUAL, "equal replicates"), (ALL, "all replicates"), (COMMON, "pairs informative in every sampler")):
            rowset = pick(ratios, ratio=label, variant=variant, coordinates="target", pair="x,x")
            if not len(rowset):
                continue
            s[f"ratio of OR two originals by target distance, {label}, {tag}"] = {
                int(r.distance): [r.ratio_of_odds_ratios, r.lo, r.hi, r.q_bh] for r in rowset.itertuples() if r.distance in shown}
            above = (rowset.q_bh.to_numpy() < 0.05) & (rowset.lo.to_numpy() > 1)
            below = (rowset.q_bh.to_numpy() < 0.05) & (rowset.hi.to_numpy() < 1)
            s[f"target distances at which {label} > 1 (BH 5 %), {tag}"] = [int(v) for v in np.flatnonzero(above) + 1]
            s[f"target distances at which {label} < 1 (BH 5 %), {tag}"] = [int(v) for v in np.flatnonzero(below) + 1]
        for pair in ("x,_ identical letters", "x,_ different letters", "_,_"):
            rowset = pick(ratios, ratio=label, variant=EQUAL, coordinates="source", pair=pair, distance=1)
            if len(rowset):
                r = rowset.iloc[0]
                s[f"ratio of OR adjacent source letters, {pair}, {label}, equal replicates"] = [r.ratio_of_odds_ratios, r.lo, r.hi]
    for name in deleted:
        r = pick(near_table, sampler=name, wrongly_deleted_original="all originals", window_originals=5).iloc[0]
        s[f"P(another wrong deletion within +-5 originals | wrong deletion), {name}"] = {
            "same run": [r.p_another_same_run, r.p_same_lo, r.p_same_hi],
            "other run of the same source": [r.p_another_other_run, r.p_other_lo, r.p_other_hi],
            "risk ratio": [r.risk_ratio, r.rr_lo, r.rr_hi],
            "excess other wrong deletions per wrong deletion": [r.excess_others, r.excess_lo, r.excess_hi],
            "n wrong deletions": int(r.n_wrong_deletions)}
        real, scrambled = cluster_boot[name, "real"], cluster_boot[name, "letters scrambled"]
        diff = real[1] - scrambled[1]
        s[f"wrong deletions per cluster that has one, {name}"] = {
            "real": real[0], "letters scrambled": scrambled[0],
            "extra per cluster": [real[0] - scrambled[0], *np.percentile(diff, [2.5, 97.5])]}
    for r in by_class_table.itertuples():
        s[f"wrong deletions of originals that are {r.original}, {r.sampler}"] = {
            "per source and run": [r.wrong_deletions_per_source_and_run, r.lo, r.hi],
            "the perfect sampler expects (sum of the exact posterior)": r.perfect_sampler_expects,
            "surplus over the perfect sampler": [r.surplus_over_the_perfect_sampler, r.surplus_lo, r.surplus_hi],
            "P(another wrong deletion within +-5 originals), same run": [r.p_another_wrong_deletion_within_5_same_run, r.p_same_lo, r.p_same_hi],
            "the same, other run of the same source": [r.p_another_wrong_deletion_within_5_other_run, r.p_other_lo, r.p_other_hi]}
    for name in errors:
        for label in PAIR_CLASSES:
            for variant, tag in ((EQUAL, f"{r_common} replicates"), (ALL, f"all {reps[name]} replicates")):
                rowset = pick(table, sampler=name, variant=variant, coordinates="target", pair=label)
                if len(rowset):
                    s[f"OR {label}, by target distance, {name}, {tag}"] = {
                        int(r.distance): [r.odds_ratio, r.lo, r.hi, int(r.n_both_errors)] for r in rowset.itertuples() if r.distance in (1, 2, 3, 5, 10)}
    for label in (f"M / {PERFECT}", f"I2 / {PERFECT}", "M / I2"):
        for pair in PAIR_CLASSES:
            rowset = pick(ratios, ratio=label, variant=EQUAL, coordinates="target", pair=pair)
            if len(rowset):
                s[f"ratio of OR {pair}, {label}, equal replicates"] = {
                    int(r.distance): [r.ratio_of_odds_ratios, r.lo, r.hi, r.q_bh] for r in rowset.itertuples() if r.distance in (1, 2, 3, 5, 10)}
    for r in follower_table.itertuples():
        s[f"wrong deletions per source and run, {r.sampler}, split by an earlier wrong deletion within +-{r.window_originals} originals"] = {
            "all": [r.wrong_deletions_per_source, r.wd_lo, r.wd_hi],
            "first of their neighbourhood": [r.first_of_their_neighbourhood_per_source, r.first_lo, r.first_hi],
            "after an earlier one nearby": [r.after_an_earlier_one_nearby_per_source, r.after_lo, r.after_hi],
            "the same, earlier one read from another run of the same source (chance)": [
                r.after_an_earlier_one_in_another_run_per_source, r.chance_lo, r.chance_hi],
            "after an earlier one nearby, more than chance": [r.more_than_chance_per_source, r.excess_lo, r.excess_hi],
            "P(an earlier wrong deletion nearby | wrong deletion), same run / other run": [
                r.p_after_an_earlier_one_same_run, r.p_after_an_earlier_one_other_run]}
    for w in FOLLOW_WINDOWS:
        if ("M", w) in followers and ("I2", w) in followers:
            (pm, bm), (pi, bi) = followers["M", w], followers["I2", w]
            lo, hi = ci(bm - bi)
            s[f"model minus exact rates, wrong deletions per source and run, window +-{w} originals"] = {
                key: [pm[i] - pi[i], lo[i], hi[i]] for i, key in enumerate(
                    ("all", "first of their neighbourhood", "after an earlier one nearby", "chance level of the latter",
                     "after an earlier one nearby, more than chance"))}
    if "M" in deleted and PERFECT in deleted:
        for other in [o for o in (PERFECT, "I2") if o in deleted]:
            a = [cluster_boot["M", k] for k in ("real", "letters scrambled")]
            b = [cluster_boot[other, k] for k in ("real", "letters scrambled")]
            dd = (a[0][1] - a[1][1]) - (b[0][1] - b[1][1])
            s[f"extra wrong deletions per cluster, M minus {other}"] = [
                (a[0][0] - a[1][0]) - (b[0][0] - b[1][0]), *np.percentile(dd, [2.5, 97.5])]
            vm, vo = near["M"][3], near[other][3]
            pm, bm = boot.ratio(vm[3] - vm[4], vm[0])
            po, bo = boot.ratio(vo[3] - vo[4], vo[0])
            s[f"excess other wrong deletions within +-5 per wrong deletion, M minus {other}"] = [
                pm - po, *np.percentile(bm - bo, [2.5, 97.5])]
    summary["1 pairs"] = s

    # ---- figures (every sampler with the same number of replicates) ----
    present = [name for name in ("M", "I2", PERFECT) if name in errors]
    d = np.arange(1, DMAX + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for name in present:
        rowset = pick(table, sampler=name, variant=EQUAL, coordinates="target", pair="x,x")
        ax.fill_between(d, rowset.lo, rowset.hi, color=SAMPLER_COLOR[name], alpha=0.18, lw=0)
        ax.plot(d, rowset.odds_ratio, "o-", color=SAMPLER_COLOR[name], label=NAMES[name])
    if reps.get(PERFECT, 0) > r_common:  # the perfect sampler with all its replicates: the same curve, less noise
        rowset = pick(table, sampler=PERFECT, variant=ALL, coordinates="target", pair="x,x")
        ax.plot(d, rowset.odds_ratio, ":", color=COLORS[6], lw=1.5, label=f"perfect sampler, all {reps[PERFECT]} runs per source")
    ax.axhline(1, color=COLORS[6], lw=1, ls="--", label="dashed lines: independent (left), the same dependence as the perfect sampler (right)")
    ax.set_yscale("log")
    plain_log_axis(ax)
    ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
    ax.set_xlabel("distance between the two original letters (letters of the target)")
    ax.set_ylabel("odds ratio: one is wrongly deleted\ngiven that the other is")
    ax.set_title("two wrong deletions in the same run")
    ax = axes[1]
    for label, colour in ((f"M / {PERFECT}", SAMPLER_COLOR["M"]), (f"I2 / {PERFECT}", SAMPLER_COLOR["I2"])):
        rowset = pick(ratios, ratio=label, variant=EQUAL, coordinates="target", pair="x,x")
        if not len(rowset):
            continue
        ax.fill_between(d, rowset.lo, rowset.hi, color=colour, alpha=0.18, lw=0)
        ax.plot(d, rowset.ratio_of_odds_ratios, "o-", color=colour)
    ax.axhline(1, color=COLORS[6], lw=1, ls="--")
    ax.set_yscale("log")
    plain_log_axis(ax)
    ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
    ax.set_xlabel("distance between the two original letters (letters of the target)")
    ax.set_ylabel("odds ratio divided by that\nof the perfect sampler")
    ax.set_title("dependence beyond that of the perfect sampler")
    legend_below(fig, axes[0], ncols=2)
    head = []
    for name in present:
        r = pick(table, sampler=name, variant=EQUAL, coordinates="target", pair="x,x", distance=1).iloc[0]
        head.append(f"{r.odds_ratio:.1f} for the {NAMES[name].split(' (')[0]}" if name != "I2" else f"{r.odds_ratio:.1f} with exact rates")
    headline(fig, "How strongly the wrong deletions of two neighbouring originals come together in one sampling run: odds ratio at "
             "distance 1 = " + ", ".join(head) + f". Source held fixed: {S} sources, {r_common} sampling runs each; bands: 95 % "
             "bootstrap over sources.")
    save_fig(fig, out / "fig1_odds_ratio_originals.png")

    with_classes = len(by_class_table) > 0
    fig, axes = plt.subplots(2 if with_classes else 1, 4, figsize=(13, 8.6 if with_classes else 4.6), squeeze=False)
    panels = (("x,x", "two originals"), ("x,_ identical letters", "original + insertion, same letter\n(next to each other: a swap)"),
              ("x,_ different letters", "original + insertion,\ndifferent letters"), ("_,_", "two insertions"))

    def curves(ax, coordinates, label, title):
        for name in present:
            rowset = pick(table, sampler=name, variant=EQUAL, coordinates=coordinates, pair=label)
            ok = (rowset.n_both_errors.to_numpy() >= 20) & np.isfinite(rowset.odds_ratio.to_numpy())
            ax.fill_between(d[ok], rowset.lo.to_numpy()[ok], rowset.hi.to_numpy()[ok], color=SAMPLER_COLOR[name], alpha=0.18, lw=0)
            ax.plot(d[ok], rowset.odds_ratio.to_numpy()[ok], "o-", color=SAMPLER_COLOR[name], label=NAMES[name], ms=3.5)
        ax.axhline(1, color=COLORS[6], lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_title(title, fontsize=9)

    for ax, (label, title) in zip(axes[0], panels):
        curves(ax, "source", label, title)
        ax.set_xlabel("distance in the source (letters)")
    plain_log_axis(axes[0])
    axes[0, 0].set_ylabel("odds ratio of the two errors\n(dashed line: independent)")
    swap = [pick(table, sampler=name, variant=EQUAL, coordinates="source", pair="x,_ identical letters", distance=1).iloc[0].odds_ratio
            for name in present]
    text = ("Which errors depend on each other. Top: by kind of pair; for the swap of an inserted letter with an identical "
            f"neighbour the odds ratio is {', '.join(f'{v:.0f}' for v in swap)} ({', '.join(NAMES[v].split(' (')[0].split(',')[0] for v in present)}).")
    if with_classes:
        titles = ("two originals,\nboth ambiguous in the source", "two originals,\none of them ambiguous", "two originals,\nneither ambiguous")
        for ax, label, title in zip(axes[1, :3], PAIR_CLASSES, titles):
            curves(ax, "target", label, title)
            ax.set_xlabel("distance in the target (letters)")
        plain_log_axis(axes[1, :3])
        axes[1, 0].set_ylabel("odds ratio of the two wrong deletions\n(dashed line: independent)")
        ax = axes[1, 3]
        x = np.arange(len(EXACT_CLASSES))
        width = 0.8 / len(present)
        for i, name in enumerate(present):
            r = by_class_table[by_class_table.sampler == name].set_index("original").loc[list(EXACT_CLASSES)]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, r.wrong_deletions_per_source_and_run, width=width * 0.92, color=SAMPLER_COLOR[name])
            ax.errorbar(pos, r.wrong_deletions_per_source_and_run, yerr=[r.wrong_deletions_per_source_and_run - r.lo,
                                                                         r.hi - r.wrong_deletions_per_source_and_run],
                        fmt="none", ecolor=COLORS[6], lw=1)
            for xi, value, top in zip(pos, r.wrong_deletions_per_source_and_run, r.hi):
                ax.text(xi, top, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["clear\nbelow 0.01", "doubtful\n0.01 to 0.1", "ambiguous\n0.1 or more"], fontsize=8.5)
        ax.set_xlabel("exact posterior of being an insertion")
        ax.set_ylabel("wrong deletions per source")
        ax.set_title("how often each kind of original\nis wrongly deleted", fontsize=9)
        ratio_rows = pick(ratios, ratio=f"M / {PERFECT}", variant=EQUAL, coordinates="target", pair=PAIR_CLASSES[0], distance=1)
        m_clear = by_class_table[(by_class_table.sampler == "M") & (by_class_table.original == EXACT_CLASSES[0])] if "M" in present else []
        if len(ratio_rows) and len(m_clear):
            r, mc = ratio_rows.iloc[0], m_clear.iloc[0]
            text += (f" Bottom: two originals of the same ambiguity (ambiguous = the exact posterior of being an insertion is 0.1 or "
                     f"more). For two ambiguous originals next to each other the odds ratio of the model is "
                     f"{r.ratio_of_odds_ratios:.2f} times that of the perfect sampler (95 % interval {r.lo:.2f} to {r.hi:.2f}); originals "
                     f"that are clear in the source are wrongly deleted {mc.wrong_deletions_per_source_and_run:.1f} times per source by "
                     f"the model, {mc.perfect_sampler_expects:.1f} times by the perfect sampler.")
    legend_below(fig, axes[0, 0], ncols=3)
    headline(fig, text + f" Same run, source held fixed, {S} sources, {r_common} runs each; points with fewer than 20 joint errors omitted.")
    save_fig(fig, out / "fig2_odds_ratio_by_pair_type.png")


# ---- parts 2 and 3: the logged q --------------------------------------------------------------------------------


def base_group(first: int, wrong: int, b: int) -> int:
    return first * 8 + wrong * 4 + b  # b: break size 0, 1, 2; 3 = fewer than two letters on one side


EVENT_GROUPS = {
    "correct deletion": [base_group(f, 0, b) for f in (0, 1) for b in range(4)],
    "correct deletion, no earlier wrong deletion nearby": [base_group(1, 0, b) for b in range(4)],
    "wrong deletion": [base_group(f, 1, b) for f in (0, 1) for b in range(4)],
    "wrong deletion, break 0": [base_group(f, 1, 0) for f in (0, 1)],
    "wrong deletion, break 1": [base_group(f, 1, 1) for f in (0, 1)],
    "wrong deletion, break 2": [base_group(f, 1, 2) for f in (0, 1)],
    "first wrong deletion": [base_group(1, 1, b) for b in range(4)],
    "first wrong deletion, break 0": [base_group(1, 1, 0)],
    "first wrong deletion, break 1": [base_group(1, 1, 1)],
    "first wrong deletion, break 2": [base_group(1, 1, 2)],
    "later wrong deletion, break 0": [base_group(0, 1, 0)],
    "later wrong deletion, break 1": [base_group(0, 1, 1)],
    "later wrong deletion, break 2": [base_group(0, 1, 2)],
    "correct deletion, break 0": [base_group(f, 0, 0) for f in (0, 1)],
    "correct deletion, break 1": [base_group(f, 0, 1) for f in (0, 1)],
    "correct deletion, break 2": [base_group(f, 0, 2) for f in (0, 1)],
}
KINDS = ("original", "inserted")
# the four series of the event-study figures: (group, label, colour, marker)
SERIES = (("first wrong deletion, break 0", "wrong deletion, the string stays allowed (break 0)", OTHER_COLORS[0], "o"),
          ("first wrong deletion, break 1", "wrong deletion, one window becomes forbidden (break 1)", OTHER_COLORS[1], "s"),
          ("first wrong deletion, break 2", "wrong deletion, both windows become forbidden (break 2)", OTHER_COLORS[2], "^"),
          ("correct deletion, no earlier wrong deletion nearby", "correct deletion (an insertion disappears)", OTHER_COLORS[3], "D"))


def event_groups(del_step, offsets, ins, letters, eligible, n_steps):
    """Group of every deletion, 8 * first + 4 * wrong + break size (-1 for letters that stay), its break size
    (`editflows.errors.break_size` on the sequence present at its step; 3 = too near an end of the sequence),
    whether it is isolated (no other deletion of its step within ISOLATION letters) and whether it is the first
    wrong deletion of its neighbourhood."""
    ctx, near, first = nb_context(del_step, offsets, ins, ISOLATION, n_steps)
    events = np.flatnonzero(del_step >= 0)
    complete = (ctx[events] >= 0).all(1)
    five = np.zeros((len(events), 5), np.int64)
    five[:, [0, 1, 3, 4]] = np.where(ctx[events] >= 0, letters[np.maximum(ctx[events], 0)], 0)
    brk = np.full(len(del_step), 3, np.int64)
    brk[events[complete]] = break_size(five.ravel(), 5 * np.arange(len(events)) + 2, eligible)[complete]
    group = np.full(len(del_step), -1, np.int64)
    group[events] = first[events] * 8 + (~ins[events]) * 4 + brk[events]
    return group, brk, near > ISOLATION, first


def matched_means(total: np.ndarray, count: np.ndarray, width: int = 5) -> np.ndarray:
    """Mean per cell (step, ...); cells with fewer than MIN_CONTROLS controls use the block of `width` steps
    around them, and are NaN (no match) if that is not enough either."""
    n_steps = total.shape[0]
    edges = np.arange(0, n_steps, width)
    block = np.arange(n_steps) // width
    exact = total / count
    coarse_t, coarse_c = np.add.reduceat(total, edges, axis=0), np.add.reduceat(count, edges, axis=0)
    coarse = np.where(coarse_c >= MIN_CONTROLS, coarse_t / coarse_c, np.nan)[block]
    return np.where(count >= MIN_CONTROLS, exact, coarse)


def part_events(args, sources: Sources, eligible: np.ndarray, out: Path, summary: dict, missing: list):
    print("parts 2 and 3: event study and origin of the wrong deletions (logged q)")
    z = load_npz(args.errors / "qlog.npz")
    if z is None or "qlog" not in z:
        missing.append(f"{args.errors / 'qlog.npz'}: parts 2, 3 and the rates of part 4 skipped")
        return None
    del_step = z["del_step"].astype(np.int64)
    n = len(del_step)
    S = int(z["first"]) if "first" in z else int(np.searchsorted(sources.offsets, n, side="right") - 1)
    assert int(sources.offsets[S]) == n, "qlog.npz does not end at a source boundary"
    q = z["qlog"].astype(np.float32)
    n_steps = q.shape[1]
    offsets = sources.offsets[: S + 1]
    ins = sources.inserted[:n]
    letters = sources.letters[:n].astype(np.int64)
    seq = np.repeat(np.arange(S), np.diff(offsets))
    hazard, lam, schedule_source = schedule(args.run, n_steps)
    p_scale = hazard / n_steps
    boot = Boot(S, args.boot, args.seed)
    deleted = del_step >= 0
    wrong, correct = deleted & ~ins, deleted & ins
    print(f"  {S} sources, {n:,} letters, {int(wrong.sum()):,} wrong and {int(correct.sum()):,} correct deletions")

    # ---- events and matched controls ----
    group, brk, isolated, first = event_groups(del_step, offsets, ins, letters, eligible, n_steps)
    group[~isolated] = -1
    n_groups = 16
    shape = (n_steps, 2, len(QEDGES) - 1)
    c_sum, c_lsum, c_jump, c_cnt, c_lcnt = (np.zeros(shape + (len(TAUS),)) for _ in range(5))
    c_del, c_n, base = np.zeros(shape + (len(HORIZONS),)), np.zeros(shape), np.zeros((2, 5))
    nb_controls(del_step, q, offsets, ins, TAUS, HORIZONS, QEDGES, ISOLATION, c_sum, c_lsum, c_jump, c_cnt, c_lcnt,
                c_del, c_n, base)
    m_dq, m_jump, m_ldq = matched_means(c_sum, c_cnt), matched_means(c_jump, c_cnt), matched_means(c_lsum, c_lcnt)
    m_del = matched_means(c_del, c_n[..., None] * np.ones(len(HORIZONS)))

    dims = (S, n_groups, 2, NEIGHBOURS)
    e_n = np.zeros((S, n_groups))
    e_raw, e_adj, e_ladj, e_jadj, e_cnt, e_lcnt, e_pre = (np.zeros(dims + (len(TAUS),)) for _ in range(7))
    e_side, e_side_n = np.zeros((S, n_groups, 2, 2, NEIGHBOURS)), np.zeros((S, n_groups, 2, 2, NEIGHBOURS))
    e_del, e_dadj, e_dn = np.zeros(dims + (len(HORIZONS),)), np.zeros(dims + (len(HORIZONS),)), np.zeros(dims)
    nb_events(group, del_step, q, offsets, ins, TAUS, HORIZONS, QEDGES, NEIGHBOURS, m_dq, m_ldq, m_jump, m_del,
              e_n, e_raw, e_adj, e_ladj, e_jadj, e_cnt, e_lcnt, e_pre, e_side, e_side_n, e_del, e_dadj, e_dn)

    def count(mask):
        return [int((wrong & mask).sum()), int((correct & mask).sum())]

    counts = pd.DataFrame({
        "kind": ["wrong", "correct"], "deletions": count(np.ones(n, bool)),
        f"isolated (no other deletion of the step within {ISOLATION} letters)": count(isolated),
        "break 0": count(brk == 0), "break 1": count(brk == 1), "break 2": count(brk == 2),
        "too near an end of the sequence": count(brk == 3),
        f"first (no earlier wrong deletion within {ISOLATION} letters)": count(first),
    })
    counts.to_csv(out / "event_counts.csv", index=False)

    rows, dq = [], {}
    for name, members in EVENT_GROUPS.items():
        def agg(x):
            return x[:, members].sum(1)
        cnt, lcnt = agg(e_cnt), agg(e_lcnt)
        point, b = boot.ratio(agg(e_adj), cnt)
        lo, hi = ci(b)
        raw, pre = agg(e_raw).sum(0) / cnt.sum(0), agg(e_pre).sum(0) / cnt.sum(0)
        jump, jb = boot.ratio(agg(e_jadj), cnt)
        jlo, jhi = ci(jb)
        locf, lb = boot.ratio(agg(e_ladj), lcnt)
        llo, lhi = ci(lb)
        dq[name] = (point, lo, hi, locf, llo, lhi, cnt.sum(0))
        for kind in range(2):
            for d in range(NEIGHBOURS):
                for t, tau in enumerate(TAUS):
                    rows.append({"event": name, "n_events": int(agg(e_n).sum()), "neighbour": KINDS[kind], "distance": d + 1,
                                 "steps_after": int(tau), "n_neighbours": int(cnt.sum(0)[kind, d, t]),
                                 "q_before": pre[kind, d, t], "change_of_q": raw[kind, d, t],
                                 "change_in_matched_controls": raw[kind, d, t] - point[kind, d, t],
                                 "change_vs_controls": point[kind, d, t], "lo": lo[kind, d, t], "hi": hi[kind, d, t],
                                 "share_rising_by_0.1_vs_controls": jump[kind, d, t], "rise_lo": jlo[kind, d, t],
                                 "rise_hi": jhi[kind, d, t],
                                 "change_vs_controls_last_value_carried": locf[kind, d, t], "carried_lo": llo[kind, d, t],
                                 "carried_hi": lhi[kind, d, t]})
    pd.DataFrame(rows).to_csv(out / "event_change_of_q.csv", index=False)

    rows = []
    for name, members in EVENT_GROUPS.items():
        num, den = e_side[:, members].sum(1), e_side_n[:, members].sum(1)
        point, b = boot.ratio(num, den)
        lo, hi = ci(b)
        for side in range(2):
            for kind in range(2):
                for d in range(3):
                    rows.append({"event": name, "side": ("left", "right")[side], "neighbour": KINDS[kind], "distance": d + 1,
                                 "n_neighbours": int(den.sum(0)[side, kind, d]), "change_vs_controls_next_step": point[side, kind, d],
                                 "lo": lo[side, kind, d], "hi": hi[side, kind, d]})
    pd.DataFrame(rows).to_csv(out / "event_left_right.csv", index=False)

    def horizon_label(h):
        return "until the end" if h > n_steps else int(h)

    rows, follow, extra = [], {}, {}
    for name, members in EVENT_GROUPS.items():
        den = e_dn[:, members].sum(1)
        treated = e_del[:, members].sum(1).sum(0) / den.sum(0)[..., None]
        point, b = boot.ratio(e_dadj[:, members].sum(1), den[..., None])
        lo, hi = ci(b)
        follow[name] = (point, lo, hi, treated)
        events_per_source = e_n[:, members].sum(1)
        for kind in range(2):
            for d in range(NEIGHBOURS):
                for h, horizon in enumerate(HORIZONS):
                    rows.append({"event": name, "neighbour": KINDS[kind], "distance": d + 1, "within_steps": horizon_label(horizon),
                                 "n_neighbours": int(den.sum(0)[kind, d]), "p_deleted": treated[kind, d, h],
                                 "p_deleted_matched_controls": treated[kind, d, h] - point[kind, d, h],
                                 "excess": point[kind, d, h], "lo": lo[kind, d, h], "hi": hi[kind, d, h]})
        for reach in (2, 5, NEIGHBOURS):  # extra deletions per event among the neighbours up to `reach` on both sides
            per_event, pb = boot.ratio(e_dadj[:, members].sum(1)[:, :, :reach].sum(2), events_per_source[:, None, None])
            extra[name, reach] = (per_event, *ci(pb), int(events_per_source.sum()))
    pd.DataFrame(rows).to_csv(out / "event_followup_deletions.csv", index=False)
    rows = []
    for (name, reach), (point, lo, hi, n_events) in extra.items():
        for kind in range(2):
            for h, horizon in enumerate(HORIZONS):
                rows.append({"event": name, "n_events": n_events, "neighbours_within": reach, "neighbour": KINDS[kind],
                             "within_steps": horizon_label(horizon),
                             "extra_deletions_per_event": point[kind, h], "lo": lo[kind, h], "hi": hi[kind, h]})
    pd.DataFrame(rows).to_csv(out / "event_extra_deletions_per_event.csv", index=False)

    t1 = int(np.flatnonzero(TAUS == 1)[0])
    pre1 = int(np.flatnonzero(TAUS == -1)[0])
    h_end = len(HORIZONS) - 1
    s = {"n_sources": S, "schedule read from": schedule_source,
         "events": {r["kind"]: {k: v for k, v in r.items() if k != "kind"} for r in counts.to_dict("records")},
         "note": "break 0 is a near-placebo: the windows across the gap are allowed but can be improbable",
         "matching": f"controls: same step, same kind of letter, same bin of q ({len(QEDGES) - 1} bins), no deletion within "
                     f"{ISOLATION} letters in that step; events: isolated deletions",
         "share of neighbours without matched controls": 1 - e_dn.sum() / max(e_n.sum() * 2 * NEIGHBOURS, 1)}
    for name in ("first wrong deletion, break 0", "first wrong deletion, break 1", "first wrong deletion, break 2",
                 "later wrong deletion, break 0", "later wrong deletion, break 1", "later wrong deletion, break 2",
                 "wrong deletion, break 0", "wrong deletion, break 1", "wrong deletion, break 2", "wrong deletion",
                 "first wrong deletion", "correct deletion", "correct deletion, no earlier wrong deletion nearby"):
        point, lo, hi, locf, llo, lhi, cnt = dq[name]
        fp, flo, fhi, treated = follow[name]
        entry = {"n events": int(e_n[:, EVENT_GROUPS[name]].sum())}
        for kind in range(2):
            entry[f"change of q vs matched controls, {KINDS[kind]} neighbours, next step, distance 1..6"] = [
                [point[kind, d, t1], lo[kind, d, t1], hi[kind, d, t1]] for d in range(6)]
            entry[f"share of {KINDS[kind]} neighbours whose q rises by more than 0.1, vs controls, distance 1..3"] = [
                float(v) for v in (e_jadj[:, EVENT_GROUPS[name]].sum((0, 1)) / e_cnt[:, EVENT_GROUPS[name]].sum((0, 1)))[kind, :3, t1]]
            entry[f"the same one step before the event (placebo), {KINDS[kind]}, distance 1, 2"] = [
                [point[kind, d, pre1], lo[kind, d, pre1], hi[kind, d, pre1]] for d in range(2)]
            entry[f"distance 1 after {[int(v) for v in TAUS]} steps, {KINDS[kind]}, survivors"] = [point[kind, 0, t] for t in range(len(TAUS))]
            entry[f"distance 1 after {[int(v) for v in TAUS]} steps, {KINDS[kind]}, last value carried"] = [locf[kind, 0, t] for t in range(len(TAUS))]
            entry[f"excess probability that a {KINDS[kind]} neighbour is deleted later, distance 1..6"] = [
                [fp[kind, d, h_end], flo[kind, d, h_end], fhi[kind, d, h_end]] for d in range(6)]
            for reach in (2, NEIGHBOURS):
                per_event, lo_e, hi_e, _ = extra[name, reach]
                entry[f"extra deletions of {KINDS[kind]} neighbours per event (within {reach} letters) after "
                      f"{[int(v) for v in HORIZONS[:-1]]} steps and at the end"] = [
                    [per_event[kind, h], lo_e[kind, h], hi_e[kind, h]] for h in range(len(HORIZONS))]
        s[name] = entry
    seeds = EVENT_GROUPS["first wrong deletion"]
    per_source, pb = boot.ratio(np.stack([e_n[:, seeds].sum(1), e_dadj[:, seeds].sum(1)[:, 0, :, h_end].sum(1),
                                          np.bincount(seq[wrong], minlength=S).astype(float)], 1), np.ones((S, 1)))
    lo, hi = ci(pb)
    s["per source"] = {
        "wrong deletions": [per_source[2], lo[2], hi[2]],
        "isolated first wrong deletions (no earlier wrong deletion nearby)": [per_source[0], lo[0], hi[0]],
        f"extra wrong deletions that follow them within {NEIGHBOURS} letters (matched estimate)": [per_source[1], lo[1], hi[1]],
        "the latter as a share of all wrong deletions": [per_source[1] / per_source[2], *np.percentile(pb[:, 1] / pb[:, 2], [2.5, 97.5])]}

    # ---- the same effect within subsets: noise level and length of the source, time and position of the event ----
    def subset_rows(kind, label, adj, cnt, dadj, n_ev, weight):
        result = []
        for b in range(3):
            g = base_group(1, 1, b)
            point, bb = boot.ratio(adj[:, g, 0, 0, t1] * weight, cnt[:, g, 0, 0, t1] * weight)
            extra_p, eb = boot.ratio(dadj[:, g, 0, :, h_end].sum(1) * weight, n_ev[:, g] * weight)
            result.append({"split_by": kind, "subset": label, "event": f"first wrong deletion, break {b}",
                           "n_events": int((n_ev[:, g] * weight).sum()),
                           "change_of_q_original_at_distance_1": point, "dq_lo": ci(bb)[0], "dq_hi": ci(bb)[1],
                           f"extra_wrong_deletions_per_event_within_{NEIGHBOURS}": extra_p, "extra_lo": ci(eb)[0], "extra_hi": ci(eb)[1]})
        return result

    rows = subset_rows("all", "all", e_adj, e_cnt, e_dadj, e_n, np.ones(S))
    rho, length = sources.rho[:S], np.diff(offsets)
    for kind, values, names in (("noise level of the source (share of inserted letters)", rho, ("low", "medium", "high")),
                                ("length of the source", length, ("short", "medium", "long"))):
        tercile = np.digitize(values, np.quantile(values, [1 / 3, 2 / 3]))
        for t, label in enumerate(names):
            rows += subset_rows(kind, label, e_adj, e_cnt, e_dadj, e_n, (tercile == t).astype(float))
    relative = (np.arange(n) - offsets[seq]) / np.repeat(length, length)
    letter_splits = (
        ("time of the event", (("first fifth of the steps", del_step < 0.2 * n_steps),
                               ("steps 20 to 50 %", (del_step >= 0.2 * n_steps) & (del_step < 0.5 * n_steps)),
                               ("second half of the steps", del_step >= 0.5 * n_steps))),
        ("position of the event in the source", (("first third", relative < 1 / 3), ("middle third", (relative >= 1 / 3) & (relative < 2 / 3)),
                                                 ("last third", relative >= 2 / 3))))
    for kind, subsets in letter_splits:
        for label, keep in subsets:
            x_n = np.zeros((S, n_groups))
            x_raw, x_adj, x_ladj, x_jadj, x_cnt, x_lcnt, x_pre = (np.zeros(dims + (len(TAUS),)) for _ in range(7))
            x_side, x_side_n = np.zeros((S, n_groups, 2, 2, NEIGHBOURS)), np.zeros((S, n_groups, 2, 2, NEIGHBOURS))
            x_del, x_dadj, x_dn = np.zeros(dims + (len(HORIZONS),)), np.zeros(dims + (len(HORIZONS),)), np.zeros(dims)
            nb_events(np.where(keep, group, -1), del_step, q, offsets, ins, TAUS, HORIZONS, QEDGES, NEIGHBOURS, m_dq, m_ldq, m_jump,
                      m_del, x_n, x_raw, x_adj, x_ladj, x_jadj, x_cnt, x_lcnt, x_pre, x_side, x_side_n, x_del, x_dadj, x_dn)
            rows += subset_rows(kind, label, x_adj, x_cnt, x_dadj, x_n, np.ones(S))
    robustness = pd.DataFrame(rows)
    robustness.to_csv(out / "event_robustness.csv", index=False)
    s["robustness: change of q of the original at distance 1 after a first wrong deletion with break 2, by subset"] = {
        f"{r.split_by}: {r.subset}": [r.change_of_q_original_at_distance_1, r.dq_lo, r.dq_hi, int(r.n_events)]
        for r in robustness[robustness.event.str.endswith("break 2")].itertuples()}
    s["robustness: the same with break 0 (near-placebo)"] = {
        f"{r.split_by}: {r.subset}": [r.change_of_q_original_at_distance_1, r.dq_lo, r.dq_hi, int(r.n_events)]
        for r in robustness[robustness.event.str.endswith("break 0")].itertuples()}
    summary["2 events"] = s

    # ---- figures of the event study ----
    d = np.arange(1, NEIGHBOURS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for kind, ax in enumerate(axes):
        for name, label, colour, marker in SERIES:
            point, lo, hi = dq[name][:3]
            n_events = int(e_n[:, EVENT_GROUPS[name]].sum())
            ax.fill_between(d, lo[kind, :, t1], hi[kind, :, t1], color=colour, alpha=0.18, lw=0)
            ax.plot(d, point[kind, :, t1], marker=marker, color=colour, label=f"{label}, {n_events:,} events")
        ax.axhline(0, color=COLORS[6], lw=1, ls="--", label="dashed line: the same change as in matched controls")
        ax.axvline(2.5, color=COLORS[7], lw=1, ls=":")
        ax.text(2.6, ax.get_ylim()[1], "reach of the grammar:\n2 letters", va="top", fontsize=8, color="#444444")
        ax.set_xticks(d)
        ax.set_xlabel("distance of the neighbour from the deleted letter (letters present)")
        ax.set_title(f"neighbours that are {'originals' if kind == 0 else 'leftover insertions'}")
    axes[0].set_ylabel("change of q one step after the deletion\nminus the change in matched controls")
    legend_below(fig, axes[0], ncols=2)
    b2, b1, b0 = (dq[f"first wrong deletion, break {v}"][0][0, 0, t1] for v in (2, 1, 0))
    headline(fig, f"What a wrong deletion does to its neighbours in the next step: q of the original at distance 1 changes by "
             f"{b2:+.2f} if both windows across the gap become forbidden, {b1:+.2f} if one does, {b0:+.3f} if the string stays allowed. "
             f"q = the model's probability that a letter is an insertion; isolated first wrong deletions, {S} sources; bands: 95 % "
             f"bootstrap over sources.")
    save_fig(fig, out / "fig3_event_study_distance.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    x = np.arange(len(TAUS))
    for ax, version in zip(axes[:2], ("survivors", "carried")):
        for name, label, colour, marker in SERIES:
            point, lo, hi, locf, llo, lhi, _ = dq[name]
            y, ylo, yhi = (point, lo, hi) if version == "survivors" else (locf, llo, lhi)
            ax.fill_between(x, ylo[0, 0], yhi[0, 0], color=colour, alpha=0.18, lw=0)
            ax.plot(x, y[0, 0], marker=marker, color=colour, label=label)
        ax.axhline(0, color=COLORS[6], lw=1, ls="--")
        ax.axvline(1.5, color=COLORS[7], lw=1, ls=":")
        ax.text(1.45, ax.get_ylim()[1], "the deletion ", ha="right", va="top", fontsize=8, color="#444444")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(v):+d}" for v in TAUS])
        ax.set_xlabel("steps relative to the deletion (not to scale)")
        ax.set_title("q of the original at distance 1,\nwhile it is still there" if version == "survivors"
                     else "the same, keeping its last q\nonce it has been deleted")
    axes[0].set_ylabel("change of q since the step of the deletion\nminus matched controls")
    ax = axes[2]
    hx = np.arange(len(HORIZONS))
    for name, label, colour, marker in SERIES:
        per_event, lo_e, hi_e, _ = extra[name, NEIGHBOURS]
        ax.fill_between(hx, lo_e[0], hi_e[0], color=colour, alpha=0.18, lw=0)
        ax.plot(hx, per_event[0], marker=marker, color=colour, label=label)
    ax.axhline(0, color=COLORS[6], lw=1, ls="--")
    ax.set_xticks(hx)
    ax.set_xticklabels([str(int(v)) for v in HORIZONS[:-1]] + ["end"])
    ax.set_xlabel("steps after the deletion (not to scale)")
    ax.set_ylabel(f"extra wrong deletions among the {NEIGHBOURS} letters\non each side, per event, vs matched controls")
    ax.set_title("wrong deletions that follow,\ncumulative")
    legend_below(fig, axes[0], ncols=2)
    e2 = extra["first wrong deletion, break 2", NEIGHBOURS][0][0]
    h5, h20 = (int(np.flatnonzero(HORIZONS == v)[0]) for v in (5, 20))
    keep20 = dq["first wrong deletion, break 2"][3][0, 0, int(np.flatnonzero(TAUS == 20)[0])]
    headline(fig, f"How long it lasts: after a wrong deletion that breaks both windows the q of the original next to it is still "
             f"{keep20:+.2f} above matched controls 20 steps later (last value kept for deleted letters); {e2[h_end]:.2f} extra wrong "
             f"deletions follow, {e2[h5]:.2f} of them within 5 steps and {e2[h20]:.2f} within 20 "
             f"(isolated first wrong deletions, {S} sources; bands: 95 % bootstrap over sources).")
    save_fig(fig, out / "fig4_event_study_time.png")

    # ---- part 3: where the wrong deletions come from ----
    w_idx = np.flatnonzero(wrong)
    w_src = seq[w_idx]
    k_del = del_step[w_idx]
    q0 = q[w_idx, 0].astype(np.float64)
    q_del = q[w_idx, k_del].astype(np.float64)
    start, end = offsets[seq], offsets[seq + 1]
    jump_step, rise, t_dist, t_kind, earlier = nb_triggers(w_idx, del_step, q, start, end, ins, NEIGHBOURS)

    def shares(masks: dict, within=None):
        base_mask = np.ones(len(w_idx), bool) if within is None else within
        den = np.bincount(w_src[base_mask], minlength=S).astype(float)
        result = {}
        for key, mask in masks.items():
            point, b = boot.ratio(np.bincount(w_src[mask & base_mask], minlength=S).astype(float), den)
            result[key] = (point, *ci(b), int((mask & base_mask).sum()))
        return result

    rows, origin = [], {}
    for threshold in Q_SENSITIVITY:
        a = q0 >= threshold
        b_ = ~a & (q_del >= threshold)
        c = ~a & (q_del < threshold)
        result = shares({"a ambiguous from the start": a, "b became suspicious during sampling": b_,
                         "c deleted although not suspicious": c, "a, of which no longer suspicious when deleted": a & (q_del < threshold)})
        origin[threshold] = (a, b_, c, result)
        for key, (point, lo, hi, n_class) in result.items():
            rows.append({"threshold_q": threshold, "class": key, "share": point, "lo": lo, "hi": hi, "n": n_class,
                         "n_wrong_deletions": len(w_idx), "n_sources": S})
    pd.DataFrame(rows).to_csv(out / "origin_classes.csv", index=False)
    grid = pd.crosstab(pd.cut(q0, Q0_BINS, right=False), pd.cut(q_del, Q0_BINS, right=False), dropna=False)
    grid.index.name, grid.columns.name = "q at t = 0", "q at the step of the deletion"
    grid.to_csv(out / "origin_q0_by_q_at_deletion.csv")

    a, b_, c, result = origin[Q_HIGH]
    near2 = (t_dist >= 1) & (t_dist <= 2)
    damaged = (earlier >= 1) & (earlier <= 2)
    trigger_masks = {
        "after a wrong deletion within 2 letters": near2 & (t_kind >= 2),
        "after a correct deletion within 2 letters, an earlier wrong deletion within 2 letters": near2 & (t_kind == 1) & damaged,
        "after a correct deletion within 2 letters, no earlier wrong deletion within 2 letters": near2 & (t_kind == 1) & ~damaged,
        "nearest deletion of that step 3 to 10 letters away": t_dist >= 3,
        "no deletion within 10 letters in that step": t_dist < 0,
    }
    rows, trigger = [], {}
    for label, cls in (("b became suspicious during sampling", b_), ("a ambiguous from the start", a & (k_del > 0)),
                       ("c deleted although not suspicious", c & (k_del > 0))):
        res = shares(trigger_masks, within=cls)
        trigger[label] = res
        for key, (point, lo, hi, n_class) in res.items():
            rows.append({"class": label, "step_of_the_largest_rise_of_q": key, "share": point, "lo": lo, "hi": hi, "n": n_class,
                         "n_in_class": int(cls.sum())})
    pd.DataFrame(rows).to_csv(out / "origin_trigger.csv", index=False)
    lag = k_del - jump_step
    lag_lam = lam[k_del] - lam[np.maximum(jump_step, 0)]
    one_jump = rise / (q_del - q0)
    rows = []
    for key, mask in list(trigger_masks.items()) + [("all of class b", np.ones(len(w_idx), bool))]:
        m = b_ & mask
        if m.sum() == 0:
            continue
        qs = boot_quantiles(lag[m].astype(float), w_src[m], boot, (0.5,))
        rows.append({"class b, trigger": key, "n": int(m.sum()), "median_steps_from_rise_to_deletion": float(np.median(lag[m])),
                     "lo": np.nanpercentile(qs[:, 0], 2.5), "hi": np.nanpercentile(qs[:, 0], 97.5),
                     "quartile_1_steps": float(np.percentile(lag[m], 25)), "quartile_3_steps": float(np.percentile(lag[m], 75)),
                     "median_operational_time": float(np.median(lag_lam[m])),
                     "median_distance": float(np.median(t_dist[m])) if (t_dist[m] > 0).all() else np.nan,
                     "median_share_of_the_rise_made_in_that_one_step": float(np.median(one_jump[m])),
                     "median_q_before_the_rise": float(np.median(q[w_idx[m], jump_step[m]])),
                     "median_q_after_the_rise": float(np.median(q[w_idx[m], jump_step[m] + 1]))})
    timing = pd.DataFrame(rows)
    timing.to_csv(out / "origin_trigger_timing.csv", index=False)
    dist_rows = []
    for dd in range(1, NEIGHBOURS + 1):
        for kind_label, kind_code in (("wrong", 2), ("correct", 1), ("one of each", 3)):
            dist_rows.append({"distance": dd, "nearest_deletion_in_the_step_of_the_rise": kind_label,
                              "n_class_b": int((b_ & (t_dist == dd) & (t_kind == kind_code)).sum())})
    pd.DataFrame(dist_rows).to_csv(out / "origin_trigger_distance.csv", index=False)

    every_b = timing[timing["class b, trigger"] == "all of class b"]
    s3 = {"n_sources": S, "n wrong deletions": len(w_idx), "threshold": Q_HIGH,
          "classes": {key: list(v) for key, v in result.items()},
          "classes at other thresholds": {str(t): {key: v[0] for key, v in origin[t][3].items()} for t in Q_SENSITIVITY},
          "class b, what happened in the step of the largest rise of q": {
              key: list(v) for key, v in trigger["b became suspicious during sampling"].items()},
          "baseline: share of original letter-steps with a deletion within 2 letters in the step": base[0, 2] / base[0, 0],
          "baseline: share of original letter-steps with a deletion within 10 letters in the step": base[0, 4] / base[0, 0]}
    if len(every_b):
        r = every_b.iloc[0]
        s3["class b, steps from the rise to the deletion: median [CI], quartiles"] = [
            r.median_steps_from_rise_to_deletion, r.lo, r.hi, r.quartile_1_steps, r.quartile_3_steps]
        s3["class b, median operational time from the rise to the deletion"] = r.median_operational_time
        s3["class b, median share of the total rise made in one step"] = r.median_share_of_the_rise_made_in_that_one_step
        s3["class b, median q before and after that step"] = [r.median_q_before_the_rise, r.median_q_after_the_rise]
    for key, mask in (("right after a wrong deletion within 2 letters", near2 & (t_kind >= 2)),
                      ("right after a correct deletion within 2 letters that uncovered an earlier wrong deletion", near2 & (t_kind == 1) & damaged),
                      ("right after a correct deletion within 2 letters, no earlier wrong deletion nearby", near2 & (t_kind == 1) & ~damaged)):
        s3[f"share of ALL wrong deletions that became suspicious {key}"] = list(shares({"x": b_ & mask})["x"])
    q_all, q_name = exact_posterior(args.study)
    if q_all is not None:
        # how ambiguous was the wrongly deleted original in the source itself? The perfect sampler deletes an original
        # with its exact posterior probability, so the sum of that probability is what it gets wrong.
        q_exact = q_all[:n].astype(np.float64)
        s3["exact posterior used"] = f"{q_name} of posterior.npz"
        cls_all = np.digitize(q_exact, EXACT_EDGES)
        cls_w = cls_all[w_idx]
        origin_masks = {"a ambiguous for the model from the start": a,
                        "b after a wrong deletion within 2 letters": b_ & near2 & (t_kind >= 2),
                        "b after a correct deletion within 2 letters": b_ & near2 & (t_kind == 1),
                        "b nearest deletion further away or none": b_ & ~near2,
                        "c never suspicious": c}
        rows, by_exact = [], {}
        for ci_, label in enumerate(EXACT_CLASSES):
            in_class = cls_w == ci_
            originals = ~ins & (cls_all == ci_)
            expected = np.bincount(seq[originals], weights=q_exact[originals], minlength=S)
            observed = np.bincount(w_src[in_class], minlength=S).astype(float)
            per_source, ps_b = boot.ratio(np.stack([observed, expected, observed - expected], 1), np.ones((S, 1)))
            lo, hi = ci(ps_b)
            res = shares(origin_masks, within=in_class)
            row = {"original": label, "n_sources": S, "originals_per_source": float(originals.sum() / S),
                   "model_wrong_deletions_per_source": per_source[0], "lo": lo[0], "hi": hi[0],
                   "perfect_sampler_expects_per_source": per_source[1],
                   "surplus_of_the_model_per_source": per_source[2], "surplus_lo": lo[2], "surplus_hi": hi[2],
                   "n_wrong_deletions": int(in_class.sum())}
            for key, (point, r_lo, r_hi, n_k) in res.items():
                row[f"share: {key}"], row[f"n: {key}"] = point, n_k
            rows.append(row)
            by_exact[label] = {"model wrong deletions per source": [per_source[0], lo[0], hi[0]],
                               "the perfect sampler expects": per_source[1],
                               "surplus of the model per source": [per_source[2], lo[2], hi[2]],
                               "n wrong deletions": int(in_class.sum()),
                               "origin of these wrong deletions (shares)": {key: [v[0], v[1], v[2]] for key, v in res.items()}}
        pd.DataFrame(rows).to_csv(out / "origin_by_exact_posterior.csv", index=False)
        s3["wrong deletions by how ambiguous the original is in the source (exact posterior at t = 0)"] = by_exact
        # the surplus of the model over the perfect sampler, per source, split by the origin of the wrong deletion:
        # originals that are not ambiguous in the source (exact posterior below 0.1) are almost never deleted by it
        not_amb = cls_w < len(EXACT_CLASSES) - 1
        surplus = {}
        for key, mask in origin_masks.items():
            point, bb = boot.ratio(np.bincount(w_src[mask & not_amb], minlength=S).astype(float), np.ones(S))
            surplus[key] = [point, *ci(bb), int((mask & not_amb).sum())]
        s3["wrong deletions per source of originals that are NOT ambiguous in the source, by origin"] = surplus
        originals = ~ins & (cls_all < len(EXACT_CLASSES) - 1)
        s3["... the perfect sampler expects per source"] = float(q_exact[originals].sum() / S)
    summary["3 origin"] = s3
    return {"del_step": del_step, "q": q, "S": S, "p_scale": p_scale, "n_wrong": len(w_idx),
            "origin": (result, trigger["b became suspicious during sampling"]), "follow": follow, "h_end": h_end}


def part_followups(args, sources: Sources, eligible: np.ndarray, m100, i2, qdata, out: Path, summary: dict):
    """What follows a first wrong deletion, by break size, without any matching: the trained model against the
    sampler with exact rates (all sources). Then figure 5."""
    print("part 2b: wrong deletions that follow a first wrong deletion, model against exact rates")
    groups = {"first wrong deletion, break 0": [base_group(1, 1, 0)], "first wrong deletion, break 1": [base_group(1, 1, 1)],
              "first wrong deletion, break 2": [base_group(1, 1, 2)],
              "correct deletion, no earlier wrong deletion nearby": [base_group(1, 0, b) for b in range(4)]}
    rows, results, per_source, quoted = [], {}, {}, {}
    also = n_covered(i2, sources) if i2 is not None else None
    for label, data in (("M", m100), ("I2", i2)):
        if data is None:
            continue
        del_step = data["del_step"].astype(np.int64)
        n = len(del_step)
        S = n_covered(data, sources)
        n_steps = int(data["steps"])
        offsets = sources.offsets[: S + 1]
        ins = sources.inserted[:n]
        group, brk, isolated, first = event_groups(del_step, offsets, ins, sources.letters[:n].astype(np.int64), eligible, n_steps)
        wrong = (del_step >= 0) & ~ins
        seq = np.repeat(np.arange(S), np.diff(offsets))
        group[~isolated] = -1
        stats = np.zeros((S, 16, 6))
        nb_followups(group, del_step, offsets, ins, n_steps, NEIGHBOURS, stats)
        # every wrong deletion is either the first of its neighbourhood or comes after another one nearby
        kinds_of_wrong = np.stack([np.bincount(seq[mask], minlength=S).astype(float)
                                   for mask in (wrong, wrong & first, wrong & ~first)], 1)
        sets = source_sets(sources, S, also if label == "M" else None)
        quoted[label] = list(sets)
        for half, (lo_s, hi_s) in sets.items():
            boot = Boot(hi_s - lo_s, args.boot, args.seed)
            part = stats[lo_s:hi_s]
            in_half = (seq >= lo_s) & (seq < hi_s)
            composition = [float((wrong & in_half & (brk == v)).sum() / max((wrong & in_half).sum(), 1)) for v in range(3)]
            per_source[label, half] = kinds_of_wrong[lo_s:hi_s]
            mean_kinds, kinds_b = boot.ratio(kinds_of_wrong[lo_s:hi_s], np.ones((hi_s - lo_s, 1)))
            k_lo, k_hi = ci(kinds_b)
            entry = {"n_sources": hi_s - lo_s, "wrong deletions": int((wrong & in_half).sum()),
                     "wrong deletions per source": [mean_kinds[0], k_lo[0], k_hi[0]],
                     f"... the first of their neighbourhood (no earlier wrong deletion within {ISOLATION} letters), per source": [
                         mean_kinds[1], k_lo[1], k_hi[1]],
                     "... after an earlier wrong deletion nearby, per source": [mean_kinds[2], k_lo[2], k_hi[2]],
                     "share of the wrong deletions with break 0, 1, 2": composition,
                     "isolated first wrong deletions per source": float(part[:, [base_group(1, 1, b) for b in range(4)], 0].sum() / (hi_s - lo_s))}
            for name, members in groups.items():
                v = part[:, members].sum(1)
                point, b = boot.ratio(v[:, 1:5], v[:, :1])
                lo, hi = ci(b)
                entry[name] = {"n events": int(v[:, 0].sum()),
                               "P(a wrong deletion follows within 2 letters)": [point[1], lo[1], hi[1]],
                               "P(a wrong deletion follows within 5 letters)": [point[2], lo[2], hi[2]],
                               f"P(a wrong deletion follows within {NEIGHBOURS} letters)": [point[3], lo[3], hi[3]],
                               f"wrong deletions that follow within {NEIGHBOURS} letters, per event": [point[0], lo[0], hi[0]]}
                # the same after a correct deletion is what the neighbourhood of any deletion gives: the excess over it
                ref = part[:, groups["correct deletion, no earlier wrong deletion nearby"]].sum(1)
                ref_point, ref_b = boot.ratio(ref[:, 1:5], ref[:, :1])
                excess, excess_b = point - ref_point, b - ref_b
                e_lo, e_hi = ci(excess_b)
                if not name.startswith("correct"):
                    entry[name][f"... minus the same after a correct deletion (wrong deletions within {NEIGHBOURS} letters)"] = [
                        excess[0], e_lo[0], e_hi[0]]
                rows.append({"sampler": label, "sources": half, "n_sources": hi_s - lo_s, "event": name, "n_events": int(v[:, 0].sum()),
                             "p_followed_within_2": point[1], "p2_lo": lo[1], "p2_hi": hi[1],
                             "p_followed_within_5": point[2], "p5_lo": lo[2], "p5_hi": hi[2],
                             "p_followed_within_10": point[3], "p10_lo": lo[3], "p10_hi": hi[3],
                             "later_wrong_deletions_within_10_per_event": point[0], "n_lo": lo[0], "n_hi": hi[0],
                             "the_same_minus_after_a_correct_deletion": excess[0], "excess_lo": e_lo[0], "excess_hi": e_hi[0]})
            results[f"{NAMES[label]}, {half}"] = entry
    common = [half for half in quoted.get("I2", []) if ("M", half) in per_source][:1]
    if common:  # the same sources for both samplers: where does the model make more wrong deletions?
        a, b = per_source["M", common[0]], per_source["I2", common[0]]
        boot = Boot(len(a), args.boot, args.seed)
        diff, diff_b = boot.ratio(a - b, np.ones((len(a), 1)))
        d_lo, d_hi = ci(diff_b)
        results[f"trained model minus exact rates, {common[0]}"] = {
            "n_sources": len(a),
            "wrong deletions per source": [diff[0], d_lo[0], d_hi[0]],
            "... first of their neighbourhood": [diff[1], d_lo[1], d_hi[1]],
            "... after an earlier wrong deletion nearby": [diff[2], d_lo[2], d_hi[2]],
            "share of the surplus that comes after an earlier wrong deletion nearby": [
                diff[2] / diff[0], *np.nanpercentile(diff_b[:, 2] / diff_b[:, 0], [2.5, 97.5])]}
    if rows:
        table = pd.DataFrame(rows)
        table.to_csv(out / "followups_model_vs_exact_rates.csv", index=False)
        summary["2b follow-ups without matching"] = results

    if qdata is None and not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    d = np.arange(1, NEIGHBOURS + 1)
    text = "What follows a wrong deletion."
    if qdata is not None:
        follow, h_end = qdata["follow"], qdata["h_end"]
        for name, label, colour, marker in SERIES:
            point, lo, hi, _ = follow[name]
            ax.fill_between(d, lo[0, :, h_end], hi[0, :, h_end], color=colour, alpha=0.18, lw=0)
            ax.plot(d, point[0, :, h_end], marker=marker, color=colour, label=label)
        ax.axhline(0, color=COLORS[6], lw=1, ls="--", label="dashed line: as often as matched controls")
        ax.axvline(2.5, color=COLORS[7], lw=1, ls=":")
        ax.text(2.45, ax.get_ylim()[1], "reach of the grammar: 2 letters", rotation=90, ha="right", va="top", fontsize=8, color="#444444")
        ax.set_xticks(d)
        ax.set_xlabel("distance of the original from the deleted letter (letters present)")
        ax.set_ylabel("probability that the original is deleted later\nminus the same for matched controls")
        ax.set_title(f"trained model: how far the damage reaches ({qdata['S']} sources)")
        ax.legend(loc="upper right", fontsize=7.5)
        f2 = follow["first wrong deletion, break 2"][0][0, :, h_end]
        text = (f"After a wrong deletion that breaks both windows the original next to it is deleted {f2[0]:+.2f} more often than a "
                f"matched control, at distance 2 {f2[1]:+.2f}, at distance 5 {f2[4]:+.2f}.")
    ax = axes[1]
    if rows:
        use = {}
        if "M" in quoted:
            use["M"] = common[0] if common else quoted["M"][0]
        if "I2" in quoted:
            use["I2"] = quoted["I2"][0]
        rep = pd.concat([table[(table.sampler == k) & (table.sources == v)] for k, v in use.items()])
        names = list(groups)
        x = np.arange(len(names))
        samplers = [v for v in ("M", "I2") if v in set(rep.sampler)]
        width = 0.8 / len(samplers)
        for i, label in enumerate(samplers):
            r = rep[rep.sampler == label].set_index("event").loc[names]
            pos = x - 0.4 + width * (i + 0.5)
            ax.bar(pos, r.p_followed_within_5, width=width * 0.94, color=SAMPLER_COLOR[label],
                   label=f"{NAMES[label]} ({use[label].split(' (')[0]})")
            ax.errorbar(pos, r.p_followed_within_5, yerr=[r.p_followed_within_5 - r.p5_lo, r.p5_hi - r.p_followed_within_5],
                        fmt="none", ecolor=COLORS[6], lw=1)
            for xi, value, top in zip(pos, r.p_followed_within_5, r.p5_hi):
                ax.text(xi, top + 0.01, f"{100 * value:.0f} %", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["wrong deletion,\nbreak 0", "wrong deletion,\nbreak 1", "wrong deletion,\nbreak 2",
                            "correct deletion\n(for comparison)"], fontsize=8.5)
        ax.set_ylabel("probability that a wrong deletion follows\nwithin 5 letters (no matching)")
        ax.set_ylim(0, 1.08)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title("the same reaction with exact rates?")
        if len(samplers) == 2:
            m = rep[rep.sampler == "M"].set_index("event").p_followed_within_5
            e = rep[rep.sampler == "I2"].set_index("event").p_followed_within_5
            key = "first wrong deletion, break 2"
            text += (f" A sampler with exact rates reacts to a broken grammar too: a further wrong deletion within 5 letters follows in "
                     f"{100 * e[key]:.0f} % of its break-2 wrong deletions, {100 * m[key]:.0f} % for the model.")
    headline(fig, text + " Isolated first wrong deletions; bands and bars: 95 % bootstrap over sources.")
    save_fig(fig, out / "fig5_followup_deletions.png")


def part_exposure(args, sources: Sources, m100, out: Path, summary: dict, origin):
    """Expected against observed wrong deletions by q at t = 0 (all sources of m100.npz) and figure 6."""
    print("part 3b: expected and observed wrong deletions by q at t = 0")
    s = summary.setdefault("3 origin", {})
    table = None
    if m100 is not None:
        n = len(m100["del_step"])
        S = n_covered(m100, sources)
        n_steps = int(m100["steps"])
        hazard, _, _ = schedule(args.run, n_steps)
        p_step = hazard / n_steps
        ins = sources.inserted[:n]
        seq = np.repeat(np.arange(S), np.diff(sources.offsets[: S + 1]))
        original = ~ins
        q0 = m100["q0"].astype(np.float64)
        wrong = (m100["del_step"] >= 0) & original

        def stay(qv):  # probability of a deletion if q stayed at this value through all steps
            log_survival = np.zeros(len(qv))
            for p in p_step:
                log_survival += np.log1p(-np.clip(qv * p, 0, 1 - 1e-12))
            return 1 - np.exp(log_survival)

        static = np.zeros(n)
        static[original] = stay(q0[original])
        floor = np.zeros(n)
        floor[original] = stay(np.minimum(q0[original], 1e-3))
        cell = np.digitize(q0, Q0_BINS) - 1
        bin_names = [f"[{Q0_BINS[c]:g}, {min(Q0_BINS[c + 1], 1):g})" for c in range(len(Q0_BINS) - 1)]
        rows = []
        halves = {half: (seq >= lo_s) & (seq < hi_s) for half, (lo_s, hi_s) in source_sets(sources, S).items()}
        quote = next(iter(halves))
        for half, in_half in halves.items():
            src_ids = np.unique(seq[in_half])
            boot = Boot(len(src_ids), args.boot, args.seed)
            local = seq - src_ids[0]
            n_orig = np.bincount(local[in_half & original], minlength=len(src_ids)).astype(float)
            total_wrong = np.bincount(local[in_half & wrong], minlength=len(src_ids)).astype(float)
            for c in list(range(len(Q0_BINS) - 1)) + [-1]:
                mask = in_half & original & ((cell == c) if c >= 0 else True)
                per = [np.bincount(local[mask], weights=v[mask].astype(float), minlength=len(src_ids))
                       for v in (np.ones(n), static, m100["exposure"], wrong, floor)]
                cols = np.stack(per[1:], 1)
                share, sb = boot.ratio(per[3], total_wrong)
                per300, pb = boot.ratio(cols * 300, n_orig[:, None])
                lo, hi = ci(pb)
                rows.append({"sources": half, "n_sources": len(src_ids), "q0_bin": "all" if c < 0 else bin_names[c],
                             "originals": int(per[0].sum()), "expected_if_q_stayed_at_q0": per[1].sum(),
                             "expected_from_the_realised_rates (exposure)": per[2].sum(), "observed_wrong_deletions": int(per[3].sum()),
                             "share_of_all_wrong_deletions": share, "share_lo": np.percentile(sb, 2.5), "share_hi": np.percentile(sb, 97.5),
                             "per_300_originals_if_q_stayed": per300[0], "per_300_originals_exposure": per300[1],
                             "per_300_originals_observed": per300[2], "obs_lo": lo[2], "obs_hi": hi[2],
                             "per_300_originals_floor_only (q0 capped at 0.001)": per300[3]})
        table = pd.DataFrame(rows)
        table.to_csv(out / "origin_expected_by_q0.csv", index=False)
        rep = table[table.sources == quote]
        every = rep[rep.q0_bin == "all"].iloc[0]
        low = rep[rep.q0_bin.isin([bin_names[c] for c in range(len(Q0_BINS) - 1) if Q0_BINS[c + 1] <= 1e-3])]
        s["expected counts (m100)"] = {
            "sources": quote,
            "n_sources": int(every.n_sources), "originals": int(every.originals),
            "wrong deletions per 300 originals, observed": [every.per_300_originals_observed, every.obs_lo, every.obs_hi],
            "per 300 originals if every original had q = 0.001 throughout (P4)": 300 * float(stay(np.array([1e-3]))[0]),
            "per 300 originals from the real q0 capped at 0.001 (the floor alone)": every["per_300_originals_floor_only (q0 capped at 0.001)"],
            "per 300 originals if q stayed at q0": every.per_300_originals_if_q_stayed,
            "per 300 originals from the realised rates": every.per_300_originals_exposure,
            "originals with q0 < 0.001: share of all originals": low.originals.sum() / every.originals,
            "originals with q0 < 0.001: observed wrong deletions": int(low.observed_wrong_deletions.sum()),
            "originals with q0 < 0.001: expected if q stayed at q0": low.expected_if_q_stayed_at_q0.sum(),
            "originals with q0 < 0.001: share of all wrong deletions": low.observed_wrong_deletions.sum() / every.observed_wrong_deletions,
        }

    if origin is None and table is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    head = "Where the wrong deletions come from."
    if origin is not None:
        result, trigger = origin["origin"]
        b_share = result["b became suspicious during sampling"][0]
        key_w = "after a wrong deletion within 2 letters"
        key_cd = "after a correct deletion within 2 letters, an earlier wrong deletion within 2 letters"
        key_c = "after a correct deletion within 2 letters, no earlier wrong deletion within 2 letters"
        rest = trigger["nearest deletion of that step 3 to 10 letters away"][0] + trigger["no deletion within 10 letters in that step"][0]
        parts = [("ambiguous from the start (q at t = 0 at least 0.1)", result["a ambiguous from the start"][0], SAMPLER_COLOR["M"]),
                 ("became suspicious right after a wrong deletion within 2 letters", b_share * trigger[key_w][0], OTHER_COLORS[1]),
                 ("... right after a correct deletion within 2 letters that uncovered\nan earlier wrong deletion", b_share * trigger[key_cd][0], OTHER_COLORS[0]),
                 ("... right after a correct deletion within 2 letters,\nno wrong deletion nearby before", b_share * trigger[key_c][0], OTHER_COLORS[2]),
                 ("... when the nearest deletion was further away", b_share * rest, OTHER_COLORS[3]),
                 ("never suspicious (q below 0.1 even when deleted)", result["c deleted although not suspicious"][0], OTHER_COLORS[4])]
        bottom = 0.0
        slots = np.linspace(0.08, 0.95, len(parts))
        for (label, value, colour), slot in zip(parts, slots):
            ax.bar(0, value, bottom=bottom, color=colour, width=0.5, edgecolor="white", linewidth=1.5)
            ax.plot([0.27, 0.42], [bottom + value / 2, slot], color="#555555", lw=0.8)
            ax.text(0.45, slot, f"{100 * value:.0f} %  {label}", va="center", fontsize=8.5)
            bottom += value
        ax.set_xlim(-0.3, 3.2)
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
        ax.set_ylabel("share of the wrong deletions")
        ax.set_title(f"origin of {origin['n_wrong']:,} wrong deletions ({origin['S']} sources)")
        ax.grid(False)
        head = (f"{100 * result['a ambiguous from the start'][0]:.0f} % of the wrong deletions hit originals that the model suspected "
                f"from the start (q at t = 0 at least 0.1), {100 * b_share:.0f} % hit originals that became suspicious during sampling "
                f"({100 * b_share * trigger[key_w][0]:.0f} % right after a wrong deletion next to them), "
                f"{100 * result['c deleted although not suspicious'][0]:.0f} % hit originals that were never suspicious.")
    ax = axes[1]
    if table is not None:
        rep = table[(table.sources == quote) & (table.q0_bin != "all")]
        x = np.arange(len(rep))
        ax.bar(x - 0.2, rep.observed_wrong_deletions, width=0.38, color=SAMPLER_COLOR["M"], label="observed wrong deletions")
        ax.bar(x + 0.2, rep.expected_if_q_stayed_at_q0, width=0.38, color=OTHER_COLORS[3],
               label="expected if q had stayed at its value at t = 0")
        ax.set_xticks(x)
        ax.set_xticklabels([b.split(",")[0].strip("[") for b in rep.q0_bin], rotation=45, ha="right")
        ax.set_xlabel("q of the original at t = 0 (lower edge of the bin)")
        ax.set_ylabel("wrong deletions")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title(f"by the q at the start ({int(rep.n_sources.iloc[0])} sources, {int(rep.originals.sum()):,} originals)")
        low = s["expected counts (m100)"]
        head += (f" The {100 * low['originals with q0 < 0.001: share of all originals']:.0f} % of the originals with q below 0.001 at the "
                 f"start (the rate floor) give {100 * low['originals with q0 < 0.001: share of all wrong deletions']:.0f} % of the wrong deletions.")
    headline(fig, head)
    save_fig(fig, out / "fig6_origin_of_wrong_deletions.png")


# ---- part 4: same-step coincidences -----------------------------------------------------------------------------


def cluster_frame(del_step, ins, letters, offsets, q=None, p_scale=None):
    """Clusters that hold a wrong deletion: size, first wrong deletion and what else was deleted in its step; and
    the same-step information of every wrong deletion."""
    S = len(offsets) - 1
    seq = np.repeat(np.arange(S), np.diff(offsets))
    deleted = del_step >= 0
    cid, n_clusters = nb_cluster_ids(deleted, ins, offsets, GAP)
    wrong = deleted & ~ins
    w = np.flatnonzero(wrong)
    start, end = offsets[seq], offsets[seq + 1]
    has_q = q is not None
    co_correct, co_wrong, twin, predicted = nb_same_step(
        w, del_step, ins, letters, start, end, q if has_q else np.zeros((1, 1), np.float32), has_q,
        p_scale if has_q else np.zeros(1), 2)
    order = np.lexsort((w, del_step[w], cid[w]))
    ws, cs = w[order], cid[w][order]
    is_first = np.r_[True, cs[1:] != cs[:-1]]
    first_pos = np.flatnonzero(is_first)
    n_x = np.diff(np.r_[first_pos, len(ws)])
    second_step = np.where(n_x >= 2, del_step[ws[np.minimum(first_pos + 1, len(ws) - 1)]], -1)
    pos_in_w = order[first_pos]
    clusters = pd.DataFrame({
        "source": seq[ws[first_pos]], "n_x": n_x, "first_step": del_step[ws[first_pos]],
        "co_correct": co_correct[pos_in_w], "co_wrong": co_wrong[pos_in_w], "twin": twin[pos_in_w],
        "predicted": predicted[pos_in_w], "tie": second_step == del_step[ws[first_pos]],
    })
    clusters["double"] = clusters.co_correct | clusters.co_wrong
    every = pd.DataFrame({"source": seq[w], "double": co_correct | co_wrong, "predicted": predicted})
    return clusters, every


def same_step_summary(clusters, every, src_lo, src_hi, args, label):
    """Shares with bootstrap over the sources src_lo .. src_hi - 1."""
    S = src_hi - src_lo
    boot = Boot(S, args.boot, args.seed)
    cl = clusters[(clusters.source >= src_lo) & (clusters.source < src_hi)]
    ev = every[(every.source >= src_lo) & (every.source < src_hi)]

    def share(mask, base, frame=cl):
        num = np.bincount(frame.source.to_numpy()[mask & base] - src_lo, minlength=S).astype(float)
        den = np.bincount(frame.source.to_numpy()[base] - src_lo, minlength=S).astype(float)
        point, b = boot.ratio(num, den)
        return [point, *ci(b), int((mask & base).sum()), int(base.sum())]

    def mean(values, base, frame=cl):
        num = np.bincount(frame.source.to_numpy()[base] - src_lo, weights=values[base], minlength=S)
        den = np.bincount(frame.source.to_numpy()[base] - src_lo, minlength=S).astype(float)
        point, b = boot.ratio(num, den)
        return [point, *ci(b), int(base.sum())]

    multi = (cl.n_x >= 2).to_numpy()
    double = cl.double.to_numpy()
    tie = cl.tie.to_numpy() & ~double
    single = ~double & ~tie
    everything = np.ones(len(cl), bool)
    res = {"data": label, "n_sources": S, "clusters with a wrong deletion": len(cl), "of which with 2 or more": int(multi.sum()),
           "share of multi clusters that start with a double event (another deletion within 2 letters in the same step)": share(double, multi),
           "... the partner is a correct deletion": share(cl.co_correct.to_numpy() & ~cl.co_wrong.to_numpy(), multi),
           "... the partner is a wrong deletion": share(cl.co_wrong.to_numpy(), multi),
           "... the partner is the adjacent identical letter (both twins deleted)": share(cl.twin.to_numpy(), multi),
           "share of multi clusters whose first two wrong deletions fall in one step but more than 2 letters apart": share(tie, multi),
           "share of multi clusters that start with a single wrong deletion, the others follow in later steps": share(single, multi),
           "share of ALL clusters with a wrong deletion that start with a double event": share(double, everything),
           "P(2 or more wrong deletions | starts with a single wrong deletion)": share(multi, single | tie),
           "P(2 or more wrong deletions | double with a correct deletion)": share(multi, cl.co_correct.to_numpy() & ~cl.co_wrong.to_numpy()),
           "mean wrong deletions per cluster | starts single": mean(cl.n_x.to_numpy().astype(float), single | tie),
           "mean wrong deletions per cluster | starts with a double event": mean(cl.n_x.to_numpy().astype(float), double),
           "share of all wrong deletions that lie in clusters started by a double event": None,
           "share of all wrong deletions made in the same step as another deletion within 2 letters": share(
               ev.double.to_numpy(), np.ones(len(ev), bool), ev)}
    num = np.bincount(cl.source.to_numpy()[double] - src_lo, weights=cl.n_x.to_numpy()[double], minlength=S)
    den = np.bincount(cl.source.to_numpy() - src_lo, weights=cl.n_x.to_numpy(), minlength=S)
    point, b = boot.ratio(num, den)
    res["share of all wrong deletions that lie in clusters started by a double event"] = [point, *ci(b), int(num.sum()), int(den.sum())]
    if np.isfinite(cl.predicted).any():
        pred = cl.predicted.to_numpy()
        res["predicted by independent draws at the logged rates: P(double) at the first wrong deletion of a multi cluster"] = mean(pred, multi)
        res["predicted P(double) at the first wrong deletion of any cluster"] = mean(pred, everything)
        res["predicted P(double) for all wrong deletions (must equal the observed share)"] = mean(
            ev.predicted.to_numpy(), np.ones(len(ev), bool), ev)
    return res


def part_same_step(args, sources: Sources, qdata, m100, i2, out: Path, summary: dict):
    print("part 4: same-step coincidences")
    results, rows = {}, []

    also = n_covered(i2, sources) if i2 is not None else None

    def run(label, del_step, q=None, p_scale=None, like_for_like=None):
        n = len(del_step)
        S = int(np.searchsorted(sources.offsets, n, side="right") - 1)
        offsets = sources.offsets[: S + 1]
        clusters, every = cluster_frame(del_step.astype(np.int64), sources.inserted[:n], sources.letters[:n].astype(np.int64), offsets, q, p_scale)
        for name, (lo, hi) in source_sets(sources, S, like_for_like).items():
            results[f"{label}, {name}"] = same_step_summary(clusters, every, lo, hi, args, f"{label}, {name}")

    if qdata is not None:
        run("model with logged rates", qdata["del_step"], qdata["q"], qdata["p_scale"])
    if m100 is not None:
        run("model", m100["del_step"], like_for_like=also)
    if i2 is not None:
        run("exact rates (I2)", i2["del_step"])
    for name, res in results.items():
        for key, value in res.items():
            if isinstance(value, list):
                rows.append({"data": name, "quantity": key, "value": value[0], "lo": value[1], "hi": value[2],
                             "n": value[3], "n_base": value[4] if len(value) > 4 else value[3]})
            elif key != "data" and value is not None:
                rows.append({"data": name, "quantity": key, "value": value, "lo": np.nan, "hi": np.nan, "n": np.nan, "n_base": np.nan})
    if not rows:
        return
    pd.DataFrame(rows).to_csv(out / "same_step.csv", index=False)
    summary["4 same step"] = results

    key_double = "share of multi clusters that start with a double event (another deletion within 2 letters in the same step)"
    key_single = "share of multi clusters that start with a single wrong deletion, the others follow in later steps"
    key_tie = "share of multi clusters whose first two wrong deletions fall in one step but more than 2 letters apart"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    shown = [name for name in results if "exploring" not in name]
    y = np.arange(len(shown))[::-1]
    left = np.zeros(len(shown))
    for key, label, colour in ((key_single, "one wrong deletion first, the others in later steps", OTHER_COLORS[0]),
                               (key_double, "a double event: another deletion within 2 letters in the same step", OTHER_COLORS[1]),
                               (key_tie, "two wrong deletions in one step, more than 2 letters apart", OTHER_COLORS[3])):
        values = np.array([results[name][key][0] for name in shown])
        ax.barh(y, values, left=left, color=colour, height=0.55, edgecolor="white", linewidth=1.5, label=label)
        for yi, value, l in zip(y, values, left):
            if value > 0.05:
                ax.text(l + value / 2, yi, f"{100 * value:.0f} %", ha="center", va="center", color="#111111", fontsize=9, fontweight="bold")
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([f"{name.split(' (')[0].replace(', sources', chr(10) + 'sources')}\n{results[name]['of which with 2 or more']:,} clusters"
                        if not name.startswith("exact") else
                        f"exact rates\n{name.split(', ')[1]}\n{results[name]['of which with 2 or more']:,} clusters"
                        for name in shown], fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlabel("share of the clusters with 2 or more wrong deletions")
    ax.grid(False)
    ax.set_title("how clusters with several wrong deletions start")
    legend_below(fig, ax, ncols=1)
    ax = axes[1]
    name = next((k for k in results if "logged" in k), None)
    tail = ""
    if name is not None:
        res = results[name]
        bars = [("all wrong\ndeletions", res["share of all wrong deletions made in the same step as another deletion within 2 letters"],
                 res["predicted P(double) for all wrong deletions (must equal the observed share)"]),
                ("first wrong deletion\nof any cluster", res["share of ALL clusters with a wrong deletion that start with a double event"],
                 res["predicted P(double) at the first wrong deletion of any cluster"]),
                ("first wrong deletion\nof a cluster with\n2 or more", res[key_double],
                 res["predicted by independent draws at the logged rates: P(double) at the first wrong deletion of a multi cluster"])]
        x = np.arange(len(bars))
        obs = np.array([b[1][:3] for b in bars])
        pre = np.array([b[2][:3] for b in bars])
        ax.bar(x - 0.2, obs[:, 0], width=0.38, color=SAMPLER_COLOR["M"], label="observed (trained model)")
        ax.errorbar(x - 0.2, obs[:, 0], yerr=[obs[:, 0] - obs[:, 1], obs[:, 2] - obs[:, 0]], fmt="none", ecolor=COLORS[6], lw=1)
        ax.bar(x + 0.2, pre[:, 0], width=0.38, color=OTHER_COLORS[3], label="expected from independent draws\nat the logged rates")
        for xi, o, top, p in zip(x, obs[:, 0], obs[:, 2], pre[:, 0]):
            ax.text(xi - 0.2, top + 0.001, f"{100 * o:.1f} %", ha="center", va="bottom", fontsize=8)
            ax.text(xi + 0.2, p + 0.001, f"{100 * p:.1f} %", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
        ax.set_ylim(0, 1.35 * max(obs[:, 2].max(), pre[:, 0].max()))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("made in the same step as another\ndeletion within 2 letters")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.set_title(f"same-step coincidences against chance ({res['n_sources']} sources)")
        tail = (f" Independent draws at the logged rates predict {100 * pre[2, 0]:.1f} % of such starts, "
                f"{100 * obs[2, 0]:.1f} % are observed.")
    main = next((k for k in results if k.startswith("model, ")), shown[0])
    headline(fig, f"How clusters with several wrong deletions start: {100 * results[main][key_single][0]:.0f} % with a single wrong "
             f"deletion that is followed by others in later steps, {100 * results[main][key_double][0]:.0f} % with a double event "
             f"(another deletion within 2 letters in the same step)." + tail)
    save_fig(fig, out / "fig7_same_step.png")


# ---- part 5: chains ---------------------------------------------------------------------------------------------

CHAIN_CLASSES = ("grows to the right", "grows to the left", "grows on both sides of the first", "separate seeds merge")


def chain_data(del_step, ins, offsets, lam, seed):
    S = len(offsets) - 1
    seq = np.repeat(np.arange(S), np.diff(offsets))
    deleted = del_step >= 0
    cid, _ = nb_cluster_ids(deleted, ins, offsets, GAP)
    wrong = deleted & ~ins
    target_pos = np.cumsum(~ins) - 1
    w = np.flatnonzero(wrong)
    size = np.bincount(cid[w], minlength=cid.max() + 2)
    w = w[size[cid[w]] >= 3]
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(w)), del_step[w], cid[w]))  # by chain, then time, ties in random order
    w = w[order]
    c = cid[w]
    starts = np.r_[np.flatnonzero(np.r_[True, c[1:] != c[:-1]]), len(w)] if len(w) else np.zeros(1, np.int64)
    return {"letters": w, "starts": starts, "pos": target_pos[w], "step": del_step[w], "source": seq[w][starts[:-1]],
            "size": np.diff(starts), "pool": del_step[wrong], "pool_source": seq[wrong], "lam": lam, "S": S}


def chain_summary(ch, src_lo, src_hi, args, label, out_rows):
    S = src_hi - src_lo
    boot = Boot(S, args.boot, args.seed)
    keep = (ch["source"] >= src_lo) & (ch["source"] < src_hi)
    n_chains = int(keep.sum())
    if n_chains < 5:
        return None
    cls, seeds, null_cls, null_seeds = nb_chains(ch["starts"], ch["pos"], SEED_REACH, 20, args.seed)
    src = ch["source"][keep] - src_lo
    res = {"data": label, "n_sources": S, "chains (clusters with 3 or more wrong deletions)": n_chains,
           "chains per 100 sources": 100 * n_chains / S, "median wrong deletions per chain": float(np.median(ch["size"][keep])),
           "mean wrong deletions per chain": float(np.mean(ch["size"][keep]))}
    den = np.bincount(src, minlength=S).astype(float)
    for c, name in enumerate(CHAIN_CLASSES):
        point, b = boot.ratio(np.bincount(src, weights=(cls[keep] == c).astype(float), minlength=S), den)
        null, nb_ = boot.ratio(np.bincount(src, weights=null_cls[keep, c], minlength=S), den)
        res[f"share: {name}"] = [point, *ci(b), int((cls[keep] == c).sum())]
        res[f"share under a random order of time: {name}"] = [null, *ci(nb_)]
        out_rows.append({"data": label, "quantity": f"share of chains: {name}", "value": point, "lo": ci(b)[0], "hi": ci(b)[1],
                         "random_order": null, "n": n_chains})
    point, b = boot.ratio(np.bincount(src, weights=seeds[keep].astype(float), minlength=S), den)
    null, nb_ = boot.ratio(np.bincount(src, weights=null_seeds[keep], minlength=S), den)
    res["mean number of seeds per chain"] = [point, *ci(b)]
    res["mean number of seeds under a random order of time"] = [null, *ci(nb_)]
    out_rows.append({"data": label, "quantity": "mean seeds per chain", "value": point, "lo": ci(b)[0], "hi": ci(b)[1],
                     "random_order": null, "n": n_chains})
    # single-seed chains: is the first wrong deletion at an end of the final hole more often than chance (2 / size)?
    single = keep & (seeds == 1)
    one_sided = (cls <= 1) & single
    chance = np.where(single, 2.0 / ch["size"], 0.0)
    den1 = np.bincount(ch["source"][single] - src_lo, minlength=S).astype(float)
    point, b = boot.ratio(np.bincount(ch["source"][one_sided] - src_lo, minlength=S).astype(float), den1)
    null, nb_ = boot.ratio(np.bincount(ch["source"][single] - src_lo, weights=chance[single], minlength=S), den1)
    res["single-seed chains: share that grow in one direction only"] = [point, *ci(b), int(single.sum())]
    res["single-seed chains: the same if the first deletion were a random member (2 / size)"] = [null, *ci(nb_)]
    right = (cls == 0) & single
    point, b = boot.ratio(np.bincount(ch["source"][right] - src_lo, minlength=S).astype(float),
                          np.bincount(ch["source"][one_sided] - src_lo, minlength=S).astype(float))
    res["one-directional chains: share that grow to the right"] = [point, *ci(b), int(one_sided.sum())]

    # time between successive wrong deletions of a chain
    inside = np.ones(len(ch["step"]), bool)
    inside[ch["starts"][:-1]] = False  # not the first of its chain
    chain_of = np.repeat(np.arange(len(ch["size"])), ch["size"])
    rank = np.arange(len(chain_of)) - ch["starts"][:-1][chain_of]
    use = inside & keep[chain_of]
    gap = (ch["step"] - np.r_[0, ch["step"][:-1]])[use].astype(float)
    lam = ch["lam"]
    gap_lam = (lam[ch["step"]] - lam[np.r_[0, ch["step"][:-1]]])[use]
    gap_src = ch["source"][chain_of][use] - src_lo
    first_gap = rank[use] == 1
    # chance: the same number of times drawn from all wrong deletions of the same sources
    rng = np.random.default_rng(args.seed + 5)
    pool_keep = (ch["pool_source"] >= src_lo) & (ch["pool_source"] < src_hi)
    pool = ch["pool"][pool_keep]
    null_gap, null_lam = [], []
    for size in np.unique(ch["size"][keep]):
        count = int((ch["size"][keep] == size).sum())
        draws = np.sort(rng.choice(pool, size=(count * 5, int(size))), axis=1)
        null_gap.append(np.diff(draws, axis=1).ravel())
        null_lam.append(np.diff(lam[draws], axis=1).ravel())
    null_gap, null_lam = np.concatenate(null_gap), np.concatenate(null_lam)
    qs = (0.25, 0.5, 0.75)
    for name, values, mask in (("steps between successive wrong deletions of a chain", gap, np.ones(len(gap), bool)),
                               ("operational time between successive wrong deletions", gap_lam, np.ones(len(gap), bool)),
                               ("steps from the first to the second wrong deletion", gap, first_gap),
                               ("steps between later wrong deletions", gap, ~first_gap)):
        b = boot_quantiles(values[mask], gap_src[mask], boot, qs)
        point = np.quantile(values[mask], qs)
        res[f"{name}: quartiles"] = [float(v) for v in point]
        res[f"{name}: median"] = [float(point[1]), float(np.nanpercentile(b[:, 1], 2.5)), float(np.nanpercentile(b[:, 1], 97.5)), int(mask.sum())]
        out_rows.append({"data": label, "quantity": f"{name}: median", "value": point[1], "lo": np.nanpercentile(b[:, 1], 2.5),
                         "hi": np.nanpercentile(b[:, 1], 97.5), "random_order": np.nan, "n": int(mask.sum())})
    res["steps between successive wrong deletions if their times were unrelated: quartiles"] = [float(v) for v in np.quantile(null_gap, qs)]
    res["operational time between them if their times were unrelated: quartiles"] = [float(v) for v in np.quantile(null_lam, qs)]
    for limit in (0, 1, 5):
        den_g = np.bincount(gap_src, minlength=S).astype(float)
        point, b = boot.ratio(np.bincount(gap_src, weights=(gap <= limit).astype(float), minlength=S), den_g)
        res[f"share of successive wrong deletions at most {limit} steps apart"] = [point, *ci(b), float((null_gap <= limit).mean())]
    res["_gaps"] = (gap, null_gap)
    res["_classes"] = (cls[keep], null_cls[keep].mean(0))
    return res


def part_chains(args, sources: Sources, m100, i2, out: Path, summary: dict):
    print("part 5: chains")
    results, rows = {}, []
    also = n_covered(i2, sources) if i2 is not None else None
    for label, data in (("model", m100), ("exact rates (I2)", i2)):
        if data is None:
            continue
        n = len(data["del_step"])
        S = n_covered(data, sources)
        n_steps = int(data["steps"])
        _, lam, _ = schedule(args.run, n_steps)
        ch = chain_data(data["del_step"].astype(np.int64), sources.inserted[:n], sources.offsets[: S + 1], lam, args.seed)
        for name, (lo, hi) in source_sets(sources, S, also if label == "model" else None).items():
            res = chain_summary(ch, lo, hi, args, f"{label}, {name}", rows)
            if res is not None:
                results[f"{label}, {name}"] = res
    if not results:
        return
    pd.DataFrame(rows).to_csv(out / "chains.csv", index=False)
    summary["5 chains"] = {name: {k: v for k, v in res.items() if not k.startswith("_")} for name, res in results.items()}

    main = next((k for k in results if k.startswith("model, ")), next(iter(results)))
    res = results[main]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    x = np.arange(len(CHAIN_CLASSES))
    obs = np.array([res[f"share: {name}"][:3] for name in CHAIN_CLASSES])
    null = np.array([res[f"share under a random order of time: {name}"][0] for name in CHAIN_CLASSES])
    ax.bar(x - 0.2, obs[:, 0], width=0.38, color=SAMPLER_COLOR["M"], label="observed order of the deletions")
    ax.errorbar(x - 0.2, obs[:, 0], yerr=[obs[:, 0] - obs[:, 1], obs[:, 2] - obs[:, 0]], fmt="none", ecolor=COLORS[6], lw=1)
    ax.bar(x + 0.2, null, width=0.38, color=COLORS[7], label="the same holes, order of the deletions shuffled")
    for xi, o, top, nl in zip(x, obs[:, 0], obs[:, 2], null):
        ax.text(xi - 0.2, top + 0.005, f"{100 * o:.0f} %", ha="center", va="bottom", fontsize=8)
        ax.text(xi + 0.2, nl + 0.005, f"{100 * nl:.0f} %", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["one hole,\ngrows to\nthe right", "one hole,\ngrows to\nthe left", "one hole,\ngrows on\nboth sides",
                        "separate\nseeds that\nmerge"], fontsize=8.5)
    ax.set_ylim(0, 1.3 * max(obs[:, 2].max(), null.max()))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("share of the chains")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"how {res['chains (clusters with 3 or more wrong deletions)']:,} chains grow ({main.split(', ')[1].split(' (')[0]})")
    ax = axes[1]
    gap, null_gap = res["_gaps"]
    bins = np.r_[np.arange(0, 21) - 0.5, [25.5, 30.5, 40.5, 50.5, 100.5]]
    centres = (bins[:-1] + bins[1:]) / 2
    for values, colour, lab in ((gap, SAMPLER_COLOR["M"], "observed"), (null_gap, COLORS[7], "if the times of the deletions were unrelated")):
        hist = np.histogram(values, bins)[0] / len(values) / np.diff(bins)
        ax.plot(centres, hist, "o-", color=colour, label=lab, ms=3.5)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks([0, 1, 2, 5, 10, 20, 50, 100])
    ax.set_xticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("steps between successive wrong deletions of a chain (of 100 steps)")
    ax.set_ylabel("share of the gaps, per step")
    ax.legend(loc="upper right", fontsize=8.5)
    med = res["steps between successive wrong deletions of a chain: median"]
    quart = res["steps between successive wrong deletions of a chain: quartiles"]
    ax.set_title(f"time between successive wrong deletions ({med[3]:,} gaps)")
    lam_med = res["operational time between successive wrong deletions: median"][0]
    headline(fig, f"Chains of 3 or more wrong deletions: {100 * (obs[0, 0] + obs[1, 0]):.0f} % are one hole growing in one direction, "
             f"{100 * obs[2, 0]:.0f} % one hole growing on both sides, {100 * obs[3, 0]:.0f} % separate seeds that merge "
             f"({100 * null[3]:.0f} % if the order were random). Successive wrong deletions are {med[0]:.0f} steps apart "
             f"(median; quartiles {quart[0]:.0f} and {quart[2]:.0f}; operational time {lam_med:.2f}).")
    save_fig(fig, out / "fig8_chains.png")


# ---- main -------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=Path("runs/lap-01"))
    parser.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (over sources)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact", default=None, help="file in runs/error-study with the exact-rate sampler on all sources "
                                                      "(default: i2-<steps of m100>.npz)")
    args = parser.parse_args()
    args.run = args.run if args.run.is_absolute() else ROOT / args.run
    args.errors = args.run / "errors"
    args.study = ROOT / "runs" / "error-study"
    out = args.errors / "results" / "dependence"
    (out / "scratch").mkdir(parents=True, exist_ok=True)
    np.seterr(all="ignore")  # empty cells give NaN, which is what the tables should show
    started = time.time()
    sources = Sources.load(args.study / "sources.npz")
    eligible = transition_table(np.random.default_rng(0), 20, 5, 1.0) > 0
    summary: dict = {"run": args.run.name}
    missing: list = []

    part_pairs(args, sources, out, summary, missing)
    qdata = part_events(args, sources, eligible, out, summary, missing)
    m100 = load_npz(args.errors / "m100.npz")
    if m100 is None:
        missing.append(f"{args.errors / 'm100.npz'}: expected counts (part 3), parts 4 and 5 for the model skipped")
    steps = int(m100["steps"]) if m100 is not None and "steps" in m100 else 100
    exact_file = args.study / (args.exact or f"i2-{steps}.npz")  # exact rates with the same number of Euler steps
    i2 = load_npz(exact_file)
    if i2 is None:
        missing.append(f"{exact_file}: no exact-rate comparison in parts 2b, 4 and 5")
    part_followups(args, sources, eligible, m100, i2, qdata, out, summary)
    part_exposure(args, sources, m100, out, summary, qdata)
    part_same_step(args, sources, qdata, m100, i2, out, summary)
    part_chains(args, sources, m100, i2, out, summary)

    summary["missing"] = missing
    summary["seconds"] = round(time.time() - started, 1)
    (out / "summary.json").write_text(json.dumps(r3(summary), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done in {time.time() - started:.0f} s -> {out}")
    for line in missing:
        print("  missing:", line)


if __name__ == "__main__":
    main()
