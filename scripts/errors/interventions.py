#!/usr/bin/env python
r"""Intervention experiments of the error study: does a wrong deletion make the model delete more around it?

    python scripts/errors/interventions.py --run runs/lap-01              # everything, resumable
    python scripts/errors/interventions.py --run runs/lap-01 --scale 0.05   # a small trial (own folder)

A *state* is what the flow would hold at time t0: the source with every inserted letter removed independently with
probability kappa(t0). A *site* is an original letter of that state; removing it is a forced wrong deletion. Sites
are stratified by the **break size** b = number of the two windows straddling the gap that the removal makes
forbidden. b = 0 is the placebo: the string stays grammatical, only the length and the positions shift. `clean`
sites have no leftover insertion within 3 letters (the primary analysis), `dirty` sites have one.

1. one forward pass with and without the removal: the change of the model's q on the 10 neighbours on each side,
   next to the same change of the exact posterior (the reaction of an ideal model);
2. sampling from t0 to the end with and without the removal, with common random numbers;
3. the mirror: an inserted letter that is forced to stay (`keep`), against the free run;
4. experiment 2 with exact rates instead of the model's;
5. the minimal system: the clean target at t0 = 0.9 with one letter removed (family `tgt09`).

Stages (each one caches under `<run>/errors/results/interventions/scratch/`, so the command can be repeated until
it is through): `sites` (CPU), `model` (GPU; `--minutes` stops it after a time budget with exit code 3, `--passes`
and `--families` choose what it samples), `exact` (CPU on purpose: the float64 forward-backward loop is several
times faster there than on a shared GPU), `analyse`. The default sample sizes are those of the first run of the
study (GPU time was scarce): 800 clean + 400 dirty sites per break size and family for experiment 1, 200 + 75 for
experiment 2, 100 per visibility stratum for experiment 3, 150 per break size for experiment 4.
Outputs: CSV tables, `summary.json`, `examples.md` and figures in `<run>/errors/results/interventions/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows import exact  # noqa: E402
from editflows.errors import Sources, break_size, minimal_repair, model_euler, violated_windows  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

N_STEPS = 100
REACH = 10  # neighbours on each side that count as "around the site"
MARGIN = REACH + 2  # letters a site needs on each side (the repair windows look two letters further)
P_TABLE = transition_table(np.random.default_rng(0), 20, 5, 1.0)
ELIGIBLE = P_TABLE > 0
ALPHABET = np.array(list("ABCDEFGHIJKLMNOPQRST"))
COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]


@dataclass(frozen=True)
class Family:
    key: str
    t0: float
    kind: str  # "flow": target + leftover insertions at t0; "target": the clean target
    label: str

    @property
    def k0(self) -> int:
        return int(round(self.t0 * N_STEPS))


FAMILIES = [
    Family("f00", 0.0, "flow", "t0 = 0"),
    Family("f03", 0.3, "flow", "t0 = 0.3"),
    Family("f06", 0.6, "flow", "t0 = 0.6"),
    Family("f09", 0.9, "flow", "t0 = 0.9"),
    Family("tgt09", 0.9, "target", "clean target, t0 = 0.9"),
]
FAMILY = {f.key: f for f in FAMILIES}
VISIBILITY = [("vis_hi", 0.95, 1.01), ("vis_mid", 0.6, 0.95), ("vis_lo", -1.0, 0.6)]


# ---- setup ---------------------------------------------------------------------------------------------------------


class Study:
    """Paths, sources, schedule and the states of one run."""

    def __init__(self, args):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        self.args = args
        self.run = Path(args.run)
        self.study = self.run.parent / "error-study"
        base = self.run / "errors" / "results" / "interventions"
        self.out = base if args.scale == 1 else base / "scratch" / f"trial-{args.scale:g}"
        self.scratch = self.out / "scratch"
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.sources = Sources.load(self.study / "sources.npz")
        self.rho = self.sources.rho
        self.checkpoint = torch.load(self.run / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
        self.config = OmegaConf.create(self.checkpoint["config"])
        self.kappa = instantiate(self.config.task.path).kappa
        self.bos = self.config.data.vocab_size - 1
        self.kappa_at = {f.key: float(self.kappa(torch.tensor(f.t0, dtype=torch.float64))) for f in FAMILIES}
        self._alive: dict = {}
        self._model = None
        self.guard()

    def guard(self):
        """Cached results belong to one checkpoint and one set of sample sizes."""
        a = self.args
        meta = {"epoch": int(self.checkpoint["epoch"]), "global_step": int(self.checkpoint["global_step"]),
                "seed": a.seed, "first": a.first, "scale": a.scale, "max_tokens": a.max_tokens,
                "sizes": [a.n_one, a.n_one_dirty, a.n_run, a.n_run_dirty, a.run_sources, a.n_mirror, a.n_exact, a.exact_batch],
                "exact_families": a.exact_families}
        path = self.scratch / "meta.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old != meta:
                sys.exit(f"{path} was written for {old}, this call is {meta}: delete the scratch folder to start over")
        else:
            path.write_text(json.dumps(meta))
        self.meta = meta

    def n(self, full: int) -> int:
        return max(1, int(round(full * self.args.scale)))

    def alive(self, fam: Family, i: int) -> np.ndarray:
        """The letters of source i that are present in the state of this family (deterministic in the seed)."""
        key = (fam.key, i)
        if key not in self._alive:
            ins = self.sources.mask(i)
            if fam.kind == "target":
                out = ~ins
            elif self.kappa_at[fam.key] == 0:
                out = np.ones(len(ins), dtype=bool)
            else:
                rng = np.random.default_rng([self.args.seed, FAMILIES.index(fam), i])
                out = ~ins | (rng.random(len(ins)) >= self.kappa_at[fam.key])
            self._alive[key] = out
        return self._alive[key]

    def prior(self, fam: Family, i) -> np.ndarray:
        """Share of leftover insertions an ideal model expects at t0, given the true noise level of the source."""
        return exact.leftover_share(torch.as_tensor(self.rho[i], dtype=torch.float64), self.kappa_at[fam.key]).numpy()

    def model(self, device):
        """The trained model (EMA weights); loaded when the first forward pass needs it."""
        if self._model is None:
            from evaluate import load_model

            self._model = load_model(self.checkpoint, self.config, ema=True).to(device)
        return self._model


# ---- sites ---------------------------------------------------------------------------------------------------------


def site_pool(study: Study, fam: Family, ids) -> pd.DataFrame:
    """Every original letter with `MARGIN` state letters on each side: its break size and whether leftover
    insertions sit within +-3 / +-MARGIN letters of it."""
    rows = []
    for i in ids:
        idx = np.flatnonzero(study.alive(fam, i))
        if len(idx) < 2 * MARGIN + 1:
            continue
        ins = study.sources.mask(i)[idx]
        state = study.sources.seq(i)[idx]
        s = np.arange(MARGIN, len(idx) - MARGIN)
        s = s[~ins[s]]
        c = np.concatenate([[0], np.cumsum(ins)])
        rows.append(pd.DataFrame({"source": i, "pos": idx[s], "spos": s, "b": break_size(state, s, ELIGIBLE),
                                  "clean3": c[s + 4] - c[s - 3] == 0, "clean12": c[s + MARGIN + 1] - c[s - MARGIN] == 0}))
    return pd.concat(rows, ignore_index=True)


def build_sites(study: Study) -> tuple[pd.DataFrame, dict]:
    """Sample the sites of all experiments. `exp`: `one` (one forward pass), `run` (sampling), `mirror`."""
    path, pool_path = study.scratch / "sites.csv", study.scratch / "pool.json"
    if path.exists():
        return pd.read_csv(path), json.loads(pool_path.read_text())
    a = study.args
    rng = np.random.default_rng(a.seed)
    ids = np.arange(a.first)
    subset = np.sort(rng.choice(ids, size=min(len(ids), max(20, study.n(a.run_sources))), replace=False))
    parts, counts = [], {}
    for fam in FAMILIES:
        pool = site_pool(study, fam, ids)
        counts[fam.key] = {}
        for b in range(3):
            for clean in (True, False):
                stratum = "clean" if clean else "dirty"
                sel = pool[(pool.b == b) & (pool.clean3 == clean)]
                counts[fam.key][f"b{b}_{stratum}"] = len(sel)
                take = sel.iloc[rng.permutation(len(sel))[: study.n(a.n_one if clean else a.n_one_dirty)]]
                parts.append(take.assign(exp="one", fam=fam.key, stratum=stratum, order=np.arange(len(take))))
                sel = sel[sel.source.isin(subset)]
                take = sel.iloc[rng.permutation(len(sel))[: study.n(a.n_run if clean else a.n_run_dirty)]]
                take = take.assign(exp="run", fam=fam.key, stratum=stratum, order=np.arange(len(take)))
                parts.append(take.assign(exact=clean & (take["order"] < study.n(a.n_exact)) & (fam.key in a.exact_families)))
    # the mirror: inserted letters of the untouched source, by how visible they are to an ideal model
    q_true = np.load(study.study / "posterior.npz")["q_true"]
    rows = []
    for i in subset:
        span, ins = study.sources.span(i), study.sources.mask(i)
        pos = np.flatnonzero(ins)
        pos = pos[(pos >= MARGIN) & (pos < len(ins) - MARGIN)]
        rows.append(pd.DataFrame({"source": i, "pos": pos, "spos": pos, "q_true": q_true[span][pos]}))
    pool = pd.concat(rows, ignore_index=True)
    for stratum, lo, hi in VISIBILITY:
        sel = pool[(pool.q_true >= lo) & (pool.q_true < hi)]
        counts["f00"][stratum] = len(sel)
        take = sel.iloc[rng.permutation(len(sel))[: study.n(a.n_mirror)]]
        parts.append(take.assign(exp="mirror", fam="f00", stratum=stratum, order=np.arange(len(take)), b=-1))
    sites = pd.concat(parts, ignore_index=True)
    sites["exact"] = sites["exact"].fillna(False).astype(bool)
    for col in ("clean3", "clean12"):
        sites[col] = sites[col].fillna(False).astype(bool)
    sites.insert(0, "site", np.arange(len(sites)))
    counts["run_sources"] = len(subset)
    sites.to_csv(path, index=False)
    pool_path.write_text(json.dumps(counts, indent=1))
    return sites, counts


# ---- stage: one forward pass (experiment 1) --------------------------------------------------------------------------


def onepass_states(study: Study, fam: Family, sites: pd.DataFrame):
    """The sequences of experiment 1: the state of every source that has a site, then one state per site with the
    site removed. Returns the sequences, the source of each, and for each site the row of its control state."""
    srcs = np.unique(sites.source)
    row = {int(i): j for j, i in enumerate(srcs)}
    seqs = [study.sources.seq(i)[study.alive(fam, i)] for i in srcs]
    owner = list(srcs)
    for i, s in zip(sites.source, sites.spos):
        seqs.append(np.delete(seqs[row[int(i)]], s))
        owner.append(i)
    return seqs, np.array(owner), np.array([row[int(i)] for i in sites.source])


def neighbour_q(q: list, control: np.ndarray, spos: np.ndarray):
    """q of the site and its `REACH` neighbours on each side, without (`qc`) and with (`qi`) the removal."""
    n_ctl = len(q) - len(spos)
    qc = np.stack([q[c][s - REACH : s + REACH + 1] for c, s in zip(control, spos)])
    qi = np.full_like(qc, np.nan)
    for j, s in enumerate(spos):
        v = q[n_ctl + j]
        qi[j, :REACH], qi[j, REACH + 1 :] = v[s - REACH : s], v[s : s + REACH]
    return qc, qi


def token_chunks(lengths, max_tokens: int):
    chunk, total = [], 0
    for j in np.argsort(lengths, kind="stable"):
        if chunk and total + lengths[j] > max_tokens:
            yield chunk
            chunk, total = [], 0
        chunk.append(int(j))
        total += int(lengths[j])
    if chunk:
        yield chunk


def stage_onepass_model(study: Study, sites: pd.DataFrame, device, deadline) -> bool:
    for fam in FAMILIES:
        path = study.scratch / f"onepass-model-{fam.key}.npz"
        s = sites[(sites.exp == "one") & (sites.fam == fam.key)]
        if path.exists() or not len(s):
            continue
        if time.time() > deadline:
            return False
        start = time.time()
        seqs, _, control = onepass_states(study, fam, s)
        q: list = [None] * len(seqs)
        for chunk in token_chunks([len(x) for x in seqs], study.args.max_tokens):
            res = model_euler(study.model(device), [seqs[j] for j in chunk], study.kappa, study.bos, device=device,
                              start_step=fam.k0, end_step=fam.k0 + 1)["q0"]
            at = 0
            for j in chunk:
                q[j] = res[at : at + len(seqs[j])]
                at += len(seqs[j])
        qc, qi = neighbour_q(q, control, s.spos.to_numpy())
        np.savez_compressed(path, site=s.site.to_numpy(), qc=qc, qi=qi)
        print(f"one pass, model, {fam.key}: {len(s)} sites, {len(seqs)} states in {time.time() - start:.0f} s", flush=True)
    return True


def stage_onepass_exact(study: Study, sites: pd.DataFrame, device):
    P = torch.tensor(P_TABLE, dtype=torch.float64, device=device)
    for fam in FAMILIES:
        path = study.scratch / f"onepass-exact-{fam.key}.npz"
        s = sites[(sites.exp == "one") & (sites.fam == fam.key)]
        if path.exists() or not len(s):
            continue
        start = time.time()
        seqs, owner, control = onepass_states(study, fam, s)
        prior = study.prior(fam, owner)
        q: list = [None] * len(seqs)
        order = np.argsort([len(x) for x in seqs], kind="stable")
        for at in range(0, len(order), 512):
            chunk = order[at : at + 512]
            S, valid = exact.pad([seqs[j] for j in chunk], device)
            post = exact.posterior(S, valid, torch.tensor(prior[chunk], dtype=torch.float64, device=device), P,
                                   budget=study.args.exact_budget).cpu().numpy()
            for r, j in enumerate(chunk):
                q[j] = post[r, : len(seqs[j])]
        qc, qi = neighbour_q(q, control, s.spos.to_numpy())
        np.savez_compressed(path, site=s.site.to_numpy(), qc=qc, qi=qi)
        print(f"one pass, exact, {fam.key}: {len(s)} sites, {len(seqs)} states in {time.time() - start:.0f} s", flush=True)


# ---- stage: sampling with common random numbers (experiments 2, 3, 5 and 4) ------------------------------------------
#
# `model_euler` keys its random numbers by the position of a letter in the concatenation of the sequences of the
# call. A control run and an intervened run therefore share their draws only if every letter sits at the same
# position in both calls. Every family has one fixed layout (its sources in a fixed order); a call may contain any
# subset of the sources: the letters of the others are passed as dead filler of the same length, so the positions do
# not move. The attention pads every sequence of a call to the longest one, so filler is merged into few sequences
# and calls are filled up to `max_tokens` letters.

PASSES = ["clean", "mirror", "dirty"]


def run_sites(sites: pd.DataFrame, fam: Family, which: str) -> pd.DataFrame:
    s = sites[(sites.fam == fam.key) & sites.exp.isin(["run", "mirror"])]
    if which == "exact":
        return s[s.exact]
    if which == "all":
        return s
    return s[s.stratum.str.startswith("vis")] if which == "mirror" else s[(s.exp == "run") & (s.stratum == which)]


def layout(study: Study, sites: pd.DataFrame, fam: Family):
    """The sources of the sampled runs of a family, ordered by the length of their state, and the offset of each
    in the concatenation of their letters."""
    srcs = np.unique(run_sites(sites, fam, "all").source)
    srcs = srcs[np.argsort([int(study.alive(fam, i).sum()) for i in srcs], kind="stable")]
    return srcs, np.concatenate([[0], np.cumsum(np.diff(study.sources.offsets)[srcs])])


def pack_calls(study: Study, sites: pd.DataFrame, fam: Family, which: str) -> list[list[tuple[int, int]]]:
    """The sampler calls of one pass: lists of (source, site) with distinct sources and at most `max_tokens` letters
    alive. Site -1 is the control run of the source; the control runs belong to the first pass."""
    srcs, _ = layout(study, sites, fam)
    rank = {int(i): r for r, i in enumerate(srcs)}
    s = run_sites(sites, fam, which)
    first = which == PASSES[0]
    runs = [(0, rank[int(i)], int(i), -1) for i in srcs] if first else []
    runs += [(int(k) + first, rank[int(i)], int(i), int(sid)) for k, i, sid in zip(s.groupby("source").cumcount(), s.source, s.site)]
    calls: list = []  # runs, their sources, letters alive
    for _, _, i, sid in sorted(runs):
        n = int(study.alive(fam, i).sum())
        for call in calls:
            if i not in call[1] and call[2] + n <= study.args.max_tokens:
                break
        else:
            call = [[], set(), 0]
            calls.append(call)
        call[0].append((i, sid))
        call[1].add(i)
        call[2] += n
    return [call[0] for call in calls]


def model_call(study: Study, model, device, fam: Family, srcs, off, runs, site_info: dict) -> dict:
    """Sample the runs of one call; returns source -> (deletion steps, q at the start) over the letters of the source."""
    wanted = dict(runs)
    seqs, alive, keep, gap = [], [], [], 0

    def filler(n):
        seqs.append(np.zeros(n, dtype=np.int64))
        alive.append(np.zeros(n, dtype=bool))
        keep.append(np.zeros(n, dtype=bool))

    for r, i in enumerate(srcs):
        if int(i) not in wanted:
            gap += int(off[r + 1] - off[r])
            continue
        if gap:
            filler(gap)
            gap = 0
        a, k = study.alive(fam, i).copy(), np.zeros(int(off[r + 1] - off[r]), dtype=bool)
        if wanted[int(i)] >= 0:
            kind, pos = site_info[wanted[int(i)]]
            if kind == "mirror":
                k[pos] = True
            else:
                a[pos] = False
        seqs.append(study.sources.seq(i))
        alive.append(a)
        keep.append(k)
    if gap:
        filler(gap)
    if len(seqs) < len(srcs):  # merged filler: keep the number of random draws per step the same in every call
        filler(len(srcs) - len(seqs) - 1)
    assert sum(len(x) for x in seqs) + len(seqs) == off[-1] + len(srcs)
    res = model_euler(model, seqs, study.kappa, study.bos, device=device, n_steps=N_STEPS, seed=study.args.seed,
                      start_step=fam.k0, alive=np.concatenate(alive), keep=np.concatenate(keep))
    rank = {int(i): r for r, i in enumerate(srcs)}
    return {i: (res["del_step"][off[rank[i]] : off[rank[i] + 1]], res["q0"][off[rank[i]] : off[rank[i] + 1]]) for i in wanted}


def exact_batches(study: Study, sites: pd.DataFrame, fam: Family) -> list[list[int]]:
    """Sources with the same number of sites share a batch, so that no call carries empty rows."""
    counts = run_sites(sites, fam, "exact").source.value_counts()
    out = []
    for k in sorted(counts.unique()):
        srcs = np.sort(counts.index[counts == k].to_numpy())
        srcs = srcs[np.argsort([int(study.alive(fam, i).sum()) for i in srcs], kind="stable")]
        out += [[int(i) for i in srcs[at : at + study.args.exact_batch]] for at in range(0, len(srcs), study.args.exact_batch)]
    return out


def call_table(s: pd.DataFrame, batch: list[int]) -> np.ndarray:
    """`table[c, r]` = the site that source r of the batch gets in call c (-1: none), in sampling order."""
    per = [s.site[s.source == i].to_numpy() for i in batch]
    table = np.full((max((len(x) for x in per), default=0), len(batch)), -1, dtype=np.int64)
    for r, x in enumerate(per):
        table[: len(x), r] = x
    return table


def batch_layout(study: Study, fam: Family, batch: list[int]):
    seqs = [study.sources.seq(i) for i in batch]
    off = np.concatenate([[0], np.cumsum([len(x) for x in seqs])])
    return seqs, off, np.concatenate([study.alive(fam, i) for i in batch])


def interventions(table_row, off, base, site_info: dict):
    """`alive` and `keep` of one call: every source with a site gets its state, minus the site or with it locked.
    `site_info[site] = (exp, pos)`."""
    alive, keep = np.zeros_like(base), np.zeros_like(base)
    for r, sid in enumerate(table_row):
        if sid < 0:
            continue
        alive[off[r] : off[r + 1]] = base[off[r] : off[r + 1]]
        kind, pos = site_info[int(sid)]
        if kind == "mirror":
            keep[off[r] + pos] = True
        else:
            alive[off[r] + pos] = False
    return alive, keep


def site_lookup(sites: pd.DataFrame) -> dict:
    return {int(s): (e, int(q)) for s, e, q in zip(sites.site, sites.exp, sites.pos)}


def stage_runs_model(study: Study, sites: pd.DataFrame, device, deadline, passes, families=None) -> bool:
    site_info = site_lookup(sites)
    for which in passes:
        for fam in [FAMILY[k] for k in families] if families else reversed(FAMILIES):  # by default the short runs first
            srcs, off = layout(study, sites, fam)
            for c, runs in enumerate(pack_calls(study, sites, fam, which)):
                path = study.scratch / f"calls-{fam.key}-{which}-{c}.npz"
                if path.exists():
                    continue
                if time.time() > deadline:
                    return False
                start = time.time()
                res = model_call(study, study.model(device), device, fam, srcs, off, runs, site_info)
                null_diff = -1
                if which == PASSES[0] and c == 0:  # no intervention, another composition of the call: the same result?
                    again = model_call(study, study.model(device), device, fam, srcs, off, runs[::2], site_info)
                    null_diff = int(sum((again[i][0] != res[i][0]).sum() for i in again))
                np.savez_compressed(path, source=np.array([i for i, _ in runs]), site=np.array([sid for _, sid in runs]),
                                    length=np.array([len(res[i][0]) for i, _ in runs]), null_diff=null_diff,
                                    del_step=np.concatenate([res[i][0] for i, _ in runs]),
                                    q0=np.concatenate([res[i][1] for i, _ in runs]).astype(np.float16))
                print(f"model runs {fam.key} {which} call {c}: {len(runs)} runs, {sum(int(study.alive(fam, i).sum()) for i, _ in runs)} "
                      f"letters in {time.time() - start:.0f} s" + (f", differences without intervention: {null_diff}" if null_diff >= 0 else ""),
                      flush=True)
    return True


def stage_runs_exact(study: Study, sites: pd.DataFrame, device):
    P = torch.tensor(P_TABLE, dtype=torch.float64, device=device)
    by_site = site_lookup(sites)
    for fam in reversed(FAMILIES):
        s = run_sites(sites, fam, "exact")
        if not len(s):
            continue
        for b, batch in enumerate(exact_batches(study, sites, fam)):
            path = study.scratch / f"exact-{fam.key}-{b}.npz"
            if path.exists():
                continue
            start = time.time()
            table = call_table(s, batch)
            seqs, off, base = batch_layout(study, fam, batch)
            letters, valid = exact.pad(seqs, device)
            rho0 = torch.tensor(study.rho[batch], dtype=torch.float64, device=device)

            def call(alive):
                grid = torch.zeros_like(valid)
                grid[valid] = torch.as_tensor(alive, device=device)
                steps = exact.exact_euler(letters, valid, rho0, P, study.kappa, n_steps=N_STEPS, seed=study.args.seed,
                                          start_step=fam.k0, alive=grid, budget=study.args.exact_budget)
                return steps[valid].cpu().numpy()

            rows = [call(base)]
            for c in range(len(table)):
                rows.append(call(interventions(table[c], off, base, by_site)[0]))
            np.savez_compressed(path, sources=np.array(batch), table=table, del_step=np.stack(rows))
            print(f"exact runs {fam.key} batch {b}: {len(batch)} sources, {len(rows)} calls in {time.time() - start:.0f} s",
                  flush=True)


# ---- analysis: helpers -----------------------------------------------------------------------------------------------


class Boot:
    """Bootstrap over sources. One set of resampling weights serves every statistic, so that differences between
    strata and mixtures of strata are resampled coherently."""

    def __init__(self, n_boot: int, n_sources: int, seed: int):
        rng = np.random.default_rng(seed + 1)
        self.w = rng.multinomial(n_sources, np.full(n_sources, 1.0 / n_sources), size=n_boot).astype(np.float64)

    def draws(self, values, source, weight=None):
        """The (weighted) mean of `values` over sites (first axis) and its replicates."""
        values = np.asarray(values, dtype=np.float64)
        shape = values.shape[1:]
        if len(values) == 0:
            return np.full(shape, np.nan), np.full((len(self.w),) + shape, np.nan)
        flat = values.reshape(len(values), -1)
        weight = np.ones(len(flat)) if weight is None else np.asarray(weight, dtype=np.float64)
        u, inv = np.unique(np.asarray(source), return_inverse=True)
        sums = np.zeros((len(u), flat.shape[1]))
        np.add.at(sums, inv, flat * weight[:, None])
        w = self.w[:, u]
        with np.errstate(invalid="ignore", divide="ignore"):
            rep = (w @ sums) / (w @ np.bincount(inv, weights=weight))[:, None]
        point = (flat * weight[:, None]).sum(0) / weight.sum()
        return point.reshape(shape), rep.reshape((len(rep),) + shape)

    def quantile(self, values, source, q: float = 0.5, weight=None):
        """The (weighted) q-quantile of `values` and its replicates."""
        values = np.asarray(values, dtype=np.float64)
        if len(values) == 0:
            return np.nan, np.full(len(self.w), np.nan)
        order = np.argsort(values)
        weight = np.ones(len(values)) if weight is None else np.asarray(weight, dtype=np.float64)
        cum = np.cumsum(self.w[:, np.asarray(source)[order]] * weight[order], axis=1)
        pick = (cum >= q * cum[:, -1:]).argmax(1)
        base = np.cumsum(weight[order])
        return float(values[order][(base >= q * base[-1]).argmax()]), np.where(cum[:, -1] > 0, values[order][pick], np.nan)


def est(point, rep, name: str = "") -> dict:
    """`name`, `name_lo`, `name_hi`: the estimate and its 95 % bootstrap interval."""
    with np.errstate(all="ignore"):
        lo, hi = (np.nanpercentile(rep, [2.5, 97.5]) if np.isfinite(rep).any() else (np.nan, np.nan))
    return {name or "mean": float(point), f"{name}_lo" if name else "lo": float(lo), f"{name}_hi" if name else "hi": float(hi)}


def mixture(parts):
    """Weighted combination of (weight, point, replicates) that skips empty parts."""
    parts = [(w, p, r) for w, p, r in parts if w > 0 and np.all(np.isfinite(p))]
    total = sum(w for w, _, _ in parts)
    if not parts:
        return np.nan, np.full(1, np.nan)
    return sum(w * p for w, p, _ in parts) / total, sum(w * r for w, _, r in parts) / total


def show(table: pd.DataFrame, keys: list[str], metrics: list[str], digits: int = 2) -> str:
    """A table for the console: every metric as `estimate [low, high]`."""
    out = table[keys].copy()
    for m in metrics:
        if m in table:
            out[m] = [f"{a:.{digits}f} [{b:.{digits}f}, {c:.{digits}f}]" for a, b, c in zip(table[m], table[f"{m}_lo"], table[f"{m}_hi"])]
    return out.to_string(index=False)


def minimal_hole(state: np.ndarray, s: int) -> int:
    """The smallest contiguous hole (letters, the site included) around the site whose two junction windows are
    allowed; `REACH + 2` if no hole of up to `REACH + 1` letters works."""
    for size in range(1, REACH + 2):
        for a in range(s - size + 1, s + 1):
            z = a + size
            if ELIGIBLE[state[a - 2], state[a - 1], state[z]] and ELIGIBLE[state[a - 1], state[z], state[z + 1]]:
                return size
    return REACH + 2


def annotate(study: Study, sites: pd.DataFrame) -> pd.DataFrame:
    """What the grammar says about every removal: the minimal repair of the state with (`mr_int`) and without
    (`mr_ctl`) the removal, the smallest contiguous hole (`lmin`), and how probable the two windows across the gap
    are (`gap_min`: the smaller of the two transition probabilities, 0 if one is forbidden)."""
    path = study.scratch / "annotations.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for fam in FAMILIES:
        s = sites[(sites.fam == fam.key) & (sites.exp != "mirror")]
        for sid, i, spos, clean12 in zip(s.site, s.source, s.spos, s.clean12):
            state = study.sources.seq(i)[study.alive(fam, i)]
            l2, l1, x, r1, r2 = state[spos - 2 : spos + 3]
            rows.append({
                "site": sid,
                "mr_int": minimal_repair(np.delete(state, spos), spos - REACH, spos + REACH, ELIGIBLE),
                "mr_ctl": 0 if clean12 else minimal_repair(state, spos - REACH, spos + REACH + 1, ELIGIBLE),
                "lmin": minimal_hole(state, spos),
                "gap_min": min(P_TABLE[l2, l1, r1], P_TABLE[l1, r1, r2]),
                "gap_p": P_TABLE[l2, l1, r1] * P_TABLE[l1, r1, r2],
                "old_p": P_TABLE[l2, l1, x] * P_TABLE[l1, x, r1] * P_TABLE[x, r1, r2],
            })
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def natural_weights(pool: dict, fam: Family) -> dict:
    """Share of the clean / dirty sites of every break size among all sites of the family."""
    counts = {(b, st): pool[fam.key].get(f"b{b}_{st}", 0) for b in range(3) for st in ("clean", "dirty")}
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


# ---- analysis: experiment 1 (one forward pass) -----------------------------------------------------------------------

OFFSETS = np.arange(-REACH, REACH + 1)


def analyse_onepass(study: Study, sites: pd.DataFrame, pool: dict, boot: Boot, notes: list):
    """Per-site table of the change of q around the removal, for the model and for the exact posterior."""
    frames, delta = [], {}
    for fam in FAMILIES:
        s = sites[(sites.exp == "one") & (sites.fam == fam.key)].reset_index(drop=True)
        s["row"] = np.arange(len(s))
        for engine in ("model", "exact"):
            path = study.scratch / f"onepass-{engine}-{fam.key}.npz"
            if not path.exists():
                notes.append(f"experiment 1: {path.name} is missing")
                continue
            z = np.load(path)
            assert np.array_equal(z["site"], s.site.to_numpy())
            d = (z["qi"] - z["qc"]).astype(np.float64)
            delta[engine, fam.key] = d
            s[f"S_{engine}"] = np.nansum(d, axis=1)
            s[f"near_{engine}"] = np.nansum(d[:, REACH - 2 : REACH + 3], axis=1)
            s[f"peak_{engine}"] = np.nanmax(np.abs(d), axis=1)
            s[f"qsite_{engine}"] = z["qc"][:, REACH]
        frames.append(s)
    one = pd.concat(frames, ignore_index=True)
    weights = {f.key: natural_weights(pool, f) for f in FAMILIES}
    one["weight"] = [weights[f][b, st] for f, b, st in zip(one.fam, one.b, one.stratum)]
    one["weight"] /= one.groupby(["fam", "b", "stratum"]).site.transform("size")

    rows, profile = [], []
    for fam in FAMILIES:
        for stratum in ("clean", "dirty", "all"):
            for b in range(3):
                row = {"family": fam.key, "t0": fam.t0, "stratum": stratum, "break_size": b}
                parts = {}
                for st in ("clean", "dirty"):
                    parts[st] = one[(one.fam == fam.key) & (one.b == b) & (one.stratum == st)]
                use = [st for st in parts if stratum in (st, "all") and len(parts[st])]
                if not use:
                    continue
                row["n"] = int(sum(len(parts[st]) for st in use))
                row["n_sources"] = int(pd.concat([parts[st] for st in use]).source.nunique())
                for engine in ("model", "exact"):
                    if (engine, fam.key) not in delta:
                        continue
                    for col, name in ((f"S_{engine}", f"S_{engine}"), (f"near_{engine}", f"S_within2_{engine}"),
                                      (f"qsite_{engine}", f"q_of_site_{engine}")):
                        row.update(est(*mixture([(weights[fam.key][b, st], *boot.draws(parts[st][col], parts[st].source)) for st in use]), name))
                    d = delta[engine, fam.key]
                    point, rep = mixture([(weights[fam.key][b, st], *boot.draws(np.nan_to_num(d[parts[st].row.to_numpy()]), parts[st].source)) for st in use])
                    lo, hi = np.nanpercentile(rep, [2.5, 97.5], axis=0)
                    profile += [{"family": fam.key, "t0": fam.t0, "stratum": stratum, "break_size": b, "engine": engine,
                                 "offset": int(o), "dq": point[j], "dq_lo": lo[j], "dq_hi": hi[j], "n": row["n"]}
                                for j, o in enumerate(OFFSETS) if o != 0]
                rows.append(row)
    return one, pd.DataFrame(rows), pd.DataFrame(profile)


def onepass_extras(one: pd.DataFrame, boot: Boot):
    """The placebo in detail, and the suspicion against the size of the minimal repair."""
    placebo, by_repair = [], []
    for fam in FAMILIES:
        s = one[(one.fam == fam.key) & (one.stratum == "clean")]
        z = s[s.b == 0]
        for label, part in (("all", z), ("both new windows have p >= 0.05", z[z.gap_min >= 0.05]),
                            ("a new window has p < 0.05", z[z.gap_min < 0.05])):
            row = {"family": fam.key, "t0": fam.t0, "placebo_sites": label, "n": len(part), "n_sources": part.source.nunique()}
            for engine in ("model", "exact"):
                if f"S_{engine}" in part and len(part):
                    row.update(est(*boot.draws(part[f"S_{engine}"], part.source), f"S_{engine}"))
                    row[f"largest_change_median_{engine}"] = float(part[f"peak_{engine}"].median())
                    row[f"largest_change_p95_{engine}"] = float(part[f"peak_{engine}"].quantile(0.95))
                    row[f"largest_change_max_{engine}"] = float(part[f"peak_{engine}"].max())
                    row[f"share_change_above_0.05_{engine}"] = float((part[f"peak_{engine}"] > 0.05).mean())
            placebo.append(row)
        strict = one[(one.fam == fam.key) & one.clean12 & (one.mr_int >= 0)]
        for k in range(6):
            part = strict[(strict.mr_int == k) if k < 5 else (strict.mr_int >= 5)]
            if not len(part):
                continue
            row = {"family": fam.key, "t0": fam.t0, "minimal_repair": f"{k}" if k < 5 else "5+", "n": len(part),
                   "n_sources": part.source.nunique()}
            for engine in ("model", "exact"):
                if f"S_{engine}" in part:
                    row.update(est(*boot.draws(part[f"S_{engine}"], part.source, part.weight), f"S_{engine}"))
            by_repair.append(row)
    return pd.DataFrame(placebo), pd.DataFrame(by_repair)


# ---- analysis: sampled runs (experiments 2, 4, 5) --------------------------------------------------------------------


def load_runs(study: Study, sites: pd.DataFrame, fam: Family, engine: str, notes: list) -> dict:
    """site -> (deletion steps of the control run, of the intervened run, q at the start of both; model only)."""
    out, missing = {}, 0
    if engine == "model":
        control, raw = {}, {}
        for which in PASSES:
            for c in range(len(pack_calls(study, sites, fam, which))):
                path = study.scratch / f"calls-{fam.key}-{which}-{c}.npz"
                if not path.exists():
                    missing += 1
                    continue
                z = np.load(path)
                bounds = np.concatenate([[0], np.cumsum(z["length"])])
                for j, (i, sid) in enumerate(zip(z["source"], z["site"])):
                    item = (int(i), z["del_step"][bounds[j] : bounds[j + 1]], z["q0"][bounds[j] : bounds[j + 1]].astype(np.float32))
                    if sid < 0:
                        control[int(i)] = item
                    else:
                        raw[int(sid)] = item
        out = {sid: (control[i][1], d, control[i][2], q) for sid, (i, d, q) in raw.items() if i in control}
    else:
        for b, batch in enumerate(exact_batches(study, sites, fam)):
            off = np.concatenate([[0], np.cumsum([len(study.sources.seq(i)) for i in batch])])
            path = study.scratch / f"exact-{fam.key}-{b}.npz"
            if not path.exists():
                missing += 1
                continue
            z = np.load(path)
            for c, r in zip(*np.nonzero(z["table"] >= 0)):
                seg = slice(off[r], off[r + 1])
                out[int(z["table"][c, r])] = (z["del_step"][0][seg], z["del_step"][1 + c][seg], None, None)
    if missing:
        notes.append(f"sampled runs, {engine}, {fam.key}: {missing} result file(s) missing")
    return out


def grammatical(seq: np.ndarray, present: np.ndarray, lo: int, hi: int) -> bool:
    """No forbidden window of the output touches the source positions lo .. hi."""
    kept = np.flatnonzero(present)
    bad = violated_windows(seq[kept], ELIGIBLE)
    first = kept[np.maximum(np.arange(len(kept)) - 2, 0)][bad]
    return not bool(((first <= hi) & (kept[bad] >= lo)).any())


def removal_outcome(seq, ins, alive, pos, d_ctl, d_int, k0, q_ctl=None, q_int=None) -> dict:
    """What a forced wrong deletion at `pos` changed within `REACH` state letters on each side (the forced letter
    itself does not count)."""
    idx = np.flatnonzero(alive)
    s = int(np.searchsorted(idx, pos))
    nb = np.concatenate([idx[s - REACH : s], idx[s + 1 : s + REACH + 1]])
    offsets = np.concatenate([np.arange(-REACH, 0), np.arange(1, REACH + 1)])
    orig = ~ins[nb]
    gone_c, gone_i = d_ctl[nb] >= 0, d_int[nb] >= 0
    new, rescued = orig & gone_i & ~gone_c, orig & gone_c & ~gone_i
    present_c = alive & (d_ctl < 0)
    present_i = alive & (d_int < 0)
    present_i[pos] = False
    left = right = pos  # the hole: letters around the site that are absent from the output, without a gap
    while left > 0 and not present_i[left - 1]:
        left -= 1
    while right < len(seq) - 1 and not present_i[right + 1]:
        right += 1
    in_hole = (nb >= left) & (nb <= right)
    far = alive & ~ins
    far[nb[0] : nb[-1] + 1] = False
    out = {
        "n_new": int(new.sum()), "n_rescued": int(rescued.sum()), "extra_wrong": int(new.sum()) - int(rescued.sum()),
        "wrong_ctl": int((orig & gone_c).sum()),
        "extra_missed": int((~orig & ~gone_i).sum()) - int((~orig & ~gone_c).sum()),
        "extra_deletions": int(gone_i.sum()) - int(gone_c.sum()),
        "extra_wrong_far": int(((d_int >= 0) & far).sum()) - int(((d_ctl >= 0) & far).sum()),
        "gram_ctl": grammatical(seq, present_c, nb[0], nb[-1]), "gram_int": grammatical(seq, present_i, nb[0], nb[-1]),
        "hole": int((~ins[left : right + 1]).sum()), "new_in_hole": int((new & in_hole).sum()),
        "delay": float(d_int[nb][new].min() - k0) if new.any() else np.nan,
        "new_offsets": ";".join(str(o) for o in offsets[new]), "new_steps": ";".join(str(k) for k in d_int[nb][new]),
    }
    if q_ctl is not None:
        out["S_start"] = float(np.nansum(q_int[nb] - q_ctl[nb]))
    else:
        out["S_start"] = np.nan
    return out


def analyse_runs(study: Study, sites: pd.DataFrame, notes: list) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        for engine in ("model", "exact"):
            runs = load_runs(study, sites, fam, engine, notes)
            s = sites[(sites.fam == fam.key) & (sites.exp == "run") & sites.site.isin(list(runs))]
            for sid, i, pos, spos, b, stratum, clean12, exact_site in zip(s.site, s.source, s.pos, s.spos, s.b, s.stratum, s.clean12, s.exact):
                d_ctl, d_int, q_ctl, q_int = runs[sid]
                row = {"site": sid, "engine": engine, "fam": fam.key, "t0": fam.t0, "source": i, "pos": pos, "spos": spos,
                       "b": b, "stratum": stratum, "clean12": clean12, "paired": exact_site}
                row.update(removal_outcome(study.sources.seq(i), study.sources.mask(i), study.alive(fam, i), pos, d_ctl, d_int,
                                           fam.k0, q_ctl, q_int))
                rows.append(row)
    return pd.DataFrame(rows)


RUN_METRICS = ["extra_wrong", "any_new", "n_new", "n_rescued", "extra_missed", "extra_wrong_far", "gram_int", "gram_ctl", "hole", "S_start"]


def summarise_runs(runs: pd.DataFrame, pool: dict, boot: Boot):
    """Per engine, family, stratum and break size: the outcomes with bootstrap intervals; `all` mixes clean and
    dirty sites with their natural shares; break size `any` mixes the break sizes of the clean sites likewise."""
    rows, dist = [], []
    for engine in ("model", "exact"):
        for fam in FAMILIES:
            weights = natural_weights(pool, fam)
            r = runs[(runs.engine == engine) & (runs.fam == fam.key)]
            if not len(r):
                continue
            for stratum in ("clean", "dirty", "all"):
                cells = {(b, st): r[(r.b == b) & (r.stratum == st)] for b in range(3) for st in ("clean", "dirty")}
                for b in (0, 1, 2, "1+2", "any"):
                    bs = {"1+2": (1, 2), "any": (0, 1, 2)}.get(b, (b,))
                    use = [(bb, st) for bb in bs for st in ("clean", "dirty") if stratum in (st, "all") and len(cells[bb, st])]
                    if not use or len({bb for bb, _ in use}) < len(bs):
                        continue
                    if stratum == "all" and any(weights[bb, "dirty"] > 0 and not len(cells[bb, "dirty"]) for bb in bs):
                        continue  # the dirty sites were not sampled (yet)
                    both = pd.concat([cells[k] for k in use])
                    row = {"engine": engine, "family": fam.key, "t0": fam.t0, "stratum": stratum, "break_size": b,
                           "n": len(both), "n_sources": both.source.nunique()}
                    for m in RUN_METRICS:
                        row.update(est(*mixture([(weights[k], *boot.draws(cells[k][m], cells[k].source)) for k in use]), m))
                    hit = {k: cells[k][cells[k].n_new > 0] for k in use}
                    row["n_with_new"] = int(sum(len(v) for v in hit.values()))
                    row.update(est(*mixture([(weights[k] * len(hit[k]) / len(cells[k]),
                                              *boot.draws(hit[k].new_in_hole == hit[k].n_new, hit[k].source)) for k in use]), "all_new_in_hole"))
                    if len(use) == 1:
                        row.update(est(*boot.quantile(hit[use[0]].delay, hit[use[0]].source), "delay_median"))
                    rows.append(row)
            for b in range(3):
                c = r[(r.b == b) & (r.stratum == "clean")]
                for k, share in c.extra_wrong.clip(-2, 10).value_counts(normalize=True).sort_index().items():
                    dist.append({"engine": engine, "family": fam.key, "t0": fam.t0, "break_size": b, "extra_wrong_deletions": k,
                                 "share": share, "n": len(c)})
    return pd.DataFrame(rows), pd.DataFrame(dist)


def contrast_runs(runs: pd.DataFrame, boot: Boot) -> pd.DataFrame:
    """Clean sites: every break size against the placebo (b = 0), resampled together."""
    rows = []
    for engine in ("model", "exact"):
        for fam in FAMILIES:
            r = runs[(runs.engine == engine) & (runs.fam == fam.key) & (runs.stratum == "clean")]
            if not len(r):
                continue
            for m in ("extra_wrong", "any_new", "gram_int"):
                p0, r0 = boot.draws(r[r.b == 0][m], r[r.b == 0].source)
                for b in (1, 2):
                    pb, rb = boot.draws(r[r.b == b][m], r[r.b == b].source)
                    rows.append({"engine": engine, "family": fam.key, "t0": fam.t0, "metric": m, "break_size": b,
                                 **est(pb - p0, rb - r0, "minus_placebo"), "n": int((r.b == b).sum()), "n_placebo": int((r.b == 0).sum())})
    return pd.DataFrame(rows)


SCOPES = {"clean": "no leftover insertion within 3 letters", "strict": "no leftover insertion within 12 letters"}


def repair_tables(runs: pd.DataFrame, pool: dict, boot: Boot):
    """The number of further deletions against the minimal repair, and the fate of the neighbourhood.

    `need` is the minimal repair of the state after the removal minus that of the state before it (the control
    state needs a repair where leftover insertions sit within 10 letters; on `strict` sites it needs none and
    every further deletion is a wrong one). Break sizes 1 and 2 are mixed with their natural shares."""
    cross, fate = [], []
    usable = runs[(runs.mr_int >= 0) & (runs.mr_ctl >= 0) & (runs.stratum == "clean")]
    usable = usable.assign(need=usable.mr_int - usable.mr_ctl)
    for scope in SCOPES:
        for engine in ("model", "exact"):
            for fam in FAMILIES:
                r = usable[(usable.engine == engine) & (usable.fam == fam.key) & (usable.clean12 | (scope == "clean"))]
                if not len(r):
                    continue
                weights = natural_weights(pool, fam)
                key = {"sites": scope, "engine": engine, "family": fam.key, "t0": fam.t0}
                for (need, extra), n in r.groupby([r.need.clip(0, 6), r.extra_deletions.clip(-1, 8)]).size().items():
                    cross.append({**key, "minimal_repair": need, "extra_deletions": extra, "n": n})
                broken = r[r.b > 0]
                w = broken.b.map({b: weights[b, "clean"] / max((broken.b == b).sum(), 1) for b in (1, 2)})
                classes = {
                    "fewer_than_minimal": broken.extra_deletions < broken.need,
                    "equal_to_minimal": broken.extra_deletions == broken.need,
                    "more_than_minimal": broken.extra_deletions > broken.need,
                    "repaired": broken.gram_int,
                    "repaired_with_minimal": broken.gram_int & (broken.extra_deletions == broken.need),
                    "repaired_with_more": broken.gram_int & (broken.extra_deletions > broken.need),
                    "repaired_with_fewer": broken.gram_int & (broken.extra_deletions < broken.need),
                    "unresolved": ~broken.gram_int,
                    "unresolved_untouched": ~broken.gram_int & (broken.n_new == 0),
                    "hole_equals_smallest_contiguous": broken.hole == broken.lmin,
                }
                row = {**key, "n": len(broken), "n_sources": broken.source.nunique()}
                for name, flag in classes.items():
                    row.update(est(*boot.draws(flag, broken.source, w), name))
                hit = broken[broken.n_new > 0]
                row.update(est(*boot.draws(hit.new_in_hole == hit.n_new, hit.source, w[hit.index]), "all_new_in_hole_given_new"))
                for name in ("extra_deletions", "need", "hole", "lmin"):
                    row.update(est(*boot.draws(broken[name], broken.source, w), f"mean_{name}"))
                fate.append(row)
    return pd.DataFrame(cross), pd.DataFrame(fate)


def hole_survival(runs: pd.DataFrame, pool: dict, boot: Boot) -> pd.DataFrame:
    """P(hole > L) over clean sites with the natural mix of break sizes, next to what the grammar alone requires
    (smallest contiguous hole, minimal repair + 1) and the junction law (1 - v^2)^(L(L+1)/2), v = 1/4. The grammar
    curves use only sites without a leftover insertion within `MARGIN` letters: elsewhere the minimal repair would
    also count the leftover insertions."""
    rows = []
    sizes = np.arange(0, REACH + 2)
    for engine in ("model", "exact"):
        for fam in FAMILIES:
            r = runs[(runs.engine == engine) & (runs.fam == fam.key) & (runs.stratum == "clean") & (runs.mr_int >= 0)]
            if not len(r):
                continue
            weights = natural_weights(pool, fam)
            strict = r[r.clean12]
            for name, part, col in (("hole of the run", r, r.hole), ("smallest contiguous hole (grammar)", strict, strict.lmin),
                                    ("minimal repair + 1 (grammar)", strict, strict.mr_int + 1)):
                if part.b.nunique() < 3:
                    continue
                w = part.b.map({b: weights[b, "clean"] / (part.b == b).sum() for b in range(3)})
                point, rep = boot.draws(col.to_numpy()[:, None] > sizes[None], part.source, w)
                lo, hi = np.nanpercentile(rep, [2.5, 97.5], axis=0)
                mean = est(*boot.draws(col, part.source, w))
                med = est(*boot.quantile(col, part.source, 0.5, w))
                rows += [{"engine": engine, "family": fam.key, "t0": fam.t0, "curve": name, "L": int(L), "survival": point[j],
                          "survival_lo": lo[j], "survival_hi": hi[j], "law": (1 - 1 / 16) ** (L * (L + 1) / 2), "n": len(part),
                          "mean_size": mean["mean"], "mean_size_lo": mean["lo"], "mean_size_hi": mean["hi"],
                          "median_size": med["mean"], "median_size_lo": med["lo"], "median_size_hi": med["hi"]} for j, L in enumerate(sizes)]
    return pd.DataFrame(rows)


def paired_engines(runs: pd.DataFrame, boot: Boot) -> pd.DataFrame:
    """Experiment 4: the sites that were sampled with the model's rates and with exact rates."""
    cols = ["site", "extra_wrong", "any_new", "gram_int", "hole", "extra_missed"]
    both = runs[(runs.engine == "model") & runs.paired].merge(runs[runs.engine == "exact"][cols], on="site", suffixes=("_model", "_exact"))
    rows = []
    for fam in FAMILIES:
        for b in (0, 1, 2):
            c = both[(both.fam == fam.key) & (both.b == b)]
            if not len(c):
                continue
            row = {"family": fam.key, "t0": fam.t0, "break_size": b, "n": len(c), "n_sources": c.source.nunique()}
            for m in cols[1:]:
                pm, rm = boot.draws(c[f"{m}_model"], c.source)
                pe, re = boot.draws(c[f"{m}_exact"], c.source)
                row.update(est(pm, rm, f"{m}_model"))
                row.update(est(pe, re, f"{m}_exact"))
                row.update(est(pm - pe, rm - re, f"{m}_model_minus_exact"))
            rows.append(row)
    return pd.DataFrame(rows)


def confounders(runs: pd.DataFrame, boot: Boot) -> pd.DataFrame:
    """The effect on clean sites (b >= 1 against the placebo) within thirds of the noise level of the source, of its
    length and of the position of the site."""
    rows = []
    r = runs[(runs.engine == "model") & (runs.stratum == "clean")]
    for fam in FAMILIES:
        f = r[r.fam == fam.key]
        for name in ("rho", "length", "position"):
            third = pd.qcut(f[name], 3, labels=False, duplicates="drop")
            for level in sorted(third.dropna().unique()):
                part = f[third == level]
                p0, r0 = boot.draws(part[part.b == 0].extra_wrong, part[part.b == 0].source)
                p1, r1 = boot.draws(part[part.b > 0].extra_wrong, part[part.b > 0].source)
                rows.append({"family": fam.key, "t0": fam.t0, "covariate": name, "third": ["low", "middle", "high"][int(level)], "n": len(part),
                             "range": f"{part[name].min():.3g} - {part[name].max():.3g}",
                             **est(p0, r0, "extra_wrong_placebo"), **est(p1, r1, "extra_wrong_broken"),
                             **est(p1 - p0, r1 - r0, "difference")})
    return pd.DataFrame(rows)


# ---- analysis: the mirror (experiment 3) -----------------------------------------------------------------------------


def mirror_outcome(seq, ins, pos, d_free, d_keep) -> dict:
    nb = np.concatenate([np.arange(pos - REACH, pos), np.arange(pos + 1, pos + REACH + 1)])
    orig = ~ins[nb]
    gone_f, gone_k = d_free[nb] >= 0, d_keep[nb] >= 0
    new, rescued = orig & gone_k & ~gone_f, orig & gone_f & ~gone_k

    def beside(d):  # is the nearest letter that is not a correctly deleted insertion a wrongly deleted original?
        out = []
        for step in (-1, 1):
            j = pos + step
            while 0 <= j < len(seq) and ins[j] and d[j] >= 0:
                j += step
            out.append(bool(0 <= j < len(seq) and not ins[j] and d[j] >= 0))
        return out

    lo, hi = pos - REACH, pos + REACH
    window = slice(lo, hi + 1)
    target = seq[window][~ins[window]]
    same = lambda d: bool(np.array_equal(seq[window][d[window] < 0], target))  # noqa: E731
    (lf, rf), (lk, rk) = beside(d_free), beside(d_keep)
    wrong_k = np.zeros(2 * REACH + 1)
    wrong_k[np.r_[0:REACH, REACH + 1 : 2 * REACH + 1]] = (orig & gone_k).astype(float) - (orig & gone_f).astype(float)
    return {
        "deleted_in_free_run": bool(d_free[pos] >= 0), "step_in_free_run": int(d_free[pos]),
        "n_new": int(new.sum()), "n_rescued": int(rescued.sum()), "extra_wrong": int(new.sum()) - int(rescued.sum()),
        "any_new": bool(new.any()), "wrong_free": int((orig & gone_f).sum()), "wrong_keep": int((orig & gone_k).sum()),
        "extra_missed_others": int((~orig & ~gone_k).sum()) - int((~orig & ~gone_f).sum()),
        "left_x_free": lf, "right_x_free": rf, "left_x_keep": lk, "right_x_keep": rk,
        "beside_free": lf or rf, "beside_keep": lk or rk, "both_sides_keep": lk and rk, "both_sides_free": lf and rf,
        "gram_free": grammatical(seq, d_free < 0, lo, hi), "gram_keep": grammatical(seq, d_keep < 0, lo, hi),
        "same_string_free": same(d_free), "same_string_keep": same(d_keep), "profile": wrong_k,
    }


def analyse_mirror(study: Study, sites: pd.DataFrame, pool: dict, boot: Boot, notes: list):
    fam = FAMILY["f00"]
    runs = load_runs(study, sites, fam, "model", [])
    s = sites[(sites.exp == "mirror") & sites.site.isin(list(runs))]
    if len(s) < (sites.exp == "mirror").sum():
        notes.append(f"experiment 3: {(sites.exp == 'mirror').sum() - len(s)} of {(sites.exp == 'mirror').sum()} sites not sampled yet")
    rows = []
    for sid, i, pos, stratum, q in zip(s.site, s.source, s.pos, s.stratum, s.q_true):
        d_free, d_keep, _, _ = runs[sid]
        rows.append({"site": sid, "source": i, "pos": pos, "stratum": stratum, "q_true": q,
                     **mirror_outcome(study.sources.seq(i), study.sources.mask(i), pos, d_free, d_keep)})
    mirror = pd.DataFrame(rows)
    if not len(mirror):
        return mirror, pd.DataFrame(), pd.DataFrame()
    share = {st: pool["f00"][st] for st, _, _ in VISIBILITY}
    metrics = ["deleted_in_free_run", "extra_wrong", "any_new", "n_new", "n_rescued", "wrong_free", "wrong_keep", "extra_missed_others",
               "beside_free", "beside_keep", "left_x_keep", "right_x_keep", "left_x_free", "right_x_free", "both_sides_keep",
               "gram_free", "gram_keep", "same_string_free", "same_string_keep"]
    table, profile = [], []
    for group in [st for st, _, _ in VISIBILITY] + ["all insertions"]:
        use = [group] if group != "all insertions" else [st for st, _, _ in VISIBILITY]
        parts = {st: mirror[mirror.stratum == st] for st in use}
        if any(len(p) == 0 for p in parts.values()):
            continue
        for subset in ("all sites", "deleted in the free run"):
            sub = {st: (p if subset == "all sites" else p[p.deleted_in_free_run]) for st, p in parts.items()}
            wts = {st: share[st] * (len(sub[st]) / len(parts[st])) for st in use}
            row = {"visibility": group, "sites": subset, "n": int(sum(len(p) for p in sub.values())),
                   "n_sources": int(pd.concat(list(sub.values())).source.nunique()),
                   "share_of_insertions": sum(share[st] for st in use) / sum(share.values())}
            for m in metrics:
                row.update(est(*mixture([(wts[st], *boot.draws(sub[st][m], sub[st].source)) for st in use]), m))
            row.update(est(*mixture([(wts[st], *boot.draws(sub[st].beside_keep.astype(float) - sub[st].beside_free.astype(float),
                                                           sub[st].source)) for st in use]), "beside_keep_minus_free"))
            table.append(row)
            if subset == "all sites":
                point, rep = mixture([(wts[st], *boot.draws(np.stack(list(sub[st].profile)), sub[st].source)) for st in use])
                lo, hi = np.nanpercentile(rep, [2.5, 97.5], axis=0)
                profile += [{"visibility": group, "offset": int(o), "extra_wrong_deletions": point[j], "lo": lo[j], "hi": hi[j],
                             "n": row["n"]} for j, o in enumerate(OFFSETS) if o != 0]
    return mirror.drop(columns="profile"), pd.DataFrame(table), pd.DataFrame(profile)


# ---- figures ---------------------------------------------------------------------------------------------------------

B_LABEL = {0: "break size 0 (placebo)", 1: "break size 1", 2: "break size 2"}
ENGINE_LABEL = {"model": "trained model", "exact": "ideal model (exact posterior)"}
INK, MUTED = "#222222", "#666666"


def figure(rows: int, cols: int, size, title: str, **kw):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, **kw)
    lines = textwrap.wrap(title, int(size[0] * 9.6))
    fig.text(0.012, 1 - 0.1 / size[1], "\n".join(lines), ha="left", va="top", fontsize=11.5, fontweight="bold", color=INK, linespacing=1.25)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#999999")
        ax.grid(True, color="#E3E3E3", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5)
    fig._top = 1 - (0.2 * len(lines) + 0.32) / size[1]
    return fig, axes


def shade(colour: str, f: float) -> str:
    """The colour mixed with white (f = 0: unchanged)."""
    rgb = [int(colour[k : k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * f):02X}" for c in rgb)


def looks(n: int) -> list[dict]:
    """Curves of one sampler under n ordered conditions: one hue, lighter along the order, another marker each."""
    return [{"color": shade(COLOURS[0], 0.6 * k / max(n - 1, 1)), "marker": "osD^v"[k % 5]} for k in range(n)]


def legend_above(fig, ax, ncol: int = 4):
    """One legend for all panels, between the title and the panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.004, fig._top + 0.04 / fig.get_size_inches()[1]), ncol=ncol,
               fontsize=8.5, frameon=False)
    fig._top -= 0.3 / fig.get_size_inches()[1]


def save(fig, path: Path, note: str = ""):
    import matplotlib.pyplot as plt

    bottom = 0.0
    if note:
        lines = textwrap.wrap(note, int(fig.get_size_inches()[0] * 15.5))
        bottom = (0.17 * len(lines) + 0.08) / fig.get_size_inches()[1]
        fig.text(0.012, 0.012, "\n".join(lines), ha="left", va="bottom", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, bottom, 1, fig._top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bars(ax, x, e, name, colour, label=None, marker="o", shift=0.0, annotate="", digits=2, hollow=False):
    """Points with their bootstrap interval from a list of table rows."""
    y = np.array([r[name] for r in e], dtype=float)
    lo = np.array([r[f"{name}_lo"] for r in e], dtype=float)
    hi = np.array([r[f"{name}_hi"] for r in e], dtype=float)
    xs = np.asarray(x, dtype=float) + shift
    ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt=marker, color=colour, markersize=6, linewidth=1.4, capsize=2.5, label=label,
                markeredgecolor=colour if hollow else "white", markerfacecolor="white" if hollow else colour,
                markeredgewidth=1.4 if hollow else 0.8)
    if annotate:  # "top", "left" or "right" of the point
        for xx, yy, top in zip(xs, y, hi):
            xy, move, ha, va = {"top": ((xx, top), (0, 4), "center", "bottom"), "left": ((xx, yy), (-7, 0), "right", "center"),
                                "right": ((xx, yy), (7, 0), "left", "center")}[annotate]
            ax.annotate(f"{yy:.{digits}f}", xy, textcoords="offset points", xytext=move, ha=ha, va=va, fontsize=8, color=INK)
    return xs, y


def versus(a: float, b: float, words=("less than", "about as much as", "more than")) -> str:
    """The wording of a comparison, so that a title stays true for another checkpoint."""
    return words[0] if a < 0.85 * b else (words[2] if a > 1.15 * b else words[1])


def pick(table: pd.DataFrame, **where) -> list[dict]:
    m = np.ones(len(table), dtype=bool)
    for k, v in where.items():
        m &= (table[k].astype(str) == str(v)).to_numpy()
    return table[m].to_dict("records")


def fig_profile(out: Path, profile: pd.DataFrame, suspicion: pd.DataFrame):
    fams = [f for f in FAMILIES if len(pick(profile, family=f.key, stratum="clean"))]
    engines = [e for e in ("model", "exact") if len(pick(profile, engine=e))]
    ref = pick(suspicion, family="f06", stratum="clean", break_size=2)[0]
    zero = pick(suspicion, family="f06", stratum="clean", break_size=0)[0]
    title = ("One forward pass after removing an original letter: the neighbours become suspicious if the removal breaks the grammar "
             f"(t0 = 0.6, break size 2: summed change of q {ref.get('S_model', np.nan):+.2f} letters; placebo "
             f"{zero.get('S_model', np.nan):+.3f}), and the trained model reacts "
             f"{versus(ref.get('S_model', np.nan), ref.get('S_exact', np.nan))} an ideal model would ({ref.get('S_exact', np.nan):+.2f}).")
    fig, axes = figure(len(engines), len(fams), (3.0 * len(fams), 2.9 * len(engines) + 1.1), title, sharex=True, sharey=True)
    for r, engine in enumerate(engines):
        for c, fam in enumerate(fams):
            ax = axes[r, c]
            for b in range(3):
                rows = pd.DataFrame(pick(profile, family=fam.key, stratum="clean", engine=engine, break_size=b))
                if not len(rows):
                    continue
                for side in (rows[rows.offset < 0], rows[rows.offset > 0]):
                    ax.fill_between(side.offset, side.dq_lo, side.dq_hi, color=COLOURS[b], alpha=0.18, linewidth=0)
                    ax.plot(side.offset, side.dq, color=COLOURS[b], linewidth=1.6, marker="o", markersize=3,
                            label=B_LABEL[b] if side.offset.iloc[0] < 0 else None)
            ax.axvline(0, color="#999999", linewidth=0.8, linestyle=":")
            if r == 0:
                ax.set_title(fam.label, fontsize=9.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{ENGINE_LABEL[engine]}\nchange of q of a neighbour", fontsize=9, color=INK)
            if r == len(engines) - 1:
                ax.set_xlabel("position relative to the removed letter", fontsize=9, color=INK)
    legend_above(fig, axes[0, 0])
    save(fig, out / "fig1_suspicion_profile.png",
         "q = probability that a letter is a leftover insertion (model: its deletion rate divided by the hazard; ideal model: exact posterior "
         "with the prior a model can have at t0). Sites without a leftover insertion within 3 letters; bands: 95 % bootstrap intervals over sources. "
         "Break size = how many of the two letter windows across the gap the grammar forbids.")


def fig_totals(out: Path, suspicion: pd.DataFrame, by_repair: pd.DataFrame):
    fams = [f for f in FAMILIES if len(pick(suspicion, family=f.key, stratum="clean"))]
    r6 = {b: pick(suspicion, family="f06", stratum="clean", break_size=b)[0] for b in range(3)}
    three = (pick(by_repair, family="f06", minimal_repair=3) or [{}])[0]
    title = ("Summed suspicion on the neighbours right after a removal: an ideal model raises it by the size of the minimal repair "
             f"({three.get('S_exact', np.nan):.2f} letters where 3 further deletions are needed, t0 = 0.6), the trained model by "
             f"{three.get('S_model', np.nan):.2f} letters there; after the placebo the change is "
             f"{r6[0].get('S_model', np.nan):+.3f} (model) and {r6[0].get('S_exact', np.nan):+.3f} (ideal).")
    fig, axes = figure(2, len(fams), (2.9 * len(fams), 7.4), title, sharey=True)
    groups = ["0", "1", "2", "3", "4", "5+"]
    for c, fam in enumerate(fams):
        ax = axes[0, c]
        rows = [pick(suspicion, family=fam.key, stratum="clean", break_size=b)[0] for b in range(3)]
        for k, engine in enumerate(("model", "exact")):
            if f"S_{engine}" in rows[0] and np.isfinite(rows[0][f"S_{engine}"]):
                bars(ax, range(3), rows, f"S_{engine}", COLOURS[k], ENGINE_LABEL[engine], "os"[k], (-0.1, 0.1)[k], ("left", "right")[k])
        ax.set_xticks(range(3), ["0 (placebo)", "1", "2"])
        ax.set_xlim(-0.75, 2.75)
        ax.set_title(fam.label, fontsize=9.5, color=INK)
        ax.set_xlabel("break size of the removal", fontsize=9, color=INK)
        ax = axes[1, c]
        rows = [r for g in groups for r in pick(by_repair, family=fam.key, minimal_repair=g)]
        xs = [groups.index(str(r["minimal_repair"])) for r in rows]
        ax.plot([0, 4], [0, 4], color=COLOURS[7], linewidth=1, linestyle="--")
        if c == 0:
            ax.annotate("suspicion =\nminimal repair", (2.9, 3.0), xytext=(0.3, 4.6), fontsize=8, color=MUTED,
                        arrowprops={"arrowstyle": "-", "color": COLOURS[7], "linewidth": 0.6})
        for k, engine in enumerate(("model", "exact")):
            if rows and f"S_{engine}" in rows[0] and np.isfinite(rows[0][f"S_{engine}"]):
                bars(ax, xs, rows, f"S_{engine}", COLOURS[k], None, "os"[k], (-0.1, 0.1)[k])
        ax.set_xticks(range(len(groups)), groups)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_xlabel("minimal repair (further deletions)", fontsize=9, color=INK)
    for r in range(2):
        axes[r, 0].set_ylabel("summed change of q on the\n20 neighbours (in letters)", fontsize=9, color=INK)
        for ax in axes[r]:
            ax.axhline(0, color="#999999", linewidth=0.8)
    legend_above(fig, axes[0, 0])
    save(fig, out / "fig2_suspicion_total.png",
         "Points: mean over sites, bars: 95 % bootstrap intervals over sources. Top: sites without a leftover insertion within 3 letters. "
         "Bottom: sites without one within 12 letters, grouped by the fewest further deletions that make the neighbourhood grammatical again "
         "(0 = placebo; break sizes 1 and 2 with their natural shares). A value of 1 means: one more letter is expected to be deleted.")


def family_axis(ax, fams):
    ax.set_xticks(range(len(fams)), [f.label.replace("clean target, ", "clean target\n").replace("t0 = ", "") for f in fams])
    ax.set_xlim(-0.5, len(fams) - 0.5)
    ax.set_xlabel("time t0 of the forced wrong deletion", fontsize=9, color=INK)


def fig_extra(out: Path, outcomes: pd.DataFrame):
    fams = [f for f in FAMILIES if len(pick(outcomes, engine="model", family=f.key, stratum="clean", break_size=2))]
    if not fams:
        return
    cell = lambda f, b: pick(outcomes, engine="model", family=f.key, stratum="clean", break_size=b)[0]  # noqa: E731
    mid = fams[0]
    late = [f for f in fams if f.kind == "flow"][-1]
    title = (f"Sampling on after a forced wrong deletion ({mid.label}): {cell(mid, 2)['extra_wrong']:.2f} further wrong deletions follow "
             f"within 10 letters if it broke two windows, {cell(mid, 1)['extra_wrong']:.2f} if one, {cell(mid, 0)['extra_wrong']:.2f} "
             f"after the placebo; at {late.label} the figures are {cell(late, 2)['extra_wrong']:.2f}, {cell(late, 1)['extra_wrong']:.2f} "
             f"and {cell(late, 0)['extra_wrong']:.2f}.")
    fig, axes = figure(1, 2, (11.5, 4.6), title)
    for ax, name, label in ((axes[0, 0], "extra_wrong", "further wrong deletions within 10 letters\n(forced run minus control run, per site)"),
                            (axes[0, 1], "any_new", "share of sites with at least one\nfurther wrong deletion")):
        for b in range(3):
            rows = [cell(f, b) for f in fams]
            xs, y = bars(ax, range(len(fams)), rows, name, COLOURS[b], B_LABEL[b], "osD"[b], (b - 1) * 0.12)
            flow = [j for j, f in enumerate(fams) if f.kind == "flow"]
            ax.plot(xs[flow], y[flow], color=COLOURS[b], linewidth=1.4)
            ax.annotate(f"{y[0]:.2f}", (xs[0], y[0]), textcoords="offset points", xytext=(-8, 0), ha="right", va="center", fontsize=8, color=INK)
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_ylabel(label, fontsize=9, color=INK)
        family_axis(ax, fams)
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[0, 0].legend(fontsize=8.5, frameon=False, loc="upper right")
    save(fig, out / "fig3_extra_wrong_deletions.png",
         "Control and intervened run use the same random numbers and differ only in the removed letter, which is not counted. "
         "Sites without a leftover insertion within 3 letters; bars: 95 % bootstrap intervals over sources. "
         "'Clean target': the source without any insertion, one letter removed at t0 = 0.9.")


def fig_repair(out: Path, cross: pd.DataFrame, fate: pd.DataFrame):
    c = cross[(cross.engine == "model") & (cross.t0 <= 0.6) & (cross.sites == "strict")]
    f = fate[(fate.engine == "model") & (fate.sites == "strict")]
    if not len(c) or not len(f):
        return
    grid = c.groupby(["minimal_repair", "extra_deletions"]).n.sum().unstack(fill_value=0).T
    grid = grid.reindex(index=range(-1, 9), columns=range(0, 7), fill_value=0)
    share = grid / grid.sum(axis=0).replace(0, np.nan)
    early = f[f.t0 <= 0.6]
    title = ("Size of the cascade against the repair the grammar needs: with time left (t0 <= 0.6) "
             f"{100 * early.equal_to_minimal.mean():.0f} % of the sites with a broken grammar get exactly the minimal number of further "
             f"deletions, {100 * early.fewer_than_minimal.mean():.0f} % get fewer and {100 * early.more_than_minimal.mean():.0f} % more "
             f"(mean {early.mean_extra_deletions.mean():.2f} further deletions, minimal repair {early.mean_need.mean():.2f}).")
    fig, axes = figure(1, 2, (12, 5.2), title, gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0, 0]
    ax.grid(False)
    ax.imshow(share.to_numpy(dtype=float), origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto",
              extent=(-0.5, 6.5, -1.5, 8.5))
    for (y, x), v in np.ndenumerate(share.to_numpy(dtype=float)):
        if np.isfinite(v) and v >= 0.005:
            ax.text(x, y - 1, f"{100 * v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
    ax.plot([-0.5, 6.5], [-0.5, 6.5], color=COLOURS[1], linewidth=1.2, linestyle="--")
    ax.text(-0.3, 7.4, "dashed: cascade = minimal repair", color=COLOURS[1], fontsize=8, ha="left", va="center")
    ax.set_xticks(range(7), [str(k) for k in range(6)] + ["6+"])
    ax.set_yticks(range(-1, 9), ["-1"] + [str(k) for k in range(8)] + ["8+"])
    for x, n in enumerate(grid.sum(axis=0)):
        ax.text(x, 8.75, f"n={int(n)}", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_xlabel("minimal repair: fewest further deletions that make the neighbourhood grammatical", fontsize=9, color=INK)
    ax.set_ylabel("further deletions within 10 letters in the run\n(intervened minus control)", fontsize=9, color=INK)
    ax.set_title("share of sites (%), per column; t0 = 0, 0.3, 0.6 pooled", fontsize=9.5, color=INK, pad=16)
    ax = axes[0, 1]
    names = [("fewer_than_minimal", "fewer than the minimal repair"), ("equal_to_minimal", "exactly the minimal repair"),
             ("more_than_minimal", "more than the minimal repair")]
    labels = [FAMILY[k].label for k in f.family]
    left = np.zeros(len(f))
    for j, (col, label) in enumerate(names):
        v = f[col].to_numpy(dtype=float)
        ax.barh(range(len(f)), v, left=left, color=COLOURS[j], edgecolor="white", linewidth=1.5, height=0.62, label=label)
        for y, (a, w) in enumerate(zip(left, v)):
            if w > 0.06:
                ax.text(a + w / 2, y, f"{100 * w:.0f} %", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    ax.set_yticks(range(len(f)), [f"{lab}\n(n = {n})" for lab, n in zip(labels, f.n)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of sites with break size 1 or 2", fontsize=9, color=INK)
    ax.set_title("further deletions in the run against the minimal repair", fontsize=9.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    save(fig, out / "fig4_cascade_vs_minimal_repair.png",
         "Sites without any leftover insertion within 12 letters, so that every further deletion is a wrong one and the minimal repair is "
         "unambiguous. Left: all break sizes as sampled (column 0 is the placebo). Right: break sizes 1 and 2 with their natural shares.")


def fig_hole(out: Path, survival: pd.DataFrame):
    s = survival[survival.engine == "model"]
    if not len(s):
        return
    fams = [f for f in FAMILIES if (s.family == f.key).any()]
    run = s[s.curve == "hole of the run"]
    first = run[run.family == fams[0].key].iloc[0]
    # what the grammar requires: from the clean-target experiment (one removal from a clean target), else from all families
    ref_fam = "tgt09" if (s.family == "tgt09").any() else None
    ref = s[s.family == ref_fam] if ref_fam else s
    need = ref[ref.curve == "minimal repair + 1 (grammar)"].mean_size.mean()
    exact = survival[(survival.engine == "exact") & (survival.curve == "hole of the run")]
    exact = exact[exact.family == exact.family.iloc[0]] if len(exact) else exact
    words = ("smaller than", "about as large as", "larger than")
    title = (f"The holes the model leaves are {versus(first.mean_size, need, words)} the repair the grammar needs: mean hole "
             f"{first.mean_size:.2f} letters (forced deletion included) at {fams[0].label}, against {need:.2f} for the minimal repair "
             + (f"({exact.mean_size.iloc[0]:.2f} with exact rates) " if len(exact) else "") + "and 4.97 for the junction-repair law.")
    fig, axes = figure(1, 1, (9.2, 5.6), title)
    ax = axes[0, 0]
    sizes = np.arange(0, REACH + 2)
    ax.plot(sizes, (1 - 1 / 16) ** (sizes * (sizes + 1) / 2), color=COLOURS[6], linewidth=2, label="junction-repair law  (1 - v²)^(L(L+1)/2),  v = 1/4")
    for name, style, label in (("smallest contiguous hole (grammar)", "--", "grammar: smallest contiguous hole that repairs"),
                               ("minimal repair + 1 (grammar)", ":", "grammar: minimal repair (any shape) + the forced letter")):
        g = ref[ref.curve == name].groupby("L").survival.mean()
        ax.plot(g.index, g.values, color=COLOURS[7], linewidth=1.6, linestyle=style, label=label)
    if len(exact):
        ax.plot(exact.L, exact.survival, color=COLOURS[1], linewidth=1.8, marker="s", markersize=4,
                label=f"exact rates (ideal model), {FAMILY[exact.family.iloc[0]].label}")
    for fam, look in zip(fams, looks(len(fams))):
        g = run[run.family == fam.key]
        ax.fill_between(g.L, g.survival_lo, g.survival_hi, color=look["color"], alpha=0.15, linewidth=0)
        ax.plot(g.L, g.survival, linewidth=1.8, markersize=4.5, label=f"trained model, {fam.label}", **look)
    ax.set_xlabel("L = hole size in letters (the forced deletion included)", fontsize=9, color=INK)
    ax.set_ylabel("share of sites whose hole is larger than L", fontsize=9, color=INK)
    ax.set_xticks(sizes)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, out / "fig5_hole_size_survival.png",
         "Hole = the original letters around the forced deletion that are missing from the output without a kept letter in between. "
         "Sites without a leftover insertion within 3 letters, break sizes mixed with their natural shares; bands: 95 % bootstrap intervals "
         "over sources. Grey curves: computed from the grammar alone on the sites of "
         + ("the clean-target experiment (one letter removed from a clean target)." if ref_fam else
            "all experiments that have no leftover insertion within 12 letters."))


def fig_mirror(out: Path, table: pd.DataFrame, profile: pd.DataFrame):
    if not len(table):
        return
    groups = [g for g in ("vis_hi", "vis_mid", "vis_lo") if len(pick(table, visibility=g, sites="all sites"))]
    names = {"vis_hi": "clearly visible\n(exact q >= 0.95)", "vis_mid": "ambiguous\n(0.6 - 0.95)", "vis_lo": "hidden\n(below 0.6)"}
    rows = [pick(table, visibility=g, sites="all sites")[0] for g in groups]
    short = [names[g].split(chr(10))[0] for g in groups]
    title = ("Mirror experiment: an inserted letter that is forced to stay adds "
             + ", ".join(f"{r['extra_wrong']:+.2f} ({k})" for r, k in zip(rows, short)) + " wrong deletions within 10 letters; a wrong "
             "deletion directly beside it occurs in " + ", ".join(f"{100 * r['beside_keep']:.0f}" for r in rows) + " % of the sites, against "
             + ", ".join(f"{100 * r['beside_free']:.0f}" for r in rows) + " % in the free run.")
    fig, axes = figure(1, 3, (13.5, 4.5), title)
    ax = axes[0, 0]
    bars(ax, range(len(groups)), rows, "extra_wrong", COLOURS[0], annotate="top")
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(range(len(groups)), [f"{names[g]}\nn = {r['n']}" for g, r in zip(groups, rows)])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("further wrong deletions within 10 letters\n(forced-to-stay run minus free run)", fontsize=9, color=INK)
    ax = axes[0, 1]
    bars(ax, range(len(groups)), rows, "beside_free", COLOURS[0], "free run", "o", -0.1, annotate="left")
    bars(ax, range(len(groups)), rows, "beside_keep", COLOURS[0], "letter forced to stay", "s", 0.1, annotate="right", hollow=True)
    ax.set_xticks(range(len(groups)), [names[g] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("share of sites with a wrong deletion directly\nbeside the letter (patterns x_ , _x , x_x)", fontsize=9, color=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax = axes[0, 2]
    for g, look in zip(groups, looks(len(groups))):
        p = profile[profile.visibility == g]
        for side in (p[p.offset < 0], p[p.offset > 0]):
            ax.fill_between(side.offset, side.lo, side.hi, color=look["color"], alpha=0.15, linewidth=0)
            ax.plot(side.offset, side.extra_wrong_deletions, linewidth=1.6, markersize=3.5,
                    label=names[g].replace("\n", " ") if side.offset.iloc[0] < 0 else None, **look)
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.set_xticks(range(-REACH, REACH + 1, 2))
    ax.set_xlabel("source position relative to the letter that stays", fontsize=9, color=INK)
    ax.set_ylabel("further wrong deletions at this position (per site)", fontsize=9, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    save(fig, out / "fig6_mirror_forced_insertion.png",
         "Both runs start from the untouched source (t0 = 0) with the same random numbers. Visibility: exact posterior probability that the "
         "letter is an insertion. 'Beside' ignores correctly deleted insertions in between. Bars and bands: 95 % bootstrap intervals over sources.")


def fig_engines(out: Path, paired: pd.DataFrame):
    if not len(paired):
        return
    fams = [f for f in FAMILIES if (paired.family == f.key).any()]
    ref = pick(paired, family=fams[min(1, len(fams) - 1)].key, break_size=2)[0]
    lab = fams[min(1, len(fams) - 1)].label
    words = ("a smaller cascade than", "about the same cascade as", "a larger cascade than")
    title = (f"With exact rates the same forced mistake triggers {versus(ref['extra_wrong_exact'], ref['extra_wrong_model'], words)} "
             f"with the trained model ({lab}, break size 2: {ref['extra_wrong_exact']:.2f} against {ref['extra_wrong_model']:.2f} further "
             f"wrong deletions) and the neighbourhood ends grammatical in {100 * ref['gram_int_exact']:.0f} % against "
             f"{100 * ref['gram_int_model']:.0f} % of the cases"
             + (": cascading is what a delete-only flow should do after a wrong deletion." if ref["extra_wrong_exact"] > 0.5 else "."))
    fig, axes = figure(2, len(fams), (3.1 * len(fams) + 0.6, 7.2), title, sharex=True, sharey="row")
    for c, fam in enumerate(fams):
        rows = [r for b in range(3) for r in pick(paired, family=fam.key, break_size=b)]
        xs = [r["break_size"] for r in rows]
        for r, (metric, label) in enumerate((("extra_wrong", "further wrong deletions\nwithin 10 letters (mean per site)"),
                                             ("gram_int", "share of neighbourhoods that are\ngrammatical at the end"))):
            ax = axes[r, c]
            bars(ax, xs, rows, f"{metric}_model", COLOURS[0], "trained model", "o", -0.1, annotate="left")
            bars(ax, xs, rows, f"{metric}_exact", COLOURS[1], "exact rates (ideal model)", "s", 0.1, annotate="right")
            ax.set_xticks(range(3), ["0\n(placebo)", "1", "2"])
            ax.set_xlim(-0.8, 2.8)
            if r == 0:
                ax.set_title(f"{fam.label}\nn = {rows[0]['n']} sites per break size", fontsize=9.5, color=INK)
                ax.axhline(0, color="#999999", linewidth=0.8)
            else:
                ax.set_ylim(-0.03, 1.1)
                ax.set_xlabel("break size of the removal", fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
    legend_above(fig, axes[0, 0])
    save(fig, out / "fig7_model_vs_exact_rates.png",
         "The same sites, the same forced removal; each sampler is compared with its own control run under common random numbers. "
         "Exact rates = hazard times the exact posterior of 'target plus leftover insertions'. Bars: 95 % bootstrap intervals over sources.")


def fig_fate(out: Path, fate: pd.DataFrame, runs: pd.DataFrame, study: Study):
    f = fate[(fate.engine == "model") & (fate.sites == "strict")]
    r = runs[(runs.engine == "model") & (runs.stratum == "clean") & (runs.b > 0)]
    if not len(f) or not len(r):
        return
    title = ("Fate of the neighbourhood after a forced mistake that broke the grammar: it ends grammatical in "
             + ", ".join(f"{100 * v:.0f} %" for v in f.repaired) + " of the cases at " + ", ".join(FAMILY[k].label for k in f.family)
             + f"; the first further deletion comes after a median of {np.nanmedian(r[r.fam == f.family.iloc[0]].delay):.0f} steps at "
             f"{FAMILY[f.family.iloc[0]].label}.")
    fig, axes = figure(1, 2, (12.5, 4.9), title)
    ax = axes[0, 0]
    f = f.assign(repaired_other=f.repaired_with_more + f.repaired_with_fewer)
    names = [("repaired_with_minimal", "grammatical, exactly the minimal repair"), ("repaired_other", "grammatical, another number of deletions"),
             ("unresolved", "a forbidden window is left (unresolved)")]
    left = np.zeros(len(f))
    for j, (col, label) in enumerate(names):
        v = f[col].to_numpy(dtype=float)
        ax.barh(range(len(f)), v, left=left, color=COLOURS[j], edgecolor="white", linewidth=1.5, height=0.62, label=label)
        for y, (a, w) in enumerate(zip(left, v)):
            if w > 0.06:
                ax.text(a + w / 2, y, f"{100 * w:.0f} %", ha="center", va="center", fontsize=8.5, color="white")
        left += v
    ax.set_yticks(range(len(f)), [f"{FAMILY[k].label}\n(n = {n})" for k, n in zip(f.family, f.n)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of sites with break size 1 or 2", fontsize=9, color=INK)
    ax.set_title("state of the neighbourhood at the end of the run", fontsize=9.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax = axes[0, 1]
    steps = np.arange(0, N_STEPS + 1)
    for fam, look in zip([FAMILY[k] for k in f.family], looks(len(f))):
        part = r[r.fam == fam.key]
        delay = part.delay.to_numpy()
        mine = steps[: N_STEPS - fam.k0 + 1]
        ax.step(mine, [(delay < d).mean() for d in mine], where="post", color=look["color"], linewidth=1.8,
                linestyle="-" if fam.kind == "flow" else "--", label=f"{fam.label} (n = {len(part)})")
        t = torch.tensor(np.minimum(fam.t0 + steps / N_STEPS, 1.0), dtype=torch.float64)
        base = study.kappa.cumulative_hazard(t) - study.kappa.cumulative_hazard(torch.tensor(fam.t0, dtype=torch.float64))
        if fam.kind == "flow":
            ax.plot(mine, (1 - torch.exp(-base)).numpy()[: len(mine)], color=look["color"], linewidth=1, linestyle=":")
    ax.set_xlabel("sampler steps since the forced deletion", fontsize=9, color=INK)
    ax.set_ylabel("share of sites that had their first\nfurther wrong deletion by then", fontsize=9, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, N_STEPS)
    ax.legend(fontsize=8, frameon=False, loc="lower right", title="dotted: one letter with q = 1\nfrom the same time on", title_fontsize=8)
    save(fig, out / "fig8_fate_and_delay.png",
         "Left: sites without a leftover insertion within 12 letters, break sizes 1 and 2 with their natural shares. Right: sites without one "
         "within 3 letters, break size 1 or 2 as sampled; a curve ends with its run, at the share of sites that ever had a further wrong deletion.")


# ---- examples --------------------------------------------------------------------------------------------------------


def write_examples(study: Study, sites: pd.DataFrame, runs: pd.DataFrame, path: Path):
    """Ten cascades of the trained model with their letters, verdicts and deletion steps."""
    r = runs[(runs.engine == "model") & (runs.stratum == "clean") & (runs.b > 0) & (runs.n_new > 0)]
    chosen = []
    for fam in FAMILIES:
        f = r[r.fam == fam.key]
        wanted = [("repaired by a contiguous hole of exactly the minimal size",
                   f.clean12 & f.gram_int & (f.extra_deletions == f.mr_int) & (f.new_in_hole == f.n_new) & (f.n_new >= 2)),
                  ("cascade that leaves a forbidden window", ~f.gram_int & (f.n_new >= 2))]
        for label, flag in wanted:
            pool_ = f[flag & ~f.site.isin([c[1].site for c in chosen])]
            if not len(pool_):
                pool_ = f[~f.site.isin([c[1].site for c in chosen])]
                label = "cascade"
            if len(pool_):
                chosen.append((label, pool_.sort_values("site").iloc[0]))
    lines = ["# Ten cascades after a forced wrong deletion (trained model)", "",
             "Each block shows 15 letters of the state on each side of the forced site. Row `letters`: the state at t0 "
             "(capital = original letter, lower case = leftover insertion, `[ ]` = the letter that is removed by force). "
             "Rows `control` and `forced`: the verdict of every letter at the end of the run without and with the removal "
             "(`.` original kept, `x` original deleted = wrong deletion, `_` insertion kept, `+` insertion deleted, `X` the forced deletion). "
             "Both runs use the same random numbers. `steps`: offset from the site and sampler step (0-99) of every deletion of the forced run; "
             "`=` marks deletions that the control run made as well.", ""]
    cache: dict = {}
    for n, (label, row) in enumerate(chosen, 1):
        fam = FAMILY[row.fam]
        if fam.key not in cache:
            cache[fam.key] = load_runs(study, sites, fam, "model", [])
        d_ctl, d_int, _, _ = cache[fam.key][row.site]
        seq, ins, alive = study.sources.seq(row.source), study.sources.mask(row.source), study.alive(fam, row.source)
        idx = np.flatnonzero(alive)
        s = int(np.searchsorted(idx, row.pos))
        window = idx[max(0, s - 15) : s + 16]

        def verdicts(d, forced):
            out = []
            for j in window:
                v = ("+" if d[j] >= 0 else "_") if ins[j] else ("x" if d[j] >= 0 else ".")
                out.append(("[X]" if forced else f"[{v}]") if j == row.pos else v)
            return "".join(out)

        letters = "".join((f"[{ALPHABET[seq[j]]}]" if j == row.pos else (ALPHABET[seq[j]].lower() if ins[j] else ALPHABET[seq[j]])) for j in window)
        steps = [f"{int(np.searchsorted(idx, j)) - s:+d}:{int(d_int[j])}{'=' if d_ctl[j] >= 0 else ''}" for j in window if d_int[j] >= 0]
        kept = lambda d, forced: "".join(ALPHABET[seq[j]] for j in window if d[j] < 0 and not (forced and j == row.pos))  # noqa: E731
        lines += [f"## {n}. {fam.label}: {label}", "",
                  f"source {row.source}, site at source position {row.pos}, break size {row.b}; the grammar needs {row.mr_int - row.mr_ctl} "
                  f"further deletion(s) within 10 letters (minimal repair, net of what the state needed before the removal), the smallest "
                  f"contiguous hole that repairs has {row.lmin} letter(s). Result: {row.n_new} further wrong deletion(s), hole of {row.hole} letter(s), "
                  f"neighbourhood grammatical at the end: {'yes' if row.gram_int else 'no'} (control: {'yes' if row.gram_ctl else 'no'}); "
                  f"first further deletion {int(row.delay)} step(s) after the forced one.", "",
                  "```", f"letters  {letters}", f"control  {verdicts(d_ctl, False)}", f"forced   {verdicts(d_int, True)}",
                  f"steps    {'  '.join(steps) if steps else '-'}", f"output of the control run  {kept(d_ctl, False)}",
                  f"output of the forced run   {kept(d_int, True)}", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---- analysis: driver ------------------------------------------------------------------------------------------------


def records(table: pd.DataFrame, digits: int = 4) -> list[dict]:
    out = []
    for row in table.to_dict("records"):
        out.append({k: (round(float(v), digits) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer, np.bool_)) else v))
                    for k, v in row.items() if not (isinstance(v, float) and np.isnan(v))})
    return out


def analyse(study: Study, sites: pd.DataFrame, pool: dict):
    import matplotlib

    matplotlib.use("Agg")
    out, notes = study.out, []
    boot = Boot(study.args.boot, len(study.sources), study.args.seed)
    ann = annotate(study, sites)
    lengths = np.diff(study.sources.offsets)

    one, suspicion, profile = analyse_onepass(study, sites, pool, boot, notes)
    one = one.merge(ann, on="site", how="left")
    placebo, by_repair = onepass_extras(one, boot)
    one.drop(columns=["row", "order", "exact", "q_true", "exp"]).to_csv(out / "exp1_sites.csv", index=False)
    suspicion.to_csv(out / "exp1_suspicion.csv", index=False)
    profile.to_csv(out / "exp1_profile.csv", index=False)
    placebo.to_csv(out / "exp1_placebo.csv", index=False)
    by_repair.to_csv(out / "exp1_by_minimal_repair.csv", index=False)

    runs = analyse_runs(study, sites, notes)
    tables = {}
    if len(runs):
        runs = runs.merge(ann, on="site", how="left")
        runs["any_new"] = runs.n_new > 0
        runs["rho"], runs["length"] = study.rho[runs.source], lengths[runs.source]
        runs["position"] = runs.pos / runs.length
        runs.to_csv(out / "exp2_sites.csv", index=False)
        tables["outcomes"], tables["distribution"] = summarise_runs(runs, pool, boot)
        tables["vs_placebo"] = contrast_runs(runs, boot)
        tables["cross"], tables["fate"] = repair_tables(runs, pool, boot)
        tables["hole"] = hole_survival(runs, pool, boot)
        tables["confounders"] = confounders(runs, boot)
        tables["paired"] = paired_engines(runs, boot)
        for name, file in (("outcomes", "exp2_outcomes.csv"), ("distribution", "exp2_distribution.csv"), ("vs_placebo", "exp2_vs_placebo.csv"),
                           ("cross", "exp2_minimal_repair_cross.csv"), ("fate", "exp2_fate.csv"), ("hole", "exp2_hole_survival.csv"),
                           ("confounders", "exp2_confounders.csv"), ("paired", "exp4_model_vs_exact.csv")):
            tables[name].to_csv(out / file, index=False)
    mirror, mirror_table, mirror_profile = analyse_mirror(study, sites, pool, boot, notes)
    if len(mirror):
        mirror.to_csv(out / "exp3_sites.csv", index=False)
        mirror_table.to_csv(out / "exp3_mirror.csv", index=False)
        mirror_profile.to_csv(out / "exp3_profile.csv", index=False)

    null = {}
    for fam in FAMILIES:
        path = study.scratch / f"calls-{fam.key}-{PASSES[0]}-0.npz"
        if path.exists():
            null[fam.key] = int(np.load(path)["null_diff"])
    if len(profile):
        fig_profile(out, profile, suspicion)
        fig_totals(out, suspicion, by_repair)
    if len(runs):
        fig_extra(out, tables["outcomes"])
        fig_repair(out, tables["cross"], tables["fate"])
        fig_hole(out, tables["hole"])
        fig_engines(out, tables["paired"])
        fig_fate(out, tables["fate"], runs, study)
        write_examples(study, sites, runs, out / "examples.md")
    fig_mirror(out, mirror_table, mirror_profile)

    keep = lambda t, **where: records(pd.DataFrame(pick(t, **where))) if len(t) else []  # noqa: E731
    def triple(rows, name):
        return [round(rows[0][k], 3) for k in (name, f"{name}_lo", f"{name}_hi")] if rows and name in rows[0] else None

    headline = {"reading": "estimate, lower and upper end of the 95 % bootstrap interval over sources; clean sites"}
    for fam in FAMILIES:
        block = {}
        for b in range(3):
            one_ = pick(suspicion, family=fam.key, stratum="clean", break_size=b)
            run_ = pick(tables["outcomes"], engine="model", family=fam.key, stratum="clean", break_size=b) if tables else []
            ex_ = pick(tables["outcomes"], engine="exact", family=fam.key, stratum="clean", break_size=b) if tables else []
            block[f"break_size_{b}"] = {
                "suspicion_model": triple(one_, "S_model"), "suspicion_ideal": triple(one_, "S_exact"),
                "further_wrong_deletions_model": triple(run_, "extra_wrong"), "further_wrong_deletions_exact_rates": triple(ex_, "extra_wrong"),
                "grammatical_at_end_model": triple(run_, "gram_int"), "grammatical_at_end_exact_rates": triple(ex_, "gram_int"),
                "sites_one_pass": one_[0]["n"] if one_ else 0, "sites_sampled": run_[0]["n"] if run_ else 0, "sites_exact_rates": ex_[0]["n"] if ex_ else 0}
        headline[fam.label] = block
    headline["mirror_further_wrong_deletions"] = {r["visibility"]: triple([r], "extra_wrong") for r in pick(mirror_table, sites="all sites")} if len(mirror_table) else {}
    summary = {
        "headline": headline,
        "run": study.run.as_posix(), "checkpoint": study.meta, "families": {f.key: {"label": f.label, "t0": f.t0, "kappa": study.kappa_at[f.key]} for f in FAMILIES},
        "sites": {f"{e}/{f}/{st}": int(n) for (e, f, st), n in sites.groupby(["exp", "fam", "stratum"]).size().items()},
        "site_pool": pool, "missing": notes,
        "checks": {"letters_that_differ_without_intervention": null,
                   "placebo_change_of_exact_posterior": keep(placebo, placebo_sites="all")},
        "experiment_1_suspicion_clean_sites": keep(suspicion, stratum="clean"),
        "experiment_1_suspicion_all_sites": keep(suspicion, stratum="all"),
        "experiment_1_by_minimal_repair": records(by_repair),
        "experiment_2_outcomes_clean_sites": keep(tables["outcomes"], engine="model", stratum="clean") if tables else [],
        "experiment_2_outcomes_all_sites": keep(tables["outcomes"], engine="model", stratum="all") if tables else [],
        "experiment_2_against_placebo": records(tables["vs_placebo"]) if tables else [],
        "experiment_2_fate_and_minimal_repair": records(tables["fate"]) if tables else [],
        "experiment_2_hole_size": records(tables["hole"][tables["hole"].L == 0].drop(columns=["L", "survival", "survival_lo", "survival_hi", "law"])) if tables else [],
        "experiment_3_mirror": records(mirror_table),
        "experiment_4_model_against_exact_rates": records(tables["paired"]) if tables else [],
        "experiment_4_outcomes_exact_rates": keep(tables["outcomes"], engine="exact", stratum="clean") if tables else [],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    clean = suspicion[suspicion.stratum == "clean"]
    print("\nexperiment 1, clean sites: summed change of q on the 20 neighbours\n" +
          show(clean, ["family", "break_size", "n", "n_sources"], ["S_model", "S_exact"], 3))
    if tables:
        t = tables["outcomes"]
        print("\nexperiment 2, clean sites: outcomes within 10 letters of the forced deletion\n" +
              show(t[t.stratum == "clean"], ["engine", "family", "break_size", "n", "n_sources"],
                   ["extra_wrong", "any_new", "extra_missed", "gram_int", "gram_ctl", "hole", "delay_median"]))
        print("\nfate of the sites with break size 1 or 2 (clean: no leftover insertion within 3 letters, strict: within 12)\n" +
              show(tables["fate"], ["sites", "engine", "family", "n"], ["fewer_than_minimal", "equal_to_minimal", "more_than_minimal",
                                                                        "repaired", "mean_extra_deletions", "mean_need", "mean_hole", "mean_lmin"]))
        if len(tables["paired"]):
            print("\nexperiment 4, the same sites with the model's and with exact rates\n" +
                  show(tables["paired"], ["family", "break_size", "n"], ["extra_wrong_model", "extra_wrong_exact", "gram_int_model", "gram_int_exact"]))
    if len(mirror_table):
        print("\nexperiment 3, an inserted letter that is forced to stay\n" +
              show(mirror_table, ["visibility", "sites", "n"], ["deleted_in_free_run", "extra_wrong", "any_new", "beside_free", "beside_keep"], 3))
    print(f"\nletters whose fate differs without intervention (must be 0): {null}")
    for line in notes:
        print("missing:", line)
    print(f"analysis written to {out}", flush=True)


# ---- main ----------------------------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/lap-01"))
    p.add_argument("--stages", default="sites,model,exact,analyse", help="comma-separated subset of sites,model,exact,analyse")
    p.add_argument("--scale", type=float, default=1.0, help="multiplies every sample size; a trial writes to its own folder")
    p.add_argument("--minutes", type=float, default=0.0, help="stop the model stage after this time (exit code 3); 0 = no limit")
    p.add_argument("--passes", default=",".join(PASSES), help="which site groups the model stage samples")
    p.add_argument("--families", default="", help="model stage: sample only these families, in this order (default: all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first", type=int, default=3000, help="sites come from the sources 0 .. first - 1")
    p.add_argument("--n-one", type=int, default=800, help="experiment 1: clean sites per break size and family")
    p.add_argument("--n-one-dirty", type=int, default=400, help="... sites with a leftover insertion within +-3 letters")
    p.add_argument("--n-run", type=int, default=200, help="experiment 2: clean sites per break size and family")
    p.add_argument("--n-run-dirty", type=int, default=75)
    p.add_argument("--run-sources", type=int, default=300, help="sources the sampled sites are drawn from")
    p.add_argument("--n-mirror", type=int, default=100, help="experiment 3: sites per visibility stratum")
    p.add_argument("--n-exact", type=int, default=150, help="experiment 4: clean sites per break size and family")
    p.add_argument("--exact-families", default="f00,f06,f09,tgt09")
    p.add_argument("--exact-batch", type=int, default=100)
    p.add_argument("--exact-budget", type=float, default=2e7)
    p.add_argument("--exact-device", default="cpu")
    p.add_argument("--max-tokens", type=int, default=20_000, help="letters per forward pass of the model")
    p.add_argument("--gpu-fraction", type=float, default=0.1, help="share of the GPU memory this process may use")
    p.add_argument("--boot", type=int, default=1000)
    args = p.parse_args()
    args.exact_families = args.exact_families.split(",")
    stages = args.stages.split(",")
    if torch.cuda.is_available() and "model" in stages:
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    study = Study(args)
    sites, pool = build_sites(study)
    print(f"{len(sites)} sites -> {study.out}", flush=True)
    if "model" in stages:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        deadline = time.time() + 60 * args.minutes if args.minutes else float("inf")
        passes = args.passes.split(",")
        done = stage_onepass_model(study, sites, device, deadline)
        done = done and stage_runs_model(study, sites, device, deadline, passes, [k for k in args.families.split(",") if k])
        if not done:
            print("model stage: time budget used up, repeat the command to continue", flush=True)
            sys.exit(3)
    if "exact" in stages:
        device = torch.device(args.exact_device)
        stage_onepass_exact(study, sites, device)
        stage_runs_exact(study, sites, device)
    if "analyse" in stages:
        analyse(study, sites, pool)


if __name__ == "__main__":
    main()
