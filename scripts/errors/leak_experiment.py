#!/usr/bin/env python
"""A perfect model with a known defect: how much damage does one unwarranted deletion do?

The exact-rate sampler is run with a deliberate *leak*: no letter's probability of being an insertion is taken
to be smaller than a small constant, so now and then a letter is deleted without reason, at a rate that is
known. Everything else, in particular the reaction to a broken sequence, stays exact.

    python scripts/build_error_corpus.py exact --first 500 --leak 0.001 --tag i2-leak0.001     # the corpora
    python scripts/errors/leak_experiment.py                                                  # this analysis

Writes `runs/error-study/results/leak/` (`leak.csv`, `summary.json`, `leak.png`). No trained model is involved.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from editflows.errors import Sources  # noqa: E402

STUDY = ROOT / "runs" / "error-study"
BLUE, VERMILLION, GREEN, ORANGE, SKY, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#7F7F7F"
KIND_COLOUR = {"Z": SKY, "V": BLUE, "R": ORANGE, "U": VERMILLION}
CUMULATIVE_HAZARD = 6.96  # Lambda(1) of the schedule 3 exp(1.5 t): a constant q = c deletes with probability 1 - exp(-c Lambda)


def per_source(tag: str, sources: Sources, n_sources: int) -> pd.DataFrame:
    """Per source: wrong deletions, missed insertions, originals, letters, and error letters by cluster kind."""
    z = np.load(STUDY / f"{tag}.npz")
    deleted = z["del_step"] >= 0
    clusters = pd.read_csv(STUDY / "results" / "clusters" / f"{tag}.csv")
    clusters = clusters[clusters.source < n_sources]
    by_kind = (clusters.n_x + clusters.n_miss).groupby([clusters.source, clusters.kind]).sum().unstack(fill_value=0)
    by_kind = by_kind.reindex(index=range(n_sources), columns=list("ZVRU"), fill_value=0)
    rows = []
    for i in range(n_sources):
        span, inserted = sources.span(i), sources.mask(i)
        d = deleted[span]
        rows.append({"wrong": int((d & ~inserted).sum()), "missed": int((~d & inserted).sum()),
                     "originals": int((~inserted).sum()), "letters": len(inserted)})
    return pd.concat([pd.DataFrame(rows), by_kind], axis=1)


def main():
    out = STUDY / "results" / "leak"
    out.mkdir(parents=True, exist_ok=True)
    sources = Sources.load(STUDY / "sources.npz")
    tags = sorted((p.stem for p in STUDY.glob("i2-leak*.npz")), key=lambda t: float(t[7:]))
    if not tags:
        raise SystemExit("no runs/error-study/i2-leak*.npz: build them first (see the docstring)")
    n = min(int(np.load(STUDY / f"{t}.npz")["first"]) for t in tags + ["i2-100"])
    base = per_source("i2-100", sources, n)
    rng = np.random.default_rng(0)
    draws = rng.integers(0, n, size=(1000, n))
    rows = []
    for tag in tags:
        leak = float(tag[7:])
        t = per_source(tag, sources, n)
        # originals the leak alone deletes (it acts on every original for the whole path)
        seeds = (1 - np.exp(-leak * CUMULATIVE_HAZARD)) * t.originals
        extra = t.wrong - base.wrong
        ratio = extra.sum() / seeds.sum()
        boot = extra.values[draws].sum(1) / seeds.values[draws].sum(1)
        row = {"leak": leak, "sources": n, "seeds_per_1000_letters": seeds.sum() / t.letters.sum() * 1000,
               "extra_wrong_per_1000_letters": extra.sum() / t.letters.sum() * 1000,
               "extra_missed_per_1000_letters": (t.missed - base.missed).sum() / t.letters.sum() * 1000,
               "wrong_per_seed": ratio, "wrong_per_seed_lo": np.quantile(boot, 0.025), "wrong_per_seed_hi": np.quantile(boot, 0.975)}
        for kind in "ZVRU":
            row[f"{kind}_per_1000_letters"] = t[kind].sum() / t.letters.sum() * 1000
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / "leak.csv", index=False)
    base_kinds = {kind: float(base[kind].sum() / base.letters.sum() * 1000) for kind in "ZVRU"}
    repair = json.loads((STUDY / "results" / "repair-law" / "summary.json").read_text()) if (STUDY / "results" / "repair-law" / "summary.json").exists() else {}
    summary = {"sources": n, "exact_rates_without_leak": base_kinds, "by_leak": table.to_dict(orient="records"),
               "wrong_deletions_per_seed_pooled": float(sum(r["extra_wrong_per_1000_letters"] for r in rows) / sum(r["seeds_per_1000_letters"] for r in rows)),
               "grammar_prediction_hole_per_seed": repair.get("mean_hole")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    x, y = table.seeds_per_1000_letters, table.extra_wrong_per_1000_letters
    ax.plot(x, y, "o", color=BLUE, ms=9, label="exact rates with a leak (measured)")
    top = x.max() * 1.1
    slope = summary["wrong_deletions_per_seed_pooled"]
    ax.plot([0, top], [0, slope * top], color=BLUE, lw=1.5, label=f"{slope:.1f} wrong deletions per unwarranted deletion")
    ax.plot([0, top], [0, top], color=GREY, lw=1.2, ls=":", label="if the damage stayed with the one letter")
    if repair:
        ax.plot([0, top], [0, repair["mean_hole"] * top], color=VERMILLION, lw=1.2, ls="--",
                label=f"grammar alone: smallest repair, {repair['mean_hole']:.1f} letters per hole")
    ax.set_xlabel("unwarranted deletions caused by the leak itself, per 1 000 letters")
    ax.set_ylabel("additional wrong deletions, per 1 000 letters")
    ax.set_title("One unwarranted deletion costs more than three letters,\neven when the reaction to it is exact", fontsize=11, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.3)
    ax = axes[1]
    labels = ["no leak"] + [f"leak {r['leak']:g}" for r in rows]
    left = np.zeros(len(labels))
    for kind, name in (("Z", "Z  same string, other label"), ("V", "V  other valid reading"), ("R", "R  repaired hole"), ("U", "U  left broken")):
        values = np.array([base_kinds[kind]] + [r[f"{kind}_per_1000_letters"] for r in rows])
        ax.barh(range(len(labels)), values, left=left, color=KIND_COLOUR[kind], edgecolor="white", label=name)
        left += values
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlabel("wrongly labelled letters per 1 000 source letters")
    ax.set_title("The additional errors are holes: repaired (R)\nor, if the seed came late, left broken (U)", fontsize=11, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16)); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "leak.png", dpi=150)
    print(json.dumps(summary, indent=2)[:1800])


if __name__ == "__main__":
    main()
