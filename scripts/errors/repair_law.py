#!/usr/bin/env python
"""What the grammar alone says about a removed letter: how often the gap is visible and how many further letters
must go until the sequence is grammatical again (no model involved).

    python scripts/errors/repair_law.py            # -> runs/error-study/results/repair-law/

For random letters of the held-out targets: the break size (0, 1 or 2 forbidden windows across the gap) and the
minimal repair (fewest further deletions within +-W letters that leave no forbidden window). Compared with two
closed forms for the size of the hole, seed included:

* junction repair: a contiguous hole of length l containing the seed has l placements, each grammatical with
  probability v^2, so P(size > L) = (1 - v^2)^(L (L + 1) / 2);
* one-sided deletion (delete to the right until it fits): P(size > L) = (1 - v^2)^L.
"""

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import Sources, break_size, minimal_repair  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#7F7F7F"


def contiguous_repair(seq: np.ndarray, gap: int, eligible: np.ndarray, limit: int) -> int:
    """Fewest further deletions if only a contiguous stretch around the gap may go (`gap` = index of the first
    letter after the gap); -1 if none of at most `limit` letters works."""
    for extra in range(limit + 1):
        for left in range(extra + 1):  # letters removed before the gap; extra - left after it
            a, b = gap - left, gap + extra - left
            l2, l1, r1, r2 = seq[a - 2], seq[a - 1], seq[b], seq[b + 1]
            if eligible[l2, l1, r1] and eligible[l1, r1, r2]:
                return extra
    return -1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", type=int, default=20, help="removed letters per target")
    p.add_argument("--window", type=int, default=12, help="letters on each side that a repair may use")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = ROOT / "runs" / "error-study" / "results" / "repair-law"
    out.mkdir(parents=True, exist_ok=True)

    eligible = transition_table(np.random.default_rng(0), 20, 5, 1.0) > 0
    v = float(eligible.mean())
    sources = Sources.load(ROOT / "runs" / "error-study" / "sources.npz")
    rng = np.random.default_rng(args.seed)
    W = args.window
    rows = []
    for i in range(len(sources)):
        target = sources.seq(i)[~sources.mask(i)].astype(np.int64)
        if len(target) < 2 * W + 8:
            continue
        sites = rng.choice(np.arange(W + 2, len(target) - W - 3), size=args.sites, replace=False)
        sizes = break_size(target, sites, eligible)
        for site, b in zip(sites, sizes):
            broken = np.delete(target, site)
            rows.append({"source": i, "site": int(site), "break_size": int(b),
                         "repair": minimal_repair(broken, site - W, site + W, eligible),
                         "contiguous": contiguous_repair(broken, site, eligible, W)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "sites.csv", index=False)

    n = len(table)
    solved = table[table.repair >= 0]
    size = solved.repair + 1  # the hole, seed included
    L = np.arange(0, 13)
    survival = np.array([(size > k).mean() for k in L])
    contiguous = table[table.contiguous >= 0].contiguous + 1
    survival_contiguous = np.array([(contiguous > k).mean() for k in L])
    junction = (1 - v**2) ** (L * (L + 1) / 2)
    one_sided = (1 - v**2) ** L
    summary = {
        "sites": n, "v": v,
        "break_size_share": {int(b): float((table.break_size == b).mean()) for b in (0, 1, 2)},
        "break_size_predicted": {0: v**2, 1: 2 * v * (1 - v), 2: (1 - v) ** 2},
        "no_repair_within_window": float((table.repair < 0).mean()),
        "mean_hole": float(size.mean()), "median_hole": float(size.median()),
        "mean_hole_contiguous": float(contiguous.mean()),
        "mean_hole_junction_law": float(sum((1 - v**2) ** (k * (k + 1) / 2) for k in range(60))),
        "mean_hole_one_sided": 1 / v**2,
        "repair_is_contiguous": float((solved.repair == solved.contiguous).mean()),
        "hole_size_share": {int(k): float((size == k).mean()) for k in range(1, 11)},
        "survival": {int(k): float(s) for k, s in zip(L, survival)},
        "survival_junction_law": {int(k): float(s) for k, s in zip(L, junction)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.bar([0, 1, 2], [summary["break_size_share"][b] for b in (0, 1, 2)], color=BLUE, width=0.6, label="held-out targets")
    ax.plot([0, 1, 2], [summary["break_size_predicted"][b] for b in (0, 1, 2)], "o", color=ORANGE, ms=9,
            label="prediction: 1/16, 6/16, 9/16")
    ax.set_xticks([0, 1, 2]); ax.set_xlabel("forbidden windows across the gap after one letter is removed")
    ax.set_ylabel("share of letters")
    ax.set_title("Removing a letter is visible in 15 of 16 cases")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.semilogy(L, survival, "o-", color=BLUE, label="smallest repair in the real grammar")
    ax.semilogy(L, survival_contiguous, "s--", color=GREEN, label="... if only a contiguous hole is allowed")
    ax.semilogy(L, junction, "-", color=ORANGE, label="junction repair: $(1-v^2)^{L(L+1)/2}$")
    ax.semilogy(L, one_sided, ":", color=GREY, label="one-sided deletion: $(1-v^2)^{L}$")
    ax.set_ylim(1e-4, 1.3); ax.set_xlabel("L = letters missing in the end, the first one included")
    ax.set_ylabel("share of removals that need more than L")
    ax.set_title(f"A removed letter costs {size.mean():.1f} letters on average before the gap fits again")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "repair-law.png", dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
