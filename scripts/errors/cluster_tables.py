#!/usr/bin/env python
"""Tables of error clusters and per-source scores for one corpus or for all corpora of a run and the references.

    python scripts/errors/cluster_tables.py --run runs/lap-01        # the run's corpora + the references
    python scripts/errors/cluster_tables.py --corpus runs/error-study/ffbs.npz

For `<dir>/<tag>.npz` it writes `<dir>/results/clusters/<tag>.csv` (one row per cluster: source, span, counts,
kind, pattern, noise level, step of the first deletion in it) and `<dir>/results/clusters/<tag>-sources.csv`
(one row per source: letters, label-level and string-level counts). The notebook `notebooks/error-clusters.ipynb`
reads these tables. Existing tables are kept unless `--force`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows.errors import Sources, canonical_counts, clusters  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

STUDY = ROOT / "runs" / "error-study"
REFERENCES = ["g", "ffbs", "i2-100", "i3-100", "i2-50", "i2-200", "i2-400", "i2-leak0.0003", "i2-leak0.001", "i2-leak0.003"]
MODEL = ["m100", "m50", "m200", "m400"]


def tables(corpus: Path, sources: Sources, eligible: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    z = np.load(corpus)
    first = int(z["first"])
    del_step = z["del_step"]
    rho = sources.rho
    cluster_rows, source_rows = [], []
    for i in range(first):
        span = sources.span(i)
        letters, inserted, steps = sources.seq(i), sources.mask(i), del_step[span]
        deleted = steps >= 0
        c = clusters(letters, inserted, deleted, eligible)
        if len(c):
            c.insert(0, "source", i)
            c["rho"] = rho[i]
            # the first deletion inside the cluster: its step, and whether it removed an original
            first_step, first_wrong, n_steps = [], [], []
            for start, end in zip(c.start, c.end):
                inside = np.arange(start, end + 1)
                gone = inside[deleted[inside]]
                if len(gone):
                    k = gone[np.argmin(steps[gone])]
                    first_step.append(int(steps[k])); first_wrong.append(bool(~inserted[k]))
                    n_steps.append(int(len(np.unique(steps[gone]))))
                else:
                    first_step.append(-1); first_wrong.append(False); n_steps.append(0)
            c["first_step"], c["first_is_wrong"], c["distinct_steps"] = first_step, first_wrong, n_steps
            cluster_rows.append(c)
        counts = canonical_counts(letters, inserted, deleted)
        source_rows.append({"source": i, "letters": len(letters), "rho": rho[i], "inserted": counts["inserted"],
                            "deleted": counts["deleted"], "correct": int((deleted & inserted).sum()),
                            "wrong": int((deleted & ~inserted).sum()), "missed": int((~deleted & inserted).sum()),
                            "string_wrong": counts["wrong"], "string_missed": counts["missed"], "equal": counts["equal"]})
    return pd.concat(cluster_rows, ignore_index=True), pd.DataFrame(source_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, help="run directory: its corpora and the references")
    p.add_argument("--corpus", type=Path, action="append", default=[], help="a single corpus file")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    corpora = list(args.corpus)
    if args.run:
        corpora += [args.run / "errors" / f"{tag}.npz" for tag in MODEL] + [STUDY / f"{tag}.npz" for tag in REFERENCES]
    sources = Sources.load(STUDY / "sources.npz")
    eligible = transition_table(np.random.default_rng(0), 20, 5, 1.0) > 0
    for corpus in corpora:
        if not corpus.exists():
            print(f"missing: {corpus}")
            continue
        out = corpus.parent / "results" / "clusters"
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{corpus.stem}.csv"
        if target.exists() and not args.force and target.stat().st_mtime > corpus.stat().st_mtime:
            continue
        c, s = tables(corpus, sources, eligible)
        c.to_csv(target, index=False)
        s.to_csv(out / f"{corpus.stem}-sources.csv", index=False)
        n = s.letters.sum()
        per_kind = (c.groupby("kind")[["n_x", "n_miss"]].sum().sum(axis=1) / n * 1000).round(2).to_dict()
        print(f"{corpus}: {len(s)} sources, {len(c)} clusters; error letters per 1000 letters by kind {per_kind}")


if __name__ == "__main__":
    main()
