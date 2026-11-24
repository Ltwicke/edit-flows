#!/usr/bin/env python
"""The perfect sampler under the coupling's own prior, as corpora for the other scripts.

`editflows.exact` (and with it `ffbs*.npz`, `i2-*.npz`, `posterior.npz: q_true`) assumes that every source letter
is an insertion independently, which makes the number of insertions between two originals geometric. The
coupling throws its K letters into the gaps uniformly, so that number is binomial (Poisson-like): after an
insertion a further one is *less* likely. `scripts/errors/references.py` implements the posterior for this prior
(the hidden state also counts the insertions of the current gap); it predicts the labels better (log loss 0.066
against 0.074) and is calibrated. This script stores it for all sources:

    python scripts/errors/gap_reference.py [--replicates 32]

* `runs/error-study/posterior.npz` gains `q_gap` (the probability that a letter is an insertion);
* `runs/error-study/g.npz`: one exact joint sample per source, same format as `ffbs.npz`;
* `runs/error-study/g-rep<k>.npz`: replicates of the first 500 sources, same format as `ffbs-rep<k>.npz`.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "errors"))

from editflows.errors import Sources  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402
from references import _gap_both, gap_hazards  # noqa: E402

STUDY = ROOT / "runs" / "error-study"


def sample(sources: Sources, P: np.ndarray, first: int, seed: int):
    """Posterior and one joint posterior sample for the first `first` sources."""
    rng = np.random.default_rng(seed)
    n = int(sources.offsets[first])
    q = np.zeros(n)
    drawn = np.zeros(n, dtype=bool)
    for i in range(first):
        seq = sources.seq(i)
        k = int(sources.mask(i).sum())
        q[sources.span(i)], drawn[sources.span(i)] = _gap_both(seq, P, gap_hazards(k, len(seq) - k), rng.random(len(seq) + 1))
    return q, drawn


def save(tag: str, drawn: np.ndarray, first: int, seed: int, inserted: np.ndarray):
    np.savez_compressed(STUDY / f"{tag}.npz", first=first, steps=0, seed=seed, del_step=np.where(drawn, 0, -1).astype(np.int16))
    tp = int((drawn & inserted[: len(drawn)]).sum())
    print(f"{tag}: {first} sources; precision {tp / drawn.sum():.4f} recall {tp / inserted[: len(drawn)].sum():.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replicates", type=int, default=32)
    p.add_argument("--first", type=int, default=500, help="sources per replicate")
    args = p.parse_args()
    sources = Sources.load(STUDY / "sources.npz")
    P = transition_table(np.random.default_rng(0), 20, 5, 1.0)
    if not (STUDY / "g.npz").exists():
        q, drawn = sample(sources, P, len(sources), seed=0)
        posterior = dict(np.load(STUDY / "posterior.npz"))
        posterior["q_gap"] = q.astype(np.float32)
        np.savez_compressed(STUDY / "posterior.npz", **posterior)
        save("g", drawn, len(sources), 0, sources.inserted)
    for k in range(args.replicates):
        if not (STUDY / f"g-rep{k}.npz").exists():
            _, drawn = sample(sources, P, args.first, seed=300 + k)
            save(f"g-rep{k}", drawn, args.first, 300 + k, sources.inserted)


if __name__ == "__main__":
    main()
