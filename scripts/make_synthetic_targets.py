#!/usr/bin/env python
"""Write a placeholder dataset of target sequences.

The sequences come from an order-2 Markov chain over the first 20 letters of the alphabet: for every pair of
preceding letters only a few next letters are possible. This local structure is what makes randomly inserted
letters detectable. (Independent uniform letters would make the deletion task unlearnable: an inserted uniform
letter is indistinguishable from an original one.) The real target sequences of the project replace this file
without any code change; the loader only needs one sequence per row.

Usage:
    python scripts/make_synthetic_targets.py --n 20000 --out data/synthetic/targets.xlsx
"""

import argparse
from pathlib import Path

import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRST"


def transition_table(rng: np.random.Generator, n_letters: int, n_allowed: int, concentration: float):
    """P[a, b, :] is the distribution of the next letter given the two preceding letters a, b.

    Only `n_allowed` letters per context get non-zero probability.
    """
    P = np.zeros((n_letters, n_letters, n_letters))
    for a in range(n_letters):
        for b in range(n_letters):
            allowed = rng.choice(n_letters, size=n_allowed, replace=False)
            P[a, b, allowed] = rng.dirichlet(np.full(n_allowed, concentration))
    return P


def sample_sequence(rng: np.random.Generator, cumulative: np.ndarray, length: int) -> np.ndarray:
    """Sample one sequence of letter ids from the chain with cumulative transition table `cumulative`."""
    n_letters = cumulative.shape[0]
    seq = np.empty(length, dtype=np.int64)
    seq[:2] = rng.integers(0, n_letters, size=2)
    u = rng.random(length)
    for i in range(2, length):
        seq[i] = min(np.searchsorted(cumulative[seq[i - 2], seq[i - 1]], u[i]), n_letters - 1)
    return seq


def write_sequences(path: Path, sequences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        from openpyxl import Workbook

        wb = Workbook(write_only=True)
        ws = wb.create_sheet("targets")
        ws.append(["sequence"])
        for s in sequences:
            ws.append([s])
        wb.save(path)
    else:
        path.write_text("sequence\n" + "\n".join(sequences) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="number of sequences")
    p.add_argument("--min-len", type=int, default=100)
    p.add_argument("--max-len", type=int, default=500)
    p.add_argument("--n-allowed", type=int, default=5, help="possible next letters per context")
    p.add_argument("--concentration", type=float, default=1.0, help="Dirichlet concentration of the transitions")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/synthetic/targets.xlsx"), help=".xlsx, .csv or .txt")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    P = transition_table(rng, len(ALPHABET), args.n_allowed, args.concentration)
    cumulative = P.cumsum(axis=-1)
    lengths = rng.integers(args.min_len, args.max_len + 1, size=args.n)
    sequences = ["".join(ALPHABET[i] for i in sample_sequence(rng, cumulative, int(n))) for n in lengths]
    write_sequences(args.out, sequences)

    print(f"wrote {len(sequences)} sequences to {args.out}")
    print(f"length: min {lengths.min()}, mean {lengths.mean():.1f}, max {lengths.max()}")
    print(f"letters per context: {args.n_allowed} of {len(ALPHABET)} (seed {args.seed})")


if __name__ == "__main__":
    main()
