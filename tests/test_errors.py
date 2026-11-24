"""The instruments of the error analysis: verdicts, clusters, grammar annotations, exact references, the event loop."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from editflows import exact, grammar
from editflows.errors import (
    break_size,
    canonical_counts,
    cluster_ids,
    clusters,
    lcs_length,
    minimal_repair,
    model_euler,
    verdict_codes,
    verdict_string,
)
from editflows.schedules import ExponentialKappaSchedule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from make_synthetic_targets import transition_table  # noqa: E402

# A grammar with exactly one allowed letter per context: c = (a + b) mod 3, the Fibonacci numbers mod 3 (period 8)
CHAIN = np.array([[[c == (a + b) % 3 for c in range(3)] for b in range(3)] for a in range(3)])
PERIOD = [0, 1, 1, 2, 0, 2, 2, 1]
FREE = np.ones((3, 3, 3), dtype=bool)  # every letter may follow every context


def test_verdicts():
    inserted = np.array([0, 0, 1, 1], dtype=bool)
    deleted = np.array([0, 1, 0, 1], dtype=bool)
    assert verdict_string(verdict_codes(inserted, deleted)) == ".x_+"


def test_clusters_are_formed_in_output_coordinates():
    codes = verdict_codes(np.array(list("0010000001"), dtype=int).astype(bool), np.array(list("0110010000"), dtype=int).astype(bool))
    assert verdict_string(codes) == ".x+..x..._"
    # two kept originals between the first two errors, three before the last one; the correct deletion does not count
    assert cluster_ids(codes, gap=2).tolist() == [-1, 0, -1, -1, -1, 0, -1, -1, -1, 1]
    assert cluster_ids(codes, gap=1).tolist() == [-1, 0, -1, -1, -1, 1, -1, -1, -1, 2]
    assert cluster_ids(codes, gap=3).tolist() == [-1, 0, -1, -1, -1, 0, -1, -1, -1, 0]


def test_cluster_types_from_ground_truth():
    target = np.array(PERIOD + PERIOD[:4])
    # a second 1 inserted between the two 1s; deleting the original instead gives the same string: label-only
    letters = np.insert(target, 2, 1)
    inserted = np.zeros(len(letters), dtype=bool); inserted[2] = True
    deleted = np.zeros(len(letters), dtype=bool); deleted[1] = True
    row = clusters(letters, inserted, deleted, CHAIN).iloc[0]
    assert (row.kind, row.pattern, row.n_x, row.n_miss, row.violated) == ("Z", "x_", 1, 1, 0)

    # an inserted letter that stays and breaks the chain: unresolved
    letters = np.insert(target, 5, 1)
    inserted = np.zeros(len(letters), dtype=bool); inserted[5] = True
    row = clusters(letters, inserted, np.zeros(len(letters), dtype=bool), CHAIN).iloc[0]
    assert row.kind == "U" and row.pattern == "_" and row.violated >= 1

    # with a grammar that allows everything nothing is ever violated: one wrong deletion is a valid alternative,
    # two are a hole
    plain = np.zeros(len(target), dtype=bool)
    one = plain.copy(); one[4] = True
    assert clusters(target, plain, one, FREE).kind.tolist() == ["V"]
    two = plain.copy(); two[[4, 5]] = True
    row = clusters(target, plain, two, FREE).iloc[0]
    assert (row.kind, row.pattern) == ("R", "xx")
    assert len(clusters(target, plain, plain, FREE)) == 0


def test_break_size_and_minimal_repair():
    seq = np.array(PERIOD * 4)
    assert break_size(seq, np.array([12]), CHAIN).tolist() == [2]
    assert break_size(seq, np.array([12]), FREE).tolist() == [0]
    # in a chain with one allowed letter the only repair of a removal is to remove the rest of the period
    broken = np.delete(seq, 12)
    assert minimal_repair(broken, 4, 20, CHAIN) == 7
    assert minimal_repair(seq, 4, 20, CHAIN) == 0
    assert minimal_repair(broken, 10, 14, CHAIN) == -1  # too short a stretch to repair
    assert minimal_repair(broken, 4, 20, FREE) == 0


def test_string_level_comparison():
    assert lcs_length([1, 2, 3, 4], [2, 4]) == 2
    assert lcs_length([1, 2, 3], [4, 5]) == 0
    assert lcs_length([], [1]) == 0
    letters = np.array([0, 1, 1, 1, 2])
    inserted = np.array([0, 0, 1, 0, 0], dtype=bool)
    wrong_copy = np.array([0, 1, 0, 0, 0], dtype=bool)  # the label says wrong deletion + missed insertion
    assert canonical_counts(letters, inserted, wrong_copy) == {"wrong": 0, "missed": 0, "deleted": 1, "inserted": 1, "equal": 1}
    nothing = np.zeros(5, dtype=bool)
    assert canonical_counts(letters, inserted, nothing)["missed"] == 1


def test_exact_references_agree():
    rng = np.random.default_rng(0)
    P = transition_table(np.random.default_rng(0), 20, 5, 1.0)
    seqs = [rng.integers(0, 20, int(n)) for n in (5, 17, 30, 30, 41)]
    rho = [0.1, 0.3, 0.5, 0.2, 0.4]
    reference = grammar.insertion_posterior(seqs, rho, P)
    device = torch.device("cpu")
    Pt, rho_t = torch.tensor(P), torch.tensor(rho, dtype=torch.float64)
    S, valid = exact.pad(seqs, device)
    q = exact.posterior(S, valid, rho_t, Pt)
    for i, s in enumerate(seqs):
        assert np.allclose(q[i, : len(s)].numpy(), reference[i], atol=1e-10)
    assert float(q[~valid].abs().max()) == 0

    # joint samples: their mean is the posterior
    generator = torch.Generator().manual_seed(0)
    n = 600
    mean = sum(exact.posterior_sample(S, valid, rho_t, Pt, generator=generator).double() for _ in range(n)) / n
    z = (mean - q) / torch.sqrt(q * (1 - q) / n).clamp(min=1e-3)
    assert float(z[valid].abs().max()) < 5
    assert not bool(exact.posterior_sample(S, valid, rho_t, Pt, generator=generator)[~valid].any())

    # the exact-rate sampler deletes, and only real positions; the same seed gives the same result
    kappa = ExponentialKappaSchedule(3.0, normalize=False, growth=1.5)
    steps = exact.exact_euler(S, valid, rho_t, Pt, kappa, n_steps=20, seed=3)
    assert bool((steps[~valid] == -1).all()) and bool((steps[valid] >= 0).any())
    assert torch.equal(steps, exact.exact_euler(S, valid, rho_t, Pt, kappa, n_steps=20, seed=3))
    # a leak is a deliberate defect: every letter keeps at least that probability, so more letters go
    leaky = exact.exact_euler(S, valid, rho_t, Pt, kappa, n_steps=20, seed=3, leak=0.3)
    assert int((leaky >= 0).sum()) > int((steps >= 0).sum()) and bool((leaky[~valid] == -1).all())


class HotLetters:
    """A stand-in model: an overwhelming deletion rate for the letters in `hot`, none for the others."""

    def __init__(self, *hot):
        self.hot = torch.tensor(hot)

    def __call__(self, x_t, t):
        return SimpleNamespace(log_lambda_del=torch.where(torch.isin(x_t.x, self.hot), 20.0, -20.0))


def test_event_loop_records_deletions_and_interventions():
    kappa = ExponentialKappaSchedule(3.0, normalize=False, growth=1.5)
    seqs = [np.array([1, 7, 2, 7, 3]), np.array([7, 7, 4])]
    run = lambda **kw: model_euler(HotLetters(7), seqs, kappa, 20, device=torch.device("cpu"), n_steps=5, **kw)  # noqa: E731

    out = run()
    assert out["del_step"].tolist() == [-1, 0, -1, 0, -1, 0, 0, -1]
    assert np.allclose(out["q0"][[1, 3, 5, 6]], 1.0) and np.all(out["q0"][[0, 2, 4, 7]] < 1e-6)

    keep = np.zeros(8, dtype=bool); keep[1] = True
    assert run(keep=keep)["del_step"].tolist() == [-1, -1, -1, 0, -1, 0, 0, -1]

    alive = np.ones(8, dtype=bool); alive[[0, 5]] = False  # letters removed before the start
    out = run(alive=alive)
    assert out["del_step"].tolist() == [-1, 0, -1, 0, -1, -1, 0, -1] and np.isnan(out["q0"][[0, 5]]).all()

    # at most one deletion per window: the two neighbouring hot letters go in different steps
    out = run(exclusion=1)
    assert sorted(out["del_step"][[5, 6]].tolist()) == [0, 1] and out["del_step"][[1, 3]].tolist() == [0, 0]

    logged = run(log_q=True)["qlog"]
    assert logged.shape == (8, 5) and logged[1, 0] == 1 and np.isnan(logged[1, 1]) and not np.isnan(logged[0]).any()
