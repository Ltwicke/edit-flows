"""The exact references for the synthetic targets (editflows/grammar.py)."""

import numpy as np

from editflows.grammar import (
    auroc,
    brute_force_posterior,
    deletion_scores,
    ideal_flow,
    ideal_scores,
    insertion_posterior,
    ranking_scores,
    rule_auroc,
    rule_probes,
)


def table(n_letters=5, n_allowed=2, seed=0):
    """A sparse order-2 transition table like the one of scripts/make_synthetic_targets.py."""
    rng = np.random.default_rng(seed)
    P = np.zeros((n_letters,) * 3)
    for a in range(n_letters):
        for b in range(n_letters):
            P[a, b, rng.choice(n_letters, size=n_allowed, replace=False)] = rng.dirichlet(np.ones(n_allowed))
    return P


def sample(P, length, rng):
    n = P.shape[0]
    seq = list(rng.integers(0, n, size=2))
    while len(seq) < length:
        seq.append(rng.choice(n, p=P[seq[-2], seq[-1]]))
    return np.array(seq)


def with_insertions(target, k, n_letters, rng):
    source = list(target)
    for position in sorted(rng.integers(0, len(target) + 1, size=k), reverse=True):
        source.insert(position, rng.integers(0, n_letters))
    return np.array(source)


def test_posterior_equals_brute_force():
    P, rng = table(), np.random.default_rng(1)
    for k in (0, 1, 3):
        source = with_insertions(sample(P, 7, rng), k, P.shape[0], rng)
        rho = max(k, 1) / len(source)
        exact = insertion_posterior([source], rho, P)[0]
        assert np.allclose(exact, brute_force_posterior(source, rho, P), atol=1e-12)


def test_posterior_of_a_batch_equals_the_single_sequences():
    P, rng = table(), np.random.default_rng(2)
    sources = [with_insertions(sample(P, n, rng), 2, P.shape[0], rng) for n in (5, 9, 14, 9)]
    rhos = [0.1, 0.3, 0.2, 0.25]
    together = insertion_posterior(sources, rhos, P, budget=300)  # tiny budget: several chunks, padded rows
    for source, rho, q in zip(sources, rhos, together):
        assert np.allclose(q, insertion_posterior([source], rho, P)[0], atol=1e-12)


def test_auroc_and_ranking_scores():
    assert auroc([0.1, 0.2, 0.9], [False, False, True]) == 1.0
    assert auroc([0.9, 0.2, 0.1], [False, False, True]) == 0.0
    assert auroc([0.5, 0.5], [False, True]) == 0.5
    assert np.isnan(auroc([0.1, 0.2], [True, True]))

    masks = [np.array([True, False, False, True]), np.array([False, True, False, False, False, False])]
    perfect = ranking_scores([m.astype(float) for m in masks], masks)
    assert perfect["rank_p_at_k"] == 1.0
    # A random ranking hits k * k / n insertions in a sequence with k of n tokens inserted
    assert np.isclose(perfect["rank_p_at_k_random"], (2 * 2 / 4 + 1 * 1 / 6) / 3)


def test_ideal_scores_of_a_perfect_posterior():
    masks = [np.array([True, False, False, True]), np.array([False, True, False])]
    scores = ideal_scores(masks, [m.astype(float) for m in masks])
    for key in ("top_k_precision", "sampling_precision", "sampling_recall", "map_f1", "auroc"):
        assert np.isclose(scores[key], 1.0), key
    assert np.isclose(scores["chance"], 3 / 7)


def test_rule_probes_cover_every_rule():
    n, left, repeats = 4, 6, 3
    P, rng = table(n_letters=n, n_allowed=3, seed=3), np.random.default_rng(4)  # 3 of 4 letters: every pair occurs
    seqs = [sample(P, 400, rng) for _ in range(12)]
    tokens = np.concatenate(seqs).astype(np.uint8)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in seqs])])
    probes, context, letter = rule_probes(tokens, offsets, np.arange(12), n_letters=n, repeats=repeats, left=left,
                                          subsample=1.0)
    assert probes.shape == (n * n * n * repeats, left + 1)
    assert (probes[:, left] == letter).all()
    assert (probes[:, left - 2] * n + probes[:, left - 1] == context).all()
    # The prefix is a piece of a target: all its windows are allowed
    assert (P[probes[:, :-3], probes[:, 1:-2], probes[:, 2:-1]] > 0).all()

    # A scorer that knows the table separates eligible from ineligible probe letters in every context
    eligible = P.reshape(n * n, n)[context, letter] > 0
    assert (rule_auroc((~eligible).astype(float), context, eligible, n_contexts=n * n) == 1.0).all()


def test_ideal_flow_deletes_what_the_table_reveals():
    P, rng = table(n_letters=6, n_allowed=2, seed=5), np.random.default_rng(6)
    targets = [sample(P, 60, rng) for _ in range(6)]
    sources, masks = [], []
    for target in targets:
        mask = np.zeros(70, dtype=bool)
        mask[rng.choice(70, size=10, replace=False)] = True
        source = np.empty(70, dtype=np.int64)
        source[~mask], source[mask] = target, rng.integers(0, 6, size=10)
        sources.append(source)
        masks.append(mask)
    deleted = ideal_flow(sources, masks, P, hazard=lambda t: 6.0, n_steps=60)
    scores = deletion_scores(deleted, masks)
    # A sparse table (2 of 6 letters allowed) exposes most insertions, and a random deletion would reach 1/7
    assert scores["precision"] > 0.6 and scores["recall"] > 0.6
    assert deletion_scores(masks, masks) == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "len_err": 0.0, "exact_match": 1.0}
