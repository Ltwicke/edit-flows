#!/usr/bin/env python
"""The headline numbers of `docs/ERROR-CLUSTERS.md` for one checkpoint, derived from the saved corpora.

    python scripts/errors/cluster_tables.py --run runs/lap-01      # once, the cluster tables
    python scripts/errors/headline.py --run runs/lap-01            # -> runs/lap-01/errors/results/headline.json

Everything here is computed from the raw corpora and the cluster tables with the primitives of
`editflows/errors.py`, independently of the analysis scripts `patterns.py`, `references.py`, `dependence.py`,
`interventions.py` and `solutions.py`; where those report the same quantity the two must agree.
Unit: wrongly labelled letters (wrong deletions + missed insertions) per 1 000 source letters, with 95 %
bootstrap intervals over sources.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from editflows.errors import Sources  # noqa: E402

STUDY = ROOT / "runs" / "error-study"
KINDS = list("ZVRU")
KAPPA_1 = 0.9991  # share of the insertions that the path removes by t = 1


def interval(numerator, denominator, draws):
    sums_n, sums_d = numerator[draws].sum(1), denominator[draws].sum(1)
    lo, hi = np.quantile(sums_n / sums_d, [0.025, 0.975])
    return [float(numerator.sum() / denominator.sum()), float(lo), float(hi)]


def per_source(corpus: Path, n: int | None = None) -> pd.DataFrame:
    """Per source: letters, label and string counts, error letters by cluster kind."""
    stem = corpus.parent / "results" / "clusters" / corpus.stem
    clusters, sources = pd.read_csv(f"{stem}.csv"), pd.read_csv(f"{stem}-sources.csv")
    n = len(sources) if n is None else n
    clusters = clusters[clusters.source < n]
    size = clusters.n_x + clusters.n_miss
    kinds = size.groupby([clusters.source, clusters.kind]).sum().unstack(fill_value=0).reindex(index=range(n), columns=KINDS, fill_value=0)
    return pd.concat([sources.iloc[:n].reset_index(drop=True), kinds.reset_index(drop=True)], axis=1)


def scores(t: pd.DataFrame, draws) -> dict:
    letters = t.letters.values.astype(float)
    out = {"sources": len(t), "errors": [1000 * v for v in interval((t.wrong + t.missed).values.astype(float), letters, draws)],
           "wrong": 1000 * t.wrong.sum() / letters.sum(), "missed": 1000 * t.missed.sum() / letters.sum(),
           "string_errors": 1000 * (t.string_wrong + t.string_missed).sum() / letters.sum(),
           "exact_match": float(t.equal.mean())}
    for kind in KINDS:
        out[kind] = [1000 * v for v in interval(t[kind].values.astype(float), letters, draws)]
    for level, wrong, missed in (("label", t.wrong.sum(), t.missed.sum()), ("string", t.string_wrong.sum(), t.string_missed.sum())):
        precision, recall = (t.deleted.sum() - wrong) / t.deleted.sum(), (t.inserted.sum() - missed) / t.inserted.sum()
        out[f"{level}_precision"], out[f"{level}_recall"] = float(precision), float(recall)
        out[f"{level}_f1"] = float(2 * precision * recall / (precision + recall))
    return out


def difference(a: pd.DataFrame, b: pd.DataFrame, draws) -> dict:
    """a - b on the same sources (paired), total and by kind."""
    letters = a.letters.values.astype(float)
    out = {"sources": len(a), "errors": [1000 * v for v in interval(((a.wrong + a.missed) - (b.wrong + b.missed)).values.astype(float), letters, draws)]}
    for kind in KINDS:
        out[kind] = [1000 * v for v in interval((a[kind] - b[kind]).values.astype(float), letters, draws)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=Path("runs/lap-01"))
    args = p.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    rng = np.random.default_rng(0)
    sources = Sources.load(STUDY / "sources.npz")
    inserted = sources.inserted
    posterior = np.load(STUDY / "posterior.npz")
    q = posterior["q_gap"].astype(np.float64)
    q_simple = posterior["q_true"].astype(np.float64)
    out: dict = {"run": run.name, "letters": int(len(inserted)), "inserted_share": float(inserted.mean())}

    # ---- the floor, from the exact posterior alone
    logloss = lambda v: float(-np.where(inserted, np.log(np.clip(v, 1e-7, 1)), np.log(np.clip(1 - v, 1e-7, 1))).mean())  # noqa: E731
    best = q > 0.5
    out["floor"] = {
        "law_2q(1-q)": float(2000 * (q * (1 - q)).mean()), "decision_min(q,1-q)": float(1000 * np.minimum(q, 1 - q).mean()),
        "sampler_f1": float((q**2).sum() / q.sum()), "decision_f1": float(2 * (best & inserted).sum() / (best.sum() + inserted.sum())),
        "log_loss": {"q_gap": logloss(q), "q_true (independent marks)": logloss(q_simple)},
        "law_simple_prior": float(2000 * (q_simple * (1 - q_simple)).mean()),
    }

    # ---- the ladder of samplers
    corpora = {"perfect sampler": STUDY / "g.npz", "perfect sampler, simpler prior": STUDY / "ffbs.npz",
               "exact rates, 100 steps": STUDY / "i2-100.npz", "exact rates, noise level estimated": STUDY / "i3-100.npz",
               "trained model": run / "errors" / "m100.npz"}
    tables = {name: per_source(path) for name, path in corpora.items() if (path.parent / "results" / "clusters" / f"{path.stem}.csv").exists()}
    out["samplers"] = {}
    for name, t in tables.items():
        out["samplers"][name] = scores(t, rng.integers(0, len(t), size=(1000, len(t))))
    steps = [("trained model", "perfect sampler"), ("perfect sampler, simpler prior", "perfect sampler"),
             ("exact rates, 100 steps", "perfect sampler, simpler prior"), ("exact rates, noise level estimated", "exact rates, 100 steps"),
             ("trained model", "exact rates, noise level estimated")]
    out["steps"] = {}
    for a, b in steps:
        if a in tables and b in tables:
            n = min(len(tables[a]), len(tables[b]))
            out["steps"][f"{a} - {b}"] = difference(tables[a].iloc[:n], tables[b].iloc[:n], rng.integers(0, n, size=(1000, n)))
    m, g = out["samplers"]["trained model"], out["samplers"]["perfect sampler"]
    out["share_of_model_errors_a_perfect_sampler_makes"] = {"label": g["errors"][0] / m["errors"][0], "string": g["string_errors"] / m["string_errors"]}

    # ---- the trained model against the law, and its first reading
    z = np.load(run / "errors" / "m100.npz")
    n = len(z["del_step"])
    deleted, q0, steps_ = z["del_step"] >= 0, z["q0"].astype(np.float64), z["del_step"]
    ins, qq = inserted[:n], q[:n]
    original = ~ins
    sure = original & (qq < 1e-4)
    perfect = np.load(STUDY / "g.npz")["del_step"][:n] >= 0
    out["model"] = {
        "epoch": int(z["epoch"]) + 1, "sources": int(z["first"]),
        "certain_originals_deleted": {"model": float(deleted[sure].mean()), "perfect sampler": float(perfect[sure].mean()), "share_of_originals": float(sure.sum() / original.sum())},
        "obvious_insertions_kept": {"model": float((~deleted[ins & (qq > 0.999)]).mean()), "perfect sampler": float((~perfect[ins & (qq > 0.999)]).mean())},
        "wrong_deletions_per_1000": {"exact posterior asks for": float(1000 * qq[original].sum() / n),
                                     "model's first reading asks for": float(1000 * KAPPA_1 * q0[original].sum() / n),
                                     "observed": float(1000 * (deleted & original).sum() / n)},
        "missed_per_1000": {"exact posterior leaves": float(1000 * (1 - qq[ins]).sum() / n),
                            "model's first reading leaves": float(1000 * (1 - KAPPA_1 * q0[ins]).sum() / n),
                            "observed": float(1000 * (~deleted & ins).sum() / n)},
        "median_step": {"correct deletions": float(np.median(steps_[deleted & ins])),
                        "wrong, first reading >= 0.2": float(np.median(steps_[deleted & original & (q0 >= 0.2)])),
                        "wrong, first reading < 0.01": float(np.median(steps_[deleted & original & (q0 < 0.01)]))},
        "wrong_deletions_by_first_reading": {f"q0 < {t:g}": float((q0[deleted & original] < t).mean()) for t in (0.01, 0.05, 0.2)},
        "log_loss_first_reading": float(-np.where(ins, np.log(np.clip(q0, 1e-7, 1)), np.log(np.clip(1 - q0, 1e-7, 1))).mean()),
    }
    decision = q0 > 0.5
    tp = (decision & ins).sum()
    out["model"]["one_pass_decision_label"] = {"precision": float(tp / decision.sum()), "recall": float(tp / ins.sum()), "f1": float(2 * tp / (decision.sum() + ins.sum()))}

    # ---- step sweeps, the grammar and the leak experiment (if their tables exist)
    sweep = {}
    for family, folder, tags in (("trained model", run / "errors", ("m50", "m100", "m200", "m400")),
                                 ("exact rates", STUDY, ("i2-50", "i2-100", "i2-200", "i2-400"))):
        present = [t for t in tags if (folder / "results" / "clusters" / f"{t}.csv").exists()]
        if present:
            k = min(len(pd.read_csv(folder / "results" / "clusters" / f"{t}-sources.csv")) for t in present)
            draws = rng.integers(0, k, size=(1000, k))
            sweep[family] = {t: {key: v for key, v in scores(per_source(folder / f"{t}.npz", k), draws).items() if key in ("sources", "errors", "R", "U", "label_precision", "label_recall")} for t in present}
    out["step_sweep"] = sweep
    for name, path in (("grammar", STUDY / "results" / "repair-law" / "summary.json"), ("leak", STUDY / "results" / "leak" / "summary.json")):
        if path.exists():
            s = json.loads(path.read_text())
            out[name] = ({k: s[k] for k in ("sites", "break_size_share", "mean_hole", "median_hole", "mean_hole_contiguous", "repair_is_contiguous")} if name == "grammar"
                         else {"wrong_deletions_per_seed": s["wrong_deletions_per_seed_pooled"], "by_leak": [{k: r[k] for k in ("leak", "seeds_per_1000_letters", "extra_wrong_per_1000_letters", "wrong_per_seed", "wrong_per_seed_lo", "wrong_per_seed_hi", "R_per_1000_letters", "U_per_1000_letters")} for r in s["by_leak"]]})

    target = run / "errors" / "results" / "headline.json"
    target.write_text(json.dumps(out, indent=1))

    r3 = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"  # noqa: E731
    print(f"== {run.name}: epoch {out['model']['epoch']} of its run, {out['model']['sources']} sources")
    f = out["floor"]
    print(f"floor: law {f['law_2q(1-q)']:.1f}, single decision {f['decision_min(q,1-q)']:.1f} per 1000; best F1 sampler {f['sampler_f1']:.3f}, decision {f['decision_f1']:.3f}")
    print(f"{'sampler':36s} {'n':>5} {'errors':>18}  {'Z':>5} {'V':>5} {'R':>5} {'U':>5}   label P/R/F1          string F1  exact")
    for name, s in out["samplers"].items():
        print(f"{name:36s} {s['sources']:>5} {r3(s['errors']):>18}  {s['Z'][0]:5.1f} {s['V'][0]:5.1f} {s['R'][0]:5.1f} {s['U'][0]:5.1f}   "
              f"{s['label_precision']:.3f}/{s['label_recall']:.3f}/{s['label_f1']:.3f}   {s['string_f1']:.3f}      {s['exact_match']:.3f}")
    for name, s in out["steps"].items():
        print(f"  step {name} (n = {s['sources']}): {r3(s['errors'])};  R {r3(s['R'])}  U {r3(s['U'])}  Z {s['Z'][0]:+.1f}  V {s['V'][0]:+.1f}")
    mm = out["model"]
    print(f"model: certain originals deleted {100 * mm['certain_originals_deleted']['model']:.2f} % (perfect {100 * mm['certain_originals_deleted']['perfect sampler']:.3f} %);"
          f" obvious insertions kept {100 * mm['obvious_insertions_kept']['model']:.2f} %")
    print("  wrong deletions per 1000:", {k: round(v, 1) for k, v in mm["wrong_deletions_per_1000"].items()})
    print("  missed per 1000:", {k: round(v, 1) for k, v in mm["missed_per_1000"].items()})
    print("  median step:", mm["median_step"], " one-pass decision:", {k: round(v, 3) for k, v in mm["one_pass_decision_label"].items()})
    for family, s in out["step_sweep"].items():
        print(f"  steps, {family}: " + "; ".join(f"{t}: {v['errors'][0]:.1f} (R {v['R'][0]:.1f}, U {v['U'][0]:.1f})" for t, v in s.items()))
    print(f"-> {target}")


if __name__ == "__main__":
    main()
