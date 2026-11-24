#!/usr/bin/env python
"""The key numbers of the error study for several checkpoints side by side (a markdown table).

    python scripts/errors/compare_checkpoints.py runs/lap-01 runs/lap-long

Reads what the other scripts wrote for each run (`errors/results/headline.json` and the `summary.json` /
tables of patterns, interventions, dependence, sceptic, solutions); cells whose source is missing show a dash.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def ci(v, digits=1):
    return f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def rows_for(run: Path) -> dict:
    res = run / "errors" / "results"
    out: dict = {}

    def put(name, fn):
        try:
            out[name] = fn()
        except Exception:  # a missing file or key: the cell stays empty
            out[name] = "—"

    h = load(res / "headline.json")
    put("checkpoint: best epoch of its run", lambda: str(h["model"]["epoch"]))
    m = (h or {}).get("samplers", {}).get("trained model", {})
    put("C  wrongly labelled letters per 1 000", lambda: ci(m["errors"]))
    for kind, label in zip("ZVRU", ("Z same string, other label", "V other valid reading", "R repaired hole", "U left broken")):
        put(f"C  … of kind {label}", lambda kind=kind: f"{m[kind][0]:.1f}")
    put("C  surplus over the perfect sampler", lambda: ci(h["steps"]["trained model - perfect sampler"]["errors"]))
    put("C  share of the model's label errors a perfect sampler makes too", lambda: f"{100 * h['share_of_model_errors_a_perfect_sampler_makes']['label']:.0f} %")
    put("C  label precision / recall / F1", lambda: f"{m['label_precision']:.3f} / {m['label_recall']:.3f} / {m['label_f1']:.3f}")
    put("C  string F1; outputs exactly right", lambda: f"{m['string_f1']:.3f}; {100 * m['exact_match']:.1f} %")
    put("C  wrong deletions per 1 000: first reading asks for / observed (perfect sampler 21.1)",
        lambda: f"{h['model']['wrong_deletions_per_1000']['model\'s first reading asks for']:.1f} / {h['model']['wrong_deletions_per_1000']['observed']:.1f}")
    put("C  missed insertions per 1 000: first reading leaves / observed (perfect sampler 21.1)",
        lambda: f"{h['model']['missed_per_1000']['model\'s first reading leaves']:.1f} / {h['model']['missed_per_1000']['observed']:.1f}")
    put("C  originals with exact probability below 0.01 % that are deleted", lambda: f"{100 * h['model']['certain_originals_deleted']['model']:.2f} %")

    iv = load(res / "interventions" / "summary.json")

    def cell(t0, b, key, digits=2):
        v = iv["headline"][t0][f"break_size_{b}"][key]
        return f"{v[0]:.{digits}f}"

    put("A  forced removal at t0 = 0: further wrong deletions, break size 0 / 1 / 2",
        lambda: " / ".join(cell("t0 = 0", b, "further_wrong_deletions_model") for b in (0, 1, 2)))
    put("A  … with exact rates, break size 1 / 2", lambda: " / ".join(cell("t0 = 0", b, "further_wrong_deletions_exact_rates") for b in (1, 2)))
    put("A  forced removal at t0 = 0.6: further wrong deletions, break size 0 / 1 / 2",
        lambda: " / ".join(cell("t0 = 0.6", b, "further_wrong_deletions_model") for b in (0, 1, 2)))
    put("A  forced removal at t0 = 0.9: further wrong deletions, break size 0 / 1 / 2",
        lambda: " / ".join(cell("t0 = 0.9", b, "further_wrong_deletions_model") for b in (0, 1, 2)))
    put("A  one forward pass at t0 = 0: summed rise of the neighbours' probability, break size 0 / 1 / 2 (ideal 0.0 / 3.2 / 3.6)",
        lambda: " / ".join(cell("t0 = 0", b, "suspicion_model") for b in (0, 1, 2)))
    put("A  neighbourhood valid at the end after a removal at t0 = 0, break size 1 / 2 (exact rates 100 % / 99 %)",
        lambda: " / ".join(f"{100 * iv['headline']['t0 = 0'][f'break_size_{b}']['grammatical_at_end_model'][0]:.0f} %" for b in (1, 2)))
    put("A  an insertion forced to stay: further wrong deletions (natural mix of insertions)",
        lambda: ci(iv["headline"]["mirror_further_wrong_deletions"]["all insertions"], 2))

    d = load(res / "dependence" / "summary.json")

    def first(prefix):
        return next(v for k, v in d["1 pairs"].items() if k.startswith(prefix))

    put("A  odds ratio, two adjacent originals wrongly deleted in one run (perfect sampler 2.8, exact rates 4.8)",
        lambda: ci(first("OR two originals by target distance, M, all")["1"], 2))
    put("A  ordinary runs: rise of the neighbour's probability after a wrong deletion, break size 0 / 1 / 2",
        lambda: " / ".join(f"{d['2 events'][f'first wrong deletion, break {b}']['change of q vs matched controls, original neighbours, next step, distance 1..6'][0][0]:+.3f}" for b in (0, 1, 2)))
    put("A  wrong deletions that hit a letter suspected in the first reading / suspicious later / never",
        lambda: " / ".join(f"{100 * d['3 origin']['classes'][k][0]:.0f} %" for k in ("a ambiguous from the start", "b became suspicious during sampling", "c deleted although not suspicious")))

    s = load(res / "sceptic" / "summary.json")
    put("A  surplus of wrong deletions over the perfect sampler: already in the first reading / added by sampling",
        lambda: " / ".join(f"{v[0]:.0f} %" for v in s["attack_c"]["surplus of wrong deletions over the perfect sampler: % already in the first reading, % added by sampling"]))
    put("A  wrong deletions that sampling adds to the first reading, per 1 000",
        lambda: ci(s["attack_c"]["sampling adds wrong deletions to the first reading, per 1000"]))
    put("A  places with a wrong deletion per 1 000 letters; wrong deletions per place (perfect sampler 12.1; 1.75)",
        lambda: (lambda n: f"{n['neighbourhoods with a wrong deletion per 1000'][0]:.1f}; {1 + n['further wrong deletions per neighbourhood'][0]:.2f}")(s["attack_c"]["neighbourhoods (wrong deletions less than 13 originals apart)"]["the sampler (m100)"]))
    put("A  wrong deletions in runs of 4+ originals per 1 000: first reading alone / sampler",
        lambda: (lambda w: f"{w['one independent draw from the first reading']['wrong deletions in runs of 4+ per 1000'][0]:.1f} / {w['the sampler (m100)']['wrong deletions in runs of 4+ per 1000'][0]:.1f}")(s["attack_c"]["what the first reading alone would give"]))

    cat = res / "patterns" / "categories.csv"

    def category(code, corpus="m100"):
        t = pd.read_csv(cat)
        r = t[(t.noise == "all") & (t.category == code) & (t.corpus == corpus)].iloc[0]
        return r.clusters_per_1000_letters, r.clusters_lo, r.clusters_hi

    put("B  hole runs per 1 000 letters (perfect sampler 0.50, exact rates 1.40)", lambda: ci(category("HH"), 2))
    put("B  twin swaps (same string) per 1 000 letters (perfect sampler 9.87)", lambda: f"{category('S=')[0]:.2f}")
    put("B  lone `x` / lone `_` per 1 000 letters (perfect sampler 2.16 / 2.18)", lambda: f"{category('X')[0]:.2f} / {category('L')[0]:.2f}")

    def variant(tag, short=None):
        t = pd.read_csv(res / "solutions" / "variants_heldout.csv")
        t = t[t.tag == tag] if short is None else t[t.short == short]
        return t.iloc[0]

    put("S  held-out sources: string F1, standard sampler", lambda: f"{variant('baseline').f1_string:.3f}")
    put("S  … anchoring 0.2 (gain)", lambda: (lambda r: f"{r.f1_string:.3f} ({r.minus_standard_f1_string:+.4f} [{r.minus_standard_f1_string_lo:+.4f}, {r.minus_standard_f1_string_hi:+.4f}])")(variant("anchor0.2")))
    put("S  … veto 0.25", lambda: f"{variant(None, 'veto 0.25').f1_string:.3f}")
    put("S  … one pass, identical neighbours together", lambda: f"{variant('one-shot-runs').f1_string:.3f}")
    put("S  wrong deletions in repaired holes per 1 000: standard / anchoring 0.2",
        lambda: f"{variant('baseline').wrong_in_R_per1000:.1f} / {variant('anchor0.2').wrong_in_R_per1000:.1f}")
    so = load(res / "solutions" / "summary.json")
    put("S  bracket for the unavoidable label errors per 1 000: lower bound / truth / upper bound (first reading)",
        lambda: (lambda b: f"{b['lower_embedding'][0]:.1f} / {b['truth'][0]:.1f} / {b['upper'][0]:.1f}")(so["bracket"]["per_1000_letters_held_out"]))
    put("S  deletions beyond the first reading among letters read as original, per 1 000 (of them wrong)",
        lambda: (lambda c: f"{c['low'][0]:+.1f} ({c['low_originals'][0]:+.1f})")(so["self_consistency"]["model_one_run_exposure"]))
    return out


def main():
    runs = [Path(a) if Path(a).is_absolute() else ROOT / a for a in sys.argv[1:]] or [ROOT / "runs" / "lap-01"]
    columns = {run.name: rows_for(run) for run in runs}
    names = list(next(iter(columns.values())))
    print("| | " + " | ".join(f"`{name}`" for name in columns) + " |")
    print("|---|" + "---|" * len(columns))
    for name in names:
        print(f"| {name} | " + " | ".join(str(columns[c][name]) for c in columns) + " |")


if __name__ == "__main__":
    main()
