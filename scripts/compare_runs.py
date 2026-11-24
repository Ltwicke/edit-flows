#!/usr/bin/env python
"""Compare runs: one table (markdown) and one figure of the sampled deletion quality per epoch.

Reads, per run directory, `eval/grammar.json` (written by `scripts/eval_grammar.py`), the per-epoch table
`eval/deletion_metrics.csv` and the command-line overrides hydra recorded.

    python scripts/compare_runs.py runs/a runs/b ... [--out figures/search.png] [--table docs/table.md]

A run can carry its legend label: `"runs/a=cubic schedule"`.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# Okabe & Ito's colour-blind safe palette, assigned in a fixed order
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7F7F7F"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Overrides every run of a search shares: they do not distinguish arms
PROTOCOL = ("experiment=", "name=", "from_ckpt=", "task.coupling.original_fraction=", "trainer.max_epochs=",
            "+trainer.limit_val_batches=", "run_dir=")


def describe(run: Path) -> str:
    """What distinguishes the run: its overrides without the ones every run shares."""
    path = run / "hydra" / ".hydra" / "overrides.yaml"
    if not path.is_file():
        return ""
    overrides = [str(o) for o in OmegaConf.load(path) if not str(o).startswith(PROTOCOL)]
    return " ".join(o.replace("task.", "", 1) if o.startswith("task.") else o for o in overrides) or "(unchanged)"


def load(run: Path, label: str = "") -> dict:
    row = {"run": run.name, "label": label or run.name, "what": describe(run)}
    table = run / "eval" / "deletion_metrics.csv"
    if table.is_file():
        t = pd.read_csv(table)
        row["epochs"] = t[(t.stage == "val") & (t.target_len == "all")].sort_values("epoch")
    grammar = run / "eval" / "grammar.json"
    if grammar.is_file():
        row["grammar"] = json.loads(grammar.read_text(encoding="utf-8"))
    config = run / "config.yaml"
    if config.is_file():
        c = OmegaConf.load(config)
        row["per_step"] = c.data.batch_size * c.get("gpus", 1) * c.trainer.get("accumulate_grad_batches", 1)
    return row


def markdown(rows: list[dict]) -> str:
    head = ["run", "what differs", "epoch", "distance covered", "AUROC x_t", "AUROC t=0", "rules >= 0.9",
            "precision / recall / F1", "len_err", "gradient norm (median / max)", "clipped", "spikes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    ideal = None
    for r in rows:
        g = r.get("grammar")
        if g is None:
            lines.append(f"| {r['run']} | {r['what']} |" + " |" * (len(head) - 2))
            continue
        L, R, PR, S, T = (g[k] for k in ("loss", "ranking_t0", "probe", "sampled", "training"))
        ideal = ideal or (L, R, PR)
        norm = f"{T['grad_norm_median']:.0f} / {T['grad_norm_max']:.0f}" if "grad_norm_median" in T else "-"
        clipped = f"{100 * T['clipped_share']:.0f} %" if "clipped_share" in T else "-"
        lines.append(
            f"| {r['run']} | {r['what']} | {g['checkpoint']['epoch']} | {100 * L['covered']:.0f} % | {L['auroc']:.3f} "
            f"| {R['auroc']:.3f} | {PR['rules_ge_0.9']} / 400 | {S['del_precision']:.2f} / {S['del_recall']:.2f} / "
            f"{S['del_f1']:.2f} | {S['len_err']:.1f} | {norm} | {clipped} | {T.get('loss_spikes', '-')} |"
        )
    if ideal is not None:
        L, R, PR = ideal
        i = R["ideal"]
        lines.append(
            f"| *exact Bayes model* | | | 100 % | {L['auroc_bayes']:.3f} | {R['auroc_bayes']:.3f} | "
            f"{PR['ideal_rules_ge_0.9']} / 400 | {i['sampling_precision']:.2f} / {i['sampling_recall']:.2f} / "
            f"{i['sampling_f1']:.2f} | | | | |"
        )
    return "\n".join(lines)


def figure(rows: list[dict], out: Path):
    rows = [r for r in rows if "epochs" in r][: len(PALETTE)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
    panels = (("del_f1", "sampled deletions: F1"), ("rank_p_at_k", "ranking at t = 0: precision in the top k"))
    for ax, (metric, title) in zip(axes, panels):
        for i, r in enumerate(rows):
            e = r["epochs"]
            x = e.step * r.get("per_step", 32) / 1e3
            ax.plot(x, e[metric], color=PALETTE[i], marker=MARKERS[i], ms=5, lw=2, label=r["label"])
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("training sequences seen (thousands)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    reference = next((r["grammar"]["ranking_t0"] for r in rows if "grammar" in r), None)
    if reference is not None:
        ideal = reference["ideal"]
        axes[0].axhline(ideal["sampling_f1"], color="#7F7F7F", lw=1.5, ls="--")
        axes[0].annotate("exact Bayes model", (0.01, ideal["sampling_f1"]), xycoords=("axes fraction", "data"),
                         xytext=(0, 4), textcoords="offset points", fontsize=8, color="#555555")
        axes[1].axhline(reference["rank_p_at_k_ideal"], color="#7F7F7F", lw=1.5, ls="--")
        axes[1].axhline(reference["rank_p_at_k_random"], color="#7F7F7F", lw=1.5, ls=":")
        axes[1].annotate("exact Bayes model", (0.01, reference["rank_p_at_k_ideal"]), xycoords=("axes fraction", "data"),
                         xytext=(0, 4), textcoords="offset points", fontsize=8, color="#555555")
        axes[1].annotate("random ranking", (0.99, reference["rank_p_at_k_random"]), xycoords=("axes fraction", "data"),
                         xytext=(0, 4), textcoords="offset points", fontsize=8, color="#555555", ha="right")
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories, optionally as <directory>=<legend label>")
    p.add_argument("--out", type=Path, help="write the figure here (.png)")
    p.add_argument("--table", type=Path, help="write the markdown table here as well")
    args = p.parse_args()

    named = [run.partition("=") for run in args.runs]
    rows = [load(Path(run), label) for run, _, label in named if Path(run).is_dir()]
    table = markdown(rows)
    print(table)
    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(table + "\n", encoding="utf-8")
    if args.out is not None:
        figure(rows, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
