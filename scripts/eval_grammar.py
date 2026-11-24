#!/usr/bin/env python
"""Score a run against what is possible on the synthetic targets.

For one checkpoint of a run (default `last.ckpt`: the best epoch so far by the monitored metric), with the run's
own coupling and schedule:

* the loss of the model next to two references on the same samples - the best function of the time alone (it
  never looks at a letter) and the exact Bayes model (it knows the transition table) - and the share of the
  distance between the two that the model has covered;
* ranking quality (mean AUROC per sequence) on those x_t and, at t = 0, on the sources of the per-epoch evaluation;
* the rule probe: for each of the 400 contexts, does the model tell the 15 ineligible next letters from the 5
  eligible ones;
* the sampled deletion metrics of that epoch next to what the exact posterior achieves on the same sources;
* gradient norms and loss spikes from the training log.

    python scripts/eval_grammar.py runs/<name> [--checkpoint last|best|periodic|<file>] [-n 512]

Writes `runs/<name>/eval/grammar.json`.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from editflows import DataBatch, EditFlow  # noqa: E402
from editflows.coupling import KnownAligner  # noqa: E402
from editflows.deletion_task import to_device  # noqa: E402
from editflows.editflows import rm_blanks  # noqa: E402
from editflows.grammar import (  # noqa: E402
    ideal_scores,
    insertion_posterior,
    mean_auroc,
    ranking_scores,
    rule_auroc,
    rule_probes,
)
from editflows.metrics import Buckets, DeletionCounts  # noqa: E402
from evaluate import load_model  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

EPS = EditFlow.EPSILON_TOKEN


def find_checkpoint(run_dir: Path, which: str) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if which in ("last", "periodic"):
        return ckpt_dir / f"{which}.ckpt"
    if which == "best":
        return max(ckpt_dir.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime)
    return Path(which)


def rebuilt_table(dm, args) -> np.ndarray:
    """The generator's transition table, checked against the trigrams of the sequence file."""
    P = transition_table(np.random.default_rng(args.table_seed), dm.n_letters, args.n_allowed, args.concentration)
    tok = dm.train_data.tokens[:3_000_000].astype(np.int64)
    seq_of = np.repeat(np.arange(len(dm.train_data.offsets) - 1), np.diff(dm.train_data.offsets))[: len(tok)]
    inside = seq_of[:-2] == seq_of[2:]
    assert (P[tok[:-2], tok[1:-1], tok[2:]][inside] > 0).all(), (
        "trigrams outside the rebuilt table - the sequence file was generated with other --table-seed / --n-allowed"
    )
    return P


def loss_references(model, dm, config, P, n: int, device) -> dict:
    """Model loss, time-only optimum and exact Bayes loss on the same (t, x_t), in float64."""
    path = instantiate(config.task.path)
    flow = EditFlow(probability_path=path, bos_token=dm.bos_token, low_discrepancy_sampling=True)
    coupling, aligner = instantiate(config.task.coupling), KnownAligner()
    cap = float(config.task.model.get("max_log_lambda", 10.0))
    floor, ceiling = math.exp(-cap), math.exp(cap)
    gen_cpu, gen = torch.Generator().manual_seed(0), torch.Generator(device).manual_seed(0)
    B = 32
    rows, letters, labels, scores = [], [], [], []
    for b in range(n // B):
        x_1 = DataBatch.from_sequences([dm.val_data[i] for i in range(b * B, (b + 1) * B)])
        x_0, x_1 = coupling.couple(None, x_1, generator=gen_cpu)
        inserted = [int(m.sum()) for m in torch.split(x_0.inserted, list(x_0.seq_lens))]
        z_0, z_1 = aligner.align(x_0, x_1)
        z_0, z_1 = z_0.to(device), z_1.to(device)
        t = flow._sample_t(batch_size=B, device=device, generator=gen)
        z_t, _ = path.sample(t=t, z_0=z_0, z_1=z_1, generator=gen)
        x_t = rm_blanks(z_t)
        leftover = (z_1 == EPS)[z_t != EPS]
        not_bos = x_t.x != dm.bos_token
        w = path.kappa.loss_coefficient(t).double()
        with torch.inference_mode():
            ll = model(x_t, t).log_lambdas.double()

        def per_seq(values):
            return torch.zeros(B, dtype=torch.float64, device=device).index_add_(0, x_t.seq_idx, values)

        loss = per_seq(ll[0].exp() + (ll[1].exp() + ll[2].exp()) * not_bos) - w * per_seq(ll[2] * leftover)
        lens = list(x_t.seq_lens)
        for i, (s, m, sc) in enumerate(zip(torch.split(x_t.x, lens), torch.split(leftover, lens), torch.split(ll[2], lens))):
            letters.append(s[1:].cpu().numpy().astype(np.int64))
            labels.append(m[1:].cpu().numpy())
            scores.append(sc[1:].cpu().numpy())
            rows.append({"t": float(t[i]), "w": float(w[i]), "N": lens[i] - 1, "R": int(m.sum()), "K": inserted[i],
                         "n1": x_1.seq_lens[i] - 1, "model": float(loss[i])})
    D = pd.DataFrame(rows)

    Q = insertion_posterior(letters, [float(m.mean()) for m in labels], P)
    idle = (2 * D.N + 1) * floor  # insert and substitute at the lowest rate the head can express
    bayes = []
    for q, m, w in zip(Q, labels, D.w):
        lam = np.clip(w * q, floor, ceiling)
        bayes.append(lam.sum() - w * np.log(lam[m]).sum())
    D["bayes"] = np.array(bayes) + idle
    remaining = 1 - path.kappa(torch.tensor(D.t.to_numpy())).numpy()
    share = D.K.mean() * remaining / (D.n1.mean() + D.K.mean() * remaining)
    c = np.clip(D.w * share, floor, ceiling)
    D["time_only"] = D.N * c - D.w * D.R * np.log(c) + idle

    def se(v):
        return float(v.std() / math.sqrt(len(v)))

    model_loss, time_only, bayes_loss = D.model.mean(), D.time_only.mean(), D.bayes.mean()
    return {
        "n": len(D),
        "model": float(model_loss), "model_se": se(D.model),
        "time_only": float(time_only), "bayes": float(bayes_loss), "bayes_se": se(D.bayes),
        "covered": float((time_only - model_loss) / (time_only - bayes_loss)),
        "auroc": mean_auroc(scores, labels), "auroc_bayes": mean_auroc(Q, labels),
    }


def evaluation_sources(dm, config):
    """The sources of the per-epoch sampling evaluation: the first validation targets, coupled with its seed."""
    n = min(config.task.eval.n_sample_seqs, len(dm.val_data))
    x_1 = DataBatch.from_sequences([dm.val_data[i] for i in range(n)])
    generator = torch.Generator().manual_seed(config.task.eval.get("seed", 1234))
    return instantiate(config.task.coupling).couple(None, x_1, generator=generator)


def encoder_probe(model, x_0, device, n_sources: int = 64) -> dict:
    """What the encoder output still says about the letters: accuracy of a linear read-out (ridge regression,
    fitted on one half of the tokens, scored on the other) of the letter itself and of its neighbours. An
    untrained encoder gives 1.0 for the letter itself and chance (0.05) for the neighbours."""
    lens = list(x_0.seq_lens)[:n_sources]
    part = to_device(DataBatch.from_sequences(list(torch.split(x_0.x, list(x_0.seq_lens))[:n_sources])), device)
    with torch.inference_mode():
        H = model.encoder(part).double()
    tokens, pos = part.x.long(), part.token_pos
    inner = torch.nonzero((pos >= 3) & (torch.roll(pos, -2) > pos))[:, 0]  # two letters on each side in the sequence
    inner = inner[torch.randperm(len(inner), device=device, generator=torch.Generator(device).manual_seed(0))]
    H = torch.cat([H[inner], torch.ones(len(inner), 1, device=device, dtype=H.dtype)], dim=1)
    half = len(inner) // 2
    gram = H[:half].T @ H[:half] + 1e-2 * half * torch.eye(H.shape[1], device=device, dtype=H.dtype)
    out = {}
    for name, shift in (("itself", 0), ("before", -1), ("after", 1)):
        target = tokens[inner + shift]
        weights = torch.linalg.solve(gram, H[:half].T @ torch.nn.functional.one_hot(target[:half], 21).to(H.dtype))
        out[name] = float(((H[half:] @ weights).argmax(dim=1) == target[half:]).float().mean())
    return out


def ranking_at_zero(model, x_0, P, device, fraction_bins) -> dict:
    lens = list(x_0.seq_lens)
    seqs = [s[1:].numpy().astype(np.int64) for s in torch.split(x_0.x, lens)]
    masks = [m[1:].numpy() for m in torch.split(x_0.inserted, lens)]
    scores = []
    with torch.inference_mode():
        for start in range(0, len(lens), 64):
            part = DataBatch.from_sequences([torch.split(x_0.x, lens)[i] for i in range(start, min(start + 64, len(lens)))])
            rate = model(to_device(part, device), torch.zeros(part.batch_size, device=device))
            scores += [s[1:].float().cpu().numpy() for s in torch.split(rate.log_lambda_del, list(part.seq_lens))]
    Q = insertion_posterior(seqs, [float(m.mean()) for m in masks], P)
    ideal = ideal_scores(masks, Q)
    # the same per bucket of the original fraction (bucketed like the per-epoch table): where is the gap?
    bucket = np.searchsorted(np.asarray(fraction_bins[1:-1]), [1 - m.mean() for m in masks], side="right")
    by_fraction = []
    for b in range(len(fraction_bins) - 1):
        idx = np.flatnonzero(bucket == b)
        if len(idx):
            part = ideal_scores([masks[i] for i in idx], [Q[i] for i in idx])
            mine = ranking_scores([scores[i] for i in idx], [masks[i] for i in idx])
            by_fraction.append({"fraction": f"{fraction_bins[b]:g}-{fraction_bins[b + 1]:g}", "n_seq": len(idx),
                                "rank_p_at_k": mine["rank_p_at_k"], "rank_p_at_k_ideal": part["top_k_precision"],
                                "sampling_f1_ideal": part["sampling_f1"]})
    return {"auroc": mean_auroc(scores, masks), "auroc_bayes": ideal["auroc"], **ranking_scores(scores, masks),
            "rank_p_at_k_ideal": ideal["top_k_precision"], "ideal": ideal, "by_fraction": by_fraction}


def probe(model, dm, config, P, repeats: int, device) -> dict:
    """Per context: does the deletion rate of a letter that closes a clean stretch tell ineligible from eligible?"""
    kappa = instantiate(config.task.path).kappa
    lo, hi = 0.0, 1.0
    target = min(0.95, float(kappa(torch.tensor(0.999))))  # the time at which 5 % of the insertions are left
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if float(kappa(torch.tensor(mid))) < target else (lo, mid)
    t_probe = hi
    rows, context, letter = rule_probes(dm.train_data.tokens, dm.train_data.offsets, dm.train_data.indices,
                                        n_letters=dm.n_letters, repeats=repeats)
    eligible = (P > 0).reshape(-1, dm.n_letters)[context, letter]
    width = rows.shape[1] + 1
    scores = []
    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            part = torch.as_tensor(rows[start : start + 256], dtype=torch.int32)
            part = torch.cat((torch.full((len(part), 1), dm.bos_token, dtype=torch.int32), part), dim=1).to(device)
            rate = model(DataBatch.from_equilength_batch(part), torch.full((len(part),), t_probe, device=device))
            scores.append(rate.log_lambda_del.float().view(len(part), width)[:, -1].cpu().numpy())
    model_auroc = rule_auroc(np.concatenate(scores), context, eligible)
    ideal = np.array([q[-1] for q in insertion_posterior(rows, 1 / rows.shape[1], P)])
    ideal_auroc = rule_auroc(ideal, context, eligible)
    return {"t": t_probe, "median": float(np.median(model_auroc)), "rules_ge_0.9": int((model_auroc >= 0.9).sum()),
            "rules_ge_0.7": int((model_auroc >= 0.7).sum()), "ideal_median": float(np.median(ideal_auroc)),
            "ideal_rules_ge_0.9": int((ideal_auroc >= 0.9).sum()), "per_context": [round(float(v), 4) for v in model_auroc]}


def sampled_metrics(run_dir: Path, epoch: int, model, dm, config, x_0, x_1, device, resample: bool) -> dict:
    keys = ["del_precision", "del_recall", "del_f1", "del_chance", "len_err", "exact_match", "ins_per_seq", "sub_per_seq"]
    table = run_dir / "eval" / "deletion_metrics.csv"
    if table.is_file() and not resample:
        t = pd.read_csv(table)
        row = t[(t.stage == "val") & (t.target_len == "all") & (t.epoch == epoch)]
        if len(row):
            return {"source": "per-epoch evaluation", **{k: float(row.iloc[0][k]) for k in keys}}
    buckets = Buckets(config.task.eval.target_len_bins, config.task.eval.fraction_bins)
    flow = EditFlow(probability_path=instantiate(config.task.path), bos_token=dm.bos_token,
                    low_discrepancy_sampling=config.task.low_discrepancy_sampling)
    counts = DeletionCounts(buckets, device=device)
    generator = torch.Generator(device).manual_seed(0)
    lens = list(x_0.seq_lens)
    for start in range(0, len(lens), 64):
        take = range(start, min(start + 64, len(lens)))
        s_0 = type(x_0)(seq_lens=tuple(lens[i] for i in take), x=torch.cat([torch.split(x_0.x, lens)[i] for i in take]),
                        inserted=torch.cat([torch.split(x_0.inserted, lens)[i] for i in take]))
        s_1 = DataBatch.from_sequences([x_1.split_sequences()[i] for i in take])
        s_0, s_1 = to_device(s_0, device), to_device(s_1, device)
        s_0.origin = torch.arange(len(s_0.x), device=device)
        x_hat = flow.sample_euler(model, x_0=s_0, n_steps=config.task.eval.sample_steps, generator=generator)
        counts.update_sampling(s_0, s_1, x_hat, buckets.index_pair(s_0, s_1))
    summary = counts.summary()
    return {"source": "resampled", **{k: float(summary[k]) for k in keys}}


def training_stats(run_dir: Path, config) -> dict:
    files = sorted((run_dir / "logs").glob("version_*/metrics.csv"))
    if not files:
        return {}
    logs = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out = {"steps": int(logs.step.max()) + 1, "clip": float(config.trainer.get("gradient_clip_val") or 0)}
    per_step = config.data.batch_size * config.get("gpus", 1) * config.trainer.get("accumulate_grad_batches", 1)
    out["sequences"] = out["steps"] * per_step
    loss = logs["train/loss"].dropna()
    spread = loss.quantile(0.9) - loss.quantile(0.1)
    out["loss_spikes"] = int((loss > loss.median() + 20 * max(spread, 1e-9)).sum())
    if "train/grad_norm" in logs:
        norm = logs["train/grad_norm"].dropna()
        late = norm.iloc[len(norm) // 5 :]  # after the first fifth: the settled regime
        out.update(grad_norm_median=float(late.median()), grad_norm_max=float(logs["train/grad_norm_max"].max()),
                   clipped_share=float((late > out["clip"]).mean()) if out["clip"] else 0.0,
                   encoder_grad_share=float(logs["train/grad_encoder_share"].dropna().iloc[len(norm) // 5 :].median()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory")
    p.add_argument("--checkpoint", default="last", help="last (default), best, periodic or a checkpoint file")
    p.add_argument("-n", type=int, default=512, help="validation sequences for the loss references")
    p.add_argument("--probe-repeats", type=int, default=2, help="probe sequences per (context, letter)")
    p.add_argument("--resample", action="store_true", help="sample again instead of reading the per-epoch table")
    p.add_argument("--table-seed", type=int, default=0, help="--seed of scripts/make_synthetic_targets.py")
    p.add_argument("--n-allowed", type=int, default=5)
    p.add_argument("--concentration", type=float, default=1.0)
    args = p.parse_args()

    run_dir = args.run.resolve()
    os.chdir(ROOT)  # data paths in the config are relative to the repository
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except ValueError:
        pass
    path = find_checkpoint(run_dir, args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, config, ema=True).to(device)
    dm = instantiate(config.data, seed_sequence=np.random.SeedSequence(0))
    dm.prepare_data()
    dm.setup("fit")
    P = rebuilt_table(dm, args)

    x_0, x_1 = evaluation_sources(dm, config)
    result = {
        "run": run_dir.name,
        "checkpoint": {"file": path.name, "epoch": int(checkpoint["epoch"]), "step": int(checkpoint["global_step"])},
        "schedule": OmegaConf.to_container(config.task.path.kappa),
        "original_fraction": list(config.task.coupling.original_fraction),
        "loss": loss_references(model, dm, config, P, args.n, device),
        "ranking_t0": ranking_at_zero(model, x_0, P, device, list(config.task.eval.fraction_bins)),
        "probe": probe(model, dm, config, P, args.probe_repeats, device),
        "encoder_probe": encoder_probe(model, x_0, device),
        "sampled": sampled_metrics(run_dir, int(checkpoint["epoch"]), model, dm, config, x_0, x_1, device, args.resample),
        "training": training_stats(run_dir, config),
    }
    out = run_dir / "eval" / "grammar.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    L, R, PR, S, T = (result[k] for k in ("loss", "ranking_t0", "probe", "sampled", "training"))
    ideal = R["ideal"]
    print(f"{result['run']}: {path.name}, epoch {result['checkpoint']['epoch']}, step {result['checkpoint']['step']}")
    print(f"  loss        model {L['model']:8.1f}   time-only {L['time_only']:8.1f}   Bayes {L['bayes']:8.1f}"
          f"   -> {100 * L['covered']:5.1f} % of the distance covered")
    print(f"  AUROC       on x_t {L['auroc']:.3f} (Bayes {L['auroc_bayes']:.3f})   at t = 0 on the sources "
          f"{R['auroc']:.3f} (Bayes {R['auroc_bayes']:.3f})")
    print(f"  rank_p_at_k {R['rank_p_at_k']:.3f}   random {R['rank_p_at_k_random']:.3f}   ideal {R['rank_p_at_k_ideal']:.3f}")
    print(f"  rule probe  median AUROC {PR['median']:.3f}   rules >= 0.9: {PR['rules_ge_0.9']} / 400"
          f"   >= 0.7: {PR['rules_ge_0.7']}   (ideal: median {PR['ideal_median']:.3f}, {PR['ideal_rules_ge_0.9']} / 400)")
    E = result["encoder_probe"]
    print(f"  encoder     linear read-out of the letter itself {E['itself']:.3f}, the letter before {E['before']:.3f}, "
          f"after {E['after']:.3f}   (untrained: 1.0 / 0.05 / 0.05)")
    print(f"  sampled     precision {S['del_precision']:.3f}  recall {S['del_recall']:.3f}  F1 {S['del_f1']:.3f}"
          f"  (chance {S['del_chance']:.3f}; ideal {ideal['sampling_precision']:.3f} / {ideal['sampling_recall']:.3f}"
          f" / {ideal['sampling_f1']:.3f})   len_err {S['len_err']:.1f}   ins {S['ins_per_seq']:.3f}  sub {S['sub_per_seq']:.3f}")
    if "grad_norm_median" in T:
        print(f"  training    {T['steps']} steps, {T['sequences']:,} sequences; gradient norm median {T['grad_norm_median']:.0f}"
              f", max {T['grad_norm_max']:.0f}, clip {T['clip']:g} active on {100 * T['clipped_share']:.0f} % of the steps;"
              f" encoder share {T['encoder_grad_share']:.3f}; loss spikes {T['loss_spikes']}")
    elif T:
        print(f"  training    {T['steps']} steps, {T['sequences']:,} sequences; loss spikes {T['loss_spikes']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
