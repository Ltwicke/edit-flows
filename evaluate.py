#!/usr/bin/env python
"""Evaluate a checkpoint: build sources from held-out targets with the extended coupling, sample from them
and report the deletion confusion matrix, precision and recall per length bucket.

    python evaluate.py runs/<name>/checkpoints/last.ckpt -n 256 --show 4 --out metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from editflows import DataBatch, EditFlow
from editflows.coupling import CoupledBatch, KnownAligner
from editflows.deletion_task import to_device
from editflows.metrics import Buckets, DeletionCounts


def load_model(checkpoint: dict, config, *, ema: bool):
    model = instantiate(config.task.model)
    prefix = "ema_model.ema_model." if ema else "model."
    state_dict = {k[len(prefix) :]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
    model.load_state_dict(state_dict)
    return model.eval()


def render(datamodule, x_0: CoupledBatch, x_1: DataBatch, x_hat: DataBatch, i: int) -> str:
    """Three lines per example: the target, the source (inserted letters in lower case) and the verdict per
    source token: `+` correctly deleted, `x` wrongly deleted, `_` insertion that was kept, `.` kept original."""
    lens_0, lens_hat = list(x_0.seq_lens), list(x_hat.seq_lens)
    src = torch.split(x_0.x, lens_0)[i][1:]
    ins = torch.split(x_0.inserted, lens_0)[i][1:]
    tgt = x_1.split_sequences()[i][1:]
    offset = sum(lens_0[:i]) + 1
    origin = torch.split(x_hat.origin, lens_hat)[i]
    kept = torch.zeros(len(src), dtype=torch.bool)
    kept[origin[origin >= offset] - offset] = True
    n_ins_events = int((origin < 0).sum())

    source = "".join(c.lower() if is_ins else c for c, is_ins in zip(datamodule.decode(src.tolist()), ins.tolist()))
    verdict = "".join(
        ("_" if k else "+") if is_ins else ("." if k else "x") for is_ins, k in zip(ins.tolist(), kept.tolist())
    )
    return (
        f"target  : {datamodule.decode(tgt.tolist())}\n"
        f"source  : {source}\n"
        f"verdict : {verdict}   (+ correct deletion, x wrong deletion, _ missed insertion; "
        f"{n_ins_events} insertion events)\n"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("-n", "--num-seqs", type=int, default=256, help="held-out sequences to sample from")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("-s", "--steps", type=int, default=100, help="Euler steps")
    p.add_argument("--original-fraction", type=float, nargs=2, metavar=("MIN", "MAX"),
                   help="override the coupling's original-fraction range")
    p.add_argument("--data", type=Path, help="override the sequence file of the checkpoint's config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show", type=int, default=4, help="print this many examples")
    p.add_argument("--out", type=Path, help="write the metrics and counts as JSON")
    p.add_argument("--no-ema", action="store_true", help="use the raw weights instead of the EMA weights")
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, config, ema=not args.no_ema).to(device)

    if args.data is not None:
        config.data.path = str(args.data)
    datamodule = instantiate(config.data, seed_sequence=np.random.SeedSequence(args.seed))
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    dataset = datamodule.val_data if args.split == "val" else datamodule.test_data

    coupling_config = OmegaConf.to_container(config.task.coupling, resolve=True)
    if args.original_fraction is not None:
        coupling_config["original_fraction"] = args.original_fraction
    coupling = instantiate(coupling_config)
    aligner = KnownAligner()
    edit_flow = EditFlow(
        probability_path=instantiate(config.task.path),
        bos_token=datamodule.bos_token,
        low_discrepancy_sampling=config.task.low_discrepancy_sampling,
    )
    buckets = Buckets(config.task.eval.target_len_bins, config.task.eval.fraction_bins)
    counts = DeletionCounts(buckets, device=device)

    cpu_generator = torch.Generator().manual_seed(args.seed)
    generator = torch.Generator(device).manual_seed(args.seed)
    n = min(args.num_seqs, len(dataset))
    batch_size = datamodule.eval_batch_size
    shown = 0
    for start in range(0, n, batch_size):
        x_1 = DataBatch.from_sequences([dataset[i] for i in range(start, min(start + batch_size, n))])
        x_0, x_1 = coupling.couple(None, x_1, generator=cpu_generator)
        aligner.align(x_0, x_1)  # sanity: the alignment exists
        x_0, x_1 = to_device(x_0, device), to_device(x_1, device)
        x_0.origin = torch.arange(len(x_0.x), device=device)
        bucket_ids = buckets.index_pair(x_0, x_1)
        with torch.inference_mode():
            rate = model(x_0, torch.zeros(x_0.batch_size, device=device))
            counts.update_ranking(rate.log_lambda_del, x_0, bucket_ids)
            x_hat = edit_flow.sample_euler(model, x_0=x_0, n_steps=args.steps, generator=generator)
        counts.update_sampling(x_0, x_1, x_hat, bucket_ids)
        while shown < args.show and shown - start < x_0.batch_size:
            print(render(datamodule, x_0, x_1, x_hat, shown - start))
            shown += 1

    summary = counts.summary()
    print(f"{args.split} split, {n} sequences, {args.steps} Euler steps, {'EMA' if not args.no_ema else 'raw'} weights")
    print(", ".join(f"{k}={v:.3f}" for k, v in summary.items()))
    rows = [r for r in counts.rows(stage=args.split, epoch=-1, step=-1) if r["n_seq"] > 0]
    columns = ["target_len", "fraction", "n_seq", "tp", "fp", "fn", "tn", "del_precision", "del_recall", "del_f1",
               "del_chance", "ins_per_seq", "sub_per_seq", "exact_match"]
    print(pd.DataFrame(rows)[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "buckets": rows}, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
