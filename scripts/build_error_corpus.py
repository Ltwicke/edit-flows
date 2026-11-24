#!/usr/bin/env python
"""Sample deletions and store them per source letter, for the error analysis (`editflows/errors.py`).

All corpora share one set of sources, `runs/error-study/sources.npz` (held-out targets with fresh insertions):

    python scripts/build_error_corpus.py sources                      # 3 000 validation + 3 000 test targets
    python scripts/build_error_corpus.py model runs/lap-01            # the trained model, standard sampler
    python scripts/build_error_corpus.py model runs/lap-01 --seed 1 --first 500 --log-q --tag rep1
    python scripts/build_error_corpus.py exact --steps 100            # exact rates, prior from the schedule
    python scripts/build_error_corpus.py exact --prior estimated      # ... with the noise level estimated
    python scripts/build_error_corpus.py ffbs                         # exact joint posterior sample, no time axis

`model` writes `runs/<run>/errors/<tag>.npz`, the references write `runs/error-study/<tag>.npz`. Every file holds
arrays that are flat over the letters of the first `--first` sources: `del_step` (step of the deletion, -1 = kept;
for `ffbs` 0 = sampled as an insertion) and, for `model`, `q0`, `exposure` and optionally `qlog`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from editflows import exact  # noqa: E402
from editflows.errors import Sources, build_sources, model_euler  # noqa: E402
from evaluate import load_model  # noqa: E402
from make_synthetic_targets import transition_table  # noqa: E402

STUDY = ROOT / "runs" / "error-study"
REFERENCE_RUN = ROOT / "runs" / "lap-01"  # its config defines the data, the coupling and the schedule


def table(device) -> torch.Tensor:
    return torch.tensor(transition_table(np.random.default_rng(0), 20, 5, 1.0), dtype=torch.float64, device=device)


def run_config(run: Path):
    checkpoint = torch.load(run / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    return checkpoint, OmegaConf.create(checkpoint["config"])


def padded(sources: Sources, ids, device):
    return exact.pad([sources.seq(i) for i in ids], device)


def chunks(sources: Sources, first: int, size: int):
    """Chunks of sources of similar length (shorter batches are cheaper), as index arrays."""
    order = np.argsort(np.diff(sources.offsets)[:first], kind="stable")
    return [order[i : i + size] for i in range(0, first, size)]


def token_chunks(sources: Sources, first: int, max_tokens: int):
    """Chunks of sources of similar length with at most `max_tokens` letters each: what a forward pass of the
    model costs depends on the letters, not on the number of sources."""
    lengths = np.diff(sources.offsets)[:first]
    out, current, total = [], [], 0
    for i in np.argsort(lengths, kind="stable"):
        if current and total + lengths[i] > max_tokens:
            out.append(np.array(current))
            current, total = [], 0
        current.append(i)
        total += int(lengths[i])
    return out + ([np.array(current)] if current else [])


def scatter(sources: Sources, ids, values, out):
    """Write the per-letter `values` of the sources `ids` (a padded row each) into the flat array `out`."""
    for row, i in enumerate(ids):
        span = sources.span(i)
        out[span] = values[row, : span.stop - span.start]


def cmd_sources(args):
    _, config = run_config(REFERENCE_RUN)
    dm = instantiate(config.data, seed_sequence=np.random.SeedSequence(0))
    dm.prepare_data(); dm.setup("fit"); dm.setup("test")
    sources = build_sources(dm, instantiate(config.task.coupling), args.n_val, args.n_test, args.seed)
    STUDY.mkdir(parents=True, exist_ok=True)
    sources.save(STUDY / "sources.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = table(device)
    q_true = np.zeros(len(sources.letters), dtype=np.float32)
    q_est = np.zeros(len(sources.letters), dtype=np.float32)
    rho_hat = np.zeros(len(sources))
    for ids in chunks(sources, len(sources), args.batch):
        S, valid = padded(sources, ids, device)
        rho = torch.tensor(sources.rho[ids], dtype=torch.float64, device=device)
        est = exact.estimate_rho(S, valid, P)
        rho_hat[ids] = est.cpu().numpy()
        scatter(sources, ids, exact.posterior(S, valid, rho, P).cpu().numpy(), q_true)
        scatter(sources, ids, exact.posterior(S, valid, est, P).cpu().numpy(), q_est)
        torch.cuda.empty_cache()
    np.savez_compressed(STUDY / "posterior.npz", q_true=q_true, q_est=q_est, rho_hat=rho_hat)
    print(f"{len(sources)} sources, {len(sources.letters):,} letters, {int(sources.inserted.sum()):,} inserted -> {STUDY}")


def cmd_model(args):
    sources = Sources.load(STUDY / "sources.npz")
    checkpoint, config = run_config(args.run)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, config, ema=True).to(device)
    kappa = instantiate(config.task.path).kappa
    first = min(args.first or len(sources), len(sources))
    n = int(sources.offsets[first])
    out = {"del_step": np.full(n, -1, dtype=np.int16), "q0": np.zeros(n, dtype=np.float32), "exposure": np.zeros(n, dtype=np.float32)}
    if args.log_q:
        out["qlog"] = np.full((n, args.steps), np.nan, dtype=np.float16)
    start = time.time()
    for ids in token_chunks(sources, first, args.max_tokens):
        result = model_euler(model, [sources.seq(i) for i in ids], kappa, config.data.vocab_size - 1, device=device,
                             n_steps=args.steps, seed=args.seed, exclusion=args.exclusion, sharpen=args.sharpen,
                             floor=args.floor, log_q=args.log_q)
        at = 0
        for i in ids:
            span = sources.span(i)
            length = span.stop - span.start
            for key in out:
                out[key][span] = result[key][at : at + length]
            at += length
    target = args.run / "errors"
    target.mkdir(exist_ok=True)
    np.savez_compressed(target / f"{args.tag}.npz", first=first, steps=args.steps, seed=args.seed, epoch=int(checkpoint["epoch"]), **out)
    deleted, inserted = out["del_step"] >= 0, sources.inserted[:n]
    tp = int((deleted & inserted).sum())
    print(f"{args.run.name} {args.tag}: {first} sources in {time.time() - start:.0f} s; precision {tp / max(deleted.sum(), 1):.4f} "
          f"recall {tp / max(inserted.sum(), 1):.4f}")


def cmd_exact(args):
    sources = Sources.load(STUDY / "sources.npz")
    _, config = run_config(REFERENCE_RUN)
    kappa = instantiate(config.task.path).kappa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = table(device)
    first = min(args.first or len(sources), len(sources))
    n = int(sources.offsets[first])
    del_step = np.full(n, -1, dtype=np.int16)
    rho_all = sources.rho if args.prior == "true" else np.load(STUDY / "posterior.npz")["rho_hat"]
    start = time.time()
    for ids in chunks(sources, first, args.batch):
        S, valid = padded(sources, ids, device)
        rho = torch.tensor(rho_all[ids], dtype=torch.float64, device=device)
        steps = exact.exact_euler(S, valid, rho, P, kappa, n_steps=args.steps, seed=args.seed, exclusion=args.exclusion,
                                  leak=args.leak, budget=args.budget)
        scatter(sources, ids, steps.cpu().numpy(), del_step)
    np.savez_compressed(STUDY / f"{args.tag}.npz", first=first, steps=args.steps, seed=args.seed, del_step=del_step)
    deleted, inserted = del_step >= 0, sources.inserted[:n]
    tp = int((deleted & inserted).sum())
    print(f"exact {args.tag}: {first} sources in {time.time() - start:.0f} s; precision {tp / max(deleted.sum(), 1):.4f} "
          f"recall {tp / max(inserted.sum(), 1):.4f}")


def cmd_ffbs(args):
    sources = Sources.load(STUDY / "sources.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = table(device)
    first = min(args.first or len(sources), len(sources))
    n = int(sources.offsets[first])
    del_step = np.full(n, -1, dtype=np.int16)
    rho_all = sources.rho if args.prior == "true" else np.load(STUDY / "posterior.npz")["rho_hat"]
    generator = torch.Generator(device).manual_seed(args.seed)
    for ids in chunks(sources, first, args.batch):
        S, valid = padded(sources, ids, device)
        rho = torch.tensor(rho_all[ids], dtype=torch.float64, device=device)
        sample = exact.posterior_sample(S, valid, rho, P, generator=generator, budget=args.budget)
        scatter(sources, ids, torch.where(sample, 0, -1).cpu().numpy(), del_step)
    np.savez_compressed(STUDY / f"{args.tag}.npz", first=first, steps=0, seed=args.seed, del_step=del_step)
    deleted, inserted = del_step >= 0, sources.inserted[:n]
    tp = int((deleted & inserted).sum())
    print(f"ffbs {args.tag}: {first} sources; precision {tp / max(deleted.sum(), 1):.4f} recall {tp / max(inserted.sum(), 1):.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu-fraction", type=float, default=0.18,
                   help="share of the GPU memory this process may use (a training run may share the GPU)")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sources")
    s.add_argument("--n-val", type=int, default=3000); s.add_argument("--n-test", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0); s.add_argument("--batch", type=int, default=256)
    m = sub.add_parser("model")
    m.add_argument("run", type=Path); m.add_argument("--tag", default="m100")
    m.add_argument("--log-q", action="store_true"); m.add_argument("--sharpen", type=float, default=1.0)
    m.add_argument("--floor", type=float, default=0.0)
    m.add_argument("--max-tokens", type=int, default=30_000, help="letters per forward pass")
    e = sub.add_parser("exact")
    e.add_argument("--tag", default="i2-100")
    e.add_argument("--leak", type=float, default=0.0, help="a deliberate defect: q is never taken below this")
    f = sub.add_parser("ffbs")
    f.add_argument("--tag", default="ffbs")
    for q in (m, e):
        q.add_argument("--steps", type=int, default=100); q.add_argument("--exclusion", type=int, default=0)
    for q in (m, e, f):
        q.add_argument("--seed", type=int, default=0); q.add_argument("--first", type=int, default=0)
    for q in (e, f):
        q.add_argument("--batch", type=int, default=768)
    for q in (e, f):
        q.add_argument("--prior", choices=["true", "estimated"], default="true")
        q.add_argument("--budget", type=float, default=9e7)
    args = p.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    {"sources": cmd_sources, "model": cmd_model, "exact": cmd_exact, "ffbs": cmd_ffbs}[args.command](args)


if __name__ == "__main__":
    main()
