#!/usr/bin/env python

"""Sample sequences from a trained Edit Flows checkpoint."""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from editflows import EditFlow


def load_model(config: DictConfig, state_dict: dict, *, use_ema: bool) -> torch.nn.Module:
    """Rebuild the backbone and load either the EMA or the raw weights."""

    model = instantiate(config.task.model)
    prefix = "ema_model.ema_model." if use_ema else "model."
    weights = {
        key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)
    }
    model.load_state_dict(weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to a Lightning .ckpt file")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=8, help="Number of sequences to sample"
    )
    parser.add_argument(
        "-s", "--steps", type=int, default=100, help="Euler steps for the CTMC integration"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the sampling RNG")
    parser.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="Sample with the latest weights instead of EMA weights",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device ('auto', 'cpu', 'cuda', 'cuda:0', ...)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])

    tokenizer = spm.SentencePieceProcessor(model_proto=checkpoint["tokenizer"])
    bos_token = tokenizer.bos_id()

    model = load_model(config, checkpoint["state_dict"], use_ema=args.use_ema)
    model = model.to(device)

    noise_config = config.task.noise
    p_noise = instantiate(noise_config, bos_token=bos_token)

    edit_flow = EditFlow(
        probability_path=instantiate(config.task.path),
        bos_token=bos_token,
        low_discrepancy_sampling=config.task.low_discrepancy_sampling,
    )

    generator = torch.Generator(device).manual_seed(args.seed)
    model.eval()
    with torch.inference_mode():
        x_0 = p_noise.sample(args.num_samples, generator=generator)
        x_1 = edit_flow.sample_euler(model, x_0=x_0, n_steps=args.steps, generator=generator)
    samples: list[str] = [
        tokenizer.decode(seq.numpy(force=True).tolist()) for seq in x_1.split_sequences()
    ]

    print("\n\n".join(samples))


if __name__ == "__main__":
    main()
