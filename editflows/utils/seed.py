import random
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(config: DictConfig) -> np.random.SeedSequence:
    """Set the configured seed in pytorch, numpy and python's random.

    The seed is a big int that wandb converts into a float, destroying the seed, so we
    store it as a string instead.
    """

    big_seed = int(config.seed) if config.seed is not None else None
    big_seed, seed_sequence = manual_seed(big_seed)
    config.seed = str(big_seed)
    return seed_sequence


def manual_seed(seed: Optional[int]) -> tuple[int, np.random.SeedSequence]:
    """Seed all python, numpy and pytorch RNGs.

    We derive different per-library seeds from the root seed, to avoid correlated samples in case,
    for example, numpy and CPU torch use the same algorithm.

    Note that all pytorch devices, e.g. CPU, CUDA, etc., are seeded with the same value. This
    ensures that, for example, model weights are initialized consistently in multi-GPU training. If
    you want different random values per device, manage `torch.Generator`s manually.
    """

    root_ss = np.random.SeedSequence(seed)
    std_ss, np_ss, torch_ss = root_ss.spawn(3)

    # Python uses a Mersenne twister with 624 words of state, so we provide enough seed to
    # initialize it fully
    random.seed(std_ss.generate_state(624).tobytes())

    # We seed the global RNG anyway in case some library uses it internally
    np.random.seed(int(np_ss.generate_state(1, np.uint32)[0]))

    # Pytorch takes a uint64 seed
    torch.manual_seed(int(torch_ss.generate_state(1, np.uint64)[0]))

    root_seed = root_ss.entropy
    assert isinstance(root_seed, int)
    return root_seed, root_ss
