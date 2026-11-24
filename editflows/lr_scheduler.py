import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, SequentialLR


def lerp(a: float, b: float, t: float) -> float:
    """Linearly interpolate between a and b."""
    return a + (b - a) * t


class WarmUpScheduler(LRScheduler):
    """Increase learning rate linearly from 0 to the target LR over the first few steps."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        start_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [
                lerp(self.start_lr, base_lr, self.last_epoch / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr for base_lr in self.base_lrs]


class WarmUpCosineAnnealing(SequentialLR):
    """Cosine annealing with linear warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        start_lr: float,
        end_lr: float,
        last_epoch: int = -1,
    ):
        schedulers = [
            WarmUpScheduler(optimizer, warmup_steps, start_lr, last_epoch),
            CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=end_lr,
                last_epoch=max(-1, last_epoch - warmup_steps),
            ),
        ]
        milestones = [warmup_steps]

        super().__init__(optimizer, schedulers, milestones, last_epoch=last_epoch)


class WarmUpCosineGroups(LRScheduler):
    """`WarmUpCosineAnnealing` in closed form, with a delayed start per parameter group.

    Every group follows the same shape (linear warm-up from `start_lr`, cosine annealing to `end_lr`, both
    relative to the first group's learning rate) scaled by its own base learning rate. A group with
    `delays[g] > 0` keeps the learning rate 0 for that many steps and then ramps up linearly over `ramps[g]`
    steps. For a single group without delay this is exactly `WarmUpCosineAnnealing`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_steps: int,
        start_lr: float,
        end_lr: float,
        delays: list[int] | None = None,
        ramps: list[int] | None = None,
        last_epoch: int = -1,
    ):
        n_groups = len(optimizer.param_groups)
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.delays = list(delays) if delays is not None else [0] * n_groups
        self.ramps = list(ramps) if ramps is not None else [0] * n_groups
        assert len(self.delays) == len(self.ramps) == n_groups

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        reference = self.base_lrs[0]
        start, end = self.start_lr / reference, self.end_lr / reference
        if step < self.warmup_steps:
            shape = lerp(start, 1.0, step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            shape = end + (1.0 - end) * (1 + math.cos(math.pi * progress)) / 2
        lrs = []
        for base_lr, delay, ramp in zip(self.base_lrs, self.delays, self.ramps):
            if step < delay:
                gate = 0.0
            elif ramp > 0:
                gate = min(1.0, (step - delay + 1) / ramp)
            else:
                gate = 1.0
            lrs.append(base_lr * shape * gate)
        return lrs
