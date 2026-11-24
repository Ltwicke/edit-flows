r"""Kappa schedules beyond upstream's linear and cubic ones.

In the token-wise mixture path a needed edit happens at a random time with distribution function $\kappa_t$, i.e.
with the hazard $\dot\kappa_t / (1 - \kappa_t)$. That hazard is the factor in front of every ideal rate (ideal
rate = hazard times the probability that the edit is still needed) *and* the weight of the log-rate term in the
loss, so the schedule decides how steep a function of $t$ the model has to learn and how unevenly the loss
weights the time points. Model, loss and sampler do not change with the schedule.
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Float
from torch import Tensor

from editflows.editflows import KappaSchedule


class ExponentialKappaSchedule(KappaSchedule):
    r"""$\kappa_t = (1 - e^{-ct}) / (1 - e^{-c})$: needed edits fire with a (nearly) constant hazard $c$.

    * `normalize=False`: $\kappa_t = 1 - e^{-ct}$, the hazard is exactly $c$ at every $t$. The path then ends at
      $\kappa_1 = 1 - e^{-c} < 1$: a share $e^{-c}$ of the edits is still open at $t = 1$.
    * `normalize=True`: $\kappa_1 = 1$ exactly. The hazard is $c / (1 - e^{-c(1-t)})$: close to $c$ for most of the
      interval and rising like $1 / (1 - t)$ at the very end, where $1 - \kappa_t$ is clamped as in the upstream
      schedules.

    With `normalize=True`, $c \to 0$ recovers the linear schedule and $c < 0$ gives slow-start schedules like the
    cubic one, so one parameter spans the family.

    `growth` $= g > 0$ lets the hazard rise as $c\,e^{gt}$: $\kappa_t = 1 - \exp(-c\,(e^{gt} - 1) / g)$ before the
    normalisation. Early times, where many edits are still open, then take a larger share of the interval, and
    the log-hazard stays linear in $t$.
    """

    def __init__(self, rate: float, normalize: bool = True, growth: float = 0.0):
        super().__init__()

        assert rate != 0, "rate 0 is the linear schedule, use LinearKappaSchedule"
        assert normalize or rate > 0, "without normalisation kappa is only increasing for a positive rate"
        assert growth == 0 or rate > 0, "a growing hazard needs a positive rate"
        self.rate = float(rate)
        self.normalize = normalize
        self.growth = float(growth)
        # kappa at t = 1 before the normalisation
        self.end = -math.expm1(-self.rate * (math.expm1(self.growth) / self.growth if self.growth else 1.0))

    def cumulative_hazard(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.rate * (torch.expm1(self.growth * t) / self.growth if self.growth else t)

    def forward(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        kappa = -torch.expm1(-self.cumulative_hazard(t))
        return kappa / self.end if self.normalize else kappa

    def loss_coefficient(self, t: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        hazard = self.rate * torch.exp(self.growth * t)
        if not self.normalize:
            return hazard
        dkappa = hazard * torch.exp(-self.cumulative_hazard(t)) / self.end
        return dkappa / (1 - self(t)).clamp(min=1e-3)
