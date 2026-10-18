"""The PDE family and its closed-form solutions.

    u_t + beta * u_x - nu * u_xx - rho * u * (1 - u) = 0,   x in [0, 2pi], t in [0, 1]

with periodic boundary and a prescribed initial condition. Pure convection, pure diffusion
and pure reaction each have an exact solution, so test error is measured against the truth
rather than against another solver. The hard regime is convection with large beta: the
solution is a fast-travelling wave and a PINN trained directly on it collapses to a trivial
fit, which is the failure this project is about."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TWO_PI = 2.0 * np.pi
COEFF = {"convection": "beta", "diffusion": "nu", "reaction": "rho"}


@dataclass(frozen=True)
class Problem:
    kind: str            # "convection" | "diffusion" | "reaction"
    beta: float = 0.0
    nu: float = 0.0
    rho: float = 0.0

    def __post_init__(self):
        # The closed forms hold for one mechanism at a time; a stray second coefficient
        # would enter the residual but not the "exact" solution it is scored against.
        if self.kind not in COEFF:
            raise ValueError(f"unknown kind {self.kind!r}")
        others = [n for n in COEFF.values() if n != COEFF[self.kind] and getattr(self, n) != 0.0]
        if others:
            raise ValueError(f"{self.kind} admits only {COEFF[self.kind]}; got nonzero {others}")

    @property
    def coeff(self) -> float:
        return getattr(self, COEFF[self.kind])

    @property
    def name(self) -> str:
        return f"{self.kind}-{self.coeff:g}"

    def initial(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "reaction":   # a Gaussian bump, as in the reaction benchmark
            return torch.exp(-((x - np.pi) ** 2) / (2 * (np.pi / 4) ** 2))
        return torch.sin(x)

    def exact(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.kind == "convection":
            return torch.sin(x - self.beta * t)
        if self.kind == "diffusion":
            return torch.exp(-self.nu * t) * torch.sin(x)
        # logistic growth pointwise in x, in the form that stays finite for large rho*t
        u0 = self.initial(x)
        return u0 / (u0 + (1 - u0) * torch.exp(-self.rho * t))

    def residual(self, u: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """PDE residual at points where u was computed from (x, t) with grad enabled. Valid
        only when du_i/dx_j vanishes for i != j, which both models guarantee."""
        g = lambda y, v: torch.autograd.grad(y, v, torch.ones_like(y), create_graph=True)[0]
        u_t, u_x = g(u, t), g(u, x)
        r = u_t + self.beta * u_x - self.rho * u * (1 - u)
        if self.nu:
            r = r - self.nu * g(u_x, x)
        return r


def grid(nx: int = 256, nt: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluation grid over the whole domain, flattened to column vectors (x-major)."""
    x = torch.linspace(0, TWO_PI, nx + 1)[:-1]
    t = torch.linspace(0, 1, nt)
    X, T = torch.meshgrid(x, t, indexing="ij")
    return X.reshape(-1, 1), T.reshape(-1, 1)


def sample_collocation(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand(n, 1, generator=gen) * TWO_PI
    t = torch.rand(n, 1, generator=gen)
    return x, t
