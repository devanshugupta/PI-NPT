"""Test metrics against the closed-form solution on a dense grid.

Prediction is chunked and the grid is shuffled first: the NPT's output at a point depends on
the set it attends over, so evaluation feeds it random interior sets of the size it was
trained with (`train.context_size`) rather than one contiguous slab, and the full grid never
has to fit through attention at once."""
from __future__ import annotations

import torch

from .pde import Problem, grid


@torch.no_grad()
def predict(model, x: torch.Tensor, t: torch.Tensor, chunk: int, seed: int = 0) -> torch.Tensor:
    n = x.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    out = torch.empty(n, 1)
    for i in range(0, n, chunk):
        idx = perm[i:i + chunk]
        out[idx] = model(x[idx], t[idx])
    return out


def errors(model, problem: Problem, chunk: int, nx: int = 256, nt: int = 100) -> dict[str, float]:
    x, t = grid(nx, nt)
    true = problem.exact(x, t)
    diff = predict(model, x, t, chunk) - true
    return {
        "rel_l2": (diff.norm() / true.norm()).item(),
        "max_abs": diff.abs().max().item(),
        "mse": diff.pow(2).mean().item(),
    }


def fields(model, problem: Problem, chunk: int, nx: int = 256, nt: int = 100):
    """Prediction, truth and absolute error as (nx, nt) arrays for plotting."""
    x, t = grid(nx, nt)
    pred = predict(model, x, t, chunk).reshape(nx, nt)
    true = problem.exact(x, t).reshape(nx, nt)
    return pred.numpy(), true.numpy(), (pred - true).abs().numpy()
