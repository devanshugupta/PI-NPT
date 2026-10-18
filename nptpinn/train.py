"""Physics-informed training.

Loss = lam_residual * MSE(PDE residual) [+ lam_initial * MSE(initial) when the initial
condition is soft] [+ MSE(observed data) when sparse observations are given]. Periodicity
in x holds exactly through the [sin x, cos x, t] features, so there is no boundary term.

Two things make a fast-moving solution learnable where a plain PINN collapses:

* A hard initial condition (`Solution(hard_initial=True)`): u = u0(x) + t * net, so the
  network cannot satisfy the loss by flattening u0 away.
* Causal weighting of the residual: collocation points are binned by t and bin m is
  weighted by exp(-eps * sum of the residual loss in earlier bins), detached. Later times
  only count once earlier times are already fit, so the solution is built forward from
  the initial condition instead of settling on a smooth average. eps is stepped up each
  time every weight exceeds `causal_tol`, until the schedule is exhausted.

Every step runs ONE forward over the concatenated point set with the collocation points
as the attention context. The NPT's output depends on that context, so evaluation chunks
by `context_size` to see the same kind of set.

A coefficient curriculum (`fit_curriculum`) warm-starts up a ladder of coefficients; it is
orthogonal to the two devices above and is kept so the benchmark can measure each."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .models import Solution
from .pde import COEFF, TWO_PI, Problem, sample_collocation

Data = tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # (x, t, u) observations


@dataclass
class TrainConfig:
    n_collocation: int = 2048
    n_initial: int = 256
    adam_steps: int = 8000
    stage_steps: int = 1500               # Adam steps for each intermediate curriculum stage
    lbfgs_steps: int = 300
    lr: float = 1e-3
    lam_residual: float = 1.0
    lam_initial: float = 1.0
    causal: bool = False
    causal_bins: int = 32
    causal_eps: tuple[float, ...] = (1e-2, 1e-1, 1.0, 10.0, 100.0)
    causal_tol: float = 0.99
    seed: int = 0


def context_size(cfg: TrainConfig) -> int:
    return cfg.n_collocation


class _Causal:
    """Tracks the eps schedule across steps."""

    def __init__(self, cfg: TrainConfig):
        self.cfg, self.i = cfg, 0

    def weighted(self, res2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        bins = (t.squeeze(-1) * cfg.causal_bins).long().clamp_(max=cfg.causal_bins - 1)
        per_bin = torch.zeros(cfg.causal_bins).index_add_(0, bins, res2.squeeze(-1))
        count = torch.zeros(cfg.causal_bins).index_add_(0, bins, torch.ones_like(res2.squeeze(-1)))
        per_bin = per_bin / count.clamp(min=1)
        earlier = torch.cumsum(per_bin, 0) - per_bin
        w = torch.exp(-cfg.causal_eps[self.i] * earlier).detach()
        if w.min() > cfg.causal_tol and self.i < len(cfg.causal_eps) - 1:
            self.i += 1
        return (w * per_bin).sum() / w.sum()


def _points(cfg: TrainConfig, gen: torch.Generator):
    x, t = sample_collocation(cfg.n_collocation, gen)
    x.requires_grad_(True), t.requires_grad_(True)
    x0 = torch.rand(cfg.n_initial, 1, generator=gen) * TWO_PI
    return x, t, x0


def _loss(sol: Solution, problem: Problem, cfg: TrainConfig, x, t, x0, data: Data | None,
          causal: _Causal | None = None) -> dict[str, torch.Tensor]:
    """Loss terms on a given point set; x, t must already require grad."""
    xs, ts = [x, x0], [t, torch.zeros_like(x0)]
    if data is not None:
        xs.append(data[0]), ts.append(data[1])
    u = sol(torch.cat(xs), torch.cat(ts), context=(x, t))
    n, n0 = x.shape[0], x0.shape[0]
    res2 = problem.residual(u[:n], x, t).pow(2)
    terms = {"residual": cfg.lam_residual * (causal.weighted(res2, t) if causal else res2.mean())}
    if not sol.hard_initial:
        terms["initial"] = cfg.lam_initial * (u[n:n + n0] - problem.initial(x0)).pow(2).mean()
    if data is not None:
        terms["data"] = (u[n + n0:] - data[2]).pow(2).mean()
    return terms


def loss_terms(sol: Solution, problem: Problem, cfg: TrainConfig, gen: torch.Generator,
               data: Data | None = None, causal: _Causal | None = None):
    """Loss terms on a fresh random point set."""
    return _loss(sol, problem, cfg, *_points(cfg, gen), data, causal)


def fit(sol: Solution, problem: Problem, cfg: TrainConfig, data: Data | None = None, log=None) -> dict[str, float]:
    """Train `sol` in place on `problem`; returns the final (unweighted) loss terms."""
    gen = torch.Generator().manual_seed(cfg.seed)
    params = list(sol.parameters())
    causal = _Causal(cfg) if cfg.causal else None
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.adam_steps)
    for step in range(cfg.adam_steps):
        opt.zero_grad()
        total = sum(loss_terms(sol, problem, cfg, gen, data, causal).values())
        total.backward(inputs=params)
        opt.step()
        sched.step()
        if log and step % 500 == 0:
            eps = f"  eps {cfg.causal_eps[causal.i]:g}" if causal else ""
            log(f"    adam {step:5d}  loss {total.item():.3e}{eps}")

    if cfg.lbfgs_steps:
        # L-BFGS line search assumes a deterministic objective: one fixed point set and the
        # plain (unweighted) residual. Tolerances sit at float32 resolution.
        fixed = _points(cfg, torch.Generator().manual_seed(cfg.seed + 1))
        lbfgs = torch.optim.LBFGS(params, max_iter=cfg.lbfgs_steps, history_size=50,
                                  tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            total = sum(_loss(sol, problem, cfg, *fixed, data).values())
            total.backward(inputs=params)
            return total

        lbfgs.step(closure)

    return {k: v.item() for k, v in loss_terms(sol, problem, cfg, gen, data).items()}


def curriculum_path(stages: list[float], final: float) -> list[float]:
    """Ascending warm-start ladder below `final`, ending at `final` itself."""
    return sorted(s for s in stages if s < final) + [final]


def fit_curriculum(sol: Solution, problem: Problem, cfg: TrainConfig, stages: list[float],
                   data: Data | None = None, log=None) -> dict[str, float]:
    """Warm-start through increasing coefficients, ending at `problem`'s own value. Only
    the final stage gets the full step budget and the L-BFGS polish."""
    coeff = COEFF[problem.kind]
    for i, c in enumerate(path := curriculum_path(stages, problem.coeff)):
        last = i == len(path) - 1
        stage_cfg = cfg if last else replace(cfg, lbfgs_steps=0, adam_steps=min(cfg.stage_steps, cfg.adam_steps))
        if log:
            log(f"  stage {coeff}={c:g}")
        terms = fit(sol, replace(problem, **{coeff: c}), stage_cfg, data, log)
    return terms
