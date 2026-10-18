"""Run the benchmark grid and write results + figures.

    python -m nptpinn.experiments                       # convection grid (CPU, tens of minutes)
    python -m nptpinn.experiments --kind reaction --coeffs 1 5 10
    python -m nptpinn.experiments --quick               # smoke run into results/quick

Method names are an architecture plus flags: `mlp`, `npt`, or `mlp-raw` / `npt-raw` (raw t,
no harmonics: the textbook PINN input); `+causal` adds the hard initial condition and
causal residual weighting; `+curriculum` warm-starts up the ladder of requested
coefficients below the target. Flags combine: `mlp+causal+curriculum`.
Every number in results.json is the relative L2 error of the trained model against the
closed-form solution on a 256 x 100 grid. Rows already present for (kind, coeff, method)
are skipped, so a run can be resumed or extended."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import models
from .evaluate import errors, fields
from .pde import COEFF, Problem
from .report import error_vs_coefficient, heatmaps
from .train import TrainConfig, context_size, fit, fit_curriculum

DEFAULT_METHODS = ["mlp-raw", "mlp", "mlp+causal", "mlp+causal+curriculum", "npt", "npt+causal"]

NPT_POINTS = 512   # attention is quadratic in the context set; the MLP keeps the full set


def run_one(kind: str, coeff: float, method: str, stages: list[float], cfg: TrainConfig, out: Path, log) -> dict:
    torch.manual_seed(cfg.seed)
    problem = Problem(kind, **{COEFF[kind]: coeff})
    arch, *flags = method.split("+")
    causal = "causal" in flags
    model = models.Solution(models.make(arch), problem.initial, hard_initial=causal)
    cfg = replace(cfg, causal=causal)
    if arch.startswith("npt"):
        cfg = replace(cfg, n_collocation=min(cfg.n_collocation, NPT_POINTS))
    t0 = time.time()
    if "curriculum" in flags:
        terms = fit_curriculum(model, problem, cfg, stages, log=log)
    else:
        terms = fit(model, problem, cfg, log=log)
    chunk = context_size(cfg)
    metrics = errors(model, problem, chunk)
    pred, true, err = fields(model, problem, chunk)
    heatmaps(pred, true, err, out / f"{problem.name}_{method}.png",
             f"{problem.name}  {method}   rel L2 = {metrics['rel_l2']:.2e}")
    row = {"kind": kind, "coeff": coeff, "method": method, **metrics, "loss": terms, "config": asdict(cfg),
           "seconds": round(time.time() - t0, 1), "params": sum(p.numel() for p in model.parameters())}
    log(f"  -> rel L2 {metrics['rel_l2']:.3e}  max|err| {metrics['max_abs']:.3e}  ({row['seconds']}s)")
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kind", default="convection", choices=list(COEFF))
    p.add_argument("--coeffs", type=float, nargs="+", default=[1, 10, 20, 30, 40])
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    p.add_argument("--out", type=Path, default=None, help="default results/, or results/quick with --quick")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    cfg = TrainConfig(seed=a.seed)
    if a.quick:
        cfg = replace(cfg, n_collocation=512, adam_steps=400, stage_steps=200, lbfgs_steps=50)
        a.coeffs = a.coeffs[:2]
    out = a.out or Path("results/quick" if a.quick else "results")
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    log = lambda s: print(s, flush=True)
    stages = sorted(a.coeffs)   # the ladder is the requested coefficients themselves

    for coeff in a.coeffs:
        for method in a.methods:
            if any(r["kind"] == a.kind and r["coeff"] == coeff and r["method"] == method for r in results):
                continue
            log(f"{a.kind} {COEFF[a.kind]}={coeff:g}  [{method}]")
            results.append(run_one(a.kind, coeff, method, stages, cfg, out, log))
            results_path.write_text(json.dumps(results, indent=1))

    error_vs_coefficient(results, out / f"error_vs_{COEFF[a.kind]}.png", a.kind)
    log("\nrel L2 error vs exact")
    for r in sorted((r for r in results if r["kind"] == a.kind), key=lambda r: (r["coeff"], r["method"])):
        log(f"  {COEFF[a.kind]}={r['coeff']:<5g} {r['method']:16} {r['rel_l2']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
