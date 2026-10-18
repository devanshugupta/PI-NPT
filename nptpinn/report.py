"""Figures: error against coefficient per method, and solution heatmaps."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def error_vs_coefficient(results: list[dict], path: Path, kind: str = "convection") -> None:
    plt = _plt()
    rows = [r for r in results if r["kind"] == kind]
    methods = sorted({r["method"] for r in rows})
    coeffs = sorted({r["coeff"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {"mlp-raw": ("#c9d3df", "x", ":"), "mlp": ("#9aa5b1", "o", "--"),
              "mlp+causal": ("#5b8dd6", "s", "-"), "mlp+causal+curriculum": ("#1f5fbf", "D", "-"),
              "npt": ("#e8a0aa", "v", "--"), "npt+causal": ("#d1495b", "^", "-")}
    for m in methods:
        ys = [next((r["rel_l2"] for r in rows if r["method"] == m and r["coeff"] == c), np.nan) for c in coeffs]
        color, marker, ls = styles.get(m, ("#333", "o", "-"))
        ax.plot(coeffs, ys, marker=marker, ls=ls, color=color, lw=2, label=m)
    ax.set_yscale("log")
    ax.set_xlabel({"convection": "beta (convection speed)", "diffusion": "nu", "reaction": "rho"}[kind])
    ax.set_ylabel("relative L2 error vs exact")
    ax.set_title(f"{kind}: test error by method")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def heatmaps(pred, true, err, path: Path, title: str) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    extent = [0, 1, 0, 2 * np.pi]
    for ax, field, name, cmap in zip(axes, (true, pred, err), ("exact", "prediction", "|error|"), ("viridis", "viridis", "magma")):
        im = ax.imshow(field, extent=extent, origin="lower", aspect="auto", cmap=cmap,
                       vmin=None if name == "|error|" else true.min(), vmax=None if name == "|error|" else true.max())
        ax.set_title(name)
        ax.set_xlabel("t")
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].set_ylabel("x")
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
