"""Two surrogates for u(x, t).

`MLP` is the standard PINN network: each point is mapped independently. `NPT` follows the
non-parametric-transformer idea of attending *between datapoints*: the batch of points is
embedded and self-attention mixes information across it, then each point reads out u by
cross-attending from its own embedding into that context.

The context is built from DETACHED inputs and a point's own (x, t) enters only through its
query. That keeps the input Jacobian diagonal: du_i/dx_j = 0 for i != j, so the autograd
derivatives the PDE residual uses are the per-point partials and not column sums that mix
in other points. Parameters still receive gradients through the context path.

Both map (x, t) through [sin x, cos x, t, sin 2*pi*k*t, cos 2*pi*k*t for k <= T_FREQS] first.
The x pair makes periodicity exact by construction, so no boundary loss is needed; the t
harmonics let a tanh network represent a solution that oscillates many times in t (a
convection wave at speed beta crosses the domain beta / 2pi times), which raw t cannot
express within a practical step budget."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


T_FREQS = 8          # harmonics in t; a convection wave at speed beta needs about beta / 2pi of them
HARMONIC_INIT = 0.1  # the first layer starts nearly blind to the harmonics and recruits them as needed,
                     # so a slow solution is not fit through high frequencies it does not have


def _features(x: torch.Tensor, t: torch.Tensor, t_freqs: int) -> torch.Tensor:
    parts = [torch.sin(x), torch.cos(x), t]
    if t_freqs:
        k = torch.arange(1, t_freqs + 1, dtype=t.dtype) * (2 * torch.pi)
        parts += [torch.sin(t * k), torch.cos(t * k)]
    return torch.cat(parts, dim=-1)


def _in_dim(t_freqs: int) -> int:
    return 3 + 2 * t_freqs


def _damp_harmonics(first: nn.Linear) -> nn.Linear:
    with torch.no_grad():
        first.weight[:, 3:] *= HARMONIC_INIT
    return first


class MLP(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4, t_freqs: int = T_FREQS):
        super().__init__()
        self.t_freqs = t_freqs
        layers: list[nn.Module] = [_damp_harmonics(nn.Linear(_in_dim(t_freqs), width)), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        return self.net(_features(x, t, self.t_freqs))


class NPT(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, ff: int = 128, t_freqs: int = T_FREQS):
        super().__init__()
        self.t_freqs = t_freqs
        self.embed = nn.Sequential(_damp_harmonics(nn.Linear(_in_dim(t_freqs), d_model)), nn.Tanh(),
                                   nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        """`context` is the (x, t) set the batch attends over; by default the batch itself."""
        cx, ct = context if context is not None else (x, t)
        q = self.embed(_features(x, t, self.t_freqs)).unsqueeze(0)                        # own point, differentiable
        ctx = self.embed(_features(cx.detach(), ct.detach(), self.t_freqs)).unsqueeze(0)  # the set, constant w.r.t. inputs
        # The residual differentiates u twice with respect to the inputs; only the math
        # attention kernel implements that double backward, the fused ones do not.
        with sdpa_kernel(SDPBackend.MATH):
            ctx = self.encoder(ctx)
            mixed, _ = self.cross(q, ctx, ctx, need_weights=False)
        return self.head(self.norm(q + mixed).squeeze(0))


def make(name: str) -> nn.Module:
    """`mlp` / `npt`, or `mlp-raw` / `npt-raw` for the same network fed raw t (no harmonics)."""
    arch, _, variant = name.partition("-")
    cls = {"mlp": MLP, "npt": NPT}[arch]
    return cls(t_freqs=0) if variant == "raw" else cls()


class Solution(nn.Module):
    """The surrogate as the solver sees it. With `hard_initial`, u = u0(x) + t * net(x, t):
    the initial condition then holds exactly and only the residual has to be learned."""

    def __init__(self, net: nn.Module, initial, hard_initial: bool = True):
        super().__init__()
        self.net, self.initial, self.hard_initial = net, initial, hard_initial

    def forward(self, x: torch.Tensor, t: torch.Tensor, context=None) -> torch.Tensor:
        out = self.net(x, t, context)
        return self.initial(x) + t * out if self.hard_initial else out
