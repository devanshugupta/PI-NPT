# Handoff: extend the PI-NPT experiments on a GPU (SageMaker)

For an agent or person picking this up on a GPU box. Everything needed to reproduce, extend
and report back is here. Read the whole file before running anything.

## What this repo is

Physics-informed training of two surrogates for u(x, t) on

    u_t + beta*u_x - nu*u_xx - rho*u*(1-u) = 0,   x in [0, 2pi], t in [0, 1], periodic in x

`mlp` is a plain PINN; `npt` attends *between datapoints* (context from detached inputs,
own point through a cross-attention query, so the input Jacobian is diagonal and the
residual is the true PDE operator). Closed forms exist for pure convection / diffusion /
reaction, so **every reported error is relative L2 against the exact solution on a
256 x 100 grid.**

Method names = architecture + flags: `mlp`, `npt`, `mlp-raw`, `npt-raw` (raw t, no
harmonics); `+causal` = hard initial condition + causal residual weighting; `+curriculum` =
warm-start up the sorted ladder of requested coefficients. Code map: `nptpinn/pde.py`,
`models.py`, `train.py`, `evaluate.py`, `report.py`, `experiments.py` (resumable grid
runner). `pytest tests -q` must pass (11 tests) before any run.

## State at handoff (CPU, seed 0, 8000 Adam + 300 L-BFGS)

The full grid is DONE and committed: `results/results.json` (39 rows) and the README
tables. Headlines: textbook PINN collapses at beta >= 20 (0.72 to 1.09);
`mlp+causal+curriculum` reaches 1.1e-2 at beta = 40; reaction rho = 10 goes 0.99 -> 6.7e-2;
the NPT wins at beta = 1 (6.7e-4) and rho = 5 (3.7e-2) but does not beat the MLP at high
beta under this budget. **Do not redo those rows**; the runner skips existing
(kind, coeff, method) cells.

## What the GPU is for (priority order)

1. **Apples-to-apples NPT.** The NPT attended over 512 points against the MLP's 2048
   (`NPT_POINTS` in `experiments.py`). Set it to 2048 and rerun `npt` and `npt+causal` for
   every coefficient. Until this is done the NPT-vs-MLP comparison is confounded.
2. **NPT with the curriculum.** The MLP's best recipe is `mlp+causal+curriculum`; the NPT
   never got the curriculum. Run `npt+causal+curriculum` for all coefficients.
3. **Seeds.** Seeds 0, 1, 2 for every cell (`--seed N`, one `--out results/seedN` each).
   Report mean +- std. One seed is not a result.
4. **Larger beta.** Extend to beta in {50, 60}. The input has `T_FREQS = 8` harmonics in t,
   enough for beta up to ~50 (needs about beta / 2pi); raise `T_FREQS` to 12 for 60 and
   say so in the write-up. Include 50 and 60 in `--coeffs` so the ladder reaches them.
5. **The NPT thesis: sparse observations.** `fit(..., data=(xd, td, ud))` already accepts
   observed points. Add `--data-points N` sampling N points of the exact solution
   (N in {0, 32, 128, 512}) as a data loss. Hypothesis: the NPT, which can attend to the
   observations, should beat the MLP by a widening margin as N grows at equal physics loss.
   If it does not, say so; that is a finding.
6. **Reaction-diffusion** (nu and rho both nonzero). No closed form: add a high-resolution
   finite-difference reference (Fourier in x, RK4 in t, 512 x 2000) in `pde.py`, allow mixed
   coefficients for that kind only, and label rows "vs FD reference".

Ablations if time remains: `T_FREQS = 0` with `+causal` (how much the harmonics carry);
`HARMONIC_INIT` in {1, 0.1, 0.01}; `lbfgs_steps = 0`; `lam_residual` in {0.1, 1, 10};
`causal_bins` in {16, 32, 64}.

## Environment

`ml.g5.xlarge` (A10G) is ample; `ml.g4dn.xlarge` works. Studio terminal or a notebook
instance with a PyTorch 2.x GPU kernel.

```bash
git clone https://github.com/devanshugupta/PI-NPT.git && cd PI-NPT
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests -q
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

GPU notes. (a) The NPT forward forces the math attention backend
(`sdpa_kernel(SDPBackend.MATH)`) because the residual needs double backward through
attention; leave it, it is correct on CUDA. (b) Tensors are created on CPU: add a
`--device` option that moves the model and the sampled points in `train.py` /
`evaluate.py`, and confirm with `nvidia-smi` that kernels run on the GPU. (c) Keep
float32; do not switch to bf16/fp16, second derivatives lose too much precision.

## Commands

```bash
# 1 + 2: fair NPT comparison (after NPT_POINTS = 2048)
python -m nptpinn.experiments --kind convection --coeffs 1 10 20 30 40 \
       --methods npt npt+causal npt+causal+curriculum --out results/seed0 --seed 0
# 3: seeds 1 and 2 for every method
python -m nptpinn.experiments --kind convection --coeffs 1 10 20 30 40 --out results/seed1 --seed 1
python -m nptpinn.experiments --kind convection --coeffs 1 10 20 30 40 --out results/seed2 --seed 2
# 4: larger beta (after T_FREQS = 12)
python -m nptpinn.experiments --kind convection --coeffs 1 10 20 30 40 50 60 --out results/large_beta
# 5: sparse data, once --data-points exists
python -m nptpinn.experiments --kind convection --coeffs 30 \
       --methods mlp+causal npt+causal --data-points 0 32 128 512 --out results/data
```

## What to check and report

Write `docs/RESULTS_GPU.md` with tables and figures. Answer:

- **Fair NPT vs MLP** at 2048 points, same recipe, per coefficient. If the NPT is not
  better, that is the result.
- **Does the curriculum help the NPT** as much as it helped the MLP?
- **Seed variance**: mean +- std over 3 seeds; flag cells where std ~ mean.
- **Sparse data**: rel L2 vs N observed points, both models, one plot.
- **Cost**: seconds per run and parameters per model.

Sanity before trusting a number: tests pass; a "good" heatmap has error in a thin band
along the characteristics, not a smooth blob (a blob means it learned an average, not the
wave); `loss.residual` small AND the prediction matches the exact at t = 0 (hard IC makes
this automatic; for soft-IC methods check `loss.initial`).

## Rules

- **Never commit credentials, AWS keys or tokens.** The repo reads nothing from the
  environment; SageMaker's execution role handles AWS access.
- Commit results under `results/` and the write-up under `docs/`. Author: Devanshu Gupta
  <devanshu0gupta@gmail.com>. **No Co-Authored-By trailer** on any commit.
- Report numbers as measured. A failed cell with a reason is a result; a rounded one is not.
- Comments and docs state the reason for a rule, never the incident that prompted it.
