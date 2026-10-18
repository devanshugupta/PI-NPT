"""The closed forms solve their PDEs, the models expose exactly the derivatives the residual
assumes, evaluation is faithful, and a short physics-only fit reaches low error."""
import pytest
import torch

from nptpinn import (MLP, NPT, Problem, Solution, TrainConfig, curriculum_path, errors, fields, fit, grid,
                     make, predict, sample_collocation)
from nptpinn.pde import TWO_PI


def test_exact_solutions_satisfy_their_pdes_and_are_periodic():
    gen = torch.Generator().manual_seed(0)
    for p in (Problem("convection", beta=20.0), Problem("diffusion", nu=2.0), Problem("reaction", rho=4.0)):
        x, t = sample_collocation(256, gen)
        x.requires_grad_(True), t.requires_grad_(True)
        assert p.residual(p.exact(x, t), x, t).abs().max() < 1e-3
        t0 = torch.rand(8, 1, generator=gen)
        assert torch.allclose(p.exact(torch.zeros(8, 1), t0), p.exact(torch.full((8, 1), TWO_PI), t0), atol=1e-4)


def test_reaction_exact_is_finite_for_large_rho():
    p = Problem("reaction", rho=100.0)
    assert torch.isfinite(p.exact(*grid(16, 16))).all()


def test_problem_rejects_mixed_coefficients():
    with pytest.raises(ValueError):
        Problem("convection", beta=1.0, nu=1.0)


def test_models_are_periodic_in_x_and_support_second_derivatives():
    x, t = torch.rand(16, 1, requires_grad=True), torch.rand(16, 1, requires_grad=True)
    for m in (MLP(), NPT(), make("mlp-raw"), make("npt-raw")):
        u = m(x, t)
        assert u.shape == (16, 1)
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        u_xx.pow(2).mean().backward()
        assert torch.isfinite(u_xx).all()
        with torch.no_grad():
            tt = torch.rand(8, 1)
            assert torch.allclose(m(torch.zeros(8, 1), tt), m(torch.full((8, 1), TWO_PI), tt), atol=1e-5)


def test_npt_input_jacobian_is_diagonal():
    """The residual reads du_i/dt_i from a vector-Jacobian product with ones, which equals the
    per-point partial only if no other point's output depends on t_i."""
    torch.manual_seed(0)
    m = NPT(d_model=16, n_heads=2, n_layers=1, ff=32)
    x = torch.rand(6, 1)
    t = torch.rand(6, 1, requires_grad=True)
    u = m(x, t)
    vjp = torch.autograd.grad(u, t, torch.ones_like(u))[0].squeeze(-1)
    jac = torch.autograd.functional.jacobian(lambda tv: m(x, tv.unsqueeze(-1)).squeeze(-1), t.squeeze(-1).detach())
    assert torch.allclose(jac - torch.diag(torch.diagonal(jac)), torch.zeros_like(jac), atol=1e-6)
    assert torch.allclose(vjp, torch.diagonal(jac), atol=1e-5)


def test_hard_initial_condition_holds_exactly():
    p = Problem("convection", beta=7.0)
    sol = Solution(MLP(), p.initial, hard_initial=True)
    x = torch.rand(32, 1) * TWO_PI
    assert torch.allclose(sol(x, torch.zeros_like(x)), p.initial(x), atol=1e-6)


def test_curriculum_path_is_sorted_and_ends_at_target():
    assert curriculum_path([30.0, 1.0, 10.0, 40.0], 20.0) == [1.0, 10.0, 20.0]
    assert curriculum_path([10.0, 20.0], 5.0) == [5.0]
    assert curriculum_path([1.0, 10.0], 10.0) == [1.0, 10.0]


def test_chunked_prediction_matches_full_and_covers_a_remainder():
    x, t = grid(13, 7)              # 91 points, not divisible by the chunk
    mlp = MLP()
    with torch.no_grad():
        assert torch.allclose(predict(mlp, x, t, chunk=20), mlp(x, t), atol=1e-6)
    out = predict(NPT(d_model=16, n_heads=2, n_layers=1, ff=32), *grid(256, 100), chunk=512)
    assert out.shape == (25600, 1) and torch.isfinite(out).all()


def test_fields_orientation_matches_grid_layout():
    p = Problem("convection", beta=3.0)
    exact = Solution(torch.nn.Identity(), p.initial, hard_initial=False)
    exact.forward = lambda x, t, context=None: p.exact(x, t)   # a perfect surrogate
    pred, true, err = fields(exact, p, chunk=512, nx=16, nt=8)
    xs = torch.linspace(0, TWO_PI, 17)[:-1]
    assert pred.shape == (16, 8) and err.max() < 1e-6
    assert torch.allclose(torch.tensor(true[:, 0]), torch.sin(xs), atol=1e-6)


def test_short_fit_learns_easy_convection():
    torch.manual_seed(0)
    p = Problem("convection", beta=1.0)
    sol = Solution(MLP(width=32, depth=3), p.initial, hard_initial=False)
    fit(sol, p, TrainConfig(n_collocation=512, adam_steps=1500, lbfgs_steps=100))
    assert errors(sol, p, chunk=512)["rel_l2"] < 0.05


def test_causal_recipe_trains_and_reports_finite_terms():
    """Hard initial condition + causal weighting: the loss has no initial term and the
    weighted residual stays finite as the eps schedule advances."""
    torch.manual_seed(0)
    p = Problem("convection", beta=5.0)
    sol = Solution(MLP(width=32, depth=3), p.initial, hard_initial=True)
    terms = fit(sol, p, TrainConfig(n_collocation=256, adam_steps=200, lbfgs_steps=0, causal=True))
    assert set(terms) == {"residual"} and all(torch.isfinite(torch.tensor(v)) for v in terms.values())
