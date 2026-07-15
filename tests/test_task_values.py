"""Value-correctness battery: losses pinned to known references.

The contract and optimizability suites prove each loss is well-behaved
and non-degenerate; this suite pins task losses to *known values* so a
silently wrong implementation (sign flip, dropped term, wrong scaling)
cannot pass. Per family:

* quadratic -- closed-form least-squares optimum via ``lstsq``,
* classification/regression -- exact loss at all-zero weights,
* BBOB/CEC2005 -- loss at the registry's optimum equals ``global_min``,
* PINNs with a closed-form solution -- substituting the exact solution
  for the network drives the physics-informed loss to ~0 (convection,
  reaction, wave, Helmholtz 2-D/3-D, viscous Burgers).

Families intentionally not covered here: inviscid Burgers, Euler and
Stokes have no closed-form solution (their operator helpers are pinned
in their own test modules); the PK-PD analytic concentration is pinned
in ``test_pkpd.py``; the stochastic BBOB/CEC cases have no fixed
reference value.

The empirical-``global_min`` tasks (gaussian classification, spiral,
MNIST-1D, gaussian-meta) have no closed-form minimum -- their
``global_min`` is benchmarked at creation -- so instead of a fixed value
this suite pins their *determinism*: rebuilding the task reproduces the
same ``global_min`` (and hence the same ``hash``), which is what keeps a
benchmarked value safe to put in ``Task`` identity.
"""

from __future__ import annotations

import bbob_jax
import equinox as eqx
import jax.nn
import jax.numpy as jnp
import jax.random as jr
import pytest

from ._contract_utils import evaluate_task_loss, fill_model, sample_model
from .task_cases import CASE_BY_ID

slow = pytest.mark.slow


# ---------------------------------------------------------------------------
# Quadratic: closed-form least-squares references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id", ["quadratic-square", "quadratic-overdetermined"]
)
def test_quadratic_loss_at_origin_is_scaled_y_norm(case_id, build_case):
    """At ``x = 0`` the loss is exactly ``scale * ||y||^2``."""
    task = build_case(case_id)
    y = task.loss_fn.keywords["y"]
    scale = task.loss_fn.keywords["scale"]
    loss = evaluate_task_loss(task, jnp.zeros_like(task.model))
    assert jnp.isclose(loss, scale * jnp.sum(y**2), rtol=1e-6)


@pytest.mark.parametrize(
    "case_id", ["quadratic-square", "quadratic-overdetermined"]
)
def test_quadratic_loss_at_lstsq_solution_is_global_min(case_id, build_case):
    """The loss at the pseudo-inverse solution equals ``global_min``.

    The factory computed ``global_min`` with this same expression
    (``quadratic_task.py``), so the overdetermined case must match to
    float precision; the square case attains exactly 0 up to float32
    ``lstsq`` round-off.
    """
    task = build_case(case_id)
    kw = task.loss_fn.keywords
    x_star, *_ = jnp.linalg.lstsq(kw["W"], kw["y"])
    loss = evaluate_task_loss(task, x_star)
    assert jnp.isclose(loss, task.global_min, rtol=1e-5, atol=1e-4)


# ---------------------------------------------------------------------------
# Classification / regression: exact loss at all-zero weights
# ---------------------------------------------------------------------------


@slow
def test_mnist1d_zero_weights_loss_is_log_num_classes(build_case):
    """Zero weights give uniform class scores -> CE == log(10)."""
    task = build_case("mnist1d")
    loss = evaluate_task_loss(task, fill_model(task, 0.0))
    assert jnp.isclose(loss, jnp.log(10.0), rtol=1e-5)


@slow
def test_gaussian_class_zero_weights_loss_is_log2(build_case):
    """Zero weights give uniform class scores and zero L2 -> log(2)."""
    task = build_case("gaussian-class")
    loss = evaluate_task_loss(task, fill_model(task, 0.0))
    assert jnp.isclose(loss, jnp.log(2.0), rtol=1e-5)


@slow
def test_spiral_zero_weights_loss_is_quarter(build_case):
    """Zero weights -> sigmoid readout 0.5 on half-0/half-1 labels.

    The GRU hidden state stays zero, the bias-free readout outputs 0
    and the sigmoid maps it to 0.5; the spiral labels are exactly half
    0 and half 1, so the MSE is exactly 0.25.
    """
    task = build_case("spiral")
    loss = evaluate_task_loss(task, fill_model(task, 0.0))
    assert jnp.isclose(loss, 0.25, rtol=1e-5)


# ---------------------------------------------------------------------------
# BBOB / CEC2005: loss at the registry optimum equals global_min
# ---------------------------------------------------------------------------
#
# The factories bake ``x_opt`` into the registry function via
# ``jax.tree_util.Partial``; rebuilding the registry entry with the
# case's exact ``(ndim, seed)`` recovers it. ``scale_input`` maps the
# unit box to the native bounds, so the unit-coordinate optimum is the
# inverse affine map of ``x_opt``. atol covers float32 round-off
# through the affine round-trip on poorly conditioned functions
# (rastrigin's alpha=10 scaling); f_opt itself can be O(100).

BBOB_BOUNDS = (-5.0, 5.0)
CEC_F1_BOUNDS = (-100.0, 100.0)


def _unit_coords(x_opt, bounds):
    """Map native coordinates back to the unit box."""
    lo, hi = bounds
    return (x_opt - lo) / (hi - lo)


@pytest.mark.parametrize(
    ("case_id", "fn_name"),
    [("bbob-sphere", "sphere"), ("bbob-rastrigin", "rastrigin")],
)
def test_bbob_loss_at_optimum_equals_global_min(case_id, fn_name, build_case):
    """The loss at the (inverse-scaled) BBOB optimum is ``global_min``."""
    task = build_case(case_id)
    fn, f_opt = bbob_jax.registry[fn_name](ndim=3, key=jr.key(0))
    x_opt = fn.keywords["x_opt"]
    assert jnp.isclose(fn(x_opt), f_opt, rtol=1e-6)  # registry sanity
    loss = evaluate_task_loss(task, _unit_coords(x_opt, BBOB_BOUNDS))
    assert jnp.isclose(loss, task.global_min, rtol=1e-4, atol=1e-2)


def test_cec2005_f1_loss_at_optimum_equals_global_min(build_case):
    """The loss at the (inverse-scaled) CEC2005 f1 optimum matches."""
    task = build_case("cec2005-f1")
    fn, f_opt = bbob_jax.cec2005_registry["f1"](ndim=3, key=jr.key(0))
    x_opt = fn.keywords["x_opt"]
    assert jnp.isclose(fn(x_opt), f_opt, rtol=1e-6)
    loss = evaluate_task_loss(task, _unit_coords(x_opt, CEC_F1_BOUNDS))
    assert jnp.isclose(loss, task.global_min, rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------------------
# Embedded BBOB: loss at the anchor point equals global_min
# ---------------------------------------------------------------------------
#
# The embedded factory anchors the offset so the optimum preimage is a
# seeded uniform draw ``x_star``, bound into the loss partial. Both the
# flat (bulk_scale=0) and curved null-space cases must attain
# ``global_min`` there exactly (the penalty vanishes at the anchor).


@pytest.mark.parametrize(
    "case_id", ["embedded-sphere", "embedded-rastrigin-bulk"]
)
def test_embedded_bbob_loss_at_anchor_equals_global_min(case_id, build_case):
    """The embedded loss attains ``global_min`` at the anchor point."""
    task = build_case(case_id)
    x_star = task.loss_fn.keywords["x_star"]
    loss = evaluate_task_loss(task, x_star)
    assert jnp.isclose(loss, task.global_min, rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------------------
# PINNs: substituting the exact solution drives the loss to ~0
# ---------------------------------------------------------------------------
#
# Every PINN loss calls the model as a generic per-point callable, so a
# module wrapping the closed-form solution can stand in for the
# network. Each test asserts an absolute tolerance (float32 second
# derivatives leave small residuals) plus a self-calibrating
# discriminance check: the exact solution must beat the untrained
# network by orders of magnitude.


class _ConvectionExact(eqx.Module):
    """Exact convection solution ``u(x, t) = sin(x - beta t)``."""

    beta: float = eqx.field(static=True)

    def __call__(self, xt):
        """Evaluate at one ``(x, t)`` point; returns shape ``(1,)``."""
        return jnp.sin(xt[0] - self.beta * xt[1])[None]


class _ReactionExact(eqx.Module):
    """Exact reaction (pointwise logistic ODE) solution.

    ``u(x, t) = h e^{rho t} / (1 - h + h e^{rho t})`` with the Gaussian
    initial condition ``h(x) = exp(-(x - pi)^2 / (2 (pi/4)^2))``.
    """

    rho: float = eqx.field(static=True)

    def __call__(self, xt):
        """Evaluate at one ``(x, t)`` point; returns shape ``(1,)``."""
        h = jnp.exp(-((xt[0] - jnp.pi) ** 2) / (2 * (jnp.pi / 4) ** 2))
        g = h * jnp.exp(self.rho * xt[1])
        return (g / (1.0 - h + g))[None]


class _WaveExact(eqx.Module):
    """Exact standing-wave solution of ``u_tt = 4 u_xx``.

    ``u = sin(pi x) cos(2 pi t) + 0.5 sin(beta pi x) cos(2 beta pi t)``
    matches the initial condition, zero initial velocity and the
    periodic-difference boundary term for integer ``beta``.
    """

    beta: float = eqx.field(static=True)

    def __call__(self, xt):
        """Evaluate at one ``(x, t)`` point; returns shape ``(1,)``."""
        x, t = xt[0], xt[1]
        u = jnp.sin(jnp.pi * x) * jnp.cos(2 * jnp.pi * t) + 0.5 * jnp.sin(
            self.beta * jnp.pi * x
        ) * jnp.cos(2 * self.beta * jnp.pi * t)
        return u[None]


class _HelmholtzExact(eqx.Module):
    """Exact Helmholtz solution ``u = prod_i sin(a_i pi x_i)``.

    The source ``q`` is manufactured from this solution, so the
    residual cancels analytically for any wavenumber ``k``.
    """

    a: tuple[float, ...] = eqx.field(static=True)

    def __call__(self, p):
        """Evaluate at one point; returns shape ``(1,)``."""
        return jnp.prod(jnp.sin(jnp.pi * jnp.asarray(self.a) * p))[None]


class _ViscousBurgersExact(eqx.Module):
    """Exact travelling-front solution of the viscous Burgers task.

    Same logistic form as ``viscous_burgers.exact_solution``:
    ``sigmoid(-(x + y - t) / (2 nu))``.
    """

    nu: float = eqx.field(static=True)

    def __call__(self, p):
        """Evaluate at one ``(x, y, t)`` point; returns shape ``(1,)``."""
        return jax.nn.sigmoid(-(p[0] + p[1] - p[2]) / (2.0 * self.nu))[None]


def _assert_exact_solution_minimizes(task, exact, atol):
    """Loss at the exact solution is ~0 and crushes an untrained net.

    The baseline network uses freshly sampled weights rather than
    ``task.model``, whose parameter values are arbitrary placeholders
    (only the static structure of ``task.model`` is contractual).
    """
    loss_exact = evaluate_task_loss(task, exact)
    untrained = sample_model(task, jr.key(24), "unbounded")
    loss_untrained = evaluate_task_loss(task, untrained)
    assert loss_exact < atol
    assert loss_exact < 1e-3 * loss_untrained


@slow
def test_convection_exact_solution_minimizes_loss(build_case):
    """``sin(x - beta t)`` drives the convection PINN loss to ~0."""
    task = build_case("pde-convection")
    exact = _ConvectionExact(beta=task.loss_fn.keywords["beta"])
    _assert_exact_solution_minimizes(task, exact, atol=1e-8)


@slow
def test_reaction_exact_solution_minimizes_loss(build_case):
    """The logistic closed form drives the reaction PINN loss to ~0."""
    task = build_case("pde-reaction")
    exact = _ReactionExact(rho=task.loss_fn.keywords["rho"])
    _assert_exact_solution_minimizes(task, exact, atol=1e-8)


@slow
def test_wave_exact_solution_minimizes_loss(build_case):
    """The standing-wave closed form drives the wave PINN loss to ~0."""
    task = build_case("pde-wave")
    exact = _WaveExact(beta=task.loss_fn.keywords["beta"])
    _assert_exact_solution_minimizes(task, exact, atol=1e-5)


@slow
@pytest.mark.parametrize(
    ("case_id", "atol"), [("helmholtz-2d", 1e-5), ("helmholtz-3d", 1e-4)]
)
def test_helmholtz_exact_solution_minimizes_loss(case_id, atol, build_case):
    """The manufactured solution drives the Helmholtz loss to ~0."""
    task = build_case(case_id)
    exact = _HelmholtzExact(a=task.loss_fn.keywords["a"])
    _assert_exact_solution_minimizes(task, exact, atol=atol)


@slow
def test_viscous_burgers_exact_solution_minimizes_loss(build_case):
    """The logistic front drives the viscous-Burgers loss to ~0."""
    task = build_case("viscous-burgers")
    exact = _ViscousBurgersExact(nu=task.loss_fn.keywords["nu"])
    _assert_exact_solution_minimizes(task, exact, atol=1e-4)


# ---------------------------------------------------------------------------
# Empirical global_min: deterministic, finite and non-negative
# ---------------------------------------------------------------------------


@slow
@pytest.mark.parametrize(
    "case_id", ["gaussian-class", "spiral", "mnist1d", "gaussian-meta"]
)
def test_empirical_global_min_is_deterministic(case_id, tmp_path):
    """Rebuilding an empirical task reproduces ``global_min`` exactly.

    The benchmark search that sets ``global_min`` is seeded, so two
    builds with identical arguments (and the same cached dataset) must
    yield the same float -- otherwise the value, which feeds
    ``tag_hashable``, would make the task's ``hash`` unstable.

    ``global_min`` must be non-negative (these are MSE / cross-entropy
    losses) but may be ``0``: the strengthened estimator's L-BFGS arm
    drives an over-parametrised tiny build (e.g. the MNIST-1D case, 1098
    parameters over 64 samples) to a perfect fit, a legitimate floor.
    """
    build = CASE_BY_ID[case_id].build
    task_a = build(tmp_path)
    task_b = build(tmp_path)
    assert task_a.global_min is not None
    assert jnp.isfinite(task_a.global_min)
    assert task_a.global_min >= 0.0
    assert task_a.global_min == task_b.global_min
    assert task_a.hash == task_b.hash
