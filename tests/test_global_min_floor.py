"""Floor battery: ``global_min`` is a true lower bound on the loss.

``global_min`` is not merely informational. ``l2co``'s
``shift_to_quality_value`` subtracts it to form the *quality value*, and
the downstream log-scaling (``min_custom``) takes ``log10`` of the
result -- so a loss *below* ``global_min`` produces negative quality
values and silently corrupts the scaling. ``global_min`` is also baked
into ``Task`` identity (``tag_hashable`` / ``hash``). It must therefore
be a genuine lower bound on the achievable loss -- a **floor**.

The three mechanisms carry different risk (see ``CONTEXT.md`` and
``docs/adr/0001-global-min-is-a-floor-validated-against-all.md``):

* **empirical** (gaussian-class, spiral, MNIST-1D, gaussian-meta) --
  ``global_min`` is a benchmarked estimate (``estimate_global_min``), not
  a guaranteed bound. The at-risk class, covered in full.
* **analytical** (BBOB / CEC2005) -- a true closed-form bound; a
  representative sample acts as a ``scale_input``-bug tripwire (the full
  grid is redundant once the affine map is correct).
* **theoretical** (PINNs) -- residual-MSE ``>= 0``, so ``global_min = 0``
  is mathematically unbreakable; a sample guards against a numerical
  artifact.

Two tiers:

* a **fast, deterministic tripwire** (this module's default tests) that
  runs the strongest deterministic search -- L-BFGS for unbounded
  weight-space tasks, projected multi-restart Adam for unit-box tasks --
  harder than the factories' estimate budget, and asserts it never dips
  below ``global_min``;
* a **slow, ``requires_l2co`` portfolio test** that drives the full
  ``all`` optimizer suite (~60 optimizers) through the real ``l2co``
  rollout at production budget and asserts the same.

The fast helpers re-implement the search (L-BFGS / projected Adam)
*independently* of ``estimate_global_min`` -- deliberately, so the guard
is not tautological against the very function that sets ``global_min``:
an estimator bug shows up as a breach here rather than being reproduced.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import pytest

from ._contract_utils import evaluate_task_loss, sample_model
from .task_cases import CASE_BY_ID

slow = pytest.mark.slow

# Empirical: global_min is a benchmarked estimate -> the at-risk class.
EMPIRICAL_IDS = ["gaussian-class", "spiral", "mnist1d", "gaussian-meta"]
# Analytical sample: true closed-form bound; scale_input-bug tripwire.
ANALYTICAL_SAMPLE_IDS = [
    "bbob-sphere",
    "bbob-rastrigin",
    "cec2005-f1",
    "cec2017-f1",
]
# Theoretical sample: residual-MSE >= 0, so global_min = 0 is unbreakable.
THEORETICAL_SAMPLE_IDS = ["pde-convection", "helmholtz-2d", "viscous-burgers"]

# Breach thresholds per class.
EMPIRICAL_RTOL, EMPIRICAL_ATOL = 1e-3, 1e-6
ANALYTICAL_ATOL = 1e-2
THEORETICAL_ATOL = 1e-6

# Fast-tier search budget. Deliberately stronger than the factories'
# estimate budget so a too-weak global_min is *caught*, not masked.
_FAST_RESTARTS = 6
_FAST_STEPS = 300

# Slow-tier portfolio settings. Budget = optimizer iterations (l2co's
# budget_fn returns n_iterations directly); mirrors the production
# per-family budgets. Realizations kept modest for the opt-in guard;
# the one-off study (docs/optimizer_sanity.ipynb) uses ~16.
_PRODUCTION_BUDGET = {
    "gaussian-class": 100,
    "spiral": 1000,
    "mnist1d": 1000,
    "gaussian-meta": 1000,
}
_PORTFOLIO_REALIZATIONS = 4


# ---------------------------------------------------------------------------
# Fast-tier deterministic search
# ---------------------------------------------------------------------------


def _finite(loss: jax.Array) -> jax.Array:
    """Map a non-finite loss to ``+inf`` so it loses every ``minimum``."""
    return jnp.where(jnp.isfinite(loss), loss, jnp.inf)


def _resample(model, key, clip_to_unit):
    """``model`` with inexact leaves resampled (uniform unit / normal)."""
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jtu.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    pairs = zip(leaves, keys, strict=True)
    if clip_to_unit:
        new = [jr.uniform(k, x.shape, x.dtype) for x, k in pairs]
    else:
        new = [jr.normal(k, x.shape, x.dtype) for x, k in pairs]
    return eqx.combine(jtu.tree_unflatten(treedef, new), static)


def _loss_closure(task, dataset):
    """Pure ``params -> scalar`` loss (fixed eval key neutralises rng)."""
    key0 = jr.key(0)

    def loss_at(params, static):
        """Loss of the model assembled from ``params``."""
        model = eqx.combine(params, static)
        if task.pass_rng:
            out = task.loss_fn(model, key=key0, **dataset)
        else:
            out = task.loss_fn(model, **dataset)
        return out[0] if task.has_aux else out

    return loss_at


def _lbfgs_best(task, start, dataset, n_steps):
    """Best loss found by deterministic L-BFGS from ``start`` (unbounded)."""
    loss_at = _loss_closure(task, dataset)
    params, static = eqx.partition(start, eqx.is_inexact_array)

    def f(p):
        """Scalar loss at trainable params ``p``."""
        return loss_at(p, static)

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(f)

    @eqx.filter_jit
    def run(params):
        """Run ``n_steps`` of L-BFGS; return the best loss seen."""
        state = opt.init(params)

        def body(carry, _):
            params, state, best = carry
            val, grad = value_and_grad(params, state=state)
            upd, state = opt.update(
                grad, state, params, value=val, grad=grad, value_fn=f
            )
            params = optax.apply_updates(params, upd)
            return (params, state, jnp.minimum(best, _finite(val))), None

        init = (params, state, jnp.asarray(jnp.inf))
        (params, _, best), _ = jax.lax.scan(body, init, None, length=n_steps)
        return jnp.minimum(best, _finite(f(params)))

    return run(params)


def _adam_best(task, start, dataset, n_steps):
    """Best loss found by deterministic projected Adam (unit box)."""
    loss_at = _loss_closure(task, dataset)
    params, static = eqx.partition(start, eqx.is_inexact_array)
    opt = optax.adam(optax.cosine_decay_schedule(1e-2, n_steps))

    @eqx.filter_jit
    def run(params):
        """Run ``n_steps`` of projected Adam; return the best loss seen."""
        state = opt.init(params)

        def body(carry, _):
            params, state, best = carry
            loss, grads = eqx.filter_value_and_grad(
                lambda p: loss_at(p, static)
            )(params)
            upd, state = opt.update(grads, state, params)
            params = eqx.apply_updates(params, upd)
            params = jtu.tree_map(lambda x: jnp.clip(x, 0.0, 1.0), params)
            return (params, state, jnp.minimum(best, _finite(loss))), None

        init = (params, state, jnp.asarray(jnp.inf))
        (params, _, best), _ = jax.lax.scan(body, init, None, length=n_steps)
        return jnp.minimum(best, _finite(loss_at(params, static)))

    return run(params)


def _strong_best(task, case, *, n_restarts, n_steps, seed=0):
    """Best loss over a multi-restart strong deterministic search.

    Unit-box tasks use projected Adam (and resample every restart from
    inside the box); unbounded weight-space tasks use L-BFGS. The first
    unbounded restart starts at ``task.model``; all unit restarts and the
    remaining unbounded restarts start from resampled leaves.
    """
    dataset = {k: jnp.asarray(v) for k, v in task.loaded_dataset.items()}
    clip = case.domain == "unit"
    restart_keys = jr.split(jr.key(seed), n_restarts)
    best = jnp.asarray(jnp.inf)
    for r in range(n_restarts):
        if r == 0 and not clip:
            start = task.model
        else:
            start = _resample(task.model, restart_keys[r], clip)
        if clip:
            b = _adam_best(task, start, dataset, n_steps)
        else:
            b = _lbfgs_best(task, start, dataset, n_steps)
        best = jnp.minimum(best, b)
    return float(best)


def _breach_msg(case_id, best, gmin, who):
    """Assertion message describing a floor breach and its margin."""
    return (
        f"{case_id}: {who} reached {best:.6g} < global_min {gmin:.6g} "
        f"(breach margin {gmin - best:.3g}) -- global_min is not a floor"
    )


# ---------------------------------------------------------------------------
# Fast tier: strongest deterministic search never breaches global_min
# ---------------------------------------------------------------------------


@slow
@pytest.mark.parametrize("case_id", EMPIRICAL_IDS)
def test_empirical_global_min_is_a_floor(case_id, build_case):
    """A strong deterministic search does not dip below ``global_min``.

    The empirical ``global_min`` is a benchmarked estimate; this asserts
    it is nonetheless a lower bound for a search stronger than the
    factory's (L-BFGS for the weight-space tasks, many-restart projected
    Adam for the unit-box meta task).
    """
    task = build_case(case_id)
    assert task.global_min is not None
    best = _strong_best(
        task,
        CASE_BY_ID[case_id],
        n_restarts=_FAST_RESTARTS,
        n_steps=_FAST_STEPS,
    )
    tol = EMPIRICAL_RTOL * abs(task.global_min) + EMPIRICAL_ATOL
    assert best >= task.global_min - tol, _breach_msg(
        case_id, best, task.global_min, "strong search"
    )


@pytest.mark.parametrize("case_id", ANALYTICAL_SAMPLE_IDS)
def test_analytical_global_min_is_a_floor(case_id, build_case):
    """No in-box search point falls below the closed-form ``global_min``.

    A multimodal function may leave the search stuck *above* the optimum
    (that is not a breach); the test only fails if a point is found
    *below* ``global_min``, which would indicate a ``scale_input`` bug.
    """
    task = build_case(case_id)
    assert task.global_min is not None
    best = _strong_best(
        task,
        CASE_BY_ID[case_id],
        n_restarts=_FAST_RESTARTS,
        n_steps=_FAST_STEPS,
    )
    assert best >= task.global_min - ANALYTICAL_ATOL, _breach_msg(
        case_id, best, task.global_min, "in-box search"
    )


@slow
@pytest.mark.parametrize("case_id", THEORETICAL_SAMPLE_IDS)
def test_theoretical_global_min_is_nonnegative_floor(case_id, build_case):
    """The PINN residual loss is ``>= 0`` and ``global_min`` is ``0``."""
    task = build_case(case_id)
    case = CASE_BY_ID[case_id]
    losses = [
        float(
            evaluate_task_loss(
                task, sample_model(task, jr.key(i), case.domain)
            )
        )
        for i in range(3)
    ]
    assert all(loss >= -THEORETICAL_ATOL for loss in losses), (
        f"{case_id}: sampled residual loss is negative: {losses}"
    )
    best = _strong_best(task, case, n_restarts=2, n_steps=_FAST_STEPS)
    assert best >= -THEORETICAL_ATOL, (
        f"{case_id}: search drove the residual MSE below 0: {best:.3g}"
    )
    assert task.global_min == 0.0, (
        f"{case_id}: theoretical global_min should be 0, got {task.global_min}"
    )


# ---------------------------------------------------------------------------
# Slow tier: the full `all` optimizer suite never breaches global_min
# ---------------------------------------------------------------------------


@pytest.mark.requires_l2co
@slow
@pytest.mark.parametrize("case_id", EMPIRICAL_IDS)
def test_empirical_global_min_floor_against_all(case_id, build_case):
    """The full ``all`` suite at production budget never breaches.

    Drives every optimizer in ``l2co``'s registry through the real
    rollout at the family's production budget (= iterations) and asserts
    none reaches a loss below ``global_min``. An optimizer that *errors*
    on a task is skipped -- it cannot establish a floor breach. Fixed
    keys make the sweep reproducible.
    """
    pytest.importorskip("l2co")
    from l2co import OptimizationStep, RolloutWrapper
    from l2co.optimizers import optimizers as all_optimizers
    from l2co.sampling import normal_sampling, random_sampling

    task = build_case(case_id)
    assert task.global_min is not None
    unit = CASE_BY_ID[case_id].domain == "unit"
    bounded = (0.0, 1.0) if unit else (None, None)
    sampler = random_sampling if unit else normal_sampling
    n_iterations = _PRODUCTION_BUDGET[case_id]

    best, worst_opt = jnp.inf, None
    for name in sorted(all_optimizers):
        try:
            rollout = RolloutWrapper.init(
                optimizer=OptimizationStep(optimizer=name),
                task=task,
                sampler=sampler,
                key=jr.key(0),
                bounded=bounded,
            )
            _, _, history = rollout.batch_evaluate(
                key=jr.key(1),
                n_iterations=n_iterations,
                n_realizations=_PORTFOLIO_REALIZATIONS,
            )
            reached = float(jnp.min(history.output_min))
        except Exception:
            continue
        if reached < best:
            best, worst_opt = reached, name

    tol = EMPIRICAL_RTOL * abs(task.global_min) + EMPIRICAL_ATOL
    assert best >= task.global_min - tol, _breach_msg(
        case_id, best, task.global_min, f"portfolio optimizer {worst_opt!r}"
    )
