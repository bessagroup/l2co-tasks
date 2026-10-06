"""Integration battery: tasks driven through the real l2co consumer.

The contract suite exercises the canonical loss-invocation contract via
a test-side replica (``_contract_utils.py``); this suite closes the
loop by driving representative tasks through the actual downstream
machinery -- ``l2co.init_run_state`` / ``RolloutWrapper`` -- and checking the
two views agree.

``l2co`` is intentionally **not** a dependency of ``l2co-tasks`` (the
package is standalone), so this module skips entirely unless ``l2co``
is importable. Run it for real from this directory via the sibling
checkout's environment::

    uv run --project ../l2co --no-sync pytest tests/test_l2co_integration.py

The case subset covers one task per model/dataset shape (vector model,
``pass_rng`` path, dataset-backed full batch, RNN + minibatch, PINN
MLP, two-network ``MultiNet``); l2co's own integration suite sweeps
the optimizer portfolio. rl2co is deliberately not covered here: it
consumes tasks exclusively through tuples of l2co ``RunState`` objects
and adds no new task-facing surface, so its coverage belongs in the
rl2co repository.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

l2co = pytest.importorskip("l2co")

from l2co import OptimizationStep, RolloutWrapper, init_run_state  # noqa: E402
from l2co.history import BatchState  # noqa: E402
from l2co.optimization import run  # noqa: E402
from l2co.sampling import normal_sampling, random_sampling  # noqa: E402

from ._contract_utils import evaluate_task_loss  # noqa: E402
from .task_cases import CASE_BY_ID  # noqa: E402

pytestmark = pytest.mark.requires_l2co

# One case per model/dataset shape the Task abstraction supports.
INTEGRATION_IDS = [
    "bbob-sphere",  # vector model, unit domain, deterministic
    "bbob-sphere-noisy",  # pass_rng path
    "cec2005-f1",  # second analytic family
    "quadratic-square",  # unbounded vector model, no dataset
    "gaussian-meta",  # dataset-backed, full batch, unit domain
    "spiral",  # eqx RNN + minibatch BatchState path
    "pde-convection",  # PINN MLP over a collocation dataset
    "inviscid-burgers",  # two-network MultiNet model
]

# Deterministic tasks without a dataset: the contract replica and l2co
# must agree on loss values exactly.
REPLICA_IDS = ["bbob-sphere", "cec2005-f1", "quadratic-square"]

UNIT_IDS = [i for i in INTEGRATION_IDS if CASE_BY_ID[i].domain == "unit"]

N_ITERATIONS = 5
N_REALIZATIONS = 2


def _params(case_id: str) -> list:
    """Pytest params for a list of case ids, inheriting registry marks."""
    return pytest.param(
        case_id, marks=list(CASE_BY_ID[case_id].marks), id=case_id
    )


def _bounded(case_id: str) -> tuple[float | None, float | None]:
    """Parameter bounds matching the case's domain."""
    if CASE_BY_ID[case_id].domain == "unit":
        return (0.0, 1.0)
    return (None, None)


def _optimizer() -> OptimizationStep:
    """The single optimizer used across this suite."""
    return OptimizationStep(
        optimizer="adam", hyperparameters={"learning_rate": 1e-3}
    )


def _rollout(task, case_id: str, key) -> RolloutWrapper:
    """Build a RolloutWrapper with domain-matched sampler and bounds.

    ``batch_evaluate`` resets each realization's parameters with the
    sampler, and samplers do not clip to ``bounded`` -- unit-domain
    tasks therefore use the uniform ``random_sampling`` so the reset
    starts inside the box (out-of-box starts can be NaN, e.g. the
    gaussian-meta task decodes adam's ``b2 > 1`` from ``x > 1``).
    """
    unit = CASE_BY_ID[case_id].domain == "unit"
    return RolloutWrapper.init(
        optimizer=_optimizer(),
        task=task,
        sampler=random_sampling if unit else normal_sampling,
        key=key,
        bounded=_bounded(case_id),
    )


@pytest.mark.parametrize("case_id", [_params(i) for i in INTEGRATION_IDS])
def test_runstate_init(case_id, build_case):
    """``init_run_state`` accepts the task and replicates its model.

    ``task.model``'s parameter values are arbitrary placeholders (only
    the static structure is contractual); ``init_run_state`` copies them
    verbatim and the sampler reset in ``evaluate``/``batch_evaluate``
    replaces them. This test pins that init mechanic.
    """
    task = build_case(case_id)
    rs = init_run_state(
        optimizer=_optimizer(),
        task=task,
        bounded=_bounded(case_id),
        key=jr.key(0),
    )
    assert rs.best_loss == jnp.inf

    model_leaves = jtu.tree_leaves(
        eqx.filter(task.model, eqx.is_inexact_array)
    )
    best_leaves = jtu.tree_leaves(
        eqx.filter(rs.best_params, eqx.is_inexact_array)
    )
    param_leaves = jtu.tree_leaves(eqx.filter(rs.params, eqx.is_inexact_array))
    assert len(model_leaves) == len(best_leaves) == len(param_leaves)
    for m, b, p in zip(model_leaves, best_leaves, param_leaves, strict=True):
        assert jnp.array_equal(b, m)  # best_params start at the model
        assert p.shape[1:] == m.shape  # leading popsize axis
        assert jnp.array_equal(p[0], m)


@pytest.mark.parametrize("case_id", [_params(i) for i in INTEGRATION_IDS])
def test_rollout_short_run_is_finite(case_id, build_case):
    """A short ``batch_evaluate`` produces finite history and best loss."""
    task = build_case(case_id)
    rollout = _rollout(task, case_id, jr.key(1))
    run_state, _, history = rollout.batch_evaluate(
        key=jr.key(2),
        n_iterations=N_ITERATIONS,
        n_realizations=N_REALIZATIONS,
    )
    assert history.output_min.shape == (N_REALIZATIONS, N_ITERATIONS)
    assert jnp.array_equal(
        history.cursor, jnp.full(N_REALIZATIONS, N_ITERATIONS)
    )
    assert jnp.all(jnp.isfinite(history.output_min))
    assert jnp.all(jnp.isfinite(run_state.best_loss))


@pytest.mark.parametrize("case_id", [_params(i) for i in REPLICA_IDS])
def test_run_matches_contract_replica(case_id, build_case):
    """l2co's recorded losses agree with the test-side contract replica.

    Uses the un-reset ``run`` path so the run starts at ``task.model``
    -- whose parameter values are arbitrary placeholders, used here
    purely as a deterministic, sampler-free starting point: the best
    loss can then never exceed the replica's loss at that start, and
    re-evaluating l2co's ``best_params`` through the replica must
    reproduce l2co's recorded ``best_loss`` exactly (deterministic,
    dataset-free tasks only).
    """
    task = build_case(case_id)
    rs = init_run_state(
        optimizer=_optimizer(),
        task=task,
        bounded=_bounded(case_id),
        key=jr.key(3),
    )
    bs = BatchState.init(
        dataset=task.loaded_dataset, batch_size=task.batch_size, key=jr.key(3)
    )
    rs, _, history = run(
        run_state=rs,
        batch_state=bs,
        dataset=task.loaded_dataset,
        n_iterations=N_ITERATIONS,
        key=jr.key(4),
        verbose=False,
    )
    init_loss = evaluate_task_loss(task, task.model)
    assert rs.best_loss <= init_loss + 1e-6
    assert jnp.isclose(rs.best_loss, jnp.min(history.output_min))
    replayed = evaluate_task_loss(task, rs.best_params)
    assert jnp.isclose(replayed, rs.best_loss, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case_id", [_params(i) for i in UNIT_IDS])
def test_bounded_unit_tasks_stay_in_box(case_id, build_case):
    """With ``bounded=(0, 1)`` the best parameters stay in the unit box."""
    task = build_case(case_id)
    rollout = _rollout(task, case_id, jr.key(5))
    run_state, _, _ = rollout.batch_evaluate(
        key=jr.key(6),
        n_iterations=N_ITERATIONS,
        n_realizations=N_REALIZATIONS,
    )
    leaves = jtu.tree_leaves(
        eqx.filter(run_state.best_params, eqx.is_inexact_array)
    )
    assert leaves
    for leaf in leaves:
        assert jnp.all((leaf >= 0.0) & (leaf <= 1.0))
