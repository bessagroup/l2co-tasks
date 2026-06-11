"""Task-contract battery: criteria every task must satisfy.

A single parametrised suite that builds each registered
:class:`~tests.task_cases.TaskCase` once (cached per session) and runs a
shared set of correctness criteria over it:

A. structure & metadata
B. loss correctness (finite, deterministic, lower-bound)
C. JAX transforms -- ``jit`` / ``vmap`` / ``grad`` (the core ask)
D. dataset-backed tasks (minibatch path)
E. ``save`` / ``load`` round-trip
F. reproducibility across identical seeds
G. NaN-input robustness (must not raise; shape preserved, value may be NaN)

Heavy cases carry ``@pytest.mark.slow`` via their registry marks, so
``pytest -m "not slow"`` runs only the cheap analytical tasks.
"""

from __future__ import annotations

import inspect

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from l2co_tasks import Task

from ._contract_utils import (
    evaluate_task_loss,
    fill_model,
    loss_closure,
    minibatch_idxs,
    sample_model,
    stack_models,
)

# A batched model of this size is used for the vmap criteria.
BATCH = 4
RTOL, ATOL = 1e-4, 1e-4


def _loss_accepts_key(loss_fn) -> bool:
    """Whether ``loss_fn`` accepts a ``key`` keyword argument."""
    try:
        sig = inspect.signature(loss_fn)
    except (ValueError, TypeError):
        return True  # builtin / un-introspectable: don't over-assert
    params = sig.parameters
    if "key" in params:
        return True
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


# ---------------------------------------------------------------------------
# A. Structure & metadata
# ---------------------------------------------------------------------------


def test_is_task_with_positive_dimensionality(tc):
    """Task builds; dimensionality counts the inexact-array leaves."""
    task = tc.task
    assert isinstance(task, Task)
    expected = sum(
        x.size
        for x in jtu.tree_leaves(eqx.filter(task.model, eqx.is_inexact_array))
    )
    assert task.dimensionality == expected > 0


def test_model_leaves_are_float(tc):
    """Every trainable leaf is a floating (inexact) array."""
    leaves = jtu.tree_leaves(eqx.filter(tc.task.model, eqx.is_inexact_array))
    assert leaves
    assert all(jnp.issubdtype(x.dtype, jnp.floating) for x in leaves)


def test_metadata_well_formed(tc):
    """Tag carries an identifier + dimensionality; global_min is sane.

    Benchmark tasks identify themselves via ``tag["fn_name"]`` (copied
    from the ``bbob_jax`` registry) rather than ``tag["task_name"]``,
    so either is accepted.
    """
    task = tc.task
    assert task.tag, "task tag must not be empty"
    assert task.tag.get("task_name") or task.tag.get("fn_name")
    assert task.tag.get("dimensionality") == task.dimensionality
    if task.global_min is not None:
        assert isinstance(task.global_min, float)
        assert jnp.isfinite(task.global_min)


def test_pass_rng_consistent_with_signature(tc):
    """A ``pass_rng`` task's loss must accept a ``key`` keyword."""
    task = tc.task
    assert isinstance(task.pass_rng, bool)
    assert isinstance(task.has_aux, bool)
    if task.pass_rng:
        assert _loss_accepts_key(task.loss_fn)


# ---------------------------------------------------------------------------
# B. Loss correctness
# ---------------------------------------------------------------------------


def test_loss_finite_at_initial_model(tc):
    """Loss at the factory's initial model is a finite scalar."""
    task = tc.task
    loss = evaluate_task_loss(task, task.model, key=jr.key(0))
    assert loss.shape == ()
    assert jnp.issubdtype(loss.dtype, jnp.floating)
    assert jnp.isfinite(loss)


def test_loss_finite_at_sampled_model(tc):
    """Loss at an in-domain sampled model is a finite scalar."""
    task = tc.task
    model = sample_model(task, jr.key(1), tc.case.domain)
    loss = evaluate_task_loss(task, model, key=jr.key(2))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_loss_determinism(tc):
    """Deterministic tasks repeat exactly; stochastic ones repeat per key."""
    task = tc.task
    model = sample_model(task, jr.key(3), tc.case.domain)
    a = evaluate_task_loss(task, model, key=jr.key(7))
    b = evaluate_task_loss(task, model, key=jr.key(7))
    assert a == b  # same key (or none) -> identical
    if task.pass_rng:
        c = evaluate_task_loss(task, model, key=jr.key(8))
        assert jnp.isfinite(c)


def test_loss_respects_global_min_lower_bound(tc):
    """When ``global_min`` is a true bound, the loss never dips below it."""
    task = tc.task
    if not tc.case.gmin_is_lower_bound or task.pass_rng:
        pytest.skip("no guaranteed lower bound for this task")
    assert task.global_min is not None
    keys = jr.split(jr.key(4), 8)
    for k in keys:
        model = sample_model(task, k, tc.case.domain)
        loss = float(evaluate_task_loss(task, model))
        tol = ATOL + RTOL * abs(task.global_min)
        assert loss >= task.global_min - tol


# ---------------------------------------------------------------------------
# C. JAX transforms
# ---------------------------------------------------------------------------


def test_loss_under_jit_matches_eager(tc):
    """``eqx.filter_jit`` of the loss matches the eager value."""
    task = tc.task
    model = sample_model(task, jr.key(5), tc.case.domain)
    f = loss_closure(task, key=jr.key(9))
    eager = f(model)
    jitted = eqx.filter_jit(f)(model)
    assert jnp.allclose(eager, jitted, rtol=RTOL, atol=ATOL)


def test_loss_under_vmap_matches_loop(tc):
    """``eqx.filter_vmap`` over a batch of models matches a Python loop."""
    task = tc.task
    keys = jr.split(jr.key(6), BATCH)
    models = [sample_model(task, k, tc.case.domain) for k in keys]
    f = loss_closure(task, key=jr.key(10))

    batched = eqx.filter_vmap(f)(stack_models(models))
    assert batched.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(batched))

    looped = jnp.stack([f(m) for m in models])
    assert jnp.allclose(batched, looped, rtol=RTOL, atol=ATOL)


def test_loss_under_jit_of_vmap(tc):
    """``jit`` composed with ``vmap`` runs and stays finite."""
    task = tc.task
    keys = jr.split(jr.key(11), BATCH)
    models = [sample_model(task, k, tc.case.domain) for k in keys]
    f = loss_closure(task, key=jr.key(12))
    out = eqx.filter_jit(eqx.filter_vmap(f))(stack_models(models))
    assert out.shape == (BATCH,)
    assert jnp.all(jnp.isfinite(out))


def test_loss_is_differentiable(tc):
    """Gradient w.r.t. the model is finite with the trainable structure."""
    task = tc.task
    model = sample_model(task, jr.key(13), tc.case.domain)
    f = loss_closure(task, key=jr.key(14))
    value, grad = eqx.filter_value_and_grad(f)(model)
    assert jnp.isfinite(value)
    grad_leaves = jtu.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    param_leaves = jtu.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    assert len(grad_leaves) == len(param_leaves)
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)


def test_vmapped_gradient_is_finite(tc):
    """Batched gradients are finite (the optimizers vmap over a population)."""
    task = tc.task
    keys = jr.split(jr.key(15), BATCH)
    models = [sample_model(task, k, tc.case.domain) for k in keys]
    f = loss_closure(task, key=jr.key(16))
    grads = eqx.filter_vmap(eqx.filter_grad(f))(stack_models(models))
    leaves = jtu.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert all(jnp.all(jnp.isfinite(g)) for g in leaves)


# ---------------------------------------------------------------------------
# D. Dataset-backed tasks
# ---------------------------------------------------------------------------


def test_dataset_keys_match_loss_and_minibatch_runs(tc):
    """Dataset keys match the loss kwargs; a minibatch eval is finite."""
    task = tc.task
    if task.dataset is None:
        pytest.skip("task carries no dataset")
    ds = task.loaded_dataset
    assert ds, "dataset declared but loaded_dataset is empty"
    idxs = minibatch_idxs(task, jr.key(17), dataset=ds)
    # Full-batch (idxs=None) and minibatch paths both go through the
    # canonical contract; a kwarg mismatch would raise TypeError here.
    loss = evaluate_task_loss(
        task, task.model, key=jr.key(18), dataset=ds, idxs=idxs
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    f = loss_closure(task, dataset=ds, idxs=idxs, key=jr.key(19))
    assert jnp.isfinite(eqx.filter_jit(f)(task.model))


# ---------------------------------------------------------------------------
# E. Serialization round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip_preserves_loss(tc, tmp_path):
    """``Task.save`` then ``Task.load`` reproduces the loss at the model."""
    task = tc.task
    before = evaluate_task_loss(task, task.model, key=jr.key(20))
    # The dataset .npz keeps its original absolute location in the cached
    # tmp dir; ``Task.save`` records its path relative to the .eqx file and
    # ``Task.load`` resolves it back, so saving into a fresh dir is fine.
    saved = Task.save(task, str(tmp_path / "task"))
    loaded = Task.load(saved)
    after = evaluate_task_loss(loaded, loaded.model, key=jr.key(20))
    assert jnp.allclose(before, after, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# F. Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_reproducible(tc, tmp_path_factory):
    """Rebuilding with the same seed yields identical model and loss."""
    case = tc.case
    rebuilt = case.build(tmp_path_factory.mktemp(case.id.replace("-", "_")))
    a = jtu.tree_leaves(eqx.filter(tc.task.model, eqx.is_inexact_array))
    b = jtu.tree_leaves(eqx.filter(rebuilt.model, eqx.is_inexact_array))
    assert len(a) == len(b)
    assert all(jnp.array_equal(x, y) for x, y in zip(a, b, strict=True))
    la = evaluate_task_loss(tc.task, tc.task.model, key=jr.key(0))
    lb = evaluate_task_loss(rebuilt, rebuilt.model, key=jr.key(0))
    assert la == lb


# ---------------------------------------------------------------------------
# G. NaN-input robustness
# ---------------------------------------------------------------------------


def test_nan_input_does_not_raise(tc):
    """All-NaN model must not raise; output shape kept (value may be NaN)."""
    task = tc.task
    nan_model = fill_model(task, jnp.nan)
    ref = evaluate_task_loss(task, task.model, key=jr.key(0))
    f = loss_closure(task, key=jr.key(0))

    eager = f(nan_model)  # must not raise
    assert eager.shape == ref.shape == ()
    assert eager.dtype == ref.dtype

    jitted = eqx.filter_jit(f)(nan_model)  # value-dependent branch would fail
    assert jitted.shape == ()
    assert jitted.dtype == ref.dtype
    # Value is intentionally unconstrained: NaN is acceptable.
