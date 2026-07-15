"""Tests for ``create_embedded_bbob_task``.

Structure-level checks of the random affine embedding: the anchor
attains the global minimum, gradients are confined to the
``intrinsic_dim``-dimensional row space, the null space is exactly
flat (``bulk_scale=0``) or uniformly ``bulk_scale``-curved, and the
Hessian spectrum is bulk-plus-outliers. Value pins live in
``test_task_values.py``; the generic battery runs via the cases
registered in ``task_cases.py``.

The factory is pure (no dataset files) so the tests run quickly and do
not need the ``slow`` marker.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks import Task, create_embedded_bbob_task


def _embedding(task: Task) -> dict:
    """The keyword bindings of the task's partial loss."""
    return task.loss_fn.keywords


def _null_space_vector(task: Task, key) -> jnp.ndarray:
    """A random ambient vector projected onto the embedding null space."""
    kw = _embedding(task)
    v = jr.normal(key, (task.dimensionality,))
    return v - kw["basis"].T @ (kw["basis"] @ v)


@pytest.mark.parametrize("fn_name", ["sphere", "rastrigin", "rosenbrock"])
@pytest.mark.parametrize("bulk_scale", [0.0, 0.5])
def test_create_embedded_bbob_task(fn_name, bulk_scale, eqx_path):
    """Embedded BBOB tasks build, carry honest tags, evaluate and
    round-trip."""
    task = create_embedded_bbob_task(
        fn_name=fn_name,
        seed=0,
        intrinsic_dim=3,
        ambient_dim=12,
        bulk_scale=bulk_scale,
    )
    assert isinstance(task, Task)
    assert task.dimensionality == 12
    assert task.pass_rng is False
    assert task.tag["fn_name"] == fn_name
    assert task.tag["seed"] == 0
    assert task.tag["noise"] == 0.0
    assert task.tag["dimensionality"] == 12
    assert task.tag["intrinsic_dimensionality"] == 3
    assert task.tag["bulk_scale"] == bulk_scale
    assert task.tag["embedded"] is True
    # The dense embedding destroys separability regardless of the
    # underlying function's own tag.
    assert task.tag["separable"] is False

    loss = float(task.loss_fn(task.model))
    assert jnp.isfinite(loss)
    assert loss >= task.global_min

    saved = Task.save(task, str(eqx_path))
    loaded = Task.load(saved)
    assert loaded == task
    assert loaded.hash == task.hash
    assert float(loaded.loss_fn(loaded.model)) == pytest.approx(loss)


@pytest.mark.parametrize("bulk_scale", [0.0, 0.5])
def test_global_min_attained_at_anchor(bulk_scale):
    """The loss equals ``global_min`` exactly at the anchor point,
    which lies inside the unit box."""
    task = create_embedded_bbob_task(
        fn_name="rastrigin",
        seed=3,
        intrinsic_dim=2,
        ambient_dim=10,
        bulk_scale=bulk_scale,
    )
    x_star = _embedding(task)["x_star"]
    assert jnp.all((x_star >= 0.0) & (x_star <= 1.0))
    loss = float(task.loss_fn(x_star))
    tol = 1e-4 * (1.0 + abs(task.global_min))
    assert abs(loss - task.global_min) <= tol


def test_gradient_confined_to_row_space():
    """With ``bulk_scale=0`` the gradient lies in the row space of the
    embedding (rank ``intrinsic_dim``), regardless of where it is
    evaluated."""
    task = create_embedded_bbob_task(
        fn_name="rosenbrock", seed=1, intrinsic_dim=2, ambient_dim=10
    )
    kw = _embedding(task)
    x = jr.uniform(jr.key(0), (10,))
    grad = jax.grad(task.loss_fn)(x)
    perp = grad - kw["basis"].T @ (kw["basis"] @ grad)
    assert jnp.linalg.norm(perp) <= 1e-4 * (1.0 + jnp.linalg.norm(grad))


def test_null_space_is_flat_without_bulk():
    """With ``bulk_scale=0`` the loss is invariant along null-space
    directions (a manifold of equivalent points)."""
    task = create_embedded_bbob_task(
        fn_name="sphere", seed=2, intrinsic_dim=2, ambient_dim=10
    )
    x = jr.uniform(jr.key(1), (10,))
    v = _null_space_vector(task, jr.key(2))
    a = float(task.loss_fn(x))
    b = float(task.loss_fn(x + v))
    assert b == pytest.approx(a, rel=1e-3)


def test_bulk_scale_curves_the_null_space():
    """With ``bulk_scale > 0`` a null-space displacement from the
    anchor costs exactly the quadratic penalty."""
    bulk_scale = 0.5
    task = create_embedded_bbob_task(
        fn_name="sphere",
        seed=2,
        intrinsic_dim=2,
        ambient_dim=10,
        bulk_scale=bulk_scale,
    )
    x_star = _embedding(task)["x_star"]
    v = _null_space_vector(task, jr.key(3))
    loss = float(task.loss_fn(x_star + v))
    expected = task.global_min + 0.5 * bulk_scale * float(jnp.sum(v**2))
    assert loss == pytest.approx(expected, rel=1e-3)


def test_hessian_spectrum_is_bulk_plus_outliers():
    """For the embedded sphere the Hessian has ``intrinsic_dim``
    eigenvalues at ``2 * scale**2`` and ``ambient_dim -
    intrinsic_dim`` eigenvalues at exactly ``bulk_scale``."""
    intrinsic_dim, ambient_dim, bulk_scale = 2, 6, 0.5
    task = create_embedded_bbob_task(
        fn_name="sphere",
        seed=4,
        intrinsic_dim=intrinsic_dim,
        ambient_dim=ambient_dim,
        bulk_scale=bulk_scale,
    )
    scale = _embedding(task)["scale"]
    x = jr.uniform(jr.key(4), (ambient_dim,))
    eigenvalues = jnp.sort(jnp.linalg.eigvalsh(jax.hessian(task.loss_fn)(x)))
    bulk = eigenvalues[: ambient_dim - intrinsic_dim]
    outliers = eigenvalues[ambient_dim - intrinsic_dim :]
    assert jnp.allclose(bulk, bulk_scale, rtol=1e-3)
    assert jnp.allclose(outliers, 2.0 * scale**2, rtol=1e-3)


def test_noise_flips_pass_rng():
    """``noise > 0`` flips ``pass_rng=True`` and the loss consumes a
    key; different keys produce different values away from zero."""
    task = create_embedded_bbob_task(
        fn_name="sphere",
        seed=0,
        intrinsic_dim=2,
        ambient_dim=8,
        noise=0.1,
    )
    assert task.pass_rng is True
    l1 = float(task.loss_fn(jnp.ones(8) * 0.2, jr.key(0)))
    l2 = float(task.loss_fn(jnp.ones(8) * 0.2, jr.key(1)))
    assert jnp.isfinite(l1) and jnp.isfinite(l2)
    assert l1 != l2


def test_ambient_dim_below_intrinsic_dim_raises():
    """The embedding needs at least as many ambient as intrinsic
    dimensions."""
    with pytest.raises(ValueError, match="ambient_dim"):
        create_embedded_bbob_task(
            fn_name="sphere", seed=0, intrinsic_dim=5, ambient_dim=3
        )


def test_non_bbob_function_raises():
    """Only BBOB-suite names are accepted (no CEC suites)."""
    with pytest.raises(ValueError, match="BBOB"):
        create_embedded_bbob_task(
            fn_name="cec2017_f1", seed=0, intrinsic_dim=2, ambient_dim=8
        )


def test_negative_bulk_scale_raises():
    """A negative null-space curvature is rejected."""
    with pytest.raises(ValueError, match="bulk_scale"):
        create_embedded_bbob_task(
            fn_name="sphere",
            seed=0,
            intrinsic_dim=2,
            ambient_dim=8,
            bulk_scale=-1.0,
        )
