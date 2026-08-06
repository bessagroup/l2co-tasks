"""Tests for ``create_uav_task`` (UAV terrain path-planning).

Covers the specifics beyond the shared contract battery: the
``3 * n_waypoints`` dimensionality, the empirical ``global_min`` (an
estimate, not a floor), terrain determinism across builds, and that the
baked-terrain objective is JIT/vmap/grad-able. Marked ``slow`` because
each build synthesises a terrain (and, by default, benchmarks the
optimum).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks._src.uav_task import create_uav_task

pytestmark = pytest.mark.slow


def test_dimensionality_is_three_per_waypoint():
    """``d = 3 * n_waypoints`` and the empirical-global_min tag is set."""
    task = create_uav_task(
        n_waypoints=6, n_iter=4, estimate_global_min=False
    )
    assert task.dimensionality == 18
    assert task.tag["task_name"] == "uav"
    assert task.tag["global_min_kind"] == "empirical"
    assert task.pass_rng is False


def test_empirical_global_min_below_centre_loss():
    """The estimated optimum is no worse than the initial-model loss."""
    task = create_uav_task(n_waypoints=4, n_iter=4, estimate_global_min=True)
    assert task.global_min is not None
    assert task.global_min <= float(task.loss_fn(task.model)) + 1e-6


def test_objective_is_jit_vmap_grad_able():
    """The baked-terrain objective compiles, vectorises and differentiates."""
    task = create_uav_task(
        n_waypoints=5, n_iter=4, estimate_global_min=False
    )
    d = task.dimensionality
    x = jr.uniform(jr.key(0), (d,))
    assert jnp.isfinite(jax.jit(task.loss_fn)(x))
    assert jax.vmap(task.loss_fn)(jr.uniform(jr.key(1), (4, d))).shape == (4,)
    grad = jax.grad(task.loss_fn)(x)
    assert jnp.all(jnp.isfinite(grad))


def test_terrain_is_deterministic_per_seed():
    """Same ``(seed, terrain_id)`` reproduces the same objective."""
    a = create_uav_task(
        terrain_id=2, seed=7, n_waypoints=4, n_iter=4,
        estimate_global_min=False,
    )
    b = create_uav_task(
        terrain_id=2, seed=7, n_waypoints=4, n_iter=4,
        estimate_global_min=False,
    )
    x = jr.uniform(jr.key(3), (a.dimensionality,))
    assert jnp.allclose(a.loss_fn(x), b.loss_fn(x))
