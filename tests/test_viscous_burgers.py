"""Tests for ``create_viscous_burgers_task``."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_viscous_burgers_task
from l2co_tasks._src.viscous_burgers import exact_solution


def _build(tmp_path):
    """Build a tiny viscous-Burgers task."""
    return create_viscous_burgers_task(
        seed=0,
        dataset_path=str(tmp_path / "vb"),
        num_res_points=16,
        num_ic_points=8,
        num_bc_points=8,
        hidden_size=4,
        num_layers=2,
    )


@pytest.mark.slow
def test_fields_and_dataset(tmp_path):
    task = _build(tmp_path)
    assert isinstance(task, Task)
    assert task.name == "viscous_burgers"
    assert task.tag["nu"] == 0.004
    ds = task.loaded_dataset
    for key in ("x_res", "x_ic", "x_bc"):
        assert ds[key].shape[1] == 3
    # IC points sit at t = 0.
    assert bool(jnp.allclose(ds["x_ic"][:, 2], 0.0))


def test_exact_solution_is_bounded():
    """The logistic exact solution stays in (0, 1) and never overflows."""
    pts = jnp.array([[0.0, 0.0, 0.0], [10.0, 10.0, -10.0], [-9.0, -9.0, 9.0]])
    u = exact_solution(pts, nu=0.004)
    assert bool(jnp.all(jnp.isfinite(u)))
    assert bool(jnp.all((u >= 0.0) & (u <= 1.0)))


@pytest.mark.slow
def test_loss_and_round_trip(tmp_path):
    task = _build(tmp_path)
    ds = task.loaded_dataset
    before = float(task.loss_fn(task.model, **ds))
    assert jnp.isfinite(before)
    saved = Task.save(task, str(tmp_path / "task"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    after = float(loaded.loss_fn(loaded.model, **loaded.loaded_dataset))
    assert after == pytest.approx(before, rel=1e-5)
