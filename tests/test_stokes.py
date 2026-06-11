"""Tests for ``create_stokes_task`` (lid-driven triangular wedge)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_stokes_task
from l2co_tasks._src.stokes import wedge_vertices


def _build(tmp_path):
    """Build a tiny Stokes-wedge task."""
    return create_stokes_task(
        seed=0,
        dataset_path=str(tmp_path / "stokes"),
        num_res_points=16,
        num_lid_points=8,
        num_wall_points=8,
        hidden_size=4,
        num_layers=2,
    )


@pytest.mark.slow
def test_fields_and_dataset(tmp_path):
    task = _build(tmp_path)
    assert isinstance(task, Task)
    assert task.name == "stokes"
    assert task.tag["angle_deg"] == 25.53
    ds = task.loaded_dataset
    for key in ("x_res", "x_lid", "x_wall"):
        assert ds[key].shape[1] == 2
    # Lid points lie on the top edge (y == height).
    assert bool(jnp.allclose(ds["x_lid"][:, 1], task.tag["height"]))


def test_wedge_vertices_geometry():
    """Apex at origin; lid corners symmetric about the y-axis."""
    v = wedge_vertices(angle_deg=25.53, height=1.0)
    assert bool(jnp.allclose(v[0], jnp.array([0.0, 0.0])))
    assert float(v[1, 0]) == pytest.approx(-float(v[2, 0]))
    assert float(v[1, 1]) == pytest.approx(1.0)


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
