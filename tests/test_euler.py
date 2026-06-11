"""Tests for ``create_euler_task`` (viscous and inviscid HLLC stages)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_euler_task
from l2co_tasks._src.euler import hllc_flux, physical_flux


def _build(tmp_path, stage):
    """Build a tiny Euler task for the given stage."""
    return create_euler_task(
        seed=0,
        dataset_path=str(tmp_path / f"euler_{stage}"),
        stage=stage,
        num_res_points=16,
        num_ic_points=8,
        num_bc_points=8,
        hidden_size=4,
        num_layers=2,
    )


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["viscous", "inviscid"])
def test_fields_and_dataset(tmp_path, stage):
    task = _build(tmp_path, stage)
    assert isinstance(task, Task)
    assert task.name == "euler"
    assert task.tag["stage"] == stage
    expected = f"euler_{stage}_loss"
    assert task.tag["loss_fn"] == expected
    ds = task.loaded_dataset
    for key in ("x_res", "x_ic", "x_bc"):
        assert ds[key].shape[1] == 2


def test_hllc_consistency_reduces_to_physical_flux():
    """For equal left/right states HLLC equals the physical flux."""
    w = (1.0, 0.3, 1.2)
    f_hllc = hllc_flux(w, w)
    f_phys = physical_flux(*w)
    assert bool(jnp.allclose(f_hllc, f_phys, atol=1e-5))


def test_invalid_stage_raises(tmp_path):
    with pytest.raises(ValueError):
        create_euler_task(
            seed=0,
            dataset_path=str(tmp_path / "bad"),
            stage="nope",
        )


@pytest.mark.slow
@pytest.mark.parametrize("stage", ["viscous", "inviscid"])
def test_loss_and_round_trip(tmp_path, stage):
    task = _build(tmp_path, stage)
    ds = task.loaded_dataset
    before = float(task.loss_fn(task.model, **ds))
    assert jnp.isfinite(before)
    saved = Task.save(task, str(tmp_path / "task"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    after = float(loaded.loss_fn(loaded.model, **loaded.loaded_dataset))
    assert after == pytest.approx(before, rel=1e-5)
