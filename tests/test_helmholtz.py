"""Tests for ``create_helmholtz_task`` (2-D and 3-D)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_helmholtz_task


def _build(tmp_path, dim):
    """Build a tiny Helmholtz task on a coarse collocation set."""
    if dim == 2:
        return create_helmholtz_task(
            a1=1.0,
            a2=4.0,
            k=1.0,
            seed=0,
            dataset_path=str(tmp_path / "hh2"),
            dim=2,
            modes=(1,),
            num_res_points=16,
            hidden_size=4,
            num_layers=2,
        )
    return create_helmholtz_task(
        a1=4.0,
        a2=4.0,
        k=1.0,
        seed=0,
        dataset_path=str(tmp_path / "hh3"),
        dim=3,
        a3=3.0,
        modes=(1, 2),
        num_res_points=16,
        hidden_size=4,
        num_layers=2,
    )


@pytest.mark.slow
@pytest.mark.parametrize("dim", [2, 3])
def test_fields_and_dataset(tmp_path, dim):
    task = _build(tmp_path, dim)
    assert isinstance(task, Task)
    assert task.name == "helmholtz"
    assert task.tag["loss_fn"] == "helmholtz_pde_loss"
    assert task.tag["dim"] == dim
    assert task.global_min == 0.0
    ds = task.loaded_dataset
    assert ds["x_res"].shape == (16, dim)


@pytest.mark.slow
@pytest.mark.parametrize("dim", [2, 3])
def test_loss_and_round_trip(tmp_path, dim):
    task = _build(tmp_path, dim)
    ds = task.loaded_dataset
    before = float(task.loss_fn(task.model, **ds))
    assert jnp.isfinite(before)
    saved = Task.save(task, str(tmp_path / "task"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    after = float(loaded.loss_fn(loaded.model, **loaded.loaded_dataset))
    assert after == pytest.approx(before, rel=1e-5)


def test_invalid_dim_raises(tmp_path):
    with pytest.raises(ValueError):
        create_helmholtz_task(
            a1=1.0,
            a2=1.0,
            k=1.0,
            seed=0,
            dataset_path=str(tmp_path / "bad"),
            dim=4,
        )
