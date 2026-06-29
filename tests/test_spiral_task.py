"""Tests for ``create_spiral_task``.

Marked ``slow`` because the factory writes an ``.npz`` dataset to disk.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_spiral_task


@pytest.mark.slow
def test_create_spiral_task_builds_dataset_and_task(tmp_path):
    ds_base = tmp_path / "spiral"
    task = create_spiral_task(
        hidden_size=4,
        dataset_size=32,
        dataset_path=str(ds_base),
        seed=0,
        batch_size=8,
        global_min_restarts=2,
        global_min_steps=10,
    )

    assert isinstance(task, Task)
    assert task.name == "spiral"
    assert task.tag["loss_fn"] == "mse"
    assert task.tag["model"] == "rnn"
    assert task.dimensionality > 0
    assert task.tag["dimensionality"] == task.dimensionality
    assert task.batch_size == 8
    assert ds_base.with_suffix(".npz").exists()

    ds = task.loaded_dataset
    assert "x" in ds and "y" in ds
    assert ds["x"].shape == (32, 16, 2)
    assert ds["y"].shape == (32, 1)


@pytest.mark.slow
def test_create_spiral_task_loss_runs_on_batch(tmp_path):
    """One loss evaluation on a mini-batch returns a finite scalar."""
    task = create_spiral_task(
        hidden_size=4,
        dataset_size=16,
        dataset_path=str(tmp_path / "spiral"),
        seed=0,
        batch_size=4,
        global_min_restarts=2,
        global_min_steps=10,
    )
    ds = task.loaded_dataset
    loss = float(task.loss_fn(task.model, ds["x"][:4], ds["y"][:4]))
    assert jnp.isfinite(loss)


@pytest.mark.slow
def test_create_spiral_task_round_trip(tmp_path):
    """Save/load preserves hash and loss value."""
    task = create_spiral_task(
        hidden_size=4,
        dataset_size=16,
        dataset_path=str(tmp_path / "spiral"),
        seed=0,
        batch_size=4,
        global_min_restarts=2,
        global_min_steps=10,
    )
    ds = task.loaded_dataset
    before = float(task.loss_fn(task.model, ds["x"][:4], ds["y"][:4]))

    saved = Task.save(task, str(tmp_path / "task.eqx"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    ds2 = loaded.loaded_dataset
    after = float(loaded.loss_fn(loaded.model, ds2["x"][:4], ds2["y"][:4]))
    assert after == pytest.approx(before, rel=1e-5)
