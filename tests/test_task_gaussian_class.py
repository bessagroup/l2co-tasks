"""Tests for ``create_gaussian_task`` (synthetic Gaussian classification)."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_gaussian_task


@pytest.mark.slow
def test_create_gaussian_task_fields(tmp_path):
    ds_base = tmp_path / "gauss"  # factory appends "_{seed}" and ".npz"
    task = create_gaussian_task(
        seed=0,
        dataset_path=str(ds_base),
        num_gaussians=2,
        num_samples_per_gaussian=4,
        dim_points=2,
        l2_regularization=0.01,
        hidden_size=2,
        num_layers=2,
    )
    assert isinstance(task, Task)
    assert task.name == "gaussian_classification"
    assert task.tag["num_gaussians"] == 2
    assert task.tag["num_samples_per_gaussian"] == 4
    assert task.tag["dim_points"] == 2
    assert task.tag["l2_regularization"] == 0.01
    assert task.tag["separable"] is False and task.tag["unimodal"] is False
    assert task.global_min == pytest.approx(0.322)
    assert task.dimensionality == task.tag["dimensionality"]
    assert Path(str(ds_base) + "_0.npz").exists()


@pytest.mark.slow
def test_create_gaussian_task_loss_and_round_trip(tmp_path):
    task = create_gaussian_task(
        seed=1,
        dataset_path=str(tmp_path / "gauss"),
        num_gaussians=2,
        num_samples_per_gaussian=4,
        dim_points=2,
        l2_regularization=0.01,
    )
    ds = task.loaded_dataset
    before = float(task.loss_fn(task.model, ds["x"], ds["y"]))
    assert jnp.isfinite(before)

    saved = Task.save(task, str(tmp_path / "task.eqx"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    ds2 = loaded.loaded_dataset
    after = float(loaded.loss_fn(loaded.model, ds2["x"], ds2["y"]))
    assert after == pytest.approx(before, rel=1e-5)
