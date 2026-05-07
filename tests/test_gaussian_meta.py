"""Tests for ``create_gaussian_meta_task``
(Adam hyperparameter meta-learning)."""

from __future__ import annotations

# import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_gaussian_meta_task


@pytest.mark.slow
def test_create_gaussian_meta_task_fields(tmp_path):
    task = create_gaussian_meta_task(
        seed=0,
        dataset_path=str(tmp_path / "gmeta"),
        num_gaussians=2,
        num_samples_per_gaussian=4,
        dim_points=2,
        inner_steps=2,
        l2_regularization=0.01,
    )
    assert isinstance(task, Task)
    assert task.name == "gaussian_classification_meta"
    # Outer model is three unit-interval Adam hyperparameters.
    assert task.dimensionality == 3
    assert task.tag["dimensionality"] == 3
    assert task.tag["num_gaussians"] == 2
