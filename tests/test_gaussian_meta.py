"""Tests for ``create_gaussian_meta_task``
(Adam hyperparameter meta-learning)."""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import pytest

from l2co_tasks import Box, Task, create_gaussian_meta_task


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
        global_min_restarts=2,
        global_min_steps=5,
    )
    assert isinstance(task, Task)
    assert task.name == "gaussian_classification_meta"
    # Outer model is three unit-interval Adam hyperparameters.
    assert task.dimensionality == 3
    assert task.tag["dimensionality"] == 3
    assert task.tag["num_gaussians"] == 2
    # The unit box is part of the task (ADR 0002): outside it, b1/b2 > 1.
    (box,) = task.constraints
    assert isinstance(box, Box)
    assert box.lower.shape == box.upper.shape == (3,)
    assert (box.lower == 0.0).all() and (box.upper == 1.0).all()


def test_loss_is_finite_on_every_corner_of_the_box(tmp_path):
    """The whole closed box is evaluable (ADR 0002).

    Clipping parks optimizers on the box's faces, so the loss must be
    finite there: the momenta stop at 0.999, short of Adam's 0/0 at 1.
    """
    task = create_gaussian_meta_task(
        seed=0,
        dataset_path=str(tmp_path / "gmeta"),
        num_gaussians=2,
        num_samples_per_gaussian=4,
        dim_points=2,
        inner_steps=2,
        l2_regularization=0.01,
        estimate_global_min=False,
    )
    data = {k: jnp.asarray(v) for k, v in task.loaded_dataset.items()}
    for corner in itertools.product([0.0, 1.0], repeat=3):
        loss = task.loss_fn(jnp.array(corner, dtype=jnp.float32), **data)
        assert jnp.isfinite(loss), corner


_GAUSSIAN_META_ARGS = dict(
    seed=0,
    num_gaussians=2,
    num_samples_per_gaussian=4,
    dim_points=2,
    inner_steps=2,
    l2_regularization=0.01,
    estimate_global_min=False,
)


@pytest.mark.parametrize(
    "changed",
    [
        {"seed": 1},
        {"num_gaussians": 3},
        {"num_samples_per_gaussian": 5},
        {"dim_points": 3},
        {"inner_steps": 3},
        {"l2_regularization": 0.02},
    ],
)
def test_gaussian_meta_hash_follows_every_argument_that_changes_the_task(
    tmp_path, changed
):
    """Same arguments, same task; one changed argument, another task."""
    base = create_gaussian_meta_task(
        dataset_path=str(tmp_path / "a"), **_GAUSSIAN_META_ARGS
    )
    again = create_gaussian_meta_task(
        dataset_path=str(tmp_path / "b"), **_GAUSSIAN_META_ARGS
    )
    other = create_gaussian_meta_task(
        dataset_path=str(tmp_path / "c"),
        **{**_GAUSSIAN_META_ARGS, **changed},
    )
    assert base.hash == again.hash
    assert base.hash != other.hash
