"""Tests for ``create_mnist1d_task``.

Marked ``slow`` because the factory generates an MNIST-1D ``.npz``
dataset. The dataset is cached once per test-session via
``tmp_path_factory`` so multiple tests don't pay the generation cost.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_mnist1d_task


@pytest.fixture(scope="session")
def _mnist1d_dataset_path(tmp_path_factory):
    """Session-scoped dataset base path; first consumer triggers generation."""
    d = tmp_path_factory.mktemp("mnist1d")
    return str(d / "mnist1d")


@pytest.fixture(scope="session")
def _mnist1d_task(_mnist1d_dataset_path):
    """Build the task once per session; reused by every test in this module."""
    return create_mnist1d_task(
        dataset_path=_mnist1d_dataset_path,
        dataset_size=64,
        seed=0,
        batch_size=8,
        global_min_restarts=2,
        global_min_steps=10,
    )


@pytest.mark.slow
def test_create_mnist1d_task_fields(_mnist1d_task):
    task = _mnist1d_task
    assert isinstance(task, Task)
    assert task.name == "mnist1d"
    assert task.tag["loss_fn"] == "mean_categorical_cross_entropy_loss_fn"
    # in=40, out=10, hidden=16, 2 layers ⇒
    # (40*16+16) + (16*16+16) + (16*10+10) = 656 + 272 + 170 = 1098
    assert task.dimensionality == 1098
    assert task.batch_size == 8
    # Note: unlike ``create_spiral_task``, ``create_mnist1d_task`` does
    # not call ``tag.update(model_tags)``, so ``tag["model"]`` is absent.
    # If that inconsistency is fixed, this assertion will flip to
    # ``== "mlp"``.
    assert "model" not in task.tag


@pytest.mark.slow
def test_create_mnist1d_task_loss_and_round_trip(_mnist1d_task, tmp_path):
    task = _mnist1d_task
    ds = task.loaded_dataset
    x = ds["x"][:8]
    y = ds["y"][:8]
    before = float(task.loss_fn(task.model, x, y))
    assert jnp.isfinite(before)

    saved = Task.save(task, str(tmp_path / "task.eqx"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    after = float(loaded.loss_fn(loaded.model, x, y))
    assert after == pytest.approx(before, rel=1e-5)
