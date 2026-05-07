"""Tests for ``create_quadratic_task``.

``create_quadratic_task`` passes ``dataset=None`` to ``Task``. ``Task``
supports ``dataset=None`` explicitly as the "no dataset attached" case;
``batch_size``/``loaded_dataset``/``to_dict``/``save``/``load`` all handle
it without crashing.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_quadratic_task


@pytest.mark.parametrize("dimensionality", [2, 10])
def test_create_quadratic_task_basic_fields(dimensionality):
    """Construction returns a Task with the right dim, name, and evaluates."""
    task = create_quadratic_task(dimensionality=dimensionality, seed=0)
    assert isinstance(task, Task)
    assert task.dimensionality == dimensionality
    assert task.name == "quadratic_functions"
    assert task.tag["seed"] == 0
    assert task.pass_rng is False

    loss = float(task.loss_fn(task.model))
    # At model=0, loss is ||y||^2 which is non-negative and finite.
    assert loss >= 0.0 and jnp.isfinite(loss)


def test_quadratic_batch_size_is_none():
    """``.batch_size`` returns ``None`` when no dataset is attached."""
    task = create_quadratic_task(dimensionality=3, seed=0)
    assert task.batch_size is None
    assert task.loaded_dataset == {}


def test_quadratic_save_load_round_trip(tmp_path):
    """A quadratic task should round-trip like every other factory."""
    task = create_quadratic_task(dimensionality=4, seed=1)
    saved = Task.save(task, str(tmp_path / "task.eqx"))
    loaded = Task.load(saved)
    assert loaded == task
