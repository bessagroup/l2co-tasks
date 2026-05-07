"""Save/load round-trip tests for the single-file ``.eqx`` Task format.

These exercise the cloudpickle-hex path for the loss function, the
``eqx.tree_serialise_leaves`` path for the model weights, and the
relative-path rewrite that keeps a task portable alongside its
``.npz`` dataset.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks import DatasetDict, Task

from .conftest import sum_of_squares


def _picklable_dataset_loss(model, x, y):
    """Module-level loss used for the dataset-portability test."""
    return jnp.mean((model - x) ** 2) + jnp.mean(y**2)


def test_round_trip_preserves_identity(simple_task, eqx_path):
    """``Task.save`` then ``Task.load`` returns an equal task with the
    same hash and the same loss-function value on the same weights."""
    saved = Task.save(simple_task, str(eqx_path))
    loaded = Task.load(saved)

    assert loaded == simple_task
    assert loaded.hash == simple_task.hash
    assert loaded.tag == simple_task.tag
    assert float(loaded.loss_fn(loaded.model)) == pytest.approx(
        float(simple_task.loss_fn(simple_task.model))
    )


def test_round_trip_preserves_weights(simple_task, eqx_path):
    """``eqx.tree_serialise_leaves`` must round-trip the model
    bytes-for-bytes."""
    saved = Task.save(simple_task, str(eqx_path))
    loaded = Task.load(saved)
    assert jnp.array_equal(loaded.model, simple_task.model)


def test_round_trip_pass_rng_and_has_aux(eqx_path, rng_key):
    """Static flags ``pass_rng`` and ``has_aux`` survive the round-trip."""

    def rng_loss(model, key):
        return jnp.sum(model**2) + jr.normal(key, ())

    t = Task(
        model=jnp.zeros(2),
        loss_fn=rng_loss,
        pass_rng=True,
        has_aux=True,
        tag={"task_name": "rng-aux"},
    )
    saved = Task.save(t, str(eqx_path))
    loaded = Task.load(saved)
    assert loaded.pass_rng is True
    assert loaded.has_aux is True


def test_to_dict_keys_and_no_dataset():
    """``to_dict`` header shape is stable and ``dataset`` is ``None``
    when empty."""
    t = Task(model=jnp.zeros(2), loss_fn=sum_of_squares, tag={"x": 1})
    header = t.to_dict()
    assert set(header) == {
        "model_shape_hex",
        "loss_fn",
        "pass_rng",
        "has_aux",
        "global_min",
        "dataset",
        "tag",
    }
    assert header["dataset"] is None


def test_dataset_path_is_relative_after_save(tmp_path):
    """The saved header stores ``dataset_path`` relative to the ``.eqx`` file
    so task + dataset can be moved together without edits."""
    ds_path = tmp_path / "dataset.npz"
    jnp.savez(ds_path, x=jnp.arange(3))
    t = Task(
        model=jnp.zeros(2),
        loss_fn=sum_of_squares,
        dataset=DatasetDict(dataset_path=str(ds_path), batch_size=1, seed=0),
        tag={"task_name": "with-dataset"},
    )

    eqx_path = tmp_path / "task.eqx"
    Task.save(t, str(eqx_path))

    import json

    with open(eqx_path, "rb") as f:
        header = json.loads(f.readline())
    stored = header["dataset"]["dataset_path"]
    assert not Path(stored).is_absolute()
    # Sibling files ⇒ the stored path is just the filename.
    assert Path(stored).name == "dataset.npz"


def test_dataset_portability_after_move(tmp_path):
    """Moving the ``.eqx`` + ``.npz`` pair together keeps
    ``loaded_dataset`` working."""
    first = tmp_path / "origin"
    first.mkdir()
    ds_path = first / "dataset.npz"
    jnp.savez(ds_path, x=jnp.arange(4), y=jnp.arange(4))
    t = Task(
        model=jnp.zeros(2),
        loss_fn=_picklable_dataset_loss,
        dataset=DatasetDict(dataset_path=str(ds_path), batch_size=2, seed=0),
        tag={"task_name": "portable"},
    )
    Task.save(t, str(first / "task.eqx"))

    second = tmp_path / "moved"
    shutil.copytree(first, second)

    loaded = Task.load(str(second / "task.eqx"))
    data = loaded.loaded_dataset
    assert jnp.array_equal(data["x"], jnp.arange(4))
