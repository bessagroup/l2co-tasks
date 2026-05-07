"""Unit tests for the Task class in ``l2co_tasks._src.task``.

Covers the properties and dunder methods that don't involve disk I/O:
``dimensionality``, ``tag_hashable``, ``hash``, ``name``, ``batch_size``,
``loaded_dataset``, ``__eq__``, ``__hash__``, ``__repr__``, ``__str__``.
Save/load is exercised in ``test_task_io.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from l2co_tasks import Task

from .conftest import sum_of_squares


def test_dimensionality_flat_array():
    """Plain JAX array → dimensionality is its size."""
    t = Task(model=jnp.zeros(7), loss_fn=sum_of_squares)
    assert t.dimensionality == 7


def test_dimensionality_nested_eqx_module(rng_key):
    """Nested eqx.Module → sum of inexact-array leaves."""
    linear = eqx.nn.Linear(3, 2, key=rng_key)  # weight 2*3 + bias 2 = 8
    t = Task(model=linear, loss_fn=sum_of_squares)
    assert t.dimensionality == 8


def test_dimensionality_ignores_integer_leaves():
    """Integer arrays are not inexact and must not contribute."""
    model = (jnp.zeros(4), jnp.arange(10, dtype=jnp.int32))
    t = Task(model=model, loss_fn=sum_of_squares)
    assert t.dimensionality == 4


def test_name_and_batch_size_defaults():
    """``name`` and ``batch_size`` return safe fallbacks when unset."""
    t = Task(model=jnp.zeros(2), loss_fn=sum_of_squares)
    assert t.name == ""
    assert t.batch_size is None


def test_name_and_batch_size_from_metadata():
    """Populated tag / dataset propagate through the properties."""
    t = Task(
        model=jnp.zeros(2),
        loss_fn=sum_of_squares,
        dataset={"dataset_path": None, "batch_size": 32, "seed": 0},
        tag={"task_name": "spiral"},
    )
    assert t.name == "spiral"
    assert t.batch_size == 32


def test_loaded_dataset_empty_without_path():
    """Missing ``dataset_path`` → ``loaded_dataset`` is ``{}``."""
    t = Task(model=jnp.zeros(2), loss_fn=sum_of_squares)
    assert t.loaded_dataset == {}


def test_loaded_dataset_reads_npz(tmp_path):
    """A real .npz on disk is loaded lazily via the property."""
    path = tmp_path / "ds.npz"
    jnp.savez(path, x=jnp.arange(3), y=jnp.arange(3) * 2)
    t = Task(
        model=jnp.zeros(2),
        loss_fn=sum_of_squares,
        dataset={"dataset_path": str(path), "batch_size": 1, "seed": 0},
    )
    loaded = t.loaded_dataset
    assert set(loaded.keys()) == {"x", "y"}
    assert jnp.array_equal(loaded["x"], jnp.arange(3))


def test_tag_hashable_converts_lists_and_dicts():
    """Lists → tuples and nested dicts → frozensets; order-independent."""
    t = Task(
        model=jnp.zeros(1),
        loss_fn=sum_of_squares,
        tag={"vals": [1, 2, 3], "nested": {"a": 1, "b": [4, 5]}},
    )
    th = dict(t.tag_hashable)
    assert th["vals"] == (1, 2, 3)
    assert th["nested"] == frozenset({("a", 1), ("b", (4, 5))})
    assert th["global_min"] is None


def test_hash_is_short_and_deterministic():
    """``hash`` returns a 16-char string that only depends on the tag."""
    t1 = Task(model=jnp.zeros(1), loss_fn=sum_of_squares, tag={"x": 1})
    t2 = Task(model=jnp.zeros(1), loss_fn=sum_of_squares, tag={"x": 1})
    t3 = Task(model=jnp.zeros(1), loss_fn=sum_of_squares, tag={"x": 2})
    assert len(t1.hash) == 16
    assert t1.hash == t2.hash
    assert t1.hash != t3.hash


def test_hash_changes_with_global_min():
    """``global_min`` participates in the hash via ``tag_hashable``."""
    t1 = Task(model=jnp.zeros(1), loss_fn=sum_of_squares, global_min=0.0)
    t2 = Task(model=jnp.zeros(1), loss_fn=sum_of_squares, global_min=1.0)
    assert t1.hash != t2.hash


def test_equality_is_tag_based_not_weight_based(rng_key):
    """Two tasks with identical tags compare equal even if weights differ.

    This is the documented invariant: ``__eq__`` compares only
    ``tag_hashable``, so Tasks are deduplicated by identity, not by
    parameter values.
    """
    k1, k2 = jr.split(rng_key)
    t1 = Task(model=jr.normal(k1, (5,)), loss_fn=sum_of_squares, tag={"x": 1})
    t2 = Task(model=jr.normal(k2, (5,)), loss_fn=sum_of_squares, tag={"x": 1})
    assert t1 == t2
    assert hash(t1) == hash(t2)


def test_equality_rejects_non_task():
    """``Task.__eq__`` is guarded against unrelated types."""
    t = Task(model=jnp.zeros(1), loss_fn=sum_of_squares, tag={"x": 1})
    assert t != {"x": 1}
    assert t != 42


def test_repr_contains_tag():
    """``repr`` surfaces the tag so logs are debuggable."""
    t = Task(
        model=jnp.zeros(1),
        loss_fn=sum_of_squares,
        tag={"task_name": "x", "seed": 7},
    )
    r = repr(t)
    assert "Task(" in r
    assert "task_name" in r and "seed" in r
