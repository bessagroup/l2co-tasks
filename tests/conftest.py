"""Shared pytest fixtures for the l2co_tasks test suite.

The helpers in this file are kept deliberately small: they build the
minimal inputs used across multiple test modules and define a picklable
loss function at module level so Tasks built from the fixtures can
round-trip through cloudpickle.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks import Task


def sum_of_squares(model):
    """Picklable loss used by ``simple_task``; lives at module level so
    ``cloudpickle`` can serialise it via ``Task.save``.
    """
    return jnp.sum(model**2)


@pytest.fixture(scope="session")
def rng_key():
    """Deterministic JAX PRNG key shared across tests."""
    return jr.key(0)


@pytest.fixture
def eqx_path(tmp_path: Path) -> Path:
    """A fresh ``.eqx`` path inside a unique tmp dir for each test."""
    return tmp_path / "task.eqx"


@pytest.fixture
def dataset_path(tmp_path: Path) -> Path:
    """A fresh ``.npz`` dataset path inside a unique tmp dir."""
    return tmp_path / "dataset.npz"


@pytest.fixture
def simple_task() -> Task:
    """A minimal hand-rolled Task: 1D array model + picklable loss."""
    return Task(
        model=jnp.arange(3, dtype=jnp.float32),
        loss_fn=sum_of_squares,
        tag={"task_name": "simple", "seed": 0},
    )
