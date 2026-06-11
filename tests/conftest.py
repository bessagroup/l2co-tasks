"""Shared pytest fixtures for the l2co_tasks test suite.

The helpers in this file are kept deliberately small: they build the
minimal inputs used across multiple test modules and define a picklable
loss function at module level so Tasks built from the fixtures can
round-trip through cloudpickle.

This file also hosts the :class:`~tests.task_cases.TaskCase` machinery
shared by the per-task suites (contract, optimizability, values,
integration): a session-scoped task cache, the indirect ``tc`` fixture,
the ``pytest_generate_tests`` hook that parametrises any test consuming
``tc`` over every registered case, and the ``build_case`` fixture for
suites that target individual cases by id.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks import Task

from .task_cases import CASE_BY_ID, TASK_CASES, TaskCase


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


@dataclass
class Built:
    """A built task paired with its registry case."""

    task: Task
    case: TaskCase


@pytest.fixture(scope="session")
def _task_cache() -> dict[str, Task]:
    """Session cache so each task is built (and its dataset written) once."""
    return {}


def _build_cached(
    case: TaskCase, cache: dict[str, Task], tmp_path_factory
) -> Task:
    """Build ``case`` once per session, reusing the cached instance."""
    if case.id not in cache:
        d = tmp_path_factory.mktemp(case.id.replace("-", "_"))
        cache[case.id] = case.build(d)
    return cache[case.id]


@pytest.fixture
def tc(request, tmp_path_factory, _task_cache) -> Built:
    """Build (and cache) the task for the parametrised case."""
    case: TaskCase = request.param
    task = _build_cached(case, _task_cache, tmp_path_factory)
    return Built(task=task, case=case)


@pytest.fixture
def build_case(_task_cache, tmp_path_factory) -> Callable[[str], Task]:
    """Build (and session-cache) a registered TaskCase by its id."""

    def _build(case_id: str) -> Task:
        return _build_cached(
            CASE_BY_ID[case_id], _task_cache, tmp_path_factory
        )

    return _build


def pytest_generate_tests(metafunc):
    """Parametrise every test that consumes the ``tc`` fixture."""
    if "tc" in metafunc.fixturenames:
        params = [
            pytest.param(c, marks=list(c.marks), id=c.id) for c in TASK_CASES
        ]
        metafunc.parametrize("tc", params, indirect=True)
