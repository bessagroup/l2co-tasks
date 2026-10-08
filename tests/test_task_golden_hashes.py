"""Golden hashes: an unconstrained task's identity must never drift.

``Task.hash`` is the databank's ``taskID``, so a change that shifts the
hash of an existing task orphans every result stored under it. The
round-trip tests elsewhere (``loaded.hash == task.hash``) cannot catch
that: both sides drift together. These values were recorded on commit
``2c2cd7e``, before constraints were added (l2co-tasks ADR 0002), and
pin three layers:

* the hashing code itself, on tasks built from fixed tags and fixed
  ``global_min`` values (hardware-independent);
* the factories whose ``global_min`` is exact, built with
  ``jax_enable_x64`` on, as every pipeline that writes a store runs
  (estimated-``global_min`` factories are left out: their estimate can
  differ between CPU and GPU);
* a ``.eqx`` file written by the pre-constraints code.

Do not update a value here unless re-keying the databank is intended.
"""

from __future__ import annotations

import json
from pathlib import Path

import bbob_jax
import jax
import jax.numpy as jnp
import pytest

from l2co_tasks import (
    Task,
    create_bbob_noisy_task,
    create_bbob_task,
    create_cec2005_task,
    create_cec2017_task,
    create_quadratic_task,
)

FIXTURE = Path(__file__).parent / "data" / "golden_quadratic_square.eqx"

#: Header keys a task file had before constraints existed.
PRE_CONSTRAINTS_HEADER_KEYS = {
    "dataset",
    "global_min",
    "has_aux",
    "loss_fn",
    "model_shape_hex",
    "pass_rng",
    "tag",
}


def _loss(model):
    return jnp.sum(model**2)


TASK_LEVEL = [
    pytest.param({}, None, "1afc2ac49a13728b", id="empty-tag-no-gmin"),
    pytest.param(
        {"task_name": "simple", "seed": 0},
        0.0,
        "abe00eaddcbeec8d",
        id="simple",
    ),
    pytest.param(
        {"task_name": "nested", "dims": [2, 3], "meta": {"a": 1}},
        1.5,
        "b56cd279e41ea485",
        id="list-and-dict",
    ),
    pytest.param(
        {
            "task_name": "bbob",
            "fn_name": "sphere",
            "dimensionality": 3,
            "seed": 0,
            "noise": 0.0,
        },
        -26.187623977661133,
        "cf276caf6843c7f2",
        id="bbob-like",
    ),
]


@pytest.mark.parametrize("tag, global_min, expected", TASK_LEVEL)
def test_task_level_hash_is_pinned(tag, global_min, expected):
    task = Task(
        model=jnp.zeros(3), loss_fn=_loss, tag=tag, global_min=global_min
    )
    assert task.hash == expected


_NEEDS_BBOB_NOISY = pytest.mark.skipif(
    not hasattr(bbob_jax, "bbob_noisy_registry"),
    reason="installed bbob-jax predates the BBOB-noisy suite",
)

FACTORY = [
    pytest.param(
        lambda: create_bbob_task(
            fn_name="sphere", seed=0, dimensionality=3, noise=0.0
        ),
        "13d7d4ff60a64f7f",
        id="bbob-sphere",
    ),
    pytest.param(
        lambda: create_bbob_task(
            fn_name="rastrigin", seed=0, dimensionality=3, noise=0.0
        ),
        "367e1bd1199584dc",
        id="bbob-rastrigin",
    ),
    pytest.param(
        lambda: create_bbob_task(
            fn_name="sphere", seed=0, dimensionality=3, noise=0.1
        ),
        "0cd204bc360894e1",
        id="bbob-sphere-noisy",
    ),
    pytest.param(
        lambda: create_bbob_noisy_task(
            fn_name="bbob_noisy_f101", seed=0, dimensionality=3
        ),
        "40efba11bf49faa7",
        id="bbob-noisy-f101",
        marks=_NEEDS_BBOB_NOISY,
    ),
    pytest.param(
        lambda: create_cec2005_task(fn_name="f1", seed=0, dimensionality=3),
        "dfbdf20255a3263f",
        id="cec2005-f1",
    ),
    pytest.param(
        lambda: create_cec2017_task(
            fn_name="cec2017_f1", seed=0, dimensionality=3
        ),
        "7291d184e56719ec",
        id="cec2017-f1",
    ),
    pytest.param(
        lambda: create_quadratic_task(
            dimensionality=4, seed=0, n_observations=None
        ),
        "32a8ec64a8595ee7",
        id="quadratic-square",
    ),
]


@pytest.mark.parametrize("build, expected", FACTORY)
def test_factory_hash_is_pinned(build, expected):
    with jax.enable_x64(True):
        task = build()
    assert task.hash == expected


def test_pre_constraints_file_loads_unchanged():
    with jax.enable_x64(True):
        task = Task.load(FIXTURE)
    assert task.constraints == ()
    assert task.hash == "32a8ec64a8595ee7"
    with open(FIXTURE, "rb") as f:
        stored = json.loads(f.readline().decode())
    assert set(stored) == PRE_CONSTRAINTS_HEADER_KEYS
    assert set(task.to_dict()) == PRE_CONSTRAINTS_HEADER_KEYS
