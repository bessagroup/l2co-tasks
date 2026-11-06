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

Re-keyed once, on purpose, for the dataset paper's protocol revision
(Section 5 item 10, 2026-11-05): ``tag["noise"]`` became a bool that
says whether the loss is stochastic, instead of the added-noise level.
That changed every benchmark value below that carried ``noise: 0.0`` or
``0.1`` (old values in the comments); bbob-noisy already carried
``True`` and quadratic carries no ``noise`` key, so those two kept
theirs.

Re-keyed a second time, on purpose, when ``global_min`` left the
task's identity (l2co-tasks ADR 0008, 2026-11-06): the hash is now the
tag plus constraints only, so an estimated minimum can be corrected
without changing the task. Every value below moved once (the value
before is in the comment next to it); the bbob sphere with added noise
also gained ``tag["noise_level"]`` in the same change. The
``global_min`` given to the task-level rows no longer enters the hash;
it stays in the table to show that.
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
    pytest.param(
        {}, None, "4f53cda18c2baa0c", id="empty-tag-no-gmin"
    ),  # was 1afc2ac49a13728b
    pytest.param(
        {"task_name": "simple", "seed": 0},
        0.0,
        "c42c78a0013db8fb",  # was abe00eaddcbeec8d
        id="simple",
    ),
    pytest.param(
        {"task_name": "nested", "dims": [2, 3], "meta": {"a": 1}},
        1.5,
        "d930f35a76d7fb34",  # was b56cd279e41ea485
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
        "7fa90fc1bae3ba5a",  # was cf276caf6843c7f2
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
        "f0206f5d4db278dd",  # was 5bd6dce2605aeeff
        id="bbob-sphere",
    ),
    pytest.param(
        lambda: create_bbob_task(
            fn_name="rastrigin", seed=0, dimensionality=3, noise=0.0
        ),
        "b53a4a511c264d6e",  # was 5cdda7a083965527
        id="bbob-rastrigin",
    ),
    pytest.param(
        lambda: create_bbob_task(
            fn_name="sphere", seed=0, dimensionality=3, noise=0.1
        ),
        "92a9802b5bf3ba41",  # was 4559d4190c9b5511
        id="bbob-sphere-noisy",
    ),
    pytest.param(
        lambda: create_bbob_noisy_task(
            fn_name="bbob_noisy_f101", seed=0, dimensionality=3
        ),
        "411b7960eeb9bc46",  # was 40efba11bf49faa7
        id="bbob-noisy-f101",
        marks=_NEEDS_BBOB_NOISY,
    ),
    pytest.param(
        lambda: create_cec2005_task(fn_name="f1", seed=0, dimensionality=3),
        "c18d2a37bf59e310",  # was 3b1c84f96f4b83af
        id="cec2005-f1",
    ),
    pytest.param(
        lambda: create_cec2005_task(fn_name="f4", seed=0, dimensionality=3),
        # Noisy by itself; tagged noise 0.0 (hash 0f6057f965f6135c)
        # before the protocol revision.
        "2a48b4eb5413283d",  # was 718e9e272b2e9880
        id="cec2005-f4",
    ),
    pytest.param(
        lambda: create_cec2017_task(
            fn_name="cec2017_f1", seed=0, dimensionality=3
        ),
        "ad68f378bb173ba6",  # was 4676ee91d04a3a4d
        id="cec2017-f1",
    ),
    pytest.param(
        lambda: create_quadratic_task(
            dimensionality=4, seed=0, n_observations=None
        ),
        "df5348e58d14aabb",  # was 32a8ec64a8595ee7
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
    assert task.loss_fn_true is None
    # was 32a8ec64a8595ee7, before global_min left identity (ADR 0008)
    assert task.hash == "df5348e58d14aabb"
    with open(FIXTURE, "rb") as f:
        stored = json.loads(f.readline().decode())
    assert set(stored) == PRE_CONSTRAINTS_HEADER_KEYS
    assert set(task.to_dict()) == PRE_CONSTRAINTS_HEADER_KEYS
