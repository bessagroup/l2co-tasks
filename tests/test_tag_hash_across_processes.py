"""A task's hash is the same in every process, even with a nested tag.

``Task.hash`` digests the printed tag. A dict inside a tag becomes a
frozenset, and a plain frozenset prints its items in an order set by the
process's string hash seed (``PYTHONHASHSEED``), so a tag holding a dict
of two or more keys hashed differently from one process to the next.
Comparing two tasks in one process cannot catch that: the seed is fixed
there. These tests hash the same tags in subprocesses under several
seeds. Without the fix, seeds 0, 1 and 2 split every multi-item shape
below.

A nested container with zero or one item always printed the same way;
its hash was pinned to the value recorded before the fix.

Every value below was re-pinned once when ``global_min`` left the
task's identity (l2co-tasks ADR 0008, 2026-11-06): these tasks carry no
``global_min``, but the hash used to digest a ``("global_min", None)``
entry, so every one of them moved (old values in the comments).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

SEEDS = (0, 1, 2)

_SCRIPT = """
import json
import jax.numpy as jnp
from l2co_tasks import Task

def loss(x):
    return jnp.sum(x**2)

TAGS = {
    "empty_dict": {"task_name": "nested", "params": {}},
    "one_key": {"task_name": "nested", "params": {"alpha": 1}},
    "one_frozen": {"task_name": "nested", "kinds": frozenset({"a"})},
    "two_keys": {
        "task_name": "nested",
        "seed": 0,
        "params": {"alpha": 1, "beta": 2},
    },
    "deep": {
        "task_name": "nested",
        "params": {"a": {"x": 1, "y": 2}, "b": [{"u": 1, "v": 2}]},
    },
    "frozen": {"task_name": "nested", "kinds": frozenset({"a", "b", "c"})},
}
print(json.dumps({
    name: Task(model=jnp.zeros(2), loss_fn=loss, tag=tag).hash
    for name, tag in TAGS.items()
}))
"""

#: Recorded before the fix, on ``21b95a1``: identical under every seed;
#: re-pinned for ADR 0008.
UNCHANGED = {
    "empty_dict": "5ba75059c6e733ec",  # was d95200dd15f5925e
    "one_key": "4f509e6aa4c8ee81",  # was 9be281e36f94efd3
    "one_frozen": "9f05ee3f6d5356da",  # was b72124bd14ecfb23
}

#: Before the fix each of these took one of several values, depending
#: on the seed; the fix keeps the one with the items in sorted order.
#: Re-pinned for ADR 0008.
CANONICAL = {
    "two_keys": "895c069d9b5ad0b2",  # was 89397f662c6cc552
    "deep": "439ea98f8b96e6b8",  # was 6922ebe31ee175be
    "frozen": "3ac9e475a7e76b9a",  # was cee0c82dcf1332ff
}


def _hashes(seed: int) -> dict[str, str]:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def hashes_by_seed() -> dict[int, dict[str, str]]:
    return {seed: _hashes(seed) for seed in SEEDS}


@pytest.mark.parametrize("shape", [*UNCHANGED, *CANONICAL])
def test_every_process_gives_the_same_hash(hashes_by_seed, shape):
    seen = {seed: hashes[shape] for seed, hashes in hashes_by_seed.items()}
    assert len(set(seen.values())) == 1, seen


@pytest.mark.parametrize("shape", list(UNCHANGED))
def test_a_nested_container_of_at_most_one_item_keeps_its_hash(
    hashes_by_seed, shape
):
    for hashes in hashes_by_seed.values():
        assert hashes[shape] == UNCHANGED[shape]


@pytest.mark.parametrize("shape", list(CANONICAL))
def test_a_nested_container_hashes_with_its_items_sorted(
    hashes_by_seed, shape
):
    for hashes in hashes_by_seed.values():
        assert hashes[shape] == CANONICAL[shape]
