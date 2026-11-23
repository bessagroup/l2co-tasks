"""The ``joint_selection_test_le50`` task set is ``joint_selection_test``
restricted to problems with at most 50 dimensions.

Its rows must be unchanged copies of the full set's rows (same factory,
same ``task_kwargs``, same row), so every task hash is the same and a run
over it can append into the ``joint_selection_test`` store.
"""

from __future__ import annotations

from collections import Counter
from importlib import resources

from omegaconf import OmegaConf

_MAX_DIM = 50


def _parts(name: str) -> list[dict]:
    text = (
        resources.files("l2co_tasks.conf.tasks")
        .joinpath(f"{name}.yaml")
        .read_text(encoding="utf-8")
    )
    return OmegaConf.to_container(OmegaConf.create(text))["parts"]


def _dim(row: dict) -> int:
    return int(row.get("ambient_dim", row.get("dimensionality")))


def _keyed_rows(name: str) -> list[tuple]:
    """``(factory, task_kwargs, row)`` for every row, as comparable tuples."""
    out = []
    for part in _parts(name):
        factory = part["data_generator"]["_target_"]
        kwargs = repr(sorted((part.get("task_kwargs") or {}).items()))
        out += [
            (factory, kwargs, repr(sorted(row.items())))
            for row in part["input_data"]
        ]
    return out


def test_le50_is_the_full_test_set_at_or_below_50_dimensions():
    full = []
    for part in _parts("joint_selection_test"):
        factory = part["data_generator"]["_target_"]
        kwargs = repr(sorted((part.get("task_kwargs") or {}).items()))
        full += [
            (factory, kwargs, repr(sorted(row.items())))
            for row in part["input_data"]
            if _dim(row) <= _MAX_DIM
        ]
    assert _keyed_rows("joint_selection_test_le50") == full


def test_le50_has_62_problems_and_no_embedded_part():
    counts = Counter(f for f, _, _ in _keyed_rows("joint_selection_test_le50"))
    assert sum(counts.values()) == 62
    assert "l2co_tasks.create_embedded_bbob_task" not in counts
    assert all(
        _dim(r) <= _MAX_DIM
        for p in _parts("joint_selection_test_le50")
        for r in p["input_data"]
    )
