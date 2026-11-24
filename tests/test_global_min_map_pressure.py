"""``estimate_global_min`` drops JAX's caches under memory-map pressure.

Every compiled CPU program holds memory mappings and Linux caps them per
process (``vm.max_map_count``). Building many tasks in one process
compiles a fresh search per task, so before each estimate the module
counts the process's mappings and, above a share of the cap, calls
``jax.clear_caches()``. The /proc files are faked here through the
module's paths to them.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import pytest

from l2co_tasks import Task, estimate_global_min
from l2co_tasks._src import global_min as gm

_KW = dict(seed=0, n_restarts=2, n_steps=50, lr=0.1, lbfgs_steps=20)


def _task():
    return Task(
        model=jnp.full(2, 0.5),
        loss_fn=lambda m: jnp.sum((m - 2.0) ** 2),
        tag={"task_name": "pull_to_two"},
    )


def _fake_proc(monkeypatch, tmp_path, *, n_maps, max_maps):
    maps = tmp_path / "maps"
    maps.write_text("r-xp\n" * n_maps)
    cap = tmp_path / "max_map_count"
    cap.write_text(f"{max_maps}\n")
    monkeypatch.setattr(gm, "_MAPS_PATH", maps)
    monkeypatch.setattr(gm, "_MAX_MAP_COUNT_PATH", cap)


@pytest.fixture
def clear_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append(1))
    return calls


def test_clears_caches_above_seventy_percent(
    monkeypatch, tmp_path, clear_calls, caplog
):
    _fake_proc(monkeypatch, tmp_path, n_maps=71, max_maps=100)
    with caplog.at_level(logging.INFO, logger=gm.__name__):
        estimate_global_min(_task(), **_KW)
    assert clear_calls == [1]
    assert "71" in caplog.text and "100" in caplog.text


def test_leaves_caches_alone_at_seventy_percent_or_below(
    monkeypatch, tmp_path, clear_calls
):
    _fake_proc(monkeypatch, tmp_path, n_maps=70, max_maps=100)
    estimate_global_min(_task(), **_KW)
    assert clear_calls == []


@pytest.mark.parametrize("missing", ["maps", "max_map_count"])
def test_is_a_no_op_without_proc(monkeypatch, tmp_path, clear_calls, missing):
    _fake_proc(monkeypatch, tmp_path, n_maps=99, max_maps=100)
    (tmp_path / missing).unlink()
    estimate_global_min(_task(), **_KW)
    assert clear_calls == []


def test_a_forced_clear_leaves_the_estimate_unchanged(monkeypatch, tmp_path):
    plain = estimate_global_min(_task(), **_KW)
    _fake_proc(monkeypatch, tmp_path, n_maps=100, max_maps=100)
    assert estimate_global_min(_task(), **_KW) == plain
