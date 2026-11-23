"""The ``cec2013lsgo_unimodal`` and ``cec2013lsgo_multimodal`` task sets
split the ``cec2013lsgo`` suite by modality.

They exist so a policy that reads a ``unimodal`` task feature can be
deployed on LSGO with a per-function fill, without adding that key to the
LSGO tags (which would change every LSGO task hash). Their rows must be
unchanged copies of the full suite's rows, built by the same factory, so
each task hashes the same and a run over either half can write into the
``cec2013lsgo`` store.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import bbob_jax
import pytest
from omegaconf import DictConfig, OmegaConf

from l2co_tasks import create_tasks_experimentdata

# Modality per function, from the CEC 2013 LSGO report (Li et al. 2013):
# elliptic and Schwefel 1.2 based functions are unimodal; Rastrigin,
# Ackley and Rosenbrock based ones are multimodal.
_UNIMODAL = {1, 4, 7, 8, 11, 13, 14, 15}
_MULTIMODAL = {2, 3, 5, 6, 9, 10, 12}


def _load(name: str) -> DictConfig:
    text = (
        resources.files("l2co_tasks.conf.tasks")
        .joinpath(f"{name}.yaml")
        .read_text(encoding="utf-8")
    )
    return OmegaConf.create(text)


def _rows(name: str) -> list[dict]:
    return OmegaConf.to_container(_load(name))["experimentdata"]["input_data"]


def _fn_numbers(rows: list[dict]) -> set[int]:
    return {int(r["fn_name"].removeprefix("cec2013lsgo_f")) for r in rows}


def _key(row: dict) -> str:
    return repr(sorted(row.items()))


@pytest.mark.parametrize(
    ("half", "expected"),
    [
        ("cec2013lsgo_unimodal", _UNIMODAL),
        ("cec2013lsgo_multimodal", _MULTIMODAL),
    ],
)
def test_half_holds_the_functions_of_its_modality(half, expected):
    assert _fn_numbers(_rows(half)) == expected


def test_halves_are_disjoint_and_cover_the_full_suite_row_for_row():
    uni = [_key(r) for r in _rows("cec2013lsgo_unimodal")]
    multi = [_key(r) for r in _rows("cec2013lsgo_multimodal")]
    full = [_key(r) for r in _rows("cec2013lsgo")]
    assert not set(uni) & set(multi)
    assert sorted(uni + multi) == sorted(full)


@pytest.mark.parametrize(
    "half", ["cec2013lsgo_unimodal", "cec2013lsgo_multimodal"]
)
def test_half_shares_everything_but_rows_with_the_full_suite(half):
    """Same factory, task_kwargs, sampler and schema as ``cec2013lsgo``."""
    full = OmegaConf.to_container(_load("cec2013lsgo"))
    part = OmegaConf.to_container(_load(half))
    for cfg in (full, part):
        del cfg["experimentdata"]["input_data"]
    assert part == full


def _hashes(name: str, project_dir: Path) -> dict[str, str]:
    """``fn_name -> Task.hash`` for every task the shipped config builds."""
    cfg = OmegaConf.merge(OmegaConf.create({"seed": 0, "tasks": _load(name)}))
    data = create_tasks_experimentdata(cfg.tasks, project_dir=project_dir)
    return {
        sample.input_data["fn_name"]: sample.output_data["task"].hash
        for _, sample in data
    }


@pytest.mark.requires_f3dasm
@pytest.mark.skipif(
    not hasattr(bbob_jax, "cec2013lsgo_registry"),
    reason="installed bbob-jax predates the CEC 2013 LSGO suite",
)
def test_half_tasks_hash_as_the_full_suite_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    full = _hashes("cec2013lsgo", tmp_path / "full")
    uni = _hashes("cec2013lsgo_unimodal", tmp_path / "uni")
    multi = _hashes("cec2013lsgo_multimodal", tmp_path / "multi")
    assert len(full) == 15
    assert {**uni, **multi} == full
