"""CUTEst problems as tasks (l2co-tasks ADR 0004).

The first group runs everywhere: the canonical SIF-parameter string, the
committed ``global_min`` table and its refusals, the opener, and the
error when pycutest is missing. The second group needs pycutest and a
CUTEst installation (``CUTEST``, ``SIFDECODE`` and ``MASTSIF`` set) and
skips without them, as it does in CI.
"""

from __future__ import annotations

import csv
import sys

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from l2co_tasks import Task, create_cutest_task
from l2co_tasks._src import cutest_task
from l2co_tasks._src.cutest_task import (
    _TABLE_COLUMNS,
    _TABLE_PATH,
    _canonical_sif_params,
    _CutestOpener,
    _read_table,
    _sif_params_str,
    _table_global_min,
)


def _cutest_available() -> bool:
    try:
        import pycutest  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


needs_cutest = pytest.mark.skipif(
    not _cutest_available(),
    reason="needs pycutest and a CUTEst installation (CUTEST, SIFDECODE, "
    "MASTSIF)",
)


def _write_table(path, rows):
    # csv quoting matters: a SIF-parameter string like "M=2,N=2" has a comma.
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_TABLE_COLUMNS)
        writer.writerows(rows)
    return path


@pytest.fixture
def table(tmp_path, monkeypatch):
    """Point the factory at a temporary table; returns a writer."""

    def write(rows):
        path = _write_table(tmp_path / "cutest_global_min.csv", rows)
        monkeypatch.setattr(cutest_task, "_TABLE_PATH", path)
        return path

    return write


# Runs everywhere -------------------------------------------------------------


def test_sif_params_have_one_canonical_string():
    params = _canonical_sif_params({"ZETA": 20.0, "N": 10})
    assert params == (("N", 10), ("ZETA", 20.0))
    assert _sif_params_str(params) == "N=10,ZETA=20.0"
    assert _sif_params_str(_canonical_sif_params(None)) == ""


def test_shipped_table_parses():
    rows = _read_table(_TABLE_PATH)
    for (problem, _), row in rows.items():
        assert problem
        assert row.excluded or row.global_min is not None


def test_table_with_wrong_columns_is_refused(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("problem,global_min\nROSENBR,0.0\n")
    with pytest.raises(ValueError, match="expected"):
        _read_table(path)


def test_table_listing_a_size_twice_is_refused(tmp_path):
    row = ("ROSENBR", "", 2, "abc", 0.0, "soltn", "")
    path = _write_table(tmp_path / "dup.csv", [row, row])
    with pytest.raises(ValueError, match="twice"):
        _read_table(path)


def test_size_that_never_built_may_have_no_n(table):
    table([("JIMACK", "M=2,N=2", "", "sha", "", "", "fails to build: x")])
    with pytest.raises(ValueError, match="fails to build"):
        _table_global_min("JIMACK", "M=2,N=2", "sha")


def test_valued_row_without_n_is_refused(tmp_path):
    row = ("ROSENBR", "", "", "sha", 0.0, "soltn", "")
    path = _write_table(tmp_path / "no_n.csv", [row])
    with pytest.raises(ValueError, match="only an excluded row"):
        _read_table(path)


def test_table_row_is_looked_up_by_problem_and_size(table):
    table(
        [
            ("EXTROSNB", "N=5", 5, "sha5", 0.0, "soltn", ""),
            ("EXTROSNB", "N=10", 10, "sha10", 1.5, "estimate", ""),
        ]
    )
    assert _table_global_min("EXTROSNB", "N=10", "sha10") == 1.5


def test_problem_missing_from_table_is_refused(table):
    table([])
    with pytest.raises(ValueError, match="not in the global_min table"):
        _table_global_min("ROSENBR", "", "sha")


def test_excluded_problem_is_refused_with_its_reason(table):
    table([("INDEF", "N=10", 10, "sha", "", "", "unbounded below")])
    with pytest.raises(ValueError, match="unbounded below"):
        _table_global_min("INDEF", "N=10", "sha")


def test_changed_sif_file_is_refused(table):
    table([("ROSENBR", "", 2, "old-sha", 0.0, "soltn", "")])
    with pytest.raises(ValueError, match="has changed"):
        _table_global_min("ROSENBR", "", "new-sha")


def test_opener_is_hashable_and_pickles():
    opener = _CutestOpener("EXTROSNB", (("N", 10),))
    assert opener == _CutestOpener("EXTROSNB", (("N", 10),))
    assert hash(opener) == hash(_CutestOpener("EXTROSNB", (("N", 10),)))
    assert opener != _CutestOpener("EXTROSNB", (("N", 5),))
    assert cloudpickle.loads(cloudpickle.dumps(opener)) == opener


def test_missing_pycutest_explains_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycutest", None)
    with pytest.raises(ImportError, match="CUTEst tasks need pycutest"):
        create_cutest_task("ROSENBR", global_min=0.0)


# Needs CUTEst ----------------------------------------------------------------


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


@needs_cutest
def test_rosenbr_starts_at_its_prescribed_point(x64):
    task = create_cutest_task("ROSENBR", global_min=0.0)
    np.testing.assert_array_equal(task.model, [-1.2, 1.0])
    assert float(jax.jit(task.loss_fn)(task.model)) == pytest.approx(24.2)
    np.testing.assert_allclose(
        jax.grad(task.loss_fn)(task.model), [-215.6, -88.0], rtol=1e-12
    )
    assert task.global_min == 0.0
    assert not task.pass_rng
    assert task.dataset is None


@needs_cutest
def test_tag_is_flat_and_complete():
    task = create_cutest_task("EXTROSNB", {"N": 5}, global_min=0.0)
    assert task.tag["task_name"] == "cutest"
    assert task.tag["fn_name"] == "EXTROSNB"
    assert task.tag["dimensionality"] == task.dimensionality == 5
    assert task.tag["noise"] == 0.0
    assert task.tag["sif_params"] == "N=5"
    assert len(task.tag["sif_sha256"]) == 64
    assert task.tag["cutest_objective"] == "sum of squares"
    assert task.tag["cutest_regular"] is True
    assert not any(isinstance(v, dict) for v in task.tag.values())


@needs_cutest
def test_two_sizes_of_one_problem_are_different_tasks(x64):
    small = create_cutest_task("EXTROSNB", {"N": 5}, global_min=0.0)
    large = create_cutest_task("EXTROSNB", {"N": 10}, global_min=0.0)
    assert small.dimensionality == 5
    assert large.dimensionality == 10
    assert small.hash != large.hash
    assert np.isfinite(float(small.loss_fn(small.model)))
    assert np.isfinite(float(large.loss_fn(large.model)))


@needs_cutest
def test_hash_is_stable_across_builds():
    first = create_cutest_task("ROSENBR", global_min=0.0)
    second = create_cutest_task("ROSENBR", global_min=0.0)
    assert first.hash == second.hash
    assert first.loss_fn == second.loss_fn


@needs_cutest
@pytest.mark.parametrize(
    ("problem", "kind"), [("ALLINIT", "bound"), ("BOX2", "fixed")]
)
def test_constrained_problem_is_refused(problem, kind):
    with pytest.raises(ValueError, match=f"has {kind} constraints"):
        create_cutest_task(problem, global_min=0.0)


@needs_cutest
def test_global_min_from_the_table(table):
    sha = cutest_task._sif_sha256("ROSENBR")
    table([("ROSENBR", "", 2, sha, 0.0, "soltn", "")])
    assert create_cutest_task("ROSENBR").global_min == 0.0


@needs_cutest
def test_stale_table_row_is_refused(table):
    table([("ROSENBR", "", 2, "0" * 64, 0.0, "soltn", "")])
    with pytest.raises(ValueError, match="has changed"):
        create_cutest_task("ROSENBR")


@needs_cutest
def test_non_finite_global_min_is_refused():
    with pytest.raises(ValueError, match="finite"):
        create_cutest_task("ROSENBR", global_min=float("nan"))


@needs_cutest
def test_save_load_round_trip(eqx_path, x64):
    task = create_cutest_task("EXTROSNB", {"N": 5}, global_min=0.0)
    loaded = Task.load(Task.save(task, str(eqx_path)))
    assert loaded == task
    assert loaded.hash == task.hash
    np.testing.assert_array_equal(loaded.model, task.model)
    assert float(loaded.loss_fn(loaded.model)) == float(
        task.loss_fn(task.model)
    )


@needs_cutest
def test_vmapped_loss_matches_a_loop(x64):
    task = create_cutest_task("EXTROSNB", {"N": 5}, global_min=0.0)
    xs = task.model + jax.random.normal(jax.random.key(0), (4, 5))
    batched = jax.jit(jax.vmap(task.loss_fn))(xs)
    looped = jnp.stack([task.loss_fn(x) for x in xs])
    np.testing.assert_array_equal(batched, looped)
