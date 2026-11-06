"""A PINN task's cached point set belongs to its tag and to nothing else.

The PINN factories cache their collocation points in an ``.npz`` file next
to ``dataset_path`` and write it only when it is missing. Since ADR 0008
the tag is the task's whole identity, so two tasks with different tags must
never share a cached file: a task rebuilt at the same ``dataset_path`` with
other point arguments, even ones that keep every array's shape (a grid
resolution, a domain range, a pkpd block boundary), must get its own
points. The same arguments must keep reusing one file, and a saved task
must still find its points after loading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from l2co_tasks import (
    Task,
    create_euler_task,
    create_helmholtz_task,
    create_inviscid_burgers_task,
    create_pde_task,
    create_pkpd_task,
    create_stokes_task,
    create_viscous_burgers_task,
)
from l2co_tasks._src._io import ensure_points_dataset

_SMALL_MLP = dict(hidden_size=4, num_layers=2)
_COUNTS = dict(num_res_points=16, num_ic_points=8, num_bc_points=8)
_PKPD_BLOCKS = ((0.0, 1.9, 8), (1.9, 4.0, 8), (4.0, 17.0, 8))

# (factory, base arguments, one point argument changed)
_CASES = {
    "pde": (
        create_pde_task,
        dict(
            pde_task_name="convection",
            beta=1.0,
            seed=0,
            x_range=(0.0, 1.0),
            t_range=(0.0, 1.0),
            xgrid_resolution=8,
            tgrid_resolution=8,
            num_ic_points=4,
            num_bc_points=4,
            num_res_points=8,
            hidden_size=4,
        ),
        {"num_res_points": 9},
    ),
    "helmholtz": (
        create_helmholtz_task,
        dict(a1=1.0, a2=4.0, k=1.0, seed=0, dim=2, num_res_points=16)
        | _SMALL_MLP,
        {"num_res_points": 17},
    ),
    "viscous_burgers": (
        create_viscous_burgers_task,
        dict(seed=0) | _COUNTS | _SMALL_MLP,
        {"num_ic_points": 9},
    ),
    "inviscid_burgers": (
        create_inviscid_burgers_task,
        dict(seed=0) | _COUNTS | _SMALL_MLP,
        {"num_ic_points": 9},
    ),
    "euler": (
        create_euler_task,
        dict(seed=0, stage="inviscid") | _COUNTS | _SMALL_MLP,
        {"num_res_points": 17},
    ),
    "stokes": (
        create_stokes_task,
        dict(seed=0, num_res_points=16, num_lid_points=8, num_wall_points=8)
        | _SMALL_MLP,
        {"num_lid_points": 9},
    ),
    "pkpd": (
        create_pkpd_task,
        dict(seed=0, point_allocation=_PKPD_BLOCKS) | _SMALL_MLP,
        {"point_allocation": ((0.0, 1.9, 9), (1.9, 4.0, 8), (4.0, 17.0, 8))},
    ),
}

# Changes that keep the shape of every stored array: a shape check alone
# cannot tell these point sets apart.
_SHAPE_PRESERVING = {
    "pde_xgrid_resolution": ("pde", {"xgrid_resolution": 9}),
    "pde_x_range": ("pde", {"x_range": (0.0, 2.0)}),
    "helmholtz_x_range": ("helmholtz", {"x_range": (0.0, 2.0)}),
    "euler_t_range": ("euler", {"t_range": (0.0, 2.0)}),
    "pkpd_block_bounds": (
        "pkpd",
        {
            "point_allocation": (
                (0.0, 2.5, 8),
                (2.5, 4.0, 8),
                (4.0, 17.0, 8),
            )
        },
    ),
}


def _points_file(task: Task) -> Path:
    return Path(task.dataset["dataset_path"])


def _points(task: Task) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in task.loaded_dataset.items()}


def _assert_same_points(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(_SHAPE_PRESERVING))
def test_shape_preserving_change_at_the_same_path_gets_its_own_points(
    tmp_path, case
):
    """Same path, other points of the same shapes: another file, own points."""
    family, changed = _SHAPE_PRESERVING[case]
    factory, args, _ = _CASES[family]
    shared = str(tmp_path / "shared" / family)
    first = factory(dataset_path=shared, **args)
    second = factory(dataset_path=shared, **{**args, **changed})
    alone = factory(
        dataset_path=str(tmp_path / "alone" / family), **{**args, **changed}
    )

    assert first.hash != second.hash
    assert _points_file(first) != _points_file(second)
    assert _points_file(first).exists() and _points_file(second).exists()
    # The second task reads the points its own arguments build, exactly
    # as if it had been the first task at that path.
    _assert_same_points(_points(second), _points(alone))
    assert any(
        not np.array_equal(v, _points(first)[k])
        for k, v in _points(second).items()
    )


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_CASES))
def test_other_point_counts_at_the_same_path_get_their_own_file(
    tmp_path, family
):
    """A changed point count no longer collides with the cached file."""
    factory, args, changed = _CASES[family]
    path = str(tmp_path / family)
    first = factory(dataset_path=path, **args)
    second = factory(dataset_path=path, **{**args, **changed})
    assert _points_file(first) != _points_file(second)
    assert _points_file(second).exists()


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_CASES))
def test_rebuild_with_the_same_arguments_reuses_the_cached_points(
    tmp_path, family
):
    """Same arguments at the same path: same task, cached file reused."""
    factory, args, _ = _CASES[family]
    path = str(tmp_path / family)
    first = factory(dataset_path=path, **args)
    written = _points_file(first).stat().st_mtime_ns
    second = factory(dataset_path=path, **args)
    assert first.hash == second.hash
    assert _points_file(second) == _points_file(first)
    assert _points_file(second).stat().st_mtime_ns == written
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        _points_file(first).name
    ]


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_CASES))
def test_the_cache_file_is_named_by_the_task_hash(tmp_path, family):
    """``<dataset_path>_<task hash>.npz``: findable from the task alone."""
    factory, args, _ = _CASES[family]
    path = str(tmp_path / family)
    task = factory(dataset_path=path, **args)
    assert _points_file(task) == Path(f"{path}_{task.hash}.npz")


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_CASES))
def test_a_saved_task_still_loads_its_points(tmp_path, family):
    """The stored reference stays relative to the ``.eqx`` file."""
    factory, args, _ = _CASES[family]
    task = factory(dataset_path=str(tmp_path / "data" / family), **args)
    (tmp_path / "tasks").mkdir()
    saved = Task.save(task, str(tmp_path / "tasks" / family))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    _assert_same_points(_points(loaded), _points(task))


def test_a_cached_file_with_other_shapes_is_still_refused(tmp_path):
    """The shape check stays as a backstop behind the tag-named file."""
    path = tmp_path / "points.npz"
    ensure_points_dataset(path, lambda: {"x": np.zeros((3, 1))})
    with pytest.raises(ValueError, match="dataset_path"):
        ensure_points_dataset(path, lambda: {"x": np.zeros((4, 1))})
