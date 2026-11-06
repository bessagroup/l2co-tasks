"""PINN factories refuse a cached point set that no longer fits the task.

The PINN factories cache their collocation points at ``dataset_path`` +
``_{seed}`` and write the file only when it is missing. Since ADR 0008 the
tag (which records the point counts) is the task's whole identity, so
silently reusing a cached file built with other counts would make the tag
claim points the task does not have. A rebuild at the same path with a
count that changes the stored arrays must fail loudly; a rebuild with the
same arguments must keep reusing the file.
"""

from __future__ import annotations

import pytest

from l2co_tasks import (
    create_euler_task,
    create_helmholtz_task,
    create_inviscid_burgers_task,
    create_pde_task,
    create_pkpd_task,
    create_stokes_task,
    create_viscous_burgers_task,
)

_SMALL_MLP = dict(hidden_size=4, num_layers=2)
_COUNTS = dict(num_res_points=16, num_ic_points=8, num_bc_points=8)

# (factory, base arguments, one argument changed so the stored arrays change)
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
        dict(
            seed=0,
            point_allocation=((0.0, 1.9, 8), (1.9, 4.0, 8), (4.0, 17.0, 8)),
        )
        | _SMALL_MLP,
        {"point_allocation": ((0.0, 1.9, 9), (1.9, 4.0, 8), (4.0, 17.0, 8))},
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_CASES))
def test_rebuild_with_other_counts_at_the_same_path_is_refused(
    tmp_path, family
):
    """A changed point count must not silently reuse the cached points."""
    factory, args, changed = _CASES[family]
    path = str(tmp_path / family)
    factory(dataset_path=path, **args)
    with pytest.raises(ValueError, match="dataset_path"):
        factory(dataset_path=path, **{**args, **changed})


@pytest.mark.slow
@pytest.mark.parametrize("family", sorted(_CASES))
def test_rebuild_with_the_same_arguments_reuses_the_cached_points(
    tmp_path, family
):
    """Same arguments at the same path: same task, cached file reused."""
    factory, args, _ = _CASES[family]
    path = str(tmp_path / family)
    first = factory(dataset_path=path, **args)
    second = factory(dataset_path=path, **args)
    assert first.hash == second.hash
