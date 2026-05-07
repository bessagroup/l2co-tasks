"""Tests for ``create_pde_task`` covering convection / reaction / wave."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Task, create_pde_task


def _build_pde_task(tmp_path, pde_name, hidden_size=4, seed=0):
    """Tiny grids so each test evaluates quickly."""
    common = dict(
        seed=seed,
        dataset_path=str(tmp_path / pde_name),
        x_range=(0.0, 2.0 * float(jnp.pi))
        if pde_name != "wave"
        else (0.0, 1.0),
        t_range=(0.0, 1.0),
        xgrid_resolution=8,
        tgrid_resolution=8,
        num_ic_points=4,
        num_bc_points=4,
        num_res_points=8,
        hidden_size=hidden_size,
    )
    if pde_name == "convection":
        return create_pde_task(pde_task_name="convection", beta=1.0, **common)
    if pde_name == "reaction":
        return create_pde_task(pde_task_name="reaction", rho=1.0, **common)
    return create_pde_task(pde_task_name="wave", beta=2.0, **common)


@pytest.mark.slow
@pytest.mark.parametrize("pde_name", ["convection", "reaction", "wave"])
def test_create_pde_task_fields_and_dataset(tmp_path, pde_name):
    task = _build_pde_task(tmp_path, pde_name)
    assert isinstance(task, Task)
    assert task.name == pde_name
    assert task.tag["loss_fn"] == f"{pde_name}_pde_loss"
    if pde_name == "reaction":
        assert task.tag["rho"] == 1.0
    else:
        assert task.tag["beta"] == (1.0 if pde_name == "convection" else 2.0)

    ds = task.loaded_dataset
    for key in (
        "x_res",
        "t_res",
        "x_ic",
        "t_ic",
        "x_bc_left",
        "x_bc_right",
        "t_bc",
    ):
        assert key in ds


@pytest.mark.slow
@pytest.mark.parametrize("pde_name", ["convection", "reaction", "wave"])
def test_create_pde_task_loss_and_round_trip(tmp_path, pde_name):
    task = _build_pde_task(tmp_path, pde_name)
    ds = task.loaded_dataset
    kwargs = dict(
        x_res=ds["x_res"],
        t_res=ds["t_res"],
        x_ic=ds["x_ic"],
        t_ic=ds["t_ic"],
        x_bc_left=ds["x_bc_left"],
        x_bc_right=ds["x_bc_right"],
        t_bc=ds["t_bc"],
    )
    before = float(task.loss_fn(task.model, **kwargs))
    assert jnp.isfinite(before)

    saved = Task.save(task, str(tmp_path / "task.eqx"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    after = float(loaded.loss_fn(loaded.model, **kwargs))
    assert after == pytest.approx(before, rel=1e-5)
