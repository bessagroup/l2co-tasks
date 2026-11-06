"""Tests for ``create_inviscid_burgers_task`` (two-network model)."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from l2co_tasks import Task, create_inviscid_burgers_task


def _build(tmp_path):
    """Build a tiny inviscid-Burgers task."""
    return create_inviscid_burgers_task(
        seed=0,
        dataset_path=str(tmp_path / "ib"),
        num_res_points=16,
        num_ic_points=8,
        num_bc_points=8,
        hidden_size=4,
        num_layers=2,
    )


@pytest.mark.slow
def test_fields_and_dataset(tmp_path):
    task = _build(tmp_path)
    assert isinstance(task, Task)
    assert task.name == "inviscid_burgers"
    assert task.tag["loss_fn"] == "inviscid_burgers_loss"
    ds = task.loaded_dataset
    for key in ("x_res", "x_ic", "x_bc"):
        assert ds[key].shape[1] == 2


@pytest.mark.slow
def test_dimensionality_counts_both_networks(tmp_path):
    """The model bundles a solution and a flux network."""
    task = _build(tmp_path)
    assert task.model.names == ("u", "flux")
    per_net = [
        sum(
            p.size
            for p in jtu.tree_leaves(eqx.filter(net, eqx.is_inexact_array))
        )
        for net in task.model.nets
    ]
    assert len(per_net) == 2
    assert task.dimensionality == sum(per_net) > 0


@pytest.mark.slow
def test_loss_and_round_trip(tmp_path):
    task = _build(tmp_path)
    ds = task.loaded_dataset
    before = float(task.loss_fn(task.model, **ds))
    assert jnp.isfinite(before)
    saved = Task.save(task, str(tmp_path / "task"))
    loaded = Task.load(saved)
    assert loaded.hash == task.hash
    after = float(loaded.loss_fn(loaded.model, **loaded.loaded_dataset))
    assert after == pytest.approx(before, rel=1e-5)


_IB_ARGS = dict(
    seed=0,
    num_res_points=16,
    num_ic_points=8,
    num_bc_points=8,
    hidden_size=4,
    num_layers=2,
)


@pytest.mark.slow
@pytest.mark.parametrize(
    "changed",
    [
        {"seed": 1},
        {"num_res_points": 17},
        {"num_ic_points": 9},
        {"num_bc_points": 9},
    ],
)
def test_inviscid_burgers_hash_follows_every_argument_that_changes_the_task(
    tmp_path, changed
):
    """Same arguments, same task; one changed argument, another task."""
    base = create_inviscid_burgers_task(
        dataset_path=str(tmp_path / "a"), **_IB_ARGS
    )
    again = create_inviscid_burgers_task(
        dataset_path=str(tmp_path / "b"), **_IB_ARGS
    )
    other = create_inviscid_burgers_task(
        dataset_path=str(tmp_path / "c"), **{**_IB_ARGS, **changed}
    )
    assert base.hash == again.hash
    assert base.hash != other.hash
