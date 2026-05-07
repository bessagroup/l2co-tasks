"""Tests for ``create_bbob_task``, ``create_cec2005_task``, ``CEC2019Sampler``.

These factories are pure (no dataset files) so the tests run quickly and
do not need the ``slow`` marker.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import pytest
from f3dasm import ExperimentData
from f3dasm.design import Domain

from l2co_tasks import (
    CEC2019Sampler,
    Task,
    create_bbob_task,
    create_cec2005_task,
)


@pytest.mark.parametrize("fn_name", ["sphere", "rastrigin", "rosenbrock"])
@pytest.mark.parametrize("dimensionality", [2, 5])
def test_create_bbob_task_noiseless(fn_name, dimensionality, eqx_path):
    """Noiseless BBOB tasks build, have ``pass_rng=False``, evaluate and
    round-trip."""
    task = create_bbob_task(
        fn_name=fn_name, seed=0, dimensionality=dimensionality, noise=0.0
    )
    assert isinstance(task, Task)
    assert task.dimensionality == dimensionality
    assert task.pass_rng is False
    assert task.tag["fn_name"] == fn_name
    assert task.tag["seed"] == 0
    assert task.tag["noise"] == 0.0
    assert task.tag["dimensionality"] == dimensionality

    loss = float(task.loss_fn(task.model))
    assert jnp.isfinite(loss)

    saved = Task.save(task, str(eqx_path))
    loaded = Task.load(saved)
    assert loaded == task
    assert loaded.hash == task.hash
    assert float(loaded.loss_fn(loaded.model)) == pytest.approx(loss)


def test_create_bbob_task_with_noise_requires_key():
    """``noise > 0`` flips ``pass_rng=True`` and the loss consumes a key."""
    task = create_bbob_task(
        fn_name="sphere", seed=0, dimensionality=3, noise=0.1
    )
    assert task.pass_rng is True

    l1 = float(task.loss_fn(task.model, jr.key(0)))
    l2 = float(task.loss_fn(task.model, jr.key(1)))
    assert jnp.isfinite(l1) and jnp.isfinite(l2)
    # At x=0 the sphere value is 0, so the multiplicative noise term is
    # also 0 for both keys; we only assert the call works. For any
    # non-zero point, different keys should produce different values:
    l3 = float(task.loss_fn(jnp.ones(3) * 0.2, jr.key(0)))
    l4 = float(task.loss_fn(jnp.ones(3) * 0.2, jr.key(1)))
    assert l3 != l4


@pytest.mark.parametrize("fn_name", ["f1", "f4"])
def test_create_cec2005_task(fn_name):
    """CEC2005: ``f1`` deterministic (noise=0 ⇒ pass_rng False);
    ``f4`` is inherently stochastic and forces ``pass_rng=True``."""
    task = create_cec2005_task(
        fn_name=fn_name, seed=0, dimensionality=3, noise=0.0
    )
    assert isinstance(task, Task)
    assert task.dimensionality == 3
    assert task.tag["fn_name"] == fn_name
    if fn_name in {"f4", "f17", "f24", "f25"}:
        assert task.pass_rng is True
    else:
        assert task.pass_rng is False


@pytest.mark.requires_f3dasm
def test_cec2019_sampler_expands_dim_mapping():
    """The sampler emits one row per input category with the hard-coded
    dimensionality from ``dim_mapping``."""
    fns = ["F12019", "F22019", "F42019"]
    d = Domain()
    d.add_category("fn_name", categories=fns)
    ed = ExperimentData(domain=d, input_data=pd.DataFrame({"fn_name": fns}))

    out = CEC2019Sampler().call(ed)
    rows = [es.input_data for _, es in out]
    dim_by_name = {r["fn_name"]: r["dimensionality"] for r in rows}
    assert dim_by_name == {"F12019": 9, "F22019": 16, "F42019": 10}
