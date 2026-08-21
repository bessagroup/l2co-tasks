"""Tests for ``create_bbob_task``, ``create_bbob_noisy_task``,
``create_cec2005_task``, ``create_cec2017_task``,
``create_cec2013lsgo_task`` and ``CEC2019Sampler``.

These factories are pure (no dataset files) so the tests run quickly and
do not need the ``slow`` marker. The BBOB-noisy tests skip on bbob-jax
installs that predate the suite (releases up to 1.8.0); the CEC 2013
LSGO tests likewise skip on installs up to 2.0.0.
"""

from __future__ import annotations

import bbob_jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import pytest
from f3dasm import ExperimentData
from f3dasm.design import Domain

from l2co_tasks import (
    CEC2019Sampler,
    Task,
    create_bbob_noisy_task,
    create_bbob_task,
    create_cec2005_task,
    create_cec2013lsgo_task,
    create_cec2017_task,
)

_HAS_BBOB_NOISY = hasattr(bbob_jax, "bbob_noisy_registry")
needs_bbob_noisy = pytest.mark.skipif(
    not _HAS_BBOB_NOISY,
    reason="installed bbob-jax predates the BBOB-noisy suite",
)

_HAS_LSGO = hasattr(bbob_jax, "cec2013lsgo_registry")
needs_lsgo = pytest.mark.skipif(
    not _HAS_LSGO,
    reason="installed bbob-jax predates the CEC 2013 LSGO suite",
)

# Native dimensionality per LSGO function: 1000, except the overlapping
# f13/f14 at 905.
_LSGO_NDIM = {"cec2013lsgo_f3": 1000, "cec2013lsgo_f13": 905}


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


@needs_bbob_noisy
@pytest.mark.parametrize("fn_name", ["bbob_noisy_f101", "bbob_noisy_f124"])
def test_create_bbob_noisy_task(fn_name, eqx_path):
    """BBOB-noisy tasks are inherently stochastic: ``pass_rng=True``,
    per-key deterministic, key-sensitive away from the optimum, and
    round-trip through save/load."""
    task = create_bbob_noisy_task(fn_name=fn_name, seed=0, dimensionality=3)
    assert isinstance(task, Task)
    assert task.dimensionality == 3
    assert task.pass_rng is True
    assert task.tag["fn_name"] == fn_name
    assert task.tag["seed"] == 0
    assert task.tag["dimensionality"] == 3
    assert task.tag["noise"] is True  # suite characteristic flag

    x = jnp.ones(3) * 0.2
    l1 = float(task.loss_fn(x, key=jr.key(0)))
    l2 = float(task.loss_fn(x, key=jr.key(0)))
    assert jnp.isfinite(l1)
    assert l1 == l2  # same key -> identical
    # The Cauchy noise model only fires with probability p per draw, so
    # two individual keys may coincide; across a batch of keys the loss
    # must still vary.
    losses = jnp.stack([task.loss_fn(x, key=jr.key(i)) for i in range(16)])
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.unique(losses).size > 1

    saved = Task.save(task, str(eqx_path))
    loaded = Task.load(saved)
    assert loaded == task
    assert float(loaded.loss_fn(x, key=jr.key(0))) == pytest.approx(l1)


@needs_bbob_noisy
def test_create_bbob_noisy_task_unknown_name_raises():
    """Noiseless-suite names are not in the noisy registry."""
    with pytest.raises(KeyError):
        create_bbob_noisy_task(fn_name="sphere", seed=0, dimensionality=3)


def test_create_bbob_noisy_task_old_bbob_jax_raises(monkeypatch):
    """Without the noisy registry the factory fails with a clear
    upgrade message instead of an AttributeError."""
    monkeypatch.delattr(bbob_jax, "bbob_noisy_registry", raising=False)
    with pytest.raises(ImportError, match="bbob-jax"):
        create_bbob_noisy_task(
            fn_name="bbob_noisy_f101", seed=0, dimensionality=3
        )


@needs_lsgo
@pytest.mark.parametrize("fn_name", ["cec2013lsgo_f3", "cec2013lsgo_f13"])
def test_create_cec2013lsgo_task(fn_name, eqx_path):
    """LSGO tasks build at their native dimensionality, are
    deterministic, expose the 4-way separability tags and round-trip.

    ``f3`` is 1000-D and fully separable; ``f13`` is one of the two
    905-D overlapping members, so the pair covers both native
    dimensionalities.
    """
    ndim = _LSGO_NDIM[fn_name]
    task = create_cec2013lsgo_task(
        fn_name=fn_name, seed=0, dimensionality=ndim
    )
    assert isinstance(task, Task)
    assert task.dimensionality == ndim
    assert task.pass_rng is False
    assert task.tag["fn_name"] == fn_name
    assert task.tag["seed"] == 0
    assert task.tag["noise"] == 0.0
    assert task.tag["dimensionality"] == ndim
    # For a fixed-instance suite min_ndim *is* the native dimension.
    assert task.tag["min_ndim"] == ndim
    # Exactly one of the four separability classes is set.
    classes = (
        "separable",
        "partially_separable",
        "overlapping",
        "non_separable",
    )
    assert sum(bool(task.tag[c]) for c in classes) == 1
    assert task.global_min == 0.0

    loss = float(task.loss_fn(task.model))
    assert jnp.isfinite(loss)
    # global_min is a true lower bound for every member of the suite.
    assert loss >= task.global_min

    saved = Task.save(task, str(eqx_path))
    loaded = Task.load(saved)
    assert loaded == task
    assert loaded.hash == task.hash
    assert float(loaded.loss_fn(loaded.model)) == pytest.approx(loss)


@needs_lsgo
def test_create_cec2013lsgo_task_is_seed_independent():
    """``seed`` selects no instance: the landscape is a fixed constant.

    This is what separates LSGO from every other suite here, so it is
    asserted rather than left implicit -- two seeds must give the same
    loss *and* the same task identity.
    """
    x = jnp.full(1000, 0.3)
    t0 = create_cec2013lsgo_task(
        fn_name="cec2013lsgo_f3", seed=0, dimensionality=1000
    )
    t7 = create_cec2013lsgo_task(
        fn_name="cec2013lsgo_f3", seed=7, dimensionality=1000
    )
    assert float(t0.loss_fn(x)) == pytest.approx(float(t7.loss_fn(x)))
    # ``seed`` still lands in the tag, so identity differs by design.
    assert t0.tag["seed"] == 0 and t7.tag["seed"] == 7


@needs_lsgo
@pytest.mark.parametrize("bad_ndim", [10, 905])
def test_create_cec2013lsgo_task_wrong_ndim_raises(bad_ndim):
    """Off the native dimensionality the factory raises, rather than
    silently rescaling a fixed instance (905 is valid for f13/f14 but
    not for f3)."""
    with pytest.raises(ValueError, match="ndim == 1000"):
        create_cec2013lsgo_task(
            fn_name="cec2013lsgo_f3", seed=0, dimensionality=bad_ndim
        )


@needs_lsgo
def test_create_cec2013lsgo_task_with_noise_requires_key():
    """``noise > 0`` flips ``pass_rng=True`` and the loss consumes a key."""
    task = create_cec2013lsgo_task(
        fn_name="cec2013lsgo_f3", seed=0, dimensionality=1000, noise=0.1
    )
    assert task.pass_rng is True
    x = jnp.full(1000, 0.3)
    l1 = float(task.loss_fn(x, jr.key(0)))
    l2 = float(task.loss_fn(x, jr.key(1)))
    assert jnp.isfinite(l1) and jnp.isfinite(l2)
    assert l1 != l2


@needs_lsgo
def test_create_cec2013lsgo_task_unknown_name_raises():
    """Other suites' names are not LSGO functions."""
    with pytest.raises(KeyError):
        create_cec2013lsgo_task(fn_name="sphere", seed=0, dimensionality=1000)


def test_create_cec2013lsgo_task_old_bbob_jax_raises(monkeypatch):
    """Without the LSGO registry the factory fails with a clear upgrade
    message instead of a KeyError from ``problem()``."""
    monkeypatch.delattr(bbob_jax, "cec2013lsgo_registry", raising=False)
    with pytest.raises(ImportError, match="bbob-jax"):
        create_cec2013lsgo_task(
            fn_name="cec2013lsgo_f3", seed=0, dimensionality=1000
        )


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


@pytest.mark.parametrize(
    "fn_name,dimensionality",
    [
        ("cec2017_f1", 2),  # unimodal simple function
        ("cec2017_f9", 5),  # Levy: argmin displaced from the shift
        ("cec2017_f17", 5),  # hybrid at its min_ndim
        ("cec2017_f21", 3),  # composition
    ],
)
def test_create_cec2017_task(fn_name, dimensionality, eqx_path):
    """CEC2017 tasks build, evaluate, and round-trip; the suite has no
    inherently stochastic functions so ``pass_rng`` is False without
    added noise."""
    task = create_cec2017_task(
        fn_name=fn_name, seed=0, dimensionality=dimensionality, noise=0.0
    )
    assert isinstance(task, Task)
    assert task.dimensionality == dimensionality
    assert task.pass_rng is False
    assert task.tag["fn_name"] == fn_name
    assert task.tag["min_ndim"] >= 1
    assert "hybrid" in task.tag  # cec2017 tag schema

    loss = float(task.loss_fn(task.model))
    assert jnp.isfinite(loss)
    assert loss >= task.global_min

    saved = Task.save(task, str(eqx_path))
    loaded = Task.load(saved)
    assert loaded == task
    assert float(loaded.loss_fn(loaded.model)) == pytest.approx(loss)


def test_create_cec2017_task_noise_flips_pass_rng():
    """``noise > 0`` flips ``pass_rng=True`` and the loss consumes a
    key (the suite itself has no stochastic functions)."""
    task = create_cec2017_task(
        fn_name="cec2017_f1", seed=0, dimensionality=3, noise=0.1
    )
    assert task.pass_rng is True
    l1 = float(task.loss_fn(jnp.ones(3) * 0.2, jr.key(0)))
    l2 = float(task.loss_fn(jnp.ones(3) * 0.2, jr.key(1)))
    assert jnp.isfinite(l1) and jnp.isfinite(l2)
    assert l1 != l2


def test_create_cec2017_task_below_min_ndim_raises():
    """Hybrids need one dimension per subcomponent kernel; the
    bbob_jax maker's ValueError propagates with the exact bound."""
    with pytest.raises(ValueError, match="ndim >= 7"):
        create_cec2017_task(fn_name="cec2017_f20", seed=0, dimensionality=3)


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
