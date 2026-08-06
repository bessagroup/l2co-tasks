"""Tests for ``create_cec2013lsgo_task`` (CEC 2013 LSGO wrapper).

Covers the specifics beyond the shared contract battery: the fixed native
dimension (905 for the overlapping F13/F14, 1000 otherwise), the
analytical ``global_min = 0``, the F14 "optimum not attained" flag, and
the ``ndim`` validation. Skips when the installed ``bbob-jax`` predates
the LSGO suite.
"""

from __future__ import annotations

import bbob_jax
import pytest

from l2co_tasks import create_cec2013lsgo_task

pytestmark = pytest.mark.skipif(
    not hasattr(bbob_jax, "cec2013lsgo_registry"),
    reason="installed bbob-jax predates the CEC 2013 LSGO suite",
)


@pytest.mark.parametrize(
    "fn_name,expected_dim",
    [
        ("cec2013lsgo_f1", 1000),
        ("cec2013lsgo_f7", 1000),
        ("cec2013lsgo_f13", 905),
        ("cec2013lsgo_f14", 905),
        ("cec2013lsgo_f15", 1000),
    ],
)
def test_native_dimension_and_global_min(fn_name, expected_dim):
    """The task is built at the native dim with analytical ``global_min=0``."""
    task = create_cec2013lsgo_task(fn_name)
    assert task.dimensionality == expected_dim
    assert task.global_min == 0.0
    assert task.tag["fn_name"] == fn_name
    assert task.pass_rng is False


def test_f14_optimum_not_attained_flag():
    """F14's conflicting overlap makes 0 a lower bound, not attained."""
    assert create_cec2013lsgo_task("cec2013lsgo_f14").tag[
        "optimum_attained"
    ] is False
    assert create_cec2013lsgo_task("cec2013lsgo_f1").tag[
        "optimum_attained"
    ] is True


def test_wrong_dimension_raises():
    """A non-native ``dimensionality`` is rejected by the maker."""
    with pytest.raises(ValueError):
        create_cec2013lsgo_task("cec2013lsgo_f1", dimensionality=500)


def test_seed_does_not_change_the_instance():
    """The deterministic instance is identical across seeds (up to the tag)."""
    import jax.numpy as jnp
    import jax.random as jr

    a = create_cec2013lsgo_task("cec2013lsgo_f1", seed=0)
    b = create_cec2013lsgo_task("cec2013lsgo_f1", seed=123)
    x = jr.uniform(jr.key(0), (a.dimensionality,))
    assert jnp.allclose(a.loss_fn(x), b.loss_fn(x))
