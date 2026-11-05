"""The ``noise`` tag: one boolean rule across every benchmark suite.

A benchmark task is *noisy* when its loss is stochastic: the function is
noisy by itself (bbob-jax's own per-function ``noise`` characteristic --
the whole BBOB-noisy suite and CEC 2005 F4, F17, F24 and F25) or
multiplicative noise is added (``noise > 0``). ``tag["noise"]`` and
``pass_rng`` both record exactly that, as a Python ``bool``. The added
level is a builder argument only: it is not in the tag, so it is not
in the hash either (dataset paper protocol revision, Section 5 item 10,
decided 2026-11-05).
"""

from __future__ import annotations

import bbob_jax
import numpy as np
import pytest

from l2co_tasks import (
    Task,
    create_bbob_noisy_task,
    create_bbob_task,
    create_cec2005_task,
    create_cec2017_task,
    create_embedded_bbob_task,
)

_CEC2005_NOISY = {"f4", "f17", "f24", "f25"}


def _embedded(noise):
    return create_embedded_bbob_task(
        fn_name="sphere", seed=0, intrinsic_dim=2, ambient_dim=6, noise=noise
    )


CASES = [
    pytest.param(
        lambda: create_bbob_task("sphere", 0, 3), False, id="bbob-sphere"
    ),
    pytest.param(
        lambda: create_bbob_task("sphere", 0, 3, noise=0.1),
        True,
        id="bbob-sphere-added",
    ),
    pytest.param(
        lambda: create_bbob_noisy_task("bbob_noisy_f101", 0, 3),
        True,
        id="bbob-noisy-f101",
    ),
    pytest.param(
        lambda: create_bbob_noisy_task("bbob_noisy_f124", 0, 3),
        True,
        id="bbob-noisy-f124",
    ),
    pytest.param(
        lambda: create_cec2005_task("f1", 0, 3), False, id="cec2005-f1"
    ),
    pytest.param(
        lambda: create_cec2005_task("f1", 0, 3, noise=0.1),
        True,
        id="cec2005-f1-added",
    ),
    *[
        pytest.param(
            lambda fn=fn: create_cec2005_task(fn, 0, 3),
            True,
            id=f"cec2005-{fn}",
        )
        for fn in sorted(_CEC2005_NOISY)
    ],
    pytest.param(
        lambda: create_cec2017_task("cec2017_f1", 0, 3),
        False,
        id="cec2017-f1",
    ),
    pytest.param(
        lambda: create_cec2017_task("cec2017_f1", 0, 3, noise=0.1),
        True,
        id="cec2017-f1-added",
    ),
    pytest.param(lambda: _embedded(0.0), False, id="embedded"),
    pytest.param(lambda: _embedded(0.1), True, id="embedded-added"),
]


@pytest.mark.parametrize("build, noisy", CASES)
def test_noise_tag_is_a_bool_and_matches_pass_rng(build, noisy):
    """``tag["noise"]`` is a Python bool, never a level, and equals
    ``pass_rng``."""
    task = build()
    assert type(task.tag["noise"]) is bool
    assert task.tag["noise"] is noisy
    assert task.pass_rng is noisy


@pytest.mark.parametrize("fn_name", sorted(bbob_jax.cec2005_registry))
def test_cec2005_noise_follows_bbob_jax_characteristic(fn_name):
    """Every CEC 2005 function is tagged by bbob-jax's own ``noise``
    characteristic, so exactly F4, F17, F24 and F25 are noisy."""
    task = create_cec2005_task(fn_name, 0, 2)
    expected = bbob_jax.cec2005_function_characteristics[fn_name]["noise"]
    assert task.tag["noise"] is expected
    assert task.pass_rng is expected
    assert expected is (fn_name in _CEC2005_NOISY)


def test_added_noise_level_is_not_part_of_identity():
    """The level is dropped from the tag, so two levels on one function
    are the same task (an accepted consequence of the bool-only tag;
    no shipped config sets an added level)."""
    low = create_bbob_task("sphere", 0, 3, noise=0.1)
    high = create_bbob_task("sphere", 0, 3, noise=0.2)
    assert low.tag == high.tag
    assert low.hash == high.hash
    assert low.hash != create_bbob_task("sphere", 0, 3).hash


@pytest.mark.parametrize("fn_name", sorted(_CEC2005_NOISY))
def test_added_noise_on_a_noisy_function_raises(fn_name):
    """Added noise on a function that is noisy by itself is refused: the
    two sources would share one key, and the result would be tagged the
    same as the function without added noise."""
    with pytest.raises(ValueError, match="already noisy"):
        create_cec2005_task(fn_name, 0, 3, noise=0.1)


@pytest.mark.parametrize("noise", [-0.1, float("nan")])
def test_invalid_noise_level_raises(noise):
    """A negative or NaN level is refused rather than read as 0."""
    with pytest.raises(ValueError, match="noise"):
        create_bbob_task("sphere", 0, 3, noise=noise)


def test_numpy_noise_level_gives_a_python_bool(eqx_path):
    """A level read from a DataFrame row is a numpy float; the tag and
    ``pass_rng`` must still be Python bools, so the JSON header
    writes."""
    task = create_bbob_task("sphere", 0, 3, noise=np.float64(0.1))
    assert type(task.tag["noise"]) is bool
    assert type(task.pass_rng) is bool
    loaded = Task.load(Task.save(task, str(eqx_path)))
    assert loaded.tag["noise"] is True
    assert loaded.hash == task.hash
