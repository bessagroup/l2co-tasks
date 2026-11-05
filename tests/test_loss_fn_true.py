"""The optional noise-free loss, ``Task.loss_fn_true``.

A noisy benchmark task also carries its loss without the noise, so l2co
can score a run on the noise-free value at the points it evaluated
(dataset paper protocol revision, decision 12). It takes the arguments
of ``loss_fn`` without the key, goes through the same input scaling,
survives ``Task.save`` / ``Task.load``, and is ``None`` on a noise-free
task. Like ``constraints``, it is written to the file header only when
present, and it is never part of the task's identity.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import bbob_jax
import jax
import jax.numpy as jnp
import jax.random as jr
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

from .conftest import sum_of_squares

FIXTURE = Path(__file__).parent / "data" / "golden_quadratic_square.eqx"


def _noisy_loss(model, key):
    """Picklable stochastic loss for the hand-built tasks below."""
    return jnp.sum(model**2) * (1.0 + 0.1 * jr.normal(key, ()))


def _task(**kwargs) -> Task:
    """A hand-built noisy task; ``kwargs`` override the defaults."""
    fields = dict(
        model=jnp.arange(3.0),
        loss_fn=_noisy_loss,
        pass_rng=True,
        tag={"task_name": "noisy-sum", "seed": 0, "noise": True},
    )
    fields.update(kwargs)
    return Task(**fields)


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


# Task field ------------------------------------------------------------------


def test_defaults_to_none(simple_task):
    assert simple_task.loss_fn_true is None


def test_absent_field_writes_no_header_key(simple_task, eqx_path):
    path = Task.save(simple_task, str(eqx_path))
    with open(path, "rb") as f:
        assert "loss_fn_true" not in json.loads(f.readline().decode())
    assert Task.load(path).loss_fn_true is None


def test_pre_existing_file_resaves_byte_for_byte(tmp_path):
    """A file written before the field existed saves back unchanged, so
    the field adds nothing to a noise-free task's ``.eqx`` bytes."""
    with jax.enable_x64(True):
        task = Task.load(FIXTURE)
        resaved = Task.save(task, str(tmp_path / "resaved"))
    assert task.loss_fn_true is None
    assert Path(resaved).read_bytes() == FIXTURE.read_bytes()


def test_round_trip_keeps_the_noise_free_loss(eqx_path):
    task = _task(loss_fn_true=sum_of_squares)
    path = Task.save(task, str(eqx_path))
    with open(path, "rb") as f:
        assert "loss_fn_true" in json.loads(f.readline().decode())
    loaded = Task.load(path)
    assert loaded == task
    assert loaded.hash == task.hash
    x = jnp.array([0.5, -1.0, 2.0])
    assert float(loaded.loss_fn_true(x)) == float(task.loss_fn_true(x))


def test_is_not_part_of_identity():
    """The noise-free loss follows from what the tag already holds, so
    adding it changes neither the hash nor equality."""
    without = _task()
    with_true = _task(loss_fn_true=sum_of_squares)
    assert with_true.hash == without.hash
    assert with_true == without
    assert hash(with_true) == hash(without)


def test_replace_keeps_the_noise_free_loss():
    task = _task(loss_fn_true=sum_of_squares)
    replaced = dataclasses.replace(task, model=jnp.ones(3))
    assert replaced.loss_fn_true is sum_of_squares


def test_needs_a_stochastic_loss():
    """On a deterministic loss a noise-free loss would only be a second
    copy of ``loss_fn``."""
    with pytest.raises(ValueError, match="pass_rng"):
        Task(
            model=jnp.zeros(2),
            loss_fn=sum_of_squares,
            loss_fn_true=sum_of_squares,
        )


def test_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        _task(loss_fn_true=1.0)


# Builders: which tasks carry it ----------------------------------------------


NOISE_FREE = [
    pytest.param(lambda: create_bbob_task("sphere", 0, 3), id="bbob"),
    pytest.param(lambda: create_cec2005_task("f1", 0, 3), id="cec2005-f1"),
    pytest.param(
        lambda: create_cec2017_task("cec2017_f1", 0, 3), id="cec2017-f1"
    ),
    pytest.param(
        lambda: create_embedded_bbob_task("sphere", 0, 2, 6), id="embedded"
    ),
]


@pytest.mark.parametrize("build", NOISE_FREE)
def test_noise_free_task_has_none(build):
    task = build()
    assert task.loss_fn_true is None
    assert task.pass_rng is False
    assert task.tag["noise"] is False


# Builders: the noise-free value is bbob-jax's fn_true ------------------------


INTRINSIC = [
    ("bbob_noisy", "bbob_noisy_f101"),  # Gaussian, moderate
    ("bbob_noisy", "bbob_noisy_f102"),  # uniform, moderate
    ("bbob_noisy", "bbob_noisy_f103"),  # Cauchy, moderate
    ("bbob_noisy", "bbob_noisy_f122"),  # Gaussian, severe
    ("cec2005", "f4"),
    ("cec2005", "f17"),
    ("cec2005", "f24"),
    ("cec2005", "f25"),
]
_BUILDERS = {
    "bbob_noisy": create_bbob_noisy_task,
    "cec2005": create_cec2005_task,
}


@pytest.mark.parametrize("suite, fn_name", INTRINSIC)
def test_noise_free_loss_is_bbob_jax_fn_true(suite, fn_name, x64):
    """On a function that is noisy by itself, the noise-free loss is
    bbob-jax's ``fn_true`` of the same instance, through the same map
    from the unit box onto the function's bounds."""
    d, seed = 4, 3
    task = _BUILDERS[suite](fn_name, seed, d)
    assert task.pass_rng is True
    assert task.loss_fn_true is not None

    problem = bbob_jax.problem(fn_name, ndim=d, key=jr.key(seed))
    lower, upper = problem.bounds
    for x in jr.uniform(jr.key(11), (8, d), dtype=jnp.float64):
        expected = problem.fn_true(lower + x * (upper - lower))
        np.testing.assert_allclose(
            task.loss_fn_true(x), expected, rtol=1e-12, atol=0.0
        )


@pytest.mark.parametrize("fn_name", ["bbob_noisy_f101", "bbob_noisy_f122"])
def test_noisy_suite_is_exact_at_its_optimum(fn_name, x64):
    """The noise-free loss reaches ``global_min`` at the optimum, where
    the noise models leave the value alone."""
    d = 4
    task = create_bbob_noisy_task(fn_name, 0, d)
    problem = bbob_jax.problem(fn_name, ndim=d, key=jr.key(0))
    lower, upper = problem.bounds
    x_opt = (problem.x_opt - lower) / (upper - lower)
    np.testing.assert_allclose(
        task.loss_fn_true(x_opt), task.global_min, rtol=1e-12
    )


def test_noisy_suite_matches_its_noiseless_twin(x64):
    """``bbob_noisy_f101`` is a noisy sphere: its noise-free loss is the
    plain BBOB sphere of the same seed."""
    noisy = create_bbob_noisy_task("bbob_noisy_f101", 5, 4)
    twin = create_bbob_task("sphere", 5, 4)
    assert noisy.global_min == twin.global_min
    for x in jr.uniform(jr.key(2), (8, 4), dtype=jnp.float64):
        np.testing.assert_allclose(
            noisy.loss_fn_true(x), twin.loss_fn(x), rtol=1e-12
        )


def _mean_over_keys(task, x, n):
    keys = jr.split(jr.key(99), n)
    return jax.vmap(lambda k: task.loss_fn(x, key=k))(keys)


# bbob-jax keeps COCO's noise gate: a disturbed residual above 1e-8 is
# shifted up by 1.01e-8, so a BBOB-noisy loss with its noise draw at zero
# is the noise-free value plus this shift. CEC 2005 has no gate.
_COCO_GATE_SHIFT = 1.01e-8

# The functions whose only random draws are standard normals in a
# Gaussian noise factor. The uniform and Cauchy models (f102, f103) are
# not undisturbed at a zero normal draw, so they are left out.
ZERO_DRAW = [
    ("bbob_noisy", "bbob_noisy_f101", _COCO_GATE_SHIFT),
    ("bbob_noisy", "bbob_noisy_f122", _COCO_GATE_SHIFT),
    ("cec2005", "f4", 0.0),
    ("cec2005", "f17", 0.0),
    ("cec2005", "f24", 0.0),  # through its noisy sphere component
    ("cec2005", "f25", 0.0),
]


def _zero_normal(key, shape=(), dtype=None):
    """``jax.random.normal`` with every draw at zero."""
    return jnp.zeros(shape, dtype=dtype)


@pytest.mark.parametrize("suite, fn_name, shift", ZERO_DRAW)
def test_noisy_loss_with_a_zero_draw_is_the_noise_free_loss(
    suite, fn_name, shift, monkeypatch, x64
):
    """With its normal draw at zero, a Gaussian noise factor is 1, so
    the noisy loss must give ``loss_fn_true``. Unlike the comparison
    with ``fn_true`` above, this checks the noise-free loss against the
    noisy loss itself, and it holds for F4's ``1 + 0.4 N`` and for the
    report's ``1 + 0.4 |N|`` alike."""
    d = 4
    task = _BUILDERS[suite](fn_name, 3, d)  # instance drawn unpatched
    xs = jr.uniform(jr.key(11), (8, d), dtype=jnp.float64)
    true = [float(task.loss_fn_true(x)) for x in xs]
    # Inside the box and clear of the gate, where the shift applies.
    assert min(true) - task.global_min > 1e-6

    monkeypatch.setattr(jax.random, "normal", _zero_normal)
    for x, t in zip(xs, true, strict=True):
        np.testing.assert_allclose(
            task.loss_fn(x, key=jr.key(0)), t + shift, rtol=1e-12, atol=0.0
        )


def test_bbob_noisy_gaussian_is_centred_on_the_noise_free_value(x64):
    """f101's noise is ``(f_true - f_opt) * exp(0.01 N)`` plus 1.01e-8,
    whose mean is ``exp(5e-5)`` times the excess: the mean sits within
    5e-4 of the noise-free value, relative to the excess over f_opt
    (bias 5e-5, sampling error 5e-5 per standard deviation)."""
    task = create_bbob_noisy_task("bbob_noisy_f101", 0, 4)
    x = jnp.full(4, 0.7)
    true = float(task.loss_fn_true(x))
    mean = float(_mean_over_keys(task, x, 40_000).mean())
    excess = true - task.global_min
    assert excess > 0
    assert abs(mean - true) < 5e-4 * excess


# Builders: added noise -------------------------------------------------------


ADDED = [
    pytest.param(
        lambda n: create_bbob_task("rastrigin", 0, 3, noise=n), 3, id="bbob"
    ),
    pytest.param(
        lambda n: create_cec2005_task("f1", 0, 3, noise=n), 3, id="cec2005"
    ),
    pytest.param(
        lambda n: create_cec2017_task("cec2017_f1", 0, 3, noise=n),
        3,
        id="cec2017",
    ),
    pytest.param(
        lambda n: create_embedded_bbob_task("rastrigin", 0, 2, 6, noise=n),
        6,
        id="embedded",
    ),
]


@pytest.mark.parametrize("build, d", ADDED)
def test_added_noise_keeps_the_unwrapped_loss(build, d, x64):
    """With added noise, the noise-free loss is the loss without the
    noise wrapper: the noiseless task's loss, and the mean of the noisy
    one."""
    noisy, clean = build(0.1), build(0.0)
    assert noisy.loss_fn_true is not None
    for x in jr.uniform(jr.key(5), (4, d), dtype=jnp.float64):
        assert float(noisy.loss_fn_true(x)) == float(clean.loss_fn(x))

    x = jnp.full(d, 0.3)
    true = float(noisy.loss_fn_true(x))
    draws = _mean_over_keys(noisy, x, 20_000)
    spread = 0.1 * abs(true) / np.sqrt(draws.size)
    assert abs(float(draws.mean()) - true) < 5 * spread


# Builders: save / load -------------------------------------------------------


def test_noisy_builder_task_round_trips(eqx_path, x64):
    task = create_cec2005_task("f17", 0, 10)
    loaded = Task.load(Task.save(task, str(eqx_path)))
    assert loaded == task
    assert loaded.hash == task.hash
    x = jr.uniform(jr.key(8), (10,), dtype=jnp.float64)
    assert float(loaded.loss_fn_true(x)) == float(task.loss_fn_true(x))
    k = jr.key(1)
    assert float(loaded.loss_fn(x, key=k)) == float(task.loss_fn(x, key=k))


# Builders: names stay suite-checked ------------------------------------------


def test_bbob_task_refuses_another_suites_name():
    """``problem()`` resolves names across every suite; the BBOB builder
    must still refuse a CEC 2005 name instead of building it."""
    with pytest.raises(KeyError):
        create_bbob_task("f4", 0, 3)


def test_bbob_noisy_task_refuses_a_noiseless_name():
    with pytest.raises(KeyError):
        create_bbob_noisy_task("sphere", 0, 3)
