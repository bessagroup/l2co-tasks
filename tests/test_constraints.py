"""Constraints on a Task (l2co-tasks ADR 0002).

Covers the three constraint kinds, the rules a ``Task`` enforces when it
is built, the box cast, identity, and the ``.eqx`` round trip.
"""

from __future__ import annotations

import dataclasses
import json

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

from l2co_tasks import Box, Equality, Inequality, Task


def sum_of_squares(model):
    return jnp.sum(model**2)


def first_below_half(model):
    """``x0 - 0.5 <= 0``."""
    return model[0] - 0.5


def sum_is_one(model):
    """``sum(x) - 1 = 0``."""
    return jnp.sum(model) - 1.0


def returns_int(model):
    return jnp.zeros(2, dtype=jnp.int32)


def returns_tuple(model):
    return model, model


def needs_a_key(model, key):
    return model[0]


START = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32)


def _task(constraints=(), model=START, **kw):
    return Task(
        model=model,
        loss_fn=sum_of_squares,
        tag={"task_name": "constrained", "seed": 0},
        constraints=constraints,
        **kw,
    )


# ----------------------------------------------------------- the three kinds


def test_inequality_raw_values_and_violation():
    c = Inequality(lambda m: m - 0.4, name="g")
    np.testing.assert_allclose(c(START), [-0.2, 0.1, -0.1], atol=1e-6)
    np.testing.assert_allclose(c.violation(START), [0.0, 0.1, 0.0], atol=1e-6)
    assert c.tol == 0.0


def test_inequality_violation_counts_beyond_tolerance():
    c = Inequality(lambda m: m - 0.4, name="g", tol=0.05)
    np.testing.assert_allclose(c.violation(START), [0.0, 0.05, 0.0], atol=1e-6)


def test_equality_raw_values_and_violation():
    c = Equality(lambda m: m - 0.5, name="h")
    assert c.tol == 1e-4
    np.testing.assert_allclose(c(START), [-0.3, 0.0, -0.2], atol=1e-6)
    np.testing.assert_allclose(
        c.violation(START), [0.3 - 1e-4, 0.0, 0.2 - 1e-4], atol=1e-6
    )


def test_any_output_shape_is_flattened():
    c = Inequality(lambda m: jnp.reshape(jnp.tile(m, 2), (2, 3)), name="g")
    assert c(START).shape == (6,)
    assert Equality(sum_is_one, name="h")(START).shape == (1,)


@pytest.mark.parametrize("kind", [Inequality, Equality])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(name=""),
        dict(name="g", tol=-1e-3),
        dict(name="g", tol=float("nan")),
        dict(name="g", tol=float("inf")),
    ],
    ids=["empty-name", "negative-tol", "nan-tol", "inf-tol"],
)
def test_bad_name_or_tolerance_is_rejected(kind, kwargs):
    with pytest.raises(ValueError):
        kind(first_below_half, **kwargs)


def test_box_on_its_own_takes_single_numbers():
    box = Box(lower=0.0, upper=0.4)
    np.testing.assert_allclose(
        box(START), [-0.2, -0.5, -0.3, -0.2, 0.1, -0.1], atol=1e-6
    )
    np.testing.assert_allclose(
        box.violation(START), [0, 0, 0, 0, 0.1, 0], atol=1e-6
    )


def test_box_evaluates_under_jit():
    (box,) = _task([Box(0.0, 1.0)]).constraints
    out = eqx.filter_jit(lambda b, m: b(m))(box, START)
    assert out.shape == (6,)
    assert bool(jnp.all(out <= 0))


# ------------------------------------------------- building a Task: the cast


def test_unconstrained_task_has_empty_tuple():
    assert _task().constraints == ()
    assert Task(model=START, loss_fn=sum_of_squares).constraints == ()


def test_any_iterable_is_stored_as_a_tuple():
    task = _task(c for c in [Inequality(first_below_half, name="g")])
    assert isinstance(task.constraints, tuple)
    assert len(task.constraints) == 1


def test_single_number_box_is_cast_to_the_model():
    (box,) = _task([Box(0.0, 1.0)]).constraints
    for bound, value in ((box.lower, 0.0), (box.upper, 1.0)):
        assert bound.shape == START.shape
        assert bound.dtype == START.dtype
        np.testing.assert_array_equal(bound, np.full(3, value))


def test_box_cast_mirrors_a_module_model():
    model = eqx.nn.Linear(3, 2, key=jr.key(0))
    lim = 1e3
    (box,) = _task([Box(-lim, lim)], model=model).constraints
    params = eqx.filter(model, eqx.is_inexact_array)
    assert jtu.tree_structure(box.lower) == jtu.tree_structure(params)
    assert [x.shape for x in jtu.tree_leaves(box.upper)] == [
        x.shape for x in jtu.tree_leaves(params)
    ]


def test_full_pytree_box_is_accepted():
    lower = jnp.array([0.0, 0.0, -1.0])
    (box,) = _task([Box(lower, jnp.ones(3))]).constraints
    np.testing.assert_array_equal(box.lower, lower)


def test_open_sides_are_infinite():
    (box,) = _task([Box(-jnp.inf, 1.0)]).constraints
    assert bool(jnp.all(jnp.isneginf(box.lower)))


def test_pinned_parameter_is_allowed():
    pin = jnp.array([0.2, 0.0, 0.0])
    _task([Box(pin, jnp.array([0.2, 1.0, 1.0]))])


def test_rebuilding_keeps_the_box():
    task = _task([Box(0.0, 1.0)])
    again = dataclasses.replace(task, global_min=0.0)
    np.testing.assert_array_equal(
        again.constraints[0].lower, task.constraints[0].lower
    )
    assert again.hash != task.hash  # global_min changed, nothing else


# ---------------------------------------------- building a Task: rejections


def test_rejects_a_non_constraint():
    with pytest.raises(TypeError):
        _task([first_below_half])


def test_rejects_two_boxes():
    with pytest.raises(ValueError, match="at most one Box"):
        _task([Box(0.0, 1.0), Box(-1.0, 2.0)])


@pytest.mark.parametrize(
    "lower",
    [jnp.zeros(2), jnp.zeros((3, 1)), jnp.zeros(1)],
    ids=["too-short", "wrong-rank", "no-partial-broadcast"],
)
def test_rejects_box_not_matching_the_model(lower):
    with pytest.raises(ValueError, match="does not match|shape"):
        _task([Box(lower, 1.0)])


def test_rejects_lower_above_upper():
    with pytest.raises(ValueError, match="exceeds"):
        _task([Box(jnp.array([0.0, 0.9, 0.0]), jnp.array([1.0, 0.8, 1.0]))])


def test_rejects_nan_bound():
    with pytest.raises(ValueError, match="NaN"):
        _task([Box(jnp.nan, 1.0)])


def test_rejects_start_outside_box():
    with pytest.raises(ValueError, match="outside"):
        _task([Box(0.0, 0.4)])


def test_rejects_duplicate_names():
    with pytest.raises(ValueError, match="unique"):
        _task(
            [
                Inequality(first_below_half, name="g"),
                Equality(sum_is_one, name="g"),
            ]
        )


@pytest.mark.parametrize(
    "fn", [returns_int, returns_tuple, needs_a_key], ids=lambda f: f.__name__
)
def test_rejects_constraint_that_does_not_evaluate_to_floats(fn):
    with pytest.raises(ValueError, match="'bad'"):
        _task([Inequality(fn, name="bad")])


# ------------------------------------------------------------------ identity


def test_constraints_change_identity():
    plain = _task()
    boxed = _task([Box(0.0, 1.0)])
    assert boxed.hash != plain.hash
    assert boxed != plain


def test_identity_ignores_order():
    a = Inequality(first_below_half, name="a")
    b = Equality(sum_is_one, name="b")
    box = Box(0.0, 1.0)
    assert _task([a, b, box]).hash == _task([box, b, a]).hash
    assert _task([a, b, box]) == _task([box, b, a])


@pytest.mark.parametrize(
    "other",
    [
        [Inequality(first_below_half, name="renamed")],
        [Inequality(first_below_half, name="g", tol=1e-3)],
        [Equality(first_below_half, name="g")],
    ],
    ids=["name", "tol", "kind"],
)
def test_name_tolerance_and_kind_enter_identity(other):
    base = _task([Inequality(first_below_half, name="g")])
    assert _task(other).hash != base.hash


def test_box_values_enter_identity():
    assert _task([Box(0.0, 1.0)]).hash != _task([Box(0.0, 2.0)]).hash


def test_single_number_and_full_box_share_identity():
    full = Box(jnp.zeros(3), jnp.ones(3))
    assert _task([Box(0.0, 1.0)]).hash == _task([full]).hash


# ------------------------------------------------------------------- storage


def test_unconstrained_header_has_no_constraints_key(eqx_path):
    path = Task.save(_task(), eqx_path)
    with open(path, "rb") as f:
        assert "constraints" not in json.loads(f.readline().decode())


def test_constrained_task_round_trips(eqx_path):
    task = _task(
        [
            Inequality(first_below_half, name="g", tol=1e-3),
            Equality(sum_is_one, name="h"),
            Box(0.0, 1.0),
        ]
    )
    loaded = Task.load(Task.save(task, eqx_path))
    assert loaded.hash == task.hash
    assert loaded == task
    assert [type(c) for c in loaded.constraints] == [
        Inequality,
        Equality,
        Box,
    ]
    for got, want in zip(loaded.constraints, task.constraints, strict=True):
        np.testing.assert_allclose(got(START), want(START))


def test_box_bounds_are_not_parameters():
    assert _task([Box(0.0, 1.0)]).dimensionality == 3
