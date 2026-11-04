"""Host objectives behind an ordinary JAX loss (l2co-tasks ADR 0003).

The host objective here is a numpy Rosenbrock with an analytic gradient,
checked against the same function written in JAX. Covers values and
gradients, the transformations optimizers apply (reverse and forward
mode, nested vmap, scan), host-call counts and the last-point cache, an
objective without a gradient, the float32 warning, and lazy opening
across a ``Task`` save/load.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections import Counter

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from l2co_tasks import HostLoss, Task, host_loss

X0 = np.array([-1.2, 1.0, 0.5, -0.3])

#: Objectives opened so far, by opener name, so tests can read call
#: counts; and how often each name was opened.
_OBJECTIVES: dict[str, ValueOnlyRosenbrock] = {}
_OPENS: Counter = Counter()


class ValueOnlyRosenbrock:
    """Host objective without a gradient; counts its calls."""

    def __init__(self):
        self.calls: Counter = Counter()

    def value(self, x):
        self.calls["value"] += 1
        return float(
            np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
        )


class Rosenbrock(ValueOnlyRosenbrock):
    """Host objective with an analytic gradient."""

    def value_and_grad(self, x):
        self.calls["value_and_grad"] += 1
        d = x[1:] - x[:-1] ** 2
        f = float(np.sum(100.0 * d**2 + (1.0 - x[:-1]) ** 2))
        g = np.zeros_like(x)
        g[:-1] += -400.0 * x[:-1] * d - 2.0 * (1.0 - x[:-1])
        g[1:] += 200.0 * d
        return f, g


class InconsistentRosenbrock(Rosenbrock):
    """``value_and_grad`` returns a value one part in 1e12 off ``value``."""

    def value_and_grad(self, x):
        f, g = super().value_and_grad(x)
        return f * (1.0 + 1e-12), g


class WrongGradientShape(ValueOnlyRosenbrock):
    def value_and_grad(self, x):
        return self.value(x), np.zeros(x.size + 1)


class TwiceDifferentiableRosenbrock(Rosenbrock):
    """Host objective with an analytic Hessian too (ADR 0005)."""

    def hessian(self, x):
        self.calls["hessian"] += 1
        h = np.zeros((x.size, x.size))
        i = np.arange(x.size - 1)
        h[i, i] += 1200.0 * x[:-1] ** 2 - 400.0 * x[1:] + 2.0
        h[i + 1, i + 1] += 200.0
        h[i, i + 1] = h[i + 1, i] = -400.0 * x[:-1]
        return h


class WrongHessianShape(Rosenbrock):
    def hessian(self, x):
        return np.zeros((x.size, x.size + 1))


@dataclasses.dataclass(frozen=True)
class RosenbrockOpener:
    """Picklable, hashable opener. ``name`` gives each test its own
    objective, so call counts and last-point caches don't leak between
    tests."""

    name: str
    kind: str = "gradient"

    def __call__(self):
        _OPENS[self.name] += 1
        objective = {
            "gradient": Rosenbrock,
            "value_only": ValueOnlyRosenbrock,
            "inconsistent": InconsistentRosenbrock,
            "wrong_gradient": WrongGradientShape,
            "hessian": TwiceDifferentiableRosenbrock,
            "wrong_hessian": WrongHessianShape,
        }[self.kind]()
        _OBJECTIVES[self.name] = objective
        return objective


def rosenbrock(x):
    """The JAX reference."""
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


@pytest.fixture(autouse=True)
def x64():
    """The host computes in float64; compare in float64."""
    with jax.enable_x64(True):
        yield


@pytest.fixture
def opener(request) -> RosenbrockOpener:
    return RosenbrockOpener(request.node.name)


@pytest.fixture
def loss(opener) -> HostLoss:
    return host_loss(opener)


def calls(opener) -> Counter:
    return _OBJECTIVES[opener.name].calls


# Values and gradients -------------------------------------------------------


def test_value_matches_reference(loss):
    x = jnp.asarray(X0)
    expected = rosenbrock(x)
    np.testing.assert_allclose(loss(x), expected, rtol=1e-13)
    np.testing.assert_allclose(jax.jit(loss)(x), expected, rtol=1e-13)


def test_gradient_matches_reference(loss):
    x = jnp.asarray(X0)
    np.testing.assert_allclose(
        jax.jit(jax.grad(loss))(x), jax.grad(rosenbrock)(x), rtol=1e-12
    )


def test_value_and_gradient_cost_one_host_evaluation(loss, opener):
    f, g = jax.jit(jax.value_and_grad(loss))(jnp.asarray(X0))
    np.testing.assert_allclose(f, rosenbrock(jnp.asarray(X0)), rtol=1e-13)
    assert calls(opener) == Counter({"value_and_grad": 1})


def test_forward_mode_and_transpose(loss):
    """optimistix's path: ``linearize``, then transpose into a gradient."""
    x = jnp.asarray(X0)
    f, lin = jax.linearize(loss, x)
    (g,) = jax.linear_transpose(lin, x)(jnp.array(1.0))
    np.testing.assert_allclose(f, rosenbrock(x), rtol=1e-13)
    np.testing.assert_allclose(g, jax.grad(rosenbrock)(x), rtol=1e-12)
    t = jnp.arange(1.0, 5.0)
    np.testing.assert_allclose(
        jax.jvp(loss, (x,), (t,))[1],
        jax.jvp(rosenbrock, (x,), (t,))[1],
        rtol=1e-12,
    )


def test_nested_vmap_reaches_the_host_once_per_point(loss, opener):
    """Realizations x population, as the fused run loop batches them."""
    xs = X0 + jr.normal(jr.key(0), (5, 3, X0.size))
    batched = jax.vmap(jax.vmap(rosenbrock))
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.vmap(loss)))(xs), batched(xs), rtol=1e-12
    )
    assert calls(opener)["value"] == 15

    grads = jax.jit(jax.vmap(jax.vmap(jax.grad(loss))))(xs)
    np.testing.assert_allclose(
        grads, jax.vmap(jax.vmap(jax.grad(rosenbrock)))(xs), rtol=1e-12
    )
    assert calls(opener)["value_and_grad"] == 15


def test_gradient_descent_in_scan(loss):
    def descend(f):
        def body(x, _):
            value, g = jax.value_and_grad(f)(x)
            return x - 1e-4 * g, value

        return jax.lax.scan(body, jnp.asarray(X0), None, length=20)[1]

    np.testing.assert_allclose(
        jax.jit(lambda: descend(loss))(),
        jax.jit(lambda: descend(rosenbrock))(),
        rtol=1e-10,
    )


def test_pytree_model_is_flattened_in_ravel_order(loss):
    model = {
        "a": jnp.asarray(X0[:2]),
        "b": jnp.asarray(X0[2:]),
        "count": 3,  # not a floating-point array: not a parameter
    }
    np.testing.assert_allclose(
        loss(model), rosenbrock(jnp.asarray(X0)), rtol=1e-13
    )


# Last-point cache ------------------------------------------------------------


def test_repeated_point_is_served_from_the_cache(loss, opener):
    x = jnp.asarray(X0)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    value_and_grad(x)
    value_and_grad(x)
    assert calls(opener) == Counter({"value_and_grad": 1})

    value = jax.jit(loss)
    value(x)
    value(x)
    assert calls(opener) == Counter({"value_and_grad": 1, "value": 1})

    value_and_grad(x + 0.1)
    assert calls(opener)["value_and_grad"] == 2


def test_value_is_never_served_from_a_gradient_call(request):
    """The two routines of an objective may differ in the last bit; a
    value must not depend on whether a gradient call came first."""
    opener = RosenbrockOpener(request.node.name, "inconsistent")
    loss = host_loss(opener)
    x = jnp.asarray(X0)
    f_grad, _ = jax.jit(jax.value_and_grad(loss))(x)
    f_value = jax.jit(loss)(x)
    assert float(f_value) == InconsistentRosenbrock().value(np.asarray(X0))
    assert float(f_grad) != float(f_value)


# Objectives without a gradient ----------------------------------------------


def test_objective_without_gradient_evaluates(request):
    loss = host_loss(RosenbrockOpener(request.node.name, "value_only"))
    x = jnp.asarray(X0)
    np.testing.assert_allclose(jax.jit(loss)(x), rosenbrock(x), rtol=1e-13)


@pytest.mark.parametrize(
    "differentiate",
    [lambda f, x: jax.grad(f)(x), lambda f, x: jax.linearize(f, x)],
    ids=["grad", "linearize"],
)
def test_differentiating_an_objective_without_gradient_raises(
    request, differentiate
):
    opener = RosenbrockOpener(request.node.name, "value_only")
    with pytest.raises(TypeError, match="without value_and_grad"):
        differentiate(host_loss(opener), jnp.asarray(X0))


def test_second_derivatives_without_a_hessian_fail_loudly(loss):
    with pytest.raises(TypeError, match="without hessian"):
        jax.hessian(loss)(jnp.asarray(X0))


# Second derivatives (ADR 0005) -----------------------------------------------


@pytest.fixture
def twice(request) -> tuple[RosenbrockOpener, HostLoss]:
    opener = RosenbrockOpener(request.node.name, "hessian")
    return opener, host_loss(opener)


@pytest.mark.parametrize(
    "second",
    [
        jax.hessian,
        lambda f: jax.jacrev(jax.jacrev(f)),
        lambda f: jax.jacfwd(jax.grad(f)),
    ],
    ids=["hessian", "rev-over-rev", "fwd-over-rev"],
)
def test_hessian_matches_reference(twice, second):
    _, loss = twice
    x = jnp.asarray(X0)
    np.testing.assert_allclose(
        second(loss)(x), jax.hessian(rosenbrock)(x), rtol=1e-12
    )
    np.testing.assert_allclose(
        jax.jit(second(loss))(x), jax.hessian(rosenbrock)(x), rtol=1e-12
    )


def test_a_hessian_costs_one_host_hessian_call(twice):
    opener, loss = twice
    jax.hessian(loss)(jnp.asarray(X0))
    assert calls(opener)["hessian"] == 1
    # The same point again is served from the Hessian cache.
    jax.hessian(loss)(jnp.asarray(X0))
    assert calls(opener)["hessian"] == 1


def test_first_derivatives_do_not_ask_for_a_hessian(twice):
    opener, loss = twice
    x = jnp.asarray(X0)
    jax.grad(loss)(x)
    jax.linearize(loss, x)
    assert calls(opener)["hessian"] == 0
    np.testing.assert_allclose(
        jax.grad(loss)(x), jax.grad(rosenbrock)(x), rtol=1e-13
    )


def test_hessian_under_vmap(twice):
    _, loss = twice
    points = jnp.asarray(X0) + jnp.linspace(-0.3, 0.3, 5)[:, None]
    np.testing.assert_allclose(
        jax.vmap(jax.hessian(loss))(points),
        jax.vmap(jax.hessian(rosenbrock))(points),
        rtol=1e-12,
    )


def test_wrong_hessian_shape_is_reported(request):
    loss = host_loss(RosenbrockOpener(request.node.name, "wrong_hessian"))
    with pytest.raises(Exception, match="hessian returned shape"):
        jax.block_until_ready(jax.hessian(loss)(jnp.asarray(X0)))


def test_wrong_gradient_shape_is_reported(request):
    loss = host_loss(RosenbrockOpener(request.node.name, "wrong_gradient"))
    with pytest.raises(Exception, match="gradient of shape"):
        jax.block_until_ready(jax.jit(jax.grad(loss))(jnp.asarray(X0)))


# Precision -------------------------------------------------------------------


def test_float32_parameters_warn(loss):
    with jax.enable_x64(False):
        x = jnp.asarray(X0, dtype=jnp.float32)
        with pytest.warns(UserWarning, match="float32"):
            value = jax.jit(loss)(x)
        assert value.dtype == jnp.float32
        np.testing.assert_allclose(value, rosenbrock(X0), rtol=1e-5)


def test_float64_parameters_do_not_warn(loss):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jax.jit(loss)(jnp.asarray(X0))


# The loss object -------------------------------------------------------------


def test_dataset_batch_or_key_is_refused(loss):
    with pytest.raises(TypeError, match="no dataset batch or random key"):
        loss(jnp.asarray(X0), key=jr.key(0))


def test_equal_openers_share_one_objective(request):
    name = request.node.name
    first = host_loss(RosenbrockOpener(name))
    second = host_loss(RosenbrockOpener(name))
    assert first == second
    assert hash(first) == hash(second)
    assert first != host_loss(RosenbrockOpener(name + "-other"))

    x = jnp.asarray(X0)
    jax.jit(first)(x)
    jax.jit(second)(x + 1.0)
    assert _OPENS[name] == 1


def test_opener_must_be_callable():
    with pytest.raises(TypeError, match="callable"):
        host_loss(3)


def test_opener_must_be_hashable():
    class Unhashable:
        __hash__ = None

        def __call__(self):
            return Rosenbrock()

    with pytest.raises(TypeError, match="hashable"):
        host_loss(Unhashable())


def test_task_round_trip_opens_lazily(opener, eqx_path):
    """A saved task holds only the opener: nothing is opened by saving
    or loading, and the loaded task evaluates once traced."""
    task = Task(
        model=jnp.asarray(X0),
        loss_fn=host_loss(opener),
        global_min=0.0,
        tag={"task_name": "host_rosenbrock"},
    )
    loaded = Task.load(Task.save(task, str(eqx_path)))
    assert _OPENS[opener.name] == 0
    assert loaded == task
    assert loaded.loss_fn == task.loss_fn

    np.testing.assert_allclose(
        jax.jit(loaded.loss_fn)(loaded.model),
        rosenbrock(jnp.asarray(X0)),
        rtol=1e-13,
    )
    assert _OPENS[opener.name] == 1
