"""``estimate_global_min``'s ``restart_sampler`` (ADR 0004).

With ``lr=0`` and no L-BFGS arm, a restart never moves, so the estimate
is exactly the lowest loss among the restarts' starting points. That
makes where each restart starts directly observable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks import Box, Task, estimate_global_min

#: Restarts that never move: the estimate is the best start.
_FROZEN = dict(n_steps=1, lr=0.0, use_lbfgs=False)


def distance_to_target(model):
    """Loss 0 at (3, 3), the point no default restart is near."""
    return jnp.sum((model - 3.0) ** 2)


def _task(model=(0.0, 0.0), constraints=()):
    return Task(
        model=jnp.asarray(model),
        loss_fn=distance_to_target,
        global_min=0.0,
        tag={"task_name": "restart_sampler_test"},
        constraints=constraints,
    )


def _fixed_starts(*points):
    """A sampler returning ``points`` as the draws, recording its calls."""
    calls = []

    def sampler(key, params, n_samples):
        calls.append((params, n_samples))
        return jnp.asarray(points, dtype=params.dtype)

    sampler.calls = calls
    return sampler


def test_without_a_sampler_restarts_are_unchanged():
    task = _task()
    assert estimate_global_min(
        task, n_restarts=3, **_FROZEN
    ) == estimate_global_min(
        task, n_restarts=3, restart_sampler=None, **_FROZEN
    )


def test_every_restart_starts_from_a_draw():
    """The target is reachable only from the last draw."""
    sampler = _fixed_starts([0.0, 0.0], [1.0, 1.0], [3.0, 3.0])
    estimate = estimate_global_min(
        _task(), n_restarts=3, restart_sampler=sampler, **_FROZEN
    )
    assert estimate == 0.0


def test_sampler_gets_the_parameters_and_the_restart_count():
    sampler = _fixed_starts([1.0, 1.0], [2.0, 2.0])
    estimate_global_min(
        _task(model=(0.5, -0.5)),
        n_restarts=2,
        restart_sampler=sampler,
        **_FROZEN,
    )
    ((params, n_samples),) = sampler.calls
    assert n_samples == 2
    assert jnp.array_equal(params, jnp.array([0.5, -0.5]))


def test_first_draw_at_the_model_restarts_from_the_model():
    """A relative_normal-style sampler: draw 0 is the parameters."""

    def around_model(key, params, n_samples):
        noise = jr.normal(key, (n_samples, *params.shape), params.dtype)
        return (params + noise).at[0].set(params)

    estimate = estimate_global_min(
        _task(model=(3.0, 3.0)),
        n_restarts=3,
        restart_sampler=around_model,
        **_FROZEN,
    )
    assert estimate == 0.0


def test_random_sampler_is_deterministic_in_seed():
    def normal(key, params, n_samples):
        return jr.normal(key, (n_samples, *params.shape), params.dtype)

    task = _task()
    kwargs = dict(n_restarts=4, restart_sampler=normal, **_FROZEN)
    first = estimate_global_min(task, seed=7, **kwargs)
    assert estimate_global_min(task, seed=7, **kwargs) == first
    assert estimate_global_min(task, seed=8, **kwargs) != first


def test_draws_are_clipped_into_the_box():
    """(5, 5) clipped into [0, 4]^2 is (4, 4), at loss 2."""
    task = _task(model=(1.0, 1.0), constraints=[Box(0.0, 4.0)])
    estimate = estimate_global_min(
        task,
        n_restarts=1,
        restart_sampler=_fixed_starts([5.0, 5.0]),
        **_FROZEN,
    )
    assert estimate == pytest.approx(2.0)


@pytest.mark.parametrize(
    "points",
    [
        [[0.0, 0.0]],  # one draw for two restarts
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],  # wrong parameter shape
    ],
    ids=["too-few-draws", "wrong-shape"],
)
def test_draws_that_do_not_match_are_refused(points):
    with pytest.raises(ValueError, match="leading axis of 2"):
        estimate_global_min(
            _task(),
            n_restarts=2,
            restart_sampler=_fixed_starts(*points),
            **_FROZEN,
        )


def test_restarts_still_train():
    """With steps left on, restarts from the draws are optimized."""
    sampler = _fixed_starts([0.0, 0.0], [1.0, 1.0])
    with jax.enable_x64(True):
        estimate = estimate_global_min(
            _task(), n_restarts=2, n_steps=200, restart_sampler=sampler
        )
    assert estimate == pytest.approx(0.0, abs=1e-6)
