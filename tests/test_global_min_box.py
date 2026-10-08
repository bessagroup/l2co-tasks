"""``estimate_global_min`` searches inside a task's Box (ADR 0002).

A task that declares a box is searched inside it: Adam steps are
clipped into the box, restarts are drawn from it, and L-BFGS is skipped.
Constraints the search can't respect are refused.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from l2co_tasks import Box, Inequality, Task, estimate_global_min


def pull_to_two(model):
    """Minimum 0 at x = 2, outside the unit box; 2.0 at the box's corner."""
    return jnp.sum((model - 2.0) ** 2)


def _task(constraints=()):
    return Task(
        model=jnp.full(2, 0.5),
        loss_fn=pull_to_two,
        tag={"task_name": "pull_to_two"},
        constraints=constraints,
    )


_KW = dict(seed=0, n_restarts=2, n_steps=200, lr=0.1, lbfgs_steps=50)


def test_boxed_search_stays_inside_the_box():
    est = estimate_global_min(_task([Box(0.0, 1.0)]), **_KW)
    # The best point inside [0, 1]^2 is (1, 1), loss 2.0.
    assert est == pytest.approx(2.0, abs=1e-3)
    assert est >= 2.0 - 1e-6


def test_unboxed_search_is_free():
    assert estimate_global_min(_task(), **_KW) == pytest.approx(0.0, abs=1e-3)


def test_refuses_an_inequality():
    task = _task([Inequality(lambda m: m[0] - 1.0, name="g")])
    with pytest.raises(ValueError, match="only respect a Box"):
        estimate_global_min(task, **_KW)


def test_refuses_an_open_box():
    with pytest.raises(ValueError, match="finite"):
        estimate_global_min(_task([Box(-jnp.inf, 1.0)]), **_KW)


def test_clip_to_unit_is_gone():
    with pytest.raises(TypeError):
        estimate_global_min(_task(), clip_to_unit=True, **_KW)
