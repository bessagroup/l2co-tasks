"""Tests for ``create_pkpd_task`` including a c(t) correctness gate."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from l2co_tasks import Task, create_pkpd_task
from l2co_tasks._src.pkpd import (
    _HOURS_PER_DAY,
    _K10_PER_HOUR,
    _K12_PER_HOUR,
    _K21_PER_HOUR,
    concentration,
)


def _build(tmp_path):
    """Build a tiny PK-PD task on a coarse time grid."""
    return create_pkpd_task(
        seed=0,
        dataset_path=str(tmp_path / "pkpd"),
        point_allocation=((0.0, 1.9, 8), (1.9, 4.0, 8), (4.0, 17.0, 8)),
        hidden_size=4,
        num_layers=2,
    )


@pytest.mark.slow
def test_fields_and_dataset(tmp_path):
    task = _build(tmp_path)
    assert isinstance(task, Task)
    assert task.name == "pkpd"
    assert task.tag["loss_fn"] == "pkpd_loss"
    assert task.loaded_dataset["t_res"].shape == (24, 1)


def test_analytic_concentration_matches_ode_integration():
    """The analytic c(t) reproduces a direct integration of Eq. 78-79.

    Integrates the two-compartment ODEs ``dA1/dt`` / ``dA2/dt`` with a
    single unit bolus at ``t = 0`` and checks the resulting central
    concentration ``c = A1`` against :func:`concentration`.
    """
    k10 = _K10_PER_HOUR * _HOURS_PER_DAY
    k12 = _K12_PER_HOUR * _HOURS_PER_DAY
    k21 = _K21_PER_HOUR * _HOURS_PER_DAY

    dt = 1e-4
    n = int(round(2.0 / dt))
    a1, a2 = 1.0, 0.0
    sample_times = {0.5: None, 1.0: None, 2.0: None}
    for step in range(1, n + 1):
        d1 = -(k10 + k12) * a1 + k21 * a2
        d2 = k12 * a1 - k21 * a2
        a1 += dt * d1
        a2 += dt * d2
        t = step * dt
        for target in sample_times:
            if sample_times[target] is None and abs(t - target) < dt / 2:
                sample_times[target] = a1

    ts = jnp.array([[t] for t in sample_times])
    analytic = concentration(ts, k10, k12, k21, dose=1.0, dose_times=(0.0,))
    for value, target in zip(analytic, sample_times, strict=True):
        assert float(value) == pytest.approx(sample_times[target], abs=1e-3)


def test_concentration_switches_on_at_dose_time():
    """Concentration is zero before the (single, positive) dose time."""
    t = jnp.array([[0.0], [1.0], [3.0]])
    c = concentration(t, 21.0, 0.14, 2.0, dose=1.0, dose_times=(2.0,))
    assert float(c[0]) == 0.0
    assert float(c[1]) == 0.0
    assert float(c[2]) > 0.0
    assert np.isfinite(np.asarray(c)).all()


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
