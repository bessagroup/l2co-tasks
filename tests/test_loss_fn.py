"""Unit tests for loss functions in ``_src.loss_fn``."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

from l2co_tasks._src.loss_fn import (
    mean_categorical_cross_entropy_loss_fn,
    mean_categorical_cross_entropy_loss_fn_l2,
    mse_loss_fn,
)


def _identity_model(x):
    return x


def test_mse_loss_zero_on_exact_match():
    """Identity model with y == x ⇒ loss is 0."""
    x = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    y = x.copy()
    assert float(mse_loss_fn(_identity_model, x, y)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_mse_loss_matches_manual():
    """MSE is averaged over every element, not every sample."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.array([[1.5, 1.5], [2.5, 4.5]])
    # residual squares sum to 1.0, averaged over 4 elements ⇒ 0.25.
    assert float(mse_loss_fn(_identity_model, x, y)) == pytest.approx(
        0.25, rel=1e-6
    )


def test_cross_entropy_lower_for_correct_labels():
    """CE loss on correct labels must be strictly less than on
    shuffled ones."""
    num_classes = 3

    def model(x):
        # Peaked logits at ``int(x)`` ⇒ argmax of softmax == int(x).
        return jnp.zeros(num_classes).at[x.astype(jnp.int32)].set(10.0)

    x = jnp.array([0, 1, 2, 0, 1, 2], dtype=jnp.float32)
    correct = jnp.array([0, 1, 2, 0, 1, 2])
    shuffled = jnp.array([2, 0, 1, 2, 0, 1])

    loss_correct = mean_categorical_cross_entropy_loss_fn(model, x, correct)
    loss_shuffled = mean_categorical_cross_entropy_loss_fn(model, x, shuffled)
    assert float(loss_correct) < float(loss_shuffled)


def test_cross_entropy_l2_decomposes_additively(rng_key):
    """``ce_l2`` == ``ce`` + ``l2_lambda * sum_of_squares(leaves(model))``.

    Uses a real ``eqx.nn.Linear`` model so the L2 sum iterates over the
    actual parameter pytree the way the loss implementation does.
    """
    linear = eqx.nn.Linear(3, 2, key=rng_key)
    x = jr.normal(rng_key, (4, 3))
    y = jnp.array([0, 1, 0, 1])
    l2_lambda = 0.05

    base = float(mean_categorical_cross_entropy_loss_fn(linear, x, y))
    combined = float(
        mean_categorical_cross_entropy_loss_fn_l2(
            linear, x, y, l2_lambda=l2_lambda
        )
    )
    import jax.tree_util as jtu

    l2_penalty = sum(float(jnp.sum(p**2)) for p in jtu.tree_leaves(linear))
    assert combined == pytest.approx(base + l2_lambda * l2_penalty, rel=1e-5)
