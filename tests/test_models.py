"""Unit tests for the RNN/MLP/CNN constructors in ``_src.models``."""

from __future__ import annotations

import equinox as eqx
import jax.nn
import jax.numpy as jnp
import pytest

from l2co_tasks._src.models import cnn, mlp, rnn


def test_rnn_builds_and_outputs_shape(rng_key):
    model, tags = rnn(in_size=2, out_size=1, hidden_size=4, key=rng_key)
    assert isinstance(model, eqx.Module)
    assert tags == {
        "model": "rnn",
        "in_size": 2,
        "out_size": 1,
        "hidden_size": 4,
    }
    out = model(jnp.zeros((3, 2)))
    assert out.shape == (1,)
    # RNN ends in sigmoid → outputs must lie in (0, 1).
    assert bool(jnp.all((out > 0.0) & (out < 1.0)))


@pytest.mark.parametrize(
    "in_size,hidden,out_size,num_layers",
    [
        (4, 8, 2, 2),
        (1, 3, 5, 3),
    ],
)
def test_mlp_builds_and_outputs_shape(
    rng_key, in_size, hidden, out_size, num_layers
):
    model, tags = mlp(
        in_size=in_size,
        out_size=out_size,
        hidden_size=hidden,
        num_layers=num_layers,
        key=rng_key,
    )
    assert isinstance(model, eqx.Module)
    assert tags["model"] == "mlp"
    assert tags["in_size"] == in_size and tags["out_size"] == out_size
    assert tags["hidden_size"] == hidden and tags["num_layers"] == num_layers
    # The default output activation is a fresh ``lambda x: x``, so the
    # identity-sentinel check in ``mlp`` never matches and the tag is
    # ``"<lambda>"``. Documented here so a future fix that makes the
    # sentinel work (e.g. a module-level ``_identity``) is caught.
    assert tags["output_activation"] == "<lambda>"

    out = model(jnp.zeros(in_size))
    assert out.shape == (out_size,)


def test_mlp_named_activations_in_tags(rng_key):
    """Non-identity output activations are tagged by their ``__name__``."""
    _, tags = mlp(
        in_size=2,
        out_size=2,
        hidden_size=2,
        num_layers=2,
        hidden_activation=jax.nn.relu,
        output_activation=jax.nn.softmax,
        key=rng_key,
    )
    assert tags["hidden_activation"] == "relu"
    assert tags["output_activation"] == "softmax"


def test_cnn_builds_and_outputs_shape(rng_key):
    model, tags = cnn(
        in_size=8,
        out_size=3,
        hidden_channels=2,
        num_layers=2,
        kernel_size=3,
        height=1,
        width=8,
        in_channels=1,
        key=rng_key,
    )
    assert isinstance(model, eqx.Module)
    assert tags["model"] == "cnn"
    assert tags["out_size"] == 3
    out = model(jnp.zeros(8))
    assert out.shape == (3,)
    # Final activation is log_softmax ⇒ exp(out) sums to 1.
    assert float(jnp.sum(jnp.exp(out))) == pytest.approx(1.0, rel=1e-4)
