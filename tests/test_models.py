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


# ---------------------------------------------------------------------------
# FourierMLP / MultiNet (PINN building blocks)
# ---------------------------------------------------------------------------


def test_fourier_mlp_feature_dim_and_periodicity(rng_key):
    """Embedding width is ``in_dim * len(modes) * 2`` and is periodic."""
    import jax.numpy as jnp

    from l2co_tasks._src.models import fourier_mlp

    model, tags = fourier_mlp(
        in_dim=2,
        out_size=1,
        modes=(1,),
        length=2.0,
        hidden_size=8,
        num_layers=4,
        key=rng_key,
    )
    assert tags["model"] == "fourier_mlp"
    assert tags["modes"] == (1,)
    # First linear layer consumes the embedded features (2 dims * 1 mode * 2).
    assert model.mlp.layers[0].weight.shape[1] == 4
    # Periodicity with period ``length`` is exact.
    x = jnp.array([0.3, -0.7])
    assert jnp.allclose(model(x), model(x + jnp.array([2.0, 0.0])), atol=1e-5)


def test_fourier_mlp_3d_kmax_two(rng_key):
    """3-D ``k_max = 2`` gives 12 input features (paper Eq. 40)."""
    from l2co_tasks._src.models import fourier_mlp

    model, _ = fourier_mlp(
        in_dim=3,
        out_size=1,
        modes=(1, 2),
        length=2.0,
        hidden_size=8,
        num_layers=4,
        key=rng_key,
    )
    assert model.mlp.layers[0].weight.shape[1] == 12


def test_multinet_keyed_and_positional_access(rng_key):
    """``MultiNet`` exposes sub-networks by name and position."""
    import jax.random as jr

    from l2co_tasks._src.models import mlp, multinet

    u, _ = mlp(2, 1, 4, 2, key=jr.key(0))
    f, _ = mlp(2, 1, 4, 2, key=jr.key(1))
    model, tags = multinet({"u": u, "flux": f})
    assert tags["names"] == ("u", "flux")
    assert model["u"] is model.nets[0]
    assert model["flux"] is model[1]
