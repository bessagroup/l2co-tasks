"""Dataset and model loading helpers.

Copied from ``l2co._src.tree_utils`` so that ``l2co_tasks`` can load
task artifacts without depending on the rest of ``l2co``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
import cloudpickle
import equinox as eqx
import jax.numpy as jnp

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def load_model(model_path: str, shape_path: str) -> eqx.Module:
    """Load model from disk.

    Parameters
    ----------
    model_path : str
        Path to the model weights file (.eqx).
    shape_path : str
        Path to the model shape file (.pkl).

    Returns
    -------
    eqx.Module
        Loaded model.
    """
    _model_path = Path(model_path)
    _shape_path = Path(shape_path)
    with open(_shape_path.with_suffix(".pkl"), "rb") as f:
        model_shape = cloudpickle.load(f)

    model = eqx.tree_deserialise_leaves(
        _model_path.with_suffix(".eqx"), model_shape
    )

    return model


def load_jnparray(path: str | Path) -> jnp.ndarray:
    """Load JAX numpy array from disk.

    Parameters
    ----------
    path : str | Path
        Path to the numpy file.

    Returns
    -------
    jnp.ndarray
        Loaded array (flattened).
    """
    # TODO: investigate this flatten behaviour
    return jnp.load(Path(path)).flatten()


def load_dataset(path: str | Path | None) -> dict[str, jnp.ndarray]:
    """Load dataset from npz file.

    Parameters
    ----------
    path : str | Path | None
        Path to the dataset file. If None, returns empty dict.

    Returns
    -------
    dict[str, jnp.ndarray]
        Loaded dataset dictionary.
    """
    if path is None:
        return {}
    loaded_data = jnp.load(Path(path).with_suffix(".npz"))
    return {k: jnp.array(v) for k, v in loaded_data.items()}
