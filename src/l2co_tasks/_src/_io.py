"""Dataset loading helpers."""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
import jax.numpy as jnp
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


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
    with np.load(Path(path).with_suffix(".npz")) as loaded_data:
        return {k: jnp.asarray(loaded_data[k]) for k in loaded_data.files}
