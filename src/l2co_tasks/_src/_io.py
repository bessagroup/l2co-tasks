"""Dataset loading and saving helpers."""

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


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path) -> None:
    """Save a dataset of arrays to an ``.npz`` file.

    Creates the parent directory if needed. The write-side twin of
    :func:`load_dataset`; the task factories delegate here rather than
    each carrying its own copy.

    Parameters
    ----------
    dataset : dict[str, jnp.ndarray]
        Dictionary of arrays to persist.
    path : str | Path
        Destination file path.
    """
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)
    jnp.savez(path, **dataset)
