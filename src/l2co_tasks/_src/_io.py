"""Dataset loading and saving helpers."""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Callable
from pathlib import Path

# Third-party
import jax
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


def ensure_points_dataset(
    path: str | Path, build: Callable[[], dict[str, jnp.ndarray]]
) -> None:
    """Write a PINN point set once; refuse a cached one that no longer fits.

    The PINN factories cache their collocation points at ``path`` and
    write the file only when it is missing. A file already there is
    reused only when it holds the same arrays, with the same shapes, as
    ``build()`` would give; otherwise this raises instead of letting a
    task whose tag records new point counts run on the old points
    (ADR 0008: the tag is the task's whole identity).

    Only array names and shapes are compared, so a cached file written
    under another dtype or JAX release is still reused; a change that
    keeps every shape (a grid resolution, or a pkpd block boundary) is
    not caught.

    Parameters
    ----------
    path : str | Path
        The ``.npz`` file of the cached point set.
    build : Callable[[], dict[str, jnp.ndarray]]
        Builds the point set the task asks for.

    Raises
    ------
    ValueError
        If the cached file holds other arrays or other shapes than
        ``build()``.
    """
    _path = Path(path)
    if not _path.exists():
        save_dataset(build(), _path)
        return
    expected = {k: tuple(v.shape) for k, v in jax.eval_shape(build).items()}
    with np.load(_path) as cached:
        found = {k: tuple(cached[k].shape) for k in cached.files}
    if found != expected:
        raise ValueError(
            f"The cached point set {_path} holds arrays {found}, but this "
            f"task asks for {expected}. It was written by a task with other "
            "point counts; pass another dataset_path or delete the file."
        )
