"""Random quadratic least-squares optimisation task.

Builds tasks of the form

    min_x  scale * ||W x - y||^2

where ``W`` (shape ``(m, n)``) and ``y`` (shape ``(m,)``) have i.i.d.
standard-normal entries and ``x`` (length ``n``) is the parameter
vector, initialised at the origin.

By default the problem is square (``m = n``) and unnormalised
(``scale = 1``), so its minimiser is the unique solution of ``W x = y``
with zero residual. Passing ``n_observations > n`` makes it
over-determined (a tall, well-conditioned least-squares problem with a
non-zero residual minimum), and ``normalize=True`` sets
``scale = 1 / (2 m)``, reproducing the linear-regression objective of
Maheswaranathan et al. (2019).

The model is a plain JAX vector and the loss is deterministic (no RNG,
no batching).

References
----------
Maheswaranathan et al., "Guided Evolutionary Strategies: Augmenting
random search with surrogate gradients", ICML 2019.

Public API
----------
create_quadratic_task
    Build a quadratic :class:`Task` from ``dimensionality`` and
    ``seed`` (optionally ``n_observations`` and ``normalize``).
"""

from functools import partial
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from .task import Task

# =============================================================================


def generate_dataset(
    dimensionality: int,
    *,
    n_observations: int | None = None,
    key: jr.PRNGKey,
):
    """Generate a random quadratic problem ``(W, y)``.

    Entries of both ``W`` and ``y`` are drawn i.i.d. from a standard
    normal distribution, matching Maheswaranathan et al. (2019).

    Parameters
    ----------
    dimensionality : int
        Number of unknowns ``n`` (columns of ``W``, length of ``x``).
    n_observations : int, optional
        Number of rows ``m`` of ``W`` (equations / observations). When
        ``None`` (the default) the problem is square (``m = n``). Pass
        ``m > n`` for an over-determined least-squares problem (the
        paper uses ``m = 2 n``).
    key : jr.PRNGKey
        Random key for sampling.

    Returns
    -------
    dict[str, jnp.ndarray]
        ``'W'`` matrix of shape ``(m, n)`` and ``'y'`` target vector of
        shape ``(m,)``.
    """
    m = dimensionality if n_observations is None else n_observations
    key_W, key_y = jr.split(key)
    Ws = jr.normal(key_W, shape=(m, dimensionality))
    ys = jr.normal(key_y, shape=(m,))
    return {"W": Ws, "y": ys}


def save_dataset(dataset: dict[str, jnp.ndarray], path: str | Path):
    """Save a dataset dictionary to an ``.npz`` file.

    Parameters
    ----------
    dataset : dict[str, jnp.ndarray]
        Dictionary of arrays to persist.
    path : str or Path
        Destination file path.
    """
    _path = Path(path)
    _path.parent.mkdir(parents=True, exist_ok=True)

    jnp.savez(path, **dataset)


# =============================================================================


def compute_quadratic_loss(model, W, y, scale=1.0):
    """Compute the (optionally scaled) squared residual loss.

    Returns ``scale * ||W x - y||^2``. With the default ``scale=1.0``
    this is the plain sum of squared residuals; passing
    ``scale = 0.5 / W.shape[0]`` reproduces the ``1 / (2 M)`` mean
    least-squares objective of Maheswaranathan et al. (2019).

    Parameters
    ----------
    model : jnp.ndarray
        Current parameter vector of shape ``(n,)``.
    W : jnp.ndarray
        Coefficient matrix of shape ``(m, n)``.
    y : jnp.ndarray
        Target vector of shape ``(m,)``.
    scale : float, optional
        Multiplicative factor applied to the summed squared residual.
        Defaults to ``1.0``.

    Returns
    -------
    jnp.ndarray
        Scalar (scaled) sum of squared residuals.
    """
    residual = jnp.dot(W, model) - y
    return scale * jnp.sum(residual**2)


def create_quadratic_task(
    dimensionality: int,
    seed: int,
    *,
    n_observations: int | None = None,
    normalize: bool = False,
) -> Task:
    """Create a quadratic optimisation task ``min ||W x - y||^2``.

    By default the task is the square, unnormalised sum-of-squares
    problem ``min ||W x - y||^2`` with ``W`` of shape ``(d, d)``. To
    reproduce the linear-regression objective of Maheswaranathan et al.
    (2019), ``f(x) = 1 / (2 M) ||A x - b||^2`` with ``M = 2 N``, pass
    ``n_observations=2 * dimensionality`` and ``normalize=True``.

    Parameters
    ----------
    dimensionality : int
        Number of unknowns ``n`` (length of the parameter vector).
    seed : int
        Random seed for matrix and vector generation.
    n_observations : int, optional
        Number of rows ``m`` of ``W``. ``None`` (default) gives a square
        problem (``m = n``); ``m > n`` gives an over-determined
        least-squares problem.
    normalize : bool, optional
        When ``True`` the loss is scaled by ``1 / (2 m)`` (the paper's
        mean least-squares normalisation). When ``False`` (default) the
        plain summed squared residual is returned.

    Returns
    -------
    Task
        Configured quadratic optimisation task. Its ``global_min`` is
        set to the least-squares optimum ``f*``: ``0.0`` when ``m <= n``
        (an exact fit exists) and the scaled minimum residual otherwise.
    """
    # Coerce numpy scalars (e.g. produced by the f3dasm random
    # sampler over an int-typed domain) back to Python ints so the
    # tag round-trips through ``json.dumps`` in ``Task.save``.
    seed = int(seed)
    dimensionality = int(dimensionality)
    m = dimensionality if n_observations is None else int(n_observations)

    tag = {}
    tag["seed"] = seed
    tag["task_name"] = "quadratic_functions"
    tag["dimensionality"] = dimensionality
    tag["n_observations"] = m
    tag["normalize"] = normalize

    key = jr.key(seed)

    model_key, dataset_key, batch_key = jr.split(key, 3)

    model = jnp.zeros(
        dimensionality,
    )

    dataset = generate_dataset(
        dimensionality=dimensionality,
        n_observations=m,
        key=dataset_key,
    )

    scale = 0.5 / m if normalize else 1.0
    loss_fn = partial(
        compute_quadratic_loss,
        W=dataset["W"],
        y=dataset["y"],
        scale=scale,
    )

    # The least-squares optimum f* of ``scale * ||W x - y||^2``. When
    # ``m <= n`` the system has an exact solution (zero residual);
    # otherwise it is over-determined and f* is the scaled residual at
    # the pseudo-inverse solution.
    if m <= dimensionality:
        global_min = 0.0
    else:
        x_star, *_ = jnp.linalg.lstsq(dataset["W"], dataset["y"])
        residual = dataset["W"] @ x_star - dataset["y"]
        global_min = float(scale * jnp.sum(residual**2))

    return Task(
        model=model,
        loss_fn=loss_fn,
        dataset=None,
        tag=tag,
        global_min=global_min,
    )
