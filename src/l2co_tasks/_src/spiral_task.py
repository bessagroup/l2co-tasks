"""Two-spiral binary classification task with a GRU-RNN.

The synthetic dataset arranges points along two interleaved spirals,
one per class. Each sample is a length-16 sequence of ``(sin, cos)``
features (shape ``(16, 2)``) traced along a spiral, with binary labels
in ``{0, 1}``; the two classes are mirror images of one another.

The model is a recurrent network: a ``GRUCell`` (input size 2, width
``hidden_size``) scanned over the 16 timesteps with ``jax.lax.scan``,
followed by a bias-free linear read-out and a sigmoid, producing a
scalar probability per sequence. The loss is mean-squared error
against the ``{0, 1}`` label. The dataset is generated once, cached to
``.npz`` and trained in mini-batches.

``global_min`` is set *empirically*: a short, seeded multi-restart Adam
benchmark records the best achievable MSE at task-creation time (see
:func:`l2co_tasks._src.global_min.estimate_global_min`), since a
finite-width GRU need not drive the MSE to 0. Pass
``estimate_global_min=False`` to skip the benchmark.

Public API
----------
create_spiral_task
    Build a spiral :class:`Task` from ``hidden_size``,
    ``dataset_size``, ``dataset_path``, ``seed`` and ``batch_size``.
"""

# Local
import dataclasses
import math
from pathlib import Path

# Third-party
import jax.numpy as jnp
import jax.random as jrd

from ._io import save_dataset
from .global_min import estimate_global_min as _estimate_global_min
from .loss_fn import mse_loss_fn
from .models import rnn

# Local
from .task import Task, count_parameters, dataset_dict

# =============================================================================


def spiral_dataset(dataset_size, *, key: jrd.PRNGKey):
    """Generate a two-class spiral dataset.

    Parameters
    ----------
    dataset_size : int
        Total number of samples (split evenly between classes).
    key : jrd.PRNGKey
        Random key for offset sampling.

    Returns
    -------
    dict[str, jnp.ndarray]
        ``'x'`` features ``(N, 16, 2)`` and ``'y'`` binary
        labels ``(N, 1)``.
    """
    t = jnp.linspace(0, 2 * math.pi, 16)
    offset = jrd.uniform(key, (dataset_size, 1), minval=0, maxval=2 * math.pi)
    x1 = jnp.sin(t + offset) / (1 + t)
    x2 = jnp.cos(t + offset) / (1 + t)
    y = jnp.ones((dataset_size, 1))

    half_dataset_size = dataset_size // 2
    x1 = x1.at[:half_dataset_size].multiply(-1)
    y = y.at[:half_dataset_size].set(0)
    x = jnp.stack([x1, x2], axis=-1)

    return {"x": x, "y": y}


def create_spiral_task(
    hidden_size: int,
    dataset_size: int,
    dataset_path: str,
    seed: int,
    batch_size: int,
    estimate_global_min: bool = True,
    global_min_restarts: int = 3,
    global_min_steps: int = 1000,
) -> Task:
    """Create a spiral binary classification task using an RNN.

    Parameters
    ----------
    hidden_size : int
        RNN hidden state size.
    dataset_size : int
        Number of spiral samples.
    dataset_path : str
        Path prefix for the dataset file.
    seed : int
        Random seed for reproducibility.
    batch_size : int
        Mini-batch size for training.
    estimate_global_min : bool, optional
        Whether to benchmark ``global_min`` at creation with a short
        multi-restart Adam search, by default True. When False,
        ``global_min`` is left as ``None``.
    global_min_restarts : int, optional
        Number of Adam restarts for the ``global_min`` benchmark, by
        default 3.
    global_min_steps : int, optional
        Number of Adam steps per restart for the benchmark, by default
        1000.

    Returns
    -------
    Task
        Configured spiral classification task. ``global_min`` is an
        *empirical* best-achievable-loss estimate, not a guaranteed
        lower bound.
    """
    # Coerce numpy scalars (e.g. produced by the f3dasm random
    # sampler over an int-typed domain) back to Python ints so the
    # tag and dataset header round-trip through ``json.dumps`` in
    # ``Task.save``.
    seed = int(seed)
    hidden_size = int(hidden_size)
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)

    key = jrd.key(seed)

    model_key, dataset_key, batch_key = jrd.split(key, 3)

    tag = {"task_name": "spiral"}

    # model
    model, model_tags = rnn(
        in_size=2, out_size=1, hidden_size=hidden_size, key=model_key
    )

    # Count trainable parameters
    num_params = count_parameters(model)
    tag["dimensionality"] = num_params

    tag.update(model_tags)

    if not Path(dataset_path).with_suffix(".npz").exists():
        dataset = spiral_dataset(dataset_size=dataset_size, key=key)
        save_dataset(dataset, dataset_path)

    # loss function
    loss_fn = mse_loss_fn

    tag.update({"loss_fn": "mse"})

    task = Task(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset_dict(dataset_path, seed, batch_size),
        tag=tag,
    )

    if estimate_global_min:
        gmin = _estimate_global_min(
            task,
            seed=seed,
            n_restarts=global_min_restarts,
            n_steps=global_min_steps,
        )
        task = dataclasses.replace(task, global_min=gmin)

    return task
