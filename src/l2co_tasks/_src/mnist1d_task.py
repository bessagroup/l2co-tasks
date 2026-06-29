"""Ten-class MNIST-1D classification task with an MLP.

MNIST-1D (Greydanus, 2020) is a compact 1-D analogue of MNIST: each
sample is a length-40 vector (shape ``(40,)``) and an integer label in
``0..9``. Samples are generated with the ``mnist1d`` package, cached to
``.npz`` and trained in mini-batches.

The model is a fixed-architecture MLP, ``40 -> 16 -> 16 -> 10`` with
``tanh`` hidden activations and a linear (logit) output layer; its
width and depth are not configurable from the factory. The loss is the
softmax cross-entropy between the output logits and the integer labels
(no L2 regularisation).

``global_min`` is set *empirically*: a short, seeded multi-restart Adam
benchmark records the best achievable cross-entropy at task-creation
time (see :func:`l2co_tasks._src.global_min.estimate_global_min`), since
the irreducible loss floor of this data is strictly positive. Pass
``estimate_global_min=False`` to skip the benchmark.

References
----------
Greydanus, "Scaling down Deep Learning", 2020 (MNIST-1D).

Public API
----------
create_mnist1d_task
    Build an MNIST-1D :class:`Task` from ``dataset_path``,
    ``dataset_size``, ``seed`` and ``batch_size``.
"""

import dataclasses
from pathlib import Path

import jax.random as jr
from mnist1d.data import get_dataset_args, make_dataset

from ._io import save_dataset
from .global_min import estimate_global_min as _estimate_global_min
from .loss_fn import mean_categorical_cross_entropy_loss_fn
from .models import mlp
from .task import Task, count_parameters, dataset_dict


def mnist1d_dataset(dataset_size, seed: int):
    """Generate an MNIST-1D dataset.

    Parameters
    ----------
    dataset_size : int
        Number of samples to generate.
    seed : int
        Random seed for dataset generation.

    Returns
    -------
    dict[str, np.ndarray]
        ``'x'`` features and ``'y'`` labels.
    """
    defaults = get_dataset_args()
    defaults.num_samples = dataset_size
    defaults.seed = seed
    data = make_dataset(defaults)
    return {"x": data["x"], "y": data["y"]}


def create_mnist1d_task(
    dataset_path: str,
    dataset_size: int,
    seed: int,
    batch_size: int,
    estimate_global_min: bool = True,
    global_min_restarts: int = 3,
    global_min_steps: int = 1000,
) -> Task:
    """Create an MNIST-1D classification task.

    Parameters
    ----------
    dataset_path : str
        Path prefix for the dataset file.
    dataset_size : int
        Number of samples in the dataset.
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
        Configured MNIST-1D classification task. ``global_min`` is an
        *empirical* best-achievable-loss estimate, not a guaranteed
        lower bound.
    """
    # Coerce numpy scalars (e.g. produced by the f3dasm random
    # sampler over an int-typed domain) back to Python ints so the
    # tag round-trips through ``json.dumps`` in ``Task.save`` and so
    # ``mnist1d.utils.set_seed`` accepts the value (stdlib
    # ``random.seed`` rejects numpy integer scalars).
    seed = int(seed)
    dataset_size = int(dataset_size)
    batch_size = int(batch_size)

    tag = {"task_name": "mnist1d"}
    tag["seed"] = seed

    if not Path(dataset_path).with_suffix(".npz").exists():
        dataset = mnist1d_dataset(dataset_size=dataset_size, seed=seed)
        save_dataset(dataset, dataset_path)

    model, model_tags = mlp(
        in_size=40, out_size=10, hidden_size=16, num_layers=2, key=jr.key(seed)
    )

    # Count trainable parameters
    num_params = count_parameters(model)
    tag["dimensionality"] = num_params

    # Loss fn
    loss_fn = mean_categorical_cross_entropy_loss_fn

    tag.update({"loss_fn": "mean_categorical_cross_entropy_loss_fn"})

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
