"""Gaussian-blob binary classification task with an MLP.

The synthetic dataset draws ``num_gaussians`` clusters, each a
multivariate normal with a random mean in ``[-5, 5]^d`` and a random
PSD covariance, and ``num_samples_per_gaussian`` points per cluster.
Each cluster is assigned a binary label in ``{0, 1}`` (with both
classes guaranteed present), giving inputs of shape
``(N, dim_points)``.

The model is an MLP (input ``dim_points``, output 2, ``relu`` hidden
activations and a ``softmax`` output) with configurable ``hidden_size``
and ``num_layers``. The loss is softmax cross-entropy plus an L2 weight
penalty scaled by ``l2_regularization``. The true minimum is unreachable
-- overlapping Gaussian clusters forbid perfect separation -- so
``global_min`` is set *empirically*: a short, seeded multi-restart Adam
benchmark records the best achievable loss at task-creation time (see
:func:`l2co_tasks._src.global_min.estimate_global_min`). Pass
``estimate_global_min=False`` to skip the benchmark and leave
``global_min`` as ``None``.

Public API
----------
create_gaussian_task
    Build a Gaussian-classification :class:`Task` from ``seed``,
    ``dataset_path``, the dataset shape (``num_gaussians``,
    ``num_samples_per_gaussian``, ``dim_points``),
    ``l2_regularization`` and the model size (``hidden_size``,
    ``num_layers``).
"""

# Standard
import dataclasses
from functools import partial
from pathlib import Path

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jrd
import jax.random as random

from ._io import save_dataset
from .global_min import estimate_global_min as _estimate_global_min
from .loss_fn import mean_categorical_cross_entropy_loss_fn_l2
from .models import mlp

# Local
from .task import Task, count_parameters, dataset_dict

# =============================================================================


def generate_one_gaussian_dataset(
    num_gaussians=4, num_samples_per_gaussian=25, dim=2, *, key: jrd.PRNGKey
) -> dict[str, jnp.ndarray]:
    """
    Generate a synthetic dataset of Gaussian distributions.

    Parameters
    ----------
    num_gaussians : int, optional
        Number of Gaussian distributions, by default 4.
    num_samples_per_gaussian : int, optional
        Number of samples per Gaussian, by default 25.
    dim : int, optional
        Dimensionality of the data, by default 2.
    key : jrd.PRNGKey
        Random key for reproducibility.

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary containing the generated samples and labels.
    """

    # total_samples = num_gaussians * num_samples_per_gaussian

    keys = random.split(key, num_gaussians * 2 + 1)
    dataset_key = keys[0]

    mean_keys = keys[1 : 1 + num_gaussians]
    cov_keys = keys[1 + num_gaussians : 1 + 2 * num_gaussians]
    label_key = keys[-1]

    # Random means in given dimension
    means = jnp.stack([random.normal(k, (dim,)) * 5.0 for k in mean_keys])
    covs = jnp.stack([random.uniform(k, (dim, dim)) for k in cov_keys])
    # Ensure positive semi-definite covariance
    covs = covs @ jnp.transpose(covs, (0, 2, 1))

    # Assign labels to Gaussians ensuring at least one of each label
    gaussian_labels = random.choice(
        label_key, jnp.array([0, 1]), shape=(num_gaussians,)
    )
    if jnp.all(gaussian_labels == 0) or jnp.all(gaussian_labels == 1):
        gaussian_labels = gaussian_labels.at[0].set(1 - gaussian_labels[0])

    samples = []
    labels = []

    for i in range(num_gaussians):
        gaussian_key = random.fold_in(dataset_key, i)
        points = random.multivariate_normal(
            gaussian_key, means[i], covs[i], (num_samples_per_gaussian,)
        )
        samples.append(points)
        labels.append(
            jnp.full((num_samples_per_gaussian,), gaussian_labels[i])
        )

    samples = jnp.vstack(samples)
    labels = jnp.concatenate(labels)

    return {"x": samples, "y": labels}


# =============================================================================


def create_gaussian_task(
    seed: int,
    dataset_path: str,
    num_gaussians: int,
    num_samples_per_gaussian: int,
    dim_points: int,
    l2_regularization: float,
    hidden_size: int = 2,
    num_layers: int = 2,
    estimate_global_min: bool = True,
    global_min_restarts: int = 3,
    global_min_steps: int = 1000,
) -> Task:
    """
    Create a Gaussian classification task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path to save the generated dataset.
    num_gaussians : int
        Number of Gaussian distributions.
    num_samples_per_gaussian : int
        Number of samples per Gaussian.
    dim_points : int
        Dimensionality of the data points.
    l2_regularization : float
        L2 regularization parameter for the loss function.
    hidden_size : int, optional
        Size of the hidden layers in the MLP, by default 2.
    num_layers : int, optional
        Number of layers in the MLP, by default 2.
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
        The created Gaussian classification task.

    Notes
    -----
    The classes overlap, so the loss cannot reach 0; ``global_min`` is
    therefore an *empirical* estimate of the best achievable loss rather
    than a guaranteed lower bound (see
    :func:`l2co_tasks._src.global_min.estimate_global_min`).
    """

    tag = {}

    # add the seed to the tags (this is the identifier for the task)
    tag["seed"] = seed

    # add '_seed' to dataset_path
    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")

    key = jrd.key(int(seed))

    model_key, dataset_key, batch_key = jrd.split(key, 3)

    tag["task_name"] = "gaussian_classification"
    tag["num_gaussians"] = num_gaussians
    tag["num_samples_per_gaussian"] = num_samples_per_gaussian
    tag["dim_points"] = dim_points
    tag["hidden_size"] = hidden_size
    tag["num_layers"] = num_layers

    model, model_tags = mlp(
        in_size=dim_points,
        out_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.relu,
        output_activation=jax.nn.softmax,
        key=model_key,
    )

    # Count trainable parameters
    num_params = count_parameters(model)
    tag["dimensionality"] = num_params

    tag.update(model_tags)

    if not _path.exists():
        dataset = generate_one_gaussian_dataset(
            num_gaussians=num_gaussians,
            num_samples_per_gaussian=num_samples_per_gaussian,
            dim=dim_points,
            key=dataset_key,
        )
        save_dataset(dataset, _path)

    loss_fn = partial(
        mean_categorical_cross_entropy_loss_fn_l2, l2_lambda=l2_regularization
    )

    tag["loss_fn"] = "cross_entropy_loss"
    tag["l2_regularization"] = l2_regularization
    tag["separable"] = False
    tag["unimodal"] = False

    task = Task(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset_dict(_path, seed),
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


# =============================================================================
