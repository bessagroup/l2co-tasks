"""Physics-informed neural network (PINN) tasks for the Helmholtz equation.

A single factory builds the 2-D or 3-D Helmholtz problem on the box
``[-1, 1]^dim`` (Jnini et al. 2026, Section 4.1):

- 2-D: ``u_xx + u_yy + k**2 u - q(x, y) = 0`` with
  ``q = (k**2 - (a1*pi)**2 - (a2*pi)**2) * sin(a1*pi*x) sin(a2*pi*y)`` and
  exact solution ``u = sin(a1*pi*x) sin(a2*pi*y)``.
- 3-D: the analogous problem with a third wavenumber ``a3``.

Periodic boundary conditions are enforced *structurally* by a periodic
Fourier-feature embedding (:class:`~l2co_tasks._src.models.FourierMLP`,
paper Eq. 40), so the training loss is the interior PDE-residual
mean-squared error alone. Second derivatives are taken with
``jax.hessian``.

Public API
----------
create_helmholtz_task
    Build a Helmholtz PINN :class:`Task` for ``dim`` in ``{2, 3}``.
HelmholtzTaskSampler
    ``f3dasm.Block`` enumerating the paper's ``(a1, a2[, a3], k)``
    configurations across hidden-layer sizes.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path

# Third-party
import jax
import jax.numpy as jnp
import jax.random as jr
from f3dasm import Block, ExperimentData
from jaxtyping import Array, PyTree

# Local
from ._io import save_dataset
from .experimentdata import build_task_experimentdata
from .models import fourier_mlp
from .pde import mse
from .pinn import latin_hypercube
from .task import Task, count_parameters, dataset_dict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def helmholtz_pde_loss(
    model: PyTree,
    x_res: Array,
    a: tuple[float, ...],
    k: float,
    loss_fn_res: Callable[[Array, Array], Array],
) -> Array:
    """Compute the interior PDE-residual loss for the Helmholtz equation.

    Residual ``Laplacian(u) + k**2 u - q`` with
    ``q = (k**2 - sum_i (a_i*pi)**2) * prod_i sin(a_i*pi*x_i)``. The
    Laplacian is the trace of the per-point Hessian of the network.

    Parameters
    ----------
    model : PyTree
        Neural network approximating the solution ``u``.
    x_res : Array
        Interior collocation points (``dim`` is 2 or 3).
    a : tuple[float, ...]
        Wavenumbers ``(a1, a2[, a3])``, one per spatial dimension.
    k : float
        Helmholtz wavenumber ``k``.
    loss_fn_res : Callable
        Loss function applied to the residual (mean-squared error).

    Returns
    -------
    Array
        Scalar residual loss.
    """
    a_arr = jnp.asarray(a)

    def u_point(p: Array) -> Array:
        """Scalar network output at a single point ``p``."""
        return model(p)[0]

    def residual(p: Array) -> Array:
        """PDE residual at a single point ``p``."""
        u = u_point(p)
        lap = jnp.trace(jax.hessian(u_point)(p))
        coeff = k**2 - jnp.sum((a_arr * jnp.pi) ** 2)
        q = coeff * jnp.prod(jnp.sin(jnp.pi * a_arr * p))
        return lap + k**2 * u - q

    res = jax.vmap(residual)(x_res)
    return loss_fn_res(res, jnp.zeros_like(res))


def create_points_dataset(
    x_range: tuple[float, float],
    dim: int,
    num_res_points: int,
    *,
    key: Array,
) -> dict[str, jnp.ndarray]:
    """Create interior collocation points for a Helmholtz task.

    Parameters
    ----------
    x_range : tuple[float, float]
        Per-axis ``(lo, hi)`` range of the box domain.
    dim : int
        Spatial dimensionality (2 or 3).
    num_res_points : int
        Number of interior residual points.
    key : Array
        Random key for reproducibility.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary with key ``"x_res"`` of shape ``(num_res_points, dim)``.
    """
    bounds = [(float(x_range[0]), float(x_range[1]))] * dim
    return {"x_res": latin_hypercube(num_res_points, bounds, key=key)}


def create_helmholtz_task(
    a1: float,
    a2: float,
    k: float,
    seed: int,
    dataset_path: str,
    dim: int = 2,
    a3: float = jnp.nan,
    modes: tuple[int, ...] = (1,),
    x_range: tuple[float, float] = (-1.0, 1.0),
    num_res_points: int = 10000,
    hidden_size: int = 30,
    num_layers: int = 4,
) -> Task:
    """Create a Helmholtz PINN task.

    Parameters
    ----------
    a1, a2 : float
        Wavenumbers in the first two spatial directions.
    k : float
        Helmholtz wavenumber ``k``.
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path stem used to save the generated collocation dataset.
    dim : int, optional
        Spatial dimensionality, 2 (default) or 3.
    a3 : float, optional
        Wavenumber in the third direction (used when ``dim == 3``).
    modes : tuple[int, ...], optional
        Integer Fourier-feature frequencies (paper Eq. 40). Use ``(1,)``
        for ``k = 1`` and a higher single mode for large ``k``; ``(1, 2)``
        for the 3-D ``k_max = 2`` setting.
    x_range : tuple[float, float], optional
        Per-axis box range, by default ``(-1.0, 1.0)``.
    num_res_points : int, optional
        Number of interior collocation points, by default 10000.
    hidden_size : int, optional
        Hidden-layer width of the MLP, by default 30.
    num_layers : int, optional
        Number of MLP layers, by default 4.

    Returns
    -------
    Task
        The created Helmholtz task.
    """
    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")
    key = jr.key(int(seed))
    model_key, dataset_key = jr.split(key)

    a = (a1, a2) if dim == 2 else (a1, a2, a3)
    length = float(x_range[1] - x_range[0])

    model, model_tags = fourier_mlp(
        in_dim=dim,
        out_size=1,
        modes=modes,
        length=length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=model_key,
    )

    loss_fn = partial(
        helmholtz_pde_loss,
        a=a,
        k=k,
        loss_fn_res=mse,
    )

    num_params = count_parameters(model)
    tag = {
        "task_name": "helmholtz",
        "loss_fn": "helmholtz_pde_loss",
        "dim": dim,
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "k": k,
        "modes": tuple(int(m) for m in modes),
        "dimensionality": num_params,
    }
    tag.update(model_tags)

    if not _path.exists():
        dataset = create_points_dataset(
            x_range=x_range,
            dim=dim,
            num_res_points=num_res_points,
            key=dataset_key,
        )
        save_dataset(dataset, _path)

    return Task(
        model=model,
        loss_fn=loss_fn,
        global_min=0.0,
        dataset=dataset_dict(_path, seed),
        tag=tag,
    )


class HelmholtzTaskSampler(Block):
    """f3dasm Block that generates Helmholtz optimization tasks.

    Produces the paper's 2-D and 3-D Helmholtz configurations across
    multiple hidden-size settings for use in the L2CO experiment
    pipeline.

    Parameters
    ----------
    num_res_points : int, optional
        Number of interior collocation points, by default 10000.
    """

    def __init__(self, num_res_points: int = 10000):
        self.num_res_points = num_res_points

        # (a1, a2, a3, k, dim, modes) from Section 4.1.
        self.configs = [
            dict(a1=1.0, a2=4.0, a3=float("nan"), k=1.0, dim=2, modes=(1,)),
            dict(a1=6.0, a2=6.0, a3=float("nan"), k=1.0, dim=2, modes=(1,)),
            dict(a1=6.0, a2=6.0, a3=float("nan"), k=10.0, dim=2, modes=(10,)),
            dict(a1=6.0, a2=6.0, a3=float("nan"), k=100.0, dim=2, modes=(10,)),
            dict(a1=10.0, a2=10.0, a3=float("nan"), k=1.0, dim=2, modes=(1,)),
            dict(a1=4.0, a2=4.0, a3=3.0, k=1.0, dim=3, modes=(1, 2)),
        ]

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        """Generate Helmholtz tasks for each hidden size and config.

        Parameters
        ----------
        data : ExperimentData
            Input experiment data with a ``hidden_size`` categorical
            parameter.
        seed : int
            Random seed for dataset generation.

        Returns
        -------
        ExperimentData
            Experiment data containing all Helmholtz task configurations.
        """
        hidden_size = data.domain.input_space["hidden_size"].categories

        task_list = []
        for cfg in self.configs:
            for h in hidden_size:
                t = deepcopy(cfg)
                t.update(
                    dict(
                        seed=seed,
                        # Collocation points depend on the spatial
                        # dimension, so 2-D and 3-D configs must not share
                        # a dataset file (other config fields leave the
                        # interior points unchanged).
                        dataset_path=f"./data/helmholtz_{cfg['dim']}d",
                        num_res_points=self.num_res_points,
                        hidden_size=h,
                    )
                )
                task_list.append(t)

        return build_task_experimentdata(
            task_list,
            (
                "a1",
                "a2",
                "a3",
                "k",
                "dim",
                "modes",
                "seed",
                "dataset_path",
                "num_res_points",
                "hidden_size",
            ),
            data.project_dir,
        )
