"""PINN task for the (2+1)-D viscous Burgers equation.

Builds the time-dependent two-dimensional viscous Burgers problem
(Jnini et al. 2026, Section 4.3, Eq. 43):

    u_t + (u**2 / 2)_x + (u**2 / 2)_y = nu * (u_xx + u_yy),

on ``(x, y) in [0, 1]^2`` and ``t in [0, 1]`` with viscosity ``nu``. The
problem has the closed-form solution

    u(x, y, t) = 1 / (1 + exp((x + y - t) / (2 nu))),

from which the initial- and boundary-condition targets are computed
(numerically stably via the logistic ``sigmoid``). The solution is
approximated by an MLP ``3 -> h -> ... -> 1`` with ``tanh`` activations
(10 hidden layers in the paper). The loss is the sum of mean-squared
errors over the PDE residual, the initial condition and the boundary
condition; derivatives are taken with ``jax.grad`` / ``jax.hessian``.

Public API
----------
create_viscous_burgers_task
    Build the viscous-Burgers PINN :class:`Task`.
ViscousBurgersTaskSampler
    ``f3dasm.Block`` generating the task across hidden-layer sizes.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Callable
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
from .models import mlp
from .pde import mse
from .pinn import latin_hypercube
from .task import Task, count_parameters, dataset_dict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def exact_solution(points: Array, nu: float) -> Array:
    """Evaluate the exact viscous-Burgers solution at ``points``.

    Uses the numerically stable logistic form
    ``sigmoid(-(x + y - t) / (2 nu))``.

    Parameters
    ----------
    points : Array
        Points ``(x, y, t)``.
    nu : float
        Viscosity coefficient.

    Returns
    -------
    Array
        Exact solution values of shape ``(n,)``.
    """
    x, y, t = points[:, 0], points[:, 1], points[:, 2]
    return jax.nn.sigmoid(-(x + y - t) / (2.0 * nu))


def viscous_burgers_loss(
    model: PyTree,
    x_res: Array,
    x_ic: Array,
    x_bc: Array,
    nu: float,
    loss_fn_res: Callable[[Array, Array], Array],
    loss_fn_ic: Callable[[Array, Array], Array],
    loss_fn_bc: Callable[[Array, Array], Array],
) -> Array:
    """Compute the physics-informed loss for the viscous Burgers equation.

    Parameters
    ----------
    model : PyTree
        Neural network approximating ``u(x, y, t)``.
    x_res : Array
        Interior collocation points.
    x_ic : Array
        Initial-condition points (with ``t = 0``).
    x_bc : Array
        Spatial-boundary points.
    nu : float
        Viscosity coefficient.
    loss_fn_res, loss_fn_ic, loss_fn_bc : Callable
        Loss functions for the residual, initial and boundary terms.

    Returns
    -------
    Array
        Combined residual + initial + boundary loss.
    """

    def u_point(p: Array) -> Array:
        """Scalar network output at a single point."""
        return model(p)[0]

    def residual(p: Array) -> Array:
        """PDE residual at a single point ``(x, y, t)``."""
        u = u_point(p)
        grad = jax.grad(u_point)(p)
        u_x, u_y, u_t = grad[0], grad[1], grad[2]
        hess = jax.hessian(u_point)(p)
        u_xx, u_yy = hess[0, 0], hess[1, 1]
        # f = g = u**2 / 2  ->  f_x = u u_x, g_y = u u_y
        return u_t + u * u_x + u * u_y - nu * (u_xx + u_yy)

    res = jax.vmap(residual)(x_res)
    loss_res = loss_fn_res(res, jnp.zeros_like(res))

    u_ic = jax.vmap(u_point)(x_ic)
    loss_ic = loss_fn_ic(u_ic, exact_solution(x_ic, nu))

    u_bc = jax.vmap(u_point)(x_bc)
    loss_bc = loss_fn_bc(u_bc, exact_solution(x_bc, nu))

    return loss_res + loss_ic + loss_bc


def _boundary_points(num_bc_points: int, *, key: Array) -> Array:
    """Sample points on the spatial boundary of the unit square.

    Points are split evenly across the four edges; each carries a random
    free coordinate and a random time in ``[0, 1]``.

    Parameters
    ----------
    num_bc_points : int
        Total number of boundary points (rounded down to a multiple of 4).
    key : Array
        Random key.

    Returns
    -------
    Array
        Boundary points ``(x, y, t)``.
    """
    per_edge = max(num_bc_points // 4, 1)
    keys = jr.split(key, 4)
    edges = []
    for i, (axis, value) in enumerate(
        [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0)]
    ):
        st = latin_hypercube(per_edge, [(0.0, 1.0), (0.0, 1.0)], key=keys[i])
        free, t = st[:, 0], st[:, 1]
        x = jnp.where(axis == 0, value, free)
        y = jnp.where(axis == 1, value, free)
        edges.append(jnp.stack([x, y, t], axis=-1))
    return jnp.concatenate(edges, axis=0)


def create_points_dataset(
    num_res_points: int,
    num_ic_points: int,
    num_bc_points: int,
    *,
    key: Array,
) -> dict[str, jnp.ndarray]:
    """Create collocation points for the viscous-Burgers task.

    Parameters
    ----------
    num_res_points : int
        Number of interior residual points.
    num_ic_points : int
        Number of initial-condition points.
    num_bc_points : int
        Number of boundary-condition points.
    key : Array
        Random key for reproducibility.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary with keys ``"x_res"``, ``"x_ic"`` and ``"x_bc"``.
    """
    res_key, ic_key, bc_key = jr.split(key, 3)
    x_res = latin_hypercube(
        num_res_points, [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], key=res_key
    )
    xy_ic = latin_hypercube(
        num_ic_points, [(0.0, 1.0), (0.0, 1.0)], key=ic_key
    )
    x_ic = jnp.concatenate([xy_ic, jnp.zeros((num_ic_points, 1))], axis=-1)
    x_bc = _boundary_points(num_bc_points, key=bc_key)
    return {"x_res": x_res, "x_ic": x_ic, "x_bc": x_bc}


def create_viscous_burgers_task(
    seed: int,
    dataset_path: str,
    nu: float = 0.004,
    num_res_points: int = 10000,
    num_ic_points: int = 1000,
    num_bc_points: int = 1000,
    hidden_size: int = 20,
    num_layers: int = 10,
) -> Task:
    """Create a (2+1)-D viscous Burgers PINN task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path stem used to save the generated collocation dataset.
    nu : float, optional
        Viscosity coefficient, by default 0.004.
    num_res_points : int, optional
        Number of interior residual points, by default 10000.
    num_ic_points : int, optional
        Number of initial-condition points, by default 1000.
    num_bc_points : int, optional
        Number of boundary-condition points, by default 1000.
    hidden_size : int, optional
        Hidden-layer width of the MLP, by default 20.
    num_layers : int, optional
        Number of MLP layers, by default 10.

    Returns
    -------
    Task
        The created viscous-Burgers task.
    """
    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")
    key = jr.key(int(seed))
    model_key, dataset_key = jr.split(key)

    model, model_tags = mlp(
        in_size=3,
        out_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=model_key,
    )

    loss_fn = partial(
        viscous_burgers_loss,
        nu=nu,
        loss_fn_res=mse,
        loss_fn_ic=mse,
        loss_fn_bc=mse,
    )

    num_params = count_parameters(model)
    tag = {
        "task_name": "viscous_burgers",
        "loss_fn": "viscous_burgers_loss",
        "nu": nu,
        "dimensionality": num_params,
    }
    tag.update(model_tags)

    if not _path.exists():
        dataset = create_points_dataset(
            num_res_points=num_res_points,
            num_ic_points=num_ic_points,
            num_bc_points=num_bc_points,
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


class ViscousBurgersTaskSampler(Block):
    """f3dasm Block that generates viscous-Burgers tasks.

    Parameters
    ----------
    nu : float, optional
        Viscosity coefficient, by default 0.004.
    num_res_points : int, optional
        Number of interior residual points, by default 10000.
    num_ic_points : int, optional
        Number of initial-condition points, by default 1000.
    num_bc_points : int, optional
        Number of boundary-condition points, by default 1000.
    """

    def __init__(
        self,
        nu: float = 0.004,
        num_res_points: int = 10000,
        num_ic_points: int = 1000,
        num_bc_points: int = 1000,
    ):
        self.nu = nu
        self.num_res_points = num_res_points
        self.num_ic_points = num_ic_points
        self.num_bc_points = num_bc_points

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        """Generate viscous-Burgers tasks for each hidden size.

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
            Experiment data containing the viscous-Burgers tasks.
        """
        hidden_size = data.domain.input_space["hidden_size"].categories

        task_list = [
            dict(
                seed=seed,
                dataset_path="./data/viscous_burgers",
                nu=self.nu,
                num_res_points=self.num_res_points,
                num_ic_points=self.num_ic_points,
                num_bc_points=self.num_bc_points,
                hidden_size=h,
            )
            for h in hidden_size
        ]

        return build_task_experimentdata(
            task_list,
            (
                "seed",
                "dataset_path",
                "nu",
                "num_res_points",
                "num_ic_points",
                "num_bc_points",
                "hidden_size",
            ),
            data.project_dir,
        )
