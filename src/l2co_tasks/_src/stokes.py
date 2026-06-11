"""PINN task for 2-D Stokes flow in a lid-driven triangular wedge.

Implements the Stokes problem of Jnini et al. 2026, Section 4.2
(Eq. 42), for a viscous lid-driven flow in a wedge:

    p_x - (u_xx + u_yy) = 0,
    p_y - (v_xx + v_yy) = 0,
    u_x + v_y           = 0,

on a triangular domain with apex at the origin, a driving lid along the
top edge (tangential velocity ``u_lid``, ``v = 0``) and no-slip walls.
Resolving the rapidly decaying Moffatt eddies near the apex demands an
accurate optimizer. There is no closed-form reference solution (the
paper uses a Nektar++ spectral-element reference, not reproduced here),
so the loss is the interior Stokes residual plus the lid/wall
boundary-condition mean-squared errors. The solution ``(u, v, p)`` is
approximated by an MLP ``2 -> h -> ... -> 3`` with ``tanh`` activations
(eight hidden layers of width 64 in the paper); derivatives are taken
with ``jax.grad`` / ``jax.hessian``.

Public API
----------
create_stokes_task
    Build the Stokes-wedge PINN :class:`Task`.
StokesTaskSampler
    ``f3dasm.Block`` generating the task across hidden-layer sizes.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from pathlib import Path

# Third-party
import equinox as eqx
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
from .pinn import sample_triangle
from .task import Task, count_parameters, dataset_dict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def wedge_vertices(angle_deg: float, height: float) -> Array:
    """Return the three vertices of the lid-driven wedge.

    The apex sits at the origin; the lid is the horizontal top edge at
    ``y = height`` and the full opening angle is ``angle_deg``.

    Parameters
    ----------
    angle_deg : float
        Full wedge opening angle in degrees.
    height : float
        Vertical extent (apex to lid).

    Returns
    -------
    Array
        Vertices ``[apex, top_left, top_right]``.
    """
    half_width = height * math.tan(math.radians(angle_deg) / 2.0)
    return jnp.array([[0.0, 0.0], [-half_width, height], [half_width, height]])


def stokes_loss(
    model: PyTree,
    x_res: Array,
    x_lid: Array,
    x_wall: Array,
    u_lid: float,
    loss_fn: Callable[[Array, Array], Array],
) -> Array:
    """Compute the physics-informed loss for the Stokes-wedge problem.

    Parameters
    ----------
    model : PyTree
        Network mapping ``(x, y)`` to ``(u, v, p)``.
    x_res : Array
        Interior collocation points.
    x_lid : Array
        Lid (top-edge) points.
    x_wall : Array
        Wall points (the two slanted edges).
    u_lid : float
        Prescribed tangential lid velocity.
    loss_fn : Callable
        Loss function applied to the residual and BC terms.

    Returns
    -------
    Array
        Combined residual + lid + wall loss.
    """

    def comp(q: Array, i: int) -> Array:
        """The ``i``-th output (u, v or p) at a single point."""
        return model(q)[i]

    def residual(q: Array) -> Array:
        """Stokes residual ``(r_x, r_y, r_div)`` at a single point."""
        g_u = jax.grad(comp, argnums=0)(q, 0)
        g_v = jax.grad(comp, argnums=0)(q, 1)
        g_p = jax.grad(comp, argnums=0)(q, 2)
        h_u = jax.hessian(comp, argnums=0)(q, 0)
        h_v = jax.hessian(comp, argnums=0)(q, 1)

        u_x = g_u[0]
        v_y = g_v[1]
        p_x, p_y = g_p[0], g_p[1]
        lap_u = h_u[0, 0] + h_u[1, 1]
        lap_v = h_v[0, 0] + h_v[1, 1]

        r_x = p_x - lap_u
        r_y = p_y - lap_v
        r_div = u_x + v_y
        return jnp.array([r_x, r_y, r_div])

    res = jax.vmap(residual)(x_res)
    loss_res = loss_fn(res, jnp.zeros_like(res))

    uv_lid = eqx.filter_vmap(lambda q: model(q)[:2])(x_lid)
    target_lid = jnp.broadcast_to(jnp.array([u_lid, 0.0]), uv_lid.shape)
    loss_lid = loss_fn(uv_lid, target_lid)

    uv_wall = eqx.filter_vmap(lambda q: model(q)[:2])(x_wall)
    loss_wall = loss_fn(uv_wall, jnp.zeros_like(uv_wall))

    return loss_res + loss_lid + loss_wall


def create_points_dataset(
    angle_deg: float,
    height: float,
    num_res_points: int,
    num_lid_points: int,
    num_wall_points: int,
    *,
    key: Array,
) -> dict[str, jnp.ndarray]:
    """Create collocation points for the Stokes-wedge task.

    Parameters
    ----------
    angle_deg : float
        Full wedge opening angle in degrees.
    height : float
        Vertical extent of the wedge.
    num_res_points : int
        Number of interior residual points (uniform over the triangle).
    num_lid_points : int
        Number of lid (top-edge) points.
    num_wall_points : int
        Number of wall points (split between the two slanted edges).
    key : Array
        Random key for reproducibility.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary with keys ``"x_res"``, ``"x_lid"`` and ``"x_wall"``.
    """
    v = wedge_vertices(angle_deg, height)
    res_key, lid_key, wall_key = jr.split(key, 3)

    x_res = sample_triangle(num_res_points, v, key=res_key)

    s_lid = jr.uniform(lid_key, (num_lid_points,))
    x_lid = v[1] + s_lid[:, None] * (v[2] - v[1])

    half = max(num_wall_points // 2, 1)
    wl_key, wr_key = jr.split(wall_key)
    s_l = jr.uniform(wl_key, (half,))
    s_r = jr.uniform(wr_key, (half,))
    left = v[0] + s_l[:, None] * (v[1] - v[0])
    right = v[0] + s_r[:, None] * (v[2] - v[0])
    x_wall = jnp.concatenate([left, right], axis=0)

    return {"x_res": x_res, "x_lid": x_lid, "x_wall": x_wall}


def create_stokes_task(
    seed: int,
    dataset_path: str,
    angle_deg: float = 25.53,
    height: float = 1.0,
    u_lid: float = 1.0,
    num_res_points: int = 120000,
    num_lid_points: int = 1000,
    num_wall_points: int = 1000,
    hidden_size: int = 64,
    num_layers: int = 8,
) -> Task:
    """Create a 2-D Stokes-wedge PINN task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path stem used to save the generated collocation dataset.
    angle_deg : float, optional
        Full wedge opening angle in degrees, by default 25.53.
    height : float, optional
        Vertical extent of the wedge, by default 1.0.
    u_lid : float, optional
        Prescribed tangential lid velocity, by default 1.0.
    num_res_points : int, optional
        Number of interior residual points, by default 120000.
    num_lid_points : int, optional
        Number of lid points, by default 1000.
    num_wall_points : int, optional
        Number of wall points, by default 1000.
    hidden_size : int, optional
        Hidden-layer width of the MLP, by default 64.
    num_layers : int, optional
        Number of MLP layers, by default 8.

    Returns
    -------
    Task
        The created Stokes-wedge task.
    """
    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")
    key = jr.key(int(seed))
    model_key, dataset_key = jr.split(key)

    model, model_tags = mlp(
        in_size=2,
        out_size=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=model_key,
    )

    loss_fn = partial(stokes_loss, u_lid=u_lid, loss_fn=mse)

    num_params = count_parameters(model)
    tag = {
        "task_name": "stokes",
        "loss_fn": "stokes_loss",
        "angle_deg": angle_deg,
        "height": height,
        "u_lid": u_lid,
        "dimensionality": num_params,
    }
    tag.update(model_tags)

    if not _path.exists():
        dataset = create_points_dataset(
            angle_deg=angle_deg,
            height=height,
            num_res_points=num_res_points,
            num_lid_points=num_lid_points,
            num_wall_points=num_wall_points,
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


class StokesTaskSampler(Block):
    """f3dasm Block that generates Stokes-wedge tasks.

    Parameters
    ----------
    angle_deg : float, optional
        Full wedge opening angle in degrees, by default 25.53.
    height : float, optional
        Vertical extent of the wedge, by default 1.0.
    u_lid : float, optional
        Prescribed tangential lid velocity, by default 1.0.
    num_res_points : int, optional
        Number of interior residual points, by default 120000.
    num_lid_points : int, optional
        Number of lid points, by default 1000.
    num_wall_points : int, optional
        Number of wall points, by default 1000.
    """

    def __init__(
        self,
        angle_deg: float = 25.53,
        height: float = 1.0,
        u_lid: float = 1.0,
        num_res_points: int = 120000,
        num_lid_points: int = 1000,
        num_wall_points: int = 1000,
    ):
        self.angle_deg = angle_deg
        self.height = height
        self.u_lid = u_lid
        self.num_res_points = num_res_points
        self.num_lid_points = num_lid_points
        self.num_wall_points = num_wall_points

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        """Generate Stokes-wedge tasks for each hidden size.

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
            Experiment data containing the Stokes-wedge tasks.
        """
        hidden_size = data.domain.input_space["hidden_size"].categories

        task_list = [
            dict(
                seed=seed,
                dataset_path="./data/stokes",
                angle_deg=self.angle_deg,
                height=self.height,
                u_lid=self.u_lid,
                num_res_points=self.num_res_points,
                num_lid_points=self.num_lid_points,
                num_wall_points=self.num_wall_points,
                hidden_size=h,
            )
            for h in hidden_size
        ]

        return build_task_experimentdata(
            task_list,
            (
                "seed",
                "dataset_path",
                "angle_deg",
                "height",
                "u_lid",
                "num_res_points",
                "num_lid_points",
                "num_wall_points",
                "hidden_size",
            ),
            data.project_dir,
        )
