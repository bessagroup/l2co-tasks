"""PINN task for the inviscid Burgers equation (entropy / flux relaxation).

Implements the entropy-consistent flux-relaxation formulation of
Jnini et al. 2026, Section 4.4.2 (Eq. 51-52) for

    u_t + (u**2 / 2)_x = 0,   x in [-1, 1],  t in [0, 1],

with initial condition ``u(x, 0) = -sin(pi x)``. Two networks are
trained jointly: a solution network ``u(x, t)`` and a flux network
``F(x, t)`` (bundled in a
:class:`~l2co_tasks._src.models.MultiNet`). The conservative residual is
imposed on the flux network while an algebraic constraint ties the flux
to ``u**2`` and a Tadmor entropy-inequality penalty selects the
physically admissible weak solution.

Total loss (Eq. 52)::

    L = lam_ic * L_ic + lam_bc * L_bc + lam_f * L_f
        + lam_I * L_I  + lam_F * L_F

with ``L_F = MSE(F, u**2)``, ``L_f = MSE(u_t + F_x, 0)``,
``L_I = mean(max(0, eta_t + psi_x))`` for the entropy pair
``eta = u**2 / 2``, ``psi = u**3 / 3``, and the usual initial/boundary
mean-squared-error terms.

Public API
----------
create_inviscid_burgers_task
    Build the inviscid-Burgers PINN :class:`Task`.
InviscidBurgersTaskSampler
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
from .models import mlp, multinet
from .pde import mse
from .pinn import latin_hypercube
from .task import Task, count_parameters, dataset_dict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def inviscid_burgers_loss(
    model: PyTree,
    x_res: Array,
    x_ic: Array,
    x_bc: Array,
    lam_ic: float,
    lam_bc: float,
    lam_f: float,
    lam_i: float,
    lam_flux: float,
    loss_fn: Callable[[Array, Array], Array],
) -> Array:
    """Compute the entropy/flux-relaxation loss for inviscid Burgers.

    Parameters
    ----------
    model : PyTree
        :class:`MultiNet` with sub-networks ``"u"`` (solution) and
        ``"flux"`` (flux ``F``).
    x_res : Array
        Interior collocation points ``(x, t)``.
    x_ic : Array
        Initial-condition points (with ``t = 0``).
    x_bc : Array
        Boundary points (at ``x = +/-1``).
    lam_ic, lam_bc, lam_f, lam_i, lam_flux : float
        Weights of the initial, boundary, flux-residual, entropy and
        flux-constraint loss terms.
    loss_fn : Callable
        Loss function applied to the equality terms (mean-squared error).

    Returns
    -------
    Array
        Combined weighted loss.
    """
    u_net = model["u"]
    f_net = model["flux"]

    def u_point(p: Array) -> Array:
        """Scalar ``u`` at a single point."""
        return u_net(p)[0]

    def f_point(p: Array) -> Array:
        """Scalar flux ``F`` at a single point."""
        return f_net(p)[0]

    def residual_terms(p: Array) -> Array:
        """Return ``(flux_residual, entropy_residual, flux_gap)`` at ``p``."""
        u = u_point(p)
        u_grad = jax.grad(u_point)(p)
        u_x, u_t = u_grad[0], u_grad[1]
        f_grad = jax.grad(f_point)(p)
        f_x = f_grad[0]
        flux_res = u_t + f_x
        # eta = u^2/2 -> eta_t = u u_t ; psi = u^3/3 -> psi_x = u^2 u_x
        entropy_res = u * u_t + u**2 * u_x
        flux_gap = f_point(p) - u**2
        return jnp.array([flux_res, entropy_res, flux_gap])

    terms = jax.vmap(residual_terms)(x_res)
    flux_res, entropy_res, flux_gap = terms[:, 0], terms[:, 1], terms[:, 2]

    loss_f = loss_fn(flux_res, jnp.zeros_like(flux_res))
    loss_i = jnp.mean(jnp.maximum(0.0, entropy_res) ** 2)
    loss_flux = loss_fn(flux_gap, jnp.zeros_like(flux_gap))

    u_ic = jax.vmap(u_point)(x_ic)
    loss_ic = loss_fn(u_ic, -jnp.sin(jnp.pi * x_ic[:, 0]))

    u_bc = jax.vmap(u_point)(x_bc)
    loss_bc = loss_fn(u_bc, jnp.zeros_like(u_bc))

    return (
        lam_ic * loss_ic
        + lam_bc * loss_bc
        + lam_f * loss_f
        + lam_i * loss_i
        + lam_flux * loss_flux
    )


def create_points_dataset(
    num_res_points: int,
    num_ic_points: int,
    num_bc_points: int,
    *,
    key: Array,
) -> dict[str, jnp.ndarray]:
    """Create collocation points for the inviscid-Burgers task.

    Parameters
    ----------
    num_res_points : int
        Number of interior residual points (LHS over ``[-1,1]x[0,1]``).
    num_ic_points : int
        Number of initial-condition points (at ``t = 0``).
    num_bc_points : int
        Number of boundary points (split between ``x = -1`` and ``x = 1``).
    key : Array
        Random key for reproducibility.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary with keys ``"x_res"``, ``"x_ic"`` and ``"x_bc"``.
    """
    res_key, ic_key, bc_key = jr.split(key, 3)
    x_res = latin_hypercube(
        num_res_points, [(-1.0, 1.0), (0.0, 1.0)], key=res_key
    )
    x_coord = latin_hypercube(num_ic_points, [(-1.0, 1.0)], key=ic_key)
    x_ic = jnp.concatenate([x_coord, jnp.zeros((num_ic_points, 1))], axis=-1)

    half = max(num_bc_points // 2, 1)
    t_bc = latin_hypercube(half, [(0.0, 1.0)], key=bc_key)
    left = jnp.concatenate([jnp.full((half, 1), -1.0), t_bc], axis=-1)
    right = jnp.concatenate([jnp.full((half, 1), 1.0), t_bc], axis=-1)
    x_bc = jnp.concatenate([left, right], axis=0)
    return {"x_res": x_res, "x_ic": x_ic, "x_bc": x_bc}


def create_inviscid_burgers_task(
    seed: int,
    dataset_path: str,
    num_res_points: int = 50000,
    num_ic_points: int = 300,
    num_bc_points: int = 300,
    hidden_size: int = 50,
    num_layers: int = 4,
    lam_ic: float = 1.0,
    lam_bc: float = 1.0,
    lam_f: float = 1.0,
    lam_i: float = 1.0,
    lam_flux: float = 1.0,
) -> Task:
    """Create an inviscid-Burgers PINN task (entropy/flux relaxation).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path stem used to save the generated collocation dataset.
    num_res_points : int, optional
        Number of interior residual points, by default 50000.
    num_ic_points : int, optional
        Number of initial-condition points, by default 300.
    num_bc_points : int, optional
        Number of boundary-condition points, by default 300.
    hidden_size : int, optional
        Hidden-layer width of each MLP, by default 50.
    num_layers : int, optional
        Number of layers in each MLP, by default 4.
    lam_ic, lam_bc, lam_f, lam_i, lam_flux : float, optional
        Loss-term weights, all 1.0 by default.

    Returns
    -------
    Task
        The created inviscid-Burgers task.
    """
    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")
    key = jr.key(int(seed))
    u_key, f_key, dataset_key = jr.split(key, 3)

    u_net, net_tags = mlp(
        in_size=2,
        out_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=u_key,
    )
    f_net, _ = mlp(
        in_size=2,
        out_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=f_key,
    )
    model, _ = multinet({"u": u_net, "flux": f_net})

    loss_fn = partial(
        inviscid_burgers_loss,
        lam_ic=lam_ic,
        lam_bc=lam_bc,
        lam_f=lam_f,
        lam_i=lam_i,
        lam_flux=lam_flux,
        loss_fn=mse,
    )

    num_params = count_parameters(model)
    tag = {
        "task_name": "inviscid_burgers",
        "loss_fn": "inviscid_burgers_loss",
        "lam_ic": lam_ic,
        "lam_bc": lam_bc,
        "lam_f": lam_f,
        "lam_i": lam_i,
        "lam_flux": lam_flux,
        "dimensionality": num_params,
    }
    # Both sub-networks share the same architecture; record it once.
    tag.update(net_tags)

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


class InviscidBurgersTaskSampler(Block):
    """f3dasm Block that generates inviscid-Burgers tasks.

    Parameters
    ----------
    num_res_points : int, optional
        Number of interior residual points, by default 50000.
    num_ic_points : int, optional
        Number of initial-condition points, by default 300.
    num_bc_points : int, optional
        Number of boundary-condition points, by default 300.
    """

    def __init__(
        self,
        num_res_points: int = 50000,
        num_ic_points: int = 300,
        num_bc_points: int = 300,
    ):
        self.num_res_points = num_res_points
        self.num_ic_points = num_ic_points
        self.num_bc_points = num_bc_points

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        """Generate inviscid-Burgers tasks for each hidden size.

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
            Experiment data containing the inviscid-Burgers tasks.
        """
        hidden_size = data.domain.input_space["hidden_size"].categories

        task_list = [
            dict(
                seed=seed,
                dataset_path="./data/inviscid_burgers",
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
                "num_res_points",
                "num_ic_points",
                "num_bc_points",
                "hidden_size",
            ),
            data.project_dir,
        )
