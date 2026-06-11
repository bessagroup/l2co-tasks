"""PINN task for the (1+1)-D compressible Euler equations (Sod shock tube).

Implements the two-stage methodology of Jnini et al. 2026,
Section 4.5.2, selected via the ``stage`` argument:

``stage="viscous"`` (warm-up, Eq. 70-73)
    A viscous regularisation of the Euler equations in primitive
    variables ``(rho, u, p)`` with an adaptive-viscosity network output
    ``nu``; the loss adds a soft penalty ``lam_nu * mean(nu)`` driving
    ``nu -> 0``.

``stage="inviscid"`` (Eq. 74-77)
    The inviscid conservative residual ``U_t + d/dx F_HLLC = 0`` where
    ``F_HLLC`` is the Harten-Lax-van Leer-Contact numerical flux
    (Toro, Eq. 75-77) built from left/right states reconstructed at
    ``x -/+ h``; its spatial derivative is taken by autodiff. The caller
    loads the converged weights of the viscous stage into the inviscid
    task before continuing training.

The conserved variables are ``U = [rho, rho u, E]`` with
``E = rho u**2 / 2 + p / (gamma - 1)`` and ``gamma = 1.4``. The network
maps ``(x, t)`` to four outputs ``(rho, u, p, nu)`` (the viscosity head
passed through ``softplus``). Densities and pressures are floored at a
small epsilon inside the flux to keep the loss finite for any weights.

Public API
----------
create_euler_task
    Build the Euler PINN :class:`Task` for a given ``stage``.
EulerTaskSampler
    ``f3dasm.Block`` generating both stages across hidden-layer sizes.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

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
from .pinn import latin_hypercube
from .task import Task, count_parameters, dataset_dict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

GAMMA = 1.4
_EPS = 1e-6

# Sod shock-tube left/right primitive states (rho, u, p), Eq. 67.
_LEFT_STATE = (1.0, 0.0, 1.0)
_RIGHT_STATE = (0.125, 0.0, 0.1)


def primitive(model: PyTree, p: Array) -> tuple[Array, Array, Array]:
    """Return the primitive variables ``(rho, u, pressure)`` at ``p``.

    Parameters
    ----------
    model : PyTree
        Network mapping ``(x, t)`` to ``(rho, u, p, nu)``.
    p : Array
        Input point ``(x, t)``.

    Returns
    -------
    tuple[Array, Array, Array]
        ``(rho, u, pressure)`` (raw network outputs).
    """
    out = model(p)
    return out[0], out[1], out[2]


def conserved(rho: Array, u: Array, pressure: Array) -> Array:
    """Convert primitive variables to the conserved vector ``[rho, rho u, E]``.

    Parameters
    ----------
    rho, u, pressure : Array
        Primitive variables.

    Returns
    -------
    Array
        Conserved state of shape ``(3,)``.
    """
    e = rho * u**2 / 2.0 + pressure / (GAMMA - 1.0)
    return jnp.array([rho, rho * u, e])


def physical_flux(rho: Array, u: Array, pressure: Array) -> Array:
    """Physical Euler flux ``F(U)`` from primitive variables.

    Parameters
    ----------
    rho, u, pressure : Array
        Primitive variables.

    Returns
    -------
    Array
        Flux vector of shape ``(3,)``.
    """
    e = rho * u**2 / 2.0 + pressure / (GAMMA - 1.0)
    return jnp.array([rho * u, rho * u**2 + pressure, u * (e + pressure)])


def hllc_flux(
    wl: tuple[Array, Array, Array],
    wr: tuple[Array, Array, Array],
) -> Array:
    """Harten-Lax-van Leer-Contact numerical flux (Toro, Eq. 75-77).

    Densities and pressures are floored at a small epsilon so the sound
    speeds and divisions stay finite for arbitrary inputs.

    Parameters
    ----------
    wl, wr : tuple[Array, Array, Array]
        Left and right primitive states ``(rho, u, pressure)``.

    Returns
    -------
    Array
        The HLLC flux vector of shape ``(3,)``.
    """
    rho_l, u_l, p_l = wl
    rho_r, u_r, p_r = wr
    rho_l = jnp.maximum(rho_l, _EPS)
    rho_r = jnp.maximum(rho_r, _EPS)
    p_l = jnp.maximum(p_l, _EPS)
    p_r = jnp.maximum(p_r, _EPS)

    a_l = jnp.sqrt(GAMMA * p_l / rho_l)
    a_r = jnp.sqrt(GAMMA * p_r / rho_r)

    # Davis wave-speed estimates.
    s_l = jnp.minimum(u_l - a_l, u_r - a_r)
    s_r = jnp.maximum(u_l + a_l, u_r + a_r)

    num = p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)
    den = rho_l * (s_l - u_l) - rho_r * (s_r - u_r)
    s_star = num / (den + _EPS * jnp.sign(den) + _EPS)

    u_l_vec = conserved(rho_l, u_l, p_l)
    u_r_vec = conserved(rho_r, u_r, p_r)
    f_l = physical_flux(rho_l, u_l, p_l)
    f_r = physical_flux(rho_r, u_r, p_r)

    def star_state(rho, u, pressure, e_state, s_k):
        """Intermediate (star) conserved state for side ``K`` (Eq. 76)."""
        factor = rho * (s_k - u) / (s_k - s_star + _EPS)
        energy_term = e_state / rho + (s_star - u) * (
            s_star + pressure / (rho * (s_k - u) + _EPS)
        )
        return factor * jnp.array([1.0, s_star, energy_term])

    u_star_l = star_state(rho_l, u_l, p_l, u_l_vec[2], s_l)
    u_star_r = star_state(rho_r, u_r, p_r, u_r_vec[2], s_r)
    f_star_l = f_l + s_l * (u_star_l - u_l_vec)
    f_star_r = f_r + s_r * (u_star_r - u_r_vec)

    flux = jnp.where(
        s_l >= 0.0,
        f_l,
        jnp.where(
            s_star >= 0.0,
            f_star_l,
            jnp.where(s_r >= 0.0, f_star_r, f_r),
        ),
    )
    return flux


def _sod_target(x: Array, x0: float) -> Array:
    """Sod shock-tube primitive target ``(rho, u, p)`` at coordinates ``x``."""
    left = jnp.asarray(_LEFT_STATE)
    right = jnp.asarray(_RIGHT_STATE)
    cond = (x < x0)[:, None]
    return jnp.where(cond, left, right)


def euler_viscous_loss(
    model: PyTree,
    x_res: Array,
    x_ic: Array,
    x_bc: Array,
    x0: float,
    lam_ic: float,
    lam_bc: float,
    lam_f: float,
    lam_nu: float,
    loss_fn: Callable[[Array, Array], Array],
) -> Array:
    """Viscous warm-up loss for the Euler equations (Eq. 70-73).

    Parameters
    ----------
    model : PyTree
        Network mapping ``(x, t)`` to ``(rho, u, p, nu)``.
    x_res : Array
        Interior collocation points.
    x_ic : Array
        Initial-condition points (at ``t = 0``).
    x_bc : Array
        Boundary points (at the domain ends).
    x0 : float
        Location of the initial discontinuity.
    lam_ic, lam_bc, lam_f, lam_nu : float
        Loss-term weights (the last drives the viscosity to zero).
    loss_fn : Callable
        Loss function applied to the equality terms.

    Returns
    -------
    Array
        Combined weighted loss.
    """

    def nu_point(q: Array) -> Array:
        """Non-negative adaptive viscosity at a single point."""
        return jax.nn.softplus(model(q)[3])

    def fields(q: Array) -> tuple[Array, Array, Array]:
        """Primitive scalar fields at a single point."""
        return primitive(model, q)

    def residual(q: Array) -> Array:
        """Viscous primitive residual ``(R0, R1, R2)`` at a single point."""
        rho, u, pressure = fields(q)
        nu = nu_point(q)

        g_rho = jax.grad(lambda z: primitive(model, z)[0])(q)
        g_u = jax.grad(lambda z: primitive(model, z)[1])(q)
        g_p = jax.grad(lambda z: primitive(model, z)[2])(q)
        h_rho = jax.hessian(lambda z: primitive(model, z)[0])(q)
        h_u = jax.hessian(lambda z: primitive(model, z)[1])(q)
        h_p = jax.hessian(lambda z: primitive(model, z)[2])(q)

        rho_x, rho_t = g_rho[0], g_rho[1]
        u_x, u_t = g_u[0], g_u[1]
        p_x, p_t = g_p[0], g_p[1]
        rho_xx, u_xx, p_xx = h_rho[0, 0], h_u[0, 0], h_p[0, 0]

        r0 = rho_t + u * rho_x + rho * u_x - nu * rho_xx
        r1 = rho * u_t + rho * u * u_x + p_x - nu * (u_xx + 2.0 * rho_x * u_x)
        r2 = (
            p_t
            + u * p_x
            + GAMMA * pressure * u_x
            - nu * (p_xx + rho * (GAMMA - 1.0) * u_x**2)
        )
        return jnp.array([r0, r1, r2])

    res = jax.vmap(residual)(x_res)
    loss_f = loss_fn(res, jnp.zeros_like(res))

    w_ic = eqx.filter_vmap(lambda q: jnp.stack(primitive(model, q)))(x_ic)
    loss_ic = loss_fn(w_ic, _sod_target(x_ic[:, 0], x0))

    w_bc = eqx.filter_vmap(lambda q: jnp.stack(primitive(model, q)))(x_bc)
    loss_bc = loss_fn(w_bc, _sod_target(x_bc[:, 0], x0))

    nu_mean = jnp.mean(eqx.filter_vmap(nu_point)(x_res))

    return (
        lam_ic * loss_ic + lam_bc * loss_bc + lam_f * loss_f + lam_nu * nu_mean
    )


def euler_inviscid_loss(
    model: PyTree,
    x_res: Array,
    x_ic: Array,
    x_bc: Array,
    x0: float,
    h: float,
    lam_ic: float,
    lam_bc: float,
    lam_f: float,
    loss_fn: Callable[[Array, Array], Array],
) -> Array:
    """Inviscid HLLC conservative-residual loss for Euler (Eq. 74-77).

    The residual ``U_t + d/dx F_HLLC`` is evaluated pointwise, with
    ``F_HLLC`` built from left/right states reconstructed at ``x -/+ h``
    and differentiated in ``x`` by autodiff.

    Parameters
    ----------
    model : PyTree
        Network mapping ``(x, t)`` to ``(rho, u, p, nu)``.
    x_res : Array
        Interior collocation points.
    x_ic : Array
        Initial-condition points (at ``t = 0``).
    x_bc : Array
        Boundary points (at the domain ends).
    x0 : float
        Location of the initial discontinuity.
    h : float
        Half-width of the left/right reconstruction stencil.
    lam_ic, lam_bc, lam_f : float
        Loss-term weights.
    loss_fn : Callable
        Loss function applied to the equality terms.

    Returns
    -------
    Array
        Combined weighted loss.
    """
    ex = jnp.array([1.0, 0.0])

    def u_state(q: Array) -> Array:
        """Conserved state at a single point."""
        rho, u, pressure = primitive(model, q)
        return conserved(rho, u, pressure)

    def flux_hllc(q: Array) -> Array:
        """HLLC flux from states reconstructed at ``q -/+ h * ex``."""
        wl = primitive(model, q - h * ex)
        wr = primitive(model, q + h * ex)
        return hllc_flux(wl, wr)

    def residual(q: Array) -> Array:
        """Conservative HLLC residual ``U_t + dF/dx`` at a single point."""
        u_t = jax.jacobian(u_state)(q)[:, 1]
        f_x = jax.jacobian(flux_hllc)(q)[:, 0]
        return u_t + f_x

    res = jax.vmap(residual)(x_res)
    loss_f = loss_fn(res, jnp.zeros_like(res))

    w_ic = eqx.filter_vmap(lambda q: jnp.stack(primitive(model, q)))(x_ic)
    loss_ic = loss_fn(w_ic, _sod_target(x_ic[:, 0], x0))

    w_bc = eqx.filter_vmap(lambda q: jnp.stack(primitive(model, q)))(x_bc)
    loss_bc = loss_fn(w_bc, _sod_target(x_bc[:, 0], x0))

    return lam_ic * loss_ic + lam_bc * loss_bc + lam_f * loss_f


def create_points_dataset(
    x_range: tuple[float, float],
    t_range: tuple[float, float],
    num_res_points: int,
    num_ic_points: int,
    num_bc_points: int,
    *,
    key: Array,
) -> dict[str, jnp.ndarray]:
    """Create collocation points for an Euler task.

    Parameters
    ----------
    x_range, t_range : tuple[float, float]
        Spatial and temporal domain ranges.
    num_res_points : int
        Number of interior residual points.
    num_ic_points : int
        Number of initial-condition points (at ``t = t_range[0]``).
    num_bc_points : int
        Number of boundary points (split between the two domain ends).
    key : Array
        Random key for reproducibility.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary with keys ``"x_res"``, ``"x_ic"`` and ``"x_bc"``.
    """
    res_key, ic_key, bc_key = jr.split(key, 3)
    x_res = latin_hypercube(
        num_res_points, [tuple(x_range), tuple(t_range)], key=res_key
    )
    x_coord = latin_hypercube(num_ic_points, [tuple(x_range)], key=ic_key)
    x_ic = jnp.concatenate(
        [x_coord, jnp.full((num_ic_points, 1), t_range[0])], axis=-1
    )
    half = max(num_bc_points // 2, 1)
    t_bc = latin_hypercube(half, [tuple(t_range)], key=bc_key)
    left = jnp.concatenate([jnp.full((half, 1), x_range[0]), t_bc], axis=-1)
    right = jnp.concatenate([jnp.full((half, 1), x_range[1]), t_bc], axis=-1)
    x_bc = jnp.concatenate([left, right], axis=0)
    return {"x_res": x_res, "x_ic": x_ic, "x_bc": x_bc}


def create_euler_task(
    seed: int,
    dataset_path: str,
    stage: str = "inviscid",
    x_range: tuple[float, float] = (-2.5, 2.5),
    t_range: tuple[float, float] = (0.0, 1.0),
    x0: float = 0.0,
    h: float = 0.0175,
    num_res_points: int = 15000,
    num_ic_points: int = 1000,
    num_bc_points: int = 1000,
    hidden_size: int = 20,
    num_layers: int = 6,
    lam_ic: float = 1.0,
    lam_bc: float = 1.0,
    lam_f: float = 1.0,
    lam_nu: float = 1.0,
) -> Task:
    """Create a (1+1)-D Euler (Sod shock tube) PINN task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path stem used to save the generated collocation dataset.
    stage : str, optional
        Either ``"viscous"`` (warm-up) or ``"inviscid"`` (HLLC), by
        default ``"inviscid"``.
    x_range, t_range : tuple[float, float], optional
        Spatial / temporal domain, by default ``(-2.5, 2.5)`` and
        ``(0.0, 1.0)``.
    x0 : float, optional
        Location of the initial discontinuity, by default 0.0.
    h : float, optional
        Half-width of the HLLC reconstruction stencil, by default 0.0175.
    num_res_points : int, optional
        Number of interior residual points, by default 15000.
    num_ic_points : int, optional
        Number of initial-condition points, by default 1000.
    num_bc_points : int, optional
        Number of boundary-condition points, by default 1000.
    hidden_size : int, optional
        Hidden-layer width of the MLP, by default 20.
    num_layers : int, optional
        Number of MLP layers, by default 6.
    lam_ic, lam_bc, lam_f, lam_nu : float, optional
        Loss-term weights (``lam_nu`` used only in the viscous stage).

    Returns
    -------
    Task
        The created Euler task for the requested stage.
    """
    if stage not in ("viscous", "inviscid"):
        raise ValueError(f"stage must be 'viscous' or 'inviscid', got {stage}")

    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")
    key = jr.key(int(seed))
    model_key, dataset_key = jr.split(key)

    model, model_tags = mlp(
        in_size=2,
        out_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=model_key,
    )

    if stage == "viscous":
        loss_fn = partial(
            euler_viscous_loss,
            x0=x0,
            lam_ic=lam_ic,
            lam_bc=lam_bc,
            lam_f=lam_f,
            lam_nu=lam_nu,
            loss_fn=mse,
        )
        loss_name = "euler_viscous_loss"
    else:
        loss_fn = partial(
            euler_inviscid_loss,
            x0=x0,
            h=h,
            lam_ic=lam_ic,
            lam_bc=lam_bc,
            lam_f=lam_f,
            loss_fn=mse,
        )
        loss_name = "euler_inviscid_loss"

    num_params = count_parameters(model)
    tag = {
        "task_name": "euler",
        "loss_fn": loss_name,
        "stage": stage,
        "x0": x0,
        "h": h,
        "dimensionality": num_params,
    }
    tag.update(model_tags)

    if not _path.exists():
        dataset = create_points_dataset(
            x_range=x_range,
            t_range=t_range,
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


class EulerTaskSampler(Block):
    """f3dasm Block that generates Euler (Sod shock tube) tasks.

    Emits both the viscous warm-up and inviscid HLLC stages for each
    hidden size.

    Parameters
    ----------
    x_range, t_range : tuple[float, float], optional
        Spatial / temporal domain.
    num_res_points : int, optional
        Number of interior residual points, by default 15000.
    num_ic_points : int, optional
        Number of initial-condition points, by default 1000.
    num_bc_points : int, optional
        Number of boundary-condition points, by default 1000.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-2.5, 2.5),
        t_range: tuple[float, float] = (0.0, 1.0),
        num_res_points: int = 15000,
        num_ic_points: int = 1000,
        num_bc_points: int = 1000,
    ):
        self.x_range = x_range
        self.t_range = t_range
        self.num_res_points = num_res_points
        self.num_ic_points = num_ic_points
        self.num_bc_points = num_bc_points

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        """Generate Euler tasks (both stages) for each hidden size.

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
            Experiment data containing the Euler tasks.
        """
        hidden_size = data.domain.input_space["hidden_size"].categories

        task_list = [
            dict(
                seed=seed,
                dataset_path="./data/euler",
                stage=stage,
                x_range=self.x_range,
                t_range=self.t_range,
                num_res_points=self.num_res_points,
                num_ic_points=self.num_ic_points,
                num_bc_points=self.num_bc_points,
                hidden_size=h,
            )
            for stage in ("viscous", "inviscid")
            for h in hidden_size
        ]

        return build_task_experimentdata(
            task_list,
            (
                "seed",
                "dataset_path",
                "stage",
                "x_range",
                "t_range",
                "num_res_points",
                "num_ic_points",
                "num_bc_points",
                "hidden_size",
            ),
            data.project_dir,
        )
