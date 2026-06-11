"""PINN task for the stiff PK-PD (pharmacokinetic-pharmacodynamic) ODE.

Builds the coupled paclitaxel chemotherapy model of Jnini et al. 2026,
Section 4.6 (Eq. 78-86), describing a cytotoxic agent acting on tumour
growth over a treatment window ``t in [0, 17]`` days.

Pharmacokinetics (analytic, Eq. 80-81)
    A two-compartment IV-bolus model gives a closed-form plasma
    concentration ``c(t)`` as a sum of fast/slow exponentials triggered
    at each dose time (a prior dose before ``t = 0`` leaving residual
    drug, plus an injection at day 2). The decay rates ``alpha >> beta``
    are the eigenvalues of the compartment system and are derived from
    the rate constants ``k10, k12, k21``.

Pharmacodynamics (residuals, Eq. 82-85)
    A four-state transit model ``x1..x4`` with a generalised-logistic
    (Hill, shape ``Psi``) growth term and a drug-kill term proportional
    to ``c(t)``. The total tumour burden is ``omega = x1 + x2 + x3 + x4``.

The network maps time ``t`` to the four states; a ``softplus`` keeps the
predicted states non-negative (so the Hill term stays finite for any
weights). The loss is the sum of the four ODE-residual MSE terms plus an
initial-condition term at ``t = 0``.

.. note::
   The transit/kill rates ``k1, k2``, the dose amplitude and the initial
   tumour weight are exposed as parameters with documented defaults; the
   exact paclitaxel values come from refs [17, 33] of the paper and can
   be overridden. The analytic ``c(t)`` is validated against direct ODE
   integration in the test suite.

Public API
----------
create_pkpd_task
    Build the PK-PD PINN :class:`Task`.
PKPDTaskSampler
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
from .task import Task, count_parameters, dataset_dict

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

# Default pharmacokinetic rate constants (per hour, paper Section 4.6).
_K10_PER_HOUR = 0.868
_K12_PER_HOUR = 0.0060
_K21_PER_HOUR = 0.0838
_HOURS_PER_DAY = 24.0


def pk_eigen_coeffs(
    k10: float, k12: float, k21: float
) -> tuple[float, float, float, float]:
    """Return ``(alpha, beta, A, B)`` of the analytic concentration.

    For a unit IV bolus into the central compartment, the plasma
    concentration is ``A exp(-alpha t) + B exp(-beta t)``, where
    ``alpha, beta`` are the (positive) eigenvalues of the two-compartment
    system and ``A + B = 1``.

    Parameters
    ----------
    k10, k12, k21 : float
        Two-compartment rate constants (same time unit as the result).

    Returns
    -------
    tuple[float, float, float, float]
        ``(alpha, beta, A, B)``.
    """
    s = k10 + k12 + k21
    disc = jnp.sqrt(s * s - 4.0 * k10 * k21)
    alpha = (s + disc) / 2.0
    beta = (s - disc) / 2.0
    a_coeff = (alpha - k21) / (alpha - beta)
    b_coeff = (k21 - beta) / (alpha - beta)
    return alpha, beta, a_coeff, b_coeff


def concentration(
    t: Array,
    k10: float,
    k12: float,
    k21: float,
    dose: float,
    dose_times: tuple[float, ...],
) -> Array:
    """Analytic plasma drug concentration ``c(t)`` (Eq. 80-81).

    Sum over dose times of the two-compartment bolus response, each
    switched on for ``t >= t_d``.

    Parameters
    ----------
    t : Array
        Times (days) of shape ``(n, 1)`` or ``(n,)``.
    k10, k12, k21 : float
        Rate constants in per-day units.
    dose : float
        Concentration jump at each dose (``D / V1``).
    dose_times : tuple[float, ...]
        Dose times in days (may be negative for a prior dose).

    Returns
    -------
    Array
        Concentration ``c(t)`` with the same leading shape as ``t``.
    """
    t = jnp.reshape(t, (-1,))
    alpha, beta, a_coeff, b_coeff = pk_eigen_coeffs(k10, k12, k21)
    c = jnp.zeros_like(t)
    for td in dose_times:
        dt = t - td
        resp = dose * (
            a_coeff * jnp.exp(-alpha * dt) + b_coeff * jnp.exp(-beta * dt)
        )
        c = c + jnp.where(dt >= 0.0, resp, 0.0)
    return c


def pkpd_loss(
    model: PyTree,
    t_res: Array,
    k10: float,
    k12: float,
    k21: float,
    k1: float,
    k2: float,
    lam1: float,
    lam2: float,
    psi: float,
    dose: float,
    dose_times: tuple[float, ...],
    omega0: float,
    loss_fn_res: Callable[[Array, Array], Array],
    loss_fn_ic: Callable[[Array, Array], Array],
) -> Array:
    """Compute the physics-informed loss for the PK-PD ODE system.

    Parameters
    ----------
    model : PyTree
        Network mapping time ``t`` (shape ``(1,)``) to four raw outputs;
        states are ``softplus`` of these (non-negative).
    t_res : Array
        Interior time collocation points (days).
    k10, k12, k21 : float
        Pharmacokinetic rate constants (per day).
    k1 : float
        Transit-compartment delay rate (per day).
    k2 : float
        Drug-kill rate (per day per concentration).
    lam1, lam2 : float
        Tumour-growth parameters ``lambda1, lambda2``.
    psi : float
        Hill shape parameter ``Psi``.
    dose : float
        Concentration jump at each dose.
    dose_times : tuple[float, ...]
        Dose times in days.
    omega0 : float
        Initial tumour weight ``omega(0)`` (assigned to ``x1(0)``).
    loss_fn_res, loss_fn_ic : Callable
        Loss functions for the residual and initial-condition terms.

    Returns
    -------
    Array
        Combined ODE-residual + initial-condition loss.
    """

    def states(p: Array) -> Array:
        """Non-negative state vector ``(x1, x2, x3, x4)`` at time ``p``."""
        return jax.nn.softplus(model(p))

    def state_i(p: Array, i: int) -> Array:
        """The ``i``-th state at time ``p`` (scalar)."""
        return states(p)[i]

    c = concentration(t_res, k10, k12, k21, dose, dose_times)

    def residual(p: Array, c_t: Array) -> Array:
        """ODE residual vector at a single time ``p``."""
        x = states(p)
        x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
        omega = x1 + x2 + x3 + x4
        dx = jnp.array([jax.grad(state_i)(p, i)[0] for i in range(4)])
        hill = (1.0 + (lam1 / lam2 * omega) ** psi) ** (1.0 / psi)
        growth = lam1 * x1 / hill
        r1 = dx[0] - (growth - k2 * c_t * x1)
        r2 = dx[1] - (k2 * c_t * x1 - k1 * x2)
        r3 = dx[2] - k1 * (x2 - x3)
        r4 = dx[3] - k1 * (x3 - x4)
        return jnp.array([r1, r2, r3, r4])

    res = jax.vmap(residual)(t_res, c)
    loss_res = loss_fn_res(res, jnp.zeros_like(res))

    x0 = states(jnp.zeros((1,)))
    target0 = jnp.array([omega0, 0.0, 0.0, 0.0])
    loss_ic = loss_fn_ic(x0, target0)

    return loss_res + loss_ic


def create_points_dataset(
    point_allocation: tuple[tuple[float, float, int], ...],
    *,
    key: Array,
) -> dict[str, jnp.ndarray]:
    """Create the non-uniform time collocation grid (Section 4.6).

    Parameters
    ----------
    point_allocation : tuple of (float, float, int)
        ``(t_lo, t_hi, n)`` blocks; points are drawn uniformly inside
        each block and concatenated, increasing sampling density around
        the early injection-driven transients.
    key : Array
        Random key for reproducibility.

    Returns
    -------
    dict[str, jnp.ndarray]
        Dictionary with key ``"t_res"`` of shape ``(total, 1)``.
    """
    keys = jr.split(key, len(point_allocation))
    blocks = []
    for (lo, hi, n), k in zip(point_allocation, keys, strict=True):
        blocks.append(jr.uniform(k, (n,), minval=lo, maxval=hi))
    t = jnp.sort(jnp.concatenate(blocks)).reshape(-1, 1)
    return {"t_res": t}


_DEFAULT_ALLOCATION = ((0.0, 1.9, 300), (1.9, 4.0, 300), (4.0, 17.0, 400))


def create_pkpd_task(
    seed: int,
    dataset_path: str,
    k1: float = 1.0,
    k2: float = 1.0,
    lam1: float = 0.273,
    lam2: float = 0.814,
    psi: float = 20.0,
    dose: float = 1.0,
    dose_times: tuple[float, ...] = (-1.0, 2.0),
    omega0: float = 2.0,
    point_allocation: tuple[tuple[float, float, int], ...] = (
        _DEFAULT_ALLOCATION
    ),
    hidden_size: int = 32,
    num_layers: int = 4,
) -> Task:
    """Create a stiff PK-PD ODE PINN task.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    dataset_path : str
        Path stem used to save the generated time-collocation dataset.
    k1 : float, optional
        Transit-compartment delay rate (per day), by default 1.0.
    k2 : float, optional
        Drug-kill rate (per day per concentration), by default 1.0.
    lam1, lam2 : float, optional
        Tumour-growth parameters, by default 0.273 and 0.814.
    psi : float, optional
        Hill shape parameter, by default 20.0.
    dose : float, optional
        Concentration jump at each dose, by default 1.0.
    dose_times : tuple[float, ...], optional
        Dose times in days, by default ``(-1.0, 2.0)`` (a prior dose plus
        the day-2 injection).
    omega0 : float, optional
        Initial tumour weight, by default 2.0.
    point_allocation : tuple of (float, float, int), optional
        Non-uniform time-grid blocks, by default the 300/300/400 split
        over ``[0, 1.9]``, ``[1.9, 4.0]`` and ``[4.0, 17.0]`` days.
    hidden_size : int, optional
        Hidden-layer width of the MLP, by default 32.
    num_layers : int, optional
        Number of MLP layers, by default 4.

    Returns
    -------
    Task
        The created PK-PD task.
    """
    _path = Path(dataset_path + f"_{seed}").with_suffix(".npz")
    key = jr.key(int(seed))
    model_key, dataset_key = jr.split(key)

    k10 = _K10_PER_HOUR * _HOURS_PER_DAY
    k12 = _K12_PER_HOUR * _HOURS_PER_DAY
    k21 = _K21_PER_HOUR * _HOURS_PER_DAY

    model, model_tags = mlp(
        in_size=1,
        out_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        hidden_activation=jax.nn.tanh,
        key=model_key,
    )

    loss_fn = partial(
        pkpd_loss,
        k10=k10,
        k12=k12,
        k21=k21,
        k1=k1,
        k2=k2,
        lam1=lam1,
        lam2=lam2,
        psi=psi,
        dose=dose,
        dose_times=tuple(float(d) for d in dose_times),
        omega0=omega0,
        loss_fn_res=mse,
        loss_fn_ic=mse,
    )

    num_params = count_parameters(model)
    tag = {
        "task_name": "pkpd",
        "loss_fn": "pkpd_loss",
        "k1": k1,
        "k2": k2,
        "lam1": lam1,
        "lam2": lam2,
        "psi": psi,
        "dose": dose,
        "dose_times": tuple(float(d) for d in dose_times),
        "omega0": omega0,
        "dimensionality": num_params,
    }
    tag.update(model_tags)

    if not _path.exists():
        dataset = create_points_dataset(point_allocation, key=dataset_key)
        save_dataset(dataset, _path)

    return Task(
        model=model,
        loss_fn=loss_fn,
        global_min=0.0,
        dataset=dataset_dict(_path, seed),
        tag=tag,
    )


class PKPDTaskSampler(Block):
    """f3dasm Block that generates stiff PK-PD ODE tasks.

    Parameters
    ----------
    point_allocation : tuple of (float, float, int), optional
        Non-uniform time-grid blocks (see :func:`create_pkpd_task`).
    """

    def __init__(
        self,
        point_allocation: tuple[tuple[float, float, int], ...] = (
            _DEFAULT_ALLOCATION
        ),
    ):
        self.point_allocation = point_allocation

    def call(self, data: ExperimentData, seed: int) -> ExperimentData:
        """Generate PK-PD tasks for each hidden size.

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
            Experiment data containing the PK-PD tasks.
        """
        hidden_size = data.domain.input_space["hidden_size"].categories

        task_list = [
            dict(
                seed=seed,
                dataset_path="./data/pkpd",
                point_allocation=self.point_allocation,
                hidden_size=h,
            )
            for h in hidden_size
        ]

        return build_task_experimentdata(
            task_list,
            (
                "seed",
                "dataset_path",
                "point_allocation",
                "hidden_size",
            ),
            data.project_dir,
        )
