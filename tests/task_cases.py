"""Registry of task cases exercised by the task-contract suite.

Each :class:`TaskCase` names one task type, a thunk that builds a *tiny*
instance of it (small dimensionality, tiny datasets/grids, few inner
steps) and the metadata the contract battery needs but that ``Task``
does not itself expose:

* ``domain`` -- ``"unit"`` for tasks optimised over the normalised box
  ``[0, 1]^d`` (BBOB, CEC2005, the meta task) or ``"unbounded"`` for
  tasks optimised over neural-network weight space / ``R^n``.
* ``gmin_is_lower_bound`` -- whether ``task.global_min`` is a true
  mathematical lower bound on the loss (BBOB / CEC2005 /
  CEC2013-LSGO noiseless, quadratic, the PDE-residual ``0`` of the
  PINN tasks) as opposed to an empirical best-achievable estimate
  (gaussian, spiral, MNIST-1D, meta) or a noiseless value under added
  noise.

Adding a new task family = adding one entry here. The companion
``test_task_registry_guard.py`` fails if a public ``create_*`` factory
has no case registered.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import bbob_jax
import pytest

from l2co_tasks import (
    Task,
    create_bbob_noisy_task,
    create_bbob_task,
    create_cec2005_task,
    create_cec2013lsgo_task,
    create_cec2017_task,
    create_cutest_task,
    create_embedded_bbob_task,
    create_euler_task,
    create_gaussian_meta_task,
    create_gaussian_task,
    create_helmholtz_task,
    create_inviscid_burgers_task,
    create_mnist1d_task,
    create_pde_task,
    create_pkpd_task,
    create_quadratic_task,
    create_spiral_task,
    create_stokes_task,
    create_viscous_burgers_task,
)


@dataclass(frozen=True)
class TaskCase:
    """A single task type plus the metadata the contract suite needs.

    Attributes
    ----------
    id : str
        Stable pytest id, e.g. ``"bbob-sphere"``.
    factory : str
        Name of the public ``create_*`` factory this case covers; used by
        the registry-guard test.
    build : Callable[[Path], Task]
        Builds a tiny task instance under the given temporary directory.
    domain : str
        ``"unit"`` or ``"unbounded"`` (see module docstring).
    gmin_is_lower_bound : bool
        Whether ``global_min`` is a genuine lower bound on the loss.
    marks : tuple
        Pytest marks (e.g. ``pytest.mark.slow``) applied to this case.
    lr : float
        Adam learning rate used by the optimizability suite
        (``test_task_optimizability.py``) for its short training run.
    """

    id: str
    factory: str
    build: Callable[[Path], Task]
    domain: str = "unbounded"
    gmin_is_lower_bound: bool = False
    marks: tuple = field(default_factory=tuple)
    lr: float = 1e-2


# ---------------------------------------------------------------------------
# Builders. Kept at module scope so they stay picklable and tiny.
# ---------------------------------------------------------------------------


def _bbob(fn_name: str, noise: float = 0.0) -> Callable[[Path], Task]:
    """Return a builder for a small BBOB task."""
    return lambda _p: create_bbob_task(
        fn_name=fn_name, seed=0, dimensionality=3, noise=noise
    )


def _cec(fn_name: str) -> Callable[[Path], Task]:
    """Return a builder for a small CEC2005 task."""
    return lambda _p: create_cec2005_task(
        fn_name=fn_name, seed=0, dimensionality=3
    )


def _cec2017(fn_name: str, dimensionality: int = 3) -> Callable[[Path], Task]:
    """Return a builder for a small CEC2017 task.

    Hybrids need one dimension per subcomponent kernel, so their cases
    pass an explicit ``dimensionality`` at or above ``min_ndim``.
    """
    return lambda _p: create_cec2017_task(
        fn_name=fn_name, seed=0, dimensionality=dimensionality
    )


def _lsgo(fn_name: str, dimensionality: int) -> Callable[[Path], Task]:
    """Return a builder for a CEC 2013 LSGO task.

    The only benchmark case that cannot be built small: LSGO is a
    fixed-instance suite defined at exactly one dimensionality per
    function (1000, or 905 for f13/f14), so ``dimensionality`` is
    passed explicitly and the case is inherently a large-scale one.
    """
    return lambda _p: create_cec2013lsgo_task(
        fn_name=fn_name, seed=0, dimensionality=dimensionality
    )


def _bbob_noisy(fn_name: str) -> Callable[[Path], Task]:
    """Return a builder for a small BBOB-noisy task."""
    return lambda _p: create_bbob_noisy_task(
        fn_name=fn_name, seed=0, dimensionality=3
    )


def _embedded_bbob(
    fn_name: str, bulk_scale: float = 0.0
) -> Callable[[Path], Task]:
    """Return a builder for a small randomly-embedded BBOB task."""
    return lambda _p: create_embedded_bbob_task(
        fn_name=fn_name,
        seed=0,
        intrinsic_dim=2,
        ambient_dim=8,
        bulk_scale=bulk_scale,
    )


def _quadratic(n_observations: int | None) -> Callable[[Path], Task]:
    """Return a builder for a small quadratic least-squares task."""
    return lambda _p: create_quadratic_task(
        dimensionality=4, seed=0, n_observations=n_observations
    )


# The empirical-global_min tasks benchmark their minimum at creation
# with a multi-restart Adam search; the contract suite only needs a
# finite, deterministic value, so a tiny budget keeps the build cheap.
_GMIN_KW = dict(global_min_restarts=2, global_min_steps=10)


def _spiral(p: Path) -> Task:
    """Build a tiny spiral RNN task."""
    return create_spiral_task(
        hidden_size=4,
        dataset_size=16,
        dataset_path=str(p / "spiral"),
        seed=0,
        batch_size=8,
        **_GMIN_KW,
    )


def _gaussian(p: Path) -> Task:
    """Build a tiny gaussian-classification MLP task."""
    return create_gaussian_task(
        seed=0,
        dataset_path=str(p / "gauss"),
        num_gaussians=2,
        num_samples_per_gaussian=4,
        dim_points=2,
        l2_regularization=0.01,
        hidden_size=2,
        num_layers=2,
        **_GMIN_KW,
    )


def _gaussian_meta(p: Path) -> Task:
    """Build a tiny gaussian meta-learning task (few inner steps)."""
    return create_gaussian_meta_task(
        seed=0,
        dataset_path=str(p / "gmeta"),
        num_gaussians=2,
        num_samples_per_gaussian=4,
        dim_points=2,
        inner_steps=2,
        l2_regularization=0.01,
        global_min_restarts=2,
        global_min_steps=5,
    )


def _mnist1d(p: Path) -> Task:
    """Build a tiny MNIST-1D MLP task."""
    return create_mnist1d_task(
        dataset_path=str(p / "mnist1d"),
        dataset_size=64,
        seed=0,
        batch_size=8,
        **_GMIN_KW,
    )


def _pde(pde_name: str) -> Callable[[Path], Task]:
    """Return a builder for a tiny PINN PDE task on a coarse grid."""

    def build(p: Path) -> Task:
        """Build the PDE task with tiny collocation grids."""
        common = dict(
            seed=0,
            dataset_path=str(p / pde_name),
            x_range=(0.0, 1.0) if pde_name == "wave" else (0.0, 2.0 * math.pi),
            t_range=(0.0, 1.0),
            xgrid_resolution=8,
            tgrid_resolution=8,
            num_ic_points=4,
            num_bc_points=4,
            num_res_points=8,
            hidden_size=4,
        )
        if pde_name == "reaction":
            return create_pde_task(pde_task_name="reaction", rho=1.0, **common)
        beta = 1.0 if pde_name == "convection" else 2.0
        return create_pde_task(pde_task_name=pde_name, beta=beta, **common)

    return build


def _helmholtz(dim: int) -> Callable[[Path], Task]:
    """Return a builder for a tiny Helmholtz PINN task."""

    def build(p: Path) -> Task:
        """Build the Helmholtz task with few collocation points."""
        if dim == 2:
            return create_helmholtz_task(
                a1=1.0,
                a2=4.0,
                k=1.0,
                seed=0,
                dataset_path=str(p / "helmholtz2"),
                dim=2,
                modes=(1,),
                num_res_points=16,
                hidden_size=4,
                num_layers=2,
            )
        return create_helmholtz_task(
            a1=4.0,
            a2=4.0,
            k=1.0,
            seed=0,
            dataset_path=str(p / "helmholtz3"),
            dim=3,
            a3=3.0,
            modes=(1, 2),
            num_res_points=16,
            hidden_size=4,
            num_layers=2,
        )

    return build


def _viscous_burgers(p: Path) -> Task:
    """Build a tiny viscous-Burgers PINN task."""
    return create_viscous_burgers_task(
        seed=0,
        dataset_path=str(p / "vburgers"),
        num_res_points=16,
        num_ic_points=8,
        num_bc_points=8,
        hidden_size=4,
        num_layers=2,
    )


def _pkpd(p: Path) -> Task:
    """Build a tiny stiff PK-PD ODE PINN task."""
    return create_pkpd_task(
        seed=0,
        dataset_path=str(p / "pkpd"),
        point_allocation=((0.0, 1.9, 8), (1.9, 4.0, 8), (4.0, 17.0, 8)),
        hidden_size=4,
        num_layers=2,
    )


def _inviscid_burgers(p: Path) -> Task:
    """Build a tiny inviscid-Burgers PINN task (two networks)."""
    return create_inviscid_burgers_task(
        seed=0,
        dataset_path=str(p / "iburgers"),
        num_res_points=16,
        num_ic_points=8,
        num_bc_points=8,
        hidden_size=4,
        num_layers=2,
    )


def _euler(stage: str) -> Callable[[Path], Task]:
    """Return a builder for a tiny Euler PINN task for a given stage."""
    return lambda p: create_euler_task(
        seed=0,
        dataset_path=str(p / f"euler_{stage}"),
        stage=stage,
        num_res_points=16,
        num_ic_points=8,
        num_bc_points=8,
        hidden_size=4,
        num_layers=2,
    )


def _stokes(p: Path) -> Task:
    """Build a tiny Stokes-wedge PINN task."""
    return create_stokes_task(
        seed=0,
        dataset_path=str(p / "stokes"),
        num_res_points=16,
        num_lid_points=8,
        num_wall_points=8,
        hidden_size=4,
        num_layers=2,
    )


_SLOW = (pytest.mark.slow,)

# The BBOB-noisy suite ships with bbob-jax releases newer than 1.8.0;
# on older installs its cases skip instead of failing the battery.
_NEEDS_BBOB_NOISY = (
    pytest.mark.skipif(
        not hasattr(bbob_jax, "bbob_noisy_registry"),
        reason="installed bbob-jax predates the BBOB-noisy suite",
    ),
)

# The CEC 2013 LSGO suite ships with bbob-jax releases newer than 2.0.0;
# on older installs its case skips instead of failing the battery.
_NEEDS_LSGO = (
    pytest.mark.skipif(
        not hasattr(bbob_jax, "cec2013lsgo_registry"),
        reason="installed bbob-jax predates the CEC 2013 LSGO suite",
    ),
)


# CUTEst cases need pycutest and a CUTEst installation (CUTEST,
# SIFDECODE, MASTSIF set); without them they skip, as they do in CI.
def _cutest_available() -> bool:
    try:
        import pycutest  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


_NEEDS_CUTEST = (
    pytest.mark.skipif(
        not _cutest_available(),
        reason="needs pycutest and a CUTEst installation",
    ),
)


def _cutest(problem: str, global_min: float) -> Callable[[Path], Task]:
    """Return a builder for a CUTEst task outside the committed table."""
    return lambda _p: create_cutest_task(problem, global_min=global_min)


TASK_CASES: list[TaskCase] = [
    # Analytical, cheap, deterministic -- the strongest correctness checks.
    TaskCase(
        "bbob-sphere",
        "create_bbob_task",
        _bbob("sphere"),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    TaskCase(
        "bbob-rastrigin",
        "create_bbob_task",
        _bbob("rastrigin"),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    # Noisy BBOB exercises the pass_rng path (no lower-bound guarantee).
    TaskCase(
        "bbob-sphere-noisy",
        "create_bbob_task",
        _bbob("sphere", noise=0.1),
        domain="unit",
        gmin_is_lower_bound=False,
    ),
    # BBOB-noisy is inherently stochastic (pass_rng=True, no lower-bound
    # guarantee): f101 = moderate Gaussian noise on a separable unimodal
    # base, f124 = severe Cauchy noise on a multimodal base.
    TaskCase(
        "bbob-noisy-f101",
        "create_bbob_noisy_task",
        _bbob_noisy("bbob_noisy_f101"),
        domain="unit",
        gmin_is_lower_bound=False,
        marks=_NEEDS_BBOB_NOISY,
    ),
    TaskCase(
        "bbob-noisy-f124",
        "create_bbob_noisy_task",
        _bbob_noisy("bbob_noisy_f124"),
        domain="unit",
        gmin_is_lower_bound=False,
        marks=_NEEDS_BBOB_NOISY,
    ),
    # A CUTEst problem in its own coordinates, evaluated in compiled
    # Fortran behind a host loss (ADRs 0003, 0004). ROSENBR is a sum of
    # squares, so 0 is a true lower bound.
    TaskCase(
        "cutest-rosenbr",
        "create_cutest_task",
        _cutest("ROSENBR", 0.0),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_NEEDS_CUTEST,
    ),
    TaskCase(
        "cec2005-f1",
        "create_cec2005_task",
        _cec("f1"),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    # f4 is inherently stochastic (pass_rng=True even without noise).
    TaskCase(
        "cec2005-f4",
        "create_cec2005_task",
        _cec("f4"),
        domain="unit",
        gmin_is_lower_bound=False,
    ),
    TaskCase(
        "cec2017-f1",
        "create_cec2017_task",
        _cec2017("cec2017_f1"),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    # A hybrid: exercises the min_ndim-constrained family (needs >= 5
    # dimensions; one per subcomponent kernel).
    TaskCase(
        "cec2017-f17",
        "create_cec2017_task",
        _cec2017("cec2017_f17", dimensionality=5),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    # CEC 2013 LSGO: the one benchmark case that is large by
    # construction (1000-D, fixed instance). f3 (shifted Ackley, fully
    # separable) is the cheapest member and the best conditioned, so it
    # exercises the 1000-D path without the 1e11-1e21 loss magnitudes of
    # the elliptic- and Schwefel-based members.
    TaskCase(
        "cec2013lsgo-f3",
        "create_cec2013lsgo_task",
        _lsgo("cec2013lsgo_f3", dimensionality=1000),
        domain="unit",
        gmin_is_lower_bound=True,
        marks=_NEEDS_LSGO,
    ),
    # Randomly-embedded BBOB: low intrinsic dimension inside a larger
    # ambient space. The optimum is attained exactly at the seeded
    # anchor point, so global_min stays a true lower bound.
    TaskCase(
        "embedded-sphere",
        "create_embedded_bbob_task",
        _embedded_bbob("sphere"),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    # bulk_scale > 0 exercises the null-space penalty path (unique
    # minimiser instead of a flat manifold of minimisers).
    TaskCase(
        "embedded-rastrigin-bulk",
        "create_embedded_bbob_task",
        _embedded_bbob("rastrigin", bulk_scale=1e-2),
        domain="unit",
        gmin_is_lower_bound=True,
    ),
    TaskCase(
        "quadratic-square",
        "create_quadratic_task",
        _quadratic(None),
        domain="unbounded",
        gmin_is_lower_bound=True,
    ),
    TaskCase(
        "quadratic-overdetermined",
        "create_quadratic_task",
        _quadratic(8),
        domain="unbounded",
        gmin_is_lower_bound=True,
    ),
    # Neural-network / dataset-backed tasks (heavier -> slow).
    TaskCase(
        "spiral",
        "create_spiral_task",
        _spiral,
        domain="unbounded",
        marks=_SLOW,
    ),
    TaskCase(
        "gaussian-class",
        "create_gaussian_task",
        _gaussian,
        domain="unbounded",
        marks=_SLOW,
    ),
    TaskCase(
        "gaussian-meta",
        "create_gaussian_meta_task",
        _gaussian_meta,
        domain="unit",
        marks=_SLOW,
        # The outer parameter is a 3-vector in [0, 1]; adam's normalised
        # steps at 1e-2 barely move it within the suite's step budget.
        lr=0.02,
    ),
    # PDE residual losses are sums of MSE terms, so global_min = 0 is a
    # genuine lower bound (theoretical minimum).
    TaskCase(
        "pde-convection",
        "create_pde_task",
        _pde("convection"),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "pde-reaction",
        "create_pde_task",
        _pde("reaction"),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "pde-wave",
        "create_pde_task",
        _pde("wave"),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "mnist1d",
        "create_mnist1d_task",
        _mnist1d,
        domain="unbounded",
        marks=_SLOW,
    ),
    # PINN benchmarks (Jnini et al. 2026). The loss is a sum of MSE /
    # penalty terms, so global_min = 0 is a genuine lower bound.
    TaskCase(
        "helmholtz-2d",
        "create_helmholtz_task",
        _helmholtz(2),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "helmholtz-3d",
        "create_helmholtz_task",
        _helmholtz(3),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "viscous-burgers",
        "create_viscous_burgers_task",
        _viscous_burgers,
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "pkpd",
        "create_pkpd_task",
        _pkpd,
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "inviscid-burgers",
        "create_inviscid_burgers_task",
        _inviscid_burgers,
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "euler-viscous",
        "create_euler_task",
        _euler("viscous"),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "euler-inviscid",
        "create_euler_task",
        _euler("inviscid"),
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
    TaskCase(
        "stokes",
        "create_stokes_task",
        _stokes,
        domain="unbounded",
        gmin_is_lower_bound=True,
        marks=_SLOW,
    ),
]

CASE_BY_ID: dict[str, TaskCase] = {case.id: case for case in TASK_CASES}
