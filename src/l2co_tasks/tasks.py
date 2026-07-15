"""
Submodule with built-in optimization tasks.

Re-exports the ``create_*`` factory functions for ergonomic access, e.g.::

    from l2co_tasks.tasks import create_bbob_task
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.benchmark_task import (
    create_bbob_noisy_task,
    create_bbob_task,
    create_cec2005_task,
    create_cec2017_task,
)
from ._src.embedded_task import create_embedded_bbob_task
from ._src.euler import create_euler_task
from ._src.gaussian_meta import create_gaussian_meta_task
from ._src.helmholtz import create_helmholtz_task
from ._src.inviscid_burgers import create_inviscid_burgers_task
from ._src.mnist1d_task import create_mnist1d_task
from ._src.pde import create_pde_task
from ._src.pkpd import create_pkpd_task
from ._src.quadratic_task import create_quadratic_task
from ._src.spiral_task import create_spiral_task
from ._src.stokes import create_stokes_task
from ._src.task_gaussian_class import create_gaussian_task
from ._src.viscous_burgers import create_viscous_burgers_task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "create_bbob_noisy_task",
    "create_bbob_task",
    "create_cec2005_task",
    "create_cec2017_task",
    "create_embedded_bbob_task",
    "create_euler_task",
    "create_gaussian_meta_task",
    "create_gaussian_task",
    "create_helmholtz_task",
    "create_inviscid_burgers_task",
    "create_mnist1d_task",
    "create_pde_task",
    "create_pkpd_task",
    "create_quadratic_task",
    "create_spiral_task",
    "create_stokes_task",
    "create_viscous_burgers_task",
]
