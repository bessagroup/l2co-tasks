"""
L2CO Tasks - Optimization tasks compatible with the L2CO library
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.benchmark_task import (
    CEC2019Sampler,
    create_bbob_noisy_task,
    create_bbob_task,
    create_cec2005_task,
    create_cec2013lsgo_task,
    create_cec2017_task,
)
from ._src.continue_from_path import retrieve_tasks
from ._src.embedded_task import create_embedded_bbob_task
from ._src.euler import EulerTaskSampler, create_euler_task
from ._src.experimentdata import create_tasks_experimentdata
from ._src.gaussian_meta import create_gaussian_meta_task
from ._src.global_min import estimate_global_min
from ._src.helmholtz import HelmholtzTaskSampler, create_helmholtz_task
from ._src.inviscid_burgers import (
    InviscidBurgersTaskSampler,
    create_inviscid_burgers_task,
)
from ._src.mnist1d_task import create_mnist1d_task
from ._src.pde import PDETaskSampler, create_pde_task
from ._src.pkpd import PKPDTaskSampler, create_pkpd_task
from ._src.quadratic_task import create_quadratic_task
from ._src.spiral_task import create_spiral_task
from ._src.stokes import StokesTaskSampler, create_stokes_task
from ._src.task import DatasetDict, Task
from ._src.task_gaussian_class import create_gaussian_task
from ._src.viscous_burgers import (
    ViscousBurgersTaskSampler,
    create_viscous_burgers_task,
)

#                                                        Authorship and Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
#
# =============================================================================

__all__ = [
    "CEC2019Sampler",
    "DatasetDict",
    "EulerTaskSampler",
    "HelmholtzTaskSampler",
    "InviscidBurgersTaskSampler",
    "PDETaskSampler",
    "PKPDTaskSampler",
    "StokesTaskSampler",
    "Task",
    "ViscousBurgersTaskSampler",
    "create_bbob_noisy_task",
    "create_bbob_task",
    "create_cec2005_task",
    "create_cec2013lsgo_task",
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
    "create_tasks_experimentdata",
    "create_viscous_burgers_task",
    "estimate_global_min",
    "retrieve_tasks",
]
