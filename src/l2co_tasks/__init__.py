"""
L2CO Tasks - Optimization tasks compatible with the L2CO library
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.benchmark_task import create_bbob_task, create_cec2005_task
from ._src.continue_from_path import retrieve_tasks
from ._src.gaussian_meta import create_gaussian_meta_task
from ._src.mnist1d_task import create_mnist1d_task
from ._src.pde import create_pde_task
from ._src.quadratic_task import create_quadratic_task
from ._src.spiral_task import create_spiral_task
from ._src.task import DatasetDict, Task
from ._src.task_gaussian_class import create_gaussian_task

#                                                        Authorship and Credits
# =============================================================================
__author__ = "Martin van der Schelling (m.p.vanderschelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
#
# =============================================================================

__all__ = [
    "DatasetDict",
    "Task",
    "create_bbob_task",
    "create_cec2005_task",
    "create_gaussian_meta_task",
    "create_gaussian_task",
    "create_mnist1d_task",
    "create_pde_task",
    "create_quadratic_task",
    "create_spiral_task",
    "retrieve_tasks",
]
