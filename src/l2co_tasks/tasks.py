"""
Submodule with built-in optimization tasks.

Re-exports the ``create_*`` factory functions for ergonomic access, e.g.::

    from l2co_tasks.tasks import create_bbob_task
"""

#                                                                       Modules
# =============================================================================

# Local
from ._src.benchmark_task import create_bbob_task, create_cec2005_task
from ._src.gaussian_meta import create_gaussian_meta_task
from ._src.mnist1d_task import create_mnist1d_task
from ._src.pde import create_pde_task
from ._src.quadratic_task import create_quadratic_task
from ._src.spiral_task import create_spiral_task
from ._src.task_gaussian_class import create_gaussian_task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "create_bbob_task",
    "create_cec2005_task",
    "create_gaussian_meta_task",
    "create_gaussian_task",
    "create_mnist1d_task",
    "create_pde_task",
    "create_quadratic_task",
    "create_spiral_task",
]
