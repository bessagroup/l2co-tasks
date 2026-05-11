"""ExperimentData construction helpers for optimization tasks."""

#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
from f3dasm import Block, ExperimentData, datagenerator
from hydra.utils import instantiate
from omegaconf import DictConfig

# Local
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def create_tasks_experimentdata(
    config: DictConfig,
    project_dir: Path,
) -> ExperimentData:
    """Build an :class:`f3dasm.ExperimentData` of optimization tasks.

    The returned ExperimentData has one row per sampled task and a
    ``task`` output column that references the materialised
    :class:`Task` on disk (via ``Task.save`` / ``Task.load``).

    The function performs four steps:

    1. Instantiate the initial ExperimentData from
       ``config.experimentdata`` (defines the task hyperparameter
       domain and any seed rows).
    2. Set its project directory to ``project_dir`` so that task
       files are written under ``project_dir/experiment_data/task/``.
    3. Expand the seed rows by calling the sampler Block defined by
       ``config.sampler`` with the kwargs from
       ``config.sampler_kwargs``.
    4. Materialise each row into a :class:`Task` via the data
       generator defined by ``config.data_generator``, passing
       ``config.task_kwargs`` to its call.

    Parameters
    ----------
    config : DictConfig
        Hydra config describing the tasks. Required keys:

        ``experimentdata`` : DictConfig
            ``f3dasm.ExperimentData.from_yaml`` spec for the initial
            task ExperimentData.
        ``sampler`` : DictConfig
            ``f3dasm.Block.from_yaml`` init spec for the sampler that
            expands the hyperparameter grid.
        ``sampler_kwargs`` : DictConfig
            Keyword arguments passed to the sampler's ``call``.
        ``data_generator`` : DictConfig
            Hydra ``_target_`` spec for the per-row Task factory; the
            target is instantiated as a partial and wrapped via
            ``f3dasm.datagenerator(output_names="task")``.
        ``task_kwargs`` : DictConfig or None
            Keyword arguments forwarded to the data generator's
            ``call``. May be ``None`` or omitted.
    project_dir : Path
        Project directory under which the task files are stored. Set
        on the ExperimentData before the data generator runs so that
        ``to_disk`` outputs land in ``project_dir/experiment_data/``.

    Returns
    -------
    ExperimentData
        The populated ExperimentData. Not yet stored to disk; call
        ``.store()`` on the result if persistence is required.
    """
    experiment_data = ExperimentData.from_yaml(config.experimentdata)
    experiment_data = experiment_data.set_project_dir(project_dir)
    experiment_data.domain.add_output(
        name="task",
        to_disk=True,
        store_function=Task.save,
        load_function=Task.load,
    )

    sampler = Block.from_yaml(
        init_config=config.sampler,
        call_config=config.sampler_kwargs,
    )
    sampler.arm(experiment_data)
    experiment_data = sampler.call(data=experiment_data)

    task_fn = datagenerator(
        output_names="task",
        domain=experiment_data.domain,
    )(instantiate(config.data_generator, _partial_=True))

    task_kwargs = config.task_kwargs or {}
    experiment_data = task_fn.call(data=experiment_data, **task_kwargs)

    return experiment_data
