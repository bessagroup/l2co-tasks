"""ExperimentData construction helpers for optimization tasks."""

#                                                                       Modules
# =============================================================================

# Standard
from collections.abc import Iterable, Sequence
from pathlib import Path

# Third-party
from f3dasm import Block, ExperimentData, datagenerator
from f3dasm.design import Domain
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

# Local
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def build_task_experimentdata(
    input_data: Iterable[dict],
    param_names: Sequence[str],
    project_dir: Path,
) -> ExperimentData:
    """Assemble an :class:`ExperimentData` of task configurations.

    Shared tail for the ``*TaskSampler`` Blocks: it builds a
    :class:`f3dasm.design.Domain` from ``param_names``, attaches the
    on-disk ``task`` output wired to :meth:`Task.save` / :meth:`Task.load`,
    and returns the populated ExperimentData. The samplers differ only in
    which rows and parameter names they pass here.

    Parameters
    ----------
    input_data : Iterable[dict]
        One mapping of input parameters per task to materialise.
    param_names : Sequence[str]
        Names of the input parameters to register on the domain.
    project_dir : Path
        Project directory under which task files are stored.

    Returns
    -------
    ExperimentData
        ExperimentData with the registered domain and ``task`` output.
    """
    domain = Domain()
    for name in param_names:
        domain.add_parameter(name)
    domain.add_output(
        name="task",
        to_disk=True,
        store_function=Task.save,
        load_function=Task.load,
    )
    return ExperimentData(
        domain=domain, input_data=list(input_data), project_dir=project_dir
    )


def create_tasks_experimentdata(
    config: DictConfig,
    project_dir: Path,
) -> ExperimentData:
    """Build an :class:`f3dasm.ExperimentData` of optimization tasks.

    The returned ExperimentData has one row per sampled task and a
    ``task`` output column that references the materialised
    :class:`Task` on disk (via ``Task.save`` / ``Task.load``).

    Two config shapes are accepted:

    * **single-generator** — the config binds one ``data_generator``
      to an ``experimentdata`` spec (see :func:`_build_tasks_part`).
    * **composite** — the config carries a top-level ``parts`` list,
      one ``{data_generator, task_kwargs, input_data}`` block per group
      sharing the enclosing ``sampler`` / ``sampler_kwargs``. Each part
      is built and the resulting task sets are row-concatenated (see
      :func:`_create_composite_tasks_experimentdata`). This expresses a
      distribution spanning several task factories — e.g. the
      ERTD-balanced ``bbob_balanced_*`` sets, whose BBOB / embedded /
      noisy suites each need a different ``create_*`` factory.

    Parameters
    ----------
    config : DictConfig
        Hydra config describing the tasks. For the single-generator
        shape the required keys are:

        ``experimentdata`` : DictConfig
            ``f3dasm.ExperimentData.from_yaml`` spec for the initial
            task ExperimentData. If it carries an inline ``input_data``
            list, those rows are used verbatim (and an identity sampler
            should be configured) instead of a sampler-expanded domain.
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

        For the composite shape, ``sampler`` / ``sampler_kwargs`` are
        shared and ``parts`` is a list of ``{data_generator,
        task_kwargs, input_data}`` blocks.
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
    if config.get("parts") is not None:
        return _create_composite_tasks_experimentdata(config, project_dir)
    return _build_tasks_part(config, project_dir)


def _create_composite_tasks_experimentdata(
    config: DictConfig,
    project_dir: Path,
) -> ExperimentData:
    """Row-concatenate one task set per ``config.parts`` entry.

    Each part reuses :func:`_build_tasks_part` with the enclosing
    ``sampler`` / ``sampler_kwargs`` and its own ``data_generator`` /
    ``task_kwargs`` / ``input_data``. Parts are materialised into the
    *same* ``project_dir`` with a cumulative ``index_offset`` so their
    on-disk ``task/<idx>`` references (keyed by row index at
    materialisation time) stay unique; :meth:`ExperimentData.__add__`
    then re-keys the rows to a contiguous ``0..N-1`` range while
    preserving those references.

    Parameters
    ----------
    config : DictConfig
        Composite tasks config with ``sampler``, ``sampler_kwargs`` and
        a ``parts`` list.
    project_dir : Path
        Directory the concatenated task files are written under.

    Returns
    -------
    ExperimentData
        The row-concatenated tasks ExperimentData (not yet stored).
    """
    combined: ExperimentData | None = None
    offset = 0
    for part in config.parts:
        part_config = OmegaConf.create(
            {
                "experimentdata": {"input_data": part.input_data},
                "sampler": config.sampler,
                "sampler_kwargs": config.sampler_kwargs,
                "data_generator": part.data_generator,
                "task_kwargs": part.get("task_kwargs"),
            }
        )
        ed_part = _build_tasks_part(
            part_config, project_dir, index_offset=offset
        )
        offset += len(ed_part)
        combined = ed_part if combined is None else combined + ed_part
    return combined


def _build_tasks_part(
    config: DictConfig,
    project_dir: Path,
    index_offset: int = 0,
) -> ExperimentData:
    """Build a single-generator tasks ExperimentData.

    Performs four steps:

    1. Instantiate the initial ExperimentData from
       ``config.experimentdata`` (defines the task hyperparameter
       domain and any seed rows). Either give a ``domain`` for the
       sampler to expand, or an inline ``input_data`` list of per-task
       rows (materialised from OmegaConf to native containers here)
       paired with a no-op sampler — the latter pins a distinct
       hyperparameter set per task, which the cross-product ``grid``
       sampler cannot express.
    2. Set its project directory to ``project_dir`` so that task
       files are written under ``project_dir/experiment_data/task/``.
    3. Expand the seed rows by calling the sampler Block defined by
       ``config.sampler`` with the kwargs from ``config.sampler_kwargs``.
    4. Materialise each row into a :class:`Task` via the data
       generator defined by ``config.data_generator``, passing
       ``config.task_kwargs`` to its call.

    Parameters
    ----------
    config : DictConfig
        Single-generator tasks config (see
        :func:`create_tasks_experimentdata`).
    project_dir : Path
        Project directory under which the task files are stored.
    index_offset : int, optional
        Shift every row's index by this amount before the data
        generator runs, so the materialised ``task`` files are named
        ``task/{index_offset}`` upwards instead of ``task/0`` upwards.
        Lets the composite path build several parts into the *same*
        ``project_dir`` without their on-disk references colliding
        (each ``task`` reference is keyed by the row index at
        materialisation time). Defaults to ``0`` (no shift).

    Returns
    -------
    ExperimentData
        The populated ExperimentData (not yet stored).
    """
    ed_config = config.experimentdata
    if isinstance(ed_config.get("input_data"), ListConfig):
        # Inline per-task rows (a list of mappings) let a config pin a
        # different hyperparameter set per task — e.g. a 400-D sphere
        # alongside a 120-D rastrigin, which the cross-product ``grid``
        # sampler cannot express. ``ExperimentData.from_yaml`` forwards the
        # value straight to ``ExperimentData(**config)``, but its
        # ``input_data`` factory only accepts a plain ``list[dict]`` (not
        # OmegaConf's ``ListConfig``/``DictConfig``), so materialise the
        # whole spec to native containers first. Pair with a no-op sampler
        # (``random`` at ``n_samples: 0``) so the rows pass through verbatim.
        experiment_data = ExperimentData(
            **OmegaConf.to_container(ed_config, resolve=True)
        )
    else:
        experiment_data = ExperimentData.from_yaml(ed_config)
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

    if index_offset:
        # Re-key rows to start at ``index_offset`` before materialisation.
        # ``_store`` names each ``to_disk`` file by the row index, so this
        # is what keeps concatenated parts from overwriting one another's
        # ``task/<idx>`` files in a shared ``project_dir``. Rebuild via
        # ``from_data`` so the index map is applied cleanly (it re-wraps
        # ``data`` and routes ``project_dir`` through the setter).
        experiment_data = ExperimentData.from_data(
            data={
                index_offset + i: es
                for i, es in enumerate(experiment_data.data.values())
            },
            domain=experiment_data.domain,
            project_dir=project_dir,
        )

    task_fn = datagenerator(
        output_names="task",
        domain=experiment_data.domain,
    )(instantiate(config.data_generator, _partial_=True))

    task_kwargs = config.task_kwargs or {}
    experiment_data = task_fn.call(data=experiment_data, **task_kwargs)

    return experiment_data
