"""End-to-end tests for the task configs shipped at
``l2co_tasks/conf/tasks/``.

Each YAML describes a *task distribution*: a sampler over an
ExperimentData domain plus a ``create_*_task`` data generator. The tests
here load every shipped YAML, run the full
:func:`l2co_tasks.create_tasks_experimentdata` pipeline against a
shrunk-down version of the domain, and assert that the resulting
ExperimentData contains real :class:`l2co_tasks.Task` objects.

The point is to verify that the *YAML wiring* is correct (Hydra
``_target_`` strings, sampler / data_generator pairing, interpolation
references) — not to enumerate the full Cartesian product, which for
``bbob.yaml`` alone is 24 × 6 = 144 tasks. Per-config overlays in
:data:`_OVERLAYS` shrink heavy grids and cap ``n_samples`` so each
parametrization stays cheap.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import bbob_jax
import pytest
from omegaconf import DictConfig, OmegaConf

from l2co_tasks import Task, create_tasks_experimentdata

# Configs whose data generator needs the BBOB-noisy suite, which ships
# with bbob-jax releases newer than 1.8.0; on older installs they skip.
# The balanced composites include a bbob_noisy ``parts`` entry.
_NEEDS_BBOB_NOISY = {
    "bbob_noisy",
    "bbob_balanced_train",
    "bbob_balanced_test",
}
_HAS_BBOB_NOISY = hasattr(bbob_jax, "bbob_noisy_registry")

# OmegaConf overlays merged on top of each raw YAML before sampling.
# Goals:
#   * cap heavy grids to a handful of tasks (bbob / bbob2 / cec2005);
#   * cap explicit ``n_samples`` knobs (random samplers);
#   * shrink dataset-building tasks (mnist1d, spirals) so the test
#     doesn't pay full dataset-construction cost.
# Configs that already ship with a tiny domain (bbob_test, bbob_holdout,
# bbob_single, cec2005_small, pde) appear here only when an overlay is
# strictly needed.
_OVERLAYS: dict[str, dict] = {
    "bbob": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["sphere", "rastrigin"],
                    },
                    "dimensionality": {"type": "category", "categories": [2]},
                }
            }
        },
    },
    "bbob2": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["sphere", "rastrigin"],
                    },
                    "dimensionality": {"type": "category", "categories": [2]},
                }
            }
        },
    },
    "bbob_small": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["sphere", "rastrigin"],
                    },
                    "dimensionality": {"type": "category", "categories": [2]},
                }
            }
        },
    },
    "bbob_embedded": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["sphere", "rastrigin"],
                    },
                    "intrinsic_dim": {"type": "category", "categories": [2]},
                    "ambient_dim": {"type": "category", "categories": [8]},
                }
            }
        },
    },
    "bbob_noisy": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["bbob_noisy_f101", "bbob_noisy_f124"],
                    },
                    "dimensionality": {"type": "category", "categories": [2]},
                }
            }
        },
    },
    "cec2005": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["f1", "f2"],
                    },
                    "dimensionality": {"type": "category", "categories": [10]},
                }
            }
        },
    },
    "cec2005_small": {
        "experimentdata": {
            "domain": {
                "input": {
                    "fn_name": {
                        "type": "category",
                        "categories": ["f1", "f2"],
                    },
                }
            }
        },
    },
    "gaussian_classification": {
        "experimentdata": {
            "domain": {
                "input": {
                    "seed": {"type": "int", "low": 0, "high": 1},
                    "hidden_size": {"type": "category", "categories": [2]},
                    "num_layers": {"type": "category", "categories": [2]},
                }
            }
        },
        # Tiny global_min benchmark so the empirical estimate stays cheap.
        "task_kwargs": {"global_min_restarts": 2, "global_min_steps": 10},
    },
    "gaussian_meta": {
        "sampler_kwargs": {"n_samples": 2},
        "task_kwargs": {"global_min_restarts": 2, "global_min_steps": 5},
    },
    "quadratic": {"sampler_kwargs": {"n_samples": 3}},
    "mnist1d": {
        "sampler_kwargs": {"n_samples": 2},
        "task_kwargs": {
            "dataset_size": 64,
            "batch_size": 8,
            "global_min_restarts": 2,
            "global_min_steps": 10,
        },
    },
    "spirals": {
        "sampler_kwargs": {"n_samples": 2},
        "task_kwargs": {
            "dataset_size": 64,
            "batch_size": 8,
            "global_min_restarts": 2,
            "global_min_steps": 10,
        },
    },
    # PINN families: shrink the collocation grids (sampler init kwargs)
    # and the network so each task is cheap to build.
    "helmholtz": {
        "sampler": {"num_res_points": 16},
        "experimentdata": {
            "domain": {
                "input": {
                    "hidden_size": {"type": "category", "categories": [4]}
                }
            }
        },
    },
    "viscous_burgers": {
        "sampler": {
            "num_res_points": 16,
            "num_ic_points": 8,
            "num_bc_points": 8,
        },
        "experimentdata": {
            "domain": {
                "input": {
                    "hidden_size": {"type": "category", "categories": [4]}
                }
            }
        },
    },
    "inviscid_burgers": {
        "sampler": {
            "num_res_points": 16,
            "num_ic_points": 8,
            "num_bc_points": 8,
        },
        "experimentdata": {
            "domain": {
                "input": {
                    "hidden_size": {"type": "category", "categories": [4]}
                }
            }
        },
    },
    "euler": {
        "sampler": {
            "num_res_points": 16,
            "num_ic_points": 8,
            "num_bc_points": 8,
        },
        "experimentdata": {
            "domain": {
                "input": {
                    "hidden_size": {"type": "category", "categories": [4]}
                }
            }
        },
    },
    "stokes": {
        "sampler": {
            "num_res_points": 32,
            "num_lid_points": 8,
            "num_wall_points": 8,
        },
        "experimentdata": {
            "domain": {
                "input": {
                    "hidden_size": {"type": "category", "categories": [4]}
                }
            }
        },
    },
    "pkpd": {
        "sampler": {
            "point_allocation": [
                [0.0, 1.9, 8],
                [1.9, 4.0, 8],
                [4.0, 17.0, 8],
            ]
        },
        "experimentdata": {
            "domain": {
                "input": {
                    "hidden_size": {"type": "category", "categories": [4]}
                }
            }
        },
    },
}

# Configs that take noticeably longer than the others — typically because
# the data generator builds an on-disk dataset (mnist1d, spirals) or
# solves PDE residuals (pde). Marked ``slow`` so ``pytest -m 'not slow'``
# skips them in fast inner loops.
_SLOW = {
    "mnist1d",
    "pde",
    "spirals",
    "helmholtz",
    "viscous_burgers",
    "inviscid_burgers",
    "euler",
    "stokes",
    "pkpd",
    # 284-task ERTD-balanced composites — materialised in full (their
    # identity sampler admits no shrinking overlay).
    "bbob_balanced_train",
    "bbob_balanced_test",
}


def _shipped_yaml_names() -> list[str]:
    """Sorted stems of every YAML under ``l2co_tasks.conf.tasks/``."""
    return sorted(
        Path(entry.name).stem
        for entry in resources.files("l2co_tasks.conf.tasks").iterdir()
        if entry.name.endswith(".yaml")
    )


def _load_shipped(name: str) -> DictConfig:
    """Load the YAML at ``l2co_tasks.conf.tasks/<name>.yaml``."""
    text = (
        resources.files("l2co_tasks.conf.tasks")
        .joinpath(f"{name}.yaml")
        .read_text(encoding="utf-8")
    )
    return OmegaConf.create(text)


def _compose(name: str) -> DictConfig:
    """Compose a tiny parent that mimics how an experiment's main
    Hydra config sees the task block.

    The resulting tree is::

        { seed: 0, tasks: { ... loaded YAML, with overlay merged ... } }

    so absolute interpolations like ``${seed}`` resolve to the parent's
    ``seed`` and relative ones like ``${......fn_name}`` resolve inside
    the ``tasks`` subtree.
    """
    raw = _load_shipped(name)
    overlay = _OVERLAYS.get(name, {})
    return OmegaConf.merge(
        OmegaConf.create({"seed": 0, "tasks": raw}),
        OmegaConf.create({"tasks": overlay}),
    )


@pytest.mark.requires_f3dasm
@pytest.mark.parametrize("config_name", _shipped_yaml_names())
def test_sample_task_distribution(
    config_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Each shipped task config produces a non-empty ExperimentData of Tasks.

    The overlay defined in :data:`_OVERLAYS` (if any) trims the sampling
    domain so the test stays fast. Heavy configs that build on-disk
    datasets or solve PDE residuals carry the ``slow`` marker.
    """
    if config_name in _SLOW:
        request.applymarker(pytest.mark.slow)
    if config_name in _NEEDS_BBOB_NOISY and not _HAS_BBOB_NOISY:
        pytest.skip("installed bbob-jax predates the BBOB-noisy suite")

    # mnist1d, spirals, gaussian_classification ship relative
    # ``task_kwargs.dataset_path`` strings; PDETaskSampler likewise
    # writes ``./data/{convection,reaction,wave}``. Run the sampler
    # inside tmp_path so those land in an isolated directory.
    monkeypatch.chdir(tmp_path)

    cfg = _compose(config_name)
    experiment_data = create_tasks_experimentdata(
        cfg.tasks, project_dir=tmp_path
    )

    n_rows = len(experiment_data)
    assert n_rows >= 1, f"{config_name!r}: sampler produced zero rows"

    for _, sample in experiment_data:
        task = sample.output_data["task"]
        assert isinstance(task, Task), (
            f"{config_name!r}: row produced {type(task).__name__}, "
            f"expected Task"
        )


@pytest.mark.requires_f3dasm
def test_composite_parts_experimentdata(tmp_path: Path) -> None:
    """A ``parts`` composite row-concatenates one task set per entry.

    Uses two different factories (``create_bbob_task`` +
    ``create_embedded_bbob_task``) to exercise the cross-factory concat.
    Asserts the parts are materialised into a single ``project_dir`` with
    unique ``task/<idx>`` files (no overwrite between parts) and that each
    row loads the Task from the factory of its part, in order.
    """
    cfg = OmegaConf.create(
        {
            "sampler": {
                "_target_": "f3dasm.create_sampler",
                "sampler": "random",
            },
            "sampler_kwargs": {"n_samples": 0},
            "parts": [
                {
                    "data_generator": {
                        "_target_": "l2co_tasks.create_bbob_task",
                        "_partial_": True,
                    },
                    "task_kwargs": None,
                    "input_data": [
                        {"fn_name": "sphere", "dimensionality": 2, "seed": 0},
                        {"fn_name": "sphere", "dimensionality": 3, "seed": 0},
                    ],
                },
                {
                    "data_generator": {
                        "_target_": "l2co_tasks.create_embedded_bbob_task",
                        "_partial_": True,
                    },
                    "task_kwargs": {"bulk_scale": 0.0},
                    "input_data": [
                        {
                            "fn_name": "sphere",
                            "intrinsic_dim": 2,
                            "ambient_dim": 64,
                            "seed": 0,
                        },
                    ],
                },
            ],
        }
    )

    project_dir = tmp_path / "tasks"
    experiment_data = create_tasks_experimentdata(cfg, project_dir)
    experiment_data.store()

    # 2 + 1 rows, re-keyed to a contiguous range.
    assert len(experiment_data) == 3
    assert list(experiment_data.data.keys()) == [0, 1, 2]

    # Parts share one project_dir; their task files must not collide.
    task_dir = project_dir / "experiment_data" / "task"
    assert sorted(p.stem for p in task_dir.iterdir()) == ["0", "1", "2"]

    tasks = [sample.output_data["task"] for _, sample in experiment_data]
    assert all(isinstance(t, Task) for t in tasks)
    # Rows land in part order and are not overwritten by later parts:
    # the first two come from create_bbob_task (dims 2, 3), the third
    # from create_embedded_bbob_task (ambient dim 64, embedded flag set).
    assert [t.tag["fn_name"] for t in tasks] == ["sphere"] * 3
    assert [t.tag["dimensionality"] for t in tasks] == [2, 3, 64]
    assert [t.tag.get("embedded", False) for t in tasks] == [
        False,
        False,
        True,
    ]
