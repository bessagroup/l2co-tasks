"""The ``bbob_metabox_train`` task set is the MetaBox bbob-10D
'difficult' training split, one instance per function.

It is composed through Hydra the way a downstream experiment selects it
(``pkg://l2co_tasks.conf`` on the search path, ``tasks=bbob_metabox_train``)
and built with :func:`l2co_tasks.create_tasks_experimentdata`, as
``test_conf_tasks.py`` builds every shipped set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from l2co_tasks import create_tasks_experimentdata

# BBOB F1 F2 F3 F5 F15 F16 F17 F21 under their bbob-jax names, in order.
EXPECTED_FN_NAMES = [
    "sphere",
    "ellipsoid_seperable",
    "rastrigin_seperable",
    "linear_slope",
    "rastrigin",
    "weierstrass",
    "schaffer_f7_condition_10",
    "gallagher_101_peaks",
]


@pytest.mark.requires_f3dasm
def test_bbob_metabox_train_is_the_metabox_10d_train_split(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "primary"
    empty.mkdir()
    (empty / "config.yaml").write_text(
        "defaults:\n  - tasks: bbob_metabox_train\n  - _self_\n"
        "seed: 0\n"
        "hydra:\n  searchpath:\n    - pkg://l2co_tasks.conf\n"
    )
    with initialize_config_dir(config_dir=str(empty), version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.resolve(cfg)

    experiment_data = create_tasks_experimentdata(
        cfg.tasks, project_dir=tmp_path / "tasks"
    )
    tags = [s.output_data["task"].tag for _, s in experiment_data]

    assert [t["fn_name"] for t in tags] == EXPECTED_FN_NAMES
    assert [t["dimensionality"] for t in tags] == [10] * 8
    assert [t["seed"] for t in tags] == [0] * 8
