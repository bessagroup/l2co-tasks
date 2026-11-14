"""The ``joint_selection_test_half`` task set is the whole base-landscape
test half of the DATASET split, before the gap-maximising selection.

The rows are read straight from the shipped YAML's ``parts[].input_data``
(no task is built here; ``test_conf_tasks.py`` builds every shipped set).
They cannot be compared with the split record itself, which lives on
Oscar; these tests check what the record implies instead.
"""

from __future__ import annotations

import re
from importlib import resources

from omegaconf import OmegaConf

from l2co_tasks import base_landscape

# The suite each part's factory builds.
_SUITE_OF_FACTORY = {
    "l2co_tasks.create_bbob_task": "bbob",
    "l2co_tasks.create_embedded_bbob_task": "bbob_embedded",
    "l2co_tasks.create_bbob_noisy_task": "bbob_noisy",
    "l2co_tasks.create_cec2005_task": "cec2005",
    "l2co_tasks.create_cec2017_task": "cec2017",
}

# The test half of the 2026-11-14 joint_selection split, per suite, as
# its record (test.json, "half") gives it.
_HALF_SUITES = {
    "bbob": 72,
    "bbob_embedded": 96,
    "bbob_noisy": 90,
    "cec2005": 36,
    "cec2017": 45,
}

# The split's base-landscape groups (ADR 0007), copied from the same
# record (train.json, "split": "train_groups" and "test_groups", seed 0).
_TRAIN_GROUPS = frozenset(
    {
        "cec2005_f15", "cec2005_f18", "cec2005_f19", "cec2005_f21",
        "cec2005_f22", "cec2005_f23", "cec2005_f24",
        "cec2017_f11", "cec2017_f13", "cec2017_f14", "cec2017_f16",
        "cec2017_f17", "cec2017_f18", "cec2017_f21", "cec2017_f25",
        "cec2017_f27", "cec2017_f29",
        "different_powers", "expanded_schaffer_f6", "gallagher", "griewank",
        "griewank_rosenbrock", "katsuura", "linear_slope",
        "non_continuous_rastrigin", "rastrigin", "schwefel", "sharp_ridge",
        "sphere", "zakharov",
    }
)  # fmt: skip
_TEST_GROUPS = frozenset(
    {
        "ackley", "attractive_sector", "bent_cigar",
        "cec2005_f16", "cec2005_f17", "cec2005_f20", "cec2005_f25",
        "cec2017_f12", "cec2017_f15", "cec2017_f19", "cec2017_f20",
        "cec2017_f22", "cec2017_f23", "cec2017_f24", "cec2017_f26",
        "cec2017_f28", "cec2017_f30",
        "discus", "ellipsoid", "levy", "lunacek_bi_rastrigin", "rosenbrock",
        "schaffer_f7", "schwefel_1_2", "schwefel_2_13", "schwefel_2_6",
        "step_ellipsoid", "weierstrass",
    }
)  # fmt: skip


def _text(name: str) -> str:
    return (
        resources.files("l2co_tasks.conf.tasks")
        .joinpath(f"{name}.yaml")
        .read_text(encoding="utf-8")
    )


def _rows(name: str) -> list[tuple[str, dict]]:
    """``(suite, row)`` for every ``input_data`` row of a composite set."""
    cfg = OmegaConf.to_container(OmegaConf.create(_text(name)))
    return [
        (_SUITE_OF_FACTORY[part["data_generator"]["_target_"]], row)
        for part in cfg["parts"]
        for row in part["input_data"]
    ]


def _keys(name: str) -> list[tuple]:
    return [(suite, tuple(sorted(row.items()))) for suite, row in _rows(name)]


def test_half_holds_each_problem_of_the_half_once_at_seed_0() -> None:
    rows = _rows("joint_selection_test_half")

    assert len(rows) == 339
    assert {row["seed"] for _, row in rows} == {0}
    keys = _keys("joint_selection_test_half")
    assert len(set(keys)) == len(keys)
    counts = {suite: 0 for suite in _HALF_SUITES}
    for suite, _ in rows:
        counts[suite] += 1
    assert counts == _HALF_SUITES


def test_header_states_the_per_suite_counts() -> None:
    match = re.search(
        r"^# 339 tasks \((.*?)\)\.$",
        _text("joint_selection_test_half"),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    stated = dict(
        (suite, int(n))
        for suite, n in re.findall(r"(\w+): (\d+)", match.group(1))
    )
    assert stated == _HALF_SUITES


def test_half_contains_the_selected_test_set() -> None:
    missing = set(_keys("joint_selection_test")) - set(
        _keys("joint_selection_test_half")
    )
    assert not missing


def test_half_shares_no_base_landscape_with_the_training_set() -> None:
    def landscapes(name: str) -> set[str]:
        return {base_landscape(row["fn_name"]) for _, row in _rows(name)}

    shared = landscapes("joint_selection_test_half") & landscapes(
        "joint_selection_train"
    )
    assert not shared


def test_half_is_exactly_the_split_test_groups() -> None:
    # Guards against a row from any of the 30 training-half landscapes,
    # not only the 10 that joint_selection_train happens to contain.
    assert len(_TRAIN_GROUPS) == 30
    assert len(_TEST_GROUPS) == 28
    assert not _TRAIN_GROUPS & _TEST_GROUPS
    landscapes = {
        base_landscape(row["fn_name"])
        for _, row in _rows("joint_selection_test_half")
    }
    assert not landscapes & _TRAIN_GROUPS
    assert landscapes == _TEST_GROUPS
