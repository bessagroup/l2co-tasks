"""Tests for the base-landscape table (ADR 0006).

The table must cover every function of the analytic suites that ship as
task configs, and nothing else, so a function added to a suite cannot
slip through a function-disjoint split unmapped. The pair tests pin the
borderline calls ADR 0006 makes, so a later edit that flips one fails
here rather than silently changing which functions a split keeps apart.
"""

from __future__ import annotations

from importlib import resources

import pytest
from omegaconf import OmegaConf

from l2co_tasks import base_landscape, base_landscapes

#: The task configs whose functions the table covers. ``bbob_embedded``
#: reuses the BBOB names.
_SUITES = ("bbob", "bbob_noisy", "bbob_embedded", "cec2005", "cec2017")


def _fn_names(config: str) -> list[str]:
    path = resources.files("l2co_tasks.conf") / "tasks" / f"{config}.yaml"
    cfg = OmegaConf.load(str(path))
    return [
        str(n)
        for n in OmegaConf.select(
            cfg, "experimentdata.domain.input.fn_name.categories"
        )
    ]


@pytest.mark.parametrize("config", _SUITES)
def test_every_function_of_the_suite_is_mapped(config: str) -> None:
    missing = [n for n in _fn_names(config) if n not in base_landscapes()]
    assert not missing, f"{config}: no base landscape for {missing}"


def test_the_table_has_no_other_rows() -> None:
    configured = {n for config in _SUITES for n in _fn_names(config)}
    assert set(base_landscapes()) == configured


def test_noisy_functions_join_noiseless_groups() -> None:
    noiseless = {base_landscape(n) for n in _fn_names("bbob")}
    noisy = {base_landscape(n) for n in _fn_names("bbob_noisy")}
    assert noisy <= noiseless


def test_the_grouping_adr_0006_records() -> None:
    assert len(set(base_landscapes().values())) == 58


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # BBOB separable/rotated, conditioning-size and generator variants
        ("rastrigin_seperable", "rastrigin"),
        ("ellipsoid_seperable", "ellipsoid"),
        ("rosenbrock", "rosenbrock_rotated"),
        ("schaffer_f7_condition_10", "schaffer_f7_condition_1000"),
        ("gallagher_101_peaks", "gallagher_21_peaks"),
        # Buche-Rastrigin is Rastrigin on a skewed, oscillated input
        ("skew_rastrigin_bueche", "rastrigin"),
        # noisy versions, including the 1e4 ellipsoid against f10's 1e6
        ("bbob_noisy_f101", "sphere"),
        ("bbob_noisy_f113", "step_ellipsoid"),
        ("bbob_noisy_f116", "ellipsoid"),
        # basic CEC functions and their BBOB base
        ("f1", "sphere"),
        ("f3", "ellipsoid"),
        ("f9", "rastrigin"),
        ("f13", "griewank_rosenbrock_f8f2"),
        ("cec2017_f1", "bent_cigar"),
        ("cec2017_f5", "rastrigin"),
        ("cec2017_f6", "schaffer_f7_condition_10"),
        ("cec2017_f7", "lunacek_bi_rastrigin"),
        ("cec2017_f10", "schwefel_xsinx"),
        # CEC 2005 Schwefel 1.2 and its noisy version
        ("f2", "f4"),
    ],
)
def test_variants_share_a_base_landscape(a: str, b: str) -> None:
    assert base_landscape(a) == base_landscape(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # a conditioning size changes, but none is added to the sphere
        ("sphere", "ellipsoid"),
        # different eigenvalue shapes
        ("ellipsoid", "discuss"),
        ("ellipsoid", "bent_cigar"),
        ("ellipsoid", "sharp_ridge"),
        # rounding makes plateaus
        ("step_ellipsoid", "ellipsoid"),
        ("cec2017_f8", "rastrigin"),
        # different formulas
        ("lunacek_bi_rastrigin", "rastrigin"),
        ("f14", "schaffer_f7_condition_10"),
        ("f2", "ellipsoid"),
        # a composition of hybrids is not those hybrids
        ("cec2017_f29", "cec2017_f15"),
    ],
)
def test_distinct_landscapes_stay_apart(a: str, b: str) -> None:
    assert base_landscape(a) != base_landscape(b)


def test_hybrids_and_compositions_are_their_own_groups() -> None:
    names = [f"f{i}" for i in range(15, 26)] + [
        f"cec2017_f{i}" for i in range(11, 31)
    ]
    groups = [base_landscape(n) for n in names]
    assert len(set(groups)) == len(names)
    owners = {g: n for n, g in base_landscapes().items() if g in groups}
    assert set(owners.values()) == set(names)


def test_unknown_name_raises_with_the_adr() -> None:
    with pytest.raises(KeyError, match="ADR 0006"):
        base_landscape("not_a_function")


def test_the_mapping_is_a_copy() -> None:
    mapping = base_landscapes()
    mapping["sphere"] = "changed"
    assert base_landscape("sphere") == "sphere"
