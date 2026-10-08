"""Sizes and recorded optima read from CUTEst SIF files (ADR 0004).

The first group writes small SIF files into a temporary ``MASTSIF`` and
runs everywhere, CI included. Each fixture mirrors a pattern found in
the real problem files. The second group reads the real MASTSIF and
skips without one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from l2co_tasks import create_cutest_task, cutest_sizes, cutest_soltn


@pytest.fixture
def sif(tmp_path, monkeypatch):
    """Write a SIF file into a temporary MASTSIF; returns the writer."""
    monkeypatch.setenv("MASTSIF", str(tmp_path))

    def write(problem: str, text: str) -> str:
        (tmp_path / f"{problem}.SIF").write_text(text)
        return problem

    return write


# Sizes -----------------------------------------------------------------------


def test_fixed_size_problem_has_one_size(sif):
    problem = sif("FIXED", "*LO SOLTN               0.0\n")
    assert cutest_sizes(problem) == [{}]


def test_one_size_parameter_lists_each_value_once_in_file_order(sif):
    problem = sif(
        "ONE",
        """\
*IE N                   10             $-PARAMETER
*IE N                   50             $-PARAMETER
 IE N                   100            $-PARAMETER     original value
*IE N                   10             $-PARAMETER     repeated
 RE ZETA                20.0           $-PARAMETER
*RE ZETA                2.0            $-PARAMETER
 IE LB                  5              $-PARAMETER
""",
    )
    # ZETA is real and LB lists one value: neither is varied.
    assert cutest_sizes(problem) == [{"N": 10}, {"N": 50}, {"N": 100}]


def test_equal_length_lists_are_paired(sif):
    problem = sif(
        "PAIRED",
        """\
*IE N                   10             $-PARAMETER
 IE N                   50             $-PARAMETER
*IE M                   20             $-PARAMETER
 IE M                   100            $-PARAMETER
""",
    )
    assert cutest_sizes(problem) == [{"N": 10, "M": 20}, {"N": 50, "M": 100}]


def test_unequal_lists_vary_the_longest_only(sif):
    problem = sif(
        "UNEQUAL",
        """\
*IE N                   19             $-PARAMETER
*IE N                   49             $-PARAMETER
 IE N                   99             $-PARAMETER
*IE M                   4              $-PARAMETER
 IE M                   6              $-PARAMETER
""",
    )
    assert cutest_sizes(problem) == [{"N": 19}, {"N": 49}, {"N": 99}]


def test_parameter_names_are_taken_as_written(sif):
    problem = sif(
        "SLASH",
        """\
*IE N/2                 5              $-PARAMETER
 IE N/2                 50             $-PARAMETER
""",
    )
    assert cutest_sizes(problem) == [{"N/2": 5}, {"N/2": 50}]


# Recorded optima -------------------------------------------------------------

PER_SIZE = """\
*IE N                   4              $-PARAMETER
 IE N                   10             $-PARAMETER
*IE N                   50             $-PARAMETER
*LO SOLTN(4)            2.24997D-4
*LO SOLTN(10)           7.08765D-5
*LO SOLTN(10)           9.0D-5
*LO SOLTN(50)           ???
*LO SOLTN               3.0
"""


@pytest.mark.parametrize(
    ("sif_params", "expected"),
    [
        ({"N": 4}, 2.24997e-4),  # Fortran exponent
        ({"N": 10}, 7.08765e-5),  # two values for one size: lowest
        ({"N": 50}, 3.0),  # ??? for this size: the un-sized value
        (None, 7.08765e-5),  # default size, N = 10
    ],
)
def test_per_size_value_wins_and_falls_back(sif, sif_params, expected):
    problem = sif("PERSIZE", PER_SIZE)
    assert cutest_soltn(problem, sif_params) == pytest.approx(expected)


def test_unsized_value_counts_for_every_size(sif):
    problem = sif(
        "UNSIZED",
        """\
*IE N                   10             $-PARAMETER
 IE N                   50             $-PARAMETER
*LO SOLTN               1.0
""",
    )
    assert cutest_soltn(problem, {"N": 10}) == 1.0
    assert cutest_soltn(problem, {"N": 50}) == 1.0


def test_keys_refer_to_the_parameter_listing_them(sif):
    """ARGLIN-style: SOLTN keys are N's values, not M's."""
    problem = sif(
        "PAIREDSOL",
        """\
*IE N                   10             $-PARAMETER
 IE N                   50             $-PARAMETER
*IE M                   20             $-PARAMETER
 IE M                   100            $-PARAMETER
*LO SOLTN(10)           5.0
*LO SOLTN(50)           25.0
""",
    )
    assert cutest_soltn(problem, {"N": 10, "M": 20}) == 5.0
    assert cutest_soltn(problem, {"N": 50, "M": 100}) == 25.0


def test_keys_that_are_not_listed_values_are_ignored(sif):
    """CRAGGLVY-style: key 2 matches no listed M (1, 4, 24)."""
    problem = sif(
        "TYPO",
        """\
*IE M                   1              $-PARAMETER
 IE M                   4              $-PARAMETER
*IE M                   24             $-PARAMETER
*LO SOLTN(2)            0.0
*LO SOLTN(4)            1.886
*LO SOLTN(24)           15.37
""",
    )
    assert cutest_soltn(problem, {"M": 4}) == pytest.approx(1.886)
    assert cutest_soltn(problem, {"M": 1}) is None
    assert cutest_soltn(problem, {"M": 2}) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("*LO SOLTN\n*LO SOLTN   ???\n*LO SOLTN   -2.0D+00\n", -2.0),
        ("*LO SOLTN\n*LO SOLTN   ???\n", None),
        ("* no recorded optimum\n", None),
    ],
    ids=["blank-and-unknown-skipped", "only-unknown", "none"],
)
def test_unknown_and_blank_values(sif, text, expected):
    assert cutest_soltn(sif("FORMATS", text)) == expected


# Errors ----------------------------------------------------------------------


def test_missing_mastsif_explains_the_setup(monkeypatch):
    monkeypatch.delenv("MASTSIF", raising=False)
    with pytest.raises(ImportError, match="MASTSIF is not set"):
        cutest_sizes("ROSENBR")


def test_unknown_problem_is_reported(sif):
    with pytest.raises(ValueError, match="no SIF file"):
        cutest_soltn("NOSUCHPROBLEM")


# The real MASTSIF ------------------------------------------------------------

_MASTSIF = os.environ.get("MASTSIF")

needs_mastsif = pytest.mark.skipif(
    not (_MASTSIF and Path(_MASTSIF).is_dir()),
    reason="needs the MASTSIF problem files (MASTSIF set)",
)


def _cutest_available() -> bool:
    try:
        import pycutest  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


needs_cutest = pytest.mark.skipif(
    not _cutest_available(),
    reason="needs pycutest and a CUTEst installation",
)


@needs_mastsif
def test_real_problem_files():
    assert cutest_sizes("ROSENBR") == [{}]
    assert cutest_soltn("ROSENBR") == 0.0
    assert cutest_sizes("ARGLINA")[:2] == [
        {"N": 10, "M": 20},
        {"N": 50, "M": 100},
    ]
    assert {tuple(size) for size in cutest_sizes("VAREIGVL")} == {("N",)}
    assert cutest_soltn("PENALTY1", {"N": 10}) == pytest.approx(7.08765e-5)
    assert cutest_soltn("CRAGGLVY", {"M": 1}) is None
    assert cutest_soltn("CRAGGLVY", {"M": 4}) is not None
    assert cutest_sizes("BROYDN7D")[0] == {"N/2": 5}


@needs_cutest
def test_every_unconstrained_problem_parses():
    import pycutest

    for problem in pycutest.find_problems(constraints="unconstrained"):
        for size in cutest_sizes(problem):
            soltn = cutest_soltn(problem, size)
            assert soltn is None or isinstance(soltn, float)


@needs_cutest
def test_listed_size_builds_a_task():
    task = create_cutest_task("BROYDN7D", {"N/2": 5}, global_min=0.0)
    assert task.dimensionality == 10
    assert task.tag["sif_params"] == "N/2=5"
