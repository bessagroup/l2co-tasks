"""
Which benchmark functions share a base landscape (ADR 0007).

A function-disjoint train/test split is only disjoint if it keeps variants
of one landscape on the same side. Names do not ensure that: all 30
BBOB-noisy functions are noisy versions of eight noiseless BBOB functions,
the basic CEC 2005 and CEC 2017 functions are shifted and rotated versions
of the sphere, Rosenbrock, Rastrigin and their kin, and BBOB itself ships
separable and rotated versions of the same function. A table that ships
with the package assigns each function of the analytic suites to a base
landscape, so a split can work on groups instead of names.

The rule (ADR 0007): two functions share a base landscape when they
evaluate the same defining formula and differ only by shift, rotation, the
size of an existing conditioning, BBOB's oscillation and asymmetry maps,
a generator parameter, noise, or embedding. Rounded variants (step
ellipsoid, non-continuous Rastrigin) and different formulas stay apart.
CEC hybrid and composition functions each form their own group.

The table is keyed by ``fn_name``, the task tag every suite sets and which
is unique across ``bbob``, ``bbob_noisy``, ``cec2005`` and ``cec2017``.
``bbob_embedded`` tasks carry the BBOB names, so an embedded function
shares the group of the BBOB function it embeds.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import csv
import functools
from pathlib import Path

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "base_landscape",
    "base_landscapes",
]

#: The committed table: one row per function, ``fn_name,suite,
#: base_landscape``.
_TABLE_PATH = Path(__file__).with_name("base_landscapes.csv")


@functools.cache
def _table() -> dict[str, str]:
    with _TABLE_PATH.open(newline="") as fh:
        return {
            row["fn_name"]: row["base_landscape"] for row in csv.DictReader(fh)
        }


def base_landscapes() -> dict[str, str]:
    """Every mapped function's base landscape.

    Returns
    -------
    dict[str, str]
        ``fn_name -> base_landscape`` for the analytic suites ``bbob``
        (whose names ``bbob_embedded`` shares), ``bbob_noisy``,
        ``cec2005`` and ``cec2017``. A fresh copy: changing it does not
        change the table.
    """
    return dict(_table())


def base_landscape(fn_name: str) -> str:
    """The base landscape a function belongs to.

    Parameters
    ----------
    fn_name : str
        The task's ``fn_name`` tag, e.g. ``"rastrigin_seperable"``,
        ``"bbob_noisy_f101"``, ``"f9"`` (CEC 2005) or ``"cec2017_f5"``.

    Returns
    -------
    str
        The group name, e.g. ``"rastrigin"``. Hybrid and composition
        functions are their own group, named after the suite and
        function (``"cec2017_f21"``).

    Raises
    ------
    KeyError
        ``fn_name`` is not in the table, which covers only the analytic
        suites listed in :func:`base_landscapes`.
    """
    try:
        return _table()[fn_name]
    except KeyError:
        raise KeyError(
            f"No base landscape for fn_name {fn_name!r}; the table covers "
            f"the bbob (and bbob_embedded), bbob_noisy, cec2005 and cec2017 "
            f"functions (ADR 0007)."
        ) from None
