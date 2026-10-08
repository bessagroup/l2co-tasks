"""
What a CUTEst problem's SIF file says about its sizes and its optimum.

Two facts the CUTEst task set needs (ADR 0004) live only in comments and
parameter lines of the SIF file, not in pycutest:

* **Which sizes a problem offers.** A variable-size problem lists the
  values of its size parameters as ``$-PARAMETER`` lines, one per value,
  with the default left uncommented::

      *IE N                   10             $-PARAMETER
       IE N                   100            $-PARAMETER

* **Its recorded optimum**, ``SOLTN``, written for every size or per size
  (keyed by the size parameter's value), sometimes with a Fortran
  exponent, sometimes unknown::

      *LO SOLTN               0.0
      *LO SOLTN(10)           7.08765D-5
      *LO SOLTN(50)           ???

Both are read here with no pycutest and no compilation; only ``MASTSIF``
must point at the problem files. The rules, settled against every
unconstrained problem in MASTSIF, are in :func:`cutest_sizes` and
:func:`cutest_soltn`.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from pathlib import Path

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "cutest_sizes",
    "cutest_soltn",
]

INSTALL_HINT = (
    "install the extra (`pip install 'l2co-tasks[cutest]'`) and a CUTEst "
    "build (SIFDecode, CUTEst and the MASTSIF problem files), and set "
    "CUTEST, SIFDECODE and MASTSIF; see ADR 0004"
)

#: A parameter line: ``[*] IE|RE <name> <value> $-PARAMETER ...``. A
#: leading ``*`` comments the line out; the uncommented one is the
#: default.
_PARAMETER = re.compile(
    r"^(?P<commented>\*?)\s*(?P<kind>IE|RE)\s+(?P<name>\S+)\s+"
    r"(?P<value>\S+)\s+\$-PARAMETER"
)

#: A recorded optimum: ``*LO SOLTN[(key)] [value]``.
_SOLTN = re.compile(
    r"^\*\s*LO\s+SOLTN(?:\((?P<key>\d+)\))?(?:\s+(?P<value>\S+))?"
)


def sif_path(problem: str) -> Path:
    """The problem's SIF file in ``$MASTSIF``.

    Raises
    ------
    ImportError
        If ``MASTSIF`` is not set.
    ValueError
        If there is no SIF file for ``problem``.
    """
    mastsif = os.environ.get("MASTSIF")
    if not mastsif:
        raise ImportError(f"MASTSIF is not set: {INSTALL_HINT}")
    path = Path(mastsif) / f"{problem}.SIF"
    if not path.is_file():
        raise ValueError(f"no SIF file for CUTEst problem {problem!r}: {path}")
    return path


def _lines(problem: str) -> list[str]:
    return sif_path(problem).read_text(errors="replace").splitlines()


def _integer_parameters(
    lines: list[str],
) -> tuple[dict[str, list[int]], dict[str, int]]:
    """Listed values (file order, no repeats) and defaults, per integer
    parameter. The default is the last uncommented listing."""
    listed: dict[str, list[int]] = {}
    defaults: dict[str, int] = {}
    for line in lines:
        match = _PARAMETER.match(line)
        if match is None or match["kind"] != "IE":
            continue
        try:
            value = int(match["value"])
        except ValueError:
            continue
        values = listed.setdefault(match["name"], [])
        if value not in values:
            values.append(value)
        if not match["commented"]:
            defaults[match["name"]] = value
    return listed, defaults


def _size_parameters(listed: dict[str, list[int]]) -> dict[str, list[int]]:
    """The integer parameters that list more than one value."""
    return {name: values for name, values in listed.items() if len(values) > 1}


def cutest_sizes(problem: str) -> list[dict[str, int]]:
    """The sizes a CUTEst problem lists, as SIF-parameter dicts.

    A **size parameter** is an integer parameter whose ``$-PARAMETER``
    lines list more than one value. Real parameters (``RE``) and integer
    parameters listed once are never varied: they stay at the problem's
    default. Values keep their order in the file, without repeats.

    * No size parameter: one size, ``[{}]``, the problem as written.
    * One size parameter: one size per listed value.
    * Several, with lists of equal length: paired by position, as
      ARGLINA lists ``N`` and ``M`` (``N=10`` with ``M=20``, ...).
    * Several, with lists of different lengths: the parameter with the
      most values varies (the first in the file on a tie), and the others
      stay at their defaults, as VAREIGVL varies ``N`` and keeps ``M``.

    The task set also caps the number of variables at 100 (ADR 0004).
    That cap needs the compiled problem, so it is not applied here.

    Parameters
    ----------
    problem : str
        CUTEst problem name.

    Returns
    -------
    list[dict[str, int]]
        One dict per size, ready for
        :func:`~l2co_tasks.create_cutest_task`'s ``sif_params``.
    """
    listed, _ = _integer_parameters(_lines(problem))
    varied = _size_parameters(listed)
    if not varied:
        return [{}]
    if len({len(values) for values in varied.values()}) == 1:
        paired = zip(*varied.values(), strict=True)
        return [dict(zip(varied, combo, strict=True)) for combo in paired]
    name = max(varied, key=lambda n: len(varied[n]))
    return [{name: value} for value in varied[name]]


def _parse_value(text: str | None) -> float | None:
    """A ``SOLTN`` value; ``None`` for ``???``, a blank or a non-number."""
    if not text:
        return None
    try:
        value = float(text.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def cutest_soltn(
    problem: str, sif_params: Mapping[str, int | float] | None = None
) -> float | None:
    """The optimum a CUTEst problem's SIF file records for one size.

    The rules:

    * A per-size value, ``SOLTN(k)``, is keyed by the value of a size
      parameter (see :func:`cutest_sizes`): the one whose listed values
      contain the most keys. A key that is not one of its listed values
      is ignored.
    * A per-size value for the requested size wins. Otherwise a value
      written without a size counts, for every size.
    * Several values for the same size give the lowest. ``???``, a blank
      or anything that isn't a number counts as no value.

    In the task set, ``global_min`` is the lower of this and the estimate
    (ADR 0004), so a recorded value that is wrong for some size can only
    lower ``global_min``, never put it above the true minimum.

    Parameters
    ----------
    problem : str
        CUTEst problem name.
    sif_params : Mapping[str, int | float] or None, optional
        The size, as passed to :func:`~l2co_tasks.create_cutest_task`.
        A size parameter left out takes its default.

    Returns
    -------
    float or None
        The recorded optimum, or ``None`` if the file records none for
        this size.
    """
    lines = _lines(problem)
    unsized: list[float] = []
    sized: dict[int, list[float]] = {}
    for line in lines:
        match = _SOLTN.match(line)
        if match is None:
            continue
        value = _parse_value(match["value"])
        if value is None:
            continue
        if match["key"] is None:
            unsized.append(value)
        else:
            sized.setdefault(int(match["key"]), []).append(value)

    if sized:
        listed, defaults = _integer_parameters(lines)
        varied = _size_parameters(listed)
        keyed_by = max(
            varied,
            key=lambda name: sum(key in varied[name] for key in sized),
            default=None,
        )
        if keyed_by is not None:
            size = dict(sif_params or {}).get(keyed_by, defaults.get(keyed_by))
            if size in varied[keyed_by] and size in sized:
                return min(sized[size])
    return min(unsized) if unsized else None
