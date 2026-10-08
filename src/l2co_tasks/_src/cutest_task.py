"""
CUTEst problems as tasks, through pycutest (ADR 0004).

CUTEst is the standard collection of nonlinear-optimization test
problems, written in SIF (a modelling format) and compiled to Fortran.
pycutest compiles a problem on first use and evaluates it, value and
gradient, in the compiled code. :func:`create_cutest_task` turns one
(problem, size) into a :class:`~l2co_tasks.Task` whose loss is a
:func:`~l2co_tasks.host_loss` (ADR 0003), so every optimizer runs on it.

What a CUTEst task looks like:

* **Its model is the problem's prescribed start** ``x0``, in the problem's
  own coordinates. This deliberately breaks the package convention of a
  ``[0, 1]^d`` model rescaled inside the loss: the start is part of the
  problem, and CUTEst variables differ in scale by orders of magnitude.
* **Unconstrained problems only.** l2co passes one ``(lo, hi)`` pair to
  every optimizer (l2co ADR 0020), and CUTEst's bounds are per variable.
* **``global_min`` comes from a committed table**, one row per (problem,
  size), so a task's identity is the same on every machine. A row also
  records the hash of the problem's SIF file: a problem corrected upstream
  no longer matches its row, and building it refuses until the table is
  regenerated. Passing ``global_min`` explicitly bypasses the table; that
  is how the table itself is built.
* **The tag is flat**, with the SIF parameters as one canonical string
  (``"N=10"``). A dict inside a tag would not hash the same in every
  process.

pycutest is GPL and needs a system CUTEst, so it is the optional
``[cutest]`` extra and is imported only when a CUTEst task is built or
evaluated. Even ``import pycutest`` fails until ``CUTEST`` is set.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import csv
import functools
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Third-party
import jax.numpy as jnp
import numpy as np

# Local
from .cutest_sif import INSTALL_HINT, sif_path
from .host_objective import host_loss
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "create_cutest_task",
]

#: The committed ``global_min`` table (ADR 0004).
_TABLE_PATH = Path(__file__).with_name("cutest_global_min.csv")

#: Its columns, in order. ``sif_params`` is the canonical string
#: (:func:`_sif_params_str`); ``source`` is ``"estimate"`` or ``"soltn"``;
#: ``excluded`` is empty, or the reason the problem is left out, in which
#: case ``global_min`` is empty.
_TABLE_COLUMNS = (
    "problem",
    "sif_params",
    "n",
    "sif_sha256",
    "global_min",
    "source",
    "excluded",
)

#: pycutest's value for "no bound".
_NO_BOUND = 1e20

SifParams = tuple[tuple[str, int | float], ...]


def _pycutest() -> Any:
    """Import pycutest, or explain what is missing."""
    try:
        import pycutest
    except (ImportError, RuntimeError) as error:
        raise ImportError(
            "CUTEst tasks need pycutest and a CUTEst installation: "
            + INSTALL_HINT
        ) from error
    return pycutest


def _canonical_sif_params(
    sif_params: Mapping[str, int | float] | None,
) -> SifParams:
    """``sif_params`` as a sorted, hashable tuple of pairs."""
    return tuple(sorted((sif_params or {}).items()))


def _sif_params_str(sif_params: SifParams) -> str:
    """The canonical string form, e.g. ``"N=10"``; empty for none."""
    return ",".join(f"{name}={value}" for name, value in sif_params)


def _sif_sha256(problem: str) -> str:
    """sha256 of the problem's SIF file in ``$MASTSIF``."""
    return hashlib.sha256(sif_path(problem).read_bytes()).hexdigest()


@dataclass(frozen=True)
class _TableRow:
    """One row of the ``global_min`` table; see :data:`_TABLE_COLUMNS`."""

    n: int
    sif_sha256: str
    global_min: float | None
    source: str
    excluded: str


@functools.cache
def _read_table(path: Path) -> dict[tuple[str, str], _TableRow]:
    """The table at ``path``, keyed by ``(problem, sif_params)``."""
    rows: dict[tuple[str, str], _TableRow] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != _TABLE_COLUMNS:
            raise ValueError(
                f"{path} has columns {reader.fieldnames}, expected "
                f"{list(_TABLE_COLUMNS)}"
            )
        for record in reader:
            key = (record["problem"], record["sif_params"])
            if key in rows:
                raise ValueError(f"{path} lists {key} twice")
            rows[key] = _TableRow(
                n=int(record["n"]),
                sif_sha256=record["sif_sha256"],
                global_min=(
                    float(record["global_min"])
                    if record["global_min"]
                    else None
                ),
                source=record["source"],
                excluded=record["excluded"],
            )
    return rows


def _table_global_min(problem: str, params_str: str, sif_sha256: str) -> float:
    """``global_min`` for one (problem, size) from the committed table.

    Raises
    ------
    ValueError
        If the table has no row for it, the row records an exclusion, or
        the problem's SIF file no longer matches the row.
    """
    label = f"{problem}({params_str})" if params_str else problem
    row = _read_table(_TABLE_PATH).get((problem, params_str))
    if row is None:
        raise ValueError(
            f"CUTEst problem {label} is not in the global_min table "
            f"({_TABLE_PATH.name}). The table is built by the "
            "cutest_global_min_table experiment in l2co_experiments; to "
            "build a task outside it, pass global_min explicitly (ADR 0004)."
        )
    if row.excluded:
        raise ValueError(
            f"CUTEst problem {label} is excluded from the task set: "
            f"{row.excluded}"
        )
    if row.sif_sha256 != sif_sha256:
        raise ValueError(
            f"CUTEst problem {label} has changed since the global_min table "
            "was built (its SIF file no longer matches the recorded hash). "
            "Regenerate the table (ADR 0004)."
        )
    if row.global_min is None:
        raise ValueError(f"the global_min table has no value for {label}")
    return row.global_min


class _CutestObjective:
    """A compiled CUTEst problem as a host objective (ADR 0003)."""

    def __init__(self, problem: Any):
        self._problem = problem

    def value(self, x: np.ndarray) -> float:
        return float(self._problem.obj(x))

    def value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        f, g = self._problem.obj(x, gradient=True)
        return float(f), np.asarray(g, dtype=np.float64)


@dataclass(frozen=True)
class _CutestOpener:
    """Opens one CUTEst (problem, size); picklable and hashable.

    Attributes
    ----------
    problem : str
        CUTEst problem name.
    sif_params : SifParams
        SIF parameters as a sorted tuple of ``(name, value)`` pairs.
    """

    problem: str
    sif_params: SifParams = ()

    def __call__(self) -> _CutestObjective:
        problem = _pycutest().import_problem(
            self.problem, sifParams=dict(self.sif_params) or None
        )
        return _CutestObjective(problem)


def create_cutest_task(
    problem: str,
    sif_params: Mapping[str, int | float] | None = None,
    *,
    global_min: float | None = None,
) -> Task:
    """Create a task from an unconstrained CUTEst problem (ADR 0004).

    Needs the ``[cutest]`` extra and a CUTEst installation (``CUTEST``,
    ``SIFDECODE`` and ``MASTSIF`` set). The problem is compiled on first
    use, into ``PYCUTEST_CACHE``.

    Parameters
    ----------
    problem : str
        CUTEst problem name, e.g. ``"ROSENBR"``.
    sif_params : Mapping[str, int | float] or None, optional
        SIF parameters selecting the size, e.g. ``{"N": 10}``; ``None``
        (the default) uses the problem's default. Use the values the SIF
        file lists, with integers for integer parameters: the table is
        keyed by the canonical string, so ``{"N": 10.0}`` is not
        ``{"N": 10}``.
    global_min : float or None, optional
        The task's ``global_min``. ``None`` (the default) reads it from the
        committed table; pass a value to build a task outside the table,
        as the experiment that builds the table does.

    Returns
    -------
    Task
        A task whose model is the problem's prescribed start ``x0``, in
        the problem's own coordinates, and whose loss is a host loss over
        the compiled problem. ``pass_rng`` is ``False`` and there is no
        dataset. The tag holds ``task_name`` (``"cutest"``), ``fn_name``
        (the problem), ``dimensionality``, ``noise`` (``0.0``),
        ``sif_params`` (canonical string), ``sif_sha256`` and the
        classification fields ``cutest_objective``, ``cutest_regular``,
        ``cutest_degree``, ``cutest_origin`` and ``cutest_internal``.

    Raises
    ------
    ImportError
        If pycutest or a CUTEst installation is missing.
    ValueError
        If the problem has constraints or bounds, integer or boolean
        variables, no SIF file, or -- when ``global_min`` is not given --
        no usable row in the table: missing, excluded, or built from a SIF
        file that has since changed.
    """
    pycutest = _pycutest()
    properties = pycutest.problem_properties(problem)
    if properties["constraints"] != "unconstrained":
        raise ValueError(
            f"CUTEst problem {problem!r} has {properties['constraints']} "
            "constraints. Only unconstrained problems are supported: "
            "optimizers take one (lo, hi) box for every parameter (l2co "
            "ADR 0020, ADR 0004)."
        )
    params = _canonical_sif_params(sif_params)
    params_str = _sif_params_str(params)
    sif_sha256 = _sif_sha256(problem)
    if global_min is None:
        global_min = _table_global_min(problem, params_str, sif_sha256)
    elif not math.isfinite(global_min):
        raise ValueError(f"global_min must be finite, got {global_min}")

    compiled = pycutest.import_problem(problem, sifParams=dict(params) or None)
    if np.any(np.asarray(compiled.vartype) != 0):
        raise ValueError(
            f"CUTEst problem {problem!r} has integer or boolean variables"
        )
    if np.any(compiled.bl > -_NO_BOUND) or np.any(compiled.bu < _NO_BOUND):
        raise ValueError(
            f"CUTEst problem {problem!r} is classified unconstrained but "
            "declares bounds; only unconstrained problems are supported "
            "(l2co ADR 0020)"
        )

    tag = {
        "task_name": "cutest",
        "fn_name": problem,
        "dimensionality": int(compiled.n),
        "noise": 0.0,
        "sif_params": params_str,
        "sif_sha256": sif_sha256,
        "cutest_objective": properties["objective"],
        "cutest_regular": properties["regular"],
        "cutest_degree": properties["degree"],
        "cutest_origin": properties["origin"],
        "cutest_internal": properties["internal"],
    }
    return Task(
        model=jnp.asarray(compiled.x0),
        loss_fn=host_loss(_CutestOpener(problem, params)),
        global_min=float(global_min),
        tag=tag,
    )
