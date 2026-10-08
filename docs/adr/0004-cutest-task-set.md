---
status: proposed
---

# CUTEst problems as a task set

CUTEst is the standard collection of nonlinear-optimization test
problems: about 1,500 problems written in SIF (a modelling format) and
compiled to Fortran. Engineering-design studies use it, and our databank
has none of it. pycutest is the Python interface: it compiles each
problem on first use and returns values and gradients from the Fortran
code. JAX can neither trace nor differentiate it.

**Decision.** CUTEst problems become a task set, built with
`create_cutest_task(name, sif_params)`. They run through the host-objective
module (ADR 0003), so every optimizer in the registry runs on them. The
rest of this ADR fixes which problems go in and how each becomes a
`Task`.

> [!IMPORTANT]
> This deliberately breaks the package convention that a task's model
> lives in `[0, 1]^d` and the loss rescales it (`scale_input`). A CUTEst
> task's model is the problem's own `x0`, in the problem's own
> coordinates. The prescribed start is part of the problem, and CUTEst
> variables differ in scale by orders of magnitude, so no single
> rescaling fits.

## Packaging

- **The adapter lives in l2co-tasks, behind a `[cutest]` extra.**
  pycutest is GPL and l2co-tasks is BSD-3. An optional extra that is
  imported lazily is the usual way for a BSD package to offer a GPL
  backend. A pip install is not enough anyway: pycutest also needs a
  system CUTEst (SIFDecode, CUTEst and the MASTSIF problem files, built
  with gfortran) and the `CUTEST`, `SIFDECODE` and `MASTSIF` environment
  variables.
- **The import must be lazy.** `import pycutest` itself raises
  `RuntimeError` when `CUTEST` isn't set. The adapter imports it inside
  its opener, and tests that need CUTEst skip on `ImportError` or
  `RuntimeError`. They always skip in CI, which has no CUTEst.
- **l2co-optimizers ADR 0003 rejected an extra for casadi.** There, a
  missing extra would surface as a failed databank cell. Here it
  surfaces while the task is built, in the create stage, before any cell
  runs, so that argument doesn't carry over.

## Which problems

- **Unconstrained problems only.** l2co passes one `(lo, hi)` pair to
  every optimizer (l2co ADR 0020), and CUTEst's bounds are per variable,
  often finite on only some variables. Building a task for a
  bound-constrained problem raises an error that cites l2co ADR 0020.
- **Excluded:** problems with integer or boolean variables, and problems
  unbounded below.
- **Both smoothness classes are included** (regular `R` and irregular
  `I`). Today this changes nothing: pycutest classifies all 293
  unconstrained problems in MASTSIF as regular. It will matter once
  bound-constrained problems come in.
- **Sizes:** only the sizes a problem lists (its `$-PARAMETER` lines),
  and only where the resulting `n` is at most 100. Fixed-size problems
  are included when their `n` is at most 100. The cap is checked against
  pycutest's actual `n`, because the size parameter isn't always `n`
  (DIXMAANA1 sets `M`, and n = 3M). Unlisted values are never used:
  SIFDecode often accepts them, but CUTEst never validated those
  instances.

Census of MASTSIF at `29adac9` (2026-06-01): 293 unconstrained problems.
133 have a variable size, and 147 have a fixed size with n ≤ 100.

## Where a run starts

The task's `model` is `x0`. A databank experiment over this set uses the
`relative_normal` sampler (l2co-optimizers ADR 0004): member 0 is
exactly `x0`, and the other members are spread relative to each
coordinate's size.

## `global_min`

ADR 0001 requires `global_min` to be a **floor**, and it is part of the
task's identity. pycutest exposes no optimal values. Many SIF files
record one in a comment, as `*LO SOLTN 1.2701`, `*LO SOLTN(10)
7.08765D-5` (per size, with a Fortran exponent) or `*LO SOLTN(50) ???`
(unknown). These values are often rounded, and some are local optima.
Of the 293 problems, 222 have a numeric value, 6 have only `???`, and 65
have none.

- **Where a numeric value exists for that size,** `global_min` is the
  lower of the estimate and `SOLTN`. A value rounded up would break the
  floor; one rounded down would silently cap the target precision. The
  estimate, at full float64 precision, covers both. **Otherwise**,
  `global_min` is the estimate.
- **The estimate always runs in x64,** whatever the process setting.
  bbob's and cec2005's ids already differ between processes with and
  without x64. The new set shouldn't inherit that.
- **Restarts start where the optimizers start:** one from `x0` and the
  rest from the relative sampler. `estimate_global_min` gains a
  generic `restart_sampler` parameter, passed in by the caller. Existing
  tasks keep their N(0, I) restarts, so their values and hashes don't
  change.
- **Floor-test-guided estimation (ADR 0001) runs before the first
  databank run.** A problem the floor test still breaches at the
  estimator's maximum strength is excluded, with the reason recorded.
  The estimator stays standalone (optax only, ADR 0001). If the
  exclusion list turns out long, adding a derivative-free arm becomes a
  separate decision.

## A committed table

The estimate runs thousands of optimizer steps, so a last-bit difference
between machines (CPU, XLA, gfortran) can change its final digits, and
with them the hash. Two create stages on different clusters would then
file one problem under two ids. So the values are computed once, offline,
and committed.

- **One row per (problem, size)** in a data file in l2co-tasks:
  - the SIF hash;
  - `global_min`;
  - its source (`estimate` or `soltn`);
  - the exclusion reason, if any.

  The table *is* the task set's problem list.
- **Building a task is a lookup.** If the problem's SIF file no longer
  matches its row's hash, building refuses and says the table is stale.
- **A new l2co_experiments experiment regenerates the table** under
  sbatch. It parses `SOLTN`, runs the estimate, runs the floor test with
  the whole `all` suite, and writes the table. The table reaches
  l2co-tasks in a PR. It lives in l2co_experiments because it needs l2co,
  for the suite, and l2co-optimizers, for the sampler, and l2co-tasks may
  import neither.

## Identity

The tag holds:
- the keys downstream code reads: `task_name: "cutest"`, `fn_name` (the
  problem), `dimensionality` (`n`) and `noise`;
- the SIF parameters;
- the classification fields (objective type, smoothness, origin), which
  also serve as task features;
- a **hash of the problem's SIF file contents**.

MASTSIF corrects problems over time while keeping their names. Without
the file hash, a corrected problem whose `global_min` didn't move (often
0.0) would keep its id, and old and new data would mix silently. With
it, exactly the changed problems get new ids.

## Compiling

pycutest compiles each (problem, size) on first import, into
`PYCUTEST_CACHE`.

- A **pre-build step** runs first in every CUTEst experiment chain. It
  compiles every row of the table into a cache on scratch, skipping
  problems already compiled, so a cache purged after 30 days simply
  rebuilds.
- Databank cells never compile, so a missing problem fails at once
  instead of many array jobs racing to compile into one folder. Ad-hoc
  use compiles on demand.

## What the spike established

Measured on Oscar on 2026-10-08. The scripts are in
`/oscar/scratch/mvander7/host_task_spike/`, which scratch purges after
30 days.

- **The build works.** SIFDecode and CUTEst build with meson (installed
  through uv; it isn't on Oscar) and gfortran 11.5 from `/usr/bin`, in
  double precision (`-Dmodules=false`). Each problem compiles in about
  2 s and evaluates (value and gradient) in about 3 µs at n = 10.
- **Two sizes of one problem coexist in one process.** Each size
  compiles to its own module, and interleaved calls stay consistent.
  Importing the same problem again returns the same instance.
- **Calls are thread-safe in practice.** 8 threads × 4,000 calls gave 0
  mismatches against sequential calls. pycutest seems to hold Python's
  global interpreter lock.
- **All 19 entries tested run on a CUTEst task**, covering every
  optimizer family. An IPOPT cell at the size cap (25,000 evaluations ×
  25 realizations) takes about 9 minutes.

## Not covered

- Bound-constrained and constrained problems.
- Sizes above 100. They can come later, as a separate task set.
- Meta-optimizers on CUTEst. How l2co's feature processing encodes the
  CUTEst tag isn't decided.
- Single or quadruple precision CUTEst.

## Considered options

- **A separate package** (`l2co-tasks-cutest`). It would keep GPL code
  out of the l2co-tasks dependency tree, at the cost of another repo with
  its own CI, releases and lockfile.
- **`global_min` from the estimate only,** with `SOLTN` as a test
  cross-check. Rejected in favour of taking the lower of the two.
- **`SOLTN` as printed,** or minus half a unit in the last digit.
  Rejected: the first breaks the floor or silently caps target
  precision; the second caps precision at about `5e-5` on every rounded
  problem.
- **Estimating when the task is built,** as the other empirical tasks
  do. Rejected: ids would depend on the machine.
- **The MASTSIF commit in the tag.** Rejected: every MASTSIF update would
  re-key every problem, changed or not.
- **Sizes forced onto bbob's {2, 3, 5, 10, 20, 40}.** Rejected: CUTEst
  never validated those instances.
- **Compiling on demand under a file lock.** Rejected: it depends on file
  locking on the shared filesystem, and the first cell for each problem
  pays the compile time inside its wall-clock limit.
