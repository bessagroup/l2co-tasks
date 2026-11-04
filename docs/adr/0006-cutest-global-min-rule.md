---
status: proposed
---

# CUTEst `global_min`: overrides, then the lowest value reached, then a confirmed `SOLTN`

Every CUTEst task reads its `global_min` from a committed table (ADR
0004). l2co_experiments' `cutest_global_min_table` experiment writes that
table, and its rule takes the lower of an estimate and the optimum the
SIF file records in a comment (`SOLTN`). The rule did not produce a
correct table on its own:

- **102 of the 475 rows were corrected by hand** before and after the
  first databank run, each from the formula or from a lower value that
  another method reached.
- **The rule let 16 impossible `SOLTN` values in.** Some were copied from
  another problem or size, some were rounded, some were a trivial bound.
  MEXHAT's `SOLTN` is −1.117; the formula gives −0.04001.
- **Rerunning the experiment reverts every correction**, because the
  rule still prefers `SOLTN`. ADR 0004 says so.

Three more CUTEst task sets are planned: larger sizes, bound-constrained
problems and generally constrained problems. Nobody will check their rows
by hand one at a time. So the rule has to give correct values by itself,
and it has to reproduce the table we already trust.

**Decision.** The table experiment sets each row from three kinds of
evidence, strongest first:

1. **An override**: a committed value or exclusion, with its basis.
2. **The lowest value reached** by the experiment's arms (the methods it
   runs on every row).
3. **`SOLTN`**, only where a reached value confirms it.

The order is the same for every CUTEst task set. Each set's ADR names
its arms.

## Overrides

- **One file next to the table:**
  `src/l2co_tasks/_src/cutest_global_min_overrides.csv`, with the columns
  `problem`, `sif_params`, `sif_sha256`, `global_min`, `excluded` and
  `basis`. A row gives either a value or an exclusion reason, never both.
  The table and the evidence for its values then change in the same PR.
- **An override names the SIF file it was derived from.** If the
  problem's file has changed since, the experiment reports the override
  as stale and does not apply it, just as building a task refuses a
  stale table row.
- **What goes in:**
  - values from the formula: an exact solve (linear least squares), a
    closed form (ARGLINB, ARGLINC), a bound that is reached (CHAINWOO is
    1 plus squares), or a derivation (MEXHAT, SCURLY, FLETCBV3);
  - values only a special run reached, such as a databank dump's float64
    best (DMN37142LS) or a grid search with Newton polish (ELATVIDU,
    FLETBV3M);
  - exclusions, with the reason from the formula (INDEF and BRKMCC are
    unbounded below) or from the code (STRATEC, AKIVA, HIELOW and PARKCH
    are unbounded below as coded).
- **What stays out:** values the arms reach again by themselves. 52 of
  the 102 corrections are such values: 37 from an exact-Hessian IPOPT
  or trust-constr run from `x0`, and 15 the lowest of the estimate and
  those two runs. They are arms now, so their rows need no override. Of
  the other 50, 39 become value overrides, 8 become exclusion overrides
  and 3 are sizes that are no longer listed at all.
- **One row per size.** A derivation that holds for every `n` is written
  out for each listed size. The file holds values, not formulas.
- **A value override is still checked against the arms.** The stored
  value is `min(override, lowest reached)`, so float noise in an
  evaluation can't breach the floor. If a reached value lies below the
  override by more than the breach tolerance, the override is wrong. The
  row stores the lower value, and the report lists it for review.
- **An exclusion override skips the arms.** The experiment's `exclude`
  setting goes away: every exclusion and its reason is committed in this
  file.
- **The file is seeded from `0004-cutest-global-min-corrections.csv`**,
  with the rows whose basis is not an arm. That CSV stays as the record
  of what changed against run `1791474722`. Nothing reads it.

## The arms for unconstrained problems

- **The estimator**, as in ADR 0004: restarts from `x0` and from
  `relative_normal` points around it, escalating until the estimate
  settles.
- **IPOPT and trust-constr with exact Hessians** (l2co-optimizers ADR
  0007), one run each from `x0`.

The table experiment runs all three, so one run produces the table.
Until now the two Newton runs were separate `cutest_floor_test` runs.

**A row whose estimate never settles is no longer excluded for that
reason alone.** It takes the lowest value reached, and the report lists
it. ADR 0004 already says that unboundedness is settled from the
formula, not from an estimate that won't settle. The first table run
excluded 25 rows for not settling. All 25 turned out bounded, and all
have values in the committed table. A row is still excluded if:

- an override excludes it;
- it fails to build;
- no arm reaches a finite value;
- an arm reaches a value below `unbounded_below`.

## When `SOLTN` counts

- **Confirmed:** `SOLTN` is at most the breach tolerance,
  `1e-6 · max(1, |v|)`, below the lowest value reached `v`. The row
  stores `min(SOLTN, v)`. That covers a correct optimum printed exactly,
  such as 0.0 for a sum of squares that reaches zero, or DIXMAAN's 1.0.
  An optimizer only gets within float noise of such a value. Storing the
  reached value instead would put `global_min` slightly above the true
  optimum. Then a better run would breach it, and the quality scale
  would stop rewarding precision past that point.
- **Unconfirmed:** `SOLTN` lies further below `v`. The row stores `v`
  and the report lists it for a formula review. A `SOLTN` this far below
  every reached value is how the 16 impossible values showed up.
- **Above `v`:** a rounded-up or local optimum. The row stores `v`.

This replaces ADR 0004's "lower of the estimate and `SOLTN`", and its
exception for rows that never settle but stay above `SOLTN`.

The `source` column says which evidence won: `override`, `reached` or
`soltn`. `source` is not part of a task's identity; only `global_min` is.

## Checked against the committed table

The new rule runs over every row of the committed table before it
produces any new row. The committed values stay where they are.

- **Reproduced:** the new value lies within the breach tolerance of the
  committed one. The committed value is kept, so a last-bit difference
  between machines moves no task id.
- **Not reproduced:** the row needs one of two explanations.
  - **The rule or the overrides miss something.** For example, a value
    only a special run reached was never written to the overrides file.
    Fix the overrides or the rule, and run again.
  - **The committed value is wrong.** An arm went below it by more than
    the tolerance. It is corrected as in a breach round, formula first.
    Its tasks are pruned from the `cutest` store and refilled.

The rule is accepted when every row is reproduced or explained. From
then on the table can be regenerated from code, which ADR 0004 says it
could not be.

## One table, filtered by size

The table keeps one row per (problem, size), whatever set the row
belongs to. `CutestTableSampler` gains `min_n` and `max_n`:
`conf/tasks/cutest.yaml` lists `n ≤ 100`, and each new set picks its own
range. `create_cutest_task(problem, sif_params)` needs no new argument,
because a (problem, size) has exactly one row. The experiment's `max_n`
becomes a `min_n`/`max_n` range too.

## Fixed variables

ADR 0004 refuses problems with fixed variables (class X). But pycutest
hides fixed variables by default (`drop_fixed_variables=True`). The six
fixed-size X problems with `n ≤ 100` all built with no bounds at all on
the variables that remain. BIGGS3, for example, is BIGGS6 with three
variables fixed: an unconstrained problem in 3 variables. So:

- **`create_cutest_task` accepts class X.** The model holds the free
  variables only, and `dimensionality` counts them.
- **X problems join the unconstrained sets:** `cutest` for `n ≤ 100`,
  and later a set for larger `n`.
- **No existing task changes.** X rows are new rows, and the tag gains
  no key, so every committed row keeps its id. `cutest` grows by the X
  rows, and the store gets them by an append.

## The databank run is the floor test

ADR 0004 floor-tested the table in a separate run before the first
databank run. For new rows, the databank run itself is now the floor
test:

1. Write the table with the rule above.
2. Run the databank.
3. Read `global_min_breaches.json`, which every dump writes since l2co
   PR #81.
4. Correct each breached row through an override.
5. Prune those tasks from the store and refill them with
   `skip_existing`.

Three things changed since ADR 0004 made the other choice:

- **A breach no longer damages anything downstream.** A dump scores a
  loss below `global_min` as reaching it (quality 0), and records the
  task's float64 best.
- **Pruning works.** The `cutest` store lost 9 tasks in place on
  2026-10-12 and was refilled with 0 errors. There is no prune command
  yet; if breach lists come back long, writing one is a separate
  decision.
- **A floor test costs as much as the databank run.** Both run the same
  suite with 25 realizations. For larger sizes that doubles the most
  expensive compute in the plan.

## Considered and rejected

- **Rerunning ADR 0004's pipeline and correcting by hand again.** It
  wouldn't scale to rows nobody checks, and it would leave the table
  impossible to regenerate.
- **Never storing `SOLTN`.** Correct printed optima, such as 0.0, would
  be replaced by values a hair above them (see "When `SOLTN` counts").
- **Replacing the committed values with the new rule's output.** Every
  row that moved would change its task id, and the `cutest` store would
  need those tasks pruned and refilled, for no gain where the committed
  value was right.
- **The overrides in l2co_experiments**, next to the experiment. The
  table and its evidence would live in two repositories.
- **A second table per task set.** `create_cutest_task` would need to
  know which file to read, although a (problem, size) only ever has one
  row.
- **Keeping fixed variables visible and pinning them with a `Box`.** It
  is the same problem, but `dimensionality` would count the pinned
  variables, and optimizers that only clip would spend steps on them.
