---
status: accepted
---

# Base landscapes group the analytic benchmark functions

A function-disjoint train/test split exists to test a learned optimizer
selector on landscapes it has not seen. Function names do not deliver
that across the analytic suites. All 30 BBOB-noisy functions are noisy
versions of eight noiseless BBOB functions. The basic CEC 2005 and CEC
2017 functions are shifted and rotated versions of the sphere, the
Rosenbrock and Rastrigin functions and their kin. BBOB itself ships
separable and rotated versions of one function (f3 and f15 are both
Rastrigin). A split by name can put a function in the training half and
its own noisy, rotated or CEC version in the test half, and a selector
then looks better on "unseen" problems than it is.

**Decision.** Every function of the analytic suites belongs to a *base
landscape*, recorded in a table that ships with the package
(`src/l2co_tasks/_src/base_landscapes.csv`, read through
`l2co_tasks.base_landscape` and `l2co_tasks.base_landscapes`). A split
assigns whole base landscapes to one side.

## The rule

Two functions share a base landscape when they evaluate the same defining
formula and differ only by

- a shift or a rotation,
- the size of an existing conditioning (the BBOB-noisy ellipsoid at 1e4
  and f10 at 1e6; Schaffer F7 at condition 10 and 1000),
- BBOB's oscillation and asymmetry maps (`T_osz`, `T_asy`),
- a parameter of the same generator (Gallagher with 101 or 21 peaks),
- noise, or
- an embedding (`bbob_embedded` tasks carry the BBOB names).

Anything else is a different landscape. In particular:

| Case | Call | Why |
|---|---|---|
| Sphere vs ellipsoid | apart | A conditioning *size* changes within a landscape; adding one where there was none makes another. |
| Ellipsoid vs discus, bent cigar, sharp ridge | apart, each its own | Different eigenvalue shapes; BBOB keeps them as distinct functions. CEC 2017 F1 joins bent cigar. |
| Step ellipsoid (f7) vs ellipsoid; CEC 2017 F8 non-continuous Rastrigin vs Rastrigin | apart | Rounding makes plateaus where the gradient is zero. The BBOB-noisy step ellipsoid joins f7. |
| Büche-Rastrigin (f4) vs Rastrigin | together | The Rastrigin formula on a skewed, oscillated input plus a boundary penalty; BBOB defines it as a Rastrigin variant. |
| Lunacek bi-Rastrigin (f24) vs Rastrigin | apart | A different formula: two sphere funnels with a Rastrigin term. CEC 2017 F7 joins f24. |
| BBOB Schwefel x·sin(√\|x\|) (f20) vs CEC 2017 modified Schwefel (F10) | together | The same Schwefel 2.26 formula; CEC changes only the boundary handling. |
| CEC 2005 expanded Schaffer F6 (F14) vs Schaffer F7 | apart | Different formulas. CEC 2017 F6 is implemented as Schaffer F7 and joins f17/f18. |
| CEC 2005 Schwefel 1.2 (F2) and its noisy version (F4) | one group of their own | F4 is F2 plus noise; no BBOB function uses this formula. |
| CEC hybrid and composition functions (CEC 2005 F15–F25, CEC 2017 F11–F30) | each its own | A hybrid splits its variables over several formulas, and a composition places several around different optima with different weights, so the global structure is new. That holds for CEC 2017 F29/F30 too, although they compose the hybrids F15–F19. |

The table has 108 rows (24 BBOB names, shared with `bbob_embedded`; 30
BBOB-noisy; 25 CEC 2005; 29 CEC 2017) in 58 groups: 18 around the BBOB
functions, 9 new CEC basic formulas, and 31 hybrid and composition
functions. Groups are named after their base (`rastrigin`), or after
suite and function for a hybrid or composition (`cec2017_f21`).

## Keyed by `fn_name`

Every analytic task carries an `fn_name` tag, and the names are unique
across the four suites (`sphere`, `bbob_noisy_f101`, `f9` for CEC 2005,
`cec2017_f5`). A consumer joins on it, so nothing about the tasks or
their stored labels changes. `tests/test_base_landscape.py` fails when a
function of the `bbob`, `bbob_noisy`, `bbob_embedded`, `cec2005` or
`cec2017` configs has no row, or a row names no configured function, and
pins each call in the table above.

## Not covered

CEC 2013 LSGO, CUTEst and the non-analytic families have no rows: no
split that needs them exists yet. `base_landscape` raises `KeyError` for
their names rather than guessing.

## Considered options

- **Isometries only** (shift, rotation, noise, embedding): cuts almost
  every CEC function off its BBOB base, since CEC's plain functions lack
  BBOB's conditioning and oscillation maps, and splits the noisy
  ellipsoid (1e4) from f10 (1e6).
- **Named families** (everything with "Rastrigin" in it together): fuzzy
  at the edges (is Lunacek a Rastrigin?) and makes very large groups.
- **A field on bbob-jax's `FunctionSpec`**: closer to the definitions, but
  needs a bbob-jax release, plumbing for the embedded tasks, and a label
  refresh of every stored dataset, for a property only a split uses.
