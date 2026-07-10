# L2CO Tasks

The shared vocabulary for `l2co-tasks` — the package of optimization
`Task` objects consumed by the L2CO optimizer-selection studies. This
glossary focuses on the terms that recur when reasoning about whether a
task is *well-posed*: its global minimum and how optimizers behave on it.

## Language

**Global minimum** (`global_min`):
The reference loss value recorded on every `Task`, against which an
optimizer's progress is measured. Not merely informational — it is
subtracted from trajectories downstream (see **Quality value**) and is
part of the task's identity (`tag_hashable`, `hash`).

**Quality value**:
A trajectory's loss shifted into `loss − global_min` space, where the
optimum is `0`. The L2CO reporting stack (ERT targets, log-scaling)
operates here. Log-scaling assumes quality values are non-negative.
_Avoid_: regret (used informally for the same quantity; "quality value"
is the term the `l2co` code uses).

**Floor**:
The property that `global_min` is a true lower bound — *no* optimizer,
at any budget, drives the loss below it. When the floor is violated the
quality value goes negative and the downstream log-scaling silently
produces NaNs. Validating the floor is the first goal of this effort.

**Analytical `global_min`**:
A `global_min` known in closed form and a genuine mathematical lower
bound — BBOB/CEC registry optima, the quadratic least-squares residual.

**Theoretical `global_min`**:
A `global_min` set to a known limit the loss approaches but a finite
model need not attain — the PDE-residual minimum `0` of the PINN tasks.
A genuine lower bound, but possibly far below any achievable loss.

**Empirical `global_min`**:
A `global_min` with no closed form, set to the *best loss found by a
short benchmark optimiser* at task-creation time (`estimate_global_min`,
currently short multi-restart Adam) — gaussian classification, spiral,
MNIST-1D, the Adam-meta task. An estimate, **not** a guaranteed floor.

**Production portfolio**:
The per-family set of optimizers the real L2CO study actually runs,
defined in `l2co_experiments/conf/optimizers/` (`medium` for
BBOB/CEC/quadratic, `pde` for PINNs, `gaussian_classification` for the
gaussian task). The hypothesis study judges "makes sense" against these.
_Avoid_: "the optimizers" (ambiguous — could mean the full registry).

**The `all` suite**:
The full ~60-optimizer registry (every evosax/optax method plus L-BFGS).
The **floor** must hold against this whole suite, not just the
production portfolio, so that `global_min` stays a valid lower bound
under any portfolio choice.

**Floor test**:
The validation that `global_min` is a floor. Two-tier: a fast,
deterministic CI *tripwire* (estimator determinism + the strongest
deterministic member never breaches) and a slow, opt-in (`requires_l2co`)
run of the full `all` suite asserting no loss falls below `global_min`.
A **breach** is a loss below `global_min` by more than float tolerance.

**Floor-test-guided estimation**:
The empirical-`global_min` workflow: set a per-task estimator strength,
run the floor test, and if a task is breached, increase *that* task's
strength and re-run until the floor holds. The floor test is the oracle
for "is this `global_min` actually a floor."

**Pairwise contrast**:
A single, literature-grounded directional hypothesis about two optimizer
classes on a task family, e.g. `L-BFGS_final < Adam_final` on PINNs
(Jnini et al.) or `CMA-ES_final < gradient_final` on multimodal BBOB
(Hansen). The unit in which the optimizer-hypothesis study states and
judges "makes sense" — judged by a paired sign/Wilcoxon test on final
**quality value**, pooled across the family's instances and realizations.
_Avoid_: "ranking" (we deliberately do not predict full per-task orders).

**Hypothesis oracle**:
The documented set of expected pairwise contrasts per family, each tied
to a citation (BBOB/Hansen for the analytic functions, Jnini et al. for
PINNs, standard DL/HPO expectations for the supervised and meta tasks),
plus the universal sanity floor: every optimizer must beat `randomsearch`.

**Realization**:
One random-start repeat of an optimizer on a fixed task instance. The
study uses ~16 (8 for the 41k-budget PINNs), pooled within a contrast
for statistical power; all seeds fixed so the study re-runs bit-for-bit.
