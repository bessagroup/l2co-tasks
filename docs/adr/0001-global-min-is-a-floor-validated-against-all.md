# `global_min` is a floor, validated against the full optimizer suite

## Context

`global_min` is not just informational: `l2co`'s `shift_to_quality_value`
subtracts it to form the quality value, and the downstream log-scaling
(`min_custom`) takes `log10` of the result, so a loss *below* `global_min`
produces negative quality values and silently corrupts the scaling. The
value is also part of `Task` identity (`tag_hashable`, `hash`). So
`global_min` must be a genuine lower bound on the achievable loss — a
**floor** — for every optimizer that could run on the task, not merely a
convenient reference number.

The **empirical** `global_min` (gaussian-class, spiral, MNIST-1D,
gaussian-meta) was set by short multi-restart Adam (`estimate_global_min`).
That is weaker than the production portfolio, which includes L-BFGS and
SepCMA-ES — so those optimizers can dig below it and break the floor.

## Decision

1. **The floor must hold against the full `all` suite** (~60 optimizers),
   not just the per-family production portfolio, so `global_min` stays a
   valid lower bound under *any* portfolio choice — important because it
   is baked into task identity.
2. **Strengthen `estimate_global_min`** to a genuine best-of-strong-search
   (add optax **L-BFGS** to the multi-restart Adam, run to convergence) so
   the floor holds *by construction*. Estimator strength is tuned
   **per task, guided by the floor test**: high-dim weight-space tasks get
   L-BFGS + a few restarts (depth); the low-dim bounded gaussian-meta task
   gets many random restarts (breadth, to approximate the global search
   that population methods would otherwise win).
3. **Stay standalone** — use only optax L-BFGS / random restarts, never an
   evosax dependency, even though the floor must hold against evosax
   methods. (Those methods don't scale to high-dim NN weight space, and
   random restarts cover the low-dim case.)
4. **Two-tier floor test in `l2co-tasks/tests/`:** a fast deterministic
   tripwire in default CI (estimator determinism + the strongest
   deterministic member, L-BFGS, never breaches) plus a slow,
   `requires_l2co`, opt-in test that runs the full `all` suite at
   production budget and asserts no loss `< global_min − tol`.
5. **Scope:** empirical tasks fully; a representative analytical sample
   (sphere, rastrigin, one CEC) as a `scale_input`-bug tripwire;
   theoretical (PINN) tasks a trivial `loss ≥ −tol` guard.
6. **Budget:** the estimate is a *converged* floor, independent of any one
   study's budget config, so `global_min` is a stable property of the task.

## Considered alternatives

- **Downstream clamp** (clamp quality value at 0 in `l2co`) instead of
  re-estimating: rejected — it hides a non-floor `global_min` rather than
  fixing it, and puts the correctness fix in the wrong package.
- **Floor only against the production portfolio:** rejected — `global_min`
  is in task identity, so it should be robust to portfolio changes.

## Consequences

Re-estimating changes empirical `global_min` values → changes `hash` →
one-time databank re-keying. Accepted as a migration cost paid now rather
than after more data accrues.

## Validation finding (floor test, first run)

Running `tests/test_global_min_floor.py` revised the a-priori premise.
The breaches are **confined to the tiny test-registry builds**
(`restarts=2, steps=10`); at *production* scale the empirical estimates
**already hold** within tolerance (spiral → ~0 perfectly separable,
gaussian-class → 0.362 irreducible overlap, gaussian-meta → not beaten,
even by the full `all` suite incl. CMA-ES/PSO/DE). So short Adam is
adequate wherever the task has a genuine loss floor; it under-estimates
only on under-budgeted / trivially-zeroable builds (e.g. the
over-parametrized tiny mnist1d). This *strengthens* the case for the
L-BFGS estimator as a *by-construction* guarantee (the floor then holds
at every build size and for future tasks without a benign floor) while
*shrinking* the hash-churn cost (production values barely move).
