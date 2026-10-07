---
status: proposed
---

# Constraints are part of the task

Where an optimizer is allowed to look is part of the problem, just like
the loss. Until now a `Task` could not say so. Box bounds lived in the
experiment config (`tasks.bounded`), outside the task, and all 36
packaged task configs left them empty. `gaussian_meta` shows the cost:
its three parameters only mean something inside `[0, 1]` (outside it,
Adam's `b1` passes 1), yet optimizers were free to leave that box, while
its `global_min` was estimated inside it (`clip_to_unit=True`). The
estimator and the optimizers were solving different problems.

**Decision.** `Task` gains one field, `constraints`, holding the task's
equality, inequality and box constraints. Constraints are part of the
task's identity. A task without constraints keeps exactly the hash and
the file it has today.

## The shape

- **One field of typed elements.** `constraints: tuple[Constraint, ...]`
  accepts any iterable and stores a tuple. Each element is an
  `Inequality`, an `Equality` or a `Box`. All three share one call
  signature, `c(model) -> 1-D float array`. The type tells a consumer
  what kind of constraint it holds, which a bare function cannot.
- **Deterministic, model only.** A constraint gets the same `model` that
  `loss_fn` gets as its first argument, and nothing else: no key, no
  data batch, whatever `pass_rng` says. `pass_rng` describes the loss.
  A noisy constraint makes feasibility a probability, which is a
  different kind of constraint. If one is ever needed it becomes its
  own type.
- **Raw values, "≤ 0 means satisfied".** Calling an `Inequality` returns
  `g(model)`, satisfied where `g ≤ 0`. Calling an `Equality` returns
  `h(model)`, satisfied where `h = 0`. Output of any shape is flattened,
  so one element can hold one constraint or a thousand. A separate
  `violation(model)` returns the non-negative amount by which each value
  misses beyond its tolerance, so it is zero exactly where the
  constraint counts as satisfied (`max(0, g − tol)`,
  `max(0, |h| − tol)`). This is the convention of the CEC 2006/2017
  constrained suites and
  COCO's bbob-constrained, so their definitions copy across unchanged.
  Flipping signs for scipy (which uses "≥ 0") is the consumer's job.
- **A tolerance per constraint.** Feasible means `g ≤ tol` or
  `|h| ≤ tol`. The defaults follow CEC: `1e-4` for `Equality`, `0` for
  `Inequality`. `Box` has no tolerance, because clipping makes it hold
  exactly. The tolerance is part of the problem: published constrained
  optima are only valid under it.
- **Named.** `Inequality` and `Equality` take a required `name`, unique
  within the task. Names make constraints identifiable in the task's
  identity and in error messages.
- **Box bounds.** `Box.lower` and `Box.upper` are each either a single
  number or a pytree that matches the model's parameters exactly: the
  floating-point array leaves that `count_parameters` counts, with the
  same structure and the same shapes. A single number is cast to those
  shapes when the `Task` is built, and the box is stored in full. Every
  `Box` on a `Task` therefore matches its model. An open side is `±inf`
  (in a pytree, `None` means an empty subtree, not "no bound"). The cast
  accepts an already-expanded box unchanged, because
  `dataclasses.replace` builds the `Task` again.

## What building a `Task` rejects

1. More than one `Box`.
2. A `Box` whose structure or leaf shapes don't match the model's
   parameters after the cast.
3. `lower > upper` anywhere, or a NaN bound. `lower == upper` is allowed
   and pins that parameter.
4. A negative tolerance.
5. An `Inequality` or `Equality` that fails on the model or doesn't
   return a float array. This is checked with `eqx.filter_eval_shape`,
   which runs no computation.
6. A starting model outside its own `Box`. A box is a hard domain; the
   other two kinds describe feasibility, and a starting point may
   violate them. Starting points drawn by a sampler are the sampler's
   responsibility.
7. Two constraints with the same name.

## Identity

The databank keys every result by `Task.hash`. So:

- **Constraints enter identity only when present.** `tag_hashable` gains
  a `constraints` entry for a constrained task and nothing otherwise.
  The entry is a set: `(kind, name, tol)` for each `Inequality` and
  `Equality`, plus the `Box` values. Order doesn't matter, matching how
  the tag is treated, so reordering a factory's list re-keys nothing.
- **Unconstrained tasks must not drift.** Today's tests compare a task
  with its own reloaded copy, which passes even if both sides drift. So
  golden hashes are recorded on the last commit before this change, with
  `jax_enable_x64=True`, in three layers:
  - `Task`s built directly from fixed tags and fixed `global_min`
    values. This tests the hashing code and doesn't depend on hardware.
  - One task each from the factories whose `global_min` is exact (bbob,
    bbob-noisy, cec2005, cec2017, quadratic). Factories with an
    estimated `global_min` are left out: the estimate can differ between
    CPU and GPU.
  - A small `.eqx` file written by today's code and committed under
    `tests/`. After the change it must load with `constraints == ()`
    and its pinned hash.
- **A box's hash depends on float precision**, as `global_min` already
  does. Bounds that float32 can't represent exactly hash differently
  with x64 off. `0` and `1` are exact, so `gaussian_meta` is not
  affected.

## Storage

The constraints tuple is pickled into the `.eqx` JSON header with
cloudpickle, exactly like `loss_fn`. The key is written only when the
tuple is non-empty, so files of unconstrained tasks stay byte-for-byte
identical to today's, and a file without the key loads with
`constraints == ()`. A large box costs about twice its binary size as
hex. That's 6 floats for `gaussian_meta`; storing bounds as binary
arrays after the model can be revisited when a large boxed task appears.

## Only the box is enforced, for now

`Inequality` and `Equality` can be declared, but every consumer refuses
a task that carries one (l2co ADR 0020). A constraint that is declared
but silently ignored is worse than none. "Best so far" in
`l2co-optimizers`' `RunState` is the lowest loss over every evaluation,
so an infeasible point with a lower loss would become "best". Its
quality value would then go negative, which is the failure ADR 0001
guards against.

Enforcing them needs three things that arrive with the first real
constrained task family:

- a "best so far" that prefers feasible points,
- `global_min` defined as the minimum over the feasible region,
- a floor test that only counts feasible points.

## The `global_min` estimator

`estimate_global_min` reads the task's `Box` instead of taking
`clip_to_unit`. A `Box` switches it into box mode: steps are clipped to
the box's bounds, restarts are drawn uniformly from them, and the L-BFGS
arm is skipped. Uniform draws need finite bounds, so it raises for a box
with an open side, and for any `Inequality` or `Equality`, which it
can't handle. For `Box(0, 1)` this reproduces the old
`clip_to_unit=True` results bit for bit.

`estimate_global_min` is public and was released with `clip_to_unit`
(v0.2.0), so removing the keyword is a **breaking change** for the next
release. No sibling repository passes it. A caller who did adds a `Box`
to the task instead.

The floor test's own restart logic (`tests/test_global_min_floor.py`)
stays keyed on the test registry's `domain`, not on the task's box: it
is an independent oracle for the estimator, and it keeps the unit-domain
benchmark cases (which declare no box) clipped to `[0, 1]`.

## First consumer: `gaussian_meta`

`create_gaussian_meta_task` gets `Box(0.0, 1.0)`. Its starting model,
`[0, 0, 0]`, lies on the box's corner, so it passes the start check.

The box made a degenerate point reachable. The momenta were decoded as
`0.85 + 0.15 * x`, so the upper face `x = 1` gave `b1` or `b2 = 1`,
where Adam's bias correction is `0/0` and the loss NaN. Unbounded
optimizers used to overshoot past it; clipping parks them on it. The
mapping is therefore shrunk to `0.85 + 0.149 * x`, so `b1, b2` stay in
`[0.85, 0.999]` and the loss is finite on the whole closed box. This
changes the loss at every point, so its estimated `global_min` changes
with it.

Its 10 task hashes change. The 100 loss files in
`databank/gaussian_meta/loss_history_raw/`, written 21–28 Aug 2026 by
optimizers free to leave the box, stay where they are, under keys new
tasks no longer match. That is intended: they describe a different
problem.

## Rollout order

l2co installs this package as an editable install, and rl2co and
l2co_experiments install l2co the same way. So a change on disk is live
for the next Python process, including the next step of an equeue chain.
If `gaussian_meta` got its box while any entry point still ignored
boxes, a run in that window would write results under the new hash
without respecting the box. The work lands in stages that are each safe
on their own:

1. **This package, types only.** Record the golden hashes first. Then
   add the types, the field, the checks, identity and storage.
   `gaussian_meta` is untouched, so no constrained task exists yet.
2. **l2co, rl2co and l2co_experiments** (l2co ADR 0020). Bounds come
   from the task, every entry point checks, and `tasks.bounded` is
   retired. Behaviour doesn't change, because every task still has no
   constraints.
3. **This package, `gaussian_meta`.** Add the box, and have the
   estimator read it.
4. **Verify.** Re-run `gaussian_meta`'s slow floor test with the box
   enforced, through `sbatch`.

Before each stage, check `squeue` and the equeue lanes so nothing is
rewired under a running job.

## Considered and rejected

- *A separate field per kind* (`lower`, `upper`, `ineq_fn`, `eq_fn`).
  One typed tuple holds the same information, and a new kind adds a
  type, not a field.
- *An untyped tuple of functions.* A function can't be inspected.
  Consumers couldn't find a box's arrays or tell an equality from an
  inequality. Rewriting `h = 0` as `h ≤ 0` and `-h ≤ 0` breaks SLSQP:
  both are always active and their gradients cancel.
- *Building only `Box` now* and the other kinds with their first task.
  Rejected to fix the full set of kinds in the API now, with consumers
  refusing what they can't enforce.
- *Returning violations instead of raw values.* A violation is flat at
  zero across the feasible region, but SLSQP needs each value and its
  gradient there.
- *One tolerance on the `Task`.* Breaks as soon as a task mixes units.
- *Scalar-only `Box`.* Per-parameter bounds would later need a format
  change.
- *Storing a box as given and expanding it on use.* Every consumer would
  have to remember to expand.
- *Leaving constraints out of identity.* Old results that ignored the
  box would be pooled with new ones that respect it, under one key.
- *Order-sensitive identity.* Reordering a factory's list would re-key
  every task it produces.
