---
status: proposed
---

# Host objectives run behind an ordinary JAX loss

Every task so far computes its loss in JAX, so every optimizer can trace,
vmap and differentiate it. Some benchmark suites can't be written that
way. CUTEst (ADR 0004) is Fortran: it supplies its own values and
gradients, and JAX can neither trace nor differentiate it. We want every
optimizer in the registry to run on such a suite, not only the scipy and
IPOPT entries that already call the objective from outside JAX
(l2co-optimizers ADRs 0002, 0003). A comparison without Adam, CMA-ES and
SHADE can't say which optimizer wins where.

**Decision.** A generic module in l2co-tasks turns a **host objective**
(an objective computed outside JAX) into an ordinary
`loss_fn(model, **sample)`. The seam is the loss. Everything above it
keeps seeing a JAX function: every optimizer, `RolloutWrapper`, rl2co
and the databank. No code on the run path changes. l2co-tasks is the
home because it owns the loss and the `Task`. l2co-optimizers must not
know about tasks anyway.

## What an adapter supplies

An adapter is the code that plugs one suite in (CUTEst is the first).
It supplies an **opener**: a picklable, hashable, zero-argument
callable that returns the objective. The objective has:

- `value(x) -> float`;
- optionally, `value_and_grad(x) -> (float, ndarray)`. Without it the
  task has no gradient.

`x` is the model's parameters flattened to one float64 vector.

## What the module does

- **The value is a `jax.pure_callback`, wrapped in a `jax.custom_jvp`.**
  The derivative rule asks the host for value and gradient in one call
  and returns `f` and `g · t`. It must be `custom_jvp`, not
  `custom_vjp`: optimistix's minimisers take gradients by `jax.linearize`
  (forward mode) and then transpose the result (`quasi_newton.py`,
  `_misc.lin_to_grad`), and forward mode fails on a `custom_vjp`.
- **One host call per value-and-gradient.** A plain call asks for the
  value only.
- **Batched evaluation under vmap.** `vmap_method="expand_dims"` hands
  the host every point of a population, across every realization, in one
  call. The host then loops over the points.
- **Last-point caches, one per kind of request.** An optimizer that asks
  for the same point twice in a row is served from the cache. A value
  request is only ever answered from an earlier value request, and a
  value-and-gradient request from an earlier value-and-gradient request.
  An objective's two routines may differ in the last bit, and a single
  shared cache made a run's numbers depend on which run came before it in
  the same process.
- **A lock around host calls.** Fortran and simulation codes are rarely
  safe to enter from two threads at once.
- **`loss_fn` is a small module-level object**, not a closure. It holds
  only the opener: the parameter count comes from the model, and whether
  there is a gradient from the objective. The traced function and the opened
  objective are built lazily, once per process, and the objective is
  opened when JAX traces the loss rather than inside the callback. A
  closure can't be saved: cloudpickle pickles it by value, together with
  the module's lock, and `Task.save` fails. Equal instances hash equal,
  so jit caches also survive a rebuilt task.
- **A warning when traced in float32.** The host computes in float64. In
  float32 its inputs arrive rounded, and its results go back rounded.
  Every pipeline that writes a databank store already runs in x64.
- **An objective without a gradient raises when JAX first traces a
  derivative through it**, with a message naming the problem. It does
  not fail later, inside a run.

## Rules every adapter must follow

1. **No side effects that depend on how often it is called.**
   `pure_callback` lets JAX skip or repeat calls. For example, the
   unbilled evaluation of the starting population in `reset` may or may
   not run. A deterministic objective (CUTEst) is fine. One that submits
   jobs or takes a licence per call is not.
2. **One host call is at most one billed evaluation.** The module never
   computes finite-difference gradients. They would hide `d` evaluations
   inside one billed evaluation, and the budget comparison would break.
   An objective without a gradient is simply non-differentiable.
3. **float64 on the host.**

## What the spike and the first implementation measured

Measured on Oscar on 2026-10-08, with ROSENBR, EXTROSNB and 19 entries
covering every optimizer family, then rechecked with the implemented
module through l2co's `RolloutWrapper`. The scripts are in
`/oscar/scratch/mvander7/host_task_spike/spike/`, which scratch purges
after 30 days.

- **The wrapper adds no error of its own, but don't expect bitwise
  equality with native JAX.** A bare `jax.pure_callback` around the same
  jitted function reproduces the module's numbers exactly. Against native
  JAX, a host version agrees only up to arithmetic drift, even when the
  host runs the same JAX function: XLA compiles it differently inside the
  fused run loop than on its own.
  - In the 2-D spike, 9 entries matched native JAX bitwise.
  - In the implementation's 3-D check across 14 entries, adam drifts by
    1 ulp from step 9, BFGS by 1e-14 from step 22, and trust-krylov only
    near the optimum (step 116, between values of 1e-13 and 1e-21). The
    rest match bitwise.

  So tests compare against native JAX with a tolerance, over a short
  horizon. SHADE is the exception: native SHADE itself depends on how
  many realizations run together (https://github.com/bessagroup/l2co-optimizers/issues/24), so it is left out of those
  comparisons until that is fixed. With CUTEst's Fortran, trajectories
  drift apart after 8–116 steps, because its arithmetic differs from
  XLA's in the last bits. The best values found still agree.
- **Every transformation works:** `grad`, `jvp`, `linearize` plus
  transpose, nested vmap, `lax.map`, `lax.scan`, and the nested
  callbacks of the scipy and IPOPT drivers. A Hessian fails loudly
  ("Pure callbacks do not support JVP").
- **Host calls against billed evaluations:** about 1:1, plus one
  unbilled evaluation of the starting population per realization.
  `lbfgs` sends 3 requests per billed evaluation, all at the same point,
  and the value-and-gradient cache absorbs them: 128 real evaluations for
  209 billed.
- **Overhead per billed evaluation, 25 realizations:**
  - 4–35 µs for the entries that run inside JAX (0.2–1.5 µs natively);
  - 0.4–0.8 ms for the scipy and IPOPT drivers (0.25 ms natively).

  The allowance is about 14 ms (l2co-optimizers ADR 0002).

## Extension points, not built

Each is added when a real adapter needs it, not before.

- **Batched objectives** that evaluate a whole population in parallel
  (simulations). The module already hands the host a batch, so only the
  adapter interface grows.
- **Stochastic objectives.** The key would reach the host as a seed. The
  last-point caches would then be wrong, so they must be keyed on the seed
  too, or switched off.
- **Objectives with side effects.** These need `io_callback` in place of
  `pure_callback`, which has tighter limits under vmap and
  differentiation.
- **A dataset or minibatch** passed through to the host.
- **Objectives without a gradient.** The decided design, deferred until
  the first such adapter:
  - `Task` gets a `differentiable` field.
  - `OptimizationStep` gets an explicit `uses_gradient` property. It
    resolves the way `.name` does: built-in entries declare a bool,
    meta-optimizers register a function of their step (`True` if any
    menu member uses gradients), and an optimizer that declares nothing
    counts as `True`. It is explicit rather than read from `family`,
    following the rule that per-optimizer properties are never inferred
    from the family.
  - l2co refuses a gradient optimizer on a non-differentiable task in
    `RolloutWrapper.init` and `init_run_state`, next to `task_bounds`.
    l2co_experiments leaves such pairs out of the grid, because a cell
    that fails in the create stage sinks the whole databank dump.

  In the first version every task has a gradient, so none of this would
  ever run.
- **Bounds that differ per variable.** l2co passes one `(lo, hi)` pair
  to every optimizer (l2co ADR 0020).

## Considered options

- **Host-loop entries only:** the scipy and IPOPT drivers call the host
  objective directly, and every entry that runs inside JAX refuses host
  tasks. Simpler, with no callback inside the scan, but a CUTEst task
  set would hold 7 optimizers out of 73. Rejected.
- **`custom_vjp`.** Rejected: optimistix's forward-mode gradients fail
  on it.
- **Finite-difference gradients for objectives without one.** Rejected
  by rule 2.
- **Building the gradient-capability machinery now.** Rejected: it would
  span four repos and never run in the first version.
