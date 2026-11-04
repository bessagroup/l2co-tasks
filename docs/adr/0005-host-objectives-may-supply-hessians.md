---
status: proposed
---

# Host objectives may supply exact Hessians

A host task's loss can be differentiated once: the gradient comes from
the host, through the `custom_jvp` rule of ADR 0003. Differentiating it a
second time failed ("Pure callbacks do not support JVP"), because the
gradient is the output of a `pure_callback`, which JAX cannot
differentiate. l2co-optimizers now has optimizers that need exact
Hessians: `ipopt` and `trustconstr` with `hessian="exact"`
(l2co-optimizers ADR 0007). They take `jax.hessian` of the loss. Their
reference results are on CUTEst (modOpt's IPOPT-2 and TrustConstr-2), a
host suite. So host tasks need second derivatives.

**Decision.** A host objective may define `hessian(x) -> ndarray`, the
dense `(n, n)` Hessian from one host evaluation. When it does, the loss
can be differentiated twice. The CUTEst adapter supplies it from
pycutest's `hess`.

## How

The gradient inside the value's derivative rule is now itself a
`custom_jvp` function, `value_and_grad(x) -> (f, g)`.

- **First derivatives** call it on primal values only, so its own rule
  never runs. They make the same host call as before and compute exactly
  what they did before; the existing tests pass unchanged.
- **Second derivatives** differentiate it. Its rule asks the host for
  the Hessian through a second `pure_callback` and returns
  `(g · t, H t)`.
  - `jax.hessian` (forward over reverse) and `jacrev(jacrev(...))` both
    work: the rule is linear in the tangent, so JAX can transpose it.
  - One `jax.hessian` makes one value-and-gradient call and one Hessian
    call. The Hessian callback takes only `x`, so the `n` tangents of a
    forward-mode Jacobian don't reach the host; `H t` is formed in JAX.
  - Under `vmap` the host gets the whole batch in one callback, as for
    values and gradients (`vmap_method="expand_dims"`).
- **A third last-point cache** serves a repeated Hessian request. As
  with the other two, no kind of request is answered from another's
  cache.
- **An objective without `hessian`** still raises at trace time when
  differentiated twice. The error is now a `TypeError` that names the
  objective and says it has no second derivatives, instead of JAX's
  callback error. l2co-optimizers checks for this before an exact-Hessian
  run starts, so such a run fails loudly instead of looking like a
  solver failure.
- **A Hessian of the wrong shape** is reported, as a wrong gradient shape
  already is.

## Rules

ADR 0003's rules for adapters apply to `hessian` too:

- it is deterministic;
- it is one billed host evaluation, with no hidden finite differences;
- it computes in float64.

The billing of a Hessian is the optimizer's business. l2co-optimizers
bills it as one evaluation (ADR 0007 there).

## Measured on CUTEst

With pycutest 1.8.2 and the CUTEst build in
`/oscar/data/mbessa/mvander7/cutest`, `jax.hessian` of the ROSENBR task
at `x0` is CUTEst's own `[[1330, 480], [480, 200]]`
(`tests/test_cutest_task.py`).

l2co-optimizers' `ipopt` and `trustconstr` ran on five CUTEst tasks with
both kinds of Hessian. The table gives the evaluations to reach
`f* + 1e-8`, with each Hessian billed as one:

| Problem | `ipopt` | `ipopt`, exact | `trustconstr` | `trustconstr`, exact |
|---|---|---|---|---|
| ROSENBR | 65 | 52 | 60 | 69 |
| BEALE | 15 | 18 | 16 | 21 |
| BROWNBS | 124 | 13 | 40 | 61 |
| HELIX | 53 | 30 | 30 | 31 |
| ARWHEAD, N = 100 | 16 | 11 | 12 | 13 |

## Not done

- **Hessian-vector products.** Only a dense `hessian` is supported, not
  `hprod` or a sparse Hessian. CUTEst's problems in the task set have at
  most a few hundred variables, where a dense Hessian is cheap.
  pycutest's `hprod` is there if a large-`n` suite needs it.
- **Stochastic host objectives** are still an extension point (ADR 0003).
