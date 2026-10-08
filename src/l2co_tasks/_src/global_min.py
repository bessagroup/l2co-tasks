"""Empirical estimation of a task's global minimum.

``l2co-tasks`` sets each task's :attr:`~l2co_tasks.Task.global_min` in one
of three ways, matching the nature of the problem:

* **analytical** -- retrieved or computed in closed form (BBOB/CEC2005
  registry optima, the quadratic least-squares residual);
* **theoretical** -- a known limit that the loss approaches but a finite
  network need not attain exactly (the PDE-residual minimum ``0`` of the
  PINN tasks);
* **empirical** -- the true minimum cannot be read off because the loss
  floor is set by dataset noise / finite model capacity (gaussian
  classification, spiral, MNIST-1D) or has no analytic form at all (the
  Adam-meta task). For these the factory records the *best loss found by
  a short benchmark optimiser* run at task-creation time.

This module implements that empirical arm: :func:`estimate_global_min`
runs a seeded, fixed-budget, multi-restart search and returns the lowest
full-batch loss observed. Each restart trains the trainable
(inexact-array) leaves of the model on the task's full-batch loss. Two
complementary search arms are taken per restart and minimised over:

* a **cosine-annealed Adam** arm (on a task with a
  :class:`~l2co_tasks.Box`, parameters are clipped back into the box
  after every step) -- this provides *breadth*, and is the only arm on
  boxed tasks, where resampled restarts approximate a global search of
  the low-dimensional box;
* an **L-BFGS** arm on the unbounded weight-space tasks -- this provides
  *depth*: a quasi-Newton search to (near-)convergence, which is what
  makes ``global_min`` an actual lower bound against the stronger
  optimisers in the downstream portfolio (see ``CONTEXT.md`` and
  ``docs/adr/0001-global-min-is-a-floor-validated-against-all.md``). It
  is skipped on boxed tasks, where its line search fights the
  projection.

The box comes from the task itself (``docs/adr/0002``): a task that
declares a ``Box`` is searched inside it, and a task with an inequality
or equality constraint is refused, since this search can't respect
one.

The floor-test battery (``tests/test_global_min_floor.py``) is the oracle
for whether the resulting value is genuinely a floor; estimator strength
(restarts / steps, per task) is tuned until that test holds.

The estimate is **deterministic** in ``seed`` -- given the same task and
budget it always returns the same float. This matters because
``global_min`` participates in :attr:`Task.tag_hashable`, so a wobbling
estimate would change a task's identity (``hash``, ``.eqx`` filename,
downstream databank keys) between rebuilds.
"""

# Standard
from __future__ import annotations

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax

# Local
from .constraints import Box
from .task import Task

# =============================================================================

# Default L-BFGS depth (steps per restart) when the caller does not set
# ``lbfgs_steps``. This is a *convergence* budget, independent of the Adam
# breadth budget ``n_steps``: on these smooth losses L-BFGS settles well
# within this many steps, so even a small-``n_steps`` build still gets a
# converged (floor-respecting) L-BFGS estimate.
_LBFGS_DEFAULT_STEPS = 300


def _finite(loss: jax.Array) -> jax.Array:
    """Map a non-finite loss to ``+inf`` so it loses every ``minimum``.

    A divergent step (e.g. the Adam-meta search clipping into a regime
    with a huge decoded inner learning rate) yields ``nan``/``inf``;
    folding it to ``+inf`` lets the running ``jnp.minimum`` ignore it
    instead of being poisoned by ``nan`` propagation.
    """
    return jnp.where(jnp.isfinite(loss), loss, jnp.inf)


def _resample_leaves(model, key: jax.Array, box: Box | None):
    """Return ``model`` with its inexact-array leaves freshly sampled.

    Mirrors ``sample_model`` in the test contract helpers: unbounded
    (weight-space) tasks draw from a standard normal, boxed tasks draw
    uniformly from the box so the restart respects its bounds.

    Parameters
    ----------
    model : PyTree
        Model whose structure is used as the template.
    key : jax.Array
        Random key.
    box : Box or None
        The task's box, already cast to the model's parameters, or
        ``None`` for an unbounded task.

    Returns
    -------
    PyTree
        A model with resampled trainable leaves.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jtu.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    if box is None:
        new = [
            jr.normal(k, x.shape, x.dtype)
            for x, k in zip(leaves, keys, strict=True)
        ]
    else:
        lower = jtu.tree_leaves(box.lower)
        upper = jtu.tree_leaves(box.upper)
        new = [
            jr.uniform(k, x.shape, x.dtype, minval=lo, maxval=hi)
            for x, k, lo, hi in zip(leaves, keys, lower, upper, strict=True)
        ]
    return eqx.combine(jtu.tree_unflatten(treedef, new), static)


def _search_box(task: Task) -> Box | None:
    """The box to search ``task`` in, or ``None`` if it has none.

    Raises
    ------
    ValueError
        If the task has an inequality or equality constraint, which this
        search can't respect, or a box with an infinite bound, which
        restarts can't be drawn uniformly from.
    """
    unsupported = [c for c in task.constraints if not isinstance(c, Box)]
    if unsupported:
        names = ", ".join(f"{c.kind} {c.name!r}" for c in unsupported)
        raise ValueError(
            "estimate_global_min can only respect a Box, but the task "
            f"also has {names}"
        )
    if not task.constraints:
        return None
    (box,) = task.constraints
    bounds = jtu.tree_leaves((box.lower, box.upper))
    if not all(bool(jnp.all(jnp.isfinite(b))) for b in bounds):
        raise ValueError(
            "estimate_global_min draws restarts uniformly from the task's "
            "Box, so every bound must be finite"
        )
    return box


def estimate_global_min(
    task: Task,
    *,
    seed: int = 0,
    n_restarts: int = 3,
    n_steps: int = 1000,
    lr: float = 1e-2,
    use_lbfgs: bool = True,
    lbfgs_steps: int | None = None,
) -> float:
    """Estimate a task's global minimum with benchmark optimisers.

    Runs ``n_restarts`` independent searches over the task's trainable
    parameters and returns the lowest loss observed across all restarts,
    arms and steps. Each restart takes the minimum of two arms:

    * a cosine-annealed **Adam** arm of ``n_steps`` steps (breadth);
    * an **L-BFGS** arm of ``lbfgs_steps`` steps (depth) -- only on
      tasks without a :class:`~l2co_tasks.Box`, where it drives the
      estimate to a genuine lower bound against the stronger portfolio
      optimisers. It is skipped on a boxed task (its line search fights
      the projection) or when ``use_lbfgs=False``.

    The result is the "empirical" ``global_min`` recorded by the
    supervised-learning and meta factories.

    The first restart starts from ``task.model`` (the factory's
    initialised weights), which is usually a strong starting point; the
    remaining restarts start from resampled leaves for diversity. On a
    boxed task the Adam steps are clipped into the box and *every*
    restart is drawn uniformly from it, since the factory's initial model
    may sit at the box corner where the projected step can stall, and
    resampled restarts approximate a global search of the
    (low-dimensional) box. The box is the task's own: there is no
    separate flag for it (``docs/adr/0002``).

    Parameters
    ----------
    task : Task
        The task to benchmark. Its ``loss_fn``, ``pass_rng``, ``has_aux``
        and ``loaded_dataset`` define the objective via the canonical
        ``loss_fn(model, [key=...], **dataset)`` contract. A
        :class:`~l2co_tasks.Box` in its ``constraints`` is searched
        inside.
    seed : int, optional
        Seed for restart initialisation and per-step keys; makes the
        estimate deterministic, by default 0.
    n_restarts : int, optional
        Number of independent restarts, by default 3.
    n_steps : int, optional
        Number of Adam steps per restart, by default 1000.
    lr : float, optional
        Initial (peak) learning rate of the cosine-decay schedule, by
        default 1e-2.
    use_lbfgs : bool, optional
        Run the L-BFGS depth arm on tasks without a box, by default
        ``True``.
    lbfgs_steps : int or None, optional
        Number of L-BFGS steps per restart; defaults to
        ``_LBFGS_DEFAULT_STEPS`` (a convergence budget independent of the
        Adam breadth budget ``n_steps``) when ``None``. On these smooth
        losses L-BFGS converges well within that many steps, so even a
        small-``n_steps`` build still gets a floor-respecting estimate.

    Returns
    -------
    float
        The lowest loss found -- the estimated global minimum.

    Raises
    ------
    ValueError
        If the task has an inequality or equality constraint, or a box
        with an infinite bound.
    """
    box = _search_box(task)
    n_steps = max(int(n_steps), 1)
    n_restarts = max(int(n_restarts), 1)
    if lbfgs_steps is None:
        lbfgs_steps = _LBFGS_DEFAULT_STEPS
    else:
        lbfgs_steps = max(int(lbfgs_steps), 1)
    # L-BFGS is a sound *strong* arm only on unbounded weight-space tasks;
    # on boxed tasks its line search fights the projection, so breadth
    # (resampled Adam restarts) sets the floor there instead.
    run_lbfgs = bool(use_lbfgs) and box is None
    # Materialise the dataset once so the loss closure is a clean
    # function of the model alone (no file I/O baked into the trace).
    dataset = {k: jnp.asarray(v) for k, v in task.loaded_dataset.items()}
    key = jr.key(int(seed))
    restart_keys = jr.split(key, n_restarts)
    opt = optax.adam(optax.cosine_decay_schedule(lr, n_steps))

    def loss_at(params, static, step_key):
        """Full-batch loss of the model assembled from ``params``."""
        model = eqx.combine(params, static)
        if task.pass_rng:
            out = task.loss_fn(model, key=step_key, **dataset)
        else:
            out = task.loss_fn(model, **dataset)
        return out[0] if task.has_aux else out

    @eqx.filter_jit
    def run_restart(start, restart_key):
        """Train ``start`` for ``n_steps`` and return the best loss seen."""
        params, static = eqx.partition(start, eqx.is_inexact_array)
        opt_state = opt.init(params)

        def body(carry, step):
            params, opt_state, best = carry
            step_key = jr.fold_in(restart_key, step)
            loss, grads = eqx.filter_value_and_grad(loss_at)(
                params, static, step_key
            )
            updates, opt_state = opt.update(grads, opt_state, params)
            params = eqx.apply_updates(params, updates)
            if box is not None:
                params = jtu.tree_map(jnp.clip, params, box.lower, box.upper)
            return (params, opt_state, jnp.minimum(best, _finite(loss))), None

        init = (params, opt_state, jnp.asarray(jnp.inf))
        (params, _, best), _ = jax.lax.scan(body, init, jnp.arange(n_steps))
        # Include the final (post-update) iterate, which the scan body
        # evaluates losses *before* applying.
        final = loss_at(params, static, jr.fold_in(restart_key, n_steps))
        return jnp.minimum(best, _finite(final))

    @eqx.filter_jit
    def run_restart_lbfgs(start, restart_key):
        """L-BFGS from ``start`` at a fixed key; return the best loss seen.

        The key is fixed for the whole restart (not folded per step) so
        the loss is deterministic across L-BFGS line-search probes, which
        ``optax.value_and_grad_from_state`` relies on to reuse cached
        value/grad.
        """
        params, static = eqx.partition(start, eqx.is_inexact_array)

        def f(p):
            """Scalar loss at trainable params ``p`` (fixed restart key)."""
            return loss_at(p, static, restart_key)

        l_opt = optax.lbfgs()
        opt_state = l_opt.init(params)
        value_and_grad = optax.value_and_grad_from_state(f)

        def body(carry, _):
            params, opt_state, best = carry
            val, grad = value_and_grad(params, state=opt_state)
            updates, opt_state = l_opt.update(
                grad, opt_state, params, value=val, grad=grad, value_fn=f
            )
            params = eqx.apply_updates(params, updates)
            return (params, opt_state, jnp.minimum(best, _finite(val))), None

        init = (params, opt_state, jnp.asarray(jnp.inf))
        (params, _, best), _ = jax.lax.scan(
            body, init, None, length=lbfgs_steps
        )
        return jnp.minimum(best, _finite(f(params)))

    best_overall = jnp.asarray(jnp.inf)
    for r in range(n_restarts):
        rk = restart_keys[r]
        if r == 0 and box is None:
            start = task.model
        else:
            start = _resample_leaves(task.model, rk, box)
        best_overall = jnp.minimum(best_overall, run_restart(start, rk))
        if run_lbfgs:
            best_overall = jnp.minimum(
                best_overall, run_restart_lbfgs(start, rk)
            )
    return float(best_overall)


# =============================================================================
