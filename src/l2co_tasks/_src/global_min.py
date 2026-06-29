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
runs a seeded, fixed-budget, multi-restart Adam search and returns the
lowest full-batch loss observed. It mirrors the short Adam loop used by
the optimizability test battery (``tests/test_task_optimizability.py``):
each restart trains the trainable (inexact-array) leaves of the model on
the task's full-batch loss, unit-box tasks are clipped back into
``[0, 1]`` after every step, and a cosine-decayed learning rate anneals
the search.

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
from .task import Task

# =============================================================================


def _finite(loss: jax.Array) -> jax.Array:
    """Map a non-finite loss to ``+inf`` so it loses every ``minimum``.

    A divergent step (e.g. the Adam-meta search clipping into a regime
    with a huge decoded inner learning rate) yields ``nan``/``inf``;
    folding it to ``+inf`` lets the running ``jnp.minimum`` ignore it
    instead of being poisoned by ``nan`` propagation.
    """
    return jnp.where(jnp.isfinite(loss), loss, jnp.inf)


def _resample_leaves(model, key: jax.Array, clip_to_unit: bool):
    """Return ``model`` with its inexact-array leaves freshly sampled.

    Mirrors ``sample_model`` in the test contract helpers: unbounded
    (weight-space) tasks draw from a standard normal, unit-box tasks draw
    uniformly from ``[0, 1]`` so the restart respects the boundaries.

    Parameters
    ----------
    model : PyTree
        Model whose structure is used as the template.
    key : jax.Array
        Random key.
    clip_to_unit : bool
        Whether the task is optimised over the unit box ``[0, 1]^d``.

    Returns
    -------
    PyTree
        A model with resampled trainable leaves.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jtu.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    pairs = zip(leaves, keys, strict=True)
    if clip_to_unit:
        new = [jr.uniform(k, x.shape, x.dtype) for x, k in pairs]
    else:
        new = [jr.normal(k, x.shape, x.dtype) for x, k in pairs]
    return eqx.combine(jtu.tree_unflatten(treedef, new), static)


def estimate_global_min(
    task: Task,
    *,
    clip_to_unit: bool = False,
    seed: int = 0,
    n_restarts: int = 3,
    n_steps: int = 1000,
    lr: float = 1e-2,
) -> float:
    """Estimate a task's global minimum with a benchmark optimiser.

    Runs ``n_restarts`` independent Adam searches over the task's
    trainable parameters -- each for ``n_steps`` steps on the full-batch
    loss with a cosine-decayed learning rate -- and returns the lowest
    loss observed across all restarts and steps. The result is the
    "empirical" ``global_min`` recorded by the supervised-learning and
    meta factories.

    The first restart starts from ``task.model`` (the factory's
    initialised weights), which is usually a strong starting point; the
    remaining restarts start from resampled leaves for diversity. For
    unit-box tasks (``clip_to_unit=True``) *every* restart is resampled,
    since the factory's initial model sits at the box corner where the
    projected step can stall.

    Parameters
    ----------
    task : Task
        The task to benchmark. Its ``loss_fn``, ``pass_rng``, ``has_aux``
        and ``loaded_dataset`` define the objective via the canonical
        ``loss_fn(model, [key=...], **dataset)`` contract.
    clip_to_unit : bool, optional
        Project parameters back into ``[0, 1]`` after each step (for
        tasks optimised over the unit box, e.g. the Adam-meta task), by
        default ``False``.
    seed : int, optional
        Seed for restart initialisation and per-step keys; makes the
        estimate deterministic, by default 0.
    n_restarts : int, optional
        Number of independent Adam restarts, by default 3.
    n_steps : int, optional
        Number of Adam steps per restart, by default 1000.
    lr : float, optional
        Initial (peak) learning rate of the cosine-decay schedule, by
        default 1e-2.

    Returns
    -------
    float
        The lowest loss found -- the estimated global minimum.
    """
    n_steps = max(int(n_steps), 1)
    n_restarts = max(int(n_restarts), 1)
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
            if clip_to_unit:
                params = jtu.tree_map(lambda x: jnp.clip(x, 0.0, 1.0), params)
            return (params, opt_state, jnp.minimum(best, _finite(loss))), None

        init = (params, opt_state, jnp.asarray(jnp.inf))
        (params, _, best), _ = jax.lax.scan(body, init, jnp.arange(n_steps))
        # Include the final (post-update) iterate, which the scan body
        # evaluates losses *before* applying.
        final = loss_at(params, static, jr.fold_in(restart_key, n_steps))
        return jnp.minimum(best, _finite(final))

    best_overall = jnp.asarray(jnp.inf)
    for r in range(n_restarts):
        rk = restart_keys[r]
        if r == 0 and not clip_to_unit:
            start = task.model
        else:
            start = _resample_leaves(task.model, rk, clip_to_unit)
        best_overall = jnp.minimum(best_overall, run_restart(start, rk))
    return float(best_overall)


# =============================================================================
