"""Optimizability battery: every task must be a non-degenerate problem.

The contract suite (``test_task_contract.py``) proves each loss is
finite, deterministic and differentiable at sampled points -- but a
loss that returned the same constant everywhere would pass all of it.
This suite closes that gap, parametrised over the same
:class:`~tests.task_cases.TaskCase` registry via the shared ``tc``
fixture:

* the loss is not constant across sampled in-domain models,
* the gradient at a sampled in-domain model is not identically zero,
* a short full-batch adam run improves on the initial loss.

``task.model``'s parameter values are arbitrary placeholders (default
Equinox initialisation, or all zeros for raw-array models) -- only its
static structure is contractual -- so every criterion here evaluates at
*sampled* parameters, obtained by partitioning ``task.model`` and
re-filling the trainable leaves (``sample_model``).

Heavy cases inherit ``@pytest.mark.slow`` from their registry marks.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax

from ._contract_utils import evaluate_task_loss, loss_closure, sample_model

N_STEPS = 40
N_SAMPLES = 4


def test_loss_is_not_constant(tc):
    """The loss varies across sampled in-domain models.

    A fixed evaluation key neutralises ``pass_rng`` stochasticity, so
    any variation must come from the model itself.
    """
    task = tc.task
    keys = jr.split(jr.key(21), N_SAMPLES)
    losses = jnp.stack(
        [
            evaluate_task_loss(
                task, sample_model(task, k, tc.case.domain), key=jr.key(0)
            )
            for k in keys
        ]
    )
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.unique(losses).size > 1, (
        "loss is constant across sampled models"
    )


def test_gradient_is_not_dead(tc):
    """The gradient at a sampled in-domain model is not all-zero.

    Sampled rather than ``task.model``: the placeholder parameters can
    sit at symmetric points (all zeros) where gradients vanish without
    the task being degenerate.
    """
    task = tc.task
    model = sample_model(task, jr.key(20), tc.case.domain)
    f = loss_closure(task, key=jr.key(0))
    grad = eqx.filter_grad(f)(model)
    leaves = jtu.tree_leaves(eqx.filter(grad, eqx.is_inexact_array))
    sq_norm = sum(jnp.sum(g**2) for g in leaves)
    assert jnp.isfinite(sq_norm)
    assert sq_norm > 0.0, "gradient at the initial model is identically zero"


def test_short_adam_run_decreases_loss(tc):
    """A short full-batch adam run improves on the initial loss.

    The run starts from a sampled in-domain model (a fixed key keeps it
    deterministic) rather than the factory's initial model: BBOB/CEC
    factories start at the unit-box corner ``0``, where the projected
    update can pin the iterate against the boundary. The training key
    is folded per step (mirroring l2co's per-step key handling) while
    the evaluation key is fixed, so the recorded loss sequence is
    deterministic even for ``pass_rng`` tasks. Unit-domain tasks are
    clipped back into ``[0, 1]`` after each update so the iterates stay
    where the task semantics hold. The assertion is
    best-seen-vs-initial, which is robust to adam overshoot but fails
    hard for constant or unoptimizable losses.
    """
    task = tc.task
    dataset = task.loaded_dataset
    start = sample_model(task, jr.key(22), tc.case.domain)
    params, static = eqx.partition(start, eqx.is_inexact_array)
    opt = optax.adam(tc.case.lr)
    opt_state = opt.init(params)
    clip_unit = tc.case.domain == "unit"

    def eval_loss(p):
        """Deterministic loss of the current iterate (fixed key)."""
        return evaluate_task_loss(
            task, eqx.combine(p, static), key=jr.key(1), dataset=dataset
        )

    @eqx.filter_jit
    def step(params, opt_state, key):
        """One adam update on the full-batch loss."""

        def f(p):
            return evaluate_task_loss(
                task, eqx.combine(p, static), key=key, dataset=dataset
            )

        grads = eqx.filter_grad(f)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        if clip_unit:
            params = jtu.tree_map(lambda x: jnp.clip(x, 0.0, 1.0), params)
        return params, opt_state

    losses = [eval_loss(params)]
    for i in range(N_STEPS):
        params, opt_state = step(params, opt_state, jr.fold_in(jr.key(2), i))
        losses.append(eval_loss(params))
    losses = jnp.stack(losses)
    assert jnp.all(jnp.isfinite(losses))
    assert jnp.min(losses[1:]) < losses[0], (
        f"adam made no progress: initial {losses[0]}, "
        f"best {jnp.min(losses[1:])}"
    )
