"""Shared helpers for the task-contract suite.

These helpers replicate, test-side, the canonical loss-invocation
contract that ``l2co``/``rl2co`` rely on (see
``l2co/_src/update_class.py`` and ``l2co/_src/tree_utils.py``):

    sample = {k: v[idxs] for k, v in task.loaded_dataset.items()}
    loss = (task.loss_fn(model, key=key, **sample) if task.pass_rng
            else task.loss_fn(model, **sample))

They are intentionally kept inside ``l2co-tasks`` (no ``l2co`` import)
so the package stays standalone. Every helper operates only on the
trainable, inexact-array part of ``task.model`` -- the same leaves that
``Task.dimensionality`` counts via ``eqx.filter(model, eqx.is_inexact_array)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree

from l2co_tasks import Task


def _inexact_leaves(model: PyTree) -> tuple[list, Any, PyTree]:
    """Split ``model`` into its inexact-array leaves and static remainder.

    Parameters
    ----------
    model : PyTree
        The model to split.

    Returns
    -------
    leaves : list
        Flat list of the inexact-array leaves.
    treedef : Any
        Tree definition of the trainable partition (for re-assembly).
    static : PyTree
        The non-trainable remainder, ready for ``eqx.combine``.
    """
    params, static = eqx.partition(model, eqx.is_inexact_array)
    leaves, treedef = jtu.tree_flatten(params)
    return leaves, treedef, static


def sample_model(task: Task, key: PRNGKeyArray, domain: str) -> PyTree:
    """Return a copy of ``task.model`` with freshly sampled parameters.

    Each inexact-array leaf is replaced by a sample of the same shape and
    dtype. ``domain="unbounded"`` draws from a standard normal;
    ``domain="unit"`` draws uniformly from ``[0, 1]`` so the sample
    *respects the boundaries* of unit-box tasks (BBOB, CEC2005, meta).

    Parameters
    ----------
    task : Task
        Task whose model structure is used as the template.
    key : PRNGKeyArray
        Random key.
    domain : str
        Either ``"unbounded"`` or ``"unit"``.

    Returns
    -------
    PyTree
        A model with sampled parameters.
    """
    leaves, treedef, static = _inexact_leaves(task.model)
    keys = jr.split(key, len(leaves))
    pairs = zip(leaves, keys, strict=True)
    if domain == "unit":
        new = [jr.uniform(k, x.shape, x.dtype) for x, k in pairs]
    elif domain == "unbounded":
        new = [jr.normal(k, x.shape, x.dtype) for x, k in pairs]
    else:
        raise ValueError(f"unknown domain {domain!r}")
    return eqx.combine(jtu.tree_unflatten(treedef, new), static)


def fill_model(task: Task, value: float) -> PyTree:
    """Return a copy of ``task.model`` with every parameter set to ``value``.

    Used with ``jnp.nan`` to probe robustness to non-finite inputs.

    Parameters
    ----------
    task : Task
        Task whose model structure is used as the template.
    value : float
        Constant value written into every inexact-array leaf.

    Returns
    -------
    PyTree
        A model whose trainable leaves are all ``value``.
    """
    params, static = eqx.partition(task.model, eqx.is_inexact_array)
    params = jtu.tree_map(lambda x: jnp.full(x.shape, value, x.dtype), params)
    return eqx.combine(params, static)


def stack_models(models: list[PyTree]) -> PyTree:
    """Stack a list of models into a single batched model.

    Only the inexact-array leaves are stacked (along a new leading axis);
    the static remainder is taken from the first model and broadcast.
    The result is suitable for ``eqx.filter_vmap`` with default
    ``in_axes``.

    Parameters
    ----------
    models : list of PyTree
        Models sharing identical structure.

    Returns
    -------
    PyTree
        A single model whose trainable leaves carry a leading batch axis.
    """
    params = [eqx.partition(m, eqx.is_inexact_array)[0] for m in models]
    _, static = eqx.partition(models[0], eqx.is_inexact_array)
    stacked = jtu.tree_map(lambda *xs: jnp.stack(xs), *params)
    return eqx.combine(stacked, static)


def minibatch_idxs(
    task: Task, key: PRNGKeyArray, dataset: dict | None = None
) -> jax.Array | None:
    """Draw minibatch indices for a dataset-backed task.

    Returns ``None`` when the task carries no dataset or its
    ``batch_size`` is ``None`` or not smaller than the dataset length
    (i.e. full-batch evaluation).

    Parameters
    ----------
    task : Task
        The task.
    key : PRNGKeyArray
        Random key for index sampling.
    dataset : dict, optional
        Already-loaded dataset; loaded from ``task`` if omitted.

    Returns
    -------
    jax.Array or None
        Integer index array of length ``batch_size``, or ``None``.
    """
    if dataset is None:
        dataset = task.loaded_dataset
    if not dataset or task.batch_size is None:
        return None
    n = jtu.tree_leaves(dataset)[0].shape[0]
    if task.batch_size >= n:
        return None
    return jr.choice(key, n, shape=(task.batch_size,), replace=False)


def loss_closure(
    task: Task,
    *,
    dataset: dict | None = None,
    idxs: jax.Array | None = None,
    key: PRNGKeyArray | None = None,
) -> Callable[[PyTree], jax.Array]:
    """Build a pure ``model -> scalar`` loss closure for ``task``.

    The dataset (and any minibatch selection) is resolved **eagerly**,
    outside the returned function, so the closure is a clean function of
    the model alone -- safe to wrap with ``eqx.filter_jit`` /
    ``filter_vmap`` / ``filter_grad`` without baking file I/O into the
    trace. ``has_aux`` losses are unwrapped to their scalar first element.

    Parameters
    ----------
    task : Task
        The task providing ``loss_fn``, ``pass_rng`` and ``has_aux``.
    dataset : dict, optional
        Pre-loaded dataset; loaded from ``task`` when omitted.
    idxs : jax.Array, optional
        Minibatch indices; when given the dataset arrays are indexed.
    key : PRNGKeyArray, optional
        Key forwarded to the loss when ``task.pass_rng`` is set.

    Returns
    -------
    Callable[[PyTree], jax.Array]
        A function mapping a model to a scalar loss.
    """
    if dataset is None:
        dataset = task.loaded_dataset
    if idxs is not None and dataset:
        sample = jtu.tree_map(lambda a: a[idxs], dataset)
    else:
        sample = dataset

    def f(model: PyTree) -> jax.Array:
        """Evaluate the task loss at ``model``."""
        if task.pass_rng:
            out = task.loss_fn(model, key=key, **sample)
        else:
            out = task.loss_fn(model, **sample)
        return out[0] if task.has_aux else out

    return f


def evaluate_task_loss(
    task: Task,
    model: PyTree,
    *,
    key: PRNGKeyArray | None = None,
    dataset: dict | None = None,
    idxs: jax.Array | None = None,
) -> jax.Array:
    """Evaluate ``task``'s loss at ``model`` via the canonical contract.

    Thin eager wrapper around :func:`loss_closure`.

    Parameters
    ----------
    task : Task
        The task.
    model : PyTree
        Model to evaluate.
    key : PRNGKeyArray, optional
        Key forwarded when ``task.pass_rng`` is set.
    dataset : dict, optional
        Pre-loaded dataset; loaded from ``task`` when omitted.
    idxs : jax.Array, optional
        Minibatch indices.

    Returns
    -------
    jax.Array
        The scalar loss value.
    """
    return loss_closure(task, dataset=dataset, idxs=idxs, key=key)(model)
