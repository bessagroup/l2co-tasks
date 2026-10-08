"""
Host objectives: losses computed outside JAX, behind an ordinary loss.

A **host objective** is a task's objective computed outside JAX -- compiled
Fortran, a simulator -- that JAX can neither trace nor differentiate.
:func:`host_loss` wraps one into an ordinary ``loss_fn(model)`` that every
optimizer can trace, vectorize and differentiate, so nothing that consumes
a :class:`~l2co_tasks.Task` needs to know (ADR 0003).

An adapter for one suite supplies an *opener*: a picklable, hashable,
zero-argument callable that returns the objective. The objective has
``value(x) -> float`` and, optionally, ``value_and_grad(x) -> (float,
ndarray)`` and ``hessian(x) -> ndarray``; ``x`` is the model's
floating-point parameters flattened into one float64 vector. Without
``value_and_grad`` the task has no gradient; without ``hessian`` it has
no second derivatives (ADR 0005).

How the loss behaves:

* **One host call per evaluation.** The value is a
  :func:`jax.pure_callback` inside a :func:`jax.custom_jvp`, whose
  derivative rule asks the host for value and gradient in one call. It has
  to be ``custom_jvp``: optimistix takes gradients in forward mode
  (``jax.linearize``), which a ``custom_vjp`` does not support.
* **Second derivatives, when the objective has them.** The gradient in
  that rule is itself a ``custom_jvp``, whose rule asks the host for the
  dense Hessian. ``jax.hessian`` of the loss therefore makes one host
  Hessian call. First derivatives compute exactly what they did before
  (ADR 0005).
* **Batched under vmap.** Every point of a vmapped population, across
  every realization, reaches the host in one callback, which evaluates
  them one at a time.
* **Last-point caches, one per kind of request.** A value request for
  the point last asked for a value, a value-and-gradient request for the
  point last asked for both, or a Hessian request for the point last
  asked for one, is answered from the cache: optimizers often ask twice
  in a row. The kinds never answer for each other,
  because an objective's value and value-and-gradient routines may differ
  in the last bit; mixing them would make a run's numbers depend on what
  ran before it in the same process.
* **Opened once per process, when first traced.** The loss object holds
  only its opener, so a ``Task`` holding it can be saved and loaded without
  opening anything; the objective is opened the first time the loss is
  traced, where a failure is an ordinary Python exception.
* **float64 on the host.** Traced with lower-precision parameters, the
  loss warns: the host then sees rounded inputs and returns rounded
  results.

An adapter must keep to three rules (ADR 0003): no side effects that
depend on how often it is called (JAX may skip or repeat a pure callback),
never more than one billed evaluation per host call (so no hidden finite
differences), and float64 arithmetic.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
import threading
import warnings
from collections.abc import Callable, Hashable
from typing import Any, Protocol

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = [
    "HostLoss",
    "HostObjective",
    "host_loss",
]


class HostObjective(Protocol):
    """An objective computed outside JAX (ADR 0003).

    Only :meth:`value` is required. An objective that can also supply its
    gradient defines ``value_and_grad(x) -> (float, ndarray)``, returning
    the value and the gradient with ``x``'s shape from one evaluation;
    without it, the task has no gradient and differentiating its loss
    raises :class:`TypeError`. One that can also supply second
    derivatives defines ``hessian(x) -> ndarray``, the dense ``(n, n)``
    Hessian from one evaluation; without it, differentiating the loss
    twice raises :class:`TypeError` (ADR 0005).

    Both methods receive ``x`` as a one-dimensional float64 array, the
    model's floating-point parameters flattened in
    :func:`jax.flatten_util.ravel_pytree` order. They must be
    deterministic: the loss caches the last point of each, and JAX may
    skip or repeat calls.
    """

    def value(self, x: np.ndarray) -> float:
        """The objective at ``x``."""
        ...


#: Objectives opened in this process, by opener.
_OPENED: dict[Hashable, _OpenObjective] = {}

#: Serializes opening and every host call. Fortran and simulation codes
#: are rarely safe to enter from two threads at once.
_LOCK = threading.RLock()


class _OpenObjective:
    """An opened host objective, its last-point caches and its traced loss.

    Attributes
    ----------
    objective : HostObjective
        The objective the opener returned.
    differentiable : bool
        Whether the objective supplies ``value_and_grad``.
    twice_differentiable : bool
        Whether it also supplies ``hessian``.
    fn : Callable
        ``fn(x) -> loss`` for a flat parameter vector ``x``: the traced
        function behind every :class:`HostLoss` with this opener.
    """

    def __init__(self, opener: Callable[[], HostObjective]):
        self._opener = opener
        self.objective = opener()
        self.differentiable = callable(
            getattr(self.objective, "value_and_grad", None)
        )
        self.twice_differentiable = self.differentiable and callable(
            getattr(self.objective, "hessian", None)
        )
        # One last-point cache per kind of request; see the module
        # docstring for why a value is never served from a gradient call.
        self._value_cache: tuple[bytes, float] | None = None
        self._grad_cache: tuple[bytes, float, np.ndarray] | None = None
        self._hessian_cache: tuple[bytes, np.ndarray] | None = None
        self.fn = _traced(self)

    def value(self, x: np.ndarray) -> float:
        """The value at ``x``; cached if the last value request was ``x``."""
        key = x.tobytes()
        if self._value_cache is None or self._value_cache[0] != key:
            self._value_cache = (key, float(self.objective.value(x)))
        return self._value_cache[1]

    def value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Value and gradient at ``x``, in one host evaluation; cached if
        the last value-and-gradient request was ``x``."""
        key = x.tobytes()
        if self._grad_cache is None or self._grad_cache[0] != key:
            f, g = self.objective.value_and_grad(x)
            g = np.asarray(g, dtype=np.float64)
            if g.shape != x.shape:
                raise ValueError(
                    f"{self._opener!r}: value_and_grad returned a gradient "
                    f"of shape {g.shape} for a point of shape {x.shape}"
                )
            self._grad_cache = (key, float(f), g)
        return self._grad_cache[1], self._grad_cache[2]

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """The Hessian at ``x``, in one host evaluation; cached if the last
        Hessian request was ``x``."""
        key = x.tobytes()
        if self._hessian_cache is None or self._hessian_cache[0] != key:
            h = np.asarray(self.objective.hessian(x), dtype=np.float64)
            if h.shape != (x.size, x.size):
                raise ValueError(
                    f"{self._opener!r}: hessian returned shape {h.shape} for "
                    f"a point of shape {x.shape}"
                )
            self._hessian_cache = (key, h)
        return self._hessian_cache[1]

    def no_hessian(self) -> TypeError:
        """The error for differentiating twice without a Hessian."""
        return TypeError(
            f"{self._opener!r} opens a host objective without hessian, so "
            "this task has no second derivatives: jax.hessian of its loss, "
            "or an optimizer that needs exact Hessians, cannot run on it "
            "(ADR 0005)"
        )

    def no_gradient(self) -> TypeError:
        """The error for differentiating an objective without a gradient."""
        return TypeError(
            f"{self._opener!r} opens a host objective without "
            "value_and_grad, so this task has no gradient: an optimizer "
            "that uses gradients cannot run on it (ADR 0003)"
        )


def _open(opener: Callable[[], HostObjective]) -> _OpenObjective:
    """The objective behind ``opener``, opened once per process."""
    with _LOCK:
        opened = _OPENED.get(opener)
        if opened is None:
            opened = _OPENED[opener] = _OpenObjective(opener)
    return opened


def _host_value(opened: _OpenObjective, x: np.ndarray) -> np.ndarray:
    """Host side of the value callback; ``x`` has shape ``(..., n)``."""
    points = np.asarray(x, dtype=np.float64)
    rows = points.reshape(-1, points.shape[-1])
    out = np.empty(rows.shape[0])
    with _LOCK:
        for i, row in enumerate(rows):
            out[i] = opened.value(row)
    return out.reshape(points.shape[:-1]).astype(x.dtype)


def _host_value_and_grad(
    opened: _OpenObjective, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Host side of the value-and-gradient callback."""
    points = np.asarray(x, dtype=np.float64)
    rows = points.reshape(-1, points.shape[-1])
    f = np.empty(rows.shape[0])
    g = np.empty(rows.shape)
    with _LOCK:
        for i, row in enumerate(rows):
            f[i], g[i] = opened.value_and_grad(row)
    return (
        f.reshape(points.shape[:-1]).astype(x.dtype),
        g.reshape(points.shape).astype(x.dtype),
    )


def _host_hessian(opened: _OpenObjective, x: np.ndarray) -> np.ndarray:
    """Host side of the Hessian callback; ``(..., n)`` to ``(..., n, n)``."""
    points = np.asarray(x, dtype=np.float64)
    rows = points.reshape(-1, points.shape[-1])
    h = np.empty((rows.shape[0], rows.shape[1], rows.shape[1]))
    with _LOCK:
        for i, row in enumerate(rows):
            h[i] = opened.hessian(row)
    return h.reshape(points.shape + points.shape[-1:]).astype(x.dtype)


def _traced(opened: _OpenObjective) -> Callable:
    """The traceable, differentiable ``x -> loss`` for one objective."""

    @jax.custom_jvp
    def value_and_grad(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.pure_callback(
            functools.partial(_host_value_and_grad, opened),
            (
                jax.ShapeDtypeStruct((), x.dtype),
                jax.ShapeDtypeStruct(x.shape, x.dtype),
            ),
            x,
            vmap_method="expand_dims",
        )

    @value_and_grad.defjvp
    def value_and_grad_jvp(primals, tangents):
        # Reached only when the loss is differentiated twice.
        (x,), (t,) = primals, tangents
        if not opened.twice_differentiable:
            raise opened.no_hessian()
        f, g = value_and_grad(x)
        h = jax.pure_callback(
            functools.partial(_host_hessian, opened),
            jax.ShapeDtypeStruct(x.shape + x.shape, x.dtype),
            x,
            vmap_method="expand_dims",
        )
        return (f, g), (jnp.dot(g, t), h @ t)

    @jax.custom_jvp
    def fn(x: jax.Array) -> jax.Array:
        return jax.pure_callback(
            functools.partial(_host_value, opened),
            jax.ShapeDtypeStruct((), x.dtype),
            x,
            vmap_method="expand_dims",
        )

    @fn.defjvp
    def fn_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        if not opened.differentiable:
            raise opened.no_gradient()
        f, g = value_and_grad(x)
        return f, jnp.dot(g, t)

    return fn


class HostLoss(eqx.Module):
    """``loss_fn(model)`` for a task whose objective is a host objective.

    Built by :func:`host_loss`. It holds only its opener, so it pickles
    with a :class:`~l2co_tasks.Task` (``Task.save``), and two losses with
    equal openers are equal and hash equal, which lets jit caches survive a
    rebuilt task. The objective is opened the first time the loss is
    called or traced in a process.

    Attributes
    ----------
    opener : Callable[[], HostObjective]
        Picklable, hashable, zero-argument callable returning the
        objective.
    """

    opener: Callable[[], HostObjective] = eqx.field(static=True)

    def __call__(self, model: PyTree, **sample: Any) -> jax.Array:
        """The objective at ``model``'s floating-point parameters.

        Parameters
        ----------
        model : PyTree
            The task's model; its floating-point array leaves, flattened,
            are the point the host objective sees.
        **sample : Any
            Must be empty: a host objective takes no dataset batch and no
            random key yet (ADR 0003).

        Returns
        -------
        jax.Array
            The objective value, a scalar in the parameters' dtype.

        Raises
        ------
        TypeError
            If a dataset batch or a key is passed.
        """
        if sample:
            raise TypeError(
                "a host objective takes no dataset batch or random key "
                f"(got {sorted(sample)}); see ADR 0003"
            )
        x, _ = ravel_pytree(eqx.filter(model, eqx.is_inexact_array))
        if x.dtype != jnp.float64:
            warnings.warn(
                f"host objective {self.opener!r} traced with {x.dtype} "
                "parameters: the host computes in float64 but receives "
                "rounded inputs and returns rounded results. Enable "
                "jax_enable_x64.",
                UserWarning,
                stacklevel=2,
            )
        return _open(self.opener).fn(x)


def host_loss(opener: Callable[[], HostObjective]) -> HostLoss:
    """Wrap a host objective into a ``loss_fn`` for a :class:`Task`.

    The returned loss is traceable, vmappable and -- when the objective
    supplies ``value_and_grad`` -- differentiable in forward and reverse
    mode, so every optimizer can run on the task (ADR 0003). When it also
    supplies ``hessian``, the loss can be differentiated twice
    (``jax.hessian``; ADR 0005). Pass it as
    ``Task(loss_fn=host_loss(opener), ...)`` with ``pass_rng=False``, no
    dataset and ``has_aux=False``.

    Parameters
    ----------
    opener : Callable[[], HostObjective]
        Zero-argument callable returning the objective. It must be
        picklable, so the task can be saved, and hashable, with equal
        openers opening the same objective: the objective is opened once
        per process and shared by every loss with an equal opener.

    Returns
    -------
    HostLoss
        The loss, ``loss_fn(model) -> scalar``.

    Raises
    ------
    TypeError
        If ``opener`` is not callable or not hashable.
    """
    if not callable(opener):
        raise TypeError(f"opener must be callable, got {opener!r}")
    try:
        hash(opener)
    except TypeError as error:
        raise TypeError(
            f"opener must be hashable, so equal openers share one opened "
            f"objective per process; got {opener!r}"
        ) from error
    return HostLoss(opener)
