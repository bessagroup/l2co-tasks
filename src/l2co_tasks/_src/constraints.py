"""
Constraints a task declares on its parameters (ADR 0002).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import abc
import math
from collections.abc import Callable, Iterable
from typing import ClassVar

# Third-party
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, PyTree

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


class Constraint(eqx.Module):
    """
    A condition every solution of a task must satisfy.

    Calling a constraint on a model returns its raw values as a 1-D
    float array, read as "``<= 0`` is satisfied" (for an
    :class:`Equality`, "``= 0``"). :meth:`violation` returns how far each
    value misses beyond its tolerance, so it is zero exactly where the
    constraint counts as satisfied.

    Constraints are deterministic functions of the model alone: they
    receive the same ``model`` as the task's ``loss_fn`` and nothing
    else -- no PRNG key and no data batch, whatever the task's
    ``pass_rng``.

    The concrete kinds are :class:`Inequality`, :class:`Equality` and
    :class:`Box`.
    """

    kind: ClassVar[str]

    @abc.abstractmethod
    def __call__(self, model: PyTree) -> Array:
        """
        Evaluate the constraint's raw values.

        Parameters
        ----------
        model : PyTree
            The task's model.

        Returns
        -------
        Array
            Raw constraint values, flattened.
        """

    @abc.abstractmethod
    def violation(self, model: PyTree) -> Array:
        """
        Evaluate how far each value misses beyond its tolerance.

        Parameters
        ----------
        model : PyTree
            The task's model.

        Returns
        -------
        Array
            Non-negative violations, zero where satisfied.
        """


def _check_name_and_tol(name: str, tol: float) -> None:
    """Reject an empty name and a negative or non-finite tolerance."""
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"a constraint needs a non-empty string name, got {name!r}"
        )
    if not math.isfinite(tol) or tol < 0:
        raise ValueError(
            f"constraint {name!r}: tolerance must be finite and "
            f"non-negative, got {tol!r}"
        )


class Inequality(Constraint):
    """
    An inequality constraint ``fn(model) <= 0``.

    Parameters
    ----------
    fn : Callable[[PyTree], Array]
        Maps the model to the constraint values. Any output shape is
        accepted and flattened, so one ``Inequality`` can hold one
        constraint or many.
    name : str
        Name of the constraint, unique within its task. Part of the
        task's identity.
    tol : float, optional
        Feasibility tolerance: satisfied where ``fn(model) <= tol``.
        Defaults to ``0.0``. Part of the task's identity.

    Attributes
    ----------
    fn : Callable[[PyTree], Array]
        The constraint function.
    name : str
        Name of the constraint.
    tol : float
        Feasibility tolerance.
    """

    kind: ClassVar[str] = "inequality"
    fn: Callable[[PyTree], Array] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tol: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        _check_name_and_tol(self.name, self.tol)

    def __call__(self, model: PyTree) -> Array:
        """See :meth:`Constraint.__call__`."""
        return jnp.ravel(self.fn(model))

    def violation(self, model: PyTree) -> Array:
        """``max(0, fn(model) - tol)``; see :meth:`Constraint.violation`."""
        return jnp.maximum(self(model) - self.tol, 0.0)


class Equality(Constraint):
    """
    An equality constraint ``fn(model) = 0``.

    Parameters
    ----------
    fn : Callable[[PyTree], Array]
        Maps the model to the constraint values. Any output shape is
        accepted and flattened, so one ``Equality`` can hold one
        constraint or many.
    name : str
        Name of the constraint, unique within its task. Part of the
        task's identity.
    tol : float, optional
        Feasibility tolerance: satisfied where ``|fn(model)| <= tol``.
        Defaults to ``1e-4``, the CEC 2006/2017 convention. Part of the
        task's identity.

    Attributes
    ----------
    fn : Callable[[PyTree], Array]
        The constraint function.
    name : str
        Name of the constraint.
    tol : float
        Feasibility tolerance.
    """

    kind: ClassVar[str] = "equality"
    fn: Callable[[PyTree], Array] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    tol: float = eqx.field(static=True, default=1e-4)

    def __check_init__(self):
        _check_name_and_tol(self.name, self.tol)

    def __call__(self, model: PyTree) -> Array:
        """See :meth:`Constraint.__call__`."""
        return jnp.ravel(self.fn(model))

    def violation(self, model: PyTree) -> Array:
        """``max(0, |fn(model)| - tol)``; see :meth:`Constraint.violation`."""
        return jnp.maximum(jnp.abs(self(model)) - self.tol, 0.0)


class Box(Constraint):
    """
    Lower and upper limits on every parameter: ``lower <= x <= upper``.

    A box is a hard domain: a task's starting model must lie inside it.
    It has no tolerance.

    Parameters
    ----------
    lower, upper : float or PyTree
        Either a single number, applied to every parameter, or a pytree
        matching the model's parameters exactly -- the floating-point
        array leaves that :func:`count_parameters` counts, with the same
        structure and the same shapes. An open side is ``-inf`` /
        ``inf``. A :class:`Task` casts a single number to its model's
        parameters when it is built, so a ``Box`` on a ``Task`` always
        matches its model.

    Attributes
    ----------
    lower : float or PyTree
        Lower bounds.
    upper : float or PyTree
        Upper bounds.
    """

    kind: ClassVar[str] = "box"
    lower: PyTree
    upper: PyTree

    def __call__(self, model: PyTree) -> Array:
        """
        ``concat(lower - x, x - upper)``; see :meth:`Constraint.__call__`.
        """
        params = _parameters(model)
        x = _ravel(params)
        lower = _ravel(_cast_bound(self.lower, params, "lower"))
        upper = _ravel(_cast_bound(self.upper, params, "upper"))
        return jnp.concatenate([lower - x, x - upper])

    def violation(self, model: PyTree) -> Array:
        """``max(0, self(model))``; see :meth:`Constraint.violation`."""
        return jnp.maximum(self(model), 0.0)


# =============================================================================


def _parameters(model: PyTree) -> PyTree:
    """The model's parameters: its floating-point array leaves."""
    return eqx.filter(model, eqx.is_inexact_array)


def _ravel(tree: PyTree) -> Array:
    """Concatenate a pytree's leaves into one flat array."""
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return jnp.zeros((0,))
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


def _is_single_number(bound: PyTree) -> bool:
    """Whether ``bound`` is one number rather than a pytree of bounds."""
    if isinstance(bound, bool):
        return False
    if isinstance(bound, int | float):
        return True
    return isinstance(bound, np.ndarray | jax.Array) and bound.ndim == 0


def _cast_bound(bound: PyTree, params: PyTree, side: str) -> PyTree:
    """
    Cast one side of a box to the shapes of ``params``.

    A single number is broadcast to every parameter leaf. A pytree must
    match ``params`` exactly (structure and leaf shapes); its leaves are
    converted to the parameters' dtypes. Casting an already-cast bound
    returns equal values, so the cast is idempotent.
    """
    if _is_single_number(bound):
        return jtu.tree_map(
            lambda p: jnp.full(p.shape, bound, dtype=p.dtype), params
        )
    if jtu.tree_structure(bound) != jtu.tree_structure(params):
        raise ValueError(
            f"Box {side} bound does not match the model's parameters: "
            f"expected structure {jtu.tree_structure(params)}, got "
            f"{jtu.tree_structure(bound)}"
        )

    def _leaf(b, p):
        if np.shape(b) != p.shape:
            raise ValueError(
                f"Box {side} bound has a leaf of shape {np.shape(b)} "
                f"where the model's parameter has shape {p.shape}"
            )
        return jnp.asarray(b, dtype=p.dtype)

    return jtu.tree_map(_leaf, bound, params)


def _resolve_box(box: Box, params: PyTree) -> Box:
    """Cast a box to ``params`` and check it against the start point."""
    lower = _cast_bound(box.lower, params, "lower")
    upper = _cast_bound(box.upper, params, "upper")
    lo = np.asarray(_ravel(lower))
    hi = np.asarray(_ravel(upper))
    x = np.asarray(_ravel(params))
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("Box bounds must not be NaN")
    if (lo > hi).any():
        raise ValueError(
            "Box lower bound exceeds its upper bound at "
            f"{int((lo > hi).sum())} parameter(s)"
        )
    outside = (x < lo) | (x > hi)
    if outside.any():
        raise ValueError(
            f"the task's starting model lies outside its Box at "
            f"{int(outside.sum())} parameter(s)"
        )
    return Box(lower=lower, upper=upper)


def _check_evaluates(constraint: Inequality | Equality, model: PyTree):
    """Check, without computing, that a constraint maps the model to a
    float array.
    """
    try:
        out = eqx.filter_eval_shape(constraint.fn, model)
    except Exception as e:
        raise ValueError(
            f"constraint {constraint.name!r} could not be evaluated on "
            f"the task's model: {e}"
        ) from e
    if not (
        isinstance(out, jax.ShapeDtypeStruct)
        and jnp.issubdtype(out.dtype, jnp.inexact)
    ):
        raise ValueError(
            f"constraint {constraint.name!r} must return a float array, "
            f"got {out!r}"
        )


def resolve_constraints(
    model: PyTree, constraints: Iterable[Constraint]
) -> tuple[Constraint, ...]:
    """
    Validate a task's constraints against its model and cast its box.

    The single place where the rules of ADR 0002 are enforced: called
    whenever a :class:`Task` is built.

    Parameters
    ----------
    model : PyTree
        The task's (starting) model.
    constraints : Iterable[Constraint]
        The task's constraints, in any order.

    Returns
    -------
    tuple[Constraint, ...]
        The constraints in their given order, with any ``Box`` cast to
        the model's parameters.

    Raises
    ------
    TypeError
        If an element is not an :class:`Inequality`, :class:`Equality`
        or :class:`Box`.
    ValueError
        If there is more than one ``Box``; a ``Box`` does not match the
        model's parameters, has a NaN bound or a lower bound above its
        upper bound; the starting model lies outside the ``Box``; two
        constraints share a name; or a constraint fails on the model or
        does not return a float array.
    """
    constraints = tuple(constraints)
    if not constraints:
        return ()
    for c in constraints:
        if not isinstance(c, Inequality | Equality | Box):
            raise TypeError(
                "a task's constraints must be Inequality, Equality or "
                f"Box, got {type(c).__name__}"
            )
    if sum(isinstance(c, Box) for c in constraints) > 1:
        raise ValueError("a task can have at most one Box")
    names = [c.name for c in constraints if not isinstance(c, Box)]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"constraint names must be unique: {duplicates}")

    params = _parameters(model)
    resolved = []
    for c in constraints:
        if isinstance(c, Box):
            c = _resolve_box(c, params)
        else:
            _check_evaluates(c, model)
        resolved.append(c)
    return tuple(resolved)


def constraints_identity(constraints: Iterable[Constraint]) -> tuple:
    """
    Describe constraints for a task's identity, independent of order.

    Parameters
    ----------
    constraints : Iterable[Constraint]
        Constraints already resolved by :func:`resolve_constraints`.

    Returns
    -------
    tuple
        Sorted entries: ``(kind, name, tol)`` per :class:`Inequality` /
        :class:`Equality`, and ``("box", lower_values, upper_values)``
        for the :class:`Box`, its bounds flattened to Python floats.
    """
    entries = []
    for c in constraints:
        if isinstance(c, Box):
            entries.append(
                ("box", _bound_values(c.lower), _bound_values(c.upper))
            )
        else:
            entries.append((c.kind, c.name, float(c.tol)))
    return tuple(sorted(entries))


def _bound_values(bound: PyTree) -> tuple[float, ...]:
    """A cast bound's values as a flat tuple of Python floats."""
    return tuple(float(v) for v in np.asarray(_ravel(bound)))
