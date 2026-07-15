"""Randomly-embedded BBOB tasks: low intrinsic dimension in a
high-dimensional ambient space.

Wraps a ``d``-dimensional BBOB instance ``f`` (with optimum ``z_opt``
and value ``f_opt``) as

    F(x) = f(z_opt + s * B (x - x_star))
           + (bulk_scale / 2) * ||P_perp (x - x_star)||^2

where ``x`` lives in the ambient unit box ``[0, 1]^D`` with
``D >= d``, ``B`` is a random ``(d, D)`` matrix with orthonormal rows,
``P_perp = I - B^T B`` projects onto the null space of ``B``, and
``x_star`` is a seeded uniform draw from the box. The scale ``s`` is
the width of the function's native box, so the per-coordinate spread
of ``z`` under ``x ~ U[0, 1]^D`` matches uniform sampling over the
native bounds.

The construction gives BBOB training data the geometric signature of
neural-network loss landscapes without using a neural network:

* gradients are confined to the ``d``-dimensional row space of ``B``
  (Gur-Ari et al., 2018);
* the Hessian spectrum is bulk-plus-outliers: ``d`` eigenvalues from
  ``f`` plus ``D - d`` eigenvalues at exactly ``bulk_scale`` (Sagun
  et al., 2017; Papyan, 2019) -- with ``bulk_scale = 0`` the minimiser
  is a ``(D - d)``-dimensional flat manifold, with ``bulk_scale > 0``
  it is the unique point ``x_star``;
* dimension- or basis-dependent algorithm components (covariance
  adaptation, diagonal preconditioning, coordinate-wise crossover) are
  penalised while rank-limited first-order behaviour is preserved.

Anchoring the offset at ``x_star`` guarantees the global minimum is
attained inside the box: ``F(x_star) = f_opt`` exactly, so
``Task.global_min`` stays a true lower bound (the reachability lesson
of random-embedding Bayesian optimisation, Wang et al., 2016). Local
minima of ``F`` correspond one-to-one to local minima of ``f`` (the
two gradient terms live in orthogonal subspaces), so the ``unimodal``
tag is inherited; separability is destroyed by the dense embedding,
so ``separable`` is always ``False``.

References
----------
Gur-Ari, Roberts & Dyer, "Gradient Descent Happens in a Tiny
Subspace", arXiv:1812.04754, 2018.
Li, Farkhoor, Liu & Yosinski, "Measuring the Intrinsic Dimension of
Objective Landscapes", ICLR 2018.
Sagun, Evci, Guney, Dauphin & Bottou, "Empirical Analysis of the
Hessian of Over-Parametrized Neural Networks", arXiv:1706.04454, 2017.
Papyan, "Measurements of Three-Level Hierarchical Structure in the
Outliers in the Spectrum of Deepnet Hessians", ICML 2019.
Wang, Hutter, Zoghi, Matheson & de Freitas, "Bayesian Optimization in
a Billion Dimensions via Random Embeddings", JAIR 55, 2016.

Public API
----------
create_embedded_bbob_task
    Build an embedded BBOB :class:`Task` from ``fn_name``, ``seed``,
    ``intrinsic_dim`` and ``ambient_dim`` (optionally ``bulk_scale``
    and ``noise``).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

import functools
from collections.abc import Callable

# Third-party
import bbob_jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

# Local
from .benchmark_task import add_noise
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


def compute_embedded_loss(
    model: Array,
    *,
    fn: Callable[[Array], Array],
    basis: Array,
    scale: float,
    x_star: Array,
    z_opt: Array,
    bulk_scale: float,
) -> Array:
    """Evaluate a BBOB function through a low-rank affine embedding.

    Computes ``fn(z_opt + scale * basis @ (model - x_star))`` plus,
    when ``bulk_scale > 0``, a quadratic penalty
    ``(bulk_scale / 2) * ||P_perp (model - x_star)||^2`` on the
    null-space component of the displacement from the anchor. The two
    terms live in orthogonal subspaces of the ambient space, so the
    minimum ``fn(z_opt)`` is attained exactly at ``model = x_star``.

    Parameters
    ----------
    model : Array
        Current parameter vector of shape ``(ambient_dim,)`` in the
        ambient unit box.
    fn : Callable
        The bound BBOB benchmark function, called as ``fn(z)``.
    basis : Array
        Embedding matrix of shape ``(intrinsic_dim, ambient_dim)``
        with orthonormal rows.
    scale : float
        Multiplier mapping row-space coordinates to the function's
        native range (the width of its native box).
    x_star : Array
        Anchor point of shape ``(ambient_dim,)``; the preimage of the
        function's optimum.
    z_opt : Array
        Location of the function's global minimum, shape
        ``(intrinsic_dim,)``.
    bulk_scale : float
        Curvature of the null-space penalty. ``0.0`` leaves the null
        space exactly flat (a manifold of minimisers); ``> 0`` gives
        ``ambient_dim - intrinsic_dim`` Hessian eigenvalues at this
        value and makes ``x_star`` the unique minimiser.

    Returns
    -------
    Array
        Scalar loss value.
    """
    residual = model - x_star
    coefficients = basis @ residual
    value = fn(z_opt + scale * coefficients)
    if bulk_scale > 0.0:
        perp = residual - basis.T @ coefficients
        value = value + 0.5 * bulk_scale * jnp.sum(perp**2)
    return value


def create_embedded_bbob_task(
    fn_name: str,
    seed: int,
    intrinsic_dim: int,
    ambient_dim: int,
    bulk_scale: float = 0.0,
    noise: float = 0.0,
) -> Task:
    """Create a BBOB task embedded in a higher-dimensional space.

    The ``intrinsic_dim``-dimensional BBOB function is hidden inside
    an ``ambient_dim``-dimensional search space through a seeded
    random affine embedding with orthonormal rows (see the module
    docstring for the construction and its properties). The model is
    an ``ambient_dim``-vector optimised over ``[0, 1]^ambient_dim``.

    Parameters
    ----------
    fn_name : str
        Name of the BBOB benchmark function (BBOB suite only).
    seed : int
        Random seed; generates the function instance, the embedding
        and the anchor point.
    intrinsic_dim : int
        Dimensionality of the underlying BBOB function.
    ambient_dim : int
        Dimensionality of the search space; must be at least
        ``intrinsic_dim``.
    bulk_scale : float, optional
        Hessian eigenvalue given to the ``ambient_dim -
        intrinsic_dim`` null-space directions, by default 0.0 (an
        exactly flat null space and a manifold of minimisers).
    noise : float, optional
        Multiplicative noise level (standard deviation), by default
        0.0.

    Returns
    -------
    Task
        Configured optimization task whose ``global_min`` is the BBOB
        instance's ``f_opt``, attained exactly at a seeded uniform
        anchor inside the box. ``tag`` carries the BBOB function
        characteristics (``separable`` forced to ``False`` by the
        dense embedding) plus ``embedded``,
        ``intrinsic_dimensionality`` and ``bulk_scale``.

    Raises
    ------
    ValueError
        If ``fn_name`` is not a BBOB-suite function, if
        ``ambient_dim < intrinsic_dim``, or if ``bulk_scale < 0``.

    Notes
    -----
    Input domain is ``[0, 1]^ambient_dim``; the embedding itself maps
    it to the function's native range (no ``scale_input`` step). If
    ``noise > 0``, the function requires a random key and applies
    multiplicative Gaussian noise to the total loss.
    """
    if fn_name not in bbob_jax.registry:
        raise ValueError(
            f"{fn_name!r} is not a BBOB-suite function; "
            "create_embedded_bbob_task only embeds the BBOB suite."
        )
    if ambient_dim < intrinsic_dim:
        raise ValueError(
            "ambient_dim must be at least intrinsic_dim (got "
            f"ambient_dim={ambient_dim} < intrinsic_dim={intrinsic_dim})."
        )
    if bulk_scale < 0.0:
        raise ValueError(f"bulk_scale must be >= 0 (got {bulk_scale}).")

    key_fn, key_basis, key_anchor = jr.split(jr.key(seed), 3)
    problem = bbob_jax.problem(fn_name, ndim=intrinsic_dim, key=key_fn)

    # Reduced QR of a (D, d) Gaussian gives d orthonormal columns;
    # transposed, an embedding with orthonormal rows.
    q, _ = jnp.linalg.qr(jr.normal(key_basis, (ambient_dim, intrinsic_dim)))
    basis = q.T
    lower, upper = problem.bounds
    scale = float(upper - lower)
    x_star = jr.uniform(key_anchor, (ambient_dim,))

    base_fn = functools.partial(
        compute_embedded_loss,
        fn=problem.fn,
        basis=basis,
        scale=scale,
        x_star=x_star,
        z_opt=problem.x_opt,
        bulk_scale=float(bulk_scale),
    )

    if noise > 0.0:
        loss_fn = add_noise(noise_level=noise)(base_fn)
        pass_rng = True
    else:
        loss_fn = base_fn
        pass_rng = False

    tag = dict(problem.tags)
    tag["separable"] = False
    tag["embedded"] = True
    tag["dimensionality"] = ambient_dim
    tag["intrinsic_dimensionality"] = intrinsic_dim
    tag["bulk_scale"] = float(bulk_scale)
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=jnp.zeros(ambient_dim),
        global_min=float(problem.f_opt),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )
