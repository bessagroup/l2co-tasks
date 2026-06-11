"""Shared utilities for physics-informed neural network (PINN) tasks.

Collects the collocation-point samplers and small evaluation helpers
reused by the PINN task families (Helmholtz, viscous/inviscid Burgers,
Euler, Stokes, PK-PD). The per-equation residual derivations live in
each family module; this module only provides the generic pieces:

- :func:`latin_hypercube` -- Latin-hypercube sampling over a box.
- :func:`sample_triangle` -- uniform sampling inside a triangle.
- :func:`mse` -- mean-squared error (re-exported from :mod:`pde`).
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from collections.abc import Sequence

# Third-party
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

# Local
from .pde import mse

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================

__all__ = ["latin_hypercube", "mse", "sample_triangle"]


def latin_hypercube(
    n: int,
    bounds: Sequence[tuple[float, float]],
    *,
    key: PRNGKeyArray,
) -> Array:
    """Draw ``n`` Latin-hypercube samples over an axis-aligned box.

    Each dimension is partitioned into ``n`` equal strata and a single
    jittered sample is drawn from each stratum, then the per-dimension
    orderings are independently permuted. This gives better space-filling
    coverage than plain uniform sampling (the strategy used for the
    Burgers and Euler collocation grids in the paper).

    Parameters
    ----------
    n : int
        Number of points to draw.
    bounds : Sequence[tuple[float, float]]
        ``(lo, hi)`` range for each dimension; its length sets ``d``.
    key : PRNGKeyArray
        Random key.

    Returns
    -------
    Array
        Sampled points of shape ``(n, d)``.
    """
    columns = []
    for i, (lo, hi) in enumerate(bounds):
        perm_key, jitter_key = jr.split(jr.fold_in(key, i))
        perm = jr.permutation(perm_key, n)
        jitter = jr.uniform(jitter_key, (n,))
        unit = (perm + jitter) / n
        columns.append(lo + (hi - lo) * unit)
    return jnp.stack(columns, axis=-1)


def sample_triangle(
    n: int,
    vertices: Array,
    *,
    key: PRNGKeyArray,
) -> Array:
    """Sample ``n`` points uniformly inside a 2-D triangle.

    Uses the standard barycentric reflection map: for ``r1, r2 ~ U(0, 1)``
    the weights ``(1 - sqrt(r1), sqrt(r1) (1 - r2), sqrt(r1) r2)`` are
    uniform over the simplex, so the convex combination of the vertices
    is uniform over the triangle.

    Parameters
    ----------
    n : int
        Number of points to draw.
    vertices : Array
        The three triangle vertices, row per vertex.
    key : PRNGKeyArray
        Random key.

    Returns
    -------
    Array
        Sampled points of shape ``(n, 2)``.
    """
    v = jnp.asarray(vertices)
    k1, k2 = jr.split(key)
    r1 = jr.uniform(k1, (n,))
    r2 = jr.uniform(k2, (n,))
    s = jnp.sqrt(r1)
    w0 = 1.0 - s
    w1 = s * (1.0 - r2)
    w2 = s * r2
    return w0[:, None] * v[0] + w1[:, None] * v[1] + w2[:, None] * v[2]
