"""UAV terrain path-planning optimisation task.

A JAX port of MetaBox's UAV benchmark: choose ``n`` 3-D waypoints
(in spherical increments) for a drone to fly from a start to a goal over
a synthetic terrain, minimising a weighted sum of path length, threat
proximity, altitude deviation, path smoothness and terrain clearance.

Ported to JAX from MetaBox (BSD 3-Clause, (c) 2023 MetaEvolution Lab),
``MetaEvo/MetaBox@5565a28``, path
``src/environment/problem/SOO/UAV/{uav_numpy.py,uav_dataset.py,utils.py}``.
Original benchmark: "Benchmarking global optimization techniques for
unmanned aerial vehicle path planning", arXiv:2501.14503 (2025); data at
Zenodo 12793991. See ``THIRD_PARTY_NOTICES.md``.

Design
------
The terrain (a 900x900 height map, the threat cylinders, and the
start/goal points) is synthesised **once in NumPy** at task-construction
time via the reference's procedural generator (a scattered-data
Delaunay-linear-interpolation pipeline that does not port cleanly to
JAX), then baked as static constants into a differentiable, JIT/vmap-able
JAX objective ``fn(x)`` over the normalised domain ``[0, 1]^d``. The
objective is differentiable w.r.t. the decision vector; the terrain is a
fixed constant.

The decision vector is ``d = 3 * n_waypoints`` (per waypoint: radial
distance, inclination, azimuth). The paper's suite is 30-D
(``n_waypoints=10``); MetaBox's *generated* instances default to
``dv=5`` (15-D). ``n_waypoints`` is a parameter here (default 10).

``global_min`` is **empirical** (``estimate_global_min``): the objective
mixes smooth terms with non-smooth ones (nearest-neighbour terrain
lookups, thresholded threat/clearance penalties), so 0 is not a known
analytical floor. It is recorded as an estimate, not a guaranteed bound.

Public API
----------
create_uav_task
    Build a UAV :class:`Task` from a terrain id / seed and
    ``n_waypoints``.
"""

#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from functools import partial

# Third-party
import jax
import jax.numpy as jnp
import numpy as np

# Local
from .global_min import estimate_global_min as _estimate_global_min
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling", "MetaEvolution Lab"]
__status__ = "Stable"
# =============================================================================

_MAP_SIZE = 900
_H0 = 130.0
_DRONE_SIZE = 1.0
_DANGER_DIST = 10.0
_J_PEN = 1e4
_ZMIN, _ZMAX = 100.0, 200.0
_WEIGHTS = (5.0, 1.0, 10.0, 1.0, 1.0)  # b1..b5 for J1..J5
_N_SAMPLES = 12  # terrain-clearance samples per segment (J5)
_ANGLE_MAX = 45.0  # turning / climb angle threshold (degrees), J4


#                                        Procedural terrain synthesis (NumPy)
# =============================================================================
# Faithful in structure to MetaBox's ``utils.generate_terrain`` /
# ``createmodel``; keyed by a NumPy ``RandomState`` (exact bitwise
# reproduction of the reference stream is not required).


def _interp_scattered(
    px: np.ndarray,
    py: np.ndarray,
    pv: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
) -> np.ndarray:
    """Scattered linear interpolation with 4 zero-valued corner anchors."""
    from scipy.interpolate import LinearNDInterpolator

    ax = np.concatenate([px, [100.0, 100.0, -100.0, -100.0]])
    ay = np.concatenate([py, [100.0, -100.0, 100.0, -100.0]])
    av = np.concatenate([pv, [0.0, 0.0, 0.0, 0.0]])
    f = LinearNDInterpolator(np.stack([ax, ay], axis=1), av)
    return f(qx, qy)


def _generate_terrain(
    rng: np.random.RandomState, r0: float, rr: float, n_iter: int = 8
) -> np.ndarray:
    """Synthesise a ``_MAP_SIZE`` x ``_MAP_SIZE`` height map.

    Grows a scattered set of height seed points over ``n_iter`` octaves
    and interpolates them onto the map grid (see MetaBox
    ``utils.generate_terrain``).
    """
    n0 = int(rng.randint(1, 5))
    x = rng.randn(n0)
    y = rng.randn(n0)
    h = r0 * rng.randn(n0) + _H0
    r = rr * rng.randn(n0) + r0

    for k in range(1, n_iter + 1):
        dxy = 0.75**k
        dh = 0.5**k
        n_new = 3 * n0
        parents = np.tile(np.arange(n0), (3, 1)).flatten("F")
        theta = 2.0 * np.pi * rng.rand(n_new)
        radius = dxy * (rng.rand(n_new) + 1.0)
        xn = x[parents] + radius * np.cos(theta)
        yn = y[parents] + radius * np.sin(theta)
        rn = _interp_scattered(x, y, r, xn, yn) + (dh * rr) * rng.randn(n_new)
        hn = _interp_scattered(x, y, h, xn, yn) + (
            dh / dxy
        ) * radius * rn * rng.randn(n_new)
        x = np.concatenate([x, xn])
        y = np.concatenate([y, yn])
        r = np.concatenate([r, rn])
        h = np.concatenate([h, hn])
        n0 += n_new

    x = (x - np.median(x)) / np.std(x)
    y = (y - np.median(y)) / np.std(y)
    grid = np.linspace(-1.0, 1.0, _MAP_SIZE)
    xm, ym = np.meshgrid(grid, grid)
    height = _interp_scattered(x, y, h, xm.ravel(), ym.ravel())
    return np.nan_to_num(height).reshape(_MAP_SIZE, _MAP_SIZE)


def _create_terrain(
    seed: int, terrain_id: int, n_threats: int, n_iter: int
) -> dict:
    """Build one procedural terrain instance (height map, threats, ends)."""
    rng = np.random.RandomState(seed + terrain_id)
    r0 = rng.rand() * 600.0 - 200.0
    rr = rng.rand() * 800.0 + 100.0
    height = _generate_terrain(rng, r0=r0, rr=rr, n_iter=n_iter)

    threats = rng.rand(n_threats, 4)
    radius = np.round(_MAP_SIZE / 15.0 + threats[:, 3] * _MAP_SIZE / 30.0)
    tx = np.round(np.minimum(radius, 0.1 * _MAP_SIZE) + threats[:, 0] * 0.7
                  * _MAP_SIZE)
    ty = np.round(np.minimum(radius, 0.1 * _MAP_SIZE) + threats[:, 1] * 0.9
                  * _MAP_SIZE)
    threats[:, 3] = radius
    threats[:, 0] = tx
    threats[:, 1] = ty

    for _ in range(20):
        xs = round((0.02 + rng.rand() * 0.17) * _MAP_SIZE)
        xe = round((1.0 - 0.02 - rng.rand() * 0.17) * _MAP_SIZE)
        ys = round((0.02 + rng.rand() * 0.17) * _MAP_SIZE)
        ye = round((1.0 - 0.02 - rng.rand() * 0.17) * _MAP_SIZE)
        dist = np.sqrt((tx - xs) ** 2 + (ty - ys) ** 2)
        if np.all(dist > radius + _DRONE_SIZE + _DANGER_DIST):
            break
    start = np.array([float(xs), float(ys), 150.0])
    end = np.array([float(xe), float(ye), 150.0])
    return {"H": height, "threats": threats, "start": start, "end": end}


def _boundaries(
    start: np.ndarray, end: np.ndarray, n_waypoints: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-variable lower/upper bounds (r, psi, phi) tiled over waypoints."""
    distance = float(np.linalg.norm(start - end))
    phi0 = float(np.arctan2(end[1] - start[1], end[0] - start[0]))
    lo_triple = np.array([0.0, -np.pi / 4.0, phi0 - np.pi / 4.0])
    hi_triple = np.array(
        [2.0 * distance / n_waypoints, np.pi / 4.0, phi0 + np.pi / 4.0]
    )
    lb = np.tile(lo_triple, n_waypoints)
    ub = np.tile(hi_triple, n_waypoints)
    return lb, ub


#                                             Differentiable JAX objective
# =============================================================================


def _dist_point_to_segment(px, py, ax, ay, bx, by):
    """2-D distance from point ``p`` to segment ``a->b`` (vectorised)."""
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    bpx, bpy = px - bx, py - by
    d_ab = jnp.hypot(abx, aby)
    d_ap = jnp.hypot(apx, apy)
    d_bp = jnp.hypot(bpx, bpy)
    dot1 = abx * apx + aby * apy
    dot2 = (-abx) * bpx + (-aby) * bpy
    perp = jnp.abs(abx * apy - aby * apx) / jnp.where(d_ab == 0, 1.0, d_ab)
    nearest = jnp.where(d_ab == 0, d_ap, jnp.minimum(d_ap, d_bp))
    return jnp.where((d_ab != 0) & (dot1 >= 0) & (dot2 >= 0), perp, nearest)


def _terrain_nearest(H, xc, yc):
    """Nearest-neighbour terrain height ``H[round(y)-1, round(x)-1]``."""
    xi = jnp.clip(jnp.round(xc).astype(jnp.int32) - 1, 0, H.shape[1] - 1)
    yi = jnp.clip(jnp.round(yc).astype(jnp.int32) - 1, 0, H.shape[0] - 1)
    return H[yi, xi]


def uav_cost(
    model,
    *,
    H,
    threats,
    start,
    end,
    lb,
    ub,
    n_waypoints,
    box,
    drone_size=_DRONE_SIZE,
    danger_dist=_DANGER_DIST,
    j_pen=_J_PEN,
):
    """UAV path-planning cost over the normalised ``[0, 1]^d`` domain.

    Parameters
    ----------
    model : jnp.ndarray
        Decision vector in ``[0, 1]^d`` (``d = 3 * n_waypoints``); scaled
        internally to the ``(r, psi, phi)`` bounds ``[lb, ub]``.
    H, threats, start, end, lb, ub : jnp.ndarray
        Baked terrain constants (height map, threat cylinders
        ``[x, y, height, radius]``, start/goal 3-vectors, bound arrays).
    n_waypoints : int
        Number of waypoints ``n``.
    box : tuple[float, float, float, float, float, float]
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` clamping box.
    drone_size, danger_dist, j_pen : float
        Threat geometry and penalty constants.

    Returns
    -------
    jnp.ndarray
        Scalar weighted-sum cost ``5*J1 + J2 + 10*J3 + J4 + J5``.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = box
    scaled = lb + model * (ub - lb)
    r = scaled[0::3]
    psi = scaled[1::3]
    phi = scaled[2::3]

    # Sequential spherical->Cartesian with per-step clamping.
    def step(carry, inp):
        xp, yp, zp = carry
        ri, psii, phii = inp
        xn = jnp.clip(xp + ri * jnp.cos(psii) * jnp.sin(phii), xmin, xmax)
        yn = jnp.clip(yp + ri * jnp.cos(psii) * jnp.cos(phii), ymin, ymax)
        zn = jnp.clip(zp + ri * jnp.sin(psii), zmin, zmax)
        return (xn, yn, zn), (xn, yn, zn)

    _, (xw, yw, zw) = jax.lax.scan(
        step, (start[0], start[1], start[2]), (r, psi, phi)
    )

    # Full path: start + waypoints + goal (N = n + 2 points).
    x_all = jnp.concatenate([start[0:1], xw, end[0:1]])
    y_all = jnp.concatenate([start[1:2], yw, end[1:2]])
    z_all = jnp.concatenate([start[2:3], zw, end[2:3]])
    z_abs = z_all + _terrain_nearest(H, x_all, y_all)

    # J1 - path length (absolute altitude).
    dx, dy, dz = jnp.diff(x_all), jnp.diff(y_all), jnp.diff(z_abs)
    j1 = jnp.sum(jnp.sqrt(dx**2 + dy**2 + dz**2))

    # J2 - threat proximity over every segment x every threat.
    ax, ay = x_all[:-1], y_all[:-1]
    bx, by = x_all[1:], y_all[1:]
    tx = threats[:, 0][:, None]
    ty = threats[:, 1][:, None]
    trad = threats[:, 3][:, None]
    dseg = _dist_point_to_segment(
        tx, ty, ax[None, :], ay[None, :], bx[None, :], by[None, :]
    )
    tc = trad + drone_size + danger_dist - dseg
    tc = jnp.where(dseg > trad + drone_size + danger_dist, 0.0, tc)
    tc = jnp.where(dseg < trad + drone_size, j_pen, tc)
    j2 = jnp.sum(tc)

    # J3 - altitude cost on the relative waypoint heights.
    mid = (zmin + zmax) / 2.0
    j3 = jnp.sum(jnp.where(zw < 0, j_pen, jnp.abs(zw - mid)))

    # J4 - turning / climb smoothness with zero-segment backfill.
    j4 = _smoothness_cost(x_all, y_all, z_abs)

    # J5 - terrain clearance (bilinear samples along each segment).
    j5 = _clearance_cost(H, x_all, y_all, z_abs, j_pen)

    b1, b2, b3, b4, b5 = _WEIGHTS
    return b1 * j1 + b2 * j2 + b3 * j3 + b4 * j4 + b5 * j5


def _smoothness_cost(x_all, y_all, z_abs):
    """J4: sum of turning- and climb-angle excess over 45 degrees."""
    seg_x = jnp.diff(x_all)
    seg_y = jnp.diff(y_all)  # (N-1,) xy projections (z zeroed)
    norms = jnp.hypot(seg_x, seg_y)
    nz = norms > 0
    n_seg = seg_x.shape[0]
    pos = jnp.arange(n_seg)

    # Backfill: seg1 = last non-zero projection at index <= j;
    # seg2 = first non-zero projection at index >= j.
    last_nz = jax.lax.cummax(jnp.where(nz, pos, -1))
    a = jnp.where(nz, pos, n_seg)
    first_nz = jnp.flip(jax.lax.cummin(jnp.flip(a)))

    def gather(idx, valid):
        safe = jnp.clip(idx, 0, n_seg - 1)
        gx = jnp.where(valid, seg_x[safe], 0.0)
        gy = jnp.where(valid, seg_y[safe], 0.0)
        return gx, gy

    total = 0.0
    for i in range(n_seg - 1):  # interior vertices i = 0 .. N-3
        i1 = last_nz[i]
        s1x, s1y = gather(i1, i1 >= 0)
        i2 = first_nz[i + 1]
        s2x, s2y = gather(i2, i2 < n_seg)

        climb1 = jnp.degrees(jnp.arctan2(z_abs[i + 1] - z_abs[i],
                                         jnp.hypot(s1x, s1y)))
        climb2 = jnp.degrees(jnp.arctan2(z_abs[i + 2] - z_abs[i + 1],
                                         jnp.hypot(s2x, s2y)))
        cross = jnp.abs(s1x * s2y - s1y * s2x)
        dot = s1x * s2x + s1y * s2y
        turning = jnp.degrees(jnp.arctan2(cross, dot))

        total = total + jnp.where(
            jnp.abs(turning) > _ANGLE_MAX, jnp.abs(turning), 0.0
        ) + jnp.where(
            jnp.abs(climb2 - climb1) > _ANGLE_MAX,
            jnp.abs(climb2 - climb1), 0.0,
        )
    return total


def _clearance_cost(H, x_all, y_all, z_abs, j_pen):
    """J5: J_pen if any bilinear terrain sample rises above the path."""
    t = jnp.linspace(0.0, 1.0, _N_SAMPLES)
    xs = x_all[:-1, None] + (x_all[1:, None] - x_all[:-1, None]) * t[None, :]
    ys = y_all[:-1, None] + (y_all[1:, None] - y_all[:-1, None]) * t[None, :]
    zs = z_abs[:-1, None] + (z_abs[1:, None] - z_abs[:-1, None]) * t[None, :]
    # RegularGridInterpolator on grid 0..size-1 at raw (0-based) coords;
    # out-of-grid -> NaN so ``z > NaN`` is False (path not clear).
    heights = jax.scipy.ndimage.map_coordinates(
        H, [ys.ravel(), xs.ravel()], order=1, mode="constant", cval=jnp.nan
    ).reshape(xs.shape)
    clear = jnp.all(zs > heights)
    return jnp.where(clear, 0.0, j_pen)


#                                                            Task factory
# =============================================================================


def create_uav_task(
    terrain_id: int = 0,
    seed: int = 3849,
    n_waypoints: int = 10,
    *,
    n_threats: int | None = None,
    n_iter: int = 8,
    estimate_global_min: bool = True,
) -> Task:
    """Create a UAV terrain path-planning task.

    Parameters
    ----------
    terrain_id : int, optional
        Terrain instance index (offsets the RNG), by default 0.
    seed : int, optional
        Base seed for terrain synthesis (MetaBox uses 3849), by default
        3849.
    n_waypoints : int, optional
        Number of waypoints ``n``; the decision vector is ``3 * n``.
        Default 10 (30-D, the paper's dimensionality). MetaBox's
        generated instances use ``n = 5`` (15-D).
    n_threats : int or None, optional
        Number of threat cylinders. Defaults to MetaBox's schedule
        (15 for the first half of the suite, 30 otherwise) keyed on
        ``terrain_id``.
    n_iter : int, optional
        Octaves of the terrain generator (default 8, the reference
        value). Lower values build faster with coarser terrain.
    estimate_global_min : bool, optional
        Estimate the empirical ``global_min`` with benchmark optimisers
        (``True``, default) or leave it ``None`` (skip the slow search).

    Returns
    -------
    Task
        Configured UAV task over ``[0, 1]^d`` (``d = 3 * n_waypoints``),
        deterministic (``pass_rng=False``). ``global_min`` is an
        empirical estimate, not a guaranteed lower bound.
    """
    terrain_id = int(terrain_id)
    n_waypoints = int(n_waypoints)
    if n_threats is None:
        n_threats = 15 if terrain_id < 28 else 30

    terrain = _create_terrain(seed, terrain_id, n_threats, n_iter)
    lb, ub = _boundaries(terrain["start"], terrain["end"], n_waypoints)
    box = (1.0, float(_MAP_SIZE), 1.0, float(_MAP_SIZE), _ZMIN, _ZMAX)

    loss_fn = partial(
        uav_cost,
        H=jnp.asarray(terrain["H"], dtype=jnp.float32),
        threats=jnp.asarray(terrain["threats"], dtype=jnp.float32),
        start=jnp.asarray(terrain["start"], dtype=jnp.float32),
        end=jnp.asarray(terrain["end"], dtype=jnp.float32),
        lb=jnp.asarray(lb, dtype=jnp.float32),
        ub=jnp.asarray(ub, dtype=jnp.float32),
        n_waypoints=n_waypoints,
        box=box,
    )

    dimensionality = 3 * n_waypoints
    model = jnp.full(dimensionality, 0.5)  # centre of the [0, 1]^d box
    tag = {
        "task_name": "uav",
        "fn_name": f"uav_terrain_{terrain_id}",
        "dimensionality": dimensionality,
        "n_waypoints": n_waypoints,
        "n_threats": n_threats,
        "terrain_id": terrain_id,
        "seed": seed,
        "global_min_kind": "empirical",
    }

    task = Task(
        model=model,
        global_min=None,
        pass_rng=False,
        loss_fn=loss_fn,
        tag=tag,
    )
    if estimate_global_min:
        gmin = _estimate_global_min(task, clip_to_unit=True, seed=seed)
        task = Task(
            model=model,
            global_min=float(gmin),
            pass_rng=False,
            loss_fn=loss_fn,
            tag=tag,
        )
    return task
