"""Black-box benchmark optimisation tasks (BBOB, BBOB-noisy, CEC2005,
CEC2017 and CEC 2013 LSGO).

Each task minimises a scalar benchmark objective ``f(x)`` drawn from
the ``bbob_jax`` registry: the BBOB suite (sphere, Rosenbrock,
Rastrigin, ...), the BBOB-noisy suite (f101-f130), the CEC2005 suite,
the CEC2017 suite or the CEC 2013 LSGO suite. The model is a
``d``-vector initialised at the origin and optimised over the
normalised domain ``[0, 1]^d``; the loss rescales it to the function's
native bounds via ``scale_input`` (``[-5, 5]^d`` for BBOB and
BBOB-noisy, per-function bounds for CEC2005 and CEC 2013 LSGO,
``[-100, 100]^d`` for CEC2017) before evaluating ``f``.

When ``noise > 0`` the objective is corrupted with multiplicative
Gaussian noise, ``f(x) * (1 + e)`` with ``e ~ N(0, noise)``, which
makes the loss stochastic (``pass_rng=True``). The CEC2005 functions
``f4``, ``f17``, ``f24`` and ``f25`` are inherently stochastic and so
always set ``pass_rng=True``, even with ``noise == 0``; the CEC2017
suite has no stochastic functions. The BBOB-noisy suite is inherently
stochastic throughout (its factory has no extra ``noise`` knob and
requires a bbob-jax newer than 1.8.0).

The CEC 2013 LSGO suite is a *fixed-instance* suite: its parameters
are official constants rather than seed-sampled, so ``seed`` selects no
instance (it only seeds the optional noise), and each function is
defined at exactly one dimensionality -- 1000, or 905 for the
overlapping ``f13`` / ``f14``. ``create_cec2013lsgo_task`` propagates
``bbob_jax``'s ``ValueError`` off that native dimensionality. Like the
BBOB-noisy factory it requires a bbob-jax newer than 2.0.0.

Some CEC2017 functions are only defined from a minimum dimensionality
upward (the hybrids need one dimension per subcomponent kernel);
``create_cec2017_task`` propagates ``bbob_jax``'s ``ValueError`` when
``dimensionality`` is below it and records the bound in
``tag["min_ndim"]``.

Public API
----------
create_bbob_task
    Build a BBOB :class:`Task` from ``fn_name``, ``seed``,
    ``dimensionality`` and optional ``noise``.
create_bbob_noisy_task
    Same, for the inherently stochastic BBOB-noisy suite (names
    ``bbob_noisy_f101`` ... ``bbob_noisy_f130``; no ``noise`` knob).
create_cec2005_task
    Same, for a CEC2005 function.
create_cec2017_task
    Same, for a CEC2017 function (names ``cec2017_f1``,
    ``cec2017_f3`` ... ``cec2017_f30``; F2 was officially withdrawn).
create_cec2013lsgo_task
    Same, for a CEC 2013 LSGO function (names ``cec2013lsgo_f1``
    ... ``cec2013lsgo_f15``; fixed instance, native dimensionality
    only).
CEC2019Sampler
    ``f3dasm.Block`` that enumerates the CEC-2019 suite, mapping each
    function name to its fixed dimensionality.
"""

from __future__ import annotations

import functools

# Standard
import bbob_jax
import jax

# Third-party
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
from f3dasm import Block, ExperimentData

# Local
from .task import Task

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================


# =============================================================================


class CEC2019Sampler(Block):
    """
    Sampler for CEC 2019 benchmark functions.

    Methods
    -------
    call(data)
        Generate experiment data for CEC 2019 benchmark functions.
    """

    def __init__(self):
        self.dim_mapping = {
            "F12019": 9,
            "F22019": 16,
            "F32019": 18,
            "F42019": 10,
            "F52019": 10,
            "F62019": 10,
            "F72019": 10,
            "F82019": 10,
            "F92019": 10,
            "F102019": 10,
        }

    def call(self, data: ExperimentData) -> ExperimentData:
        """
        Generate experiment data for CEC 2019 benchmark functions.

        Parameters
        ----------
        data : ExperimentData
            Input experiment data.

        Returns
        -------
        ExperimentData
            Experiment data with CEC 2019 benchmark functions.
        """
        fns = data.domain.input_space["fn_name"].categories

        df = pd.DataFrame(
            {
                "dimensionality": [self.dim_mapping[f] for f in fns],
                "fn_name": fns,
            }
        )

        return ExperimentData(domain=data.domain, input_data=df)


def scale_input(domain_bounds: jax.Array, function_bounds: jax.Array):
    """
    Create a decorator to scale input parameters between different bounds.

    Parameters
    ----------
    domain_bounds : jax.Array
        Domain bounds with shape (n_dims, 2), where each row contains
        [lower, upper] for normalization.
    function_bounds : jax.Array
        Target function bounds with shape (n_dims, 2), where each row
        contains [lower, upper] for scaling.

    Returns
    -------
    Callable
        Decorator that scales function inputs from domain_bounds to
        function_bounds.

    Notes
    -----
    Performs two-step transformation: first normalizes input to [0, 1]
    using domain_bounds, then scales to function_bounds.
    """

    def decorator(func):
        """Wrap a function to scale its first argument."""

        def wrapper(x, *args, **kwargs):
            """Apply domain-to-function bound scaling."""
            # Extract bounds
            dmn_lower = domain_bounds[:, 0]
            dmn_upper = domain_bounds[:, 1]
            fn_lower = function_bounds[:, 0]
            fn_upper = function_bounds[:, 1]

            # Scale x to [0, 1] based on standard bounds
            normalized_x = (x - dmn_lower) / (dmn_upper - dmn_lower)

            # Scale normalized_x to target bounds
            scaled_x = fn_lower + normalized_x * (fn_upper - fn_lower)

            # Pass transformed input to the original function
            return func(scaled_x, *args, **kwargs)

        return wrapper

    return decorator


def add_noise(noise_level: float):
    """
    Create a decorator to add multiplicative Gaussian noise to function
    output.

    Parameters
    ----------
    noise_level : float
        Standard deviation of the Gaussian noise relative to the
        function value.

    Returns
    -------
    Callable
        Decorator that adds noise to function output using provided
        random key.

    Notes
    -----
    The noise is multiplicative: $y_{noisy} = y + y \\cdot \\epsilon$,
    where $\\epsilon \\sim \\mathcal{N}(0, \\text{noise_level})$.
    Requires a 'key' argument in the wrapped function signature.
    """

    def decorator(func):
        """Wrap a function to inject multiplicative noise."""

        @functools.wraps(func)
        def wrapper(x, key, *args, **kwargs):
            """Evaluate function with added noise."""
            noise = jr.normal(key, shape=()) * noise_level
            y = func(x, *args, **kwargs)
            return y + y * noise

        return wrapper

    return decorator


def create_bbob_task(
    fn_name: str, seed: int, dimensionality: int, noise: float = 0.0
) -> Task:
    """
    Create an optimization task from a BBOB benchmark function.

    Parameters
    ----------
    fn_name : str
        Name of the BBOB benchmark function.
    seed : int
        Random seed for function instance generation.
    dimensionality : int
        Dimensionality of the optimization problem.
    noise : float, optional
        Multiplicative noise level (standard deviation), by default 0.0.

    Returns
    -------
    Task
        Configured optimization task with scaled and optionally noisy
        objective function.

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to BBOB's standard
    [-5, 5]^d range. If noise > 0, the function requires a random
    key and applies multiplicative Gaussian noise.
    """
    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 1)),
        function_bounds=jnp.tile(jnp.array([-5.0, 5.0]), (dimensionality, 1)),
    )

    if noise > 0.0:
        fn_noise = add_noise(noise_level=noise)
        pass_rng = True
    else:
        fn_noise = lambda f: f
        pass_rng = False

    bbob_fn, global_min = bbob_jax.registry[fn_name](
        ndim=dimensionality, key=jr.key(seed)
    )

    loss_fn = fn_noise(fn_scale_input(bbob_fn))
    model = jnp.zeros(dimensionality)
    tag = bbob_jax.function_characteristics[fn_name].copy()
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=model,
        global_min=float(global_min),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )


def create_bbob_noisy_task(
    fn_name: str, seed: int, dimensionality: int
) -> Task:
    """
    Create an optimization task from a BBOB-noisy benchmark function.

    The BBOB-noisy suite (``bbob_noisy_f101`` ... ``bbob_noisy_f130``)
    is inherently stochastic: eight base landscapes disturbed by the
    Gaussian, uniform or Cauchy noise model at moderate (f101-f106) or
    severe (f107-f130) severity. The noise disturbs only the residual
    above the optimum, so tasks always set ``pass_rng=True`` and there
    is no extra ``noise`` knob (unlike the deterministic suites).

    Parameters
    ----------
    fn_name : str
        Name of the BBOB-noisy benchmark function
        (``"bbob_noisy_f101"`` ... ``"bbob_noisy_f130"``).
    seed : int
        Random seed for function instance generation.
    dimensionality : int
        Dimensionality of the optimization problem.

    Returns
    -------
    Task
        Configured optimization task with scaled, inherently noisy
        objective function (``pass_rng=True``). ``tag`` carries the
        ``bbob_jax`` characteristics: ``separable`` / ``unimodal``
        describe the undisturbed base function, and the noise model is
        recorded in the ``gaussian_noise`` / ``uniform_noise`` /
        ``cauchy_noise`` / ``severe`` / ``noise`` flags.

    Raises
    ------
    ImportError
        If the installed ``bbob-jax`` predates the BBOB-noisy suite
        (releases up to 1.8.0 do not include it).

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to the suite's standard
    [-5, 5]^d range. ``global_min`` is the instance's ``f_opt``: the
    infimum of the *undisturbed* value. The Cauchy noise model is
    signed and heavy-tailed, so observed noisy values can dip below
    it.
    """
    registry = getattr(bbob_jax, "bbob_noisy_registry", None)
    if registry is None:
        raise ImportError(
            "create_bbob_noisy_task requires a bbob-jax with the "
            "BBOB-noisy suite (bbob_noisy_registry); releases up to "
            "1.8.0 do not include it. Upgrade bbob-jax."
        )

    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 1)),
        function_bounds=jnp.tile(jnp.array([-5.0, 5.0]), (dimensionality, 1)),
    )

    noisy_fn, global_min = registry[fn_name](
        ndim=dimensionality, key=jr.key(seed)
    )

    loss_fn = fn_scale_input(noisy_fn)
    model = jnp.zeros(dimensionality)
    tag = bbob_jax.bbob_noisy_function_characteristics[fn_name].copy()
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed

    return Task(
        model=model,
        global_min=float(global_min),
        pass_rng=True,
        loss_fn=loss_fn,
        tag=tag,
    )


def create_cec2005_task(
    fn_name: str, seed: int, dimensionality: int, noise: float = 0.0
) -> Task:
    """
    Create an optimization task from a CEC2005 benchmark function.

    Parameters
    ----------
    fn_name : str
        Name of the CEC2005 benchmark function.
    seed : int
        Random seed for function instance generation.
    dimensionality : int
        Dimensionality of the optimization problem.
    noise : float, optional
        Multiplicative noise level (standard deviation), by default 0.0.

    Returns
    -------
    Task
        Configured optimization task with scaled and optionally noisy
        objective function.

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to CEC2005's bounds.
    If noise > 0, the function requires a random
    key and applies multiplicative Gaussian noise.
    """

    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 1)),
        function_bounds=jnp.tile(
            jnp.array(bbob_jax.bounds.cec2005_bounds[fn_name]),
            (dimensionality, 1),
        ),
    )

    if noise > 0.0:
        fn_noise = add_noise(noise_level=noise)
        pass_rng = True
    else:
        fn_noise = lambda f: f
        pass_rng = False

    # These functions have built-in stochasticity.
    if fn_name in ["f4", "f17", "f24", "f25"]:
        pass_rng = True

    bbob_fn, global_min = bbob_jax.cec2005_registry[fn_name](
        ndim=dimensionality, key=jr.key(seed)
    )

    loss_fn = fn_noise(fn_scale_input(bbob_fn))
    model = jnp.zeros(dimensionality)
    tag = bbob_jax.cec2005_function_characteristics[fn_name].copy()
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=model,
        global_min=float(global_min),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )


def create_cec2017_task(
    fn_name: str, seed: int, dimensionality: int, noise: float = 0.0
) -> Task:
    """
    Create an optimization task from a CEC2017 benchmark function.

    Parameters
    ----------
    fn_name : str
        Name of the CEC2017 benchmark function (``"cec2017_f1"``,
        ``"cec2017_f3"`` ... ``"cec2017_f30"``; F2 was officially
        withdrawn from the suite).
    seed : int
        Random seed for function instance generation.
    dimensionality : int
        Dimensionality of the optimization problem. Must be at least
        the function's ``min_ndim`` (the hybrids F11-F20 need one
        dimension per subcomponent kernel, up to 7; F29/F30 need 5;
        F6 needs 2).
    noise : float, optional
        Multiplicative noise level (standard deviation), by default 0.0.

    Returns
    -------
    Task
        Configured optimization task with scaled and optionally noisy
        objective function. ``tag`` carries the ``bbob_jax`` function
        characteristics plus ``min_ndim``.

    Raises
    ------
    ValueError
        If ``dimensionality`` is below the function's ``min_ndim``
        (raised by the ``bbob_jax`` maker with the exact bound).

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to CEC2017's standard
    [-100, 100]^d range. If noise > 0, the function requires a random
    key and applies multiplicative Gaussian noise; the suite itself
    has no stochastic functions.
    """
    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 1)),
        function_bounds=jnp.tile(
            jnp.array(bbob_jax.bounds.cec2017_bounds[fn_name]),
            (dimensionality, 1),
        ),
    )

    if noise > 0.0:
        fn_noise = add_noise(noise_level=noise)
        pass_rng = True
    else:
        fn_noise = lambda f: f
        pass_rng = False

    # problem() bundles the same instance the registry would build,
    # plus min_ndim (and raises ValueError below it).
    problem = bbob_jax.problem(fn_name, ndim=dimensionality, key=jr.key(seed))

    loss_fn = fn_noise(fn_scale_input(problem.fn))
    model = jnp.zeros(dimensionality)
    tag = dict(problem.tags)
    tag["min_ndim"] = problem.min_ndim
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=model,
        global_min=float(problem.f_opt),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )


def create_cec2013lsgo_task(
    fn_name: str, seed: int, dimensionality: int, noise: float = 0.0
) -> Task:
    """
    Create an optimization task from a CEC 2013 LSGO benchmark function.

    The CEC 2013 Large-Scale Global Optimization suite
    (``cec2013lsgo_f1`` ... ``cec2013lsgo_f15``) is a **fixed-instance**
    suite: its shift / rotation / permutation / weight parameters are
    official constants rather than seed-sampled, and each function is
    defined only at its native dimensionality (1000, or 905 for the
    overlapping ``f13`` / ``f14``). ``seed`` therefore does not select
    an instance -- it only seeds the optional multiplicative noise --
    and every seed yields the same landscape.

    Parameters
    ----------
    fn_name : str
        Name of the CEC 2013 LSGO benchmark function
        (``"cec2013lsgo_f1"`` ... ``"cec2013lsgo_f15"``).
    seed : int
        Random seed. Kept for signature uniformity with the other
        benchmark factories and recorded in ``tag``; the landscape is
        seed-independent (see above).
    dimensionality : int
        Dimensionality of the optimization problem. Must equal the
        function's native dimensionality: 1000, or 905 for ``f13`` and
        ``f14``.
    noise : float, optional
        Multiplicative noise level (standard deviation), by default 0.0.

    Returns
    -------
    Task
        Configured optimization task with scaled and optionally noisy
        objective function. ``tag`` carries the ``bbob_jax`` function
        characteristics (``separable`` / ``partially_separable`` /
        ``overlapping`` / ``non_separable``, exactly one True, plus
        ``rotated``) and ``min_ndim``, which for this suite *is* the
        native dimensionality.

    Raises
    ------
    ImportError
        If the installed ``bbob-jax`` predates the CEC 2013 LSGO suite
        (releases up to 2.0.0 do not include it).
    KeyError
        If ``fn_name`` is not a CEC 2013 LSGO function name.
    ValueError
        If ``dimensionality`` does not equal the function's native
        dimensionality (raised by the ``bbob_jax`` maker with the exact
        bound).

    Notes
    -----
    Input domain is [0, 1]^d which is scaled to the function's own
    bounds (``[-100, 100]^d`` for most of the suite, ``[-5, 5]^d`` and
    ``[-32, 32]^d`` for the Rastrigin- and Ackley-based members). If
    noise > 0, the function requires a random key and applies
    multiplicative Gaussian noise; the suite itself has no stochastic
    functions.

    ``global_min`` is ``0.0`` for all 15 functions, and is a true lower
    bound in every case -- but two members never attain it: ``f12``'s
    Rosenbrock optimum sits at ``xopt + 1`` and ``f14``'s conflicting
    overlap makes ``0`` unreachable. Success-threshold metrics defined
    relative to ``global_min`` will therefore never fire on those two.
    """
    if not hasattr(bbob_jax, "cec2013lsgo_registry"):
        raise ImportError(
            "create_cec2013lsgo_task requires a bbob-jax with the "
            "CEC 2013 LSGO suite (cec2013lsgo_registry); releases up "
            "to 2.0.0 do not include it. Upgrade bbob-jax."
        )

    # ``problem()`` resolves names across *every* bbob-jax suite, so an
    # off-suite name would otherwise build a valid task from the wrong
    # suite (e.g. "sphere" -> a 1000-D BBOB sphere on [-5, 5]) and tag
    # it as an LSGO one. Check membership against the LSGO registry
    # first so a wrong name fails loudly.
    if fn_name not in bbob_jax.cec2013lsgo_registry:
        raise KeyError(
            f"{fn_name!r} is not a CEC 2013 LSGO function; expected one "
            f"of cec2013lsgo_f1 ... cec2013lsgo_f15"
        )

    # problem() bundles the instance, its per-function bounds and
    # min_ndim (and raises ValueError off the native dimensionality),
    # so it is the single source of truth for the affine map below.
    # ``key`` is accepted but ignored by the fixed-instance maker.
    problem = bbob_jax.problem(fn_name, ndim=dimensionality, key=jr.key(seed))

    fn_scale_input = scale_input(
        domain_bounds=jnp.tile(jnp.array([0.0, 1.0]), (dimensionality, 1)),
        function_bounds=jnp.tile(
            jnp.array(problem.bounds), (dimensionality, 1)
        ),
    )

    if noise > 0.0:
        fn_noise = add_noise(noise_level=noise)
        pass_rng = True
    else:
        fn_noise = lambda f: f
        pass_rng = False

    loss_fn = fn_noise(fn_scale_input(problem.fn))
    model = jnp.zeros(dimensionality)
    tag = dict(problem.tags)
    tag["min_ndim"] = problem.min_ndim
    tag["dimensionality"] = dimensionality
    tag["fn_name"] = fn_name
    tag["seed"] = seed
    tag["noise"] = noise

    return Task(
        model=model,
        global_min=float(problem.f_opt),
        pass_rng=pass_rng,
        loss_fn=loss_fn,
        tag=tag,
    )
