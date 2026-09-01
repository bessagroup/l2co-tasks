# Embedded BBOB tasks

This page explains the `bbob_embedded` task family in plain language: what it is, why it
exists, and what it does to optimizers. The factory is
[`create_embedded_bbob_task`](api.md#l2co_tasks.create_embedded_bbob_task) and the
matching suite is `l2co_tasks/conf/tasks/bbob_embedded.yaml`.

## The problem it solves

Real machine-learning loss landscapes have a peculiar shape. You might have a million
parameters, but the loss only actually *changes* along a few dozen directions. The other
~999,950 directions are flat — nothing happens if you move along them.

If you want to train an optimizer-selection policy on that kind of terrain, you would
normally need real neural networks, which are slow and awkward to sample in bulk. The
embedded BBOB family fakes that shape using cheap analytic BBOB functions instead.

## How it fakes it

Take a normal 10-dimensional BBOB function like `rastrigin`. Now hand the optimizer a
**256-dimensional** search box instead. Behind the scenes, the 256-dimensional guess is
squashed down to 10 numbers through a fixed random matrix, and *those* are fed to
`rastrigin`.

So the optimizer gropes around in 256 dimensions, but only 10 dimensions' worth of
information ever reaches the function. The remaining 246 directions are invisible: move
along them and the loss does not budge. That is the neural-network signature, produced by
a single matrix multiply.

```python
from l2co_tasks import create_embedded_bbob_task

task = create_embedded_bbob_task(
    fn_name="rastrigin",
    seed=0,
    intrinsic_dim=10,    # dimensions that actually matter
    ambient_dim=256,     # dimensions the optimizer thinks it has
)

loss = task.loss_fn(task.model)   # model is a 256-vector in [0, 1]^256
```

## The knobs

| Argument | What it means |
| --- | --- |
| `intrinsic_dim` (`d`) | How many dimensions actually matter. The real difficulty. |
| `ambient_dim` (`D`) | How many dimensions the optimizer sees. The camouflage. Must be `>= d`. |
| `bulk_scale` | How steeply the "useless" directions tilt. See below. |
| `noise` | Optional multiplicative Gaussian noise on the total loss. |

`bulk_scale` is the interesting optional one. Leave it at `0.0` and the useless directions
are *perfectly* flat, so there is a whole flat valley of equally-good answers. Set it to
something small like `1e-4` and they get a gentle upward tilt, which collapses that valley
to a single best point. Papers measuring real networks report the tilted version, so the
knob is there if you want it.

## Why it is a fair test

Two things had to be got right for the family to be usable as a benchmark.

**The best answer has to be findable.** The construction picks a random point `x*` inside
the unit box first, then arranges the arithmetic so that `x*` maps exactly onto the BBOB
function's optimum. So the `global_min` a task reports is not just a lower bound you can
never reach — an optimizer can actually land on it. This is easy to get wrong;
random-embedding Bayesian optimization ran into exactly this trap.

**The scaling has to be sane.** BBOB functions expect inputs in `[-5, 5]`, while the
optimizer works in `[0, 1]`. A factor of 10 sits in the embedding specifically so that
random guessing in the ambient box produces the same *spread* of function inputs as random
guessing in the function's native box — rather than accidentally squeezing everything into
a small region near the centre.

## What it does to optimizers

This is the point of the whole family. Optimizers that build a model of the search space
are badly hurt:

- CMA-ES learning a full covariance matrix,
- anything with per-coordinate step sizes or diagonal preconditioning,
- genetic algorithms that recombine individual coordinates.

They spend their effort modelling 256 dimensions when only 10 carry signal. Optimizers
that just follow the gradient are barely affected, because the gradient already lies in
the 10-dimensional subspace that matters.

That asymmetry creates a genuine, measurable *disagreement* about which optimizer to
pick — which is exactly what makes these tasks valuable as training data for a selection
policy.

## One honest wrinkle

Squashing 256 uniform numbers down to 10 produces bell-curve-shaped values, not uniform
ones. The average spread matches the native box, but the tails stick out: roughly a third
of the time a coordinate lands outside BBOB's `[-5, 5]` range, where the functions add a
boundary penalty. Nothing is broken, but the embedded landscape is not a pure copy of the
original — it includes some of that penalty region.

## Tags and features

Embedded tasks inherit the underlying BBOB function's characteristics, with two
adjustments:

- `separable` is forced to `False`. Mixing all 256 inputs into every one of the 10
  destroys any coordinate independence the original function had.
- `unimodal` is inherited. The two terms of the loss live in orthogonal subspaces, so
  local minima of the embedded task correspond one-to-one with local minima of the
  underlying function.

The tags also carry `embedded`, `intrinsic_dimensionality`, `bulk_scale`, and
`dimensionality` (which is `ambient_dim`).

The policy feature `schema`, however, is deliberately identical to plain `bbob.yaml` —
just `dimensionality`, `separable`, `unimodal`. The embedding metadata stays in the tags
but is **not** exposed as a feature, because on a real out-of-distribution problem nobody
knows the intrinsic dimension.

## The shipped suite

`bbob_embedded.yaml` is a grid of 24 functions × `{2, 5, 10, 40}` intrinsic dimensions ×
`{64, 1024}` ambient dimensions × 4 seeds = **768 tasks**, with `bulk_scale` at `0.0`.
Each seed is a distinct BBOB instance — its own shift, rotation and `global_min` — so the
seed axis grows the task pool rather than repeating work. (Rollout repeats are a separate
axis, `n_realizations`.)

Both dimensionality grids are deliberately thinned from the 6 × 3 they started as, which
came to 1728 tasks — 3× the native `bbob.yaml` suite. The dropped cells were *measured*
to be redundant on the seed-0 slice of the `bbob_embedded` databank, using a per-task
64-optimizer success signature and the transfer regret of applying one cell's best
optimizer to another:

- **ambient 256** went because 1024 → 256 regret is 0.017, the lowest of all six ordered
  ambient pairs, and 64 ↔ 256 is the most redundant adjacent pair in the grid. 64 and
  1024 both stay for a reason: 64 is the only ambient dimension where the whole
  64-optimizer portfolio runs (`cmaes` is already gone at 256, and five more optimizers
  at 1024), while 1024 supplies the extreme rank-deficiency regime — ambient/intrinsic
  up to 512.
- **intrinsic 3 and 20** went because adjacent intrinsic dimensions are near-duplicates
  (2 ↔ 3 rank correlation 0.908, 10 ↔ 20 0.916, 20 ↔ 40 0.910, all with regret ≤ 0.024)
  while distant ones are not (2 ↔ 40 is 0.558). So 3 is covered by 2, and 20 by both 10
  and 40.

All 24 functions are kept on purpose: none is uninformative, and pruning any would break
the drop-in comparability with `bbob.yaml` that the shared `schema` promises.

Override per run:

```bash
# curved bulk instead of a flat null space
tasks.task_kwargs.bulk_scale=1e-4

# a single ambient dimensionality
tasks.experimentdata.domain.input.ambient_dim.categories=[1024]
```

Embedded tasks also make up part of the mixed `bbob_balanced_train` and
`bbob_balanced_test` suites (171 and 173 rows respectively).

## The formula, for reference

For the record, the loss is

$$
F(x) = f\bigl(z_{\text{opt}} + s \cdot B (x - x^{*})\bigr)
       + \frac{\texttt{bulk\_scale}}{2} \bigl\lVert P_{\perp} (x - x^{*}) \bigr\rVert^{2}
$$

where `x` lives in `[0, 1]^D`, `B` is a `(d, D)` matrix with orthonormal rows (the reduced
QR factor of a random Gaussian), `P⊥ = I − BᵀB` projects onto its null space, `s` is the
width of the function's native box, and `x*` is a seeded uniform draw from the box. A
single `seed` generates the BBOB instance, the matrix `B`, and the anchor `x*`.

`z_opt` is where the underlying BBOB function's minimum sits in its *own* `d`-dimensional
coordinates — the instance's seeded shift, taken straight from `problem.x_opt`. It is the
term that makes the optimum reachable: at `x = x*` the displacement vanishes, the argument
collapses to exactly `z_opt`, and so `F(x*) = f(z_opt) = f_opt`. The two "optimum"
quantities are therefore distinct and both stored in the task — `z_opt` locates it in
function space, `x*` locates it in search space, and the embedding is built so the two
correspond.

## References

- Gur-Ari, Roberts & Dyer, *Gradient Descent Happens in a Tiny Subspace*,
  arXiv:1812.04754, 2018.
- Li, Farkhoor, Liu & Yosinski, *Measuring the Intrinsic Dimension of Objective
  Landscapes*, ICLR 2018.
- Sagun, Evci, Guney, Dauphin & Bottou, *Empirical Analysis of the Hessian of
  Over-Parametrized Neural Networks*, arXiv:1706.04454, 2017.
- Papyan, *Measurements of Three-Level Hierarchical Structure in the Outliers in the
  Spectrum of Deepnet Hessians*, ICML 2019.
- Wang, Hutter, Zoghi, Matheson & de Freitas, *Bayesian Optimization in a Billion
  Dimensions via Random Embeddings*, JAIR 55, 2016.
