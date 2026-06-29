![Bessa Research Group](img/bessa_group_logo.png)

# L2CO Tasks

| [**GitHub**](https://github.com/bessagroup/l2co-tasks)
| [**Documentation**](https://l2co-tasks.readthedocs.io/en/latest/)

Optimization tasks compatible with the L2CO library

**First publication:** April 13, 2026

***

## Summary

`l2co-tasks` provides a unified collection of optimization tasks for the [L2CO](https://github.com/bessagroup/l2co) ecosystem. Each `Task` bundles a JAX/Equinox model, a loss function, an optional dataset, and metadata tags into a single, serializable object that any L2CO optimizer or rollout can consume. The package ships ready-made task families — analytic black-box benchmarks ([BBOB](https://github.com/bessagroup/bbob-jax) and CEC 2005), quadratic problems, supervised-learning tasks (spiral, MNIST-1D, Gaussian classification), and PINN-style PDE problems (convection, reaction, wave) — together with `create_*_task` factories and `f3dasm` samplers for building experiment datasets.

## Statement of need

Research on learning to optimize and optimizer selection requires evaluating many optimizers across a diverse, well-characterized set of problems — but these problems usually come from incompatible sources with different input ranges, calling conventions, and metadata. `l2co-tasks` standardizes them behind a single `Task` abstraction: inputs are normalized to `[0, 1]^d` and scaled inside the loss, stochasticity and known global minima are tracked explicitly, and every task carries a hashable tag and serializes to a single `.eqx` file. This makes tasks reproducible, portable, and directly pluggable into the [`l2co`](https://github.com/bessagroup/l2co) / [`rl2co`](https://github.com/bessagroup/rl2co) rollout machinery and [`f3dasm`](https://github.com/bessagroup/f3dasm) experiment pipelines.

## Authorship

**Authors**:
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Authors affiliation:**
- Delft University of Technology (Bessa Research Group)

**Maintainer:**
- Martin van der Schelling ([m.p.vanderschelling@tudelft.nl](mailto:m.p.vanderschelling@tudelft.nl))

**Maintainer affiliation:**
- Delft University of Technology (Bessa Research Group)


## Getting started

`l2co-tasks` is `uv`-managed and depends on an editable install of a sibling [`f3dasm`](https://github.com/bessagroup/f3dasm) checkout, so lay the repositories out side-by-side before syncing:

```bash
git clone https://github.com/bessagroup/f3dasm.git
git clone https://github.com/bessagroup/l2co-tasks.git
cd l2co-tasks
uv sync
```

Create a task from one of the factories and evaluate its loss:

```python
from l2co_tasks import create_bbob_task

task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)
loss = task.loss_fn(task.model)   # model is the [0, 1]^d input vector
```

Tasks serialize to a single-file `.eqx` format via `Task.save` / `Task.load`. See the [documentation](https://l2co-tasks.readthedocs.io/en/latest/) for the full list of task families and `create_*_task` factories. To build your own task from scratch, see the [Create your own task](./docs/create_task.ipynb) guide.

## Available tasks

Every task is built by a `create_<name>_task(...)` factory and returned as a single `Task` object. The package ships the following families:

| Category | Task | Factory | `global_min` | Description | Reference |
|---|---|---|---|---|---|
| Black-box | BBOB | `create_bbob_task` | analytical | 24 analytic, noiseless black-box functions over `[0, 1]^d`; optional multiplicative Gaussian noise. | [bbob-jax](https://github.com/bessagroup/bbob-jax) |
| Black-box | CEC 2005 | `create_cec2005_task` | analytical | CEC 2005 real-parameter functions with per-function bounds; `f4/f17/f24/f25` are stochastic. | [bbob-jax](https://github.com/bessagroup/bbob-jax) |
| Least-squares | Random quadratic | `create_quadratic_task` | analytical | Minimize `\|\|W x - y\|\|^2` for random Gaussian `W`, `y` (square or over-determined). | Maheswaranathan et al. (2019) |
| Supervised | Two-spiral | `create_spiral_task` | empirical | GRU-RNN trained with MSE to separate two interleaved spirals. | — |
| Supervised | MNIST-1D | `create_mnist1d_task` | empirical | MLP softmax classifier on the 1-D MNIST surrogate dataset. | [Greydanus (2020)](https://github.com/greydanus/mnist1d) |
| Supervised | Gaussian blobs | `create_gaussian_task` | empirical | MLP classifier (cross-entropy + L2) on Gaussian-cluster data. | — |
| Meta-learning | Adam hyperparameters | `create_gaussian_meta_task` | empirical | Outer objective tunes Adam's `(lr, b1, b2)` for an inner MLP training run. | — |
| PINN | 1-D PDE | `create_pde_task` | 0 (theoretical) | MLP PINN for the `convection`, `reaction`, or `wave` equation (collocation-residual MSE). | — |
| PINN | Helmholtz | `create_helmholtz_task` | 0 (theoretical) | Fourier-feature MLP for the 2-D/3-D Helmholtz equation. | Jnini et al. (2026) |
| PINN | Stokes | `create_stokes_task` | 0 (theoretical) | MLP for lid-driven Stokes flow in a wedge (Moffatt eddies). | Jnini et al. (2026) |
| PINN | Viscous Burgers | `create_viscous_burgers_task` | 0 (theoretical) | MLP for the 2+1-D viscous Burgers equation with closed-form targets. | Jnini et al. (2026) |
| PINN | Inviscid Burgers | `create_inviscid_burgers_task` | 0 (theoretical) | Two-network `MultiNet` with entropy consistency for the shock-forming inviscid Burgers law. | Jnini et al. (2026) |
| PINN | Euler (Sod) | `create_euler_task` | 0 (theoretical) | MLP for the 1-D compressible Euler shock tube; viscous warm-up + inviscid HLLC stages. | Jnini et al. (2026) |
| PINN | Stiff PK-PD | `create_pkpd_task` | 0 (theoretical) | MLP for a stiff pharmacokinetic–pharmacodynamic ODE (paclitaxel). | Jnini et al. (2026) |

Every `Task` records a `global_min` used for regret/gap reporting, set by the mechanism appropriate to the problem:

- **analytical** — retrieved or computed in closed form (the BBOB/CEC 2005 registry optima, the quadratic least-squares residual). A true mathematical lower bound on the noiseless loss.
- **0 (theoretical)** — the loss is a sum of PDE-residual / boundary / initial-condition mean-squared-error terms, so `0` is the theoretical minimum (attained when the network solves the PDE exactly). A genuine lower bound, though a finite-width network need not reach it.
- **empirical** — the true minimum is unreachable (overlapping classes forbid perfect separation, or there is no closed form), so the factory runs a short, seeded, multi-restart Adam **benchmark at task-creation time** and records the best loss found (see `estimate_global_min`). This is a best-achievable *estimate*, not a guaranteed lower bound; it is deterministic in the task seed, so it is stable across rebuilds. Pass `estimate_global_min=False` to those factories to skip the benchmark and leave `global_min` unset.

The physics-informed suite reproduces the benchmarks of **Jnini et al. (2026)**, *Curvature-aware optimization for high-accuracy physics-informed neural networks*, [arXiv:2604.05230](https://arxiv.org/abs/2604.05230).

## Hydra task configurations

For large-scale studies, `l2co-tasks` ships ready-made [Hydra](https://hydra.cc) config groups under `l2co_tasks/conf/tasks/` (installed as package data). Each YAML describes a whole **task distribution** rather than a single task: an [`f3dasm`](https://github.com/bessagroup/f3dasm) sampler and domain (for example the grid over `fn_name × dimensionality × seed`), a `data_generator` pointing at the matching `create_*_task` factory, a feature `schema`, and the optimization bounds. Suites are provided for every family — `bbob`, `bbob_small`, `bbob_diverse` (a 7-function subset covering all five BBOB difficulty groups) and its held-out counterpart `bbob_diverse_holdout` (same functions, disjoint seeds), `bbob_holdout`, `cec2005`, `quadratic`, `spirals`, `mnist1d`, `gaussian_classification`, `gaussian_meta`, and one per PINN problem (`helmholtz`, `stokes`, `viscous_burgers`, `inviscid_burgers`, `euler`, `pkpd`, `pde`).

Downstream applications such as [`l2co_experiments`](https://github.com/bessagroup/l2co_experiments) consume these by adding `l2co-tasks` to the Hydra search path and selecting a suite by name:

```yaml
# in your primary Hydra config
hydra:
  searchpath:
    - pkg://l2co_tasks.conf

defaults:
  - tasks: bbob          # any file in l2co_tasks/conf/tasks/
```

The selected suite can be overridden from the command line (e.g. `... tasks=cec2005`) and materialized into an `f3dasm.ExperimentData` of `Task` objects via `create_tasks_experimentdata(config=config.tasks, ...)`.

## Examples and benchmarks

Two runnable notebooks demonstrate and benchmark the core functionality end-to-end. Both run against the installed package after `uv sync` and are rendered in the [documentation](https://l2co-tasks.readthedocs.io/en/latest/):

- [Create your own task](./docs/create_task.ipynb) — builds a `Task` from scratch and exercises the `Task` API (loss evaluation, serialization, metadata).
- [PINN benchmark tasks](./docs/pinn_tasks.ipynb) — instantiates and evaluates the physics-informed PDE benchmark suite (Helmholtz, Burgers, Euler, Stokes, PK–PD).

## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co-tasks/issues) to report them.

## License

Copyright (c) 2026, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](https://github.com/bessagroup/l2co-tasks/blob/main/LICENSE) for the full license text.

## Related repositories

This package is part of the L2CO ecosystem developed in the [Bessa Research Group](https://github.com/bessagroup). The repositories below work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.

