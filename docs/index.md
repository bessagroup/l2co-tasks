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

Tasks serialize to a single-file `.eqx` format via `Task.save` / `Task.load`. See the [API reference](api.md) for the full list of task families and `create_*_task` factories. To build your own task from scratch, see the [Create your own task](create_task.ipynb) guide.

## Community Support

If you find any **issues, bugs or problems** with this package, please use the [GitHub issue tracker](https://github.com/bessagroup/l2co-tasks/issues) to report them.

## License

Copyright (c) 2026, Martin van der Schelling

All rights reserved.

This project is licensed under the BSD 3-Clause License. See [LICENSE](license.md) for the full license text.

## Related repositories

This package is part of the L2CO ecosystem developed in the [Bessa Research Group](https://github.com/bessagroup). The repositories below work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.

