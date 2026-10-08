# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`l2co-tasks` is a Python 3.12+ package providing optimization tasks for the L2CO (learning-to-continuously-optimize) library. Tasks wrap a JAX/Equinox model together with a loss function, optional dataset, and metadata tags into a single serializable `Task` object.

## Common commands

Package management uses `uv`. Tests use `pytest`, linting/formatting uses `ruff`, docs use `mkdocs`.

- Install dev deps: `uv sync` (or `pip install -e '.[dev,tests,docs]'`)
- Run all tests: `make test` or `pytest`
- Run one test: `pytest tests/test_smoke.py::test_create_bbob_task_sphere`
- Skip slow/heavy tests: `pytest -m "not slow and not requires_f3dasm"`
- Lint: `make lint` or `ruff check`
- Format: `ruff format`
- Pre-commit: `pre-commit run --all-files`
- Build docs: `make docs` (output in `site/`)
- Build distribution: `make build`

`f3dasm` is a required dependency and is resolved as an **editable install from `../f3dasm`** (see `[tool.uv.sources]` in `pyproject.toml`). That sibling checkout must exist for `uv sync` to succeed.

Pytest config (`[tool.pytest.ini_options]` in `pyproject.toml`) declares two markers — `slow` (writes a dataset `.npz` or runs an inner training loop) and `requires_f3dasm` (constructs an `f3dasm.ExperimentData`) — and treats any `DeprecationWarning` originating from `l2co_tasks` as an error. Don't introduce internal deprecation warnings without also updating call sites; CI-equivalent runs will fail.

## Architecture

**Never import `l2co` or `l2co_optimizers` from `src/`.** A ruff `TID251` banned-api rule fails the lint: this package knows tasks, not optimizers, and `l2co` is the only bridge between a `Task` and an `OptimizationStep` (l2co ADR 0018).

Public API lives at the package root (`src/l2co_tasks/__init__.py`); all implementation is under `src/l2co_tasks/_src/` and is not meant to be imported directly. `l2co_tasks.tasks` is a secondary re-export module that exposes only the per-family `create_*` factories (it does **not** re-export `create_tasks_experimentdata` or `retrieve_tasks`). The top-level package additionally re-exports the `f3dasm.Block` samplers `CEC2019Sampler` and `PDETaskSampler`, plus the `f3dasm.ExperimentData` helpers `create_tasks_experimentdata` and `retrieve_tasks`, for use in `f3dasm.ExperimentData` pipelines.

### The `Task` abstraction (`_src/task.py`)

`Task` is an `equinox.Module` with these fields:
- `model`: PyTree of parameters (JAX arrays or `eqx.Module`) — the only trainable part.
- `loss_fn`: static callable with signature `loss_fn(model, ...)`; may accept a batch of data and/or an RNG key depending on the flags below.
- `global_min` (static `float | None`): the known global minimum of the loss for this task (used for regret/gap reporting); defaults to `None`.
- `pass_rng` (static): whether the optimizer must supply a `key` to `loss_fn`.
- `has_aux` (static): whether `loss_fn` returns `(loss, aux)`.
- `dataset` (static `DatasetDict | None`): `{dataset_path, batch_size, seed}` — paths only; arrays are loaded lazily via the `loaded_dataset` property. `None` (the default) means the task carries no dataset; `loaded_dataset` returns `{}` and `batch_size` returns `None` in that case.
- `tag` (static dict): free-form metadata (function name, dimensionality, seed, noise, etc.) used for identity via `tag_hashable`, `hash`, `__eq__`, `__hash__`.
- `constraints` (`tuple[Constraint, ...]`, default `()`): equality, inequality and box constraints on the model's parameters (`_src/constraints.py`, ADR 0002). Elements are `Inequality(fn, name, tol=0)`, `Equality(fn, name, tol=1e-4)` or `Box(lower, upper)`; all are called as `c(model)` (no key, no data batch) and return raw values read as "≤ 0 is satisfied". `Task.__post_init__` runs `resolve_constraints`, which enforces the rules (at most one `Box`, unique names, box matches the model and contains the starting model, …) and casts a single-number `Box` to the model's parameters — so it re-runs on `dataclasses.replace`. Constraints join `tag_hashable` **only when present**: an unconstrained task's `hash` and `.eqx` bytes are exactly what they were before constraints existed, pinned by `tests/test_task_golden_hashes.py`. Enforcement is l2co's (l2co ADR 0020, stage 2 of ADR 0002): it forwards a uniform `Box` to the optimizers and refuses a task with an `Inequality` or `Equality`. `create_gaussian_meta_task` is the one factory that sets a constraint (`Box(0, 1)`); `estimate_global_min` searches inside a task's `Box` (it no longer takes `clip_to_unit`) and refuses an `Inequality`/`Equality`.

Serialization is a **single-file `.eqx` format**: a UTF-8 JSON header line (loss function + model skeleton pickled via `cloudpickle` and hex-encoded, plus dataset reference and tags; a `constraints` key, likewise pickled, is written only when the task has constraints) followed by the binary leaf bytes from `eqx.tree_serialise_leaves`. `Task.save` / `Task.load` are the canonical (and only) entrypoints — the legacy multi-file `.json` format and its `from_dict` / `save_to_json` shims have been removed, as have `_io.load_model` and `_io.load_jnparray`. `_src/_io.py` now only provides `load_dataset` for `.npz` files. `dataset_path` is stored **relative to the `.eqx` file** so a task and its `.npz` dataset can be moved together.

Key invariant: `loss_fn` and the model skeleton are round-tripped via `cloudpickle.dumps(...).hex()`. The skeleton is read back with `_SkeletonUnpickler`, which rebuilds each `jax.ShapeDtypeStruct` from its shape and dtype only: the fields JAX pickles for that class change between releases (JAX 0.11 dropped `vma`), and unpickling straight into the running JAX's class made files written under another JAX unloadable (`tests/test_skeleton_across_jax_versions.py`). When editing loss functions, keep them picklable (module-level or closures over picklable values) — anonymous lambdas over non-picklable state will break `save`/`load`.

### Host objectives (`_src/host_objective.py`, ADR 0003)

A task whose objective is computed outside JAX (compiled Fortran, a simulator) gets an ordinary `loss_fn` from `host_loss(opener)`. The opener is a picklable, hashable, zero-argument callable returning an objective with `value(x) -> float` and, optionally, `value_and_grad(x) -> (float, ndarray)`, on the model's floating-point parameters flattened to one float64 vector.
- `HostLoss` is an `eqx.Module` holding only the opener (static), so a `Task` holding it saves and loads without opening anything, and equal openers give equal, equally hashed losses.
- The value is a `jax.pure_callback` inside a `jax.custom_jvp` whose rule asks the host for value and gradient in one call. It must stay `custom_jvp`: optimistix takes gradients with `jax.linearize` (forward mode), which a `custom_vjp` breaks. `vmap_method="expand_dims"` hands the host a whole batch.
- Per process, `_OPENED` maps each opener to its opened objective, two last-point caches and the traced function; an `RLock` serializes opening and host calls. There is one cache per kind of request (value, value-and-gradient) and they never answer for each other: an objective's two routines may differ in the last bit, and mixing them made a run's numbers depend on which run came before it in the same process. The objective is opened at first trace, in `HostLoss.__call__`, not inside the callback, where a failure would arrive wrapped in a `JaxRuntimeError`.
- Tracing with non-float64 parameters warns. A dataset batch or key raises `TypeError` (not supported yet). Differentiating an objective without `value_and_grad` raises `TypeError` at trace time.
- Adapters must be deterministic, make at most one billed evaluation per host call (no hidden finite differences) and compute in float64. Test openers must be module-level and hashable (`tests/test_host_objective.py` uses a frozen dataclass) so `Task.save` can pickle them.

### Task families (each in its own `_src/*_task.py` or module)

Every family exposes a top-level `create_<name>_task(...)` factory that builds and returns a `Task`:

- `benchmark_task.py` — `create_bbob_task`, `create_bbob_noisy_task`, `create_cec2005_task`, `create_cec2017_task`, `create_cec2013lsgo_task`. Wraps `bbob_jax` functions; applies `scale_input` (maps `[0,1]^d` to the function's native bounds) and optional multiplicative-Gaussian `add_noise`. CEC 2017 tasks propagate `bbob_jax`'s `ValueError` below a function's `min_ndim` (hybrids need one dimension per subcomponent kernel) and record the bound in `tag["min_ndim"]`. CEC 2013 LSGO is a *fixed-instance* suite (official constants, not seed-sampled) defined only at its native dimensionality — 1000, or 905 for the overlapping `f13`/`f14`; its factory and `conf/tasks/cec2013lsgo.yaml` both require a bbob-jax newer than 2.0.0. `CEC2019Sampler` is an `f3dasm.Block` that enumerates the CEC 2019 suite.
- `quadratic_task.py` — `create_quadratic_task`: random `min ||Wx − y||^2`.
- `cutest_task.py` — `create_cutest_task(problem, sif_params=None, *, global_min=None)` (ADR 0004): an unconstrained CUTEst problem through pycutest, behind a `host_loss`. Needs the optional `[cutest]` extra (pycutest is GPL) and a system CUTEst with `CUTEST`, `SIFDECODE` and `MASTSIF` set; even `import pycutest` raises `RuntimeError` without `CUTEST`, so the import is lazy and tests skip on `ImportError` or `RuntimeError` (always in CI). The model is the problem's `x0`. Bound-constrained, fixed-variable and integer problems are refused. `global_min` comes from the committed table `_src/cutest_global_min.csv` (one row per problem and canonical SIF-parameter string, with the SIF file's sha256; a changed SIF file or an excluded row refuses), and the table is written by an l2co_experiments experiment, not by hand. Passing `global_min` bypasses the table. The tag is flat: SIF parameters as one canonical string (`"N=10"`), classification fields as `cutest_*` keys.
- `spiral_task.py`, `mnist1d_task.py`, `task_gaussian_class.py`, `gaussian_meta.py` — supervised-learning tasks; build a model from `_src/models.py` (RNN/MLP/CNN) and wire it to a loss from `_src/loss_fn.py` (MSE or softmax cross-entropy with optional L2).
- `pde.py` — PINN-style tasks (`create_pde_task`, `PDETaskSampler`) for convection, reaction, and wave equations using an MLP.
- `helmholtz.py`, `viscous_burgers.py`, `inviscid_burgers.py`, `euler.py`, `stokes.py`, `pkpd.py` — the PINN benchmark suite of Jnini et al. 2026 (`create_helmholtz_task`, `create_viscous_burgers_task`, `create_inviscid_burgers_task`, `create_euler_task`, `create_stokes_task`, `create_pkpd_task` plus a `*TaskSampler` each). Shared PINN helpers (Latin-hypercube / triangle samplers, batched eval) live in `pinn.py`; the `FourierMLP` (periodic Fourier features) and `MultiNet` (multi-network) model builders live in `models.py`. Each loss is a picklable module-level function bound with `functools.partial`; `global_min=0.0`. `create_euler_task(stage=...)` selects the viscous warm-up vs the inviscid HLLC stage. `create_inviscid_burgers_task` uses a two-network `MultiNet` model.
- `continue_from_path.py` — `retrieve_tasks` extracts unique tasks from an `f3dasm.ExperimentData`.
- `experimentdata.py` — `create_tasks_experimentdata` builds an `f3dasm.ExperimentData` from a collection of tasks.
- `cifar10_task.py` — currently provides only the dataset-side helpers `download_cifar10` and `save_dataset`; `create_cifar10_task` is checked in but commented out and **not** part of the public API. Treat this module as work-in-progress and do not add it to `__init__.py` until the factory is restored.

### Conventions shared across factories

- Input domain is normalized to `[0, 1]^d` and scaled inside the loss (see `scale_input` in `benchmark_task.py`). Don't feed raw function-native coordinates to a `Task` model. The one deliberate exception is CUTEst (ADR 0004): its model is the problem's prescribed start `x0`, in the problem's own coordinates.
- `pass_rng=True` is set when the loss has stochasticity (added noise, or CEC2005 functions `f4/f17/f24/f25` which are inherently stochastic).
- `tag` is populated with dimensionality, seed, noise, and any function-characteristics metadata from the underlying registry — downstream code relies on `tag["task_name"]`, `tag["fn_name"]`, etc. for identity and filtering. Keep tags flat: a dict with two or more keys inside a tag doesn't hash the same in every process until #21's fix lands.
- The `dimensionality` property counts inexact-array leaves via `eqx.filter(model, eqx.is_inexact_array)` — so only JAX float arrays count as parameters.

## Tooling notes

- Ruff is configured with a **79-char line length** and numpy-style docstrings; `__init__.py` files are exempt from `F401`/`E402`.
- `pyproject.toml` is auto-sorted by `toml-sort` via pre-commit — editing it manually then committing will trigger a reformat.
- CI: `.github/workflows/pull_request.yml` runs on every PR -- ruff (pinned to the pre-commit rev; bump the two together), pre-commit, `pytest -m "not slow"` on Linux/macOS x Python 3.12/3.13, the package build and the docs build. It checks out `bessagroup/f3dasm` next to the repo (the `../f3dasm` editable source) at `main` for a PR into `main` and at `develop` otherwise. `requires_l2co` tests skip there (no l2co). `build_docs.yml` builds the docs on pushes to `main`; `release.yml` publishes.

## Agent skills

### Issue tracker

Issues are tracked in **GitHub Issues** for `bessagroup/l2co-tasks` via the `gh` CLI; external PRs are **not** a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. Only `wontfix` exists today; the other four are created on first triage. See `docs/agents/triage-labels.md`.

### Domain docs

**Single-context** — `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## Related repositories

Part of the L2CO ecosystem (Bessa Research Group). These repositories work together:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.
