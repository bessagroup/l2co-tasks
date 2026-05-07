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

Public API lives at the package root (`src/l2co_tasks/__init__.py`); all implementation is under `src/l2co_tasks/_src/` and is not meant to be imported directly. `l2co_tasks.tasks` is a secondary re-export module that exposes only the `create_*` factories. The top-level package additionally re-exports the `f3dasm.Block` samplers `CEC2019Sampler` and `PDETaskSampler` for use in `f3dasm.ExperimentData` pipelines.

### The `Task` abstraction (`_src/task.py`)

`Task` is an `equinox.Module` with these fields:
- `model`: PyTree of parameters (JAX arrays or `eqx.Module`) — the only trainable part.
- `loss_fn`: static callable with signature `loss_fn(model, ...)`; may accept a batch of data and/or an RNG key depending on the flags below.
- `pass_rng` (static): whether the optimizer must supply a `key` to `loss_fn`.
- `has_aux` (static): whether `loss_fn` returns `(loss, aux)`.
- `dataset` (static `DatasetDict | None`): `{dataset_path, batch_size, seed}` — paths only; arrays are loaded lazily via the `loaded_dataset` property. `None` (the default) means the task carries no dataset; `loaded_dataset` returns `{}` and `batch_size` returns `None` in that case.
- `tag` (static dict): free-form metadata (function name, dimensionality, seed, noise, etc.) used for identity via `tag_hashable`, `hash`, `__eq__`, `__hash__`.

Serialization is a **single-file `.eqx` format**: a UTF-8 JSON header line (loss function + model skeleton pickled via `cloudpickle` and hex-encoded, plus dataset reference and tags) followed by the binary leaf bytes from `eqx.tree_serialise_leaves`. `Task.save` / `Task.load` are the canonical (and only) entrypoints — the legacy multi-file `.json` format and its `from_dict` / `save_to_json` shims have been removed, as have `_io.load_model` and `_io.load_jnparray`. `_src/_io.py` now only provides `load_dataset` for `.npz` files. `dataset_path` is stored **relative to the `.eqx` file** so a task and its `.npz` dataset can be moved together.

Key invariant: `loss_fn` and the model skeleton are round-tripped via `cloudpickle.dumps(...).hex()`. When editing loss functions, keep them picklable (module-level or closures over picklable values) — anonymous lambdas over non-picklable state will break `save`/`load`.

### Task families (each in its own `_src/*_task.py` or module)

Every family exposes a top-level `create_<name>_task(...)` factory that builds and returns a `Task`:

- `benchmark_task.py` — `create_bbob_task`, `create_cec2005_task`. Wraps `bbob_jax` functions; applies `scale_input` (maps `[0,1]^d` to the function's native bounds) and optional multiplicative-Gaussian `add_noise`. `CEC2019Sampler` is an `f3dasm.Block` that enumerates the CEC 2019 suite.
- `quadratic_task.py` — `create_quadratic_task`: random `min ||Wx − y||^2`.
- `spiral_task.py`, `mnist1d_task.py`, `task_gaussian_class.py`, `gaussian_meta.py` — supervised-learning tasks; build a model from `_src/models.py` (RNN/MLP/CNN) and wire it to a loss from `_src/loss_fn.py` (MSE or softmax cross-entropy with optional L2).
- `pde.py` — PINN-style tasks (`create_pde_task`, `PDETaskSampler`) for convection, reaction, and wave equations using an MLP.
- `continue_from_path.py` — `retrieve_tasks` extracts unique tasks from an `f3dasm.ExperimentData`.
- `cifar10_task.py` — currently provides only the dataset-side helpers `download_cifar10` and `save_dataset`; `create_cifar10_task` is checked in but commented out and **not** part of the public API. Treat this module as work-in-progress and do not add it to `__init__.py` until the factory is restored.

### Conventions shared across factories

- Input domain is normalized to `[0, 1]^d` and scaled inside the loss (see `scale_input` in `benchmark_task.py`). Don't feed raw function-native coordinates to a `Task` model.
- `pass_rng=True` is set when the loss has stochasticity (added noise, or CEC2005 functions `f4/f17/f24/f25` which are inherently stochastic).
- `tag` is populated with dimensionality, seed, noise, and any function-characteristics metadata from the underlying registry — downstream code relies on `tag["task_name"]`, `tag["fn_name"]`, etc. for identity and filtering.
- The `dimensionality` property counts inexact-array leaves via `eqx.filter(model, eqx.is_inexact_array)` — so only JAX float arrays count as parameters.

## Tooling notes

- Ruff is configured with a **79-char line length** and numpy-style docstrings; `__init__.py` files are exempt from `F401`/`E402`.
- `pyproject.toml` is auto-sorted by `toml-sort` via pre-commit — editing it manually then committing will trigger a reformat.
- GitHub workflows in `.github/workflows/` are currently fully commented out; CI is effectively not running. Don't assume pushes are gated by tests.
