# Getting started

This page walks through installing `l2co-tasks` and creating your first task.

## Installation

`l2co-tasks` is managed with [`uv`](https://docs.astral.sh/uv/) and depends on an
**editable install of a sibling [`f3dasm`](https://github.com/bessagroup/f3dasm)
checkout**. `f3dasm` is pinned through `[tool.uv.sources]` in `pyproject.toml` and is
not yet published on PyPI, so the two repositories must be laid out side-by-side
before syncing:

```bash
git clone https://github.com/bessagroup/f3dasm.git
git clone https://github.com/bessagroup/l2co-tasks.git
cd l2co-tasks
uv sync
```

`uv sync` resolves `f3dasm` from `../f3dasm`, so that sibling checkout must exist or the
sync will fail. The command creates a virtual environment in `.venv/` with all runtime,
test, and documentation dependencies installed.

To work inside the environment, prefix commands with `uv run` (for example
`uv run python`, `uv run pytest`) or activate it with `source .venv/bin/activate`.

## A minimal example

Every task family exposes a `create_<name>_task(...)` factory that returns a single
[`Task`](api.md#l2co_tasks.Task) object — a serializable bundle of a model, a loss
function, an optional dataset, and metadata tags:

```python
from l2co_tasks import create_bbob_task

task = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)
loss = task.loss_fn(task.model)   # model is the [0, 1]^d input vector
```

Inputs are normalized to `[0, 1]^d` and scaled to the function's native bounds inside
the loss, so you never pass raw function-native coordinates to a task model.

Tasks serialize to a single-file `.eqx` format through `Task.save` / `Task.load`:

```python
task.save("sphere.eqx")
restored = Task.load("sphere.eqx")
```

## Where to go next

- **[API reference](api.md)** — the full list of task families and their
  `create_*_task` factories.
- **[Create your own task](create_task.ipynb)** — a guided notebook that builds a `Task`
  from scratch.
- **[PINN benchmark tasks](pinn_tasks.ipynb)** — a notebook tour of the
  physics-informed PDE benchmark suite.

Both notebooks run end-to-end against the installed package and double as the project's
reproducible example/benchmark suite.
