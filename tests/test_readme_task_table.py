"""The "Available tasks" tables in the README and the docs list every
task family.

The tables are hand-written, so nothing else stops them drifting when a
family is added: CUTEst and CEC 2013 LSGO both shipped without a row. A
family is a public ``create_*_task`` factory, and every task set under
``l2co_tasks/conf/tasks/`` builds its tasks with one. So a new task set
needs a public factory, and a new public factory needs a row in both
tables.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest

import l2co_tasks

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "docs/index.md"]

# The Factory column is the third cell of a row:
# | Category | Task | `create_<name>_task` | ...
FACTORY = re.compile(
    r"^\| [^|\n]+ \| [^|\n]+ \| `(create_\w+_task)` \|", re.MULTILINE
)
# Any factory a task-set YAML instantiates, including the nested parts of
# a composite set.
TARGET = re.compile(r"_target_:\s*l2co_tasks\.(create_\w+_task)\b")


def _public_factories() -> set[str]:
    return {
        name
        for name in l2co_tasks.__all__
        if name.startswith("create_") and name.endswith("_task")
    }


def _table_factories(path: Path) -> list[str]:
    section = path.read_text().split("## Available tasks", 1)[1]
    section = section.split("\n## ", 1)[0]
    return FACTORY.findall(section)


def _task_set_factories() -> dict[str, set[str]]:
    tasks = resources.files("l2co_tasks.conf") / "tasks"
    return {
        p.name: set(TARGET.findall(p.read_text()))
        for p in tasks.iterdir()
        if p.name.endswith(".yaml")
    }


@pytest.mark.parametrize("doc", DOCS)
def test_task_table_lists_every_public_factory(doc: str) -> None:
    listed = _table_factories(ROOT / doc)
    assert len(listed) == len(set(listed)), f"duplicate rows in {doc}"
    public = _public_factories()
    missing = sorted(public - set(listed))
    stale = sorted(set(listed) - public)
    assert not missing, f"{doc}'s task table has no row for {missing}"
    assert not stale, f"{doc}'s task table lists non-public {stale}"


def test_every_task_set_builds_with_a_public_factory() -> None:
    sets = _task_set_factories()
    assert sets, "no task-set YAMLs found"
    public = _public_factories()
    for name, factories in sorted(sets.items()):
        assert factories, f"{name} names no l2co_tasks.create_*_task"
        unlisted = sorted(factories - public)
        assert not unlisted, (
            f"{name} builds with {unlisted}, which is not a public "
            "factory, so the task tables cannot list it"
        )
