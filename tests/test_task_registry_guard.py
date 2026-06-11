"""Guard: every public ``create_*`` factory has a contract case.

This is what makes "when I add a new task, it gets tested" *enforced*:
adding a ``create_*_task`` factory to the public API without registering
a :class:`~tests.task_cases.TaskCase` for it fails this test, so the new
task can never silently skip the contract battery.
"""

from __future__ import annotations

import l2co_tasks

from .task_cases import TASK_CASES

# Factories that build a single optimization ``Task`` and must therefore
# be covered by the contract suite. ``create_tasks_experimentdata`` is an
# f3dasm pipeline helper (it builds an ``ExperimentData``, not a ``Task``)
# and is intentionally excluded.
_NON_TASK_FACTORIES = {"create_tasks_experimentdata"}


def _public_task_factories() -> set[str]:
    """Names of public ``create_*`` factories that return a single Task."""
    return {
        name
        for name in l2co_tasks.__all__
        if name.startswith("create_") and name not in _NON_TASK_FACTORIES
    }


def test_every_task_factory_has_a_case():
    """Each public task factory is exercised by at least one TaskCase."""
    covered = {case.factory for case in TASK_CASES}
    missing = _public_task_factories() - covered
    assert not missing, (
        "Public task factories without a contract TaskCase: "
        f"{sorted(missing)}. Add an entry to tests/task_cases.py."
    )


def test_cases_reference_real_factories():
    """Every TaskCase points at a factory that is actually public."""
    public = _public_task_factories()
    for case in TASK_CASES:
        assert case.factory in public, (
            f"TaskCase {case.id!r} references unknown factory {case.factory!r}"
        )
