"""Tests for ``retrieve_tasks`` in ``_src.continue_from_path``."""

from __future__ import annotations

import pandas as pd
import pytest
from f3dasm import ExperimentData
from f3dasm.design import Domain

from l2co_tasks import create_bbob_task, retrieve_tasks


@pytest.mark.requires_f3dasm
def test_retrieve_tasks_deduplicates():
    """Tasks with identical tags collapse to one entry via ``set(...)``.

    ``retrieve_tasks`` relies on ``Task.__hash__``/``__eq__`` being
    tag-based, so two BBOB tasks built with the same args are
    considered the same even though they are distinct Python objects.
    """
    t_same_a = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)
    t_same_b = create_bbob_task(fn_name="sphere", seed=0, dimensionality=2)
    t_other = create_bbob_task(fn_name="rastrigin", seed=0, dimensionality=2)

    d = Domain()
    d.add_parameter("task", to_disk=False)
    ed = ExperimentData(
        domain=d,
        input_data=pd.DataFrame({"task": [t_same_a, t_same_b, t_other]}),
    )

    out = retrieve_tasks(ed)
    extracted = [es.output_data["task"] for _, es in out]
    assert len(extracted) == 2
    assert {t.tag["fn_name"] for t in extracted} == {"sphere", "rastrigin"}
