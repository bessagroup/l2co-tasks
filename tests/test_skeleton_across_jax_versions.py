"""A task file loads under a JAX other than the one that wrote it.

A ``.eqx`` header pickles the model skeleton, whose leaves are
``jax.ShapeDtypeStruct``s, and JAX changes which fields that class
pickles between releases: a file written under JAX 0.9 sets ``vma``,
which JAX 0.11 no longer has, and failed to load there with
``AttributeError``. The test reproduces that without a second JAX, by
renaming one field in the pickled skeleton to a name this JAX lacks.
"""

from __future__ import annotations

import json

import cloudpickle
import numpy as np
import pytest

from l2co_tasks import Task

# SHORT_BINUNICODE of "weak_type" (9 characters), a field every recent
# ShapeDtypeStruct pickles, and the same-length name this JAX lacks.
_FIELD = b"\x8c\tweak_type"
_FOREIGN_FIELD = b"\x8c\tweak_tyqe"


def test_a_skeleton_pickled_by_another_jax_still_loads(simple_task, eqx_path):
    path = Task.save(simple_task, eqx_path)
    with open(path, "rb") as f:
        header = json.loads(f.readline().decode())
        leaves = f.read()

    skeleton = bytes.fromhex(header["model_shape_hex"])
    assert _FIELD in skeleton
    foreign = skeleton.replace(_FIELD, _FOREIGN_FIELD)
    # Unpickled straight into this JAX's class, it fails -- as a JAX 0.9
    # skeleton does under JAX 0.11.
    with pytest.raises(AttributeError):
        cloudpickle.loads(foreign)

    header["model_shape_hex"] = foreign.hex()
    with open(path, "wb") as f:
        f.write((json.dumps(header) + "\n").encode())
        f.write(leaves)

    loaded = Task.load(path)
    assert loaded.hash == simple_task.hash
    np.testing.assert_array_equal(loaded.model, simple_task.model)
