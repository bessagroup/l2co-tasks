"""Smoke tests for the public ``l2co_tasks`` API."""

from l2co_tasks import Task, create_bbob_task


def test_public_api_imports():
    """The top-level package re-exports Task and the create_* factories."""
    import l2co_tasks

    for name in [
        "Task",
        "DatasetDict",
        "create_bbob_task",
        "create_cec2005_task",
        "create_pde_task",
        "create_quadratic_task",
        "create_spiral_task",
        "create_mnist1d_task",
        "create_gaussian_meta_task",
        "create_gaussian_task",
        "retrieve_tasks",
    ]:
        assert hasattr(l2co_tasks, name), f"l2co_tasks missing {name}"


def test_create_bbob_task_sphere():
    """``create_bbob_task`` returns a valid ``Task`` for a simple case."""
    task = create_bbob_task(
        fn_name="sphere",
        seed=0,
        dimensionality=2,
        noise=0.0,
    )
    assert isinstance(task, Task)
    assert task.dimensionality == 2
