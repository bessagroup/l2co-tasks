# API Reference

This section contains the API reference for the `l2co_tasks` library. It is organized around the central [`Task`](#l2co_tasks.Task) object, followed by one section per task family (each exposing a `create_<name>_task` factory), and finally the helpers for using tasks inside `f3dasm` pipelines.

## The `Task` object

The single object every factory returns: a serializable bundle of a model, a loss function, an optional dataset, and metadata. See [Create your own task](create_task.ipynb) for a guided walkthrough.

::: l2co_tasks.Task

::: l2co_tasks.DatasetDict

## BBOB benchmark functions

::: l2co_tasks.create_bbob_task

## CEC 2005 benchmark functions

::: l2co_tasks.create_cec2005_task

## Quadratic least-squares

::: l2co_tasks.create_quadratic_task

## Spiral classification

::: l2co_tasks.create_spiral_task

## MNIST-1D classification

::: l2co_tasks.create_mnist1d_task

## Gaussian classification

::: l2co_tasks.create_gaussian_task

## Gaussian meta-learning

::: l2co_tasks.create_gaussian_meta_task

## PDE (physics-informed) tasks

::: l2co_tasks.create_pde_task

## `f3dasm` integration

Helpers for enumerating task suites as `f3dasm.Block` samplers and for
moving tasks in and out of an `f3dasm.ExperimentData`.

::: l2co_tasks.CEC2019Sampler

::: l2co_tasks.PDETaskSampler

::: l2co_tasks.create_tasks_experimentdata

::: l2co_tasks.retrieve_tasks
