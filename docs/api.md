# API Reference

This section contains the API reference for the `l2co_tasks` library. It is organized around the central [`Task`](#l2co_tasks.Task) object, followed by one section per task family (each exposing a `create_<name>_task` factory), and finally the helpers for using tasks inside `f3dasm` pipelines.

## The `Task` object

The single object every factory returns: a serializable bundle of a model, a loss function, an optional dataset, and metadata. See [Create your own task](create_task.ipynb) for a guided walkthrough.

::: l2co_tasks.Task

::: l2co_tasks.DatasetDict

## Constraints

Equality, inequality and box constraints a task declares on its
parameters, passed to `Task` as `constraints=[...]`. Every constraint is
called as `c(model)` and returns raw values where `<= 0` means
satisfied (`= 0` for an equality). Enforcement belongs to `l2co`,
which forwards a box to the optimizers and refuses a task carrying an
inequality or equality, since nothing downstream can enforce those yet.

::: l2co_tasks.Constraint

::: l2co_tasks.Inequality

::: l2co_tasks.Equality

::: l2co_tasks.Box

## Host objectives

A task whose objective is computed outside JAX, such as compiled Fortran
or a simulator, wraps it with `host_loss` into an ordinary loss that
every optimizer can trace, vectorize and differentiate (ADR 0003). The
suite's adapter supplies an *opener*: a picklable, hashable,
zero-argument callable returning a `HostObjective`. The objective is
opened once per process, the first time the loss is traced.

::: l2co_tasks.host_loss

::: l2co_tasks.HostObjective

::: l2co_tasks.HostLoss

## CUTEst problems

Unconstrained problems from CUTEst, the standard collection of
nonlinear-optimization test problems, evaluated in compiled Fortran
through pycutest behind a host loss (ADR 0004). Needs the optional
`[cutest]` extra and a CUTEst installation, with `CUTEST`, `SIFDECODE`
and `MASTSIF` set. Unlike the other task families, a CUTEst task's model
is the problem's own starting point `x0`, in the problem's own
coordinates. `global_min` comes from a committed table; until a
(problem, size) has a row there, pass `global_min` explicitly.

::: l2co_tasks.create_cutest_task

## Estimating the global minimum

Benchmark a task's best achievable loss with a short, seeded, multi-restart Adam search. This is how the supervised-learning and meta factories set their *empirical* `global_min`; it is also a standalone helper for custom tasks. A task with a `Box` is searched inside it.

::: l2co_tasks.estimate_global_min

## BBOB benchmark functions

::: l2co_tasks.create_bbob_task

## BBOB-noisy benchmark functions

The inherently stochastic BBOB-noisy suite (f101–f130): eight base
landscapes disturbed by the Gaussian, uniform or Cauchy noise model at
moderate or severe severity. Every loss takes a random key
(`pass_rng=True`). Requires a `bbob-jax` release newer than 1.8.0.

::: l2co_tasks.create_bbob_noisy_task

## Embedded BBOB benchmark functions

BBOB functions composed with a random low-rank affine embedding: an
`intrinsic_dim`-dimensional function hidden inside the
`[0, 1]^ambient_dim` search space. Gradients are confined to a
low-dimensional subspace and the Hessian spectrum is
bulk-plus-outliers, mimicking the measured geometry of neural-network
loss landscapes without using a neural network.

::: l2co_tasks.create_embedded_bbob_task

## CEC 2005 benchmark functions

::: l2co_tasks.create_cec2005_task

::: l2co_tasks.create_cec2017_task

::: l2co_tasks.create_cec2013lsgo_task

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

## Helmholtz equation (PINN)

::: l2co_tasks.create_helmholtz_task

## Viscous Burgers equation (PINN)

::: l2co_tasks.create_viscous_burgers_task

## Inviscid Burgers equation (PINN)

::: l2co_tasks.create_inviscid_burgers_task

## Euler equations (PINN)

::: l2co_tasks.create_euler_task

## Stokes wedge flow (PINN)

::: l2co_tasks.create_stokes_task

## Stiff PK-PD ODE (PINN)

::: l2co_tasks.create_pkpd_task

## `f3dasm` integration

Helpers for enumerating task suites as `f3dasm.Block` samplers and for
moving tasks in and out of an `f3dasm.ExperimentData`.

::: l2co_tasks.CEC2019Sampler

::: l2co_tasks.PDETaskSampler

::: l2co_tasks.HelmholtzTaskSampler

::: l2co_tasks.ViscousBurgersTaskSampler

::: l2co_tasks.InviscidBurgersTaskSampler

::: l2co_tasks.EulerTaskSampler

::: l2co_tasks.StokesTaskSampler

::: l2co_tasks.PKPDTaskSampler

::: l2co_tasks.create_tasks_experimentdata

::: l2co_tasks.retrieve_tasks
