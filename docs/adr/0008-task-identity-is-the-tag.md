---
status: accepted
---

# Task identity is the tag, not `global_min`

The databank keys every result by `Task.hash`. Until now that hash
digested the tag, the constraints (when present, ADR 0002) and
`global_min`. ADR 0001 put `global_min` there on purpose, and ADR 0004
and ADR 0006 repeated it for CUTEst.

That made a correction impossible to apply cleanly. Many `global_min`
values are estimates: the supervised tasks run a short search at
creation (ADR 0001), and 102 of the CUTEst table's rows were corrected
by hand after the first runs (ADR 0004, ADR 0006). Each correction gave
the task a new hash. A store written before the correction then no
longer matched the task, although the function, the start and the data
were the same. The minimum is a fact *about* the task that we learn
better over time. It is not part of what the task is.

The hash also missed things that do change the task. Spiral tasks did
not tag their seed, dataset size or batch size; the PINN tasks did not
tag their seed or their point counts; and the level of added noise was
not tagged at all. Two tasks that differ only there hashed the same, and
only a different `global_min` estimate (by luck) told them apart.

Decided in the grilling of 2026-11-06 (dataset paper re-run).

## Decision

**Identity is the tag plus the constraints.** `tag_hashable`, `hash`,
`__eq__` and `__hash__` no longer read `global_min`, for every task.
Constraints still join only when present (ADR 0002). Correcting
`global_min` (for example with `dataclasses.replace`) leaves the hash and
equality unchanged.

The `.eqx` format does not change: `global_min` is still saved in the
header and loaded as a property.

**Every factory argument that changes the task is in its tag.** The tag
must tell two different tasks apart on its own. Arguments that only set
`global_min` (`estimate_global_min`, `global_min_restarts`,
`global_min_steps`, CUTEst's `global_min=` and `table=`) stay out. Keys
added by this change (none removed or renamed):

| Factory | New tag keys |
| --- | --- |
| `create_spiral_task` | `seed`, `dataset_size`, `batch_size` |
| `create_mnist1d_task` | `dataset_size`, `batch_size` |
| `create_gaussian_meta_task` | `inner_steps`, `l2_regularization` |
| `create_pde_task` | `seed`, `xgrid_resolution`, `tgrid_resolution`, `num_ic_points`, `num_bc_points`, `num_res_points`, `x_range`, `t_range`, `hidden_size` |
| `create_helmholtz_task` | `seed`, `num_res_points`, `x_range` |
| `create_viscous_burgers_task`, `create_inviscid_burgers_task` | `seed`, `num_res_points`, `num_ic_points`, `num_bc_points` |
| `create_euler_task` | `seed`, `num_res_points`, `num_ic_points`, `num_bc_points`, `x_range`, `t_range`, `lam_ic`, `lam_bc`, `lam_f`, `lam_nu` |
| `create_stokes_task` | `seed`, `num_res_points`, `num_lid_points`, `num_wall_points` |
| `create_pkpd_task` | `seed`, `point_allocation` |
| bbob, CEC 2005, CEC 2017, CEC 2013 LSGO, embedded | `noise_level`, only when `noise > 0` |

Ranges are tagged as tuples of floats, counts as ints and loss weights
as floats, so the tag hashes the same in every process. `create_euler_task`
tags `lam_nu` in both stages, although only the viscous stage uses it, as
it already tagged `h` (used only by the inviscid stage).

`tag["noise"]` stays the bool that says whether the loss is
stochastic. A task without added noise keeps its tag exactly.

**A PINN task's cached points are named by its tag.** The PINN
factories cache their collocation points in an `.npz` file and write it
only when it is missing. The file is `<dataset_path>_<digest>.npz`, where
`<digest>` is the short sha256 of the tag, computed as `Task.hash` does
(`_io.points_cache_path`; a PINN task has no constraints, so it is the
task's hash). Two tasks with different tags never share a file, even when
their point arrays have the same shapes (a grid resolution, a domain
range, a pkpd block boundary). The seed is in the tag, so it still
separates files. `ensure_points_dataset` keeps its shape check as a
backstop. Cache files under the old `<dataset_path>_<seed>.npz` names are
simply not found, and each task regenerates its points once.

**CUTEst with an explicit `global_min=` or a provisional `table=` is the
same task** as the one built from the committed table. The SIF file's
hash in the tag (ADR 0004) still gives a corrected problem a new id.

## What this supersedes

- ADR 0001: "the value is also part of `Task` identity" and the
  consequence that re-estimating re-keys the databank.
- ADR 0004: "it is part of the task's identity" in the `global_min`
  section.
- ADR 0006: "only `global_min` is" part of a task's identity.

ADR 0002 stands: constraints are part of identity, only when present.

## Consequences

- **Every hash moves once.** Even a task without `global_min` used to
  digest a `("global_min", None)` entry. The golden hashes in
  `tests/test_task_golden_hashes.py` and
  `tests/test_tag_hash_across_processes.py` were re-pinned once, with
  the old values kept in comments.
- **Stores keyed by old hashes do not match.** The dataset re-run writes
  fresh stores. Old hash lists stay as records; nothing translates them.
- **A changed `global_min` no longer shows up as a new task.** l2co
  refuses to append to a store whose recorded f* differs from the
  task's, and l2co_experiments' create step does the same, before
  anything is written (exact float64 comparison; NaN equals NaN). A tool
  to update f* in place comes later.
- **A cached PINN point set belongs to its tag.** Rebuilding at the same
  `dataset_path` with other point arguments writes a new file next to the
  old one, instead of running a task whose tag claims new points on the
  old ones. Old `<dataset_path>_<seed>.npz` files are left in place and
  no longer read; the first build after this change regenerates the
  points once. A task's file name now changes whenever its tag does
  (including model keys such as `hidden_size`), so tasks that differ
  only in their network write identical points to separate files.
- **A changed function definition is not detected.** If a loss function's
  code changes while its tag does not, the hash stays the same. This was
  true before, and is out of scope here.

Rejected: keeping `global_min` in identity and re-keying stores at every
correction (the cost ADR 0001 accepted once, which the CUTEst corrections
made recurring).
