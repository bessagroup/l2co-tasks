# Third-party notices

`l2co-tasks` incorporates work derived from third-party projects. Their
licenses and copyright notices are reproduced below, as required.

---

## MetaBox

The **UAV** path-planning task (`_src/uav_task.py`) was **ported to JAX
from MetaBox's NumPy implementation**.

- Project: MetaBox — <https://github.com/MetaEvo/MetaBox>
- Version: `MetaEvo/MetaBox@5565a28` (v2.0.1+5)
- Source ported: `src/environment/problem/SOO/UAV/{uav_numpy.py,uav_dataset.py,utils.py}`
- License: BSD 3-Clause, © 2023 MetaEvolution Lab

The underlying benchmark is *Benchmarking global optimization techniques
for unmanned aerial vehicle path planning* (arXiv:2501.14503, 2025; data at
Zenodo 12793991). The terrain is synthesised procedurally at
task-construction time — no external data files are used or redistributed.

The **CEC 2013 LSGO** tasks (`create_cec2013lsgo_task` in
`_src/benchmark_task.py`) do **not** vendor any MetaBox code: they wrap the
LSGO suite provided by [`bbob-jax`](https://github.com/bessagroup/bbob-jax),
which carries its own third-party notices for that port.

Please cite MetaBox (NeurIPS 2023, arXiv:2311.02708; v2 NeurIPS 2025,
arXiv:2505.17745) and the original benchmarks alongside `l2co-tasks`.

```
BSD 3-Clause License

Copyright (c) 2023, MetaEvolution Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

## Provenance table

| l2co-tasks component | Derived from | Original benchmark |
|---|---|---|
| `_src/uav_task.py` | MetaBox UAV `uav_numpy.py`/`utils.py` @ 5565a28 (BSD-3) | Benchmarking global optimization for UAV path planning, arXiv:2501.14503 (2025) |
| `create_cec2013lsgo_task` | `bbob_jax.cec2013lsgo` (see bbob-jax notices) | Li, Tang, Omidvar, Yang & Qin, CEC 2013 LSGO (2013) |
