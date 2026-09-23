# IGAOS — Indigenous GPU-Accelerated Optimization Solver

A linear, mixed-integer and convex quadratic programming solver built from the
mathematical foundation up, for SIH 2026 problem statement **26119** (MRPL).
No third-party optimization library is linked at any layer: the sparse
factorizations, orderings, simplex, interior point method, cut separation,
presolve and branch-and-cut are all our own implementations.

## Run it

```bash
git clone https://github.com/trijalpgunaseelan/IGAOS.git
cd IGAOS
./start.sh
```

That is the whole setup. `start.sh` checks your toolchain, builds, runs the test
suite, solves an example, and opens **Solver Studio** — a live console at
<http://127.0.0.1:8420> where you pick a model, press Solve, and watch the
algorithm flow light up stage by stage as the solver reports it.

You need a C++17 compiler and CMake. Python 3 is used by the console. Nothing
else — no optimization library to procure, no licence to obtain. If something is
missing, the script names it and gives you the command to install it.

```bash
./start.sh --no-web     # build and test only, no browser
./start.sh --check      # just tell me whether this machine is ready
```

New to the codebase? **[CONTRIBUTING.md](CONTRIBUTING.md)** maps the repository,
explains where each algorithm lives, and states the three rules that must not be
broken.

```
                     Model  (MPS / C ABI / Python)
                              |
                    Presolve  →  Scaling
                              |
        ┌─────────────────────┼─────────────────────┐
   Revised simplex      Interior point         First order
   primal + dual        Mehrotra, aug. KKT     PDHG, matrix-free
        │                     └────── Crossover ─────┘
        └─────────────────────┼─────────────────────┘
                              |
                       Branch and cut
                   GMI · knapsack cover · MIR
                              |
              Sparse linear algebra core
   basis LU (threshold Markowitz) · KKT LDLᵀ (AMD) · SpMV
```

Every path in that diagram is implemented and tested, and the tree above now
also carries a **mixed-integer quadratic** branch — branch and bound over convex
QP relaxations, multi-threaded, in `src/miqp.cpp`.

The one component that has **not been executed** is the CUDA translation of the
first-order kernels (`cuda/pdhg_kernels.cu`), because this was developed without
a GPU. It does now **compile** — NVRTC to PTX, ptxas to SASS, for sm_75, sm_80
and sm_90, in CI, on a machine with no NVIDIA device in it. Compiling is not
running: nothing here claims a GPU speedup, and the file says so at the top.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure && cd ..
./build/igaos model.mps
```

**[RUNNING.md](RUNNING.md) is the step-by-step guide** — prerequisites, a model
to solve, every flag, the Python and C paths, and how to reproduce each measured
claim. Every command in it was executed against a clean unpack.

Requires only a C++17 compiler and CMake. OpenMP is used when present. Nothing
else is needed — there is no dependency to procure and no licence to obtain.

## Using it

**Command line**

```bash
igaos model.mps --algorithm interior --gap 1e-9 --solution out.sol
igaos model.mps --no-cuts            # measure what separation is buying you
```

Run `igaos` with no arguments and you get the console itself — the same thing
the browser shows, driven by keys:

```
  IGAOS SOLVER STUDIO   0.2.0 - PS 26119 - press ? for keys
  ----------------------------------------------------------------------------
  MODEL LIBRARY               | SOLVE PIPELINE   uc_l  1686x1008   * solving
    qsmooth_m      QP         |
  > uc_l           MILP       | Read model           DONE       1,686x1,008  672 integer
    uc_m           MILP       | Presolve             DONE       -0 rows  -14 cols
    uc_s           MILP       | Scaling              DONE       equilibrated
                              | Primal simplex       DONE       1,968 iterations
  OPTIONS                     | Dual simplex         RUNNING    running...
    [ ] cut separation   (c)  | Interior point       OTHER PATH works — the tree uses simplex
    [x] presolve         (p)  | First-order (PDHG)   OTHER PATH works — the tree uses simplex
    [x] scaling          (s)  | Crossover            OTHER PATH the tree already has a basis
    [x] crossover        (x)  | Cut separation       SKIPPED    switched off
                              | Branch and bound     RUNNING    4,548 nodes  gap 2.41%
  RESULT                      | Postsolve + cleanup  idle
    status     optimal        |
  ----------------------------------------------------------------------------
  j/k model  enter solve  c/p/s/x options  a algorithm  C cuts on-off  P four paths
  1 pipeline  2 console  3 timeline  4 chart  5 table  e stages  ? help  q quit
```

Pick a model with the arrows, toggle separation off with `c`, press Enter, and
watch the node count and gap move. `C` solves the model twice — separation on,
then off — and tabulates the two. `P` solves it four times, once per continuous
method. `2` is the execution console, one line per solver event; `3` is where
the time went; `4` plots the dual bound against the incumbent; `e` explains each
stage. Every run is bounded by a time limit (`[` and `]`) because a console with
no Stop key must not be able to hang, and Ctrl-C restores the terminal.

It refuses to start unless both stdin and stdout are terminals — a full-screen
application writing into a pipe is a corrupted log file — and names the batch
flags instead.

Given a model on the command line it stays a batch tool, and draws the solve
pipeline as it runs: which phase is working, which of the four continuous
methods this model took, which stages were switched off, and where the time went:

```
  IGAOS solve pipeline   finished
  preparation — every model
   + Read model           DONE        802x480  4,270 nonzeros  320 integer
   + Presolve             DONE        -60 rows  -10 cols
   + Scaling              DONE        equilibrated
  continuous solve — exactly one of these four runs
   + Primal simplex       DONE        1,203 iterations
   - Dual simplex         OTHER PATH  works — the tree solves its LPs with simplex
   - Interior point       OTHER PATH  works — the tree solves its LPs with simplex
   - First-order (PDHG)   OTHER PATH  works — the tree solves its LPs with simplex
   - Crossover            OTHER PATH  the tree already has a basis
  mixed integer — only when the model has integer columns
   + Cut separation       DONE        21 kept  (19 gomory, 6 cover, 7 mir)
   + Branch and bound     DONE        1 node
   + Postsolve + cleanup  DONE        basis rebuilt
```

`SKIPPED` means a stage you switched off. `OTHER PATH` means it works and this
model went another way — exactly one continuous method runs per solve.

On by default at a terminal, off whenever the output is piped, so
`igaos m.mps | grep IGAOS_RESULT` is unchanged and no escape sequence ever
reaches a log file. `--live` forces it on, `--no-live` off. It reads the same
event stream the browser console reads, through the same hook in the solver, so
the two cannot disagree about what happened.

Everything else the console offers is here too — a model library so you need no
paths, both comparisons, the convergence plot, and what each stage is:

```bash
igaos --models                      # the fourteen bundled instances
igaos -m uc_m                       # solve one by name (a unique prefix will do)
igaos -m uc_m --compare cuts        # solve twice, with separation and without
igaos -m blend_s --compare paths    # solve four times, one per continuous method
igaos -m uc_l --no-cuts --chart     # dual bound against incumbent, as it closed
igaos --explain cuts                # what a stage actually does
```

`--compare cuts` on `uc_m` is the demonstration worth showing: **1 node against
51, at the same objective to twelve digits.** `--compare paths` runs primal,
dual, interior point and first order over one model and prints the four answers
side by side — four algorithms sharing no code path, agreeing or not. Both run
in process; the browser console shells out to this binary, and this does not.

**C** — a stable ABI, so a program linked against one version keeps working
against later ones:

```c
#include "igaos/igaos.h"

igaos_model* m = igaos_create();
igaos_set_sense(m, IGAOS_MAXIMIZE);
int x = igaos_add_column(m, 0.0, IGAOS_INFINITY, 3.0, IGAOS_CONTINUOUS, "x");
int r = igaos_add_row(m, -IGAOS_INFINITY, 4.0, "c0");
igaos_set_element(m, r, x, 1.0);
igaos_solve(m);
double obj = igaos_get_objective(m);
igaos_destroy(m);
```

**Python** — ctypes over that ABI, so installing it needs no compiler and no
numpy. Point it at the package and the built library first
(`export PYTHONPATH=$PWD/python IGAOS_LIBRARY=$PWD/build/libigaos.so`), or
`pip install ./python`:

```python
from igaos import Model, INF

m = Model(sense="maximize")
x = m.add_variable(0, INF, cost=3.0)
y = m.add_variable(0, INF, cost=5.0)
m.add_constraint({x: 1.0}, upper=4.0)
m.add_constraint({y: 2.0}, upper=12.0)
m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)
print(m.solve().objective)          # 36.0
```

## What is implemented

**Sparse linear algebra**
- CSC/CSR sparse matrices, pattern-tracked sparse vectors
- Basis LU by right-looking Gaussian elimination with **threshold Markowitz
  pivoting** — pivots minimise `(rᵢ−1)(cⱼ−1)` subject to `|aᵢⱼ| ≥ τ·maxᵢ|aᵢⱼ|`,
  which bounds fill and element growth together
- Product-form (eta) basis updates with accuracy-triggered refactorization
- **AMD** approximate minimum degree ordering, elimination tree, column counts
- **Quasi-definite LDLᵀ** with dynamic regularization — exists for *every*
  symmetric permutation, so ordering is chosen purely for fill

**Simplex**
- Bounded-variable revised simplex on the computational form `[A −I][x;s] = 0`
- Composite phase 1 minimising primal infeasibility — no artificial variables
- **Harris two-pass ratio test** with bound flipping
- **Dual steepest edge** pricing with the Forrest–Goldfarb update
- Cost shifting for dual phase 1; bound perturbation as the anti-stalling device
- Hyper-sparse pricing rows: the tableau row is built by walking only the
  support of `B⁻ᵀeₚ` through the row-wise copy of `A`

**Interior point** (`src/ipm.cpp`)
- Mehrotra predictor–corrector on the **augmented KKT system**, not the normal
  equations — `A(Q+D)⁻¹Aᵀ` goes dense whenever `Q` is not diagonal, so the
  normal equations cannot serve QP at all
- Symbolic analysis once, numeric refactorization per iteration: the pattern
  never changes because the barrier terms only touch the diagonal
- Iterative refinement on every KKT solve, paying back the regularization
- Handles every bound pattern — both, lower only, upper only, free, fixed — and
  every row type through the same logical-variable formulation

**Crossover** (`src/crossover.cpp`)
- Basis identification by interiorness ranking, then a simplex clean-up, so the
  interior point and first-order paths hand back a genuine basic solution that
  branch and cut can warm start from

**First-order / GPU path** (`src/pdhg.cpp`, `cuda/pdhg_kernels.cu`)
- Restarted primal–dual hybrid gradient with adaptive restarts and primal
  weight adaptation; step sizes from a power-iteration estimate of `‖A‖₂`
- The row-box proximal operator in closed form via Moreau's identity, so one
  iteration is exactly one `Ax`, one `Aᵀy` and elementwise work — no
  factorization, no elimination tree, no sequential dependency chain
- **This is why the GPU claim is about this path specifically.** Sparse LU and
  sparse Cholesky are dominated by a long, irregular dependency chain through
  the elimination tree; SpMV is bandwidth-bound and perfectly parallel
- An explicit row-major copy of `A` is kept so that *both* products run on the
  parallel reduction kernel — the same trade the CUDA kernels make
- The CUDA kernels **compile to SASS** for sm_75, sm_80 and sm_90
  (`tools/cuda_compile_check.py`, no GPU required)
- Their **arithmetic is executed and checked**, also without a GPU:
  `cuda/emulate.hpp` defines the CUDA built-ins in ordinary C++ — including a
  fixed-point emulation of `__shfl_down_sync`, so the warp reduction really is
  executed lane by lane — and `tools/cuda_emulate.cpp` runs the kernel bodies
  themselves against `src/pdhg.cpp` and against the simplex. It runs under
  `ctest` as `cuda_kernels`, and the first time it ran it found a defect: the
  device `dualStep` was missing the exact-zero branch its CPU counterpart has,
  and left sign-unstable rounding noise on free rows
- They have never been *run on a device*, and **no GPU speedup is claimed
  anywhere**. What still needs real hardware: any timing number, the launch
  configuration, memory coalescing, and the host launcher

**Cut separation** (`src/cuts.cpp`)
- **Gomory mixed-integer** cuts from fractional tableau rows — note
  *mixed-integer*, not *fractional*: the fractional Gomory cut is invalid when
  any variable in the row is continuous
- **Knapsack cover** cuts with the standard extension
- **Complemented MIR** with bound substitution and a family of divisors
- Pool with normalization, duplicate suppression, density/dynamism/efficacy
  screening, a right-hand-side safety relaxation, and slack-cut purging
- Separation is at the root only, and that is a *validity* decision rather than
  a shortcut: a cut read off a node's locally tightened bounds is sound only
  inside that subtree

**Mixed-integer quadratic** (`src/miqp.cpp`)
- Branch and bound where each node relaxation is the convex QP, solved by the
  interior point method — not the LP relaxation. Branching on the linear part
  and evaluating the quadratic objective at the answer returns a feasible point
  with a wrong `optimal` label; on a two-variable model it returns +20 where the
  optimum is −25
- No cutting planes on this path, and that is a correctness decision: Gomory
  cuts are read off a simplex tableau, the interior point method does not produce
  one, and separating from the wrong tableau cuts off your own optimum
- Non-convex objectives are refused rather than answered — a relaxation of a
  non-convex QP is not a bound
- **Multi-threaded**: workers share one node pool and one incumbent, so a bound
  found by one thread prunes work on every other (`--threads N`)

**Mixed integer**
- Branch and cut with dual-simplex warm starts at every node
- **Multi-threaded tree** (`--threads N`, default all cores): each worker owns
  its own copy of the cut-augmented relaxation, so a warm start on one thread
  cannot disturb another; the node queue, the pseudocosts and the incumbent are
  the only shared state. 1.68x–2.03x on the two cores this was measured on,
  zero ThreadSanitizer reports, identical objectives at 1 and 4 threads on all
  13 models checked — `bench/results_parallel.txt`. Node counts are *not*
  reproducible above one thread and the file says so; `--threads 1` is the
  deterministic mode
- Pseudocost branching learned from measured degradation, with a reliability phase
- Hybrid plunge / best-bound node selection
- Rounding and fractional diving heuristics with one-level backtracking
- A polish step that fixes the integers and re-solves the continuous part on the
  *original* model, so the continuous values reported are exact

**Nonlinear: global** (`src/global.cpp`)
- **Nonconvex QCQP and bilinear MINLP, solved to proven GLOBAL optimality.**
  Spatial branch and bound over **McCormick envelopes** — the convex hull of
  each product `x_i·x_j` over the current box, which makes the relaxation linear
  and therefore a *valid bound*, which is the thing a nonconvex tree needs and a
  local method cannot give
- Branching splits the range of a **continuous** variable: nothing is
  fractional, what is violated is `w ≠ x_i·x_j`, and the envelope tightens
  quadratically as the box narrows
- Interval propagation at every node, optimality-based bound tightening at the
  root, and an alternating fix-and-resolve heuristic for incumbents
- The relaxation of a node is an LP — or a **MILP** when the model has integer
  variables, so integer decisions and continuous nonconvexity are searched by
  one tree, on the branch-and-cut code that was already here
- This is the **pooling problem**: the three Haverly instances are solved to
  their published optima 400 / 600 / 750 (`ctest -R pooling`)

**Nonlinear: general** (`src/expr.cpp`, `src/nlp.cpp`)
- Expression tape over `+ − × ÷`, powers, `exp`, `log`, `sqrt`, `sin`, `cos`,
  with **exact** derivatives — reverse mode for gradients, forward-over-reverse
  for Hessian-vector products. No finite differences anywhere
- Primal-dual interior point with a **filter line search** and **inertia
  correction**, on the same augmented KKT form and the same `src/ldl.cpp`
  factorization the LP and QP paths use
- **12 / 12 Hock–Schittkowski** problems match their published optima
- **Local solutions only**, and every result says so. On Haverly 1 from 40
  random starts this method returns a converged, zero-violation, *wrong* answer
  ten times — which is the argument for the global path above

**Presolve / postsolve**
- Empty and fixed columns, empty rows, singleton rows, forcing and redundant
  rows, constraint propagation with integer rounding
- Full postsolve reconstructing primal values **and a complete basis**, followed
  by a cleanup solve of the original problem that makes the reported duals exact
  and independently checks that the reduction chain preserved optimality

**Interfaces**
- MPS reader (fixed and free form, `RANGES`, `BOUNDS`, integer markers,
  `OBJSENSE`, `QUADOBJ`), MPS writer, solution writer
- Command line driver, stable C ABI, dependency-free Python bindings

## Verification

Correctness is checked several ways, because the failure mode that matters most —
an invalid cut — neither crashes nor fails a small test. It silently returns a
wrong optimum.

0. **Security.** The reader-hardening, sanitizer, fuzzing, static-analysis and
   binary-mitigation controls are described in [`SECURITY.md`](SECURITY.md),
   together with the gaps they do not close. The threat model is in
   [`docs/THREAT-MODEL.md`](docs/THREAT-MODEL.md).
1. **Unit and property tests** (`tests/test_all.cpp`, 108 assertions): kernels
   against their defining identities, hand-computed optima, MPS round trips
   including objective sense, presolve equivalence, interior point against the
   simplex, crossover returning exactly `m` basic variables, PDHG against the
   simplex, convex QP by duality gap, equality-constrained models on every path,
   and the parallel tree against the serial one — same objective at 1 and at 4
   threads, on a model with a tree worth walking and on an infeasible one.
2. **C ABI tests** (`tests/test_capi.c`, 42 assertions) compiled as C and linked
   against the shared library, plus 20 Python binding tests.
2b. **CUDA kernel arithmetic** (`tools/cuda_emulate.cpp`, 8 assertions, runs
   under `ctest` as `cuda_kernels`): the kernel bodies in `cuda/pdhg_kernels.cu`
   executed on the CPU — including the warp reduction, lane by lane — against
   `SparseMatrix::multiply`, against `src/pdhg.cpp`, and, assembled into a whole
   PDHG loop, against the simplex. No GPU and no CUDA toolkit required. It found
   a real defect the first time it ran.
2c. **Published optima** (`ctest -R "nlp|pooling"`): the three Haverly pooling
   problems and twelve Hock–Schittkowski problems, each against a value
   published decades before this project existed. These are the only checks here
   that no amount of internal consistency could fake.
2d. **MPS reader fuzzing** (`tools/fuzz_mps.cpp`, runs under `ctest` as
   `mps_fuzz`): `readMps()` is the only function here that consumes bytes nobody
   in this project produced, so it gets its own harness. 34 targeted cases plus
   20,000 mutated files per run, and the check is not only "did it crash" —
   sanitizers cover that. Every file that **parses successfully** has its model
   checked against the invariants everything downstream assumes: no NaN in the
   objective, bounds, matrix or quadratic terms; `colPtr` monotone and of length
   `ncol+1`; every row index in range. That is the failure that matters. A crash
   is loud; a file that parses into a quietly wrong model produces a confident
   number indistinguishable from a real one. It found defect 26 — `std::stod`
   accepts `nan`, and the NaN reached the constraint matrix, and the solver
   reported `optimal`.
3. **Cut validity fuzzing** (`tools/fuzz_cuts.cpp`): random mixed-integer models
   solved twice, with and without separation, plus a verification-point hook
   that hands the separators a proven optimum and asserts that no cut ever
   excludes it. **578 optimal pairs per separator combination, zero
   disagreements**, on models that include equality, ranged, one-sided and
   free rows.
4. **Checkable certificates** (`tools/verify_certificate.py`). The three above
   are all things this project does to itself. This one is not: the solver
   writes out a certificate, and a separate program decides whether to believe
   it — with its own MPS reader, in exact rational arithmetic, without a single
   floating-point operation.

   It rests on one theorem. For `min c'x` subject to `rl <= Ax <= ru`,
   `cl <= x <= cu`, write `z = c - A'y`. Then for **any** vector `y` at all,

   ```
   L(y) = SUM_j ( z_j >= 0 ? z_j*cl_j : z_j*cu_j )
        + SUM_i ( y_i >= 0 ? y_i*rl_i : y_i*ru_i )
   ```

   is a lower bound on the optimum. "For any y" is what makes it worth doing:
   the checker never reproduces the solver's reasoning, agrees with its sign
   conventions, or trusts its basis. It is handed a vector and verifies an
   inequality. The same object proves infeasibility with `c` taken as zero.

   ```bash
   ./build/igaos model.mps --certificate model.cert
   python3 tools/verify_certificate.py model.mps model.cert
   ```

   Over the 55 smallest Netlib instances: **53 certified** — one proven optimal
   in exact arithmetic, 52 with a rigorous lower bound — and zero rejected.
   Altering a single multiplier cannot forge a better bound, only a useless one,
   and the checker says so rather than passing it.

   Two paths cannot yet be certified and report that rather than implying a
   proof: the branch-and-bound tree emits no dual multipliers, and infeasibility
   detected inside presolve produces no Farkas ray.

## Benchmarks

### The standard libraries

Generated instances prove a solver handles the shapes you thought of. The public
collections prove it handles the ones you did not. All of this is reproducible:
`bench/fetch_benchmarks.sh` downloads the instances, which are not redistributed
here.

**Netlib LP** — the reference correctness test for a linear programming code
since 1985. 114 feasible models, many deliberately degenerate or badly scaled.
Reference objectives from Gurobi 10 at 1e-8, cross-checked against the values
Netlib itself publishes.

```bash
./bench/fetch_benchmarks.sh
IGAOS_BIN=./build/igaos python3 bench/netlib.py \
    --dir benchmarks/netlib --ref bench/netlib_reference.csv --time-limit 300
```

| | result |
|---|---|
| solved to proven optimality | **108 / 114** |
| objective matches the reference | **107 / 114** |
| objective mismatches | **1** — `pilot87`, at 1.7e-6 relative |
| hit the 300 s limit | 6 — `d6cube`, `dfl001`, `ken-18`, `pds-20`, `qap12`, `qap15` |
| worst primal infeasibility | 3.5e-6 (`cre-d`) |
| total solve time | 480 s for the 108 |

`pilot87` is the hardest instance in the collection and published reference
values for it disagree in the sixth digit: Netlib's own table gives
3.0171072827e+02, the Gurobi reference gives 3.0171034733e+02, and this solver
returns 3.017108552e+02 — between the two. It is reported as a mismatch rather
than argued away.

One reference value needed correcting, in the open, in `bench/netlib.py`: the
mirrored CSV gives `forplan` as −1163.915769, while
[netlib.org/lp/data/readme](https://www.netlib.org/lp/data/readme) gives
−6.6421873953e+02 and HiGHS, handed the same file, returns −664.2189613. Two
independent sources against one.

**Netlib infeasible** — 28 models with no feasible point. The failure that
matters in production is not a slow answer, it is a confident wrong one: a
solver that returns `optimal` here has handed a planner a schedule that cannot
be run.

```bash
IGAOS_BIN=./build/igaos python3 bench/infeasible.py --dir benchmarks/netlib_infeasible
```

**27 / 28 correctly reported infeasible.** The exception is `cplex2`, which is
infeasible by roughly 1e-9 — HiGHS's own minimum-violation solve puts it at
9.99e-10, below any working feasibility tolerance. IGAOS returns `optimal` with
a point violating one row by 8.1e-8 and says so in the primal-infeasibility
field. Finding this instance is what surfaced the phase-one bug described in the
phase-2 report.

**Convex quadratic programming** — the third class the problem statement names,
checked against **OSQP**, an independent solver using a completely different
method (operator splitting against an interior point method). Eighteen
instances from three industrially shaped families: refinery blend property
tracking (`Q = 2SᵀS`, positive semidefinite and rank deficient), production
smoothing (a singular tridiagonal Hessian), and factor-model risk.

```bash
python3 bench/generate_qp.py benchmarks/qp --seeds 3
IGAOS_BIN=./build/igaos python3 bench/qp_bench.py --dir benchmarks/qp
```

| | result |
|---|---|
| agreeing with OSQP | **18 / 18** |
| worst relative difference | 8.5e-08 (`qrisk_s_1`) |
| worst constraint violation, IGAOS | 2.4e-09 |
| worst constraint violation, OSQP | 1.0e-09 |

Two of the three families are deliberately **singular** — a solver that assumes a
positive definite Hessian gets them wrong. The worst difference, 8.5e-08, is at
the edge of what OSQP itself resolves: it is a first-order method run at 1e-9
with polishing, so a disagreement in the eighth digit is as likely to be the
reference as the solver under test. Both objectives are printed by the harness
so the reader can judge rather than take the verdict.

**Maros–Mészáros** — the public convex QP collection, and the one Mittelmann's
convex QP benchmark is built on. 138 instances, 20 s each, OSQP as the reference.
Generated instances prove a solver handles the shapes you thought of; this set
proves it handles the ones you did not, and it is the harshest number in this
README.

```bash
./bench/fetch_benchmarks.sh
IGAOS_BIN=./build/igaos python3 bench/qp_bench.py --dir benchmarks/qp_maros --time-limit 20
```

| | result |
|---|---|
| solved to optimality | **109 / 138** |
| still failing | 22 `numerical_error`, 5 time limit, 2 iteration limit |
| where both solved, agreeing to 1e-6 | **65 / 69** |
| solved by IGAOS that OSQP could not, in 20 s | 40 |
| solved by OSQP that IGAOS could not | 15 |

That 109 was **102** two days ago. The seven that moved exposed a real defect:
the scaling is an equilibration of `A` alone, `Q` is transformed consistently
afterwards but never gets a vote in choosing it, and a scaling that flattens `A`
can leave `Q` spanning orders of magnitude — on a KKT matrix that contains both.
`QBANDM` failed scaled and solved to seven digits unscaled. The fix is a
fallback, the principled fix (equilibrating `[A; Q]` together) is *not* written,
and 22 instances still fail.
[bench/results_maros.txt](bench/results_maros.txt) has the full lists and says
which is which.

**MIPLIB-family mixed integer** — nine instances, compared against HiGHS
branch-and-cut run on the identical file.

| | result |
|---|---|
| agreeing with HiGHS | **7 / 9** |
| worst relative difference | 8.1e-14 (`gesa2`) |
| the other two | `gt2` and `bell5` hit the 300 s limit |

`gt2` is worth being precise about: IGAOS finds the value HiGHS proves optimal,
21166, but does not prove it within the limit. `bell5` finishes about 1% above —
and *where* it finishes moves between runs now that the tree is parallel: 1.10%,
0.45% and 0.25% have all been recorded at the same limit. All three are honest
readings of a search that did not finish, and all are labelled `feasible` rather
than `optimal`. Both instances are open weaknesses in the branch-and-cut search,
not in the answers.

### Against CPLEX and Gurobi

PS 26119 names Xpress, CPLEX and Gurobi. Two of the three ship a free,
size-limited licence of the real product, and this is a comparison against those
— not against a re-implementation.

```bash
./bench/fetch_benchmarks.sh                 # creates benchmarks/standard
pip install cplex gurobipy
python3 bench/commercial.py --dir benchmarks/standard --time-limit 60
```

| | result |
|---|---|
| agreeing with CPLEX | **13 / 13 comparable** |
| agreeing with Gurobi | **21 / 22 comparable** |
| disagreements | 1 — `bell5`, unfinished and labelled `feasible` |

The caveat is the whole point and it is stated first in
[bench/results_commercial.txt](bench/results_commercial.txt): CPLEX Community
Edition caps at 1000×1000 and the restricted Gurobi licence at 2000×2000, so
every instance is small — exactly where a mature commercial solver has least
room to show its advantage. **This is a correctness result, not a performance
one.** On the mixed-integer instances IGAOS is one to three orders of magnitude
slower, `gt2` takes 60 s here against CPLEX's 0.01 s, and nothing in this
repository narrows that. Xpress has never been run at all.

### Refinery scheduling and process optimization

The two application areas PS 26119 names that had no instance family, and now
do — a discrete-time scheduling MILP with mode changeovers, minimum run lengths
and tank inventories, and the steady-state RTO convex QP.
[bench/results_scope.txt](bench/results_scope.txt) has the tables.

| | result |
|---|---|
| refinery scheduling, objective agrees with HiGHS | **9 / 9** |
| … proven optimal by IGAOS | 8 / 9 (`sched_l_2` finds the optimum, cannot close it in 900 s) |
| refinery scheduling vs CPLEX and Gurobi | **6 / 6 and 6 / 6** on the sizes their licences allow |
| process optimization, agreeing with OSQP | **9 / 9**, worst relative difference 3.2e-08 |

### Nonlinear — pooling, and the general NLP

```bash
cd build && ctest -R "nlp|pooling" --output-on-failure
```

| | result |
|---|---|
| Haverly pooling vs published global optima | **3 / 3** exact (400, 600, 750) |
| Hock–Schittkowski vs published optima | **12 / 12** |
| local solutions beating the proven global optimum, 12 instances × 40 starts | **0** — as it must be |
| multistart reaching the global optimum | **12 / 12** instances |
| exact derivatives vs central differences | 1.3e-09, which *is* the differencing error |

The number that matters is not in that table. On `haverly1` the local method
reaches the true 400 in 30 starts out of 40 and lands on **100** in the other
ten — a 75% error, reported as converged and optimal with zero constraint
violation. Nothing about those runs looks like a failure; they are KKT points,
and a local method has no way to know. That is what the global solver is for.

And the honest other half: on the nine *generated* pooling instances every start
reaches the same answer, so the global solver bought nothing there. They have
one local optimum each — a fair test that it is correct, and no test of whether
it is necessary. [bench/results_nlp.txt](bench/results_nlp.txt) says both.

### Multi-core

```bash
for t in 1 2 4; do ./build/igaos demo/models/uc_l.mps -q --no-live --threads $t; done
```

| | result |
|---|---|
| speedup at 2 threads, eight models | **1.62× – 1.97×**, geometric mean 1.81× |
| identical objectives at 1 vs 4 threads | **16 / 16 models** |
| ThreadSanitizer data races | **0** over six models at 4 threads |

Measured on a **two-core** machine, which is why the 4-thread column adds
nothing; that limit is stated rather than extrapolated past. Node counts are not
reproducible above one thread — `--threads 1` is the deterministic mode — and
[bench/results_parallel.txt](bench/results_parallel.txt) says so along with the
ceiling: branch and bound does not scale linearly and a model that closes at the
root gains nothing.

### Generated industrial instances

`bench/generate.py` builds seven families of industrial instances — refinery
crude blending, **refinery scheduling** (unit run-modes over time slots, with
changeover binaries, minimum run lengths and tank inventories), multi-period
production planning, two-echelon supply chain, degenerate transportation,
ill-conditioned blending (coefficients spanning 10⁸) and unit commitment;
`bench/generate_qp.py` adds a fourth convex-QP family, **process optimization**
— the steady-state RTO layer, moving each unit's operating point along its
measured gain matrix. Those two are the scope areas PS 26119 names that had no
instance family until now; both are measured against outside solvers in
[bench/results_scope.txt](bench/results_scope.txt). `bench/run_bench.py` runs a
head-to-head against **HiGHS**.

```bash
IGAOS_BIN=./build/igaos python3 bench/run_bench.py --seeds 3 --time-limit 120
```

48 instances, 3 seeds per family, both engines on identical hardware:

| | result |
|---|---|
| LP instances agreeing with HiGHS | **39 / 39** |
| worst relative objective difference | **3.4e-12** |
| worst primal infeasibility | **4.2e-11** |
| worst integer infeasibility | **0.0** |
| LP solve time vs HiGHS (geometric mean) | **1.13×** |
| LP instances faster than HiGHS | 18 / 39 |
| best single speedup | **15.3×** |
| MILP instances proven optimal | **8 / 9** |
| largest instance solved | 2,196 rows × 11,314 cols, 45,224 nonzeros |

The agreement figures are properties of the solver and reproduce exactly
anywhere. The time ratios are properties of the machine: the run above was on an
8-core x86-64 box, and re-running the identical command on a 2-core container
gives 1.19× geometric mean and a 12.5× best rather than 1.13× and 15.3×. Both
runs put 18 of the 39 LPs ahead of HiGHS and both report 3.4e-12 as the worst
objective difference.

### What cut separation bought

The same nine mixed-integer instances, run twice on the same machine, once with
`--no-cuts` and once with separation on (`bench/cut_effect.py`):

| instance | without cuts | with cuts | cuts kept | root gap closed |
|---|---|---|---|---|
| uc_s_0 | optimal, 67 nodes | optimal, 3 nodes | 31 | 99.7% |
| uc_s_1 | optimal, 139 nodes | optimal, 1 node | 41 | 100% |
| uc_s_2 | optimal, 33 nodes | optimal, 1 node | 27 | 100% |
| uc_m_0 | optimal, 51 nodes | optimal, 1 node | 21 | 100% |
| uc_m_1 | optimal, 8,467 nodes, 3.65 s | optimal, 17 nodes, **0.15 s** | 67 | 100% |
| uc_m_2 | optimal, 2,603 nodes, 1.21 s | optimal, 1 node, **0.06 s** | 74 | 100% |
| uc_l_0 | feasible, **2.00%** gap | feasible, **0.98%** gap | 73 | 14.0% |
| uc_l_1 | feasible, **2.31%** gap | **optimal**, 30.0 s | 91 | 49.1% |
| uc_l_2 | optimal, 36,641 nodes, 69.3 s | optimal, **1 node, 0.21 s** | 40 | 100% |
| | **7 / 9 proven** | **8 / 9 proven** | | |

`uc_l_2` is the clearest case: 36,641 nodes and 69 seconds become one node and
0.21 seconds. `uc_l_0` is the honest remainder — separation halves its gap but
does not close it, and it is still running when the time limit expires.

Re-run on different hardware, every node count and every cut count in this table
comes back identical; only the seconds move (`uc_l_2` took 83.8 s against 0.29 s
on the slower machine, the same ratio).

## Layout

```
include/igaos/   common, sparse, model, mps, lu, ldl, simplex, presolve,
                 cuts, ipm, crossover, pdhg, expr (AD + NLP), solver,
                 igaos.h (C ABI)
src/             implementations
apps/            command line driver
cuda/            pdhg_kernels.cu, and emulate.hpp which runs it on a CPU
python/          ctypes bindings, packaging, 20 tests
tests/           108 solver assertions, 42 C ABI assertions
tools/           certificate checker, cut-validity fuzzer, MPS-reader fuzzer,
                 cuda_emulate.cpp, path comparison harnesses, pin_actions.sh,
                 make_sbom.py
bench/           instance generators, the head-to-heads against HiGHS, OSQP,
                 CPLEX and Gurobi, and every recorded result file
demo/            Solver Studio: a live local console (stdlib Python) + 17 models
docs/            algorithms.md, api.md, PHASE2_REPORT.md (26 defects, in full),
                 THREAT-MODEL.md, EXTENSION-NLP-MINLP.md (what is NOT built)
SECURITY.md      attack surface, controls, and the open gaps stated plainly
sbom.json        CycloneDX: components is empty, and that emptiness is the claim
CONTRIBUTING.md  how to change it, and the three rules that are not negotiable
RUNNING.md       step-by-step run guide
.github/         CI, nine jobs: three platforms, sanitizers, two fuzzers,
                 warnings-as-errors, CodeQL, binary mitigations, supply chain
```

## Not yet built

Stated plainly, because a solver that overstates its coverage is worse than one
that is narrow and honest:

- The **spatial (global) tree is serial**, and quadratically constrained models
  get **no presolve and no scaling** — neither transforms the row quadratics, and
  a reduction that dropped them would be a silent wrong answer.
- **No file format for a general NLP.** `QCMATRIX` in MPS carries quadratic
  constraints, so the whole pooling class round-trips through a file; an `exp`
  or `log` constraint has to be built through the C++ API.
- **No convexity detection**, so a convex MINLP is not recognised as one and
  gets a weaker guarantee than it deserves; and the general-MINLP path reports
  `Feasible`, never `Optimal`, for exactly that reason.
- The CUDA kernels have never **run on a device**, so there is no measured GPU
  speedup and none is claimed. They compile (NVRTC to PTX, ptxas to SASS, for
  sm_75, sm_80 and sm_90), and their arithmetic is now executed and checked on
  the CPU by `tools/cuda_emulate.cpp` — which found and fixed a real defect in
  `dualStep`. What still needs hardware: timing, launch configuration, memory
  coalescing, and the host launcher at the bottom of the `.cu`.
- Cut separation runs at the root only; local (subtree-valid) cuts are not
  generated.
- No cut aggregation: MIR is applied to single rows, not to aggregations of
  several, which is where much of its strength lives in mature codes.
- No conflict analysis, no symmetry detection, no restarts in the tree.
- Infeasibility and unboundedness are certified by the simplex; the interior
  point and first-order paths fall back to it rather than producing their own
  certificates.
- Mixed-integer *optimality* is not certifiable end to end. A MIP certificate
  now proves an **interval**: the incumbent above (exactly feasible, exactly
  integral) and the root LP relaxation's multipliers below, giving an
  unconditional lower bound the checker verifies in exact arithmetic. What it
  still cannot prove is that no better integer point exists — that needs the
  branch-and-bound tree, which the format does not carry.

## Licence

Copyright is retained by the developing team, with a perpetual, royalty-free
licence granted to Mangalore Refinery and Petrochemicals Limited, ONGC, the
Ministry of Education and AICTE — the arrangement the Ministry's guidelines for
Smart India Hackathon winning projects describe. See [`LICENSE`](LICENSE), which
also carries the provenance audit and acknowledges the third-party software used
for validation only.
