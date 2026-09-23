# Running IGAOS

Every command below was executed on a clean unpack of `igaos-solver.zip`, and
the outputs shown are the real ones. Copy-paste order works start to finish.

**In a hurry — before a demo, say:**

```bash
unzip igaos-solver.zip && cd igaos && ./run_demo.sh
```

That runs §2 through §9 in order and prints each result with the section it
comes from: build, both test suites, the LP solve, the cuts-on/cuts-off
comparison, all four algorithms, the Python bindings, a C program linked against
the shared library, and both fuzzers. About a minute. Add `--bench`
for the HiGHS head-to-head as well (~8 more minutes, needs SciPy).

**With a screen behind you — the demo console:**

```bash
./run_demo.sh --web        # → http://127.0.0.1:8420
```

A local web console that drives this same binary: pick one of eight industrial
models (or paste your own MPS), choose an algorithm, and run. Three views —
a single solve, **cuts on against cuts off side by side**, and **all four
algorithms on one model**. Every number on the page comes from running the
solver as a subprocess; nothing is precomputed. Python standard library only,
no npm, no build step. §14 has the detail.

The rest of this guide is the same thing one command at a time, with the output
to expect from each.

---

## 1. What you need

| | |
|---|---|
| A C++17 compiler | macOS: `xcode-select --install` · Linux: `g++` or `clang++` |
| CMake ≥ 3.16 | macOS: `brew install cmake` · Linux: `apt install cmake` |
| Python 3.8+ | only for the bindings and the benchmark harness |
| SciPy | **optional** — only for the HiGHS head-to-head in §9 |

Nothing else. No solver to license, no numerical library to fetch, no package
registry involved in the build. The build, both test suites, the solver and the
Python bindings need **no** third-party packages at all — skip SciPy entirely
unless you want to run the head-to-head benchmark.

**If you do want SciPy, use a virtualenv.** Homebrew's Python (and Debian's) is
marked externally managed, so a plain `pip3 install scipy` refuses with
`error: externally-managed-environment` (PEP 668). That is the packaging policy
talking, not a broken machine:

```bash
python3 -m venv ~/igaos-venv
source ~/igaos-venv/bin/activate
pip install scipy
```

Everything after this point works the same inside or outside that venv.

**On macOS**, Apple Clang ships without OpenMP. CMake reports `Could NOT find
OpenMP` and the build continues single-threaded — this is expected and is not
an error. For the parallel kernels: `brew install libomp`.

---

## 2. Build and test

```bash
unzip igaos-solver.zip
cd igaos

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cd build && ctest --output-on-failure && cd ..
```

Expected:

```
    Start 1: unit
1/2 Test #1: unit .............................   Passed    0.10 sec
    Start 2: c_abi
2/2 Test #2: c_abi ............................   Passed    0.00 sec

100% tests passed, 0 tests failed out of 2
```

That is 71 solver assertions plus 42 C ABI assertions. `./build/igaos_tests`
alone prints them one by one, which is the better thing to show a judge.

The build produces:

| | |
|---|---|
| `build/igaos` | command-line solver |
| `build/libigaos.so` (`.dylib` on macOS) | shared library carrying the C ABI |
| `build/libigaos_core.a` | static library for C++ callers |
| `build/igaos_tests`, `build/igaos_capi_test` | the two suites |

---

## 3. Get a model to solve

The benchmark generators write real industrial instances as MPS files:

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, 'bench')
from generate import SUITE
for tag, fn, family, kind in SUITE:
    if tag.startswith('blend_s'): fn(0).write('example_lp.mps')
    if tag.startswith('uc_m'):    fn(0).write('example_milp.mps')
print("wrote example_lp.mps (refinery blending, LP) and example_milp.mps (unit commitment, MILP)")
PY
```

Any MPS file works — MPS is the universal interchange format, so a model
written for CPLEX, Gurobi or Xpress runs here unchanged.

---

## 4. Solve

```bash
./build/igaos example_lp.mps -v
```

```
IGAOS 0.2.0
  model      BLEND0
  rows       150   (12 equality, 14 ranged, 0 free)
  columns    230   (0 integer, 0 binary)
  nonzeros   880   density 2.5507%   max/min |a| 2.67e+03
  presolve   3 rows, 0 cols, 3 nonzeros removed; 173 bounds tightened
  algorithm  primal simplex

RESULT
  status         optimal
  objective      -744955.877255
  iterations     137
  primal infeas  4.235e-11
  basis cond est 2.450e+01
  time           0.002 s
```

Verbosity: `-q` silent · `-v` summary · `-vv` per phase · `-vvv` per iteration.
`--help` lists every flag.

---

## 5. The demo worth showing

Cut separation, on and off, same instance, same machine:

```bash
./build/igaos example_milp.mps -v              # cuts on
./build/igaos example_milp.mps -v --no-cuts    # cuts off
```

| | nodes | iterations | time |
|---|---|---|---|
| with cuts | **1** | 1,656 | 0.049 s |
| without cuts | 51 | 1,295 | 0.068 s |

with the log line that explains it:

```
  cuts       21 kept (19 gomory, 6 cover, 7 mir) over 1 rounds;
             root bound 76208.28398 -> 76267.64208  (+59.3581)
```

The root bound moves to the integer optimum, so the tree closes at the root.
Both runs return the same objective, `76267.6420856` — the point is that one
proves it in a single node.

**On Apple silicon and other ARM machines the counts shift slightly**: the same
run keeps 28 cuts (21 gomory, 6 cover, 7 mir) and takes 1,677 iterations with
cuts against 1,204 without. Fused multiply-add contraction is compiled
differently there, so two near-equal candidates break a tie the other way in
the ratio test and in the cut efficacy screen. What does **not** move is the
part the claim rests on: 1 node against 51, objective `76267.6420856`, and a
root bound of `76267.64208` either way.

---

## 6. The four solve paths

```bash
for a in primal dual interior pdhg; do
  echo -n "$a: "; ./build/igaos example_lp.mps -q --algorithm $a
done
```

All four reach the same optimum:

```
primal:   status=optimal obj=-744955.877255 iters=137   time=0.0017
dual:     status=optimal obj=-744955.877255 iters=1003  time=0.0170
interior: status=optimal obj=-744955.877255 iters=1028  time=0.0224
pdhg:     status=optimal obj=-744955.877255 iters=5931  time=0.0371
```

The objective is bit-identical across all four and across machines. Timings and
iteration counts are not — they are what the machine and the compiler give you.
The same four runs on aarch64 return 137 / 411 / 436 / 5,339 iterations, for the
tie-breaking reason described at the end of §5.

`--algorithm interior` runs Mehrotra predictor–corrector on the augmented KKT
system and then crossover; `--algorithm pdhg` runs the matrix-free first-order
method (the GPU path's algorithm, on CPU) and then crossover. `-v` shows the
iteration split:

```
  interior   25 iterations, 1406 factor nonzeros, 0 regularized pivots
  crossover  81 pushes, 1003 simplex iterations to a basic solution
```

A model with a quadratic objective is routed to the interior point method
whatever you pass, because the simplex optimizes a linear objective and would
otherwise silently solve a different problem.

---

## 7. Python

The bindings are ctypes over the C ABI, so there is no compiler step — but they
do need to find the package and the shared library.

```bash
export PYTHONPATH=$PWD/python
export IGAOS_LIBRARY=$PWD/build/libigaos.so      # .dylib on macOS
python3 -c "
from igaos import Model, INF
m = Model(sense='maximize')
x = m.add_variable(0, INF, cost=3.0)
y = m.add_variable(0, INF, cost=5.0)
m.add_constraint({x: 1.0}, upper=4.0)
m.add_constraint({y: 2.0}, upper=12.0)
m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)
r = m.solve(); print(r); print('x =', r.x)
"
```

```
Result(status=optimal, objective=36, iterations=2, nodes=0, algorithm='primal simplex')
x = [2.0, 6.0]
```

Or install it properly (`IGAOS_LIBRARY` is still needed unless the library is
on the system path):

```bash
pip install ./python
```

Binding tests:

```bash
IGAOS_LIBRARY=$PWD/build/libigaos.so python3 -m unittest discover -s python/tests -v
```

---

## 8. C

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

```bash
cc yours.c -Iinclude -Lbuild -ligaos -o yours
LD_LIBRARY_PATH=build ./yours          # DYLD_LIBRARY_PATH on macOS
```

`tests/test_capi.c` is a complete worked example.

---

## 9. Reproducing the measured claims

Run these in order — the first one writes the instances the second one reads.

### Head-to-head against HiGHS (needs SciPy; ~8 minutes)

```bash
IGAOS_BIN=./build/igaos python3 bench/run_bench.py --seeds 3 --time-limit 120
```

```
agreement: 47/48 instances match HiGHS to 1e-6 relative
LP time ratio vs HiGHS (geometric mean over 39): 1.19x
worst LP primal infeasibility: 4.24e-11  (blend_s_0)
MILP time ratio vs HiGHS (geometric mean over 8): 1.94x
```

All 39 LPs agree, worst relative objective difference `3.4e-12`, and 18 of the
39 are faster than HiGHS — best `blend_l_0` at 12.5×, worst `illcond_m_0` at
19.5× slower. The one instance that does not match is `uc_l_0`, a MILP that hits
the 120 s limit with a 0.66% gap still open; it is a timeout, not a wrong
answer. Time ratios move by 10–20% between machines and with load — the
agreement column does not move at all.

### What cut separation bought, as a paired run

```bash
IGAOS_BIN=./build/igaos python3 bench/cut_effect.py /tmp/inst/uc_*.mps
```

```
uc_l_2       |          optimal     0.00%     36641    83.81 |            optimal     0.00%         1     0.29    40       100.0%
uc_m_1       |          optimal     0.00%      8467     4.91 |            optimal     0.00%        17     0.19    67       100.0%
...
proven optimal without cuts: 7/9
proven optimal with cuts:    8/9
```

`uc_l_2` is the one to point at: 36,641 nodes and 84 seconds becomes a single
node and 0.29 seconds, because the root bound closes 100% of the gap.

### Cut-validity fuzzing

```bash
# mask: 1 gomory, 2 cover, 4 mir, 7 all
g++ -std=c++17 -O2 -Iinclude tools/fuzz_cuts.cpp build/libigaos_core.a -fopenmp -o fuzz_cuts
./fuzz_cuts 600 7
```

The harness solves each random model twice, with and without separation, and
also hands the separators a proven optimum as a verification point: a valid cut
can never exclude a feasible point, so any cut that does is reported immediately
instead of surfacing as a wrong objective thousands of nodes later.

### MPS-reader fuzzing

Runs automatically as part of `ctest`, so you have already done this if you ran
§3. To run it longer, or to reproduce a CI failure from the two numbers in its
log:

```bash
./build/igaos_fuzz_mps 100000 1        # iterations, seed -- deterministic
```

A different question from the fuzzer above. That one asks whether the
mathematics is right on models this program generated itself. This one asks what
a file **nobody here wrote** makes the reader do — `readMps()` is the only
function in the project that consumes bytes from outside it.

The check is not just "did it crash"; the sanitizers cover that. Every file that
**parses successfully** has its model checked against the invariants everything
downstream assumes: no NaN in the objective, bounds, matrix or quadratic terms,
`colPtr` monotone and of length `ncol+1`, every row index in range. That is the
failure mode worth catching. A crash is loud. A file that parses into a quietly
wrong model produces a confident number that looks exactly like every other
number this solver has ever produced — which is precisely what defect 26 was.

Under the sanitizers, as CI runs it:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
cmake --build build-asan --parallel
cd build-asan && ctest --output-on-failure
```

Coverage-guided, with clang:

```bash
clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -DIGAOS_LIBFUZZER \
  -Iinclude tools/fuzz_mps.cpp src/*.cpp -o fuzz_mps
mkdir -p corpus && ./fuzz_mps corpus -max_total_time=300
```

### Are the exploit mitigations actually in the binary?

Configured is not linked. `SECURITY.md` explains why the distinction matters:

```bash
checksec --file=build/igaos      # want: full RELRO, canary, NX, PIE
```

```
compared 578 optimal pairs, 0 failures
```

Both architectures return that line verbatim, and it takes about three seconds.
(Linking against `build/libigaos_core.a` as shown reuses the objects CMake
already built; `src/*.cpp` in place of the library also works and takes a few
minutes.)

---

## 10. Mixed-integer quadratic programming

The third class in the statement's "later extend to MIQP" line, and the one that
was returning wrong answers until it got its own tree.

```bash
cat > miqp.mps <<'EOF'
NAME          MIQP1
ROWS
 N  COST
 L  C1
COLUMNS
    MARKER0  'MARKER'  'INTORG'
    X  COST  -6.0
    X  C1  1.0
    Y  COST  -8.0
    Y  C1  1.0
    MARKER1  'MARKER'  'INTEND'
RHS
    RHS  C1  10.0
BOUNDS
 UP BND  X  10.0
 UP BND  Y  10.0
QUADOBJ
    X  X  2.0
    Y  Y  2.0
ENDATA
EOF
./build/igaos miqp.mps -v
```

```
  algorithm  branch and bound (convex QP relaxations, interior point)
  status         optimal
  objective      -25
  nodes          1
```

**−25 at (3, 4).** Branch and cut on the LP relaxation returns **+20**: it
optimises the linear part to (0, 10), then evaluates the quadratic objective
there and calls it optimal. A feasible point with a wrong label is the worst
thing a solver can produce, and it is what an LP tree does to a QP.

Each node relaxation is the convex QP, solved by the interior point method.
There are no cutting planes on this path and that is deliberate: Gomory cuts are
read off a simplex tableau, the interior point method does not produce one, and
separating from the wrong tableau cuts off your own optimum. A non-convex
objective is refused rather than answered.

**Multi-threaded**, sharing one node pool and one incumbent:

```bash
./build/igaos miqp.mps -q --threads 4
```

A bound found by one thread prunes work on every other, which also makes the
node count non-deterministic. Use `--threads 1` for a reproducible tree.

The **linear** branch-and-cut tree is parallel on the same terms — `--threads N`
covers both, and defaults to every core:

```bash
for t in 1 2 4; do ./build/igaos demo/models/uc_l.mps -q --no-live --threads $t; done
#   1 thread  24.6 s      2 threads  13.1 s      4 threads  12.9 s   (two-core box)
```

1.62x–1.97x at two threads across eight models, identical objectives at 1 and 4
threads on all sixteen checked, and zero ThreadSanitizer reports.
`bench/results_parallel.txt` has the table and the honest ceiling.

Validated against exhaustive enumeration: **200 random instances, 200 objectives
matching, all proven optimal.**

---

## 11. The CUDA kernels — compiled and executed, without a GPU

Two separate checks, and the difference between them is the point.

### Do they compile?

```bash
pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
python3 tools/cuda_compile_check.py cuda/pdhg_kernels.cu
```

```
sm_75: ok   5 kernels     423 PTX lines    18088 B cubin
sm_80: ok   5 kernels     423 PTX lines    18600 B cubin
sm_90: ok   5 kernels     423 PTX lines    19872 B cubin

RESULT: compiles clean to SASS for sm_75, sm_80, sm_90.
```

NVRTC turns CUDA C++ into PTX; ptxas turns PTX into the machine code the device
executes, register allocation and all. Neither needs an NVIDIA device, so this
runs in CI. Running it for the first time found two defects sitting in that file
unseen — a missing `<utility>` include, and host code mixed into a translation
unit nothing had separated.

### Do they compute the right thing?

That question was left open for a long time on the grounds that it needed a GPU.
It does not.

```bash
cd build && ctest -R cuda_kernels --output-on-failure     # no CUDA toolkit needed
```

```
spmvCsr against the CPU sparse product
  [ ok ] spmvCsr matches SparseMatrix::multiply on every shape
         7 shapes, worst relative difference 3.101e-15
primalStep against the CPU primal update ....... [ ok ] [ ok ]
dualStep  against the CPU dual update .......... [ ok ]
accumulate and axpby ........................... [ ok ] [ ok ]
a PDHG loop assembled only from the kernels, against the simplex
         instance 3: 20 x 30, kernels 13.4663941 vs simplex 13.4455313 (rel 1.4e-03)
  [ ok ] the kernel-only PDHG loop reaches the simplex optimum
ALL KERNEL CHECKS PASSED (0 failures)
```

`cuda/emulate.hpp` defines `__global__`, `blockIdx`, `threadIdx`, `blockDim` and
`__shfl_down_sync` in ordinary C++, so `tools/cuda_emulate.cpp` can include the
`.cu` and run the kernel bodies themselves — the actual text, not a
re-implementation. The warp reduction is executed lane by lane: the 32-lane
sweep is repeated to a fixed point, so every shuffle reads what the source lane
really produced in that round.

**The first run failed.** The device `dualStep` was missing the exact-zero branch
its CPU counterpart has, and left rounding noise of unstable sign on rows with no
bound on either side — the same defect whose CPU version once blew up the whole
dual convergence measure. That is defect 23 in `docs/PHASE2_REPORT.md`.

**Still not covered, and it does need hardware**: any timing number, that the
launch configuration is legal on a real device, memory coalescing, and the host
launcher at the bottom of the `.cu`. **No GPU speedup is claimed anywhere in this
project.**

---

## 11b. Nonlinear: pooling, and the general NLP

### The pooling problem, in one command

```bash
./build/igaos -m haverly1
```

```
rootlp = -500        what the linear relaxation says
final  = -400        the true global optimum, proven
```

That gap is the whole subject. Haverly's 1978 counterexample exists because the
obvious linear model of a blending network — the one that treats pool qualities
as fixed inputs — reports a blend that **cannot be made**. Blend quality is a
flow-weighted average, so quality x flow is a product of two decisions, and
fixing the quality turns a nonconvex problem into a linear one with the wrong
answer.

```bash
cd build && ctest -R pooling --output-on-failure
```

solves all three Haverly variants against their published optima (400, 600, 750)
and prints, next to each, what the same network says with the pool quality pinned
at 1%, 2% and 3%. No column is right three times.

`./build/igaos -m pool_m` runs a generated two-pool instance;
`python3 bench/generate_pooling.py outdir --seeds 3` builds the family. They are
MPS files with `QCMATRIX` sections — the CPLEX and Gurobi convention — so they
round-trip through a file and run from the command line like anything else.

### How it is solved, and what "global" means here

Spatial branch and bound over **McCormick envelopes**. Each product becomes a
variable bounded by the convex hull of the product's graph over the current box,
which makes the relaxation *linear* and therefore a **valid lower bound** — the
thing a nonconvex tree needs and a local method cannot supply. Branching splits
the range of a **continuous** variable, which has no analogue in a mixed-integer
tree. `optimal` is reported only when the tree emptied with every node bounded.

### The general NLP

```bash
cd build && ctest -R nlp --output-on-failure
```

Twelve Hock–Schittkowski problems against their published optima (**12 / 12**),
and the expression graph's exact derivatives against central differences. The
solver is a primal-dual interior point method with a filter line search and
inertia correction, on the same augmented KKT form and the same `src/ldl.cpp`
factorization the LP and QP paths use.

**It finds LOCAL solutions**, and every result says so. What that costs is
measured rather than described: on `haverly1`, from 40 random starts, it returns
a converged, zero-violation, *wrong* answer ten times.

```bash
python3 bench/generate_pooling.py benchmarks/pooling --seeds 3
./build/igaos_pooling_bench benchmarks/pooling
```

runs both solvers against each other — 12 instances, 40 starts each, 0 local
solutions beating the proven global optimum. Recorded in
`bench/results_nlp.txt`.

---

## 12. Scale

```bash
python3 bench/generate_large.py big.mps --family chain --periods 80 --scale 8
./build/igaos big.mps -q --algorithm pdhg --no-crossover
```

Multi-period supply chain, single core, first-order path:

| periods | rows | columns | nonzeros | status | wall |
|---|---|---|---|---|---|
| 20 | 28,800 | 76,800 | 153,280 | optimal | 31 s |
| 40 | 57,600 | 153,600 | 306,560 | optimal | 67 s |
| 80 | 115,200 | 307,200 | 614,080 | optimal | 90 s |

Close to linear, which is what an O(nnz)-per-iteration method should give.

**The simplex does not scale the same way on these**, and that is the finding
worth having:

| columns | simplex | first-order |
|---|---|---|
| 38,400 | 31 s | — |
| 76,800 | **145 s** | **31 s** |

4.7× the time for 2× the model. These are massively degenerate multi-period
network LPs — the class the problem statement calls out — and the revised
simplex grinds while the matrix-free method walks through. Presolve is not the
bottleneck: 0.02 s on the 76,800-column model.

**Multi-core.** How much a second core buys depends entirely on how many
nonzeros each column carries:

```bash
python3 bench/generate_large.py dense.mps --family refinery --periods 200 --scale 6
for t in 1 2; do OMP_NUM_THREADS=$t ./build/igaos dense.mps -vv \
    --algorithm pdhg --no-crossover --time-limit 1200; done
```

| matrix | nnz per column | 1 thread | 2 threads | |
|---|---|---|---|---|
| supply chain, 153,280 nnz | 2 | 31.3 s | 30.6 s | **1.02×** |
| refinery, 2,036,140 nnz | 26 | 174,720 iterations | 305,344 iterations | **1.75×** |

(The second row gives both runs the same 1200-second budget and counts
iterations, because neither converges — see below.)

With two nonzeros per column the kernel waits on random reads into `x`; it is
memory-latency bound and a second core on the same memory controller adds
nothing. With twenty-six there is enough arithmetic between the loads, and the
speedup lands where a well-parallelised reduction should — 87% efficiency on two
cores. That is also the sharpest statement of *why* a GPU: the sparse case is
short of bandwidth, not arithmetic units.

**Where it stops.** The denser refinery family at 2,036,140 nonzeros has
‖A‖₂ = 154 against 4.9 for the chain family, and the first-order path does not
converge on it inside 20 minutes — 174,720 iterations, primal infeasibility
still 1.4e-2. Scale is not what defeats it; conditioning is. That is the honest
boundary of the method as it stands, and the next thing to work on.

---

## 13. The standard benchmark libraries

Everything in §9 uses instances this repository generates. That proves the
solver handles the shapes we thought of. The public collections are how you find
out about the ones we did not.

```bash
./bench/fetch_benchmarks.sh                # ~230 MB, all publicly available
```

The instances are not redistributed here — the script pulls them from public
mirrors of Netlib and from the HiGHS test data.

### Netlib LP — the correctness test for a linear programming code

114 feasible models, many deliberately degenerate or badly scaled, with
objective values that have been agreed on since 1985.

```bash
IGAOS_BIN=./build/igaos python3 bench/netlib.py \
    --dir benchmarks/netlib --ref bench/netlib_reference.csv --time-limit 300
```

```
note: reference for forplan corrected from -1163.915769 to -664.2187395
      (netlib.org/lp/data/readme; corroborated by HiGHS)
...
solved to optimality ......... 108 / 114
objective matches reference .. 107 / 114
OBJECTIVE MISMATCHES ......... 1
not solved ................... 6: d6cube, dfl001, ken-18, pds-20, qap12, qap15
worst relative difference .... 1.68e-06  (pilot87)
worst primal infeasibility ... 3.54e-06  (cre-d)
total solve time ............. 480.1 s over 108 instances
```

Takes about 40 minutes, most of it in the six that reach the time limit. The one
mismatch is `pilot87`, the hardest instance in the set, where the published
references themselves disagree in the sixth digit.

### Netlib infeasible — the answer that must never be "optimal"

28 models with no feasible point. A solver that returns `optimal` on one of
these has handed a planner a schedule that cannot be run.

```bash
IGAOS_BIN=./build/igaos python3 bench/infeasible.py --dir benchmarks/netlib_infeasible
```

```
correctly reported infeasible .. 27 / 28
reported optimal (WRONG) ....... 1: cplex2
```

`cplex2` is infeasible by about 1e-9 — below any working feasibility tolerance —
and the returned point's violation is reported honestly as 8.1e-8. Running this
collection is what found the phase-one bug in the report's §3, item 12.

### Convex QP: the Maros and Meszaros set, against OSQP

The third class the problem statement names, on the standard public collection —
138 convex quadratic programs, the set Mittelmann's convex QP benchmark is built
on. QPLIB's own host is not reachable from a sandboxed build; this set is, and it
is the one a QP code is normally judged against.

```bash
pip install osqp
./bench/fetch_benchmarks.sh                       # includes benchmarks/qp_maros
IGAOS_BIN=./build/igaos python3 bench/qp_bench.py --dir benchmarks/qp_maros --time-limit 20
```

```
IGAOS status over all 138
  optimal            102
  numerical_error     29
  time_limit           3
  hard-timeout         2
  iteration_limit      2

Where both solved (64 instances)
  agreeing to 1e-6 relative .... 61 / 64

Solved by IGAOS but not by OSQP inside 20 s ... 38
Solved by OSQP but not by IGAOS inside 20 s ... 17
```

The 38 include the largest instances in the set: **CONT-300 at 90,298 rows by
90,597 columns solves in 15 s** while the reference does not finish, and the whole
CONT and LISWET families go the same way.

The 17 are almost entirely the **Q-prefixed family** — Netlib linear programs
with a quadratic objective attached. They are badly scaled by construction, and
they are where this solver's QP path is weakest. That is the clearest open
weakness the benchmark exposed, and it is in the table rather than in a footnote.

Three objectives disagree beyond 1e-6: AUG3DQP at 4.1e-6, DTOC3 at 8.7e-6 and
UBH1 at 5.2e-3. On the first two, running with `--ipm-tol 1e-12` moves the answer
onto OSQP's to nine digits — the default interior-point tolerance is a relative
measure that those instances satisfy while the objective is still a few parts per
million out. UBH1 is a real disagreement and is not yet explained.

**Running this set for the first time found three defects, two of them wrong
answers rather than slow ones.** They are bugs 16, 17 and 18 in
`docs/PHASE2_REPORT.md`; the shortest is that the solver read `QFORPLAN` — whose
column names contain spaces, which fixed-column MPS permits — as *infeasible*.

### Generated convex QP, against OSQP

The generated families are still run, because they cover shapes the public set
does not: rank-deficient least-squares Hessians and the singular tridiagonal
smoothing Hessian.

```bash
python3 bench/generate_qp.py benchmarks/qp --seeds 3
IGAOS_BIN=./build/igaos python3 bench/qp_bench.py --dir benchmarks/qp
```

```
agreeing with OSQP ......... 18 / 18
worst relative difference .. 8.50e-08  (qrisk_s_1)
worst IGAOS infeasibility .. 2.38e-09  (qblend_m_2)
worst OSQP  infeasibility .. 1.03e-09  (qblend_m_1)
```

Two of the three families have a **singular** Hessian on purpose — blend
property tracking gives `Q = 2SᵀS`, production smoothing gives the tridiagonal
second-difference matrix. A solver that quietly assumes positive definiteness
fails there.

### MIPLIB-family mixed integer, against HiGHS

```bash
IGAOS_BIN=./build/igaos python3 bench/mps_bench.py --dir benchmarks/milp --time-limit 300
```

```
agreeing with HiGHS ........ 7 / 9
worst relative difference .. 8.08e-14  (gesa2)
total time ................. IGAOS 23.5 s · HiGHS 3.4 s
```

`gt2` and `bell5` reach the 300 s limit. `gt2` finds 21166, the value HiGHS
proves optimal, without proving it; `bell5` finishes 0.45% above. Those two are
the honest counterweight to §5.

Both harnesses hand the reference solver the *same file*, parsed on the Python
side by `bench/mpsread.py` — written from the MPS format description rather than
from the C++ reader, so a disagreement between the two readers shows up as a
disagreement in the results instead of hiding inside a shared parser.

Results exactly as measured are committed at `bench/results_netlib.csv`,
`bench/results_milp.csv` and `bench/results_netlib_infeasible.txt`.

---

## 14. Solver Studio — the console

```bash
./run_demo.sh --web
# or, if the solver is already built:
python3 demo/server.py
```

Opens `http://127.0.0.1:8420`. Bound to localhost only — a console to stand in
front of, not a service.

### What it shows

The page is not a form that posts a model and prints an answer. It is a live
view of the solve, in three panes:

| | |
|---|---|
| **left** | fourteen models grouped LP / QP / MILP / MIQP, and every solver switch |
| **centre** | the pipeline as cards, lighting up phase by phase, over a timeline and a bound-versus-incumbent chart |
| **right** | the execution console: one card per pipeline event with its raw key/value payload, plus a Raw stdout tab holding the binary's output verbatim |

### Why it is live rather than animated

The solver is run with `--progress`, which makes it emit one line per pipeline
event at the moment the event happens, flushed:

```
IGAOS_STAGE presolve end rows=60 cols=10 nnz=120 tightened=0 t=0.0003
IGAOS_STAGE cuts end kept=32 gomory=19 cover=6 mir=7 rounds=1 before=76208.28398 after=76267.64208 t=0.0222
IGAOS_STAGE tree node nodes=264 bound=297863.7249 incumbent=303468.5171 open=176 gap=0.0184691 t=3.2306
```

`demo/server.py` reads those lines as they arrive and forwards them to the
browser over Server-Sent Events, together with every line of the human log.
Nothing on the page is on a timer and nothing is replayed from a recording: a
card lights up because the solver just entered that phase, and the node counter
moves because the tree moved. **If the solver stalls, the page stalls with it**
— which is the honest behaviour and, when you are looking for a stall, the
useful one.

You can see this for yourself: run it with the console open and watch the
`/api/stream` connection, or run the same command the page shows in its first
line and compare.

### The five card states

| state | means |
|---|---|
| `IDLE` | this phase has not reported yet |
| `RUNNING` | the solver emitted `begin` for it and has not emitted `end` |
| `DONE` | it reported `end`, with the numbers it reported |
| `SKIPPED` | switched off in the sidebar, or not on this model's path — the solver said so, the page did not infer it |
| `FELL BACK` | it ended without reaching optimality and another method finished the job |

`FELL BACK` is worth trying deliberately: pick **Refinery crude blending —
planner size**, set the algorithm to first-order, and run it. PDHG hits its
iteration limit after 500,000 iterations and the dual simplex finishes the
model in 3,496. The card goes red, the timeline shows where the twenty seconds
went, and the answer is still right. A console that could not show that would
not be worth trusting with the runs where nothing goes wrong.

### Things to try

| | |
|---|---|
| **Unit commitment — the cuts demo** | run it, then untick **cut separation** and run again. Same objective; one node against fifty-one. |
| **Unit commitment — a full day, 14 units** | about a minute. Watch the branch-and-bound card count nodes and the **Bound vs incumbent** tab close the gap in real time. |
| **Discrete lot blending (MIQP)** | the interior point card lights up as the *node* solver, and cut separation goes dashed with "no tableau to cut from" — that is the reason there is no cutting on the quadratic path, on the page. |
| **Factor-model risk (QP)** | interior point runs, crossover goes dashed with "a QP optimum need not be a vertex". |
| **Replay** | re-runs the recorded event stream from the last solve with its real timestamps stretched 8× or 30×, for models that finish faster than anyone can read. The closing line always states what the run really took. |
| **Paste an MPS file** | a model written for CPLEX, Gurobi or Xpress runs unchanged. |

### Regenerating the model library

The models are shipped rather than generated, so the console works on a machine
with no numpy. To rebuild them:

```bash
python3 demo/make_models.py
```

### What it is not

- It is Python standard library only. No Flask, no npm, no build step, no
  network access. `demo/server.py` is about 300 lines and worth reading if
  anyone asks what it does.
- There is no precomputed data anywhere in the page. Every figure comes from
  running `build/igaos` as a subprocess. If the solver is wrong, the console is
  wrong with it.
- Closing the tab or pressing **Stop** kills the solve. A sixty-second branch
  and bound does not run on after the person watching it has left.

---

## 15. Verified on two architectures

The whole of this guide was executed end to end on both:

| | x86-64 Linux, GCC 13 | aarch64 Linux, GCC 11 |
|---|---|---|
| build | warning-free | warning-free |
| solver assertions | 108 pass | 108 pass |
| C ABI assertions | 42 pass | 42 pass |
| Python binding tests | 20 pass | 20 pass |
| `pip install ./python` | imports and solves | imports and solves |
| C example against the shared library | `optimal 36` | `optimal 36` |
| `example_lp.mps` | −744955.877255, 137 iters | identical |
| four solve paths | same objective | same objective |
| cuts on / off | 1 node / 51 nodes | 1 node / 51 nodes |
| cut-validity fuzzing | 578 pairs, 0 failures | 578 pairs, 0 failures |
| MPS-reader fuzzing | 20,034 files, 0 violations | 20,034 files, 0 violations |
| LP agreement vs HiGHS | 3.4e-12 worst | 3.4e-12 worst |

Running it on the second architecture is what exposed the first-order path's
sign bug (report §3, item 11) — it was invisible on the build machine.

## 16. If something goes wrong

**`Could NOT find OpenMP`** — expected on macOS with Apple Clang. The build
continues single-threaded. `brew install libomp` if you want the parallel
kernels.

**`error: externally-managed-environment` from `pip3 install scipy`** — PEP 668.
Homebrew and Debian mark their Python as system-managed and refuse to let pip
write into it. SciPy is optional here (§9 only); if you want it, make a
virtualenv as shown in §1. `pip3 install --break-system-packages scipy` also
works but writes into the Homebrew Python, which is worth avoiding.

**`LibraryNotFound` from Python** — the exception lists every directory it
searched. Set `IGAOS_LIBRARY` to the built `libigaos.so` / `.dylib`.

**`ModuleNotFoundError: No module named 'igaos'`** — `PYTHONPATH` is not
pointing at the `python/` directory, or the package is not pip-installed.

**A build directory from another machine** — `rm -rf build` and re-run CMake.
CMake caches absolute paths, so a `build/` copied between machines will not work.

**A mixed-integer solve stops early saying it hit a memory limit** — that is
working as intended. A branch-and-bound tree that cannot close keeps growing its
list of open nodes, and a process the kernel kills has no answer at all; this one
stops, hands back the best solution it found with the bound it proved, and
reports `feasible` rather than `optimal`. The default limit is 60% of what the
machine says is free. Raise it with `--memory-limit 8000`, or turn it off with
`--memory-limit 0` and accept the risk. This is defect 24 in
`docs/PHASE2_REPORT.md`; before it was fixed, `gt2` was killed at 289 s of a
300 s limit and printed nothing at all.

**CUDA** — `-DIGAOS_ENABLE_CUDA=ON` exists, but no NVIDIA device has ever run
these kernels: there was no GPU on the machine that wrote them. Enabling it
prints a CMake warning saying so. They do compile to SASS, and their arithmetic
is checked against the CPU path by `ctest -R cuda_kernels` (§11) — which is not
the same as having run on hardware. Validate against the CPU path before trusting
any result from a real device.
