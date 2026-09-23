# Working on IGAOS

For teammates and collaborators. If you only want to *run* it, `./start.sh` and
[RUNNING.md](RUNNING.md) are all you need — this file is about changing it.

## First five minutes

```bash
git clone https://github.com/<owner>/IGAOS.git
cd IGAOS
./start.sh
```

That checks your toolchain, builds, runs the tests, solves an example, and opens
Solver Studio at <http://127.0.0.1:8420>. If it fails it tells you which tool is
missing and the command to get it.

Prerequisites are a C++17 compiler and CMake. Python 3 is needed only for the
console and the benchmark harnesses. No optimization library is required,
procured, or licensed — that is the point of the project.

While you work:

```bash
cmake --build build -j          # rebuild after an edit (seconds)
cd build && ctest --output-on-failure && cd ..
./start.sh --quick              # rebuild and reopen the console, skip tests
```

## Where everything lives

```
include/igaos/   public headers — the API surface, one per subsystem
src/             the solver, 8,813 lines of our own C++17
apps/            igaos_cli.cpp, the command line front end
tests/           test_all.cpp (C++ assertions) and test_capi.c (the C ABI)
tools/           standalone utilities: the certificate checker, the cut fuzzer,
                 the MPS-reader fuzzer, cuda_emulate.cpp (runs the CUDA kernels
                 on a CPU), benches, pin_actions.sh, make_sbom.py
bench/           benchmark harnesses and the recorded result files
python/          the Python binding and its own tests
cuda/            pdhg_kernels.cu and emulate.hpp — the kernels compile, and
                 their arithmetic is executed on the CPU; no device has run them
bench/generate_pooling.py   the pooling family: crude blending with pool
                 qualities as DECISIONS, plus the three Haverly instances
demo/            Solver Studio: server.py, index.html, models/
docs/            algorithms.md, api.md, PHASE2_REPORT.md (the defect log, 26),
                 THREAT-MODEL.md, EXTENSION-NLP-MINLP.md (what is NOT built)
SECURITY.md      attack surface, controls, and the open gaps stated plainly
sbom.json        CycloneDX; regenerate with python3 tools/make_sbom.py
```

### The solver, by file

| File | What it is |
|---|---|
| `src/simplex.cpp` | Bounded-variable revised simplex, primal and dual. Harris two-pass ratio test, Devex and dual steepest-edge pricing, Forrest–Goldfarb basis update. |
| `src/solver.cpp` | The orchestrator: chooses a path, runs branch and cut, emits the stage events the console draws. The tree is a worker pool — one copy of the relaxation per thread, one mutex over the queue, the pseudocosts and the incumbent. |
| `src/cuts.cpp` | Gomory mixed-integer, knapsack cover, complemented MIR separation, and the cut pool. |
| `src/ipm.cpp` | Mehrotra predictor–corrector on the augmented KKT system. Augmented rather than normal equations because a non-diagonal Q makes the normal form dense. |
| `src/presolve.cpp` | Reductions and the postsolve stack that undoes them. |
| `src/mps.cpp` | MPS reader — both free and fixed-column, detected by scanning the whole file. |
| `src/miqp.cpp` | Branch and bound over convex QP relaxations, multi-threaded. The pattern the LP tree's pool was built from. |
| `src/global.cpp` | Spatial branch and bound over McCormick envelopes: nonconvex QCQP and bilinear MINLP, solved to **proven global** optimality. Each node's relaxation is handed back to `Solver::solve`, so it is an LP or a MILP and the existing branch-and-cut does the work. |
| `src/expr.cpp` | The expression tape and its exact derivatives — reverse mode for gradients, forward-over-reverse for Hessian-vector products. |
| `src/nlp.cpp` | Primal-dual interior point for smooth NLP: filter line search, inertia correction. **Local solutions only**, and `NlpResult::localOnly` says so on every result. |
| `src/lu.cpp` / `src/ldl.cpp` | Sparse LU with threshold Markowitz pivoting; quasi-definite LDLᵀ with AMD ordering. |
| `src/pdhg.cpp` | Restarted matrix-free primal–dual hybrid gradient. |
| `src/crossover.cpp` | Interior point solution → a basic solution. |
| `src/certificate.cpp` | Writes the (x, y) pair that lets something else check the answer. |
| `src/capi.cpp` | The C ABI the Python binding sits on. |

Read `include/igaos/certificate.hpp` before touching anything to do with
certificates — the theorem is stated there in full, and it is short.

## Rules that are not negotiable

These come from the problem statement and from what the project claims about
itself. Breaking one is worse than not making the change.

**1. No third-party solver code.** PS 26119 requires the solver be built "from
scratch from mathematical foundation." Not a line of external optimization code
is vendored or linked. Every `#include <...>` resolves to the standard library,
to POSIX headers used only by the terminal front end, to `<omp.h>`, or to
`<cuda_runtime.h>`. To check that you have not broken it:

```bash
grep -rhE '^[[:space:]]*#[[:space:]]*include[[:space:]]*<' \
     src include apps tests tools cuda | \
    sed 's/.*include[[:space:]]*<//; s/>.*//' | sort -u

for h in $(grep -rhE '^[[:space:]]*#[[:space:]]*include[[:space:]]*"' \
                src include apps tests tools | \
           sed 's/.*include[[:space:]]*"//; s/".*//' | sort -u); do
    [ -f "include/$h" ] || [ -f "apps/$h" ] || echo "OUTSIDE: $h"
done
```

The second command must print nothing. Note the regexes match *indented*
preprocessor directives: an earlier version anchored on `^#include` and so never
saw the POSIX headers inside `#if !defined(_WIN32)` blocks. An audit command
that silently misses things is worse than none, because it is believed. HiGHS, OSQP, NumPy and SciPy appear only
in `bench/` and `demo/`, as separate programs we compare against — never linked.

**2. Never claim a number nobody measured.** The CUDA kernels compile, and
`tools/cuda_emulate.cpp` executes their arithmetic on a CPU and checks it
against `src/pdhg.cpp` and the simplex — that runs under `ctest`. No NVIDIA
device has ever run them. There is no measured GPU speedup, and the code, the
README and the docs all say so. If you touch `cuda/pdhg_kernels.cu`, run
`ctest -R cuda_kernels`; it is the only thing that will notice if the device
path stops computing what the CPU path computes, which is exactly how defect 23
was found.

The MILP tree is parallel and its node counts are not reproducible above one
thread. Any measurement you record has to say the thread count it was taken at,
and any equivalence check has to compare `--threads 1` against `--threads N`
on the OBJECTIVE, never on the node count.

CPLEX and Gurobi *are* now compared against, through their free size-limited
licences (`bench/commercial.py`) — but only on models under 1000x1000 and
2000x2000, and only as a CORRECTNESS result. On mixed-integer models we are one
to three orders of magnitude slower, that is in `bench/results_commercial.txt`,
and no run in this repository narrows it. Xpress has never been run at all.

If you add a performance claim, add the command that produced it in the same
commit.

**3a. Never say "global" without a valid bound.** `src/global.cpp` may report
`Optimal` only when its tree emptied with every node bounded by a *relaxation*.
`src/nlp.cpp` finds local solutions and must never report anything stronger —
`solveMinlp` returns `Feasible` even when it is almost certainly right, because
a local solution is not a bound and a pruned subtree may have held the optimum.
If you add a path, decide which of those two it is before you write it.

**3. Keep the limitations in.** README and `docs/PHASE2_REPORT.md` list the six
Netlib instances we do not solve, the 29 Maros–Mészáros QP instances we do not
solve, and all twenty-five defects found so far, two of which returned a
confident wrong answer.
That honesty is the most valuable thing in the repository. Do not tidy it away.

## Adding a test

`tests/test_all.cpp` is plain assertions, no framework. Add to the relevant
section and it runs under `ctest` automatically:

```cpp
{
    Model m = /* build the smallest model that shows the behaviour */;
    Solution s = solve(m, opt);
    CHECK(s.status == Status::Optimal);
    CHECK(std::abs(s.objective - 42.0) < 1e-9);
}
```

Prefer a model small enough to reason about by hand. Most of the twenty-five
defects were found by a small instance behaving oddly, not by a large one
running slowly.

For anything that produces an answer, also check it independently:

```bash
./build/igaos model.mps --certificate model.cert
python3 tools/verify_certificate.py model.mps model.cert
```

`verify_certificate.py` shares no code with the solver and uses exact rational
arithmetic — no floating point at all. Agreement between the two is evidence
rather than tautology.

## Benchmarks

Instances are not in the repository — they belong to their publishers.

```bash
bench/fetch_benchmarks.sh          # Netlib, MIPLIB, Maros–Mészáros, and the
                                   # small set the free CPLEX/Gurobi licences
                                   # can actually load
python3 bench/netlib.py            # correctness sweep, the one that matters
python3 bench/qp_bench.py          # convex QP against OSQP
python3 bench/mps_bench.py         # MILP against HiGHS
python3 bench/commercial.py --dir benchmarks/standard    # vs CPLEX and Gurobi
```

Recorded results live beside the harnesses in `bench/results_*.csv`. When a run
changes a number, commit the new result file with the change that caused it.

## Working on the console

`demo/server.py` is a small HTTP server that runs the solver as a subprocess and
forwards its stage events over Server-Sent Events. `demo/index.html` is the
whole front end in one file — no build step, no framework, no package manager.
Edit it and reload the page.

The flow diagram is driven by real solver events, not an animation. If you add a
stage to the solver, emit it with `opt.log.stage(...)` and add a matching node to
`NODES` in `index.html`; the edges are drawn from measured DOM positions, so
nothing else needs adjusting.

`python3 demo/make_models.py` regenerates the fourteen bundled instances.

## Commits

Say what changed and why it was wrong before. `docs/PHASE2_REPORT.md` is the
running defect log; a bug fix belongs there as a short narrative — what the
symptom was, how it was found, what the root cause turned out to be. Several of
those entries are the most interesting reading in the project.

If the change touches `readMps()`, the C ABI, `demo/server.py`, the build flags
or CI, it is a change to the security position too: update `SECURITY.md` and, if
it moves a trust boundary, `docs/THREAT-MODEL.md`. A control that quietly stops
being true is worse than one that was never claimed. And if you add a source
file, run `python3 tools/make_sbom.py` — `sbom.json` carries a hash of the tree,
so a stale one is detectable rather than merely wrong.

## Licence

See [LICENSE](LICENSE). Copyright stays with the team; MRPL, ONGC, the Ministry
of Education and AICTE have a perpetual royalty-free licence; all other rights
are reserved. Contributing means you are fine with your work under those terms.
