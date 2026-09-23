# API reference

Three interfaces onto the same solver: C++, a stable C ABI, and Python.

- **C++** (`include/igaos/*.hpp`) — full access, including the individual
  algorithms. Header-documented; no stability promise across versions.
- **C** (`include/igaos/igaos.h`) — the stable boundary. Opaque handle, no C++
  types, no exceptions, return codes only. This is what bindings target.
- **Python** (`python/igaos/`) — ctypes over the C ABI. No compiler, no numpy.

---

## C ABI

### Lifecycle

```c
igaos_model* m = igaos_create();   /* NULL on allocation failure */
igaos_destroy(m);                  /* destroying NULL is a no-op */
```

A handle is **not** thread-safe. Use one per thread.

### Return codes

Every `int`-returning function returns `IGAOS_OK` (0) or a negative code.
Nothing throws.

| code | meaning |
|---|---|
| `IGAOS_OK` | success |
| `IGAOS_ERROR_NULL` | null handle or buffer |
| `IGAOS_ERROR_INDEX` | row or column index out of range |
| `IGAOS_ERROR_PARAM` | unknown parameter name |
| `IGAOS_ERROR_VALUE` | parameter value rejected |
| `IGAOS_ERROR_IO` | file could not be read or written |
| `IGAOS_ERROR_NOT_SOLVED` | result requested before `igaos_solve` |
| `IGAOS_ERROR_INTERNAL` | unexpected failure; see `igaos_last_error` |

`igaos_last_error(m)` returns the message for the most recent failure on that
handle, valid until the next call on it.

### Building a model

```c
int igaos_add_column(igaos_model*, double lower, double upper, double cost,
                     int vartype, const char* name);   /* returns the index */
int igaos_add_row(igaos_model*, double lower, double upper, const char* name);
int igaos_set_element(igaos_model*, int row, int col, double value);
int igaos_set_quadratic(igaos_model*, int i, int j, double value);  /* i >= j */
int igaos_set_sense(igaos_model*, int sense);          /* IGAOS_MINIMIZE / MAXIMIZE */
int igaos_set_objective_offset(igaos_model*, double);
```

Bounds use `±IGAOS_INFINITY` (`1e30`) for absent. There is no row "sense"
argument — a row is `lower ≤ a'x ≤ upper`, and `=`, `≤`, `≥`, ranged and free
are all bound patterns of that one form.

`igaos_set_quadratic` takes the **lower triangle** of a symmetric positive
semidefinite `Q`; the objective is `c'x + ½x'Qx`. An upper-triangle index is
rejected with `IGAOS_ERROR_INDEX` rather than silently transposed. Convexity is
assumed, not checked — checking would cost an eigenvalue computation on every
solve. A model with any `Q` entry is solved by the interior point method
regardless of the `algorithm` parameter, because the simplex optimizes a linear
objective and would otherwise silently solve a different problem.

### Bulk load

```c
int igaos_load(igaos_model*, int nrow, int ncol,
               const int* colptr, const int* rowidx, const double* value,
               const double* obj,
               const double* collower, const double* colupper,
               const double* rowlower, const double* rowupper,
               const int* coltype);
```

Compressed sparse column. Any bound array may be `NULL` for the defaults
(`0`/`+inf` for columns, `±inf` for rows); `coltype` may be `NULL` for all
continuous. The objective sense and offset set beforehand are **preserved** —
a bulk load replaces the structure, not the objective's orientation.

### Parameters

```c
int igaos_set_int_param(igaos_model*, const char* name, long long value);
int igaos_set_double_param(igaos_model*, const char* name, double value);
int igaos_get_int_param(igaos_model*, const char* name, long long* out);
int igaos_get_double_param(igaos_model*, const char* name, double* out);
```

**Integer parameters**

| name | default | meaning |
|---|---|---|
| `verbosity` | 0 from the C API | 0 silent … 4 numerical diagnostics |
| `algorithm` | 0 | 0 auto, 1 primal simplex, 2 dual simplex, 3 interior point, 4 PDHG |
| `presolve` | 1 | run presolve/postsolve |
| `scaling` | 1 | geometric equilibration rounded to powers of two |
| `crash` | 1 | triangular crash basis |
| `crossover` | 1 | make the interior point / PDHG solution basic |
| `cuts` | 1 | root cutting-plane loop |
| `cut_gomory`, `cut_cover`, `cut_mir` | 1 | individual separators |
| `cut_rounds_root` | 12 | root separation rounds |
| `heuristics` | 1 | rounding and diving |
| `reliability` | 4 | pseudocost reliability threshold |
| `threads` | 0 | 0 = all available |
| `refactor_frequency` | 100 | simplex updates between refactorizations |
| `iteration_limit`, `node_limit` | very large | search limits |
| `ipm_max_iter` | 200 | interior point iteration cap |
| `pdhg_max_iter` | 500000 | first-order iteration cap |
| `pdhg_restart` | 64 | restart cadence base |

**Double parameters**

| name | default | meaning |
|---|---|---|
| `time_limit` | ∞ | wall clock seconds |
| `mip_gap_relative` | 1e-6 | relative optimality gap |
| `mip_gap_absolute` | 1e-9 | absolute optimality gap |
| `primal_tolerance` | 1e-7 | row activity violation |
| `dual_tolerance` | 1e-7 | reduced cost sign violation |
| `integrality_tolerance` | 1e-6 | distance from an integer |
| `pivot_tolerance` | 1e-9 | smallest acceptable pivot |
| `markowitz` | 0.01 | threshold pivoting factor |
| `cutoff` | ∞ | prune nodes no better than this |
| `ipm_tolerance` | 1e-8 | interior point convergence |
| `pdhg_tolerance` | 1e-8 | first-order convergence |

### Solving and results

```c
int igaos_solve(igaos_model*);

int    igaos_get_status(const igaos_model*);        /* IGAOS_STATUS_* */
double igaos_get_objective(const igaos_model*);
double igaos_get_best_bound(const igaos_model*);
double igaos_get_mip_gap(const igaos_model*);
double igaos_get_solve_time(const igaos_model*);
long long igaos_get_iterations(const igaos_model*);
long long igaos_get_nodes(const igaos_model*);

int igaos_get_solution(const igaos_model*, double* x);       /* ncol */
int igaos_get_reduced_costs(const igaos_model*, double* d);  /* ncol */
int igaos_get_row_activity(const igaos_model*, double* a);   /* nrow */
int igaos_get_row_duals(const igaos_model*, double* y);      /* nrow */
```

`IGAOS_STATUS_FEASIBLE` means an incumbent was found but optimality was not
proven — check `igaos_get_mip_gap`. It is distinct from `IGAOS_STATUS_OPTIMAL`
on purpose: a solver that reports "optimal" for a solution it has not proven is
the one thing a planning department cannot recover from.

### Diagnostics

```c
int igaos_get_cut_counts(const igaos_model*, int* applied, int* gomory,
                         int* cover, int* mir, int* rounds);
double igaos_get_root_bound_before_cuts(const igaos_model*);
double igaos_get_root_bound_after_cuts(const igaos_model*);
const char* igaos_get_algorithm(const igaos_model*);
```

The two root bounds are what let a caller measure separation rather than
trust it: `(after − before) / (optimum − before)` is the fraction of the root
gap the cuts closed.

---

## Python

```python
from igaos import Model, Result, Status, INF, IgaosError
```

### Model

| method | notes |
|---|---|
| `Model(sense="minimize", name="")` | context manager; `close()` releases the handle |
| `add_variable(lower=0, upper=INF, cost=0, vtype="continuous", name="")` | returns the index |
| `add_binary(cost=0, name="")`, `add_integer(...)` | conveniences |
| `add_constraint({col: coef, ...}, lower=-INF, upper=INF, name="")` | returns the index |
| `set_quadratic(i, j, value)` | upper-triangle indices are swapped for you |
| `load(num_rows, num_cols, colptr, rowidx, values, ...)` | CSC bulk load |
| `Model.from_mps(path)`, `read_mps`, `write_mps` | MPS interchange |
| `set_param(name, value)`, `get_param(name)` | routed to the int or double table by type |
| `solve(**params)` | parameters may be passed inline |
| `result()` | the last `Result` |
| `num_variables`, `num_constraints` | properties |

### Result

`status`, `status_name`, `optimal`, `feasible`, `objective`, `best_bound`,
`mip_gap`, `iterations`, `nodes`, `solve_time`, `algorithm`, `x`,
`reduced_costs`, `row_activity`, `row_duals`, `cuts_applied`, `cuts_gomory`,
`cuts_cover`, `cuts_mir`, `cut_rounds`, `root_bound_before_cuts`,
`root_bound_after_cuts`.

### Finding the library

In order: an explicit path, `$IGAOS_LIBRARY`, `$IGAOS_LIBRARY_PATH`, the package
directory, a sibling `build/` tree, then the platform loader. If none works,
`LibraryNotFound` explains how to build it and lists everywhere it looked.

```bash
export IGAOS_LIBRARY=/path/to/libigaos.so
python -m unittest discover -s python/tests
```

### Scale

`add_constraint` costs one Python-to-C transition per coefficient. That is fine
for thousands of nonzeros and not fine for hundreds of thousands — use
`Model.load` there, which hands the whole matrix over in one call.

---

## C++

`Solver` is the top-level driver:

```cpp
#include "igaos/solver.hpp"

igaos::Model m;                       /* addColumn / addRow / setElement / finalize */
igaos::Solver s;
s.opt.timeLimit = 60.0;
s.opt.lpAlgorithm = igaos::LpAlgorithm::InteriorPoint;
igaos::Solution sol = s.solve(m);
const igaos::SolveReport& r = s.report;   /* timings, cut counts, iteration splits */
```

The individual algorithms are also directly callable, which is what the
regression suite uses to check them against each other:

| header | entry point |
|---|---|
| `simplex.hpp` | `Simplex::load` / `solve` / `tableauRow` |
| `ipm.hpp` | `interiorPoint(model, options) -> IpmResult` |
| `crossover.hpp` | `crossover(model, options, ipmResult) -> CrossoverResult` |
| `pdhg.hpp` | `primalDualHybridGradient(model, options) -> PdhgResult` |
| `cuts.hpp` | `separateGomory` / `separateCover` / `separateMir`, `CutPool`, `runRootCutLoop` |
| `presolve.hpp` | `presolve` / `postsolve` / `computeScaling` |
| `mps.hpp` | `readMps` / `writeMps` / `writeSolution` |

### The cut verification hook

`Options::cutReference` (and `CutLimits::referencePoint`) accept a point known to
be feasible for the mixed-integer problem. Every candidate cut is tested against
it and rejected if it would exclude it; `SolveReport::cutsInvalid` counts the
rejections and must be zero on a correct solver.

```cpp
Solution reference = plainSolver.solve(model);      // cuts off
cutSolver.opt.cutReference = &reference.colValue;   // cuts on, verified
Solution checked = cutSolver.solve(model);
assert(cutSolver.report.cutsInvalid == 0);
```

Use it in tests and when bringing up a new separator. It is null in normal
solves.
