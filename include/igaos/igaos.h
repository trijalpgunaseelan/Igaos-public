/* igaos.h : the stable C interface to IGAOS.
 *
 * This is the boundary the rest of the world links against.  Everything here is
 * plain C89-compatible declarations behind an opaque handle: no C++ types cross
 * it, no exceptions escape it, and no structure layout is exposed.  That is what
 * makes it an ABI rather than merely an API -- a program built against version
 * 0.2 keeps working against a later shared library, and bindings for languages
 * that speak the C ABI (Python's ctypes, Java's JNI, Rust's extern "C", .NET
 * P/Invoke) need no compiler and no glue library.
 *
 * Conventions
 *   - Every function returning int returns IGAOS_OK (0) on success and a
 *     negative IGAOS_ERROR_* code on failure.  Nothing throws.
 *   - Indices are 0-based.  Column and row indices are those returned by
 *     igaos_add_column / igaos_add_row, or the natural order of a bulk load.
 *   - +/- IGAOS_INFINITY marks an absent bound.
 *   - Strings returned by this library are owned by it and stay valid until the
 *     next call on the same handle.
 *   - A handle is not thread-safe; use one handle per thread.
 *
 * Licence: see LICENSE at the repository root.  No third-party optimization
 * library is linked at any layer, so nothing here carries a third-party
 * obligation.
 */
#ifndef IGAOS_H
#define IGAOS_H

#ifdef __cplusplus
extern "C" {
#endif

#define IGAOS_VERSION_MAJOR 0
#define IGAOS_VERSION_MINOR 2
#define IGAOS_VERSION_PATCH 0

#define IGAOS_INFINITY 1e30

/* ---- return codes ------------------------------------------------------- */
#define IGAOS_OK                 0
#define IGAOS_ERROR_NULL        (-1)   /* null handle or null buffer */
#define IGAOS_ERROR_INDEX       (-2)   /* row or column index out of range */
#define IGAOS_ERROR_PARAM       (-3)   /* unknown parameter name */
#define IGAOS_ERROR_VALUE       (-4)   /* parameter value rejected */
#define IGAOS_ERROR_IO          (-5)   /* file could not be read or written */
#define IGAOS_ERROR_NOT_SOLVED  (-6)   /* result requested before igaos_solve */
#define IGAOS_ERROR_INTERNAL    (-7)   /* unexpected failure; see last_error */

/* ---- solve status (matches igaos::Status) -------------------------------- */
#define IGAOS_STATUS_NOT_SOLVED      0
#define IGAOS_STATUS_OPTIMAL         1
#define IGAOS_STATUS_INFEASIBLE      2
#define IGAOS_STATUS_UNBOUNDED       3
#define IGAOS_STATUS_ITERATION_LIMIT 4
#define IGAOS_STATUS_TIME_LIMIT      5
#define IGAOS_STATUS_NODE_LIMIT      6
#define IGAOS_STATUS_NUMERICAL_ERROR 7
#define IGAOS_STATUS_INTERRUPTED     8
#define IGAOS_STATUS_FEASIBLE        9   /* incumbent found, optimality unproven */

/* ---- variable types ------------------------------------------------------ */
#define IGAOS_CONTINUOUS 0
#define IGAOS_INTEGER    1
#define IGAOS_BINARY     2

/* ---- objective sense ----------------------------------------------------- */
#define IGAOS_MINIMIZE  1
#define IGAOS_MAXIMIZE (-1)

/* ---- algorithm selection (the "algorithm" integer parameter) ------------- */
#define IGAOS_ALGORITHM_AUTO           0
#define IGAOS_ALGORITHM_PRIMAL_SIMPLEX 1
#define IGAOS_ALGORITHM_DUAL_SIMPLEX   2
#define IGAOS_ALGORITHM_INTERIOR_POINT 3
#define IGAOS_ALGORITHM_PDHG           4

typedef struct igaos_model igaos_model;

/* ---- library identity ---------------------------------------------------- */
const char* igaos_version(void);
void        igaos_version_numbers(int* major, int* minor, int* patch);

/* ---- lifecycle ----------------------------------------------------------- */
igaos_model* igaos_create(void);
void         igaos_destroy(igaos_model* m);

/* ---- incremental model building ------------------------------------------ */
/* Return the new index (>= 0) or a negative error code. */
int igaos_add_column(igaos_model* m, double lower, double upper, double cost,
                     int vartype, const char* name);
int igaos_add_row(igaos_model* m, double lower, double upper, const char* name);
int igaos_set_element(igaos_model* m, int row, int col, double value);
/* Lower triangle of a symmetric positive semidefinite Q; the objective is
 * c'x + 1/2 x'Qx.  A model with any Q entry is solved by the interior point
 * method regardless of the algorithm parameter -- the simplex here optimizes a
 * linear objective, so it would otherwise silently solve a different problem. */
int igaos_set_quadratic(igaos_model* m, int i, int j, double value);
int igaos_set_sense(igaos_model* m, int sense);
int igaos_set_objective_offset(igaos_model* m, double offset);

/* ---- bulk load ----------------------------------------------------------- */
/* A is compressed sparse column: colptr has ncol+1 entries, rowidx and value
 * have colptr[ncol].  coltype may be NULL, meaning all columns continuous.
 * Any bound array may be NULL, meaning 0 / +infinity for columns and
 * -infinity / +infinity for rows. */
int igaos_load(igaos_model* m, int nrow, int ncol,
               const int* colptr, const int* rowidx, const double* value,
               const double* obj,
               const double* collower, const double* colupper,
               const double* rowlower, const double* rowupper,
               const int* coltype);

/* ---- file interchange ---------------------------------------------------- */
int igaos_read_mps(igaos_model* m, const char* path);
int igaos_write_mps(igaos_model* m, const char* path);

/* ---- parameters ---------------------------------------------------------- */
/* Integer parameters: "verbosity", "algorithm", "presolve", "scaling", "crash",
 *   "crossover", "cuts", "cut_gomory", "cut_cover", "cut_mir", "cut_rounds_root",
 *   "heuristics", "reliability", "threads", "refactor_frequency",
 *   "iteration_limit", "node_limit", "memory_limit_mb", "ipm_max_iter",
 *   "pdhg_max_iter", "pdhg_restart".
 *
 * "threads" is 0 for all cores, 1 for a serial (and reproducible) tree search.
 * "memory_limit_mb" is -1 for automatic (60% of what the machine reports free),
 *   0 for no limit, or a size in megabytes: the branch-and-bound search stops
 *   and reports its incumbent as feasible rather than being killed.
 * Double parameters: "time_limit", "mip_gap_relative", "mip_gap_absolute",
 *   "primal_tolerance", "dual_tolerance", "integrality_tolerance",
 *   "pivot_tolerance", "markowitz", "cutoff", "ipm_tolerance", "pdhg_tolerance". */
int igaos_set_int_param(igaos_model* m, const char* name, long long value);
int igaos_get_int_param(igaos_model* m, const char* name, long long* value);
int igaos_set_double_param(igaos_model* m, const char* name, double value);
int igaos_get_double_param(igaos_model* m, const char* name, double* value);

/* ---- solve --------------------------------------------------------------- */
int igaos_solve(igaos_model* m);

/* ---- results ------------------------------------------------------------- */
int    igaos_get_status(const igaos_model* m);
const char* igaos_status_string(int status);
double igaos_get_objective(const igaos_model* m);
double igaos_get_best_bound(const igaos_model* m);
double igaos_get_mip_gap(const igaos_model* m);
double igaos_get_solve_time(const igaos_model* m);
long long igaos_get_iterations(const igaos_model* m);
long long igaos_get_nodes(const igaos_model* m);
int    igaos_get_num_rows(const igaos_model* m);
int    igaos_get_num_cols(const igaos_model* m);

/* Each fills a caller-owned buffer of the stated length. */
int igaos_get_solution(const igaos_model* m, double* x);        /* ncol */
int igaos_get_reduced_costs(const igaos_model* m, double* d);   /* ncol */
int igaos_get_row_activity(const igaos_model* m, double* a);    /* nrow */
int igaos_get_row_duals(const igaos_model* m, double* y);       /* nrow */

/* ---- diagnostics --------------------------------------------------------- */
/* Cut statistics are zero unless the model is mixed-integer. */
int igaos_get_cut_counts(const igaos_model* m, int* applied, int* gomory,
                         int* cover, int* mir, int* rounds);
double igaos_get_root_bound_before_cuts(const igaos_model* m);
double igaos_get_root_bound_after_cuts(const igaos_model* m);
/* Human-readable name of the path that actually ran, e.g. "branch and cut ...". */
const char* igaos_get_algorithm(const igaos_model* m);
/* Message for the most recent failure on this handle, or "" if none. */
const char* igaos_last_error(const igaos_model* m);

#ifdef __cplusplus
}
#endif
#endif /* IGAOS_H */
