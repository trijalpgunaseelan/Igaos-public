/* test_capi.c : exercises the stable C interface the way a real consumer does.
 *
 * Deliberately compiled as C and linked against the shared library.  Including
 * the C++ headers here would test nothing about the ABI -- the point is that a
 * translation unit that has never seen a C++ declaration can build a model,
 * solve it, and read the answer back. */
#include "igaos/igaos.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int failures = 0;

static void check(int cond, const char* what) {
    printf("  [%s] %s\n", cond ? "ok" : "FAIL", what);
    if (!cond) ++failures;
}

static void checkClose(double got, double want, double tol, const char* what) {
    int ok = fabs(got - want) <= tol * (1.0 + fabs(want));
    printf("  [%s] %s (got %.12g, want %.12g)\n", ok ? "ok" : "FAIL", what, got, want);
    if (!ok) ++failures;
}

/* maximize 3x + 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18, x,y >= 0
 * The textbook Wyndham Glass problem; the optimum is x = 2, y = 6, value 36. */
static void testLinear(void) {
    igaos_model* m = igaos_create();
    int x, y, r0, r1, r2, rc;
    double sol[2], duals[3], obj;
    printf("C ABI: linear program\n");
    check(m != NULL, "handle created");
    if (!m) return;

    check(igaos_set_sense(m, IGAOS_MAXIMIZE) == IGAOS_OK, "sense set to maximize");
    x = igaos_add_column(m, 0.0, IGAOS_INFINITY, 3.0, IGAOS_CONTINUOUS, "x");
    y = igaos_add_column(m, 0.0, IGAOS_INFINITY, 5.0, IGAOS_CONTINUOUS, "y");
    check(x == 0 && y == 1, "columns indexed from zero");

    r0 = igaos_add_row(m, -IGAOS_INFINITY, 4.0,  "c0");
    r1 = igaos_add_row(m, -IGAOS_INFINITY, 12.0, "c1");
    r2 = igaos_add_row(m, -IGAOS_INFINITY, 18.0, "c2");
    check(r0 == 0 && r1 == 1 && r2 == 2, "rows indexed from zero");

    igaos_set_element(m, r0, x, 1.0);
    igaos_set_element(m, r1, y, 2.0);
    igaos_set_element(m, r2, x, 3.0);
    igaos_set_element(m, r2, y, 2.0);

    check(igaos_set_int_param(m, "verbosity", 0) == IGAOS_OK, "integer parameter accepted");
    check(igaos_set_double_param(m, "time_limit", 30.0) == IGAOS_OK, "double parameter accepted");
    check(igaos_set_int_param(m, "no_such_parameter", 1) == IGAOS_ERROR_PARAM,
          "unknown parameter rejected with IGAOS_ERROR_PARAM");
    check(strlen(igaos_last_error(m)) > 0, "error message recorded for the rejected parameter");

    rc = igaos_solve(m);
    check(rc == IGAOS_OK, "solve returned IGAOS_OK");
    check(igaos_get_status(m) == IGAOS_STATUS_OPTIMAL, "status is optimal");
    check(strcmp(igaos_status_string(IGAOS_STATUS_OPTIMAL), "optimal") == 0,
          "status string round trip");

    obj = igaos_get_objective(m);
    checkClose(obj, 36.0, 1e-9, "objective");
    check(igaos_get_solution(m, sol) == IGAOS_OK, "solution copied out");
    checkClose(sol[0], 2.0, 1e-9, "x");
    checkClose(sol[1], 6.0, 1e-9, "y");
    check(igaos_get_row_duals(m, duals) == IGAOS_OK, "row duals copied out");
    check(igaos_get_num_rows(m) == 3 && igaos_get_num_cols(m) == 2, "dimensions reported");
    check(strlen(igaos_get_algorithm(m)) > 0, "algorithm name reported");

    igaos_destroy(m);
}

/* Bulk load in compressed sparse column form, mixed-integer.
 *   min  -x0 - x1        s.t.  2x0 + 3x1 <= 7,  x integer in [0, 3]
 * The optimum is x0 = 3, x1 = 0 with value -3 (2*3 = 6 <= 7; x1 = 1 would
 * need 2x0 <= 4, giving -2-1 = -3 as well, so -3 either way). */
static void testBulkMip(void) {
    igaos_model* m = igaos_create();
    int colptr[3] = {0, 1, 2};
    int rowidx[2] = {0, 0};
    double value[2] = {2.0, 3.0};
    double obj[2] = {-1.0, -1.0};
    double lo[2] = {0.0, 0.0}, up[2] = {3.0, 3.0};
    double rlo[1] = {-IGAOS_INFINITY}, rup[1] = {7.0};
    int ctype[2] = {IGAOS_INTEGER, IGAOS_INTEGER};
    double sol[2];
    int applied = -1, gomory = -1, cover = -1, mir = -1, rounds = -1;

    printf("C ABI: bulk load, mixed-integer\n");
    check(igaos_load(m, 1, 2, colptr, rowidx, value, obj, lo, up, rlo, rup, ctype) == IGAOS_OK,
          "bulk load accepted");
    igaos_set_int_param(m, "verbosity", 0);
    check(igaos_solve(m) == IGAOS_OK, "solve returned IGAOS_OK");
    check(igaos_get_status(m) == IGAOS_STATUS_OPTIMAL, "status is optimal");
    checkClose(igaos_get_objective(m), -3.0, 1e-9, "objective");
    check(igaos_get_solution(m, sol) == IGAOS_OK, "solution copied out");
    check(fabs(sol[0] - floor(sol[0] + 0.5)) < 1e-6 &&
          fabs(sol[1] - floor(sol[1] + 0.5)) < 1e-6, "solution is integral");
    check(igaos_get_cut_counts(m, &applied, &gomory, &cover, &mir, &rounds) == IGAOS_OK,
          "cut counters readable");
    check(applied >= 0 && rounds >= 0, "cut counters are non-negative");
    igaos_destroy(m);
}

/* Convex QP: min (x-3)^2 + (y-2)^2 subject to x + y <= 4, x,y >= 0.
 * Written as x^2 + y^2 - 6x - 4y + 13.  The unconstrained minimum (3,2) sums to
 * 5 > 4, so the constraint binds and the optimum is (2.5, 1.5), value 0.5. */
static void testQuadratic(void) {
    igaos_model* m = igaos_create();
    int x, y, r;
    double sol[2];
    printf("C ABI: convex quadratic program\n");
    x = igaos_add_column(m, 0.0, IGAOS_INFINITY, -6.0, IGAOS_CONTINUOUS, "x");
    y = igaos_add_column(m, 0.0, IGAOS_INFINITY, -4.0, IGAOS_CONTINUOUS, "y");
    igaos_set_objective_offset(m, 13.0);
    /* 1/2 x'Qx with Q = 2I gives x^2 + y^2 */
    check(igaos_set_quadratic(m, x, x, 2.0) == IGAOS_OK, "Q diagonal accepted");
    check(igaos_set_quadratic(m, y, y, 2.0) == IGAOS_OK, "Q diagonal accepted");
    check(igaos_set_quadratic(m, 0, 1, 1.0) == IGAOS_ERROR_INDEX,
          "upper-triangle Q entry rejected");
    r = igaos_add_row(m, -IGAOS_INFINITY, 4.0, "budget");
    igaos_set_element(m, r, x, 1.0);
    igaos_set_element(m, r, y, 1.0);
    igaos_set_int_param(m, "verbosity", 0);
    check(igaos_solve(m) == IGAOS_OK, "solve returned IGAOS_OK");
    check(igaos_get_status(m) == IGAOS_STATUS_OPTIMAL, "status is optimal");
    check(igaos_get_solution(m, sol) == IGAOS_OK, "solution copied out");
    checkClose(sol[0], 2.5, 1e-5, "x");
    checkClose(sol[1], 1.5, 1e-5, "y");
    checkClose(igaos_get_objective(m), 0.5, 1e-5, "objective");
    igaos_destroy(m);
}

static void testErrorPaths(void) {
    igaos_model* m = igaos_create();
    double buf[4];
    printf("C ABI: error handling\n");
    check(igaos_add_column(NULL, 0, 1, 0, IGAOS_CONTINUOUS, "z") == IGAOS_ERROR_NULL,
          "null handle rejected");
    check(igaos_get_status(NULL) == IGAOS_STATUS_NOT_SOLVED, "null handle reports not-solved");
    check(igaos_get_solution(m, buf) == IGAOS_ERROR_NOT_SOLVED,
          "results before solve rejected");
    check(igaos_add_column(m, 0.0, 1.0, 1.0, 99, "bad") == IGAOS_ERROR_VALUE,
          "unknown variable type rejected");
    check(igaos_read_mps(m, "/nonexistent/path/to/model.mps") == IGAOS_ERROR_IO,
          "missing MPS file reported as IO error");
    igaos_destroy(m);
    igaos_destroy(NULL);   /* must be a no-op, not a crash */
    check(1, "destroying a null handle is safe");
}

int main(void) {
    int major = 0, minor = 0, patch = 0;
    printf("IGAOS C ABI test suite\n======================\n");
    printf("library version %s\n", igaos_version());
    igaos_version_numbers(&major, &minor, &patch);
    check(major == IGAOS_VERSION_MAJOR && minor == IGAOS_VERSION_MINOR,
          "header and library versions agree");

    testLinear();
    testBulkMip();
    testQuadratic();
    testErrorPaths();

    printf("======================\n");
    if (failures) { printf("%d FAILURES\n", failures); return 1; }
    printf("ALL C ABI TESTS PASSED\n");
    return 0;
}
