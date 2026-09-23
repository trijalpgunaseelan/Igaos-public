// capi.cpp : implementation of the stable C interface.
//
// Two rules govern everything in this file.
//
//   1. No exception may cross the boundary.  A C caller has no way to catch one
//      and the behaviour of unwinding through a C frame is undefined, so every
//      entry point is wrapped and every failure becomes a return code plus a
//      message the caller can read back.
//   2. No C++ type may appear in a signature or in the layout of anything the
//      caller holds.  The handle is opaque, which is what lets the internals
//      change without breaking a binary that is already linked.
#include "igaos/igaos.h"
#include "igaos/solver.hpp"
#include "igaos/mps.hpp"

#include <cstring>
#include <string>
#include <new>

using namespace igaos;

struct igaos_model {
    Model    model;
    Options  opt;
    Solver   solver;
    Solution solution;
    bool     solved = false;
    std::string error;
    std::string algorithm;
};

namespace {

const char* kVersion = "0.2.0";

int fail(igaos_model* m, int code, const char* msg) {
    if (m) m->error = msg;
    return code;
}

// Every entry point funnels through this, so "no exception escapes" is a
// property of one function rather than a discipline applied by hand in forty.
template <typename Fn>
int guarded(igaos_model* m, Fn&& fn) {
    if (!m) return IGAOS_ERROR_NULL;
    try {
        m->error.clear();
        return fn();
    } catch (const std::bad_alloc&) {
        return fail(m, IGAOS_ERROR_INTERNAL, "out of memory");
    } catch (const std::exception& e) {
        return fail(m, IGAOS_ERROR_INTERNAL, e.what());
    } catch (...) {
        return fail(m, IGAOS_ERROR_INTERNAL, "unknown internal error");
    }
}

bool nameIs(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

} // namespace

// ===========================================================================
extern "C" {

const char* igaos_version(void) { return kVersion; }

void igaos_version_numbers(int* major, int* minor, int* patch) {
    if (major) *major = IGAOS_VERSION_MAJOR;
    if (minor) *minor = IGAOS_VERSION_MINOR;
    if (patch) *patch = IGAOS_VERSION_PATCH;
}

igaos_model* igaos_create(void) {
    try {
        igaos_model* m = new igaos_model();
        m->opt.log.level = 0;      // a library is quiet unless asked
        return m;
    } catch (...) {
        return nullptr;
    }
}

void igaos_destroy(igaos_model* m) { delete m; }

// ---- model building -------------------------------------------------------
int igaos_add_column(igaos_model* m, double lower, double upper, double cost,
                     int vartype, const char* name) {
    return guarded(m, [&] {
        VarType t = VarType::Continuous;
        if (vartype == IGAOS_INTEGER) t = VarType::Integer;
        else if (vartype == IGAOS_BINARY) t = VarType::Binary;
        else if (vartype != IGAOS_CONTINUOUS)
            return fail(m, IGAOS_ERROR_VALUE, "unknown variable type");
        m->solved = false;
        return (int)m->model.addColumn(lower, upper, cost, t, name ? name : "");
    });
}

int igaos_add_row(igaos_model* m, double lower, double upper, const char* name) {
    return guarded(m, [&] {
        m->solved = false;
        return (int)m->model.addRow(lower, upper, name ? name : "");
    });
}

int igaos_set_element(igaos_model* m, int row, int col, double value) {
    return guarded(m, [&] {
        if (row < 0 || col < 0) return fail(m, IGAOS_ERROR_INDEX, "negative index");
        m->solved = false;
        m->model.setElement(row, col, value);
        return IGAOS_OK;
    });
}

int igaos_set_quadratic(igaos_model* m, int i, int j, double value) {
    return guarded(m, [&] {
        if (i < 0 || j < 0) return fail(m, IGAOS_ERROR_INDEX, "negative index");
        if (j > i) return fail(m, IGAOS_ERROR_INDEX,
                               "Q is stored as a lower triangle: expected column <= row");
        m->solved = false;
        m->model.setQuadratic(i, j, value);
        return IGAOS_OK;
    });
}

int igaos_set_sense(igaos_model* m, int sense) {
    return guarded(m, [&] {
        if (sense != IGAOS_MINIMIZE && sense != IGAOS_MAXIMIZE)
            return fail(m, IGAOS_ERROR_VALUE, "sense must be IGAOS_MINIMIZE or IGAOS_MAXIMIZE");
        m->model.sense = (sense == IGAOS_MAXIMIZE) ? Sense::Maximize : Sense::Minimize;
        m->solved = false;
        return IGAOS_OK;
    });
}

int igaos_set_objective_offset(igaos_model* m, double offset) {
    return guarded(m, [&] { m->model.objOffset = offset; m->solved = false; return IGAOS_OK; });
}

int igaos_load(igaos_model* m, int nrow, int ncol,
               const int* colptr, const int* rowidx, const double* value,
               const double* obj,
               const double* collower, const double* colupper,
               const double* rowlower, const double* rowupper,
               const int* coltype) {
    return guarded(m, [&] {
        if (nrow < 0 || ncol < 0) return fail(m, IGAOS_ERROR_VALUE, "negative dimension");
        if (ncol > 0 && (!colptr || (!rowidx && colptr[ncol] > 0) || (!value && colptr[ncol] > 0)))
            return fail(m, IGAOS_ERROR_NULL, "matrix arrays must not be null");

        Model& mo = m->model;
        // A bulk load replaces the *structure* of the model, not the objective
        // sense or offset the caller configured beforehand.  Resetting those
        // silently turned a maximization into a minimization.
        const Sense keepSense = mo.sense;
        const Real  keepOffset = mo.objOffset;
        mo = Model();
        mo.sense = keepSense;
        mo.objOffset = keepOffset;
        for (int j = 0; j < ncol; ++j) {
            double lo = collower ? collower[j] : 0.0;
            double up = colupper ? colupper[j] : kInf;
            double c  = obj ? obj[j] : 0.0;
            VarType t = VarType::Continuous;
            if (coltype) {
                if (coltype[j] == IGAOS_INTEGER) t = VarType::Integer;
                else if (coltype[j] == IGAOS_BINARY) t = VarType::Binary;
            }
            mo.addColumn(lo, up, c, t, "");
        }
        for (int i = 0; i < nrow; ++i) {
            double lo = rowlower ? rowlower[i] : -kInf;
            double up = rowupper ? rowupper[i] :  kInf;
            mo.addRow(lo, up, "");
        }
        for (int j = 0; j < ncol; ++j) {
            if (colptr[j] > colptr[j + 1])
                return fail(m, IGAOS_ERROR_VALUE, "colptr is not non-decreasing");
            for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
                if (rowidx[p] < 0 || rowidx[p] >= nrow)
                    return fail(m, IGAOS_ERROR_INDEX, "row index out of range in the matrix");
                mo.setElement(rowidx[p], j, value[p]);
            }
        }
        mo.finalize();
        m->solved = false;
        return IGAOS_OK;
    });
}

// ---- file interchange -----------------------------------------------------
int igaos_read_mps(igaos_model* m, const char* path) {
    return guarded(m, [&] {
        if (!path) return fail(m, IGAOS_ERROR_NULL, "null path");
        std::string err;
        Model mo;
        if (!readMps(path, mo, err))
            return fail(m, IGAOS_ERROR_IO, err.empty() ? "could not read the MPS file" : err.c_str());
        m->model = mo;
        m->solved = false;
        return IGAOS_OK;
    });
}

int igaos_write_mps(igaos_model* m, const char* path) {
    return guarded(m, [&] {
        if (!path) return fail(m, IGAOS_ERROR_NULL, "null path");
        m->model.finalize();
        std::string err;
        if (!writeMps(path, m->model, err))
            return fail(m, IGAOS_ERROR_IO, err.empty() ? "could not write the MPS file" : err.c_str());
        return IGAOS_OK;
    });
}

// ---- parameters -----------------------------------------------------------
int igaos_set_int_param(igaos_model* m, const char* name, long long v) {
    return guarded(m, [&] {
        if (!name) return fail(m, IGAOS_ERROR_NULL, "null parameter name");
        Options& o = m->opt;
        if      (nameIs(name, "verbosity"))          o.log.level = (int)v;
        else if (nameIs(name, "algorithm")) {
            if (v < 0 || v > 4) return fail(m, IGAOS_ERROR_VALUE, "algorithm out of range");
            o.lpAlgorithm = (LpAlgorithm)(int)v;
        }
        else if (nameIs(name, "presolve"))           o.presolve = v != 0;
        else if (nameIs(name, "scaling"))            o.scaling = v != 0;
        else if (nameIs(name, "crash"))              o.crash = v != 0;
        else if (nameIs(name, "crossover"))          o.crossover = v != 0;
        else if (nameIs(name, "cuts"))               o.cuts = v != 0;
        else if (nameIs(name, "cut_gomory"))         o.cutGomory = v != 0;
        else if (nameIs(name, "cut_cover"))          o.cutCover = v != 0;
        else if (nameIs(name, "cut_mir"))            o.cutMir = v != 0;
        else if (nameIs(name, "cut_rounds_root"))    o.cutRoundsRoot = (int)v;
        else if (nameIs(name, "heuristics"))         o.heuristics = v != 0;
        else if (nameIs(name, "reliability"))        o.reliability = (int)v;
        else if (nameIs(name, "threads"))            o.threads = (int)v;
        else if (nameIs(name, "refactor_frequency")) o.refactorFreq = (int)v;
        else if (nameIs(name, "iteration_limit"))    o.iterationLimit = (Long)v;
        else if (nameIs(name, "node_limit"))         o.nodeLimit = (Long)v;
        else if (nameIs(name, "memory_limit_mb"))    o.memoryLimitMb = (int)v;
        else if (nameIs(name, "ipm_max_iter"))       o.ipmMaxIter = (int)v;
        else if (nameIs(name, "pdhg_max_iter"))      o.pdhgMaxIter = (Long)v;
        else if (nameIs(name, "pdhg_restart"))       o.pdhgRestart = (int)v;
        else return fail(m, IGAOS_ERROR_PARAM, "unknown integer parameter");
        return IGAOS_OK;
    });
}

int igaos_get_int_param(igaos_model* m, const char* name, long long* out) {
    return guarded(m, [&] {
        if (!name || !out) return fail(m, IGAOS_ERROR_NULL, "null argument");
        const Options& o = m->opt;
        if      (nameIs(name, "verbosity"))          *out = o.log.level;
        else if (nameIs(name, "algorithm"))          *out = (int)o.lpAlgorithm;
        else if (nameIs(name, "presolve"))           *out = o.presolve;
        else if (nameIs(name, "scaling"))            *out = o.scaling;
        else if (nameIs(name, "crash"))              *out = o.crash;
        else if (nameIs(name, "crossover"))          *out = o.crossover;
        else if (nameIs(name, "cuts"))               *out = o.cuts;
        else if (nameIs(name, "cut_gomory"))         *out = o.cutGomory;
        else if (nameIs(name, "cut_cover"))          *out = o.cutCover;
        else if (nameIs(name, "cut_mir"))            *out = o.cutMir;
        else if (nameIs(name, "cut_rounds_root"))    *out = o.cutRoundsRoot;
        else if (nameIs(name, "heuristics"))         *out = o.heuristics;
        else if (nameIs(name, "reliability"))        *out = o.reliability;
        else if (nameIs(name, "threads"))            *out = o.threads;
        else if (nameIs(name, "refactor_frequency")) *out = o.refactorFreq;
        else if (nameIs(name, "iteration_limit"))    *out = (long long)o.iterationLimit;
        else if (nameIs(name, "node_limit"))         *out = (long long)o.nodeLimit;
        else if (nameIs(name, "memory_limit_mb"))    *out = o.memoryLimitMb;
        else if (nameIs(name, "ipm_max_iter"))       *out = o.ipmMaxIter;
        else if (nameIs(name, "pdhg_max_iter"))      *out = (long long)o.pdhgMaxIter;
        else if (nameIs(name, "pdhg_restart"))       *out = o.pdhgRestart;
        else return fail(m, IGAOS_ERROR_PARAM, "unknown integer parameter");
        return IGAOS_OK;
    });
}

int igaos_set_double_param(igaos_model* m, const char* name, double v) {
    return guarded(m, [&] {
        if (!name) return fail(m, IGAOS_ERROR_NULL, "null parameter name");
        Options& o = m->opt;
        if      (nameIs(name, "time_limit"))              o.timeLimit = v;
        else if (nameIs(name, "mip_gap_relative"))        o.tol.mipGapRel = v;
        else if (nameIs(name, "mip_gap_absolute"))        o.tol.mipGapAbs = v;
        else if (nameIs(name, "primal_tolerance"))        o.tol.primalFeas = v;
        else if (nameIs(name, "dual_tolerance"))          o.tol.dualFeas = v;
        else if (nameIs(name, "integrality_tolerance"))   o.tol.integrality = v;
        else if (nameIs(name, "pivot_tolerance"))         o.tol.pivot = v;
        else if (nameIs(name, "markowitz"))               o.tol.markowitz = v;
        else if (nameIs(name, "cutoff"))                  o.cutoff = v;
        else if (nameIs(name, "ipm_tolerance"))           o.ipmTol = v;
        else if (nameIs(name, "pdhg_tolerance"))          o.pdhgTol = v;
        else return fail(m, IGAOS_ERROR_PARAM, "unknown double parameter");
        return IGAOS_OK;
    });
}

int igaos_get_double_param(igaos_model* m, const char* name, double* out) {
    return guarded(m, [&] {
        if (!name || !out) return fail(m, IGAOS_ERROR_NULL, "null argument");
        const Options& o = m->opt;
        if      (nameIs(name, "time_limit"))            *out = o.timeLimit;
        else if (nameIs(name, "mip_gap_relative"))      *out = o.tol.mipGapRel;
        else if (nameIs(name, "mip_gap_absolute"))      *out = o.tol.mipGapAbs;
        else if (nameIs(name, "primal_tolerance"))      *out = o.tol.primalFeas;
        else if (nameIs(name, "dual_tolerance"))        *out = o.tol.dualFeas;
        else if (nameIs(name, "integrality_tolerance")) *out = o.tol.integrality;
        else if (nameIs(name, "pivot_tolerance"))       *out = o.tol.pivot;
        else if (nameIs(name, "markowitz"))             *out = o.tol.markowitz;
        else if (nameIs(name, "cutoff"))                *out = o.cutoff;
        else if (nameIs(name, "ipm_tolerance"))         *out = o.ipmTol;
        else if (nameIs(name, "pdhg_tolerance"))        *out = o.pdhgTol;
        else return fail(m, IGAOS_ERROR_PARAM, "unknown double parameter");
        return IGAOS_OK;
    });
}

// ---- solve ----------------------------------------------------------------
int igaos_solve(igaos_model* m) {
    return guarded(m, [&] {
        m->model.finalize();
        m->solver.opt = m->opt;
        m->solution = m->solver.solve(m->model);
        m->algorithm = m->solution.algorithm;
        m->solved = true;
        return IGAOS_OK;
    });
}

// ---- results --------------------------------------------------------------
int igaos_get_status(const igaos_model* m) {
    if (!m || !m->solved) return IGAOS_STATUS_NOT_SOLVED;
    return (int)m->solution.status;
}

const char* igaos_status_string(int status) {
    return statusName((Status)status);
}

double igaos_get_objective(const igaos_model* m)  { return (m && m->solved) ? m->solution.objective : 0.0; }
double igaos_get_best_bound(const igaos_model* m) { return (m && m->solved) ? m->solution.bestBound : 0.0; }
double igaos_get_mip_gap(const igaos_model* m)    { return (m && m->solved) ? m->solution.mipGap : 0.0; }
double igaos_get_solve_time(const igaos_model* m) { return (m && m->solved) ? m->solution.solveTime : 0.0; }
long long igaos_get_iterations(const igaos_model* m) { return (m && m->solved) ? (long long)m->solution.iterations : 0; }
long long igaos_get_nodes(const igaos_model* m)      { return (m && m->solved) ? (long long)m->solution.nodes : 0; }
int igaos_get_num_rows(const igaos_model* m) { return m ? (int)m->model.numRow() : 0; }
int igaos_get_num_cols(const igaos_model* m) { return m ? (int)m->model.numCol() : 0; }

namespace {
int copyOut(const igaos_model* m, const std::vector<Real>& src, double* dst, size_t expect) {
    if (!m) return IGAOS_ERROR_NULL;
    if (!m->solved) return IGAOS_ERROR_NOT_SOLVED;
    if (!dst) return IGAOS_ERROR_NULL;
    if (src.size() < expect) return IGAOS_ERROR_INTERNAL;
    for (size_t i = 0; i < expect; ++i) dst[i] = src[i];
    return IGAOS_OK;
}
} // namespace

int igaos_get_solution(const igaos_model* m, double* x) {
    return copyOut(m, m ? m->solution.colValue : std::vector<Real>(), x,
                   m ? (size_t)m->model.numCol() : 0);
}
int igaos_get_reduced_costs(const igaos_model* m, double* d) {
    return copyOut(m, m ? m->solution.colDual : std::vector<Real>(), d,
                   m ? (size_t)m->model.numCol() : 0);
}
int igaos_get_row_activity(const igaos_model* m, double* a) {
    return copyOut(m, m ? m->solution.rowValue : std::vector<Real>(), a,
                   m ? (size_t)m->model.numRow() : 0);
}
int igaos_get_row_duals(const igaos_model* m, double* y) {
    return copyOut(m, m ? m->solution.rowDual : std::vector<Real>(), y,
                   m ? (size_t)m->model.numRow() : 0);
}

// ---- diagnostics ----------------------------------------------------------
int igaos_get_cut_counts(const igaos_model* m, int* applied, int* gomory,
                         int* cover, int* mir, int* rounds) {
    if (!m) return IGAOS_ERROR_NULL;
    if (!m->solved) return IGAOS_ERROR_NOT_SOLVED;
    const SolveReport& r = m->solver.report;
    if (applied) *applied = (int)r.cutsApplied;
    if (gomory)  *gomory  = (int)r.cutsGomory;
    if (cover)   *cover   = (int)r.cutsCover;
    if (mir)     *mir     = (int)r.cutsMir;
    if (rounds)  *rounds  = (int)r.cutRounds;
    return IGAOS_OK;
}

double igaos_get_root_bound_before_cuts(const igaos_model* m) {
    return (m && m->solved) ? m->solver.report.rootBoundLp : 0.0;
}
double igaos_get_root_bound_after_cuts(const igaos_model* m) {
    return (m && m->solved) ? m->solver.report.rootBoundCut : 0.0;
}

const char* igaos_get_algorithm(const igaos_model* m) {
    return (m && m->solved) ? m->algorithm.c_str() : "";
}

const char* igaos_last_error(const igaos_model* m) {
    return m ? m->error.c_str() : "";
}

} // extern "C"
