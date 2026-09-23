// IGAOS -- Indigenous GPU-Accelerated Optimization Solver
// common.hpp : fundamental types, tolerances, status codes, timing, logging.
//
// Copyright (c) 2026 the IGAOS team. See LICENSE at the repository root.
//
// Design note: everything in IGAOS is built from the mathematical
// foundation up. No third-party optimization library is linked, and the
// numerical linear algebra (sparse LU, sparse LDL^T, orderings) is our own.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
#include <chrono>
#include <cstdio>
#include <cstdarg>
#include <functional>

namespace igaos {

using Int  = int32_t;
using Long = int64_t;
using Real = double;

constexpr Real kInf     = 1e30;          // anything >= kInf is "infinite"
constexpr Real kBigReal = std::numeric_limits<Real>::max();
constexpr Int  kNone    = -1;

inline bool isInf(Real v)     { return v >=  kInf; }
inline bool isNegInf(Real v)  { return v <= -kInf; }
inline bool isFinite(Real v)  { return v > -kInf && v < kInf; }

// ---------------------------------------------------------------------------
// Tolerances.  Grouped so that a single struct can be tuned per problem class
// (refinery blending models, for example, want a tighter dual feasibility
// tolerance than the default, and a looser Markowitz threshold).
// ---------------------------------------------------------------------------
struct Tolerances {
    Real primalFeas   = 1e-7;   // |row activity violation|
    Real dualFeas     = 1e-7;   // reduced-cost sign violation
    Real pivot        = 1e-9;   // smallest acceptable pivot magnitude
    Real markowitz    = 0.01;   // threshold pivoting factor in sparse LU
    Real zero         = 1e-12;  // numeric drop tolerance
    Real integrality  = 1e-6;   // |x - round(x)| below this counts as integral
    Real mipGapAbs    = 1e-9;
    Real mipGapRel    = 1e-6;
    Real harris       = 1e-9;   // Harris ratio-test relaxation
    Real degenerate   = 1e-10;
};

// ---------------------------------------------------------------------------
enum class Status : int {
    NotSolved      = 0,
    Optimal        = 1,
    Infeasible     = 2,
    Unbounded      = 3,
    IterationLimit = 4,
    TimeLimit      = 5,
    NodeLimit      = 6,
    NumericalError = 7,
    Interrupted    = 8,
    Feasible       = 9    // MIP: incumbent found, optimality not proven
};

const char* statusName(Status s);

enum class Sense : int { Minimize = 1, Maximize = -1 };

// Variable status in a simplex basis.
enum class VarStatus : uint8_t {
    Basic     = 0,
    AtLower   = 1,
    AtUpper   = 2,
    AtZero    = 3,   // free nonbasic sitting at zero
    Fixed     = 4
};

enum class VarType : uint8_t { Continuous = 0, Integer = 1, Binary = 2 };

// ---------------------------------------------------------------------------
// Which continuous algorithm to use.
enum class LpAlgorithm : int {
    Auto        = 0,
    PrimalSimplex = 1,
    DualSimplex   = 2,
    InteriorPoint = 3,
    PDHG          = 4   // first-order, GPU-accelerated
};

enum class GpuMode : int { Off = 0, Auto = 1, Force = 2 };

// ---------------------------------------------------------------------------
struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0{Clock::now()};
    void   reset()          { t0 = Clock::now(); }
    double elapsed() const  {
        return std::chrono::duration<double>(Clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Logging: verbosity 0 = silent, 1 = summary, 2 = per-phase, 3 = per-iteration,
// 4 = numerical diagnostics.
struct Logger {
    int level = 1;
    FILE* out = stdout;

    // Machine-readable stage events, off unless a caller asks for them.
    // The human log is written when a phase FINISHES, because that is when its
    // numbers exist; a caller that wants to show what the solver is doing WHILE
    // it does it needs events at the moments themselves, which is what this is.
    // One line, flushed, easy to parse, and impossible to confuse with the
    // human log because of the prefix.
    //
    //   IGAOS_STAGE presolve begin
    //   IGAOS_STAGE presolve end rows=3 cols=0 nnz=3 tightened=173 t=0.0004
    //
    bool progress = false;

    // A caller that wants to RENDER these events rather than read them -- the
    // command line's live pipeline, a browser console, a future IDE plugin --
    // sets this and receives every event as it happens, already formatted.
    // Without it such a caller has to re-parse our own stdout, which means a
    // second implementation of the protocol that can drift from the first.
    // The web console did exactly that; the terminal view does not have to.
    std::function<void(const char* name, const char* state, const char* detail)> onStage;

    // True when anybody is listening. The branch-and-bound heartbeat is behind
    // this rather than behind `progress` alone, because a caller that installed
    // a renderer wants the node stream just as much as one that asked for the
    // text protocol -- and gating on `progress` silently starved it.
    bool wantsEvents() const { return progress || (bool)onStage; }

    void log(int lvl, const char* fmt, ...) const {
        if (lvl > level || !out) return;
        va_list ap; va_start(ap, fmt);
        std::vfprintf(out, fmt, ap);
        va_end(ap);
        std::fflush(out);
    }

    void stage(const char* name, const char* state, const char* fmt = nullptr, ...) const {
        if (!progress && !onStage) return;
        char detail[512];
        detail[0] = '\0';
        if (fmt) {
            va_list ap; va_start(ap, fmt);
            std::vsnprintf(detail, sizeof detail, fmt, ap);
            va_end(ap);
        }
        if (onStage) onStage(name, state, detail);
        if (progress && out) {
            std::fprintf(out, "IGAOS_STAGE %s %s", name, state);
            if (detail[0]) { std::fputc(' ', out); std::fputs(detail, out); }
            std::fputc('\n', out);
            std::fflush(out);
        }
    }
};

// ---------------------------------------------------------------------------
inline Real relGap(Real primal, Real dual) {
    Real d = std::fabs(primal);
    if (d < 1.0) d = 1.0;
    return std::fabs(primal - dual) / d;
}

inline Real fracPart(Real v) { return v - std::floor(v); }

inline bool isIntegral(Real v, Real tol) {
    return std::fabs(v - std::floor(v + 0.5)) <= tol;
}

} // namespace igaos
