// model.hpp : the computational form IGAOS optimizes over.
//
//      minimize    c^T x + 1/2 x^T Q x + objOffset
//      subject to  rowLower <= A x + q_i(x) <= rowUpper
//                  colLower <=  x  <= colUpper
//                  x_j integral for j in the integer set
//
// Ranged rows subsume =, <= and >= rows, so the solver has exactly one row
// form to reason about.  Q is the lower triangle of the objective Hessian; for
// the convex QP path it is positive semidefinite, and the global path in
// src/global.cpp accepts it indefinite.
//
// q_i(x) is the OPTIONAL quadratic part of row i -- a sum of coef * x_a * x_b
// terms, held in `qcon` rather than as a matrix per row, because a refinery
// pooling model has a handful of bilinear terms in each of many rows and a
// matrix per row would be almost entirely zeros.  A model with any of these is
// a QCQP: nonconvex in general, and solved to proven global optimality by
// spatial branch and bound over McCormick relaxations.
#pragma once
#include "igaos/sparse.hpp"
#include <map>

namespace igaos {

class Model {
public:
    std::string name = "igaos_model";
    Sense       sense = Sense::Minimize;
    Real        objOffset = 0.0;

    SparseMatrix A;                 // nrow x ncol constraint matrix
    SparseMatrix Q;                 // ncol x ncol, lower triangle, empty for LP

    // One bilinear or square term inside a row: coef * x[i] * x[j].  i == j is
    // a square.  Terms are stored in whatever order they were added; nothing
    // depends on the order, and duplicates simply sum.
    struct QuadTerm {
        Int  row = kNone;
        Int  i = kNone, j = kNone;
        Real coef = 0.0;
    };
    std::vector<QuadTerm> qcon;
    std::vector<Real>    obj;       // linear objective, length ncol
    std::vector<Real>    colLower, colUpper;
    std::vector<Real>    rowLower, rowUpper;
    std::vector<VarType> colType;
    std::vector<std::string> colName, rowName;

    Int numCol() const { return (Int)obj.size(); }
    Int numRow() const { return (Int)rowLower.size(); }
    Int numInt() const {
        Int k = 0; for (VarType t : colType) if (t != VarType::Continuous) ++k; return k;
    }
    bool isMip() const { return numInt() > 0; }
    bool isQp()  const { return Q.nnz() > 0; }
    // A quadratically constrained model.  Distinct from isQp(): a quadratic
    // OBJECTIVE with a positive semidefinite Q is a convex problem the interior
    // point method solves directly, whereas a quadratic CONSTRAINT is nonconvex
    // in general and needs the global path whatever the sign of its coefficient.
    bool isQcqp() const { return !qcon.empty(); }
    // True when the objective Hessian is not positive semidefinite, tested by
    // attempting a Cholesky factorization of the dense symmetric form.  A
    // nonconvex objective makes the QP relaxation useless as a bound, which is
    // why the convex path refuses one and the global path takes it.
    bool hasNonconvexObjective() const;

    // ---- construction helpers (also the basis of the C API / Python layer) --
    Int addColumn(Real lo, Real up, Real cost, VarType type = VarType::Continuous,
                  const std::string& nm = "");
    Int addRow(Real lo, Real up, const std::string& nm = "");
    void setElement(Int row, Int col, Real v);      // buffered; call finalize()
    void setQuadratic(Int i, Int j, Real v);        // buffered; lower triangle
    // Add coef * x[i] * x[j] to row `row`.  Not buffered: quadratic terms are
    // few and are kept as a list, so there is nothing to compress.
    void addQuadraticTerm(Int row, Int i, Int j, Real coef);
    void finalize();                                // build A and Q from buffers

    void ensureNames();
    void validate() const;

    // Objective value of a candidate point in the *user's* sense.
    Real objectiveValue(const std::vector<Real>& x) const;

    // Row activities.  A*x plus the quadratic part of each row, so a caller
    // checking feasibility of a QCQP sees the TRUE activity and not the linear
    // part of it -- which is the difference between a feasible point and a
    // point that only looks feasible to a relaxation.
    void rowActivity(const std::vector<Real>& x, std::vector<Real>& act) const {
        A.multiply(x, act);
        for (const QuadTerm& t : qcon)
            act[t.row] += t.coef * x[t.i] * x[t.j];
    }

    // Maximum primal infeasibility of a candidate point.
    Real primalInfeasibility(const std::vector<Real>& x) const;
    Real integerInfeasibility(const std::vector<Real>& x, Real tol) const;

    // Diagnostics reported in the log and used to pick default algorithms.
    struct Stats {
        Int  nrow = 0, ncol = 0, nnz = 0, nint = 0, nbin = 0, nqnz = 0;
        Int  nqcon = 0;         // bilinear/square terms inside rows
        Int  nqconRows = 0;     // how many rows carry at least one
        Real density = 0, minAbs = 0, maxAbs = 0, ratio = 0;
        Int  nEqRows = 0, nRangeRows = 0, nFreeRows = 0;
        Int  maxColLen = 0, maxRowLen = 0;
    };
    Stats stats() const;

    void toMaximizationNegated();   // internal: always minimize
private:
    TripletBuilder abuf_, qbuf_;
    bool dirty_ = false;
};

// ---------------------------------------------------------------------------
// Solution container.  `basis` is populated by the simplex path so that the
// caller (and branch-and-cut) can warm start.
// ---------------------------------------------------------------------------
struct Solution {
    Status status = Status::NotSolved;
    Real   objective = 0.0;
    Real   dualObjective = 0.0;
    std::vector<Real> colValue, colDual;      // x, reduced costs
    std::vector<Real> rowValue, rowDual;      // Ax, y
    std::vector<VarStatus> colStatus, rowStatus;
    Long   iterations = 0;
    Long   nodes = 0;
    double solveTime = 0.0;
    Real   mipGap = 0.0;
    Real   bestBound = 0.0;
    Real   primalInf = 0.0, dualInf = 0.0;
    std::string algorithm;

    void resize(Int m, Int n) {
        colValue.assign(n, 0.0); colDual.assign(n, 0.0);
        rowValue.assign(m, 0.0); rowDual.assign(m, 0.0);
        colStatus.assign(n, VarStatus::AtLower);
        rowStatus.assign(m, VarStatus::Basic);
    }
};

// ---------------------------------------------------------------------------
struct Options {
    Tolerances tol;
    Logger     log;
    LpAlgorithm lpAlgorithm = LpAlgorithm::Auto;
    GpuMode    gpu = GpuMode::Auto;

    double timeLimit      = 1e30;
    Long   iterationLimit = 1000000000LL;
    Long   nodeLimit      = 1000000000LL;
    // Ceiling on the memory a branch-and-bound solve may reach, in megabytes.
    // A tree that cannot close keeps growing its open list, and a solver killed
    // by the kernel with no output is far worse than one that stops and hands
    // back the incumbent it has with an honest bound.  Hitting it is reported
    // exactly like a time limit: feasible, not proven.
    //
    //   -1  automatic -- 60% of the memory the machine reports as available
    //       when the solve starts, clamped to [512, 16384] MB.  A fixed default
    //       is wrong on both ends: 2 GB cuts a solve short on a workstation and
    //       does not save a 1 GB container.
    //    0  no limit.  The process may be killed instead.
    //   >0  that many megabytes.
    int    memoryLimitMb  = -1;

    bool   presolve       = true;
    bool   scaling        = true;
    // Equilibrate [A; Q] together rather than scaling from A alone.  Helps the
    // interior-point path on quadratic models whose Q dominates A.  Off by
    // default so recorded benchmark results stay reproducible.
    bool   scaleWithQ     = false;
    // Refuse to call a point optimal when the primal and dual objectives
    // disagree.  relP and relD each divide by a norm that grows with the model,
    // so on a large badly conditioned QP both can be tiny while the answer is
    // wrong; the primal-dual gap has no such denominator.  Defect 27.
    // On by default since 22 September 2026.  Measured cost: Netlib LP 104/114
    // unchanged, zero refused; Maros-Meszaros 108 -> 101, and all 7 refused were
    // either recorded objective mismatches against OSQP or instances OSQP never
    // finished.  None of the 65 that agree with OSQP was refused.  Reporting
    // seven fewer optima is the right trade for never reporting a wrong one.
    bool   dualityGapCheck = true;
    bool   crash          = true;      // build a triangular starting basis
    int    refactorFreq   = 100;       // simplex updates between refactorizations
    int    threads        = 0;         // 0 = all available

    // MIP controls
    bool   cuts           = true;
    bool   cutGomory      = true;    // Gomory mixed-integer cuts
    bool   cutCover       = true;    // knapsack cover cuts
    bool   cutMir         = true;    // complemented mixed-integer rounding cuts
    int    cutRoundsRoot  = 12;
    // Optional verification point for cut separation: a vector of structural
    // column values known to be feasible for the mixed-integer problem.  See
    // CutLimits::referencePoint.  Null in normal use.
    const std::vector<Real>* cutReference = nullptr;
    int    cutRoundsNode  = 2;
    bool   heuristics     = true;
    int    reliability    = 4;         // pseudocost reliability threshold
    Real   cutoff         = kInf;

    // IPM controls
    int    ipmMaxIter     = 200;
    Real   ipmTol         = 1e-8;
    // Turn the interior point solution into a basic one.  Off means the caller
    // gets the interior point itself: optimal, but with no basis, so it cannot
    // warm start branch and cut and its "reduced costs" are barrier multipliers
    // rather than simplex reduced costs.
    bool   crossover      = true;

    // PDHG controls
    Long   pdhgMaxIter    = 500000;
    Real   pdhgTol        = 1e-8;
    int    pdhgRestart    = 64;
};

} // namespace igaos
