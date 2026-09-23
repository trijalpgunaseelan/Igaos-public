// ipm.hpp : primal-dual interior point method, Mehrotra predictor-corrector.
//
// The problem is held in the same computational form as the simplex:
//
//      minimize    c'x + 1/2 x'Qx
//      subject to  A x - s = 0
//                  lx <= x <= ux,   ls <= s <= us
//
// so ranged, equality, one-sided and free rows are all the same object -- a
// bounded logical variable -- and the method never has to special-case a row
// type, only a *bound pattern*.
//
// Each Newton step solves the augmented KKT system
//
//      [ -(Q + Dx)   A'  ] [ dx ]   [ rx ]
//      [    A        Dy  ] [ dy ] = [ ry ]
//
// with Dx >= 0 from the primal-dual barrier terms on x and Dy > 0 obtained by
// eliminating ds.  This is the *quasi-definite* form, and that matters for two
// reasons.  First, a quasi-definite matrix admits an LDL' factorization for
// every symmetric permutation, so the fill-reducing ordering is computed once,
// from the pattern alone, and reused for every iteration -- the numerical phase
// never has to reorder.  Second, it is why the same factorization kernel serves
// LP and convex QP: Q enters the (1,1) block additively and changes nothing
// structural.
//
// The normal-equations form A(Q + Dx)^-1 A' is deliberately NOT used.  For an
// LP it is often the faster choice, but (Q + Dx)^-1 is dense whenever Q is not
// diagonal, so the normal equations cannot serve QP at all -- and a solver that
// switches formulations between problem classes needs two kernels, two sets of
// numerical failure modes, and two things to get right.
#pragma once
#include "igaos/model.hpp"
#include "igaos/ldl.hpp"

namespace igaos {

struct IpmResult {
    Status status = Status::NotSolved;
    Int    iterations = 0;
    Real   primalObjective = 0.0;
    Real   dualObjective = 0.0;
    Real   primalInfeasibility = 0.0;
    Real   dualInfeasibility = 0.0;
    Real   complementarityGap = 0.0;
    // |primal - dual| / (1 + |primal| + |dual|), measured in the IPM's own
    // space where both objectives are directly comparable.  This is the one
    // acceptance test that cannot be flattered by a large norm in a
    // denominator -- see defect 27.
    Real   relDualityGap = -1.0;   // negative: not measured on this path
    Int    regularizedPivots = 0;
    Long   factorNonzeros = 0;
    Real   analyzeTime = 0.0, factorTime = 0.0, solveTime = 0.0;

    std::vector<Real> x, s, y;        // primal, row activity, row duals
    std::vector<Real> zLower, zUpper; // bound multipliers on x
};

// Solve the continuous relaxation with the interior point method.
// The model must already be finalized; presolve and scaling are the caller's
// business, exactly as for the simplex path.
IpmResult interiorPoint(const Model& m, const Options& opt);

} // namespace igaos
