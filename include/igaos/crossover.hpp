// crossover.hpp : turn an interior point into an optimal *basic* solution.
//
// An interior point method converges to a point in the relative interior of the
// optimal face.  That point is optimal, but it is not a vertex, so it carries no
// basis -- and without a basis there is nothing for branch and cut to warm start
// from, no reduced costs in the usual sense, and no exact identification of
// which constraints are active.  Crossover is what makes the interior point path
// usable by the rest of the solver rather than a dead end.
//
// The method here is basis identification followed by a simplex clean-up:
//
//   1. Rank every variable -- structural and logical alike -- by how far the
//      interior point leaves it from its nearest bound.  Complementarity says
//      that a variable the barrier has driven hard against a bound is nonbasic
//      there, and one left strictly interior belongs in the basis.
//   2. Take the m most interior variables as the candidate basis and push every
//      other variable onto its nearer bound.
//   3. Hand the assignment to the simplex, which repairs any linear dependence
//      by swapping in logical variables, and re-optimizes.  From a correctly
//      identified basis this costs a handful of iterations.
//
// This is the practical identification-and-clean-up crossover, not a
// Megiddo-style strongly-polynomial push sequence.  What it guarantees is the
// part that matters downstream: on return the solution is basic, primal and dual
// feasible, and optimal to the simplex's own tolerances rather than the interior
// point method's.
#pragma once
#include "igaos/model.hpp"
#include "igaos/ipm.hpp"

namespace igaos {

struct CrossoverResult {
    Status status = Status::NotSolved;
    Long   iterations = 0;      // simplex iterations spent cleaning up
    Int    pushes = 0;          // strictly interior variables forced to a bound
    Int    repaired = 0;        // candidates rejected as linearly dependent
    Real   objective = 0.0;
    Real   primalInfeasibility = 0.0, dualInfeasibility = 0.0;

    std::vector<Real> colValue, colDual, rowValue, rowDual;
    std::vector<VarStatus> colStatus, rowStatus;
};

CrossoverResult crossover(const Model& m, const Options& opt, const IpmResult& ip);

} // namespace igaos
