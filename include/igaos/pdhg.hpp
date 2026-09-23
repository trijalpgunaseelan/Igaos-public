// pdhg.hpp : restarted primal-dual hybrid gradient -- the first-order path, and
// the one that is actually a GPU workload.
//
// The saddle-point form.  With  min c'x  s.t.  Ax in [ls, us],  x in [l, u],
// the indicator of the row box is dualized:
//
//      L(x, y) = c'x + y'Ax - sigma_C(y),      sigma_C(y) = sup_{s in C} y's
//
// and PDHG alternates a projected gradient step in x with a proximal ascent
// step in y:
//
//      x+ = proj_[l,u] ( x - tau (c + A'y) )
//      y+ = prox_{sigma * sigma_C} ( y + sigma A (2x+ - x) )
//
// For a box C the proximal operator is available in closed form through Moreau's
// identity, prox_{a sigma_C}(v) = v - a proj_C(v / a), so a full iteration costs
// exactly one A'y and one Ax and nothing else.
//
// WHY THIS IS THE GPU PATH, AND THE FACTORIZATIONS ARE NOT
// --------------------------------------------------------
// Sparse LU and sparse Cholesky are dominated by a sequential elimination tree:
// the work per node is small, the dependencies are long and irregular, and fill
// grows as elimination proceeds.  That structure does not map onto a GPU -- the
// device spends its time waiting on dependencies rather than computing, and
// published sparse-direct GPU results are dominated by the dense supernodes,
// which industrial LP bases do not have.  PDHG has no factorization at all.  Its
// only kernel is a sparse matrix-vector product, which is bandwidth-bound,
// perfectly parallel across rows, and exactly what GPU memory systems are built
// for.  That is the whole argument for the GPU claim, and it is a claim about
// this method, not about the solver as a whole.
//
// What this path buys is moderate accuracy on very large problems where a
// factorization would not fit or would fill in catastrophically.  It does not
// replace the simplex: first-order methods converge slowly in the final digits,
// which is why the driver runs crossover afterwards when an exact basic solution
// is wanted.
#pragma once
#include "igaos/model.hpp"

namespace igaos {

struct PdhgResult {
    Status status = Status::NotSolved;
    Long   iterations = 0;
    Int    restarts = 0;
    Real   primalObjective = 0.0, dualObjective = 0.0;
    Real   primalInfeasibility = 0.0, dualInfeasibility = 0.0, relativeGap = 0.0;
    Real   matrixNorm = 0.0;          // estimate of ||A||_2 from power iteration
    Real   primalWeight = 1.0;
    std::vector<Real> x, s, y;
};

// Largest singular value of A, by power iteration on A'A.  Only SpMV is used,
// which is why the step-size estimate runs on the same kernel as the solver.
Real spectralNormEstimate(const SparseMatrix& A, int maxIter = 200, Real tol = 1e-4);

PdhgResult primalDualHybridGradient(const Model& m, const Options& opt);

} // namespace igaos
