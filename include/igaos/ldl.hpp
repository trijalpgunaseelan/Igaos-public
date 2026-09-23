// ldl.hpp : fill-reducing ordering and sparse LDL^T factorization for the
// quasi-definite KKT systems solved by the interior-point method.
//
//      K = [ -(Q + Dx)   A^T ]
//          [    A         Dy ]
//
// with Dx > 0 and Dy > 0.  Such a matrix is *quasi-definite*, which means an
// LDL^T factorization exists for **every** symmetric permutation -- so the
// ordering can be chosen purely to reduce fill, with numerical stability
// recovered through dynamic regularization plus iterative refinement.
// That property is what makes a single factorization routine serve LP and
// convex QP alike.
#pragma once
#include "igaos/sparse.hpp"

namespace igaos {

// Approximate minimum degree ordering on a symmetric pattern supplied as the
// full (both triangles, no diagonal) adjacency structure.
void approximateMinimumDegree(Int n,
                              const std::vector<Int>& Ap, const std::vector<Int>& Ai,
                              std::vector<Int>& perm, std::vector<Int>& iperm);

class LdlFactor {
public:
    Int n = 0;

    // Symbolic phase: pattern of the *upper triangle* including the diagonal,
    // in the original ordering.  Computes the ordering, elimination tree and
    // column counts once; the numeric phase then reuses them each IPM iteration.
    void analyze(Int n, const std::vector<Int>& Ap, const std::vector<Int>& Ai);

    // Numeric phase.  `expectedSign[i]` is +1 or -1 and states the sign the
    // pivot for original index i must have; a pivot that comes out too small or
    // with the wrong sign is replaced by `sign * regularization`, and the
    // affected indices are recorded so the caller can compensate with iterative
    // refinement.
    bool factorize(const std::vector<Int>& Ap, const std::vector<Int>& Ai,
                   const std::vector<Real>& Ax,
                   const std::vector<int8_t>& expectedSign,
                   Real regularization, Real pivotTol);

    void solve(std::vector<Real>& b) const;      // in place, original ordering

    Long   nonzeros() const { return Lp_.empty() ? 0 : Lp_.back(); }
    Int    numRegularized() const { return nreg_; }
    Real   minAbsPivot() const { return minPivot_; }
    const std::vector<Int>& permutation() const { return perm_; }

private:
    std::vector<Int>  perm_, iperm_, parent_, Lnz_, Lp_, Li_;
    std::vector<Real> Lx_, D_;
    mutable std::vector<Real> y_, work_;
    mutable std::vector<Int>  pattern_, flag_;
    Int  nreg_ = 0;
    Real minPivot_ = 0.0;
};

} // namespace igaos
