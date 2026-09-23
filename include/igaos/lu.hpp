// lu.hpp : sparse LU factorization of a simplex basis, with threshold
// Markowitz pivoting and product-form (eta) updates between refactorizations.
//
// The basis B is m x m and its columns are drawn from the *extended* matrix
//        [ A   -I ]
// so column index j < n selects structural column A(:,j) and column index
// n + i selects the logical (slack) column -e_i.  This is the standard
// computational form and lets every row be a ranged row.
//
// Factorization computes  P B Q = L U  with L unit lower triangular and U
// upper triangular in the permuted ordering.  L is stored by column and U by
// row, which lets all four triangular solves (L, L^T, U, U^T) run as
// column-oriented sweeps with no transposed data structure.
#pragma once
#include "igaos/sparse.hpp"

namespace igaos {

class ExtendedMatrix {
public:
    const SparseMatrix* A = nullptr;
    Int m = 0, n = 0;                      // rows, structural columns
    void bind(const SparseMatrix& a) { A = &a; m = a.nrow; n = a.ncol; }
    Int  total() const { return n + m; }
    bool isLogical(Int j) const { return j >= n; }

    // Scatter column j into (idx,val) pairs appended to the given arrays.
    template <class F> void forEach(Int j, F&& f) const {
        if (j < n) { for (Int p = A->colPtr[j]; p < A->colPtr[j + 1]; ++p) f(A->rowIdx[p], A->val[p]); }
        else       { f(j - n, -1.0); }
    }
    Int length(Int j) const { return j < n ? A->colLen(j) : 1; }
};

class BasisFactor {
public:
    Int m = 0;

    // --- factorization ------------------------------------------------------
    // Returns false on singularity; on failure singularRows/singularCols report
    // which basis positions could not be pivoted so the caller can repair the
    // basis with logical columns.
    bool factorize(const ExtendedMatrix& E, const std::vector<Int>& basis,
                   const Tolerances& tol);

    std::vector<Int> singularPositions;   // basis positions left unpivoted

    // --- solves (in place on a dense work vector of length m) ---------------
    void ftranDense(std::vector<Real>& v) const;   // v <- B^{-1} v
    void btranDense(std::vector<Real>& v) const;   // v <- B^{-T} v

    // Sparse interface: pattern-tracked results used by the simplex hot loop.
    void ftran(SparseVector& v, Real dropTol) const;
    void btran(SparseVector& v, Real dropTol) const;

    // --- product-form update ------------------------------------------------
    // alphaQ must be B^{-1} a_q expressed in basis-position coordinates and
    // pivotPos the basis position leaving.  Returns false if the pivot is too
    // small, in which case the caller should refactorize.
    bool update(Int pivotPos, const SparseVector& alphaQ, Real pivotTol);

    Int  numUpdates() const { return (Int)etas_.size(); }
    void clearUpdates() { etas_.clear(); etaNnz_ = 0; }
    Long etaNonzeros() const { return etaNnz_; }
    Long factorNonzeros() const { return lnz_ + unz_; }
    Real growthFactor() const { return growth_; }
    Real smallestPivot() const { return minPivot_; }
    Real conditionEstimate() const;

private:
    // L by column, U by row, both in pivot order.
    std::vector<Int>  lPtr_, lIdx_;
    std::vector<Real> lVal_;
    std::vector<Int>  uPtr_, uIdx_;
    std::vector<Real> uVal_;
    std::vector<Real> uDiag_;
    std::vector<Int>  pivotRow_, pivotCol_;   // step k -> original row / basis pos
    std::vector<Int>  rowPos_, colPos_;       // original row / basis pos -> step k

    struct Eta { Int pivot; std::vector<Int> idx; std::vector<Real> val; Real pivotVal; };
    std::vector<Eta> etas_;
    Long etaNnz_ = 0, lnz_ = 0, unz_ = 0;
    Real growth_ = 1.0, minPivot_ = 1.0;

    mutable std::vector<Real> work_;

    void applyEtasForward(std::vector<Real>& v) const;
    void applyEtasBackward(std::vector<Real>& v) const;
};

} // namespace igaos
