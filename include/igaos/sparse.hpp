// sparse.hpp : compressed-sparse-column matrix and the sparse kernels the
// rest of the solver is built on.  Written from scratch; no external BLAS or
// sparse library is used.
#pragma once
#include "igaos/common.hpp"
#include <algorithm>
#include <cstring>

namespace igaos {

// ---------------------------------------------------------------------------
// SparseMatrix: CSC storage.  colPtr has ncol+1 entries; rowIdx/val have nnz.
// Rows within a column are kept sorted ascending, which the LU factorization
// and the cut generators both rely on.
// ---------------------------------------------------------------------------
class SparseMatrix {
public:
    Int nrow = 0, ncol = 0;
    std::vector<Int>  colPtr;
    std::vector<Int>  rowIdx;
    std::vector<Real> val;

    SparseMatrix() : colPtr(1, 0) {}
    SparseMatrix(Int m, Int n) : nrow(m), ncol(n), colPtr(n + 1, 0) {}

    Int  nnz()          const { return colPtr.empty() ? 0 : colPtr.back(); }
    Int  colStart(Int j) const { return colPtr[j]; }
    Int  colEnd(Int j)   const { return colPtr[j + 1]; }
    Int  colLen(Int j)   const { return colPtr[j + 1] - colPtr[j]; }

    void clear() { nrow = ncol = 0; colPtr.assign(1, 0); rowIdx.clear(); val.clear(); }

    void reserve(Int n, Int nz) {
        colPtr.reserve(n + 1); rowIdx.reserve(nz); val.reserve(nz);
    }

    // y += alpha * A(:,j)
    void addCol(Int j, Real alpha, std::vector<Real>& y) const {
        for (Int p = colPtr[j]; p < colPtr[j + 1]; ++p) y[rowIdx[p]] += alpha * val[p];
    }

    // returns A(:,j)' * x
    Real dotCol(Int j, const std::vector<Real>& x) const {
        Real s = 0;
        for (Int p = colPtr[j]; p < colPtr[j + 1]; ++p) s += val[p] * x[rowIdx[p]];
        return s;
    }

    // y = A * x    (dense in/out)
    void multiply(const std::vector<Real>& x, std::vector<Real>& y) const;
    // y = A^T * x
    void multiplyTranspose(const std::vector<Real>& x, std::vector<Real>& y) const;
    // y += alpha * A * x
    void multiplyAdd(Real alpha, const std::vector<Real>& x, std::vector<Real>& y) const;
    // y += alpha * A^T * x
    void multiplyTransposeAdd(Real alpha, const std::vector<Real>& x, std::vector<Real>& y) const;

    // Transpose (also the CSC->CSR conversion used by presolve and pricing).
    SparseMatrix transpose() const;

    // Drop |a_ij| <= tol.
    void dropSmall(Real tol);
    void sortRows();

    Real infNorm() const {
        Real m = 0; for (Real v : val) m = std::max(m, std::fabs(v)); return m;
    }
    // Largest / smallest nonzero magnitude, for conditioning diagnostics.
    void magnitudeRange(Real& lo, Real& hi) const;
};

// ---------------------------------------------------------------------------
// Triplet builder -> CSC.  Duplicate (i,j) entries are summed.
// ---------------------------------------------------------------------------
class TripletBuilder {
public:
    Int nrow = 0, ncol = 0;
    std::vector<Int>  ri, ci;
    std::vector<Real> vv;

    void reset(Int m, Int n) { nrow = m; ncol = n; ri.clear(); ci.clear(); vv.clear(); }
    void add(Int i, Int j, Real v) {
        if (v == 0.0) return;
        ri.push_back(i); ci.push_back(j); vv.push_back(v);
        if (i + 1 > nrow) nrow = i + 1;
        if (j + 1 > ncol) ncol = j + 1;
    }
    SparseMatrix build() const;
};

// ---------------------------------------------------------------------------
// A sparse vector with an O(1) occupancy test; the workhorse for FTRAN/BTRAN
// results, pricing rows and cut aggregation.  "dense" holds values, "idx"
// the pattern, "mark" the occupancy flags.  Reset is O(nnz), never O(n).
// ---------------------------------------------------------------------------
class SparseVector {
public:
    Int n = 0;
    std::vector<Real>    dense;
    std::vector<Int>     idx;
    std::vector<uint8_t> mark;

    void resize(Int nn) { n = nn; dense.assign(nn, 0.0); mark.assign(nn, 0); idx.clear(); }
    void clear() { for (Int i : idx) { dense[i] = 0.0; mark[i] = 0; } idx.clear(); }

    Int  nnz() const { return (Int)idx.size(); }
    Real operator[](Int i) const { return dense[i]; }

    void set(Int i, Real v) {
        if (!mark[i]) { mark[i] = 1; idx.push_back(i); }
        dense[i] = v;
    }
    void add(Int i, Real v) {
        if (!mark[i]) { mark[i] = 1; idx.push_back(i); dense[i] = v; }
        else dense[i] += v;
    }
    void scale(Real a) { for (Int i : idx) dense[i] *= a; }

    // Compress out numerical zeros.
    void prune(Real tol) {
        Int k = 0;
        for (Int t = 0; t < (Int)idx.size(); ++t) {
            Int i = idx[t];
            if (std::fabs(dense[i]) > tol) idx[k++] = i;
            else { dense[i] = 0.0; mark[i] = 0; }
        }
        idx.resize(k);
    }
    void sortIndices() { std::sort(idx.begin(), idx.end()); }

    Real infNorm() const {
        Real m = 0; for (Int i : idx) m = std::max(m, std::fabs(dense[i])); return m;
    }
    Real twoNormSq() const {
        Real s = 0; for (Int i : idx) s += dense[i] * dense[i]; return s;
    }
};

// ---------------------------------------------------------------------------
// Dense helpers (OpenMP-parallel where it pays).
// ---------------------------------------------------------------------------
Real dotProduct(const std::vector<Real>& a, const std::vector<Real>& b);
Real twoNorm(const std::vector<Real>& a);
Real infNorm(const std::vector<Real>& a);
void axpy(Real alpha, const std::vector<Real>& x, std::vector<Real>& y);
void scaleVec(Real alpha, std::vector<Real>& x);

} // namespace igaos
