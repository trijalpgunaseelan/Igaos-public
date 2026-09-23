#include "igaos/sparse.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace igaos {

const char* statusName(Status s) {
    switch (s) {
        case Status::NotSolved:      return "not_solved";
        case Status::Optimal:        return "optimal";
        case Status::Infeasible:     return "infeasible";
        case Status::Unbounded:      return "unbounded";
        case Status::IterationLimit: return "iteration_limit";
        case Status::TimeLimit:      return "time_limit";
        case Status::NodeLimit:      return "node_limit";
        case Status::NumericalError: return "numerical_error";
        case Status::Interrupted:    return "interrupted";
        case Status::Feasible:       return "feasible";
    }
    return "unknown";
}

void SparseMatrix::multiply(const std::vector<Real>& x, std::vector<Real>& y) const {
    y.assign(nrow, 0.0);
    multiplyAdd(1.0, x, y);
}

void SparseMatrix::multiplyAdd(Real alpha, const std::vector<Real>& x,
                               std::vector<Real>& y) const {
    // Column-major SpMV.  Sequential accumulation into y avoids the atomics a
    // naive column-parallel loop would need; the row-major transpose path
    // below is the one we parallelise.
    for (Int j = 0; j < ncol; ++j) {
        Real xj = x[j];
        if (xj == 0.0) continue;
        Real a = alpha * xj;
        for (Int p = colPtr[j]; p < colPtr[j + 1]; ++p) y[rowIdx[p]] += a * val[p];
    }
}

void SparseMatrix::multiplyTranspose(const std::vector<Real>& x, std::vector<Real>& y) const {
    y.assign(ncol, 0.0);
    multiplyTransposeAdd(1.0, x, y);
}

void SparseMatrix::multiplyTransposeAdd(Real alpha, const std::vector<Real>& x,
                                        std::vector<Real>& y) const {
    // Each output entry y[j] is an independent reduction -> embarrassingly
    // parallel, and this is the hot kernel inside PDHG.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (ncol > 4096)
#endif
    for (Int j = 0; j < ncol; ++j) {
        Real s = 0.0;
        for (Int p = colPtr[j]; p < colPtr[j + 1]; ++p) s += val[p] * x[rowIdx[p]];
        y[j] += alpha * s;
    }
}

SparseMatrix SparseMatrix::transpose() const {
    SparseMatrix T(ncol, nrow);
    Int nz = nnz();
    T.colPtr.assign(nrow + 1, 0);
    for (Int p = 0; p < nz; ++p) T.colPtr[rowIdx[p] + 1]++;
    for (Int i = 0; i < nrow; ++i) T.colPtr[i + 1] += T.colPtr[i];
    T.rowIdx.resize(nz); T.val.resize(nz);
    std::vector<Int> next(T.colPtr.begin(), T.colPtr.end() - 1);
    for (Int j = 0; j < ncol; ++j)
        for (Int p = colPtr[j]; p < colPtr[j + 1]; ++p) {
            Int i = rowIdx[p], q = next[i]++;
            T.rowIdx[q] = j; T.val[q] = val[p];
        }
    return T;
}

void SparseMatrix::dropSmall(Real tol) {
    std::vector<Int> newPtr(ncol + 1, 0);
    Int k = 0;
    for (Int j = 0; j < ncol; ++j) {
        newPtr[j] = k;
        for (Int p = colPtr[j]; p < colPtr[j + 1]; ++p)
            if (std::fabs(val[p]) > tol) { rowIdx[k] = rowIdx[p]; val[k] = val[p]; ++k; }
    }
    newPtr[ncol] = k;
    colPtr.swap(newPtr);
    rowIdx.resize(k); val.resize(k);
}

void SparseMatrix::sortRows() {
    std::vector<std::pair<Int, Real>> buf;
    for (Int j = 0; j < ncol; ++j) {
        Int s = colPtr[j], e = colPtr[j + 1];
        if (e - s < 2) continue;
        bool sorted = true;
        for (Int p = s + 1; p < e; ++p) if (rowIdx[p] < rowIdx[p - 1]) { sorted = false; break; }
        if (sorted) continue;
        buf.clear();
        for (Int p = s; p < e; ++p) buf.emplace_back(rowIdx[p], val[p]);
        std::sort(buf.begin(), buf.end(),
                  [](const std::pair<Int,Real>& a, const std::pair<Int,Real>& b) {
                      return a.first < b.first; });
        for (Int p = s; p < e; ++p) { rowIdx[p] = buf[p - s].first; val[p] = buf[p - s].second; }
    }
}

void SparseMatrix::magnitudeRange(Real& lo, Real& hi) const {
    lo = kBigReal; hi = 0.0;
    for (Real v : val) {
        Real a = std::fabs(v);
        if (a == 0.0) continue;
        lo = std::min(lo, a); hi = std::max(hi, a);
    }
    if (hi == 0.0) { lo = hi = 1.0; }
}

SparseMatrix TripletBuilder::build() const {
    SparseMatrix A(nrow, ncol);
    Int nz = (Int)vv.size();
    A.colPtr.assign(ncol + 1, 0);
    for (Int p = 0; p < nz; ++p) A.colPtr[ci[p] + 1]++;
    for (Int j = 0; j < ncol; ++j) A.colPtr[j + 1] += A.colPtr[j];
    A.rowIdx.resize(nz); A.val.resize(nz);
    std::vector<Int> next(A.colPtr.begin(), A.colPtr.end() - 1);
    for (Int p = 0; p < nz; ++p) {
        Int j = ci[p], q = next[j]++;
        A.rowIdx[q] = ri[p]; A.val[q] = vv[p];
    }
    A.sortRows();
    // Sum duplicates.
    Int k = 0;
    std::vector<Int> newPtr(ncol + 1, 0);
    for (Int j = 0; j < ncol; ++j) {
        newPtr[j] = k;
        Int p = A.colPtr[j];
        while (p < A.colPtr[j + 1]) {
            Int  i = A.rowIdx[p];
            Real s = A.val[p];
            ++p;
            while (p < A.colPtr[j + 1] && A.rowIdx[p] == i) { s += A.val[p]; ++p; }
            if (s != 0.0) { A.rowIdx[k] = i; A.val[k] = s; ++k; }
        }
    }
    newPtr[ncol] = k;
    A.colPtr.swap(newPtr);
    A.rowIdx.resize(k); A.val.resize(k);
    return A;
}

Real dotProduct(const std::vector<Real>& a, const std::vector<Real>& b) {
    Real s = 0; Int n = (Int)a.size();
#ifdef _OPENMP
#pragma omp parallel for reduction(+:s) schedule(static) if (n > 8192)
#endif
    for (Int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

Real twoNorm(const std::vector<Real>& a) { return std::sqrt(dotProduct(a, a)); }

Real infNorm(const std::vector<Real>& a) {
    Real m = 0; for (Real v : a) m = std::max(m, std::fabs(v)); return m;
}

void axpy(Real alpha, const std::vector<Real>& x, std::vector<Real>& y) {
    Int n = (Int)x.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 8192)
#endif
    for (Int i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void scaleVec(Real alpha, std::vector<Real>& x) {
    for (Real& v : x) v *= alpha;
}

} // namespace igaos
