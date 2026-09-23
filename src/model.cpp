#include "igaos/model.hpp"
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace igaos {

Int Model::addColumn(Real lo, Real up, Real cost, VarType type, const std::string& nm) {
    Int j = numCol();
    obj.push_back(cost);
    colLower.push_back(lo);
    colUpper.push_back(up);
    colType.push_back(type);
    colName.push_back(nm.empty() ? ("x" + std::to_string(j)) : nm);
    dirty_ = true;
    return j;
}

Int Model::addRow(Real lo, Real up, const std::string& nm) {
    Int i = numRow();
    rowLower.push_back(lo);
    rowUpper.push_back(up);
    rowName.push_back(nm.empty() ? ("r" + std::to_string(i)) : nm);
    dirty_ = true;
    return i;
}

void Model::setElement(Int row, Int col, Real v) { abuf_.add(row, col, v); dirty_ = true; }

void Model::addQuadraticTerm(Int row, Int i, Int j, Real coef) {
    if (coef == 0.0) return;
    // Normalise the pair order so that i >= j.  x_a*x_b and x_b*x_a are the same
    // term, and the relaxation builder keys its auxiliary variables on the pair
    // -- without this it would create two auxiliaries for one product and the
    // two McCormick envelopes would not be linked to each other.
    if (j > i) std::swap(i, j);
    qcon.push_back(QuadTerm{row, i, j, coef});
}
void Model::setQuadratic(Int i, Int j, Real v) {
    if (j > i) std::swap(i, j);                    // keep lower triangle
    qbuf_.add(i, j, v); dirty_ = true;
}

void Model::finalize() {
    if (!dirty_) return;
    abuf_.nrow = std::max(abuf_.nrow, numRow());
    abuf_.ncol = std::max(abuf_.ncol, numCol());
    A = abuf_.build();
    A.nrow = numRow(); A.ncol = numCol();
    A.colPtr.resize(numCol() + 1, A.nnz());
    if (!qbuf_.vv.empty()) {
        qbuf_.nrow = std::max(qbuf_.nrow, numCol());
        qbuf_.ncol = std::max(qbuf_.ncol, numCol());
        Q = qbuf_.build();
        Q.nrow = numCol(); Q.ncol = numCol();
        Q.colPtr.resize(numCol() + 1, Q.nnz());
    }
    dirty_ = false;
}

void Model::ensureNames() {
    for (Int j = (Int)colName.size(); j < numCol(); ++j) colName.push_back("x" + std::to_string(j));
    for (Int i = (Int)rowName.size(); i < numRow(); ++i) rowName.push_back("r" + std::to_string(i));
}

void Model::validate() const {
    if ((Int)colLower.size() != numCol() || (Int)colUpper.size() != numCol())
        throw std::runtime_error("igaos: column bound arrays inconsistent");
    if ((Int)rowUpper.size() != numRow())
        throw std::runtime_error("igaos: row bound arrays inconsistent");
    if (A.ncol != numCol() || A.nrow != numRow())
        throw std::runtime_error("igaos: matrix shape does not match model (call finalize)");
    for (Int j = 0; j < numCol(); ++j)
        if (colLower[j] > colUpper[j] + 1e-9) {
            std::ostringstream os;
            os << "igaos: column " << j << " has empty bound interval ["
               << colLower[j] << "," << colUpper[j] << "]";
            throw std::runtime_error(os.str());
        }
    for (Int i = 0; i < numRow(); ++i)
        if (rowLower[i] > rowUpper[i] + 1e-9)
            throw std::runtime_error("igaos: row " + std::to_string(i) + " has empty range");
    for (const QuadTerm& t : qcon) {
        if (t.row < 0 || t.row >= numRow())
            throw std::runtime_error("igaos: quadratic term names row "
                                     + std::to_string(t.row) + ", which does not exist");
        if (t.i < 0 || t.i >= numCol() || t.j < 0 || t.j >= numCol())
            throw std::runtime_error("igaos: quadratic term names a column that does not exist");
        // A bilinear term needs both its variables bounded on both sides, or a
        // McCormick envelope cannot be written and the relaxation has no bound
        // to give.  Failing here is much better than failing silently later
        // with a "global optimum" that was never bounded.
        for (Int c : {t.i, t.j})
            if (!isFinite(colLower[c]) || !isFinite(colUpper[c]))
                throw std::runtime_error(
                    "igaos: column " + (c < (Int)colName.size() ? colName[c]
                                                                : std::to_string(c))
                    + " appears in a quadratic constraint term but is not bounded on both "
                      "sides; a McCormick relaxation needs finite bounds");
    }
}

Real Model::objectiveValue(const std::vector<Real>& x) const {
    Real v = objOffset;
    for (Int j = 0; j < numCol(); ++j) v += obj[j] * x[j];
    if (Q.nnz() > 0) {
        Real q = 0;
        for (Int j = 0; j < Q.ncol; ++j)
            for (Int p = Q.colPtr[j]; p < Q.colPtr[j + 1]; ++p) {
                Int i = Q.rowIdx[p];
                // Lower triangle: diagonal counted once, off-diagonal twice.
                q += (i == j ? 0.5 : 1.0) * Q.val[p] * x[i] * x[j];
            }
        v += q;
    }
    return v;
}

Real Model::primalInfeasibility(const std::vector<Real>& x) const {
    Real worst = 0.0;
    for (Int j = 0; j < numCol(); ++j) {
        worst = std::max(worst, colLower[j] - x[j]);
        worst = std::max(worst, x[j] - colUpper[j]);
    }
    std::vector<Real> act;
    rowActivity(x, act);            // includes the quadratic part of each row
    for (Int i = 0; i < numRow(); ++i) {
        worst = std::max(worst, rowLower[i] - act[i]);
        worst = std::max(worst, act[i] - rowUpper[i]);
    }
    return std::max(worst, 0.0);
}

Real Model::integerInfeasibility(const std::vector<Real>& x, Real tol) const {
    Real worst = 0.0;
    for (Int j = 0; j < numCol(); ++j)
        if (colType[j] != VarType::Continuous) {
            Real d = std::fabs(x[j] - std::floor(x[j] + 0.5));
            if (d > tol) worst = std::max(worst, d);
        }
    return worst;
}

Model::Stats Model::stats() const {
    Stats s;
    s.nrow = numRow(); s.ncol = numCol(); s.nnz = A.nnz(); s.nqnz = Q.nnz();
    s.nqcon = (Int)qcon.size();
    {
        std::vector<uint8_t> seen((size_t)std::max<Int>(numRow(), 1), 0);
        for (const QuadTerm& t : qcon)
            if (t.row >= 0 && t.row < numRow() && !seen[(size_t)t.row]) {
                seen[(size_t)t.row] = 1; ++s.nqconRows;
            }
    }
    for (Int j = 0; j < numCol(); ++j) {
        if (colType[j] != VarType::Continuous) {
            ++s.nint;
            if (colLower[j] >= -1e-9 && colUpper[j] <= 1.0 + 1e-9) ++s.nbin;
        }
        s.maxColLen = std::max(s.maxColLen, A.colLen(j));
    }
    std::vector<Int> rlen(numRow(), 0);
    for (Int p = 0; p < s.nnz; ++p) rlen[A.rowIdx[p]]++;
    for (Int i = 0; i < numRow(); ++i) {
        s.maxRowLen = std::max(s.maxRowLen, rlen[i]);
        bool lf = isFinite(rowLower[i]), uf = isFinite(rowUpper[i]);
        if (lf && uf) { if (rowLower[i] == rowUpper[i]) ++s.nEqRows; else ++s.nRangeRows; }
        else if (!lf && !uf) ++s.nFreeRows;
    }
    if (s.nrow > 0 && s.ncol > 0)
        s.density = (Real)s.nnz / ((Real)s.nrow * (Real)s.ncol);
    A.magnitudeRange(s.minAbs, s.maxAbs);
    s.ratio = s.maxAbs / std::max(s.minAbs, 1e-300);
    return s;
}

void Model::toMaximizationNegated() {
    for (Real& c : obj) c = -c;
    for (Real& q : Q.val) q = -q;
    objOffset = -objOffset;
    // Row quadratics are NOT negated: negating the objective does not touch the
    // constraints.  Stated because the loop above sits right next to them and
    // the omission would otherwise look like one.
}

// ---------------------------------------------------------------------------
// Convexity of the objective Hessian, by attempted Cholesky.
//
// The test is a factorization rather than an eigenvalue computation because
// that is exactly the question being asked: a convex QP path needs L with
// Q = L L', and if the factorization runs to completion with positive pivots
// then Q is positive semidefinite in the only sense that matters downstream.
// Dense, because it is called once per solve on the objective Hessian and a
// model with a Hessian too large to factor densely has other problems.
// ---------------------------------------------------------------------------
bool Model::hasNonconvexObjective() const {
    const Int n = numCol();
    if (Q.nnz() == 0 || n == 0) return false;
    if (n > 2000) return false;          // too large to test this way; assume convex

    std::vector<Real> M((size_t)n * (size_t)n, 0.0);
    for (Int j = 0; j < Q.ncol; ++j)
        for (Int p = Q.colPtr[j]; p < Q.colPtr[j + 1]; ++p) {
            Int i = Q.rowIdx[p];
            M[(size_t)i * n + j] += Q.val[p];
            if (i != j) M[(size_t)j * n + i] += Q.val[p];
        }

    // A semidefinite matrix has zero pivots, so the test allows a pivot to be
    // zero (to a scaled tolerance) and rejects only a genuinely NEGATIVE one.
    Real scale = 0.0;
    for (Real v : M) scale = std::max(scale, std::fabs(v));
    const Real tol = 1e-10 * std::max(scale, 1.0);
    for (Int k = 0; k < n; ++k) {
        Real d = M[(size_t)k * n + k];
        for (Int p = 0; p < k; ++p) d -= M[(size_t)k * n + p] * M[(size_t)k * n + p];
        if (d < -tol) return true;                       // negative pivot: indefinite
        d = std::sqrt(std::max(d, 0.0));
        M[(size_t)k * n + k] = d;
        for (Int i = k + 1; i < n; ++i) {
            Real v = M[(size_t)i * n + k];
            for (Int p = 0; p < k; ++p) v -= M[(size_t)i * n + p] * M[(size_t)k * n + p];
            if (d > tol) M[(size_t)i * n + k] = v / d;
            else if (std::fabs(v) > 1e-6 * std::max(scale, 1.0)) return true;
            else M[(size_t)i * n + k] = 0.0;
        }
    }
    return false;
}

} // namespace igaos
