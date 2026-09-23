#include "igaos/presolve.hpp"
#include <algorithm>

namespace igaos {

// ===========================================================================
//  Scaling
// ===========================================================================
static Real pow2round(Real v) {
    if (v <= 0.0 || !std::isfinite(v)) return 1.0;
    int e; std::frexp(v, &e);
    return std::ldexp(1.0, e - 1);          // nearest power of two below v*2
}

void computeScaling(const Model& m, Scaling& s, int passes, bool equilibrateQ) {
    Int nr = m.numRow(), nc = m.numCol();
    s.rowScale.assign(nr, 1.0);
    s.colScale.assign(nc, 1.0);
    s.active = true;
    const bool useQ = equilibrateQ && (m.Q.nnz() > 0 || !m.qcon.empty());
    if (m.A.nnz() == 0 && !useQ) return;

    std::vector<uint8_t> fixedCol(nc, 0);
    for (Int j = 0; j < nc; ++j)
        if (m.colType[j] != VarType::Continuous) fixedCol[j] = 1;   // keep integrality

    std::vector<Real> rmin(nr), rmax(nr), cmin(nc), cmax(nc);
    for (int pass = 0; pass < passes; ++pass) {
        // --- rows ---
        std::fill(rmin.begin(), rmin.end(), kBigReal);
        std::fill(rmax.begin(), rmax.end(), 0.0);
        for (Int j = 0; j < nc; ++j)
            for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p) {
                Int i = m.A.rowIdx[p];
                Real a = std::fabs(m.A.val[p]) * s.rowScale[i] * s.colScale[j];
                if (a == 0.0) continue;
                rmin[i] = std::min(rmin[i], a); rmax[i] = std::max(rmax[i], a);
            }
        for (Int i = 0; i < nr; ++i)
            if (rmax[i] > 0.0) s.rowScale[i] /= pow2round(std::sqrt(rmin[i] * rmax[i]));

        // --- columns ---
        std::fill(cmin.begin(), cmin.end(), kBigReal);
        std::fill(cmax.begin(), cmax.end(), 0.0);
        for (Int j = 0; j < nc; ++j)
            for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p) {
                Int i = m.A.rowIdx[p];
                Real a = std::fabs(m.A.val[p]) * s.rowScale[i] * s.colScale[j];
                if (a == 0.0) continue;
                cmin[j] = std::min(cmin[j], a); cmax[j] = std::max(cmax[j], a);
            }
        // --- the quadratic block, when asked for ---
        // |Q(i,j)| scales as colScale[i]*colScale[j].  Entering it here is what
        // makes the equilibration see the whole KKT matrix rather than only A.
        if (useQ) {
            for (Int j = 0; j < m.Q.ncol; ++j)
                for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p) {
                    Int i = m.Q.rowIdx[p];
                    Real q = std::fabs(m.Q.val[p]) * s.colScale[i] * s.colScale[j];
                    if (q == 0.0) continue;
                    cmin[j] = std::min(cmin[j], q); cmax[j] = std::max(cmax[j], q);
                    if (i != j) {                       // Q is symmetric
                        cmin[i] = std::min(cmin[i], q); cmax[i] = std::max(cmax[i], q);
                    }
                }
        }

        // Quadratic row terms, scaled rowScale[r]*colScale[i]*colScale[j].
        if (useQ) {
            for (const Model::QuadTerm& t : m.qcon) {
                Real v = std::fabs(t.coef) * s.rowScale[t.row]
                       * s.colScale[t.i] * s.colScale[t.j];
                if (v == 0.0) continue;
                cmin[t.i] = std::min(cmin[t.i], v); cmax[t.i] = std::max(cmax[t.i], v);
                cmin[t.j] = std::min(cmin[t.j], v); cmax[t.j] = std::max(cmax[t.j], v);
            }
        }

        for (Int j = 0; j < nc; ++j)
            if (!fixedCol[j] && cmax[j] > 0.0)
                s.colScale[j] /= pow2round(std::sqrt(cmin[j] * cmax[j]));
    }
}

void Scaling::apply(Model& m) const {
    if (!active) return;
    for (Int j = 0; j < m.numCol(); ++j) {
        Real cs = colScale[j];
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p)
            m.A.val[p] *= rowScale[m.A.rowIdx[p]] * cs;
        m.obj[j] *= cs;
        if (isFinite(m.colLower[j])) m.colLower[j] /= cs;
        if (isFinite(m.colUpper[j])) m.colUpper[j] /= cs;
    }
    for (Int i = 0; i < m.numRow(); ++i) {
        Real rs = rowScale[i];
        if (isFinite(m.rowLower[i])) m.rowLower[i] *= rs;
        if (isFinite(m.rowUpper[i])) m.rowUpper[i] *= rs;
    }
    if (m.Q.nnz() > 0)
        for (Int j = 0; j < m.Q.ncol; ++j)
            for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
                m.Q.val[p] *= colScale[j] * colScale[m.Q.rowIdx[p]];
    // coef*x_i*x_j inside row r transforms exactly like the row it sits in.
    for (Model::QuadTerm& t : m.qcon)
        t.coef *= rowScale[t.row] * colScale[t.i] * colScale[t.j];
}

void Scaling::unapply(Model& m) const {
    if (!active) return;
    for (Int j = 0; j < m.numCol(); ++j) {
        Real cs = colScale[j];
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p)
            m.A.val[p] /= rowScale[m.A.rowIdx[p]] * cs;
        m.obj[j] /= cs;
        if (isFinite(m.colLower[j])) m.colLower[j] *= cs;
        if (isFinite(m.colUpper[j])) m.colUpper[j] *= cs;
    }
    for (Int i = 0; i < m.numRow(); ++i) {
        Real rs = rowScale[i];
        if (isFinite(m.rowLower[i])) m.rowLower[i] /= rs;
        if (isFinite(m.rowUpper[i])) m.rowUpper[i] /= rs;
    }
    if (m.Q.nnz() > 0)
        for (Int j = 0; j < m.Q.ncol; ++j)
            for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
                m.Q.val[p] /= colScale[j] * colScale[m.Q.rowIdx[p]];
    for (Model::QuadTerm& t : m.qcon)
        t.coef /= rowScale[t.row] * colScale[t.i] * colScale[t.j];
}

void Scaling::unscaleSolution(Solution& s) const {
    if (!active) return;
    for (size_t j = 0; j < s.colValue.size() && j < colScale.size(); ++j) {
        s.colValue[j] *= colScale[j];
        s.colDual[j]  /= colScale[j];
    }
    for (size_t i = 0; i < s.rowValue.size() && i < rowScale.size(); ++i) {
        s.rowValue[i] /= rowScale[i];
        s.rowDual[i]  *= rowScale[i];
    }
}

// ===========================================================================
//  Presolve
// ===========================================================================
namespace {

struct Work {
    const Model* m;
    Int nr, nc;
    std::vector<Real> cl, cu, rl, ru;
    std::vector<uint8_t> colDead, rowDead;
    std::vector<Int> rowLen, colLen;
    SparseMatrix At;                       // row-wise view
    Int tightened = 0;
    bool infeasible = false;
};

// Activity bounds of a row over the live columns.
void rowActivityBounds(const Work& w, Int i, Real& lo, Real& hi, Int& nInfLo, Int& nInfHi) {
    lo = 0; hi = 0; nInfLo = 0; nInfHi = 0;
    for (Int p = w.At.colPtr[i]; p < w.At.colPtr[i + 1]; ++p) {
        Int j = w.At.rowIdx[p];
        if (w.colDead[j]) continue;
        Real a = w.At.val[p];
        Real l = w.cl[j], u = w.cu[j];
        if (a > 0) {
            if (isNegInf(l)) ++nInfLo; else lo += a * l;
            if (isInf(u))    ++nInfHi; else hi += a * u;
        } else {
            if (isInf(u))    ++nInfLo; else lo += a * u;
            if (isNegInf(l)) ++nInfHi; else hi += a * l;
        }
    }
}

} // namespace

void presolve(const Model& m, const Options& opt, PresolveResult& r) {
    Work w;
    w.m = &m; w.nr = m.numRow(); w.nc = m.numCol();
    w.cl = m.colLower; w.cu = m.colUpper;
    w.rl = m.rowLower; w.ru = m.rowUpper;
    w.colDead.assign(w.nc, 0);
    w.rowDead.assign(w.nr, 0);
    w.At = m.A.transpose();
    w.rowLen.assign(w.nr, 0);
    w.colLen.assign(w.nc, 0);
    for (Int j = 0; j < w.nc; ++j) w.colLen[j] = m.A.colLen(j);
    for (Int i = 0; i < w.nr; ++i) w.rowLen[i] = w.At.colLen(i);

    r.fixedValue.assign(w.nc, 0.0);
    const Real ft = opt.tol.primalFeas;

    // Columns that appear anywhere in the quadratic objective.  Every reduction
    // below that reasons from the SIGN OF THE LINEAR COST is invalid for these:
    // the minimiser of c*x + q*x^2/2 over [l,u] is clamp(-c/q, l, u), which is
    // an interior point whenever -c/q lands inside the box, not a bound.  Left
    // unguarded, presolve fixes such a column to a bound and the solver returns
    // a feasible point with a confident wrong objective.
    std::vector<char> quadCol(w.nc, 0);
    for (Int j = 0; j < m.Q.ncol; ++j)
        for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p) {
            quadCol[j] = 1;
            quadCol[m.Q.rowIdx[p]] = 1;
        }

    // Fixing a column in a QP is a SUBSTITUTION, not a deletion.  With x_j = v,
    // the quadratic form 1/2 x'Qx gives up
    //     1/2 Q_jj v^2                       -> a constant, and
    //     Q_ij v x_i  for every i != j       -> a LINEAR cost on the survivors.
    // Dropping the second family is what made a QP with any fixed column come
    // back with a wrong objective, so the objective is carried in a working copy
    // that killCol updates, rather than read from the original at the end.
    // Q is stored as a lower triangle by column, so the entries touching column
    // j live partly in column j (rows i >= j) and partly in the columns k < j
    // (as entries (j, k)); the transpose gives the second half by column.
    std::vector<Real> wobj = m.obj;
    Real objAccum = 0.0;
    SparseMatrix Qt;
    if (m.Q.nnz() > 0) Qt = m.Q.transpose();

    auto killCol = [&](Int j, Real val) {
        if (w.colDead[j]) return;
        w.colDead[j] = 1;
        r.fixedValue[j] = val;
        objAccum += wobj[j] * val;
        if (m.Q.nnz() > 0 && val != 0.0) {
            for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p) {
                Int i = m.Q.rowIdx[p];                      // (i, j), i >= j
                if (i == j) objAccum += 0.5 * m.Q.val[p] * val * val;
                else        wobj[i] += m.Q.val[p] * val;
            }
            for (Int p = Qt.colPtr[j]; p < Qt.colPtr[j + 1]; ++p) {
                Int k = Qt.rowIdx[p];                       // (j, k), k <= j
                if (k != j) wobj[k] += Qt.val[p] * val;
            }
        }
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p) {
            Int i = m.A.rowIdx[p];
            if (w.rowDead[i]) continue;
            Real a = m.A.val[p];
            if (isFinite(w.rl[i])) w.rl[i] -= a * val;
            if (isFinite(w.ru[i])) w.ru[i] -= a * val;
            w.rowLen[i]--;
        }
    };
    auto killRow = [&](Int i) {
        if (w.rowDead[i]) return;
        w.rowDead[i] = 1;
        for (Int p = w.At.colPtr[i]; p < w.At.colPtr[i + 1]; ++p) {
            Int j = w.At.rowIdx[p];
            if (!w.colDead[j]) w.colLen[j]--;
        }
    };
    // An implied bound is only information if it can cut something off. Chains
    // of equality rows routinely propagate values like 9.8e29 -- infinity with a
    // rounding error -- and recording one turns a free variable into a bounded
    // one. That is not a cosmetic difference: an interior point method must then
    // start strictly inside a bound 1e30 away, and on AUG2DQP from the Maros and
    // Meszaros set that produced a starting objective of 1e61 and a run that
    // died after one iteration. With presolve switched off the same model solved
    // in 38 iterations, which is how the reduction was identified as the cause.
    //
    // The rule is asymmetric on purpose. A lower bound of -1e29 says nothing, so
    // it is dropped; a lower bound of +1e29 says the variable is enormous, which
    // is real information and is kept. Only the useless direction is discarded.
    constexpr Real kImpliedBoundLimit = 1e12;
    auto tightenCol = [&](Int j, Real lo, Real hi) {
        bool ch = false;
        if (isFinite(lo) && lo < -kImpliedBoundLimit) lo = -kInf;
        if (isFinite(hi) && hi >  kImpliedBoundLimit) hi =  kInf;
        if (m.colType[j] != VarType::Continuous) {
            if (isFinite(lo)) lo = std::ceil(lo - 1e-9);
            if (isFinite(hi)) hi = std::floor(hi + 1e-9);
        }
        if (lo > w.cl[j] + 1e-11) { w.cl[j] = lo; ch = true; }
        if (hi < w.cu[j] - 1e-11) { w.cu[j] = hi; ch = true; }
        if (ch) ++w.tightened;
        if (w.cl[j] > w.cu[j] + ft) w.infeasible = true;
    };

    // ---- fixed columns up front ------------------------------------------
    for (Int j = 0; j < w.nc; ++j)
        if (isFinite(w.cl[j]) && std::fabs(w.cu[j] - w.cl[j]) <= 1e-12) killCol(j, w.cl[j]);

    // ---- main reduction loop ----------------------------------------------
    for (int round = 0; round < 12 && !w.infeasible; ++round) {
        Int before = 0;
        for (Int j = 0; j < w.nc; ++j) before += w.colDead[j];
        for (Int i = 0; i < w.nr; ++i) before += w.rowDead[i];
        Int tightBefore = w.tightened;

        for (Int i = 0; i < w.nr && !w.infeasible; ++i) {
            if (w.rowDead[i]) continue;

            // empty row
            if (w.rowLen[i] == 0) {
                if ((isFinite(w.rl[i]) && w.rl[i] >  ft) ||
                    (isFinite(w.ru[i]) && w.ru[i] < -ft)) { w.infeasible = true; break; }
                killRow(i);
                continue;
            }

            // singleton row -> bound on the single live variable
            if (w.rowLen[i] == 1) {
                Int jj = kNone; Real a = 0;
                for (Int p = w.At.colPtr[i]; p < w.At.colPtr[i + 1]; ++p) {
                    Int j = w.At.rowIdx[p];
                    if (w.colDead[j]) continue;
                    jj = j; a = w.At.val[p]; break;
                }
                if (jj == kNone) { killRow(i); continue; }
                Real lo = -kInf, hi = kInf;
                if (a > 0) {
                    if (isFinite(w.rl[i])) lo = w.rl[i] / a;
                    if (isFinite(w.ru[i])) hi = w.ru[i] / a;
                } else {
                    if (isFinite(w.ru[i])) lo = w.ru[i] / a;
                    if (isFinite(w.rl[i])) hi = w.rl[i] / a;
                }
                tightenCol(jj, lo, hi);
                killRow(i);
                if (isFinite(w.cl[jj]) && std::fabs(w.cu[jj] - w.cl[jj]) <= 1e-12)
                    killCol(jj, w.cl[jj]);
                continue;
            }

            // activity bounds -> redundant / forcing rows and bound tightening
            Real lo, hi; Int nlo, nhi;
            rowActivityBounds(w, i, lo, hi, nlo, nhi);
            Real actLo = (nlo > 0) ? -kInf : lo;
            Real actHi = (nhi > 0) ?  kInf : hi;

            if ((isFinite(w.ru[i]) && actLo > w.ru[i] + ft) ||
                (isFinite(w.rl[i]) && actHi < w.rl[i] - ft)) { w.infeasible = true; break; }

            bool redundant = (!isFinite(w.rl[i]) || actLo >= w.rl[i] - 1e-9) &&
                             (!isFinite(w.ru[i]) || actHi <= w.ru[i] + 1e-9);
            if (redundant) { killRow(i); continue; }

            // forcing row: every variable pinned to one bound
            if (isFinite(w.ru[i]) && nlo == 0 && std::fabs(actLo - w.ru[i]) <= 1e-9) {
                for (Int p = w.At.colPtr[i]; p < w.At.colPtr[i + 1]; ++p) {
                    Int j = w.At.rowIdx[p];
                    if (w.colDead[j]) continue;
                    Real a = w.At.val[p];
                    killCol(j, a > 0 ? w.cl[j] : w.cu[j]);
                }
                killRow(i);
                continue;
            }
            if (isFinite(w.rl[i]) && nhi == 0 && std::fabs(actHi - w.rl[i]) <= 1e-9) {
                for (Int p = w.At.colPtr[i]; p < w.At.colPtr[i + 1]; ++p) {
                    Int j = w.At.rowIdx[p];
                    if (w.colDead[j]) continue;
                    Real a = w.At.val[p];
                    killCol(j, a > 0 ? w.cu[j] : w.cl[j]);
                }
                killRow(i);
                continue;
            }

            // constraint propagation onto variable bounds (rounded for integers)
            for (Int p = w.At.colPtr[i]; p < w.At.colPtr[i + 1]; ++p) {
                Int j = w.At.rowIdx[p];
                if (w.colDead[j]) continue;
                Real a = w.At.val[p];
                if (std::fabs(a) < 1e-10) continue;
                // residual activity excluding j
                Real resLo, resHi;
                Real cj_lo = w.cl[j], cj_hi = w.cu[j];
                Real termLo = (a > 0) ? (isNegInf(cj_lo) ? -kInf : a * cj_lo)
                                      : (isInf(cj_hi)    ? -kInf : a * cj_hi);
                Real termHi = (a > 0) ? (isInf(cj_hi)    ?  kInf : a * cj_hi)
                                      : (isNegInf(cj_lo) ?  kInf : a * cj_lo);
                Int nl = nlo - (isInf(std::fabs(termLo)) ? 1 : 0);
                Int nh = nhi - (isInf(std::fabs(termHi)) ? 1 : 0);
                resLo = (nl > 0) ? -kInf : (isFinite(termLo) ? lo - termLo : lo);
                resHi = (nh > 0) ?  kInf : (isFinite(termHi) ? hi - termHi : hi);

                Real nlB = -kInf, nuB = kInf;
                if (a > 0) {
                    if (isFinite(w.ru[i]) && isFinite(resLo)) nuB = (w.ru[i] - resLo) / a;
                    if (isFinite(w.rl[i]) && isFinite(resHi)) nlB = (w.rl[i] - resHi) / a;
                } else {
                    if (isFinite(w.ru[i]) && isFinite(resLo)) nlB = (w.ru[i] - resLo) / a;
                    if (isFinite(w.rl[i]) && isFinite(resHi)) nuB = (w.rl[i] - resHi) / a;
                }
                if (isFinite(nlB) || isFinite(nuB)) tightenCol(j, nlB, nuB);
                if (w.infeasible) break;
                if (isFinite(w.cl[j]) && std::fabs(w.cu[j] - w.cl[j]) <= 1e-12)
                    killCol(j, w.cl[j]);
            }
        }

        // empty / free columns: fix to the bound the objective prefers.
        // Only sound when the objective really is linear in this column.
        for (Int j = 0; j < w.nc && !w.infeasible; ++j) {
            if (w.colDead[j]) continue;
            if (w.colLen[j] != 0) continue;
            if (quadCol[j]) continue;              // see quadCol above
            Real c = wobj[j];
            if (c > 0) {
                if (isNegInf(w.cl[j])) { r.status = Status::Unbounded; return; }
                killCol(j, w.cl[j]);
            } else if (c < 0) {
                if (isInf(w.cu[j])) { r.status = Status::Unbounded; return; }
                killCol(j, w.cu[j]);
            } else {
                killCol(j, isFinite(w.cl[j]) ? w.cl[j] : (isFinite(w.cu[j]) ? w.cu[j] : 0.0));
            }
        }

        Int after = 0;
        for (Int j = 0; j < w.nc; ++j) after += w.colDead[j];
        for (Int i = 0; i < w.nr; ++i) after += w.rowDead[i];
        if (after == before && w.tightened == tightBefore) break;
    }

    if (w.infeasible) { r.status = Status::Infeasible; return; }

    // ---- build the reduced model ------------------------------------------
    r.origColToReduced.assign(w.nc, kNone);
    r.origRowToReduced.assign(w.nr, kNone);
    r.colMap.clear(); r.rowMap.clear();
    for (Int j = 0; j < w.nc; ++j)
        if (!w.colDead[j]) { r.origColToReduced[j] = (Int)r.colMap.size(); r.colMap.push_back(j); }
    for (Int i = 0; i < w.nr; ++i)
        if (!w.rowDead[i]) { r.origRowToReduced[i] = (Int)r.rowMap.size(); r.rowMap.push_back(i); }

    r.colRemoved.assign(w.nc, 0); r.rowRemoved.assign(w.nr, 0);
    for (Int j = 0; j < w.nc; ++j) r.colRemoved[j] = w.colDead[j];
    for (Int i = 0; i < w.nr; ++i) r.rowRemoved[i] = w.rowDead[i];
    r.removedCols = w.nc - (Int)r.colMap.size();
    r.removedRows = w.nr - (Int)r.rowMap.size();
    r.tightenedBounds = w.tightened;

    Model& rm = r.reduced;
    rm = Model();
    rm.name = m.name + "_presolved";
    rm.sense = Sense::Minimize;
    // objAccum already carries the linear AND quadratic contribution of every
    // column killCol removed, in the order they were removed.
    rm.objOffset = m.objOffset + objAccum;

    Int rn = (Int)r.colMap.size(), rmr = (Int)r.rowMap.size();
    rm.obj.resize(rn); rm.colLower.resize(rn); rm.colUpper.resize(rn);
    rm.colType.resize(rn); rm.colName.resize(rn);
    for (Int jj = 0; jj < rn; ++jj) {
        Int j = r.colMap[jj];
        rm.obj[jj] = wobj[j];
        rm.colLower[jj] = w.cl[j]; rm.colUpper[jj] = w.cu[j];
        rm.colType[jj] = m.colType[j];
        rm.colName[jj] = j < (Int)m.colName.size() ? m.colName[j] : ("x" + std::to_string(j));
    }
    rm.rowLower.resize(rmr); rm.rowUpper.resize(rmr); rm.rowName.resize(rmr);
    for (Int ii = 0; ii < rmr; ++ii) {
        Int i = r.rowMap[ii];
        rm.rowLower[ii] = w.rl[i]; rm.rowUpper[ii] = w.ru[i];
        rm.rowName[ii] = i < (Int)m.rowName.size() ? m.rowName[i] : ("r" + std::to_string(i));
    }
    Int keptNnz = 0;
    for (Int jj = 0; jj < rn; ++jj) {
        Int j = r.colMap[jj];
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p) {
            Int i = m.A.rowIdx[p];
            Int ii = r.origRowToReduced[i];
            if (ii == kNone) continue;
            rm.setElement(ii, jj, m.A.val[p]);
            ++keptNnz;
        }
    }
    if (m.Q.nnz() > 0)
        for (Int jj = 0; jj < rn; ++jj) {
            Int j = r.colMap[jj];
            for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p) {
                Int i = m.Q.rowIdx[p];
                Int ii = r.origColToReduced[i];
                if (ii == kNone) continue;
                rm.setQuadratic(ii, jj, m.Q.val[p]);
            }
        }
    rm.finalize();
    rm.A.nrow = rmr; rm.A.ncol = rn;
    if ((Int)rm.A.colPtr.size() != rn + 1) rm.A.colPtr.resize(rn + 1, rm.A.nnz());
    r.removedNnz = m.A.nnz() - keptNnz;
    r.status = Status::NotSolved;
}

// ---------------------------------------------------------------------------
void postsolve(const Model& orig, const PresolveResult& r,
               const Solution& red, Solution& full) {
    Int nc = orig.numCol(), nr = orig.numRow();
    full.resize(nr, nc);
    full.status = red.status;
    full.iterations = red.iterations;
    full.nodes = red.nodes;

    for (Int j = 0; j < nc; ++j) {
        Int jj = r.origColToReduced.empty() ? kNone : r.origColToReduced[j];
        if (jj != kNone && jj < (Int)red.colValue.size()) {
            full.colValue[j] = red.colValue[jj];
            full.colStatus[j] = red.colStatus[jj];
        } else {
            full.colValue[j] = r.fixedValue.empty() ? 0.0 : r.fixedValue[j];
            // A removed column is nonbasic at whichever original bound it sits on.
            Real v = full.colValue[j];
            if (isFinite(orig.colLower[j]) && std::fabs(v - orig.colLower[j]) <= 1e-9)
                full.colStatus[j] = VarStatus::AtLower;
            else if (isFinite(orig.colUpper[j]) && std::fabs(v - orig.colUpper[j]) <= 1e-9)
                full.colStatus[j] = VarStatus::AtUpper;
            else full.colStatus[j] = VarStatus::AtZero;
        }
    }
    for (Int i = 0; i < nr; ++i) {
        Int ii = r.origRowToReduced.empty() ? kNone : r.origRowToReduced[i];
        if (ii != kNone && ii < (Int)red.rowStatus.size())
            full.rowStatus[i] = red.rowStatus[ii];
        else
            full.rowStatus[i] = VarStatus::Basic;   // removed row => logical basic
    }
    orig.rowActivity(full.colValue, full.rowValue);
    full.objective = orig.objectiveValue(full.colValue);
}

} // namespace igaos
