#include "igaos/lu.hpp"
#include <numeric>

namespace igaos {

// ---------------------------------------------------------------------------
// Right-looking sparse Gaussian elimination with threshold Markowitz pivoting.
//
// At every step we choose a pivot (r, j) minimising the Markowitz cost
//        (rowCount[r] - 1) * (colCount[j] - 1)
// subject to the numerical threshold test
//        |a_rj| >= markowitz * max_i |a_ij|
// which bounds the multipliers stored in L by 1/markowitz and therefore bounds
// element growth.  Column and row singletons are pivoted first, which peels off
// the large triangular parts an LP basis almost always has before any general
// search is needed.
// ---------------------------------------------------------------------------
bool BasisFactor::factorize(const ExtendedMatrix& E, const std::vector<Int>& basis,
                            const Tolerances& tol) {
    m = (Int)basis.size();
    singularPositions.clear();
    etas_.clear(); etaNnz_ = 0;
    work_.assign(m, 0.0);

    // --- working copy of B, column-wise (columns indexed by basis position) --
    std::vector<std::vector<std::pair<Int, Real>>> cols(m);
    std::vector<std::vector<Int>> rows(m);
    std::vector<Int> colCnt(m, 0), rowCnt(m, 0);
    std::vector<uint8_t> colDone(m, 0), rowDone(m, 0);
    Real inputMax = 0.0;

    for (Int k = 0; k < m; ++k) {
        Int bj = basis[k];
        E.forEach(bj, [&](Int i, Real v) {
            if (v == 0.0) return;
            cols[k].emplace_back(i, v);
            rows[i].push_back(k);
            rowCnt[i]++;
            inputMax = std::max(inputMax, std::fabs(v));
        });
        colCnt[k] = (Int)cols[k].size();
    }
    if (inputMax == 0.0) inputMax = 1.0;

    // --- count buckets -------------------------------------------------------
    std::vector<std::vector<Int>> colBucket(m + 2), rowBucket(m + 2);
    for (Int k = 0; k < m; ++k) if (colCnt[k] <= m) colBucket[colCnt[k]].push_back(k);
    for (Int i = 0; i < m; ++i) if (rowCnt[i] <= m) rowBucket[rowCnt[i]].push_back(i);
    auto touchCol = [&](Int j) { if (!colDone[j] && colCnt[j] >= 0 && colCnt[j] <= m) colBucket[colCnt[j]].push_back(j); };
    auto touchRow = [&](Int i) { if (!rowDone[i] && rowCnt[i] >= 0 && rowCnt[i] <= m) rowBucket[rowCnt[i]].push_back(i); };

    pivotRow_.assign(m, kNone); pivotCol_.assign(m, kNone);
    rowPos_.assign(m, kNone);   colPos_.assign(m, kNone);

    // Output accumulators (original indices; remapped to pivot order at the end)
    std::vector<Int> lStart(m + 1, 0), uStart(m + 1, 0);
    std::vector<Int>  lRow, uCol;
    std::vector<Real> lv, uv;
    uDiag_.assign(m, 1.0);
    lRow.reserve(cols.size() * 4); uCol.reserve(cols.size() * 4);

    // Dense scatter workspace for the elimination update.
    std::vector<Real>    wval(m, 0.0);
    std::vector<uint8_t> wmark(m, 0);
    std::vector<Int>     wpat; wpat.reserve(m);

    Real umax = 0.0;
    minPivot_ = kBigReal;
    const Real tau = tol.markowitz;
    const Real zeroTol = tol.zero;

    std::vector<std::pair<Int, Real>> Lcol;      // (original row, multiplier)
    std::vector<Int> rowCandidates;

    Int nPivoted = 0;
    for (Int step = 0; step < m; ++step) {
        Int bestRow = kNone, bestCol = kNone;
        Real bestVal = 0.0;
        Long bestCost = std::numeric_limits<Long>::max();

        // ---- 1. column singletons: Markowitz cost 0, threshold automatic ----
        while (!colBucket[1].empty()) {
            Int j = colBucket[1].back(); colBucket[1].pop_back();
            if (colDone[j] || colCnt[j] != 1) continue;
            // locate the single active entry
            for (auto& e : cols[j]) {
                if (rowDone[e.first]) continue;
                if (std::fabs(e.second) > tol.pivot) {
                    bestRow = e.first; bestCol = j; bestVal = e.second; bestCost = 0;
                }
                break;
            }
            if (bestCost == 0) break;
        }

        // ---- 2. row singletons ----------------------------------------------
        if (bestCost != 0) {
            while (!rowBucket[1].empty()) {
                Int i = rowBucket[1].back(); rowBucket[1].pop_back();
                if (rowDone[i] || rowCnt[i] != 1) continue;
                Int foundCol = kNone; Real foundVal = 0.0;
                for (Int j : rows[i]) {
                    if (colDone[j]) continue;
                    for (auto& e : cols[j]) if (e.first == i) { foundCol = j; foundVal = e.second; break; }
                    if (foundCol != kNone) break;
                }
                if (foundCol == kNone) continue;
                Real cmax = 0.0;
                for (auto& e : cols[foundCol]) if (!rowDone[e.first]) cmax = std::max(cmax, std::fabs(e.second));
                if (std::fabs(foundVal) >= tau * cmax && std::fabs(foundVal) > tol.pivot) {
                    bestRow = i; bestCol = foundCol; bestVal = foundVal; bestCost = 0;
                    break;
                }
            }
        }

        // ---- 3. general threshold Markowitz search ---------------------------
        if (bestCost != 0) {
            int searched = 0;
            const int kMaxSearch = 8;
            for (Int c = 1; c <= m && searched < kMaxSearch; ++c) {
                if ((Long)(c - 1) * (Long)(c - 1) > bestCost) break;   // cannot improve
                auto& bk = colBucket[c];
                size_t keep = 0;
                for (size_t t = 0; t < bk.size(); ++t) {
                    Int j = bk[t];
                    if (colDone[j]) continue;
                    if (colCnt[j] != c) { continue; }   // stale; it lives in another bucket
                    bk[keep++] = j;
                    if (searched >= kMaxSearch) continue;
                    ++searched;
                    Real cmax = 0.0;
                    for (auto& e : cols[j]) if (!rowDone[e.first]) cmax = std::max(cmax, std::fabs(e.second));
                    if (cmax <= tol.pivot) continue;
                    for (auto& e : cols[j]) {
                        Int i = e.first;
                        if (rowDone[i]) continue;
                        Real a = std::fabs(e.second);
                        if (a < tau * cmax || a <= tol.pivot) continue;
                        Long cost = (Long)(rowCnt[i] - 1) * (Long)(c - 1);
                        if (cost < bestCost || (cost == bestCost && a > std::fabs(bestVal))) {
                            bestCost = cost; bestRow = i; bestCol = j; bestVal = e.second;
                        }
                    }
                }
                bk.resize(keep);
            }
        }

        if (bestCol == kNone) break;                    // structurally singular

        Int r = bestRow, jc = bestCol;
        Real piv = bestVal;
        pivotRow_[step] = r; pivotCol_[step] = jc;
        rowPos_[r] = step;  colPos_[jc] = step;
        uDiag_[step] = piv;
        minPivot_ = std::min(minPivot_, std::fabs(piv));
        umax = std::max(umax, std::fabs(piv));
        ++nPivoted;

        // ---- build L column (multipliers) and retire the pivot column -------
        Lcol.clear();
        lStart[step] = (Int)lRow.size();
        for (auto& e : cols[jc]) {
            Int i = e.first;
            if (rowDone[i]) continue;
            rowCnt[i]--;                                 // pivot column disappears
            if (i == r) continue;
            Real mult = e.second / piv;
            if (std::fabs(mult) <= zeroTol) continue;
            Lcol.emplace_back(i, mult);
            lRow.push_back(i); lv.push_back(mult);
        }
        colDone[jc] = 1; rowDone[r] = 1;
        cols[jc].clear(); cols[jc].shrink_to_fit();

        // ---- gather the pivot row and eliminate ------------------------------
        rowCandidates.clear();
        for (Int j : rows[r]) if (!colDone[j]) rowCandidates.push_back(j);
        std::sort(rowCandidates.begin(), rowCandidates.end());
        rowCandidates.erase(std::unique(rowCandidates.begin(), rowCandidates.end()),
                            rowCandidates.end());

        uStart[step] = (Int)uCol.size();
        for (Int j : rowCandidates) {
            if (colDone[j]) continue;
            // scatter column j
            wpat.clear();
            Real arj = 0.0; bool hasR = false;
            for (auto& e : cols[j]) {
                Int i = e.first;
                if (rowDone[i] && i != r) continue;
                if (i == r) { arj = e.second; hasR = true; continue; }
                if (!wmark[i]) { wmark[i] = 1; wval[i] = e.second; wpat.push_back(i); }
                else wval[i] += e.second;
            }
            if (!hasR || arj == 0.0) {
                // Row r not actually present: just compact the column.
                for (Int i : wpat) { wmark[i] = 0; wval[i] = 0.0; }
                continue;
            }
            uCol.push_back(j); uv.push_back(arj);
            umax = std::max(umax, std::fabs(arj));
            rowCnt[r]--;

            for (auto& lp : Lcol) {
                Int i = lp.first;
                Real d = lp.second * arj;
                if (!wmark[i]) {
                    wmark[i] = 1; wval[i] = -d; wpat.push_back(i);
                    rowCnt[i]++; rows[i].push_back(j);      // fill-in
                } else {
                    wval[i] -= d;
                }
            }
            // gather back
            auto& cj = cols[j];
            cj.clear();
            for (Int i : wpat) {
                Real v = wval[i];
                wmark[i] = 0; wval[i] = 0.0;
                if (std::fabs(v) > zeroTol) cj.emplace_back(i, v);
                else rowCnt[i]--;
            }
            colCnt[j] = (Int)cj.size();
            touchCol(j);
        }
        rows[r].clear(); rows[r].shrink_to_fit();

        // refresh row buckets for rows whose count changed
        for (auto& lp : Lcol) touchRow(lp.first);
        for (Int j : rowCandidates) for (auto& e : cols[j]) touchRow(e.first);
    }
    lStart[m] = (Int)lRow.size();
    uStart[m] = (Int)uCol.size();

    bool ok = (nPivoted == m);
    if (!ok) {
        // Pair the leftover rows and basis positions so the factorization stays
        // well defined; report the positions so the caller can repair the basis.
        std::vector<Int> freeRows, freeCols;
        for (Int i = 0; i < m; ++i) if (!rowDone[i]) freeRows.push_back(i);
        for (Int j = 0; j < m; ++j) if (!colDone[j]) freeCols.push_back(j);
        Int t = 0;
        for (Int step = nPivoted; step < m && t < (Int)freeCols.size(); ++step, ++t) {
            Int r = freeRows[t], jc = freeCols[t];
            pivotRow_[step] = r; pivotCol_[step] = jc;
            rowPos_[r] = step;  colPos_[jc] = step;
            uDiag_[step] = 1.0;
            lStart[step] = (Int)lRow.size();
            uStart[step] = (Int)uCol.size();
            singularPositions.push_back(jc);
        }
        lStart[m] = (Int)lRow.size();
        uStart[m] = (Int)uCol.size();
    }

    // --- remap original indices to pivot order --------------------------------
    lPtr_.assign(m + 1, 0); uPtr_.assign(m + 1, 0);
    lIdx_.resize(lRow.size()); lVal_.resize(lv.size());
    uIdx_.resize(uCol.size()); uVal_.resize(uv.size());
    for (Int k = 0; k <= m; ++k) { lPtr_[k] = lStart[k]; uPtr_[k] = uStart[k]; }
    for (size_t p = 0; p < lRow.size(); ++p) { lIdx_[p] = rowPos_[lRow[p]]; lVal_[p] = lv[p]; }
    for (size_t p = 0; p < uCol.size(); ++p) { uIdx_[p] = colPos_[uCol[p]]; uVal_[p] = uv[p]; }
    lnz_ = (Long)lIdx_.size(); unz_ = (Long)uIdx_.size() + m;
    growth_ = umax / inputMax;
    if (minPivot_ == kBigReal) minPivot_ = 1.0;
    return ok;
}

// ---------------------------------------------------------------------------
void BasisFactor::ftranDense(std::vector<Real>& v) const {
    // Input indexed by original row; output indexed by basis position.
    std::vector<Real>& w = work_;
    w.assign(m, 0.0);
    for (Int k = 0; k < m; ++k) w[k] = v[pivotRow_[k]];
    // L w = z
    for (Int k = 0; k < m; ++k) {
        Real x = w[k];
        if (x == 0.0) continue;
        for (Int p = lPtr_[k]; p < lPtr_[k + 1]; ++p) w[lIdx_[p]] -= lVal_[p] * x;
    }
    // U t = w
    for (Int k = m - 1; k >= 0; --k) {
        Real s = w[k];
        for (Int p = uPtr_[k]; p < uPtr_[k + 1]; ++p) s -= uVal_[p] * w[uIdx_[p]];
        w[k] = s / uDiag_[k];
    }
    std::fill(v.begin(), v.end(), 0.0);
    for (Int k = 0; k < m; ++k) v[pivotCol_[k]] = w[k];
    applyEtasForward(v);
}

void BasisFactor::btranDense(std::vector<Real>& v) const {
    // Input indexed by basis position; output indexed by original row.
    applyEtasBackward(v);
    std::vector<Real>& w = work_;
    w.assign(m, 0.0);
    for (Int k = 0; k < m; ++k) w[k] = v[pivotCol_[k]];
    // U^T s = d   (forward sweep using row storage)
    for (Int k = 0; k < m; ++k) {
        Real s = w[k] / uDiag_[k];
        w[k] = s;
        if (s == 0.0) continue;
        for (Int p = uPtr_[k]; p < uPtr_[k + 1]; ++p) w[uIdx_[p]] -= uVal_[p] * s;
    }
    // L^T w = s   (backward sweep using column storage)
    for (Int k = m - 1; k >= 0; --k) {
        Real s = w[k];
        for (Int p = lPtr_[k]; p < lPtr_[k + 1]; ++p) s -= lVal_[p] * w[lIdx_[p]];
        w[k] = s;
    }
    std::fill(v.begin(), v.end(), 0.0);
    for (Int k = 0; k < m; ++k) v[pivotRow_[k]] = w[k];
}

void BasisFactor::applyEtasForward(std::vector<Real>& v) const {
    for (const Eta& e : etas_) {
        Real vp = v[e.pivot] / e.pivotVal;
        if (vp == 0.0) continue;
        for (size_t t = 0; t < e.idx.size(); ++t) v[e.idx[t]] -= e.val[t] * vp;
        v[e.pivot] = vp;
    }
}

void BasisFactor::applyEtasBackward(std::vector<Real>& v) const {
    for (Int t = (Int)etas_.size() - 1; t >= 0; --t) {
        const Eta& e = etas_[t];
        Real s = v[e.pivot];
        for (size_t q = 0; q < e.idx.size(); ++q) s -= e.val[q] * v[e.idx[q]];
        v[e.pivot] = s / e.pivotVal;
    }
}

void BasisFactor::ftran(SparseVector& v, Real dropTol) const {
    std::vector<Real> d(m, 0.0);
    for (Int i : v.idx) d[i] = v.dense[i];
    ftranDense(d);
    v.clear();
    for (Int i = 0; i < m; ++i) if (std::fabs(d[i]) > dropTol) v.set(i, d[i]);
}

void BasisFactor::btran(SparseVector& v, Real dropTol) const {
    std::vector<Real> d(m, 0.0);
    for (Int i : v.idx) d[i] = v.dense[i];
    btranDense(d);
    v.clear();
    for (Int i = 0; i < m; ++i) if (std::fabs(d[i]) > dropTol) v.set(i, d[i]);
}

bool BasisFactor::update(Int pivotPos, const SparseVector& alphaQ, Real pivotTol) {
    Real ap = alphaQ.dense[pivotPos];
    if (std::fabs(ap) < pivotTol) return false;
    Eta e;
    e.pivot = pivotPos;
    e.pivotVal = ap;
    e.idx.reserve(alphaQ.idx.size());
    e.val.reserve(alphaQ.idx.size());
    for (Int i : alphaQ.idx) {
        if (i == pivotPos) continue;
        Real v = alphaQ.dense[i];
        if (v == 0.0) continue;
        e.idx.push_back(i); e.val.push_back(v);
    }
    etaNnz_ += (Long)e.idx.size() + 1;
    etas_.push_back(std::move(e));
    return true;
}

Real BasisFactor::conditionEstimate() const {
    Real lo = kBigReal, hi = 0.0;
    for (Int k = 0; k < m; ++k) {
        Real a = std::fabs(uDiag_[k]);
        if (a == 0.0) return kBigReal;
        lo = std::min(lo, a); hi = std::max(hi, a);
    }
    if (lo == kBigReal) return 1.0;
    return hi / lo;
}

} // namespace igaos
