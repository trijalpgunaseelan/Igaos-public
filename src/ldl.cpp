#include "igaos/ldl.hpp"
#include <list>

namespace igaos {

// ---------------------------------------------------------------------------
// Approximate minimum degree.
//
// Elimination is simulated on the quotient graph: each remaining node keeps a
// list of adjacent *variables* and a list of adjacent *elements* (previously
// eliminated pivots).  Eliminating p creates element Lp = union of p's
// variable list and the elements it touches.  Degrees are then refreshed with
// the standard AMD upper bound
//
//   d_i  <=  min( n - k,  d_i + |Lp \ {i}|,
//                 |Av_i \ Lp| + |Lp \ {i}| + sum_{e in Ae_i, e != p} |Le \ Lp| )
//
// which is computed in one pass with the classic w[] marking trick, avoiding
// the set unions an exact minimum-degree code would need.
// ---------------------------------------------------------------------------
void approximateMinimumDegree(Int n,
                              const std::vector<Int>& Ap, const std::vector<Int>& Ai,
                              std::vector<Int>& perm, std::vector<Int>& iperm) {
    perm.assign(n, 0); iperm.assign(n, 0);
    if (n == 0) return;

    std::vector<std::vector<Int>> Av(n), Ae(n), Le(n);
    std::vector<Int>  deg(n, 0);
    std::vector<uint8_t> alive(n, 1), isElem(n, 0);

    for (Int j = 0; j < n; ++j) {
        Av[j].reserve(Ap[j + 1] - Ap[j]);
        for (Int p = Ap[j]; p < Ap[j + 1]; ++p) {
            Int i = Ai[p];
            if (i != j) Av[j].push_back(i);
        }
        deg[j] = (Int)Av[j].size();
    }

    // Degree buckets with lazy validation.
    std::vector<std::vector<Int>> bucket(n + 1);
    for (Int j = 0; j < n; ++j) bucket[std::min(deg[j], n)].push_back(j);
    Int minDeg = 0;

    std::vector<Int>  wTag(n, -1), wCnt(n, 0);
    std::vector<uint8_t> inLp(n, 0);
    std::vector<Int>  Lp;
    Int mark = 0;

    for (Int k = 0; k < n; ++k) {
        // ---- select the minimum degree node -------------------------------
        Int p = kNone;
        while (minDeg <= n) {
            auto& bk = bucket[minDeg];
            while (!bk.empty()) {
                Int cand = bk.back(); bk.pop_back();
                if (!alive[cand] || isElem[cand]) continue;
                if (std::min(deg[cand], n) != minDeg) continue;   // stale
                p = cand; break;
            }
            if (p != kNone) break;
            ++minDeg;
        }
        if (p == kNone) {                       // isolated leftovers
            for (Int j = 0; j < n; ++j) if (alive[j] && !isElem[j]) { p = j; break; }
            if (p == kNone) break;
        }
        perm[k] = p; iperm[p] = k;
        alive[p] = 0;

        // ---- form element Lp ----------------------------------------------
        Lp.clear();
        ++mark;
        for (Int i : Av[p]) if (alive[i] && !isElem[i] && !inLp[i]) { inLp[i] = 1; Lp.push_back(i); }
        for (Int e : Ae[p]) {
            for (Int i : Le[e]) if (alive[i] && !isElem[i] && !inLp[i]) { inLp[i] = 1; Lp.push_back(i); }
            Le[e].clear(); Le[e].shrink_to_fit();
            isElem[e] = 2;                       // absorbed
        }
        Av[p].clear(); Av[p].shrink_to_fit();
        Ae[p].clear(); Ae[p].shrink_to_fit();

        if (Lp.empty()) { for (Int i : Lp) inLp[i] = 0; continue; }

        // ---- |Le \ Lp| for every element adjacent to Lp --------------------
        for (Int i : Lp) {
            size_t keep = 0;
            for (size_t t = 0; t < Ae[i].size(); ++t) {
                Int e = Ae[i][t];
                if (isElem[e] == 2 || e == p) continue;   // absorbed
                Ae[i][keep++] = e;
                if (wTag[e] != mark) { wTag[e] = mark; wCnt[e] = (Int)Le[e].size(); }
                wCnt[e]--;
            }
            Ae[i].resize(keep);
        }

        // ---- refresh degrees ------------------------------------------------
        Int lpSize = (Int)Lp.size();
        for (Int i : Lp) {
            // prune the variable list of i: drop dead nodes and nodes now in Lp
            size_t keep = 0;
            Int extAv = 0;
            for (size_t t = 0; t < Av[i].size(); ++t) {
                Int v = Av[i][t];
                if (!alive[v] || isElem[v]) continue;
                if (inLp[v]) continue;            // covered by the new element
                Av[i][keep++] = v; ++extAv;
            }
            Av[i].resize(keep);
            Int d = extAv + (lpSize - 1);
            for (Int e : Ae[i]) {
                Int ext = (wTag[e] == mark) ? std::max(0, wCnt[e]) : (Int)Le[e].size();
                d += ext;
            }
            Ae[i].push_back(p);
            d = std::min(d, n - k - 1);
            if (d < 0) d = 0;
            if (d != deg[i]) {
                deg[i] = d;
                bucket[std::min(d, n)].push_back(i);
                if (d < minDeg) minDeg = d;
            }
        }
        Le[p] = Lp;
        isElem[p] = 1;
        for (Int i : Lp) inLp[i] = 0;
    }

    // Any node never selected (shouldn't happen) gets appended.
    std::vector<uint8_t> placed(n, 0);
    Int cnt = 0;
    for (Int k = 0; k < n; ++k) if (perm[k] >= 0 && perm[k] < n) { placed[perm[k]] = 1; ++cnt; }
    if (cnt < n) {
        Int k = cnt;
        for (Int j = 0; j < n; ++j) if (!placed[j]) { perm[k] = j; iperm[j] = k; ++k; }
    }
}

// ---------------------------------------------------------------------------
void LdlFactor::analyze(Int nn, const std::vector<Int>& Ap, const std::vector<Int>& Ai) {
    n = nn;
    // Build the full symmetric adjacency (both triangles, no diagonal) for AMD.
    std::vector<Int> cnt(n + 1, 0);
    for (Int j = 0; j < n; ++j)
        for (Int p = Ap[j]; p < Ap[j + 1]; ++p) {
            Int i = Ai[p];
            if (i == j) continue;
            cnt[i + 1]++; cnt[j + 1]++;
        }
    std::vector<Int> Fp(n + 1, 0);
    for (Int j = 0; j < n; ++j) Fp[j + 1] = Fp[j] + cnt[j + 1];
    std::vector<Int> Fi(Fp[n]);
    std::vector<Int> next(Fp.begin(), Fp.end() - 1);
    for (Int j = 0; j < n; ++j)
        for (Int p = Ap[j]; p < Ap[j + 1]; ++p) {
            Int i = Ai[p];
            if (i == j) continue;
            Fi[next[i]++] = j; Fi[next[j]++] = i;
        }
    approximateMinimumDegree(n, Fp, Fi, perm_, iperm_);

    // Permuted upper-triangle pattern, then the elimination tree and column
    // counts (Davis's symbolic analysis).
    std::vector<Int> Pp(n + 1, 0), Pi;
    {
        std::vector<Int> c(n, 0);
        for (Int j = 0; j < n; ++j)
            for (Int p = Ap[j]; p < Ap[j + 1]; ++p) {
                Int i = Ai[p];
                Int pi = iperm_[i], pj = iperm_[j];
                Int a = std::min(pi, pj), b = std::max(pi, pj);
                c[b]++; (void)a;
            }
        for (Int j = 0; j < n; ++j) Pp[j + 1] = Pp[j] + c[j];
        Pi.resize(Pp[n]);
        std::vector<Int> nx(Pp.begin(), Pp.end() - 1);
        for (Int j = 0; j < n; ++j)
            for (Int p = Ap[j]; p < Ap[j + 1]; ++p) {
                Int i = Ai[p];
                Int pi = iperm_[i], pj = iperm_[j];
                Int a = std::min(pi, pj), b = std::max(pi, pj);
                Pi[nx[b]++] = a;
            }
    }

    parent_.assign(n, kNone);
    Lnz_.assign(n, 0);
    flag_.assign(n, 0);
    for (Int k = 0; k < n; ++k) {
        parent_[k] = kNone;
        flag_[k] = k;
        for (Int p = Pp[k]; p < Pp[k + 1]; ++p) {
            Int i = Pi[p];
            if (i >= k) continue;
            for (; flag_[i] != k; i = parent_[i]) {
                if (parent_[i] == kNone) parent_[i] = k;
                Lnz_[i]++;
                flag_[i] = k;
            }
        }
    }
    Lp_.assign(n + 1, 0);
    for (Int k = 0; k < n; ++k) Lp_[k + 1] = Lp_[k] + Lnz_[k];
    Li_.assign(Lp_[n], 0);
    Lx_.assign(Lp_[n], 0.0);
    D_.assign(n, 0.0);
    y_.assign(n, 0.0);
    work_.assign(n, 0.0);
    pattern_.assign(n, 0);
}

bool LdlFactor::factorize(const std::vector<Int>& Ap, const std::vector<Int>& Ai,
                          const std::vector<Real>& Ax,
                          const std::vector<int8_t>& expectedSign,
                          Real reg, Real pivotTol) {
    nreg_ = 0;
    minPivot_ = kBigReal;
    std::fill(Lnz_.begin(), Lnz_.end(), 0);
    std::fill(y_.begin(), y_.end(), 0.0);
    std::fill(D_.begin(), D_.end(), 0.0);

    // Permuted upper triangle with values, column by column.
    std::vector<std::vector<std::pair<Int, Real>>> P(n);
    for (Int j = 0; j < n; ++j)
        for (Int p = Ap[j]; p < Ap[j + 1]; ++p) {
            Int i = Ai[p];
            Int pi = iperm_[i], pj = iperm_[j];
            Int a = std::min(pi, pj), b = std::max(pi, pj);
            P[b].emplace_back(a, Ax[p]);
        }

    for (Int k = 0; k < n; ++k) {
        Int top = n;
        y_[k] = 0.0;
        flag_[k] = k;
        for (auto& e : P[k]) {
            Int i = e.first;
            if (i > k) continue;
            y_[i] += e.second;
            Int len = 0;
            for (; flag_[i] != k; i = parent_[i]) {
                pattern_[len++] = i;
                flag_[i] = k;
                if (parent_[i] == kNone) break;
            }
            while (len > 0) pattern_[--top] = pattern_[--len];
        }
        Real d = y_[k];
        y_[k] = 0.0;
        for (; top < n; ++top) {
            Int i = pattern_[top];
            Real yi = y_[i];
            y_[i] = 0.0;
            Int p2 = Lp_[i] + Lnz_[i];
            for (Int p = Lp_[i]; p < p2; ++p) y_[Li_[p]] -= Lx_[p] * yi;
            Real lki = yi / D_[i];
            d -= lki * yi;
            Li_[p2] = k; Lx_[p2] = lki; Lnz_[i]++;
        }
        int8_t s = expectedSign[perm_[k]];
        Real sd = (Real)s * d;
        if (!(sd > pivotTol)) { d = (Real)s * reg; ++nreg_; }
        D_[k] = d;
        minPivot_ = std::min(minPivot_, std::fabs(d));
    }
    return nreg_ == 0;
}

void LdlFactor::solve(std::vector<Real>& b) const {
    std::vector<Real>& x = work_;
    for (Int k = 0; k < n; ++k) x[k] = b[perm_[k]];
    for (Int k = 0; k < n; ++k) {
        Real xk = x[k];
        if (xk != 0.0)
            for (Int p = Lp_[k]; p < Lp_[k] + Lnz_[k]; ++p) x[Li_[p]] -= Lx_[p] * xk;
    }
    for (Int k = 0; k < n; ++k) x[k] /= D_[k];
    for (Int k = n - 1; k >= 0; --k) {
        Real s = x[k];
        for (Int p = Lp_[k]; p < Lp_[k] + Lnz_[k]; ++p) s -= Lx_[p] * x[Li_[p]];
        x[k] = s;
    }
    for (Int k = 0; k < n; ++k) b[perm_[k]] = x[k];
}

} // namespace igaos
