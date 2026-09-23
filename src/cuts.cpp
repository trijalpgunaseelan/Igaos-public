#include "igaos/cuts.hpp"
#include <algorithm>
#include <cstring>

namespace igaos {

// ===========================================================================
//  Pool
// ===========================================================================
namespace {

uint64_t mixBits(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

// Hash of the normalized halfspace.  Coefficients are quantized before hashing
// so that two derivations that agree to eight digits collide on purpose.
uint64_t cutKey(const Cut& c) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t t = 0; t < c.idx.size(); ++t) {
        h = mixBits(h, (uint64_t)c.idx[t]);
        long long q = (long long)std::llround(c.val[t] * 1e8);
        h = mixBits(h, (uint64_t)q);
    }
    h = mixBits(h, (uint64_t)std::llround(c.rhs * 1e8));
    return h;
}

} // namespace

bool CutPool::offer(Cut c, CutStats& st) {
    if (c.idx.empty()) { ++st.rejectedWeak; return false; }
    if ((Int)cuts_.size() >= lim_.maxTotal) return false;
    if (c.idx.size() != c.val.size()) { ++st.rejectedUnstable; return false; }

    // --- sort and merge repeated columns ------------------------------------
    // A separator that accumulates into a dense workspace can emit the same
    // column twice; silently keeping both halves would understate the
    // coefficient and make the cut invalid, so the pool merges defensively
    // rather than trusting its callers.
    {
        std::vector<std::pair<Int, Real>> terms;
        terms.reserve(c.idx.size());
        for (size_t t = 0; t < c.idx.size(); ++t) {
            if (c.idx[t] < 0 || c.idx[t] >= ncol_) { ++st.rejectedUnstable; return false; }
            terms.emplace_back(c.idx[t], c.val[t]);
        }
        std::sort(terms.begin(), terms.end(),
                  [](const std::pair<Int, Real>& a, const std::pair<Int, Real>& b) {
                      return a.first < b.first;
                  });
        c.idx.clear(); c.val.clear();
        for (const auto& pr : terms) {
            if (!c.idx.empty() && c.idx.back() == pr.first) c.val.back() += pr.second;
            else { c.idx.push_back(pr.first); c.val.push_back(pr.second); }
        }
    }

    // --- drop numeric dust, compensating the right-hand side ----------------
    Int k = 0;
    Real maxAbs = 0;
    for (size_t t = 0; t < c.idx.size(); ++t) maxAbs = std::max(maxAbs, std::fabs(c.val[t]));
    if (maxAbs <= lim_.minCoef) { ++st.rejectedWeak; return false; }

    Real dropBelow = std::max(lim_.minCoef, 1e-12 * maxAbs);
    for (size_t t = 0; t < c.idx.size(); ++t) {
        Real a = c.val[t];
        if (std::fabs(a) <= dropBelow) {
            // sum a_j x_j >= rhs stays valid without this term only if the rhs
            // absorbs the most the term could ever contribute.
            bool dropped = false;
            if (lower_ && upper_ && (size_t)c.idx[t] < lower_->size()) {
                Real l = (*lower_)[c.idx[t]], u = (*upper_)[c.idx[t]];
                Real hi = -kBigReal;
                if (a >= 0) hi = isInf(u)    ? kBigReal : a * u;
                else        hi = isNegInf(l) ? kBigReal : a * l;
                if (hi < kBigReal) { c.rhs -= hi; dropped = true; }
            }
            if (dropped) continue;
        }
        c.idx[k] = c.idx[t];
        c.val[k] = a;
        ++k;
    }
    c.idx.resize(k); c.val.resize(k);
    if (c.idx.empty()) { ++st.rejectedWeak; return false; }

    Real minAbs = kBigReal;
    maxAbs = 0;
    for (Real v : c.val) { Real a = std::fabs(v); maxAbs = std::max(maxAbs, a); minAbs = std::min(minAbs, a); }
    if (maxAbs > lim_.maxAbsCoef) { ++st.rejectedUnstable; return false; }
    if (maxAbs / std::max(minAbs, 1e-300) > lim_.maxDynamism) { ++st.rejectedUnstable; return false; }

    Real scale = 1.0 / maxAbs;
    for (Real& v : c.val) v *= scale;
    c.rhs *= scale;
    c.violation *= scale;
    if (!isFinite(c.rhs) || std::isnan(c.rhs)) { ++st.rejectedUnstable; return false; }

    if (c.nnz() > lim_.maxNnz(ncol_)) { ++st.rejectedDense; return false; }

    Real norm = 0;
    for (Real v : c.val) norm += v * v;
    norm = std::sqrt(norm);
    c.efficacy = (norm > 0) ? c.violation / norm : 0.0;
    if (c.efficacy < lim_.minEfficacy) { ++st.rejectedWeak; return false; }

    // Shave the right-hand side so floating-point noise in the derivation can
    // never make the cut strictly tighter than the exact halfspace it models.
    c.rhs -= lim_.safetyRelax * (1.0 + std::fabs(c.rhs));
    c.violation -= lim_.safetyRelax * (1.0 + std::fabs(c.rhs));
    if (c.violation <= 0) { ++st.rejectedWeak; return false; }

    if (lim_.referencePoint && lim_.referencePoint->size() >= (size_t)ncol_) {
        Real act = c.activity(*lim_.referencePoint);
        Real slack = act - c.rhs;
        Real tol = 1e-6 * (1.0 + std::fabs(c.rhs));
        if (slack < -tol) {
            ++st.rejectedInvalid;
            if (!st.firstInvalidOrigin) st.firstInvalidOrigin = c.origin;
            st.worstInvalidViolation = std::max(st.worstInvalidViolation, -slack);
            return false;
        }
    }

    uint64_t key = cutKey(c);
    for (uint64_t existing : keys_)
        if (existing == key) { ++st.rejectedDuplicate; return false; }

    keys_.push_back(key);
    cuts_.push_back(std::move(c));
    return true;
}

void CutPool::keepOnly(const std::vector<uint8_t>& keep) {
    std::vector<Cut> kept;
    std::vector<uint64_t> keptKeys;
    for (size_t i = 0; i < cuts_.size(); ++i)
        if (i < keep.size() && keep[i]) { kept.push_back(cuts_[i]); keptKeys.push_back(keys_[i]); }
    cuts_.swap(kept);
    keys_.swap(keptKeys);
}

// ===========================================================================
//  Gomory mixed-integer cuts
// ===========================================================================
//
//  A tableau row for a basic variable x_B(p) reads
//
//        x_B(p) + sum_{j nonbasic} alpha_j v_j = <constant>
//
//  Rewriting each nonbasic variable as its non-negative distance from the bound
//  it currently sits at (t_j = v_j - l_j at lower, t_j = u_j - v_j at upper)
//  puts the row in the canonical form
//
//        x_B(p) + sum_j abar_j t_j = b,     t_j >= 0
//
//  and with f0 = frac(b) the GMI inequality is
//
//        sum_{j integer} phi(abar_j) t_j + sum_{j continuous} psi(abar_j) t_j >= 1
//
//        phi(a) = f_a / f0            if f_a <= f0,   else (1 - f_a) / (1 - f0)
//        psi(a) = a / f0              if a >= 0,      else -a / (1 - f0)
//
//  A *free* nonbasic sitting at zero has no such non-negative form, so a row
//  that touches one is skipped rather than fudged -- that is exactly the case
//  where the cut would be invalid.
// ===========================================================================
Int separateGomory(Simplex& sx, const Model& m, const std::vector<uint8_t>& isIntCol,
                   const CutLimits& lim, CutPool& pool, CutStats& st, Int maxCuts)
{
    const Int n = m.numCol();
    const Int nr = m.numRow();
    const auto& basis = sx.basis();
    const auto& value = sx.values();
    const auto& stat  = sx.statuses();
    const auto& lo    = sx.lower();
    const auto& up    = sx.upper();
    if ((Int)value.size() < n + nr) return 0;

    // Row-wise A, so a logical variable's coefficient can be pushed back onto
    // the structural columns of its row.
    SparseMatrix At = m.A.transpose();     // column i of At is row i of A

    std::vector<std::pair<Real, Int>> cand;   // (-distance from integrality, basis position)
    for (Int p = 0; p < (Int)basis.size(); ++p) {
        Int k = basis[p];
        if (k < 0 || k >= n) continue;
        if (!isIntCol[k]) continue;
        Real v = value[k];
        Real f = v - std::floor(v);
        Real d = std::min(f, 1.0 - f);
        if (d < lim.minFrac) continue;
        cand.emplace_back(-d, p);
    }
    if (cand.empty()) return 0;
    std::sort(cand.begin(), cand.end());

    std::vector<Real> x(m.numCol());
    for (Int j = 0; j < n; ++j) x[j] = value[j];

    // tableauRow writes through SparseVector::set, which indexes preallocated
    // occupancy flags -- both scratch vectors must be sized before first use.
    SparseVector row, rho;
    row.resize(n + nr);
    rho.resize(nr);
    std::vector<Real>    dense(n, 0.0);
    std::vector<uint8_t> seen(n, 0);
    std::vector<Int>     touched;
    Int accepted = 0;
    Int tried = 0;
    const Int maxTried = std::max<Int>(maxCuts * 3, 30);

    for (const auto& cd : cand) {
        if (accepted >= maxCuts || tried >= maxTried) break;
        ++tried;
        Int p = cd.second;
        Int kb = basis[p];
        Real b = value[kb];
        Real f0 = b - std::floor(b);
        if (f0 < lim.minFrac || f0 > lim.maxFrac) continue;

        sx.tableauRow(p, row, rho);
        if (row.nnz() == 0) continue;
        // A tableau row with enormous entries means the basis that produced it is
        // close to singular; a cut read off it is numerical fiction.
        if (row.infNorm() > lim.maxTableauNorm) { ++st.rejectedUnstable; continue; }

        // ---- build the cut over extended variables -------------------------
        bool ok = true;
        Real rhs = 1.0;
        for (Int idx : touched) { dense[idx] = 0.0; seen[idx] = 0; }
        touched.clear();

        // Occupancy must be tracked with an explicit flag, never by testing
        // dense[j] against zero: a logical variable's substitution can cancel a
        // structural coefficient exactly, and a later deposit on the same column
        // would then register it as newly touched and emit the column twice.
        auto deposit = [&](Int j, Real v) {
            if (v == 0.0) return;
            if (!seen[j]) { seen[j] = 1; touched.push_back(j); }
            dense[j] += v;
        };

        for (Int k : row.idx) {
            Real a = row.dense[k];
            if (a == 0.0) continue;
            VarStatus s = stat[k];
            if (s == VarStatus::Basic) continue;
            if (s == VarStatus::Fixed) continue;             // t == 0 identically
            if (lo[k] == up[k]) continue;

            Real abar; int dir;
            if (s == VarStatus::AtLower) {
                if (!isFinite(lo[k])) { ok = false; break; }
                abar = a;  dir = 1;
            } else if (s == VarStatus::AtUpper) {
                if (!isFinite(up[k])) { ok = false; break; }
                abar = -a; dir = -1;
            } else {
                ok = false; break;                            // free nonbasic at zero
            }

            // The integer branch of the GMI function is only valid when t_j
            // itself takes integer values, and t_j is the distance from the
            // bound the variable currently sits at -- so a genuinely integer
            // variable resting on a *fractional* bound (which presolve's bound
            // tightening can produce) must be treated as continuous here.  The
            // continuous branch is valid for any real t_j >= 0, so this is a
            // safe downgrade rather than a lost cut.
            bool intK = (k < n) && isIntCol[k] != 0;
            if (intK) {
                Real atBound = (dir > 0) ? lo[k] : up[k];
                if (!isIntegral(atBound, 1e-9)) intK = false;
            }
            Real coef;
            if (intK) {
                Real fj = abar - std::floor(abar);
                if (fj < 0.0) fj = 0.0;
                if (fj > 1.0) fj = 1.0;
                coef = (fj <= f0) ? fj / f0 : (1.0 - fj) / (1.0 - f0);
            } else {
                coef = (abar >= 0.0) ? abar / f0 : -abar / (1.0 - f0);
            }
            if (!(coef > lim.minCoef)) continue;
            if (!std::isfinite(coef)) { ok = false; break; }

            Real signed_ = (dir > 0) ? coef : -coef;
            if (dir > 0) rhs += coef * lo[k];
            else         rhs -= coef * up[k];

            if (k < n) {
                deposit(k, signed_);
            } else {
                Int i = k - n;                                // logical of row i: s_i = A_i x
                for (Int t = At.colPtr[i]; t < At.colPtr[i + 1]; ++t)
                    deposit(At.rowIdx[t], signed_ * At.val[t]);
            }
        }
        if (!ok || touched.empty()) continue;

        Cut c;
        c.origin = "gomory";
        std::sort(touched.begin(), touched.end());
        c.idx.reserve(touched.size());
        c.val.reserve(touched.size());
        for (Int j : touched) {
            if (std::fabs(dense[j]) <= lim.minCoef) continue;
            c.idx.push_back(j);
            c.val.push_back(dense[j]);
        }
        if (c.idx.empty()) continue;
        c.rhs = rhs;
        c.violation = rhs - c.activity(x);
        if (c.violation <= 0.0) continue;
        if (pool.offer(std::move(c), st)) { ++accepted; ++st.gomory; }
    }
    for (Int idx : touched) { dense[idx] = 0.0; seen[idx] = 0; }
    return accepted;
}

// ===========================================================================
//  Knapsack cover cuts
// ===========================================================================
namespace {

// One side of a row, reduced to a pure binary knapsack  sum w_j y_j <= cap
// with every w_j > 0, where y_j is x_j or its complement 1 - x_j.
struct Knapsack {
    std::vector<Int>  col;
    std::vector<Real> w;
    std::vector<uint8_t> complemented;
    std::vector<Real> z;              // LP value of y_j
    Real cap = 0;
};

bool buildKnapsack(const SparseMatrix& At, Int i, Real rhs, Real sign,
                   const Model& m, const std::vector<Real>& x, Knapsack& kp)
{
    kp.col.clear(); kp.w.clear(); kp.complemented.clear(); kp.z.clear();
    Real cap = sign * rhs;
    for (Int t = At.colPtr[i]; t < At.colPtr[i + 1]; ++t) {
        Int j = At.rowIdx[t];
        Real a = sign * At.val[t];
        if (a == 0.0) continue;
        // binary means: integer with bounds exactly [0,1]
        if (m.colType[j] == VarType::Continuous) return false;
        if (m.colLower[j] != 0.0 || m.colUpper[j] != 1.0) return false;
        if (a > 0) {
            kp.col.push_back(j); kp.w.push_back(a);
            kp.complemented.push_back(0); kp.z.push_back(x[j]);
        } else {
            kp.col.push_back(j); kp.w.push_back(-a);
            kp.complemented.push_back(1); kp.z.push_back(1.0 - x[j]);
            cap += -a;                    // x_j = 1 - y_j moves |a| to the rhs
        }
    }
    if (kp.col.size() < 2 || kp.col.size() > 5000) return false;
    if (cap < 0) return false;
    kp.cap = cap;
    return true;
}

} // namespace

Int separateCover(const Model& m, const std::vector<Real>& x,
                  const CutLimits& lim, CutPool& pool, CutStats& st, Int maxCuts)
{
    SparseMatrix At = m.A.transpose();
    Int accepted = 0;
    Knapsack kp;

    for (Int i = 0; i < m.numRow() && accepted < maxCuts; ++i) {
        for (int side = 0; side < 2; ++side) {
            if (accepted >= maxCuts) break;
            Real rhs; Real sign;
            if (side == 0) { if (!isFinite(m.rowUpper[i])) continue; rhs = m.rowUpper[i]; sign = 1.0; }
            else           { if (isNegInf(m.rowLower[i]))  continue; rhs = m.rowLower[i]; sign = -1.0; }
            if (!buildKnapsack(At, i, rhs, sign, m, x, kp)) continue;

            const size_t nItem = kp.col.size();
            Real total = 0;
            for (Real w : kp.w) total += w;
            if (total <= kp.cap + 1e-9) continue;       // row cannot be binding

            // Greedy separation: a most-violated cover is the one that reaches
            // capacity while spending as little (1 - z_j) as possible.
            std::vector<size_t> order(nItem);
            for (size_t t = 0; t < nItem; ++t) order[t] = t;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                Real ka = (1.0 - kp.z[a]) / std::max(kp.w[a], 1e-12);
                Real kb = (1.0 - kp.z[b]) / std::max(kp.w[b], 1e-12);
                if (ka != kb) return ka < kb;
                return kp.w[a] > kp.w[b];
            });

            std::vector<uint8_t> inCover(nItem, 0);
            Real sum = 0;
            size_t taken = 0;
            for (size_t t = 0; t < nItem; ++t) {
                size_t j = order[t];
                inCover[j] = 1; sum += kp.w[j]; ++taken;
                if (sum > kp.cap + 1e-9) break;
            }
            if (sum <= kp.cap + 1e-9) continue;

            // Make it minimal: drop the most expensive members that the cover
            // can afford to lose.
            for (size_t t = nItem; t-- > 0; ) {
                size_t j = order[t];
                if (!inCover[j]) continue;
                if (taken <= 1) break;
                if (sum - kp.w[j] > kp.cap + 1e-9) { inCover[j] = 0; sum -= kp.w[j]; --taken; }
            }
            if (taken < 2) continue;

            Real coverZ = 0, maxW = 0;
            for (size_t t = 0; t < nItem; ++t)
                if (inCover[t]) { coverZ += kp.z[t]; maxW = std::max(maxW, kp.w[t]); }

            Real viol = coverZ - (Real)(taken - 1);
            if (viol <= 1e-6) continue;

            // Extension: any item at least as heavy as the heaviest cover member
            // can join without weakening the inequality.
            std::vector<uint8_t> inCut = inCover;
            for (size_t t = 0; t < nItem; ++t)
                if (!inCut[t] && kp.w[t] >= maxW - 1e-12) inCut[t] = 1;

            // sum_{E} y_j <= |C| - 1, un-complemented and flipped to ">=".
            Real rhsCut = -(Real)(taken - 1);
            std::vector<std::pair<Int, Real>> terms;
            for (size_t t = 0; t < nItem; ++t) {
                if (!inCut[t]) continue;
                if (kp.complemented[t]) { terms.emplace_back(kp.col[t],  1.0); rhsCut += 1.0; }
                else                    { terms.emplace_back(kp.col[t], -1.0); }
            }
            if (terms.empty()) continue;
            std::sort(terms.begin(), terms.end());

            Cut c;
            c.origin = "cover";
            for (auto& pr : terms) {
                if (!c.idx.empty() && c.idx.back() == pr.first) { c.val.back() += pr.second; continue; }
                c.idx.push_back(pr.first); c.val.push_back(pr.second);
            }
            c.rhs = rhsCut;
            c.violation = c.rhs - c.activity(x);
            if (c.violation <= 1e-7) continue;
            if (pool.offer(std::move(c), st)) { ++accepted; ++st.cover; }
        }
    }
    return accepted;
}

// ===========================================================================
//  Complemented mixed-integer rounding (c-MIR)
// ===========================================================================
//
//  For  sum_j a_j x'_j - s <= b  with x' integer >= 0 and s >= 0 continuous,
//  writing f0 = frac(b) > 0 and f_j = frac(a_j), the MIR inequality is
//
//        sum_j [ floor(a_j) + (f_j - f0)^+ / (1 - f0) ] x'_j
//              - s / (1 - f0)  <=  floor(b)
//
//  To get there from a real row we (1) substitute every variable against the
//  bound it currently sits nearer to, so all variables are non-negative,
//  (2) drop continuous terms with a positive coefficient, which is a valid
//  relaxation because they only push the left-hand side up, (3) collect the
//  remaining continuous terms into s, and (4) scale the row by 1/delta for a
//  family of candidate divisors, since MIR is not scale-invariant and the
//  right divisor is what turns a weak row into a strong cut.
// ===========================================================================
namespace {

struct MirTerm {
    Int  col;
    Real a;              // coefficient after bound substitution
    bool integer;
    bool upperSub;       // true if x' = u - x, false if x' = x - l
    Real bound;          // the l or u used
};

} // namespace

Int separateMir(const Model& m, const std::vector<Real>& x,
                const CutLimits& lim, CutPool& pool, CutStats& st, Int maxCuts)
{
    SparseMatrix At = m.A.transpose();
    Int accepted = 0;
    std::vector<MirTerm> terms;
    std::vector<Real> divisors;

    for (Int i = 0; i < m.numRow() && accepted < maxCuts; ++i) {
        for (int side = 0; side < 2; ++side) {
            if (accepted >= maxCuts) break;
            Real rhs, sign;
            if (side == 0) { if (!isFinite(m.rowUpper[i])) continue; rhs = m.rowUpper[i]; sign = 1.0; }
            else           { if (isNegInf(m.rowLower[i]))  continue; rhs = m.rowLower[i]; sign = -1.0; }

            Int len = At.colPtr[i + 1] - At.colPtr[i];
            if (len < 2 || len > 5000) continue;

            // ---- bound substitution ---------------------------------------
            terms.clear();
            Real b = sign * rhs;
            bool ok = true;
            bool anyInt = false;
            for (Int t = At.colPtr[i]; t < At.colPtr[i + 1]; ++t) {
                Int j = At.rowIdx[t];
                Real a = sign * At.val[t];
                if (a == 0.0) continue;
                Real l = m.colLower[j], u = m.colUpper[j];
                bool isInt = (m.colType[j] != VarType::Continuous);
                if (isInt) {
                    // Integrality of the substituted variable needs integral bounds.
                    if (isFinite(l) && !isIntegral(l, 1e-9)) isInt = false;
                    if (isInt && isFinite(u) && !isIntegral(u, 1e-9)) isInt = false;
                }
                bool useUpper;
                Real distLo = isNegInf(l) ? kBigReal : (x[j] - l);
                Real distUp = isInf(u)    ? kBigReal : (u - x[j]);
                if (distLo <= distUp && isFinite(l) && !isNegInf(l)) useUpper = false;
                else if (isFinite(u) && !isInf(u))                   useUpper = true;
                else if (isFinite(l) && !isNegInf(l))                useUpper = false;
                else { ok = false; break; }                 // free variable: no MIR

                MirTerm mt;
                mt.col = j;
                mt.integer = isInt;
                mt.upperSub = useUpper;
                if (useUpper) { mt.a = -a; mt.bound = u; b -= a * u; }
                else          { mt.a =  a; mt.bound = l; b -= a * l; }
                if (mt.integer) anyInt = true;
                terms.push_back(mt);
            }
            if (!ok || !anyInt || terms.empty()) continue;

            // ---- candidate divisors ---------------------------------------
            divisors.clear();
            divisors.push_back(1.0);
            for (const MirTerm& t : terms)
                if (t.integer && std::fabs(t.a) > 1e-9) divisors.push_back(std::fabs(t.a));
            std::sort(divisors.begin(), divisors.end());
            divisors.erase(std::unique(divisors.begin(), divisors.end(),
                          [](Real a, Real c) { return std::fabs(a - c) <= 1e-12 * std::max(a, c); }),
                          divisors.end());
            if (divisors.size() > 8) divisors.resize(8);

            Cut bestCut;
            Real bestViol = 1e-7;
            bool haveBest = false;

            for (Real delta : divisors) {
                if (delta <= 1e-12) continue;
                Real B = b / delta;
                Real f0 = B - std::floor(B);
                if (f0 < lim.minFrac || f0 > lim.maxFrac) continue;
                Real oneMinus = 1.0 - f0;

                // Build the cut in substituted space, then map back to x.
                std::vector<std::pair<Int, Real>> raw;
                Real cutRhs = std::floor(B);
                bool bad = false;
                for (const MirTerm& t : terms) {
                    Real A = t.a / delta;
                    Real coef;
                    if (t.integer) {
                        Real fl = std::floor(A);
                        Real fj = A - fl;
                        coef = fl + std::max(0.0, fj - f0) / oneMinus;
                    } else {
                        if (A > 0) continue;              // valid relaxation: drop it
                        coef = A / oneMinus;
                    }
                    if (coef == 0.0) continue;
                    if (!std::isfinite(coef)) { bad = true; break; }
                    // Undo the bound substitution.  The two cases move the
                    // constant to the right-hand side with *opposite* signs:
                    //   x' = x - l  ->  coef*x - coef*l   =>  rhs += coef*l
                    //   x' = u - x  -> -coef*x + coef*u   =>  rhs -= coef*u
                    if (t.upperSub) { raw.emplace_back(t.col, -coef); cutRhs -= coef * t.bound; }
                    else            { raw.emplace_back(t.col,  coef); cutRhs += coef * t.bound; }
                }
                if (bad || raw.empty()) continue;

                // "<= cutRhs" flipped into the pool's ">=" orientation.
                std::sort(raw.begin(), raw.end());
                Cut c;
                c.origin = "mir";
                for (auto& pr : raw) {
                    Real v = -pr.second;
                    if (!c.idx.empty() && c.idx.back() == pr.first) { c.val.back() += v; continue; }
                    c.idx.push_back(pr.first); c.val.push_back(v);
                }
                c.rhs = -cutRhs;
                c.violation = c.rhs - c.activity(x);
                if (c.violation > bestViol) { bestViol = c.violation; bestCut = std::move(c); haveBest = true; }
            }

            if (haveBest && pool.offer(std::move(bestCut), st)) { ++accepted; ++st.mir; }
        }
    }
    return accepted;
}

// ===========================================================================
//  Model surgery
// ===========================================================================
void appendCutRows(Model& m, const std::vector<Cut>& cuts, Int from) {
    Int add = (Int)cuts.size() - from;
    if (add <= 0) return;
    const Int oldRows = m.numRow();
    const Int nc = m.numCol();

    std::vector<Int> extra(nc, 0);
    for (Int c = from; c < (Int)cuts.size(); ++c)
        for (Int j : cuts[c].idx) extra[j]++;

    SparseMatrix B;
    B.nrow = oldRows + add;
    B.ncol = nc;
    B.colPtr.assign(nc + 1, 0);
    for (Int j = 0; j < nc; ++j)
        B.colPtr[j + 1] = B.colPtr[j] + m.A.colLen(j) + extra[j];
    B.rowIdx.assign(B.colPtr.back(), 0);
    B.val.assign(B.colPtr.back(), 0.0);

    std::vector<Int> pos(nc);
    for (Int j = 0; j < nc; ++j) {
        Int q = B.colPtr[j];
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p) {
            B.rowIdx[q] = m.A.rowIdx[p];
            B.val[q]    = m.A.val[p];
            ++q;
        }
        pos[j] = q;
    }
    // New row indices exceed every existing one, so appending at the tail of
    // each column keeps the ascending row order the LU factorization relies on.
    for (Int c = from; c < (Int)cuts.size(); ++c) {
        Int r = oldRows + (c - from);
        const Cut& cu = cuts[c];
        for (size_t t = 0; t < cu.idx.size(); ++t) {
            Int j = cu.idx[t];
            B.rowIdx[pos[j]] = r;
            B.val[pos[j]]    = cu.val[t];
            ++pos[j];
        }
    }
    m.A = std::move(B);
    for (Int c = from; c < (Int)cuts.size(); ++c) {
        m.rowLower.push_back(cuts[c].rhs);
        m.rowUpper.push_back(kInf);
        m.rowName.push_back("cut_" + std::to_string(c));
    }
}

// ===========================================================================
//  Root cut loop
// ===========================================================================
Status runRootCutLoop(Model& mm, Simplex& sx, const std::vector<Int>& intCols,
                      const Options& opt, const CutLimits& lim, CutStats& st)
{
    const Int nc = mm.numCol();
    const Int baseRows = mm.numRow();
    std::vector<uint8_t> isIntCol(nc, 0);
    for (Int j : intCols) isIntCol[j] = 1;

    CutPool pool(nc, lim, &mm.colLower, &mm.colUpper);
    Status status = Status::Optimal;

    std::vector<Real> x(nc);
    for (Int j = 0; j < nc; ++j) x[j] = sx.values()[j];
    st.rootBefore = 0;
    for (Int j = 0; j < nc; ++j) st.rootBefore += mm.obj[j] * x[j];
    st.rootBefore += mm.objOffset;

    Int applied = 0;
    for (int round = 0; round < opt.cutRoundsRoot; ++round) {
        Int before = pool.size();

        Int budget = std::min<Int>(lim.maxPerRound, std::max<Int>(20, nc / 8 + 20));
        if (opt.cutGomory) separateGomory(sx, mm, isIntCol, lim, pool, st, budget);
        if (opt.cutCover)  separateCover(mm, x, lim, pool, st, budget);
        if (opt.cutMir)    separateMir(mm, x, lim, pool, st, budget);

        Int fresh = pool.size() - before;
        if (fresh <= 0) break;
        ++st.rounds;

        std::vector<VarStatus> cs, rs;
        sx.extractStatus(cs, rs);

        appendCutRows(mm, pool.cuts(), before);
        applied += fresh;

        // The new logical variables enter the basis, which keeps the basis
        // matrix nonsingular and leaves the old point dual feasible -- exactly
        // the situation the dual simplex is built to re-optimize from.
        rs.resize(mm.numRow(), VarStatus::Basic);

        sx.load(mm.A, mm.obj, mm.colLower, mm.colUpper, mm.rowLower, mm.rowUpper, opt);
        sx.setBasis(cs, rs);
        Status rst = sx.solve(true);
        if (rst != Status::Optimal) {
            // Refuse to let a cut round damage a solvable root: retry cold once,
            // and if that also fails, roll the round back.
            sx.load(mm.A, mm.obj, mm.colLower, mm.colUpper, mm.rowLower, mm.rowUpper, opt);
            rst = sx.solve(false);
            if (rst != Status::Optimal) { status = rst; break; }
        }
        for (Int j = 0; j < nc; ++j) x[j] = sx.values()[j];

        Real obj = mm.objOffset;
        for (Int j = 0; j < nc; ++j) obj += mm.obj[j] * x[j];
        opt.log.log(3, "    cut round %d: +%d cuts (%d total), root bound %.10g\n",
                    round + 1, (int)fresh, (int)pool.size(), (double)obj);
    }

    st.applied = applied;

    // ---- purge cuts that ended up slack at the root ------------------------
    // A cut that is not tight has not shaped the relaxation, and every row it
    // adds is paid for again at every node of the tree.
    if (status == Status::Optimal && pool.size() > 0) {
        const auto& cuts = pool.cuts();
        std::vector<uint8_t> keep(cuts.size(), 0);
        Int nkeep = 0;
        for (size_t c = 0; c < cuts.size(); ++c) {
            Real act = cuts[c].activity(x);
            Real slack = act - cuts[c].rhs;
            Real scale = std::max(1.0, std::fabs(cuts[c].rhs));
            if (slack <= 1e-6 * scale) { keep[c] = 1; ++nkeep; }
        }
        if (nkeep < (Int)cuts.size() && nkeep > 0) {
            st.purged = (Int)cuts.size() - nkeep;
            pool.keepOnly(keep);

            // Rebuild the model from the base rows plus the surviving cuts.
            mm.A.nrow = baseRows;
            {
                // strip cut rows out of the CSC structure
                SparseMatrix B;
                B.nrow = baseRows; B.ncol = nc;
                B.colPtr.assign(nc + 1, 0);
                for (Int j = 0; j < nc; ++j) {
                    Int cnt = 0;
                    for (Int p = mm.A.colPtr[j]; p < mm.A.colPtr[j + 1]; ++p)
                        if (mm.A.rowIdx[p] < baseRows) ++cnt;
                    B.colPtr[j + 1] = B.colPtr[j] + cnt;
                }
                B.rowIdx.assign(B.colPtr.back(), 0);
                B.val.assign(B.colPtr.back(), 0.0);
                Int q = 0;
                for (Int j = 0; j < nc; ++j)
                    for (Int p = mm.A.colPtr[j]; p < mm.A.colPtr[j + 1]; ++p)
                        if (mm.A.rowIdx[p] < baseRows) { B.rowIdx[q] = mm.A.rowIdx[p]; B.val[q] = mm.A.val[p]; ++q; }
                mm.A = std::move(B);
            }
            mm.rowLower.resize(baseRows);
            mm.rowUpper.resize(baseRows);
            mm.rowName.resize(baseRows);
            appendCutRows(mm, pool.cuts(), 0);

            sx.load(mm.A, mm.obj, mm.colLower, mm.colUpper, mm.rowLower, mm.rowUpper, opt);
            Status rst = sx.solve(false);
            if (rst != Status::Optimal) status = rst;
            else for (Int j = 0; j < nc; ++j) x[j] = sx.values()[j];
        }
    }

    st.rootAfter = mm.objOffset;
    for (Int j = 0; j < nc; ++j) st.rootAfter += mm.obj[j] * x[j];
    return status;
}

} // namespace igaos
