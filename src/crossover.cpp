#include "igaos/crossover.hpp"
#include "igaos/simplex.hpp"
#include <algorithm>

namespace igaos {

namespace {

// How far the interior point leaves a variable from its nearest finite bound,
// measured relatively so a variable ranging over [0, 1e6] and one over [0, 1]
// are ranked on the same scale.  A free variable is maximally interior: it can
// never be nonbasic at a bound it does not have.
Real interiorness(Real v, Real lo, Real up) {
    bool fl = isFinite(lo) && !isNegInf(lo);
    bool fu = isFinite(up) && !isInf(up);
    if (!fl && !fu) return kBigReal;
    Real scale = 1.0;
    if (fl && fu) scale = std::max(1.0, up - lo);
    else if (fl)  scale = std::max(1.0, std::fabs(lo));
    else          scale = std::max(1.0, std::fabs(up));
    Real d = kBigReal;
    if (fl) d = std::min(d, (v - lo) / scale);
    if (fu) d = std::min(d, (up - v) / scale);
    return std::max(d, 0.0);
}

VarStatus nearerBound(Real v, Real lo, Real up) {
    bool fl = isFinite(lo) && !isNegInf(lo);
    bool fu = isFinite(up) && !isInf(up);
    if (fl && fu) {
        if (up - lo <= 0.0) return VarStatus::Fixed;
        return (v - lo <= up - v) ? VarStatus::AtLower : VarStatus::AtUpper;
    }
    if (fl) return VarStatus::AtLower;
    if (fu) return VarStatus::AtUpper;
    return VarStatus::AtZero;
}

} // namespace

CrossoverResult crossover(const Model& mo, const Options& opt, const IpmResult& ip) {
    CrossoverResult out;
    const Int n = mo.numCol(), m = mo.numRow();
    if ((Int)ip.x.size() != n || (Int)ip.s.size() != m) {
        out.status = Status::NumericalError;
        return out;
    }

    // ---- 1. rank every extended variable by interiorness --------------------
    std::vector<std::pair<Real, Int>> rank;   // (-interiorness, extended index)
    rank.reserve(n + m);
    for (Int j = 0; j < n; ++j)
        rank.emplace_back(-interiorness(ip.x[j], mo.colLower[j], mo.colUpper[j]), j);
    for (Int i = 0; i < m; ++i)
        rank.emplace_back(-interiorness(ip.s[i], mo.rowLower[i], mo.rowUpper[i]), n + i);
    std::sort(rank.begin(), rank.end());

    // ---- 2. the m most interior variables become the candidate basis --------
    std::vector<VarStatus> colStat(n, VarStatus::AtLower);
    std::vector<VarStatus> rowStat(m, VarStatus::Basic);
    std::vector<uint8_t> isBasic(n + m, 0);
    Int taken = 0;
    for (const auto& pr : rank) {
        if (taken >= m) break;
        isBasic[pr.second] = 1;
        ++taken;
    }
    const Real interiorThreshold = 1e-7;
    for (Int j = 0; j < n; ++j) {
        if (isBasic[j]) { colStat[j] = VarStatus::Basic; continue; }
        colStat[j] = nearerBound(ip.x[j], mo.colLower[j], mo.colUpper[j]);
        if (interiorness(ip.x[j], mo.colLower[j], mo.colUpper[j]) > interiorThreshold) ++out.pushes;
    }
    for (Int i = 0; i < m; ++i) {
        if (isBasic[n + i]) { rowStat[i] = VarStatus::Basic; continue; }
        rowStat[i] = nearerBound(ip.s[i], mo.rowLower[i], mo.rowUpper[i]);
        if (interiorness(ip.s[i], mo.rowLower[i], mo.rowUpper[i]) > interiorThreshold) ++out.pushes;
    }

    // ---- 3. clean up with the simplex ---------------------------------------
    Simplex sx;
    sx.load(mo.A, mo.obj, mo.colLower, mo.colUpper, mo.rowLower, mo.rowUpper, opt);
    sx.setBasis(colStat, rowStat);
    Status st = sx.solve(true);
    if (st != Status::Optimal) {
        // The identified basis was a poor guess; the simplex still has to finish
        // the job, so give it a clean start rather than reporting failure.
        sx.setBasis(colStat, rowStat);
        st = sx.solvePrimal();
        if (st != Status::Optimal) {
            sx.setSlackBasis();
            st = sx.solve(false);
        }
    }

    out.status = st;
    out.iterations = sx.iterations();
    out.colValue.assign(n, 0.0);
    out.colDual.assign(n, 0.0);
    out.rowValue.assign(m, 0.0);
    out.rowDual.assign(m, 0.0);
    const auto& v = sx.values();
    const auto& d = sx.duals();
    const auto& y = sx.rowDuals();
    for (Int j = 0; j < n; ++j) { out.colValue[j] = v[j]; out.colDual[j] = d[j]; }
    for (Int i = 0; i < m; ++i) out.rowDual[i] = y[i];
    mo.rowActivity(out.colValue, out.rowValue);
    sx.extractStatus(out.colStatus, out.rowStatus);
    out.objective = mo.objectiveValue(out.colValue);
    out.primalInfeasibility = sx.primalInfeasibility();
    out.dualInfeasibility = sx.dualInfeasibility();

    Int stillBasic = 0;
    for (VarStatus s : out.colStatus) if (s == VarStatus::Basic) ++stillBasic;
    for (VarStatus s : out.rowStatus) if (s == VarStatus::Basic) ++stillBasic;
    out.repaired = std::max(0, taken - stillBasic);
    return out;
}

} // namespace igaos
