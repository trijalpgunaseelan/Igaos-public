#include "igaos/simplex.hpp"
#include <algorithm>
#include <numeric>

namespace igaos {

// ===========================================================================
//  Setup
// ===========================================================================
void Simplex::load(const SparseMatrix& A,
                   const std::vector<Real>& obj,
                   const std::vector<Real>& cl, const std::vector<Real>& cu,
                   const std::vector<Real>& rl, const std::vector<Real>& ru,
                   const Options& options) {
    opt = options;
    A_ = A;
    At_ = A.transpose();
    m_ = A.nrow; n_ = A.ncol; nTot_ = n_ + m_;
    E_.bind(A_);
    cost_.assign(nTot_, 0.0);
    lower_.assign(nTot_, 0.0);
    upper_.assign(nTot_, 0.0);
    for (Int j = 0; j < n_; ++j) { cost_[j] = obj[j]; lower_[j] = cl[j]; upper_[j] = cu[j]; }
    for (Int i = 0; i < m_; ++i) { lower_[n_ + i] = rl[i]; upper_[n_ + i] = ru[i]; }
    allocate();
    setSlackBasis();
}

void Simplex::allocate() {
    basis_.assign(m_, 0);
    inBasis_.assign(nTot_, kNone);
    status_.assign(nTot_, VarStatus::AtLower);
    value_.assign(nTot_, 0.0);
    xB_.assign(m_, 0.0);
    dual_.assign(nTot_, 0.0);
    y_.assign(m_, 0.0);
    dseWeight_.assign(m_, 1.0);
    devex_.assign(nTot_, 1.0);
    costShift_.assign(nTot_, 0.0);
    alpha_.resize(m_); rho_.resize(m_); tau_.resize(m_);
    row_.resize(nTot_);
    dwork_.assign(std::max(m_, nTot_), 0.0);
    scratchM_.assign(m_, 0.0);
    scratchM2_.assign(m_, 0.0);
}

Real Simplex::nonbasicValue(Int k) const {
    Real lo = lower_[k], up = upper_[k];
    switch (status_[k]) {
        case VarStatus::AtLower: return isFinite(lo) ? lo : (isFinite(up) ? up : 0.0);
        case VarStatus::AtUpper: return isFinite(up) ? up : (isFinite(lo) ? lo : 0.0);
        case VarStatus::Fixed:   return isFinite(lo) ? lo : 0.0;
        default:                 return 0.0;
    }
}

void Simplex::setNonbasicToBound(Int k) {
    Real lo = lower_[k], up = upper_[k];
    if (lo == up && isFinite(lo))      status_[k] = VarStatus::Fixed;
    else if (isNegInf(lo) && isInf(up)) status_[k] = VarStatus::AtZero;
    else if (isFinite(lo))              status_[k] = VarStatus::AtLower;
    else                                status_[k] = VarStatus::AtUpper;
    value_[k] = nonbasicValue(k);
}

// A status carried over from another model (a presolved copy, or a parent
// branch-and-bound node) can name a bound that is infinite here.  Leaving it in
// place is silently fatal: a variable marked AtUpper against an infinite upper
// bound is only ever priced for the *decreasing* direction, so an improving
// increase is never considered and the solver stops at a non-optimal point.
void Simplex::sanitizeNonbasic(Int k) {
    Real lo = lower_[k], up = upper_[k];
    if (isFinite(lo) && lo == up)        { status_[k] = VarStatus::Fixed;  value_[k] = lo;  return; }
    if (isNegInf(lo) && isInf(up))       { status_[k] = VarStatus::AtZero; value_[k] = 0.0; return; }
    VarStatus st = status_[k];
    if (st == VarStatus::Fixed || st == VarStatus::AtZero || st == VarStatus::Basic)
        st = isFinite(lo) ? VarStatus::AtLower : VarStatus::AtUpper;
    if (st == VarStatus::AtUpper && !isFinite(up)) st = VarStatus::AtLower;
    if (st == VarStatus::AtLower && !isFinite(lo)) st = VarStatus::AtUpper;
    status_[k] = st;
    value_[k] = nonbasicValue(k);
}

void Simplex::setSlackBasis() {
    std::fill(inBasis_.begin(), inBasis_.end(), kNone);
    for (Int j = 0; j < n_; ++j) setNonbasicToBound(j);
    for (Int i = 0; i < m_; ++i) {
        basis_[i] = n_ + i;
        inBasis_[n_ + i] = i;
        status_[n_ + i] = VarStatus::Basic;
    }
    std::fill(dseWeight_.begin(), dseWeight_.end(), 1.0);
    std::fill(devex_.begin(), devex_.end(), 1.0);
    factor_.clearUpdates();
    sinceRefactor_ = opt.refactorFreq + 1;
}

// A triangular crash: greedily place structural columns whose pivot row is not
// yet occupied, preferring long, well-scaled columns with free or wide bounds.
// This typically removes a large fraction of phase-1 iterations on staircase
// production-planning and scheduling models.
void Simplex::crashBasis() {
    setSlackBasis();
    hasUserBasis_ = false;
    std::vector<uint8_t> rowUsed(m_, 0);
    std::vector<Int> order(n_);
    std::iota(order.begin(), order.end(), 0);
    std::vector<Real> score(n_, 0.0);
    for (Int j = 0; j < n_; ++j) {
        Real width = upper_[j] - lower_[j];
        Real w = isInf(width) ? 1e6 : width;
        score[j] = std::log(1.0 + w) - 0.15 * A_.colLen(j);
        if (lower_[j] == upper_[j]) score[j] = -1e30;
    }
    std::sort(order.begin(), order.end(),
              [&](Int a, Int b) { return score[a] > score[b]; });

    Int placed = 0;
    for (Int j : order) {
        if (placed >= m_ / 2) break;                 // keep the basis well conditioned
        if (lower_[j] == upper_[j]) continue;
        Int pivRow = kNone; Real best = 0.0;
        Real cmax = 0.0;
        for (Int p = A_.colPtr[j]; p < A_.colPtr[j + 1]; ++p) cmax = std::max(cmax, std::fabs(A_.val[p]));
        if (cmax < 1e-6) continue;
        for (Int p = A_.colPtr[j]; p < A_.colPtr[j + 1]; ++p) {
            Int i = A_.rowIdx[p];
            if (rowUsed[i]) continue;
            Real a = std::fabs(A_.val[p]);
            if (a < 0.1 * cmax) continue;            // numerical caution
            if (a > best) { best = a; pivRow = i; }
        }
        if (pivRow == kNone) continue;
        rowUsed[pivRow] = 1;
        Int leaving = basis_[pivRow];
        inBasis_[leaving] = kNone;
        setNonbasicToBound(leaving);
        basis_[pivRow] = j;
        inBasis_[j] = pivRow;
        status_[j] = VarStatus::Basic;
        ++placed;
    }
    sinceRefactor_ = opt.refactorFreq + 1;
}

void Simplex::setBasis(const std::vector<VarStatus>& colStat,
                       const std::vector<VarStatus>& rowStat) {
    std::fill(inBasis_.begin(), inBasis_.end(), kNone);
    Int p = 0;
    for (Int j = 0; j < n_; ++j) {
        status_[j] = colStat[j];
        if (colStat[j] == VarStatus::Basic && p < m_) { basis_[p] = j; inBasis_[j] = p; ++p; }
        else sanitizeNonbasic(j);
    }
    for (Int i = 0; i < m_; ++i) {
        Int k = n_ + i;
        status_[k] = rowStat[i];
        if (rowStat[i] == VarStatus::Basic && p < m_) { basis_[p] = k; inBasis_[k] = p; ++p; }
        else sanitizeNonbasic(k);
    }
    // Pad with logicals if the supplied basis is short.
    for (Int i = 0; i < m_ && p < m_; ++i) {
        Int k = n_ + i;
        if (inBasis_[k] == kNone && status_[k] != VarStatus::Basic) {
            basis_[p] = k; inBasis_[k] = p; status_[k] = VarStatus::Basic; ++p;
        }
    }
    sinceRefactor_ = opt.refactorFreq + 1;
    hasUserBasis_ = true;
}

void Simplex::extractStatus(std::vector<VarStatus>& colStat,
                            std::vector<VarStatus>& rowStat) const {
    colStat.assign(n_, VarStatus::AtLower);
    rowStat.assign(m_, VarStatus::Basic);
    for (Int j = 0; j < n_; ++j) colStat[j] = status_[j];
    for (Int i = 0; i < m_; ++i) rowStat[i] = status_[n_ + i];
}

void Simplex::changeBound(Int k, Real lo, Real up) {
    lower_[k] = lo; upper_[k] = up;
    // Keep the existing bound choice where it is still meaningful: preserving it
    // is what keeps a branch-and-bound child dual feasible for its parent basis.
    if (status_[k] != VarStatus::Basic) sanitizeNonbasic(k);
}

void Simplex::restoreBounds(const std::vector<Real>& lo, const std::vector<Real>& up) {
    lower_ = lo; upper_ = up;
    for (Int k = 0; k < nTot_; ++k)
        if (status_[k] != VarStatus::Basic) sanitizeNonbasic(k);
}

// ===========================================================================
//  Factorization and derived quantities
// ===========================================================================
bool Simplex::refactorize() {
    lastRefactorRepaired_ = false;
    for (int attempt = 0; attempt < 6; ++attempt) {
        factor_.clearUpdates();
        bool ok = factor_.factorize(E_, basis_, opt.tol);
        if (ok) { sinceRefactor_ = 0; return true; }
        lastRefactorRepaired_ = true;
        // Repair: replace the offending basis positions with logical columns.
        bool changed = false;
        for (Int pos : factor_.singularPositions) {
            Int old = basis_[pos];
            // find an unused logical
            for (Int i = 0; i < m_; ++i) {
                Int k = n_ + i;
                if (inBasis_[k] == kNone) {
                    inBasis_[old] = kNone;
                    setNonbasicToBound(old);
                    basis_[pos] = k; inBasis_[k] = pos; status_[k] = VarStatus::Basic;
                    changed = true;
                    break;
                }
            }
        }
        if (!changed) break;
    }
    // Last resort: full slack basis.
    lastRefactorRepaired_ = true;
    setSlackBasis();
    factor_.clearUpdates();
    bool ok = factor_.factorize(E_, basis_, opt.tol);
    sinceRefactor_ = 0;
    return ok;
}

void Simplex::computeBasicValues() {
    // [A -I]x = 0  =>  B x_B = -N x_N
    std::vector<Real>& r = dwork_;
    std::fill(r.begin(), r.begin() + m_, 0.0);
    for (Int k = 0; k < nTot_; ++k) {
        if (status_[k] == VarStatus::Basic) continue;
        Real v = value_[k];
        if (v == 0.0) continue;
        E_.forEach(k, [&](Int i, Real a) { r[i] -= a * v; });
    }
    std::vector<Real> rhs(r.begin(), r.begin() + m_);
    factor_.ftranDense(rhs);
    for (Int p = 0; p < m_; ++p) {
        xB_[p] = rhs[p];
        value_[basis_[p]] = rhs[p];
    }
}

void Simplex::computeDuals() {
    std::vector<Real> cb(m_, 0.0);
    for (Int p = 0; p < m_; ++p) cb[p] = cost_[basis_[p]] + costShift_[basis_[p]];
    factor_.btranDense(cb);                    // cb becomes y, indexed by row
    y_.assign(cb.begin(), cb.begin() + m_);
    for (Int k = 0; k < nTot_; ++k) {
        if (status_[k] == VarStatus::Basic) { dual_[k] = 0.0; continue; }
        Real d = cost_[k] + costShift_[k];
        if (k < n_) { for (Int p = A_.colPtr[k]; p < A_.colPtr[k + 1]; ++p) d -= A_.val[p] * y_[A_.rowIdx[p]]; }
        else        { d -= -1.0 * y_[k - n_]; }
        dual_[k] = d;
    }
}

void Simplex::updatePrimalInfeasibility() {
    Real worst = 0.0;
    for (Int p = 0; p < m_; ++p) {
        Int k = basis_[p];
        worst = std::max(worst, lower_[k] - xB_[p]);
        worst = std::max(worst, xB_[p] - upper_[k]);
    }
    primalInf_ = std::max(worst, 0.0);
}

Real Simplex::computeDualInfeasibility() {
    Real worst = 0.0;
    for (Int k = 0; k < nTot_; ++k) {
        switch (status_[k]) {
            case VarStatus::AtLower: worst = std::max(worst, -dual_[k]); break;
            case VarStatus::AtUpper: worst = std::max(worst,  dual_[k]); break;
            case VarStatus::AtZero:  worst = std::max(worst, std::fabs(dual_[k])); break;
            default: break;
        }
    }
    dualInf_ = std::max(worst, 0.0);
    return dualInf_;
}

Real Simplex::objective() const {
    Real v = 0.0;
    for (Int j = 0; j < n_; ++j) v += cost_[j] * value_[j];
    return v;
}

void Simplex::tableauColumn(Int q, SparseVector& colOut) {
    std::vector<Real>& d = scratchM2_;
    d.assign(m_, 0.0);
    E_.forEach(q, [&](Int i, Real a) { d[i] += a; });
    factor_.ftranDense(d);
    colOut.clear();
    for (Int p = 0; p < m_; ++p) if (std::fabs(d[p]) > opt.tol.zero) colOut.set(p, d[p]);
}

void Simplex::tableauRow(Int p, SparseVector& rowOut, SparseVector& rhoOut) {
    // rho = B^{-T} e_p is typically very sparse on structured industrial models
    // ("hyper-sparsity").  Building the pricing row by walking only rho's
    // support through the row-wise copy of A costs O(nnz of the touched rows)
    // instead of O(nnz of A) -- on staircase scheduling models that is often a
    // one- to two-order-of-magnitude difference per iteration.
    std::vector<Real>& e = scratchM_;
    e.assign(m_, 0.0);
    e[p] = 1.0;
    factor_.btranDense(e);
    const Real tz = opt.tol.zero;
    rhoOut.clear();
    for (Int i = 0; i < m_; ++i) if (std::fabs(e[i]) > tz) rhoOut.set(i, e[i]);
    rowOut.clear();
    for (Int i : rhoOut.idx) {
        Real ri = rhoOut.dense[i];
        for (Int t = At_.colPtr[i]; t < At_.colPtr[i + 1]; ++t) {
            Int j = At_.rowIdx[t];
            if (status_[j] == VarStatus::Basic) continue;
            rowOut.add(j, ri * At_.val[t]);
        }
        Int k = n_ + i;
        if (status_[k] != VarStatus::Basic) rowOut.add(k, -ri);
    }
    rowOut.prune(tz);
}

// ===========================================================================
//  Primal simplex
// ===========================================================================
void Simplex::buildPhaseOneCost(std::vector<Real>& c1) const {
    c1.assign(nTot_, 0.0);
    const Real ft = opt.tol.primalFeas;
    for (Int p = 0; p < m_; ++p) {
        Int k = basis_[p];
        if (xB_[p] < lower_[k] - ft)      c1[k] = -1.0;
        else if (xB_[p] > upper_[k] + ft) c1[k] =  1.0;
    }
}

// Approximate steepest-edge pricing: score d_j^2 / (1 + ||a_j||^2).  The static
// reference norms cost nothing per iteration and recover much of the iteration
// count reduction of full Devex without needing a pricing row.
Int Simplex::pricePrimal(int, const std::vector<Real>& d) const {
    Real bestScore = 0.0;
    Int  best = kNone;
    const Real dt = opt.tol.dualFeas;
    for (Int k = 0; k < nTot_; ++k) {
        VarStatus st = status_[k];
        if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
        Real dk = d[k];
        bool attractive;
        if (st == VarStatus::AtLower)      attractive = dk < -dt;
        else if (st == VarStatus::AtUpper) attractive = dk >  dt;
        else                               attractive = std::fabs(dk) > dt;
        if (!attractive) continue;
        Real score = dk * dk / devex_[k];
        if (score > bestScore) { bestScore = score; best = k; }
    }
    return best;
}

// Harris two-pass ratio test with a bound-flip alternative.  Pass 1 finds the
// largest step admissible under *relaxed* bounds; pass 2 then takes, among all
// rows admissible at that step, the one with the largest pivot magnitude.
// Trading a tolerance-sized bound violation for a numerically better pivot is
// what keeps degenerate models from destroying the factorization.
bool Simplex::ratioTestPrimal(Int q, Int dir, const SparseVector& alpha, int,
                              Int& leavePos, Real& theta, bool& boundFlip, Real& leaveBound) {
    const Real ft  = opt.tol.primalFeas;
    const Real piv = opt.tol.pivot;
    leavePos = kNone; boundFlip = false; leaveBound = 0.0;

    Real thetaFlip = kInf;
    if (isFinite(lower_[q]) && isFinite(upper_[q])) thetaFlip = upper_[q] - lower_[q];

    Real thetaMax = thetaFlip;
    for (Int p : alpha.idx) {
        Real ap = dir * alpha.dense[p];
        if (std::fabs(ap) <= piv) continue;
        Int k = basis_[p];
        Real lo = lower_[k], up = upper_[k], x = xB_[p], bnd;
        if (ap > 0) {
            if (isFinite(up) && x > up + ft)      bnd = up;
            else if (isFinite(lo) && x < lo - ft) continue;
            else if (isFinite(lo))                bnd = lo;
            else                                  continue;
        } else {
            if (isFinite(lo) && x < lo - ft)      bnd = lo;
            else if (isFinite(up) && x > up + ft) continue;
            else if (isFinite(up))                bnd = up;
            else                                  continue;
        }
        Real r = (x - bnd + (ap > 0 ? ft : -ft)) / ap;
        if (r < 0) r = 0;
        if (r < thetaMax) thetaMax = r;
    }
    if (isInf(thetaMax)) return false;

    Real bestPivot = 0.0, bestTheta = 0.0;
    for (Int p : alpha.idx) {
        Real ap = dir * alpha.dense[p];
        Real aa = std::fabs(ap);
        if (aa <= piv) continue;
        Int k = basis_[p];
        Real lo = lower_[k], up = upper_[k], x = xB_[p], bnd;
        if (ap > 0) {
            if (isFinite(up) && x > up + ft)      bnd = up;
            else if (isFinite(lo) && x < lo - ft) continue;
            else if (isFinite(lo))                bnd = lo;
            else                                  continue;
        } else {
            if (isFinite(lo) && x < lo - ft)      bnd = lo;
            else if (isFinite(up) && x > up + ft) continue;
            else if (isFinite(up))                bnd = up;
            else                                  continue;
        }
        Real r = (x - bnd) / ap;
        if (r < 0) r = 0;
        if (r <= thetaMax && aa > bestPivot) {
            bestPivot = aa; bestTheta = r; leavePos = p; leaveBound = bnd;
        }
    }
    if (leavePos == kNone || thetaFlip < bestTheta) {
        if (isInf(thetaFlip)) return false;
        boundFlip = true; theta = thetaFlip; leavePos = kNone;
        return true;
    }
    theta = bestTheta;
    return true;
}

Status Simplex::solvePrimal() {
    timer_.reset();
    if (!refactorize()) return Status::NumericalError;
    computeBasicValues();

    for (Int k = 0; k < nTot_; ++k) {
        Real s = 1.0;
        if (k < n_) for (Int p = A_.colPtr[k]; p < A_.colPtr[k + 1]; ++p) s += A_.val[p] * A_.val[p];
        else s += 1.0;
        devex_[k] = s;
    }

    std::vector<Real> c1, d(nTot_, 0.0), yy(m_, 0.0);
    Long stallCount = 0;
    int  stuckPhase1 = 0;
    bool forcePhase2 = false;

    for (;;) {
        if (iter_ >= opt.iterationLimit) return Status::IterationLimit;
        if (timer_.elapsed() > opt.timeLimit) return Status::TimeLimit;

        if (sinceRefactor_ >= opt.refactorFreq) {
            if (!refactorize()) return Status::NumericalError;
            computeBasicValues();
        }
        updatePrimalInfeasibility();
        phase_ = (!forcePhase2 && primalInf_ > opt.tol.primalFeas) ? 1 : 2;

        const std::vector<Real>* cptr;
        if (phase_ == 1) { buildPhaseOneCost(c1); cptr = &c1; }
        else             { cptr = &cost_; }
        for (Int p = 0; p < m_; ++p)
            yy[p] = (*cptr)[basis_[p]] + (phase_ == 2 ? costShift_[basis_[p]] : 0.0);
        factor_.btranDense(yy);
        for (Int k = 0; k < nTot_; ++k) {
            if (status_[k] == VarStatus::Basic) { d[k] = 0.0; continue; }
            Real v = (*cptr)[k] + (phase_ == 2 ? costShift_[k] : 0.0);
            if (k < n_) for (Int t = A_.colPtr[k]; t < A_.colPtr[k + 1]; ++t) v -= A_.val[t] * yy[A_.rowIdx[t]];
            else v += yy[k - n_];
            d[k] = v;
        }

        Int q = pricePrimal(phase_, d);
        if (q == kNone) {
            if (phase_ == 1) {
                // Phase one has priced out: no direction reduces the residual
                // infeasibility, so this IS the minimum -- and a minimum above
                // the feasibility tolerance is a proof that the model has no
                // feasible point.
                //
                // Refactorize once before believing the number: an eta file
                // that has drifted can report an infeasibility the model does
                // not have.  Re-entering the loop recomputes the basic values
                // and the phase from scratch; a residual that survives that is
                // a property of the model, not of the factorization.
                //
                // Comparing against a MULTIPLE of the tolerance here (it was
                // 1e3) means every model infeasible by less than that multiple
                // is reported optimal, with the returned point quietly
                // violating a constraint.  The tolerance is the tolerance.
                if (++stuckPhase1 == 1) {
                    if (!refactorize()) return Status::NumericalError;
                    computeBasicValues();
                    continue;
                }
                if (primalInf_ > opt.tol.primalFeas) return Status::Infeasible;
                forcePhase2 = true;
                continue;
            }
            y_.assign(yy.begin(), yy.begin() + m_);
            for (Int k = 0; k < nTot_; ++k) dual_[k] = d[k];
            computeDualInfeasibility();
            return Status::Optimal;
        }

        Int dir;
        if (status_[q] == VarStatus::AtLower)      dir = +1;
        else if (status_[q] == VarStatus::AtUpper) dir = -1;
        else                                       dir = (d[q] < 0) ? +1 : -1;

        tableauColumn(q, alpha_);
        if (alpha_.nnz() == 0) {
            if (isFinite(lower_[q]) && isFinite(upper_[q])) {
                status_[q] = (dir > 0) ? VarStatus::AtUpper : VarStatus::AtLower;
                value_[q] = nonbasicValue(q);
                ++iter_;
                continue;
            }
            return Status::Unbounded;
        }

        Int leavePos; Real theta, leaveBound; bool flip;
        if (!ratioTestPrimal(q, dir, alpha_, phase_, leavePos, theta, flip, leaveBound)) {
            if (phase_ == 1 && ++stuckPhase1 <= 2) {
                if (!refactorize()) return Status::NumericalError;
                computeBasicValues();
                continue;
            }
            return Status::Unbounded;
        }

        stuckPhase1 = 0;
        Real step = dir * theta;
        if (step != 0.0) for (Int p : alpha_.idx) xB_[p] -= step * alpha_.dense[p];
        value_[q] += step;

        if (flip) {
            status_[q] = (dir > 0) ? VarStatus::AtUpper : VarStatus::AtLower;
            value_[q] = nonbasicValue(q);
            for (Int p = 0; p < m_; ++p) value_[basis_[p]] = xB_[p];
            ++iter_;
            continue;
        }

        Int leaving = basis_[leavePos];
        inBasis_[leaving] = kNone;
        if (lower_[leaving] == upper_[leaving]) status_[leaving] = VarStatus::Fixed;
        else if (std::fabs(leaveBound - lower_[leaving]) <= std::fabs(leaveBound - upper_[leaving]))
            status_[leaving] = VarStatus::AtLower;
        else status_[leaving] = VarStatus::AtUpper;
        value_[leaving] = nonbasicValue(leaving);

        basis_[leavePos] = q;
        inBasis_[q] = leavePos;
        status_[q] = VarStatus::Basic;
        xB_[leavePos] = value_[q];
        for (Int p = 0; p < m_; ++p) value_[basis_[p]] = xB_[p];

        if (!factor_.update(leavePos, alpha_, opt.tol.pivot)) {
            if (!refactorize()) return Status::NumericalError;
            computeBasicValues();
        } else ++sinceRefactor_;
        ++iter_;

        // Anti-stalling: perturb nonbasic bounds after a long degenerate run.
        if (theta <= opt.tol.degenerate) {
            if (++stallCount > 500) {
                stallCount = 0;
                for (Int k = 0; k < nTot_; ++k) {
                    if (status_[k] == VarStatus::Basic || lower_[k] == upper_[k]) continue;
                    Real bump = 1e-9 * (1.0 + std::fabs(value_[k]))
                              * (1.0 + 0.5 * (Real)((k * 2654435761u) % 1000) / 1000.0);
                    if (status_[k] == VarStatus::AtLower && isFinite(lower_[k])) { lower_[k] -= bump; value_[k] = lower_[k]; }
                    else if (status_[k] == VarStatus::AtUpper && isFinite(upper_[k])) { upper_[k] += bump; value_[k] = upper_[k]; }
                }
                if (!refactorize()) return Status::NumericalError;
                computeBasicValues();
            }
        } else stallCount = 0;
    }
}

// ===========================================================================
//  Dual simplex
// ===========================================================================
void Simplex::makeDualFeasible() {
    computeDuals();
    const Real dt = opt.tol.dualFeas;
    bool shifted = false;
    for (Int k = 0; k < nTot_; ++k) {
        if (status_[k] == VarStatus::Basic) continue;
        if (lower_[k] == upper_[k]) { status_[k] = VarStatus::Fixed; value_[k] = lower_[k]; continue; }
        Real d = dual_[k];
        if (d < -dt) {
            if (isFinite(upper_[k])) { status_[k] = VarStatus::AtUpper; value_[k] = upper_[k]; }
            else { costShift_[k] = -d; shifted = true; }
        } else if (d > dt) {
            if (isFinite(lower_[k])) { status_[k] = VarStatus::AtLower; value_[k] = lower_[k]; }
            else { costShift_[k] = -d; shifted = true; }
        } else if (isNegInf(lower_[k]) && isInf(upper_[k])) {
            costShift_[k] = -d; shifted = true;
            status_[k] = VarStatus::AtZero; value_[k] = 0.0;
        }
    }
    if (shifted) computeDuals();
    computeBasicValues();
}

void Simplex::removeCostShifts() { std::fill(costShift_.begin(), costShift_.end(), 0.0); }

Int Simplex::priceDual(Real& delta) const {
    Real best = 0.0; Int bp = kNone;
    const Real ft = opt.tol.primalFeas;
    for (Int p = 0; p < m_; ++p) {
        Int k = basis_[p];
        Real dl;
        if (xB_[p] < lower_[k] - ft)      dl = xB_[p] - lower_[k];
        else if (xB_[p] > upper_[k] + ft) dl = xB_[p] - upper_[k];
        else continue;
        Real score = dl * dl / std::max(dseWeight_[p], 1e-8);
        if (score > best) { best = score; bp = p; delta = dl; }
    }
    return bp;
}

// Bound-flipping dual ratio test.  Candidates are walked in increasing ratio;
// any whose whole range can be absorbed by the outstanding primal infeasibility
// is flipped rather than pivoted on, so one dual iteration makes the progress
// of many.  Ties at the blocking ratio break on pivot magnitude.
bool Simplex::ratioTestDual(const SparseVector& row, Real delta, Int,
                            Int& enterIdx, Real& alphaPq, std::vector<Int>& flips) {
    flips.clear();
    enterIdx = kNone;
    const Real dt  = opt.tol.dualFeas;
    const Real piv = opt.tol.pivot;
    Real sigma = (delta > 0) ? 1.0 : -1.0;

    struct Cand { Int j; Real abar, ratio, range; };
    static thread_local std::vector<Cand> cand;
    cand.clear();

    for (Int j : row.idx) {
        VarStatus st = status_[j];
        if (st == VarStatus::Basic || st == VarStatus::Fixed) continue;
        Real abar = sigma * row.dense[j];
        if (std::fabs(abar) <= piv) continue;
        bool ok;
        if (st == VarStatus::AtLower)      ok = abar >  0;
        else if (st == VarStatus::AtUpper) ok = abar <  0;
        else                               ok = true;
        if (!ok) continue;
        Real r = dual_[j] / abar;
        if (r < 0) r = 0;
        Real range = (st == VarStatus::AtZero) ? kInf
                   : ((isFinite(lower_[j]) && isFinite(upper_[j])) ? (upper_[j] - lower_[j]) : kInf);
        cand.push_back({j, abar, r, range});
    }
    if (cand.empty()) return false;                      // dual unbounded

    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b) { return a.ratio < b.ratio; });

    Real remaining = std::fabs(delta);
    size_t b = 0;
    for (; b < cand.size(); ++b) {
        if (isInf(cand[b].range)) break;
        Real absorb = std::fabs(cand[b].abar) * cand[b].range;
        if (absorb >= remaining) break;
        remaining -= absorb;
        flips.push_back(cand[b].j);
    }
    if (b >= cand.size()) {
        b = cand.size() - 1;
        if (!flips.empty() && flips.back() == cand[b].j) flips.pop_back();
    }

    Real rb = cand[b].ratio;
    Real relax = dt / std::max(std::fabs(cand[b].abar), 1e-4);
    size_t bestIdx = b; Real bestAbs = std::fabs(cand[b].abar);
    for (size_t t = b; t < cand.size() && cand[t].ratio <= rb + relax; ++t)
        if (std::fabs(cand[t].abar) > bestAbs) { bestAbs = std::fabs(cand[t].abar); bestIdx = t; }
    for (size_t t = b; t < bestIdx; ++t)
        if (isFinite(cand[t].range)) flips.push_back(cand[t].j);

    enterIdx = cand[bestIdx].j;
    alphaPq  = row.dense[enterIdx];
    return true;
}

// Exact dual steepest edge (Forrest-Goldfarb): weights approximate
// ||B^{-1} e_i||^2 and cost one extra FTRAN per iteration.
void Simplex::updateDse(Int p, const SparseVector& alpha, Real alphaPq) {
    std::vector<Real> t(m_, 0.0);
    for (Int i : rho_.idx) t[i] = rho_.dense[i];
    factor_.ftranDense(t);
    Real wp = dseWeight_[p];
    Real inv = 1.0 / alphaPq;
    for (Int i : alpha.idx) {
        if (i == p) continue;
        Real r = alpha.dense[i] * inv;
        dseWeight_[i] = std::max(dseWeight_[i] - 2.0 * r * t[i] + r * r * wp, 1e-4);
    }
    dseWeight_[p] = std::max(wp * inv * inv, 1e-4);
}

Status Simplex::solveDual() {
    timer_.reset();
    if (!refactorize()) return Status::NumericalError;
    computeBasicValues();
    makeDualFeasible();
    std::fill(dseWeight_.begin(), dseWeight_.end(), 1.0);

    std::vector<Int> flips;
    std::vector<Real> agg(m_, 0.0), dz(m_, 0.0);
    Long stall = 0;
    bool needDuals = true;
    int  pivotMismatch = 0;
    // Dual simplex reduces primal infeasibility weakly monotonically.  If it
    // instead grows, the iteration has lost its invariant and no amount of
    // further pivoting recovers it -- bail out to the primal method rather than
    // spinning to the time limit.
    Real bestInf = kBigReal;
    Long sinceImprovement = 0;

    for (;;) {
        if (iter_ >= opt.iterationLimit) return Status::IterationLimit;
        if (timer_.elapsed() > opt.timeLimit) return Status::TimeLimit;

        if (sinceRefactor_ >= opt.refactorFreq) {
            if (!refactorize()) return Status::NumericalError;
            computeBasicValues();
            // Repairing a singular basis swaps columns out and resets their
            // status, which silently destroys the dual feasibility every later
            // ratio test assumes.  Re-establish it before continuing.
            if (lastRefactorRepaired_) { makeDualFeasible(); std::fill(dseWeight_.begin(), dseWeight_.end(), 1.0); }
            needDuals = true;
        }
        // Reduced costs are carried forward through the pricing row (see the
        // incremental update after each pivot) and recomputed from the
        // factorization only periodically, to bound accumulated drift.
        if (needDuals || (iter_ % 50) == 0) { computeDuals(); needDuals = false; }

        updatePrimalInfeasibility();
        if (primalInf_ < bestInf - 1e-12) { bestInf = primalInf_; sinceImprovement = 0; }
        else if (++sinceImprovement > 2000 ||
                 (bestInf < kBigReal && primalInf_ > 1e3 * (bestInf + 1.0))) {
            removeCostShifts();
            return solvePrimal();
        }

        Real delta = 0.0;
        Int p = priceDual(delta);
        if (p == kNone) {
            computeDuals();
            updatePrimalInfeasibility();
            computeDualInfeasibility();
            removeCostShifts();
            computeDuals();
            computeDualInfeasibility();
            if (dualInf_ > opt.tol.dualFeas) return solvePrimal();
            return Status::Optimal;
        }

        tableauRow(p, row_, rho_);
        Int q; Real alphaPqRow;
        if (!ratioTestDual(row_, delta, p, q, alphaPqRow, flips))
            return Status::Infeasible;                  // dual unbounded

        if (!flips.empty()) {
            std::fill(agg.begin(), agg.end(), 0.0);
            for (Int j : flips) {
                Real t;
                if (status_[j] == VarStatus::AtLower) { t = upper_[j] - lower_[j]; status_[j] = VarStatus::AtUpper; }
                else                                  { t = lower_[j] - upper_[j]; status_[j] = VarStatus::AtLower; }
                value_[j] = nonbasicValue(j);
                if (t == 0.0) continue;
                E_.forEach(j, [&](Int i, Real a) { agg[i] += a * t; });
            }
            dz = agg;
            factor_.ftranDense(dz);
            for (Int pp = 0; pp < m_; ++pp) xB_[pp] -= dz[pp];
            for (Int pp = 0; pp < m_; ++pp) value_[basis_[pp]] = xB_[pp];
            Int kb = basis_[p];
            if (xB_[p] < lower_[kb] - opt.tol.primalFeas)      delta = xB_[p] - lower_[kb];
            else if (xB_[p] > upper_[kb] + opt.tol.primalFeas) delta = xB_[p] - upper_[kb];
            else { ++iter_; continue; }
        }

        tableauColumn(q, alpha_);
        Real alphaPq = alpha_.dense[p];
        // The pivot element is available twice -- from the pricing row and from
        // the FTRAN of the entering column.  Disagreement means the factorization
        // has drifted, so refactorize and retry.  If it persists the two are
        // genuinely different, not stale: trust the FTRAN value, which is the
        // more accurate of the two, and only give up if it is too small to pivot
        // on.  Without the counter this retry has no state change and loops.
        if (std::fabs(alphaPq - alphaPqRow) > 1e-6 * (1.0 + std::fabs(alphaPq)) ||
            std::fabs(alphaPq) < opt.tol.pivot) {
            if (++pivotMismatch <= 2) {
                if (!refactorize()) return Status::NumericalError;
                computeBasicValues();
                needDuals = true;
                continue;
            }
            if (std::fabs(alphaPq) < opt.tol.pivot) {
                removeCostShifts();
                return solvePrimal();
            }
        }
        pivotMismatch = 0;

        Real tstep = delta / alphaPq;
        Real dq = dual_[q];
        Int leaving = basis_[p];

        updateDse(p, alpha_, alphaPq);

        for (Int i : alpha_.idx) xB_[i] -= tstep * alpha_.dense[i];
        value_[q] += tstep;

        inBasis_[leaving] = kNone;
        status_[leaving] = (delta < 0) ? VarStatus::AtLower : VarStatus::AtUpper;
        if (lower_[leaving] == upper_[leaving]) status_[leaving] = VarStatus::Fixed;
        value_[leaving] = nonbasicValue(leaving);

        basis_[p] = q; inBasis_[q] = p; status_[q] = VarStatus::Basic;
        xB_[p] = value_[q];
        for (Int pp = 0; pp < m_; ++pp) value_[basis_[pp]] = xB_[pp];

        // ---- incremental dual update: d_j -= theta_D * alpha_pj -------------
        Real thetaD = dq / alphaPq;
        if (thetaD != 0.0) {
            for (Int j : row_.idx) dual_[j] -= thetaD * row_.dense[j];
            for (Int i : rho_.idx) y_[i] += thetaD * rho_.dense[i];
        }
        dual_[leaving] = -thetaD;
        dual_[q] = 0.0;

        if (!factor_.update(p, alpha_, opt.tol.pivot)) {
            if (!refactorize()) return Status::NumericalError;
            computeBasicValues();
            needDuals = true;
        } else ++sinceRefactor_;
        ++iter_;

        if (std::fabs(tstep) <= opt.tol.degenerate) {
            if (++stall > 800) { removeCostShifts(); return solvePrimal(); }
        } else stall = 0;
    }
}

// ===========================================================================
Status Simplex::solve(bool preferDual) {
    Status st;
    if (preferDual) {
        st = solveDual();
        if (st == Status::NumericalError) {
            setSlackBasis(); removeCostShifts();
            st = solvePrimal();
        }
    } else {
        if (opt.crash && !hasUserBasis_) crashBasis();
        st = solvePrimal();
        if (st == Status::NumericalError) { setSlackBasis(); st = solvePrimal(); }
    }
    updatePrimalInfeasibility();
    computeDuals();
    computeDualInfeasibility();
    return st;
}

} // namespace igaos
