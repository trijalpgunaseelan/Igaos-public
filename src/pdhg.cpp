#include "igaos/pdhg.hpp"
#include <algorithm>

namespace igaos {

namespace {

inline Real clampTo(Real v, Real lo, Real up) {
    if (!isNegInf(lo) && v < lo) return lo;
    if (!isInf(up)    && v > up) return up;
    return v;
}

// Support function of the row box, sigma_C(y) = sup_{s in C} y's.
// Infinite when y pushes against a side the box does not have -- that is the
// dual infeasibility signal, reported rather than clipped.
Real supportFunction(const std::vector<Real>& y,
                     const std::vector<Real>& lo, const std::vector<Real>& up,
                     bool& finite) {
    // Scaled so the "is this component really non-zero?" test is relative to
    // the size of y rather than absolute.  Without it a component sitting at
    // rounding level counts as a genuine sign violation and drives the whole
    // support value to infinity -- which is a statement about arithmetic noise,
    // not about the iterate.
    Real scale = 0;
    for (Real v : y) scale = std::max(scale, std::fabs(v));
    const Real tol = 1e-12 * std::max(scale, 1.0);

    Real total = 0;
    finite = true;
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] > tol) {
            if (isInf(up[i])) { finite = false; return kBigReal; }
            total += y[i] * up[i];
        } else if (y[i] < -tol) {
            if (isNegInf(lo[i])) { finite = false; return kBigReal; }
            total += y[i] * lo[i];
        }
    }
    return total;
}

} // namespace

// ===========================================================================
Real spectralNormEstimate(const SparseMatrix& A, int maxIter, Real tol) {
    if (A.nnz() == 0) return 1.0;
    std::vector<Real> v(A.ncol, 0.0), w(A.nrow, 0.0), t(A.ncol, 0.0);
    uint64_t seed = 12345;
    for (Int j = 0; j < A.ncol; ++j) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        v[j] = ((Real)((seed >> 33) & 0xffff) / 32768.0) - 1.0;
    }
    Real nrm = twoNorm(v);
    if (nrm == 0) { v.assign(A.ncol, 1.0); nrm = twoNorm(v); }
    scaleVec(1.0 / nrm, v);

    Real lambda = 0, prev = -1;
    for (int it = 0; it < maxIter; ++it) {
        A.multiply(v, w);                       // w = A v
        std::fill(t.begin(), t.end(), 0.0);
        A.multiplyTransposeAdd(1.0, w, t);      // t = A'A v
        Real tn = twoNorm(t);
        if (tn <= 0) return 1.0;
        lambda = tn;
        scaleVec(1.0 / tn, t);
        v.swap(t);
        if (prev > 0 && std::fabs(lambda - prev) <= tol * lambda) break;
        prev = lambda;
    }
    return std::sqrt(std::max(lambda, 1e-30));
}

// ===========================================================================
PdhgResult primalDualHybridGradient(const Model& mo, const Options& opt) {
    PdhgResult res;
    Timer clock;
    const Int n = mo.numCol(), m = mo.numRow();
    if (n == 0) { res.status = Status::Optimal; return res; }
    if (mo.isQp()) { res.status = Status::NumericalError; return res; }

    const Real normA = std::max(spectralNormEstimate(mo.A), 1e-12);
    res.matrixNorm = normA;

    Real omega = 1.0;                            // primal weight
    Real tau = 1.0 / (omega * normA);
    Real sigma = omega / normA;

    std::vector<Real> x(n, 0.0), y(m, 0.0), xPrev(n, 0.0);
    for (Int j = 0; j < n; ++j) x[j] = clampTo(0.0, mo.colLower[j], mo.colUpper[j]);

    std::vector<Real> xRestart(x), yRestart(y);
    std::vector<Real> xSum(n, 0.0), ySum(m, 0.0);
    std::vector<Real> xAvg(n, 0.0), yAvg(m, 0.0);
    Real weightSum = 0;

    std::vector<Real> aty(n, 0.0), ax(m, 0.0), extrapolated(n, 0.0), v(m, 0.0);
    std::vector<Real> work(m, 0.0), reduced(n, 0.0);

    Real cNorm = 1.0, bNorm = 1.0;
    for (Real c : mo.obj) cNorm = std::max(cNorm, std::fabs(c));
    for (Int i = 0; i < m; ++i) {
        if (isFinite(mo.rowLower[i])) bNorm = std::max(bNorm, std::fabs(mo.rowLower[i]));
        if (isFinite(mo.rowUpper[i])) bNorm = std::max(bNorm, std::fabs(mo.rowUpper[i]));
    }

    // KKT error of a candidate point: primal residual, dual residual and the
    // relative duality gap, combined into one number so restarts have something
    // to compare.
    auto kktError = [&](const std::vector<Real>& xc, const std::vector<Real>& yc,
                        Real& pInf, Real& dInf, Real& gap, Real& pObj, Real& dObj) {
        mo.A.multiply(xc, work);
        pInf = 0;
        for (Int i = 0; i < m; ++i) {
            Real viol = 0;
            if (!isNegInf(mo.rowLower[i])) viol = std::max(viol, mo.rowLower[i] - work[i]);
            if (!isInf(mo.rowUpper[i]))    viol = std::max(viol, work[i] - mo.rowUpper[i]);
            pInf = std::max(pInf, viol);
        }
        std::fill(reduced.begin(), reduced.end(), 0.0);
        mo.A.multiplyTransposeAdd(1.0, yc, reduced);
        pObj = mo.objOffset;
        for (Int j = 0; j < n; ++j) pObj += mo.obj[j] * xc[j];

        bool finite = true;
        Real sup = supportFunction(yc, mo.rowLower, mo.rowUpper, finite);
        dObj = mo.objOffset;
        dInf = 0;
        if (!finite) { dInf = kBigReal; dObj = -kBigReal; }
        else {
            for (Int j = 0; j < n; ++j) {
                Real r = mo.obj[j] + reduced[j];
                Real lo = mo.colLower[j], up = mo.colUpper[j];
                if (r > 0) {
                    if (isNegInf(lo)) { dInf = std::max(dInf, r); }
                    else dObj += r * lo;
                } else if (r < 0) {
                    if (isInf(up)) { dInf = std::max(dInf, -r); }
                    else dObj += r * up;
                }
            }
            dObj -= sup;
        }
        gap = std::fabs(pObj - dObj) / (1.0 + std::fabs(pObj) + std::fabs(dObj));
        pInf /= (1.0 + bNorm);
        dInf /= (1.0 + cNorm);
        return std::max(gap, std::max(pInf, dInf));
    };

    // A row-major copy of A, so that BOTH matrix-vector products in the
    // iteration run on the parallel kernel.
    //
    // y = A x accumulates into y and cannot be parallelised over columns
    // without atomics; y = A' x is an independent reduction per output entry
    // and parallelises cleanly.  Storing the transpose explicitly turns the
    // first product into the second at the cost of one extra copy of the
    // nonzeros -- which is the same trade the CUDA kernels make, and for the
    // same reason.  Without it Amdahl caps this loop at 2x no matter how many
    // cores are available, because half of every iteration stays sequential.
    const SparseMatrix At = mo.A.transpose();

    Real pInf = 0, dInf = 0, gap = 0, pObj = 0, dObj = 0;
    Real lastRestartError = kBigReal;
    Long iter = 0;
    Int restarts = 0;
    const Int checkEvery = 64;
    const Int restartEvery = std::max(64, opt.pdhgRestart * 8);
    Status status = Status::IterationLimit;

    for (iter = 1; iter <= opt.pdhgMaxIter; ++iter) {
        // ---- primal step: x+ = proj( x - tau (c + A'y) ) -------------------
        std::fill(aty.begin(), aty.end(), 0.0);
        mo.A.multiplyTransposeAdd(1.0, y, aty);       // SpMV #1
        xPrev = x;
        for (Int j = 0; j < n; ++j)
            x[j] = clampTo(x[j] - tau * (mo.obj[j] + aty[j]), mo.colLower[j], mo.colUpper[j]);

        // ---- dual step on the extrapolated primal --------------------------
        for (Int j = 0; j < n; ++j) extrapolated[j] = 2.0 * x[j] - xPrev[j];
        // ax = A * extrapolated, computed as At' * extrapolated so the parallel
        // reduction kernel is used (see the transpose above).
        std::fill(ax.begin(), ax.end(), 0.0);
        At.multiplyTransposeAdd(1.0, extrapolated, ax);   // SpMV #2
        for (Int i = 0; i < m; ++i) {
            Real vi = y[i] + sigma * ax[i];
            // Moreau: prox_{sigma*sigma_C}(v) = v - sigma * proj_C(v / sigma).
            // When v/sigma lands strictly inside the row box the projection is
            // the identity and the exact result is zero -- but evaluating it as
            // vi - sigma*(vi/sigma) leaves rounding noise of order 1e-16*|vi|,
            // whose SIGN is not reproducible across architectures.  A dual
            // component of the wrong sign on a row with no bound on that side
            // makes the support function infinite, so a single rounding bit was
            // enough to make the whole dual measure blow up.  Take the exact
            // branch instead of subtracting two nearly equal numbers.
            Real t = vi / sigma;
            Real proj = clampTo(t, mo.rowLower[i], mo.rowUpper[i]);
            v[i] = (proj == t) ? 0.0 : vi - sigma * proj;
        }
        y.swap(v);

        // ---- running average ------------------------------------------------
        weightSum += 1.0;
        for (Int j = 0; j < n; ++j) xSum[j] += x[j];
        for (Int i = 0; i < m; ++i) ySum[i] += y[i];

        if ((iter % checkEvery) != 0) continue;
        if (clock.elapsed() > opt.timeLimit) { status = Status::TimeLimit; break; }

        for (Int j = 0; j < n; ++j) xAvg[j] = xSum[j] / weightSum;
        for (Int i = 0; i < m; ++i) yAvg[i] = ySum[i] / weightSum;

        Real errCur = kktError(x, y, pInf, dInf, gap, pObj, dObj);
        Real pInfA, dInfA, gapA, pObjA, dObjA;
        Real errAvg = kktError(xAvg, yAvg, pInfA, dInfA, gapA, pObjA, dObjA);

        bool useAvg = errAvg < errCur;
        Real err = useAvg ? errAvg : errCur;
        if (useAvg) { pInf = pInfA; dInf = dInfA; gap = gapA; pObj = pObjA; dObj = dObjA; }

        opt.log.log(3, "    pdhg %8lld  err %.3e  primal %.3e  dual %.3e  gap %.3e  omega %.3g\n",
                    (long long)iter, (double)err, (double)pInf, (double)dInf,
                    (double)gap, (double)omega);

        if (pInf <= opt.pdhgTol && dInf <= opt.pdhgTol && gap <= opt.pdhgTol) {
            if (useAvg) { x = xAvg; y = yAvg; }
            status = Status::Optimal;
            break;
        }

        // ---- adaptive restart ----------------------------------------------
        // Restarting to the average whenever the average is the better point --
        // and whenever progress since the last restart has stalled -- is what
        // turns PDHG from a method with a sublinear tail into one that keeps
        // making digits on structured problems.
        bool sufficient = err <= 0.2 * lastRestartError;
        bool stalled = (iter % restartEvery) == 0;
        if (sufficient || stalled) {
            if (useAvg) { x = xAvg; y = yAvg; }
            std::fill(xSum.begin(), xSum.end(), 0.0);
            std::fill(ySum.begin(), ySum.end(), 0.0);
            weightSum = 0;
            ++restarts;
            lastRestartError = err;

            // Primal weight adaptation.  The quantity that matters is how far
            // each block has travelled *since the previous restart* -- comparing
            // a single primal step against the norm of y instead would make the
            // weight drift in one direction until it pins at its cap, and a
            // pinned weight is a stalled method: tau shrinks to nothing and the
            // primal iterate stops moving.
            Real dxn = 0, dyn = 0;
            for (Int j = 0; j < n; ++j) { Real d = x[j] - xRestart[j]; dxn += d * d; }
            for (Int i = 0; i < m; ++i) { Real d = y[i] - yRestart[i]; dyn += d * d; }
            dxn = std::sqrt(dxn); dyn = std::sqrt(dyn);
            if (dxn > 1e-12 && dyn > 1e-12) {
                Real target = dyn / dxn;
                omega = std::exp(0.5 * std::log(target) + 0.5 * std::log(omega));
                omega = std::min(std::max(omega, 1e-4), 1e4);
                tau = 1.0 / (omega * normA);
                sigma = omega / normA;
            }
            xRestart = x;
            yRestart = y;
        }
    }

    if (iter > opt.pdhgMaxIter) iter = opt.pdhgMaxIter;
    res.status = status;
    res.iterations = iter;
    res.restarts = restarts;
    res.primalObjective = pObj;
    res.dualObjective = dObj;
    res.primalInfeasibility = pInf;
    res.dualInfeasibility = dInf;
    res.relativeGap = gap;
    res.primalWeight = omega;
    res.x = x;
    res.y = y;
    res.s.assign(m, 0.0);
    mo.A.multiply(x, res.s);
    for (Int i = 0; i < m; ++i) res.s[i] = clampTo(res.s[i], mo.rowLower[i], mo.rowUpper[i]);
    return res;
}

} // namespace igaos
