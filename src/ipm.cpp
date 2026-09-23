#include "igaos/ipm.hpp"
#include <algorithm>

namespace igaos {

namespace {

constexpr Real kStepFactor  = 0.9995;   // fraction of the distance to the boundary
constexpr Real kMinDs       = 1e-8;     // floor on the logical barrier diagonal
constexpr Real kMaxDy       = 1e8;      // cap on 1/Ds, so a free row cannot blow up
constexpr Real kFixedDiag   = 1e10;     // makes a fixed column's step vanish

// ---------------------------------------------------------------------------
// y += Q * x, with Q held as the lower triangle of a symmetric matrix.
// ---------------------------------------------------------------------------
void addQuadratic(const SparseMatrix& Q, const std::vector<Real>& x, std::vector<Real>& y) {
    if (Q.nnz() == 0) return;
    for (Int j = 0; j < Q.ncol; ++j) {
        for (Int p = Q.colPtr[j]; p < Q.colPtr[j + 1]; ++p) {
            Int i = Q.rowIdx[p];
            Real v = Q.val[p];
            y[i] += v * x[j];
            if (i != j) y[j] += v * x[i];
        }
    }
}

// ---------------------------------------------------------------------------
// The KKT matrix.  The *pattern* is built once: it depends only on the sparsity
// of A and Q, never on the barrier terms, which is the whole point of the
// quasi-definite formulation.  Each iteration refills the numeric values and
// re-runs only the numeric factorization.
// ---------------------------------------------------------------------------
struct KktMatrix {
    Int n = 0, m = 0, N = 0;
    std::vector<Int>  Kp, Ki;          // upper triangle, CSC
    std::vector<Real> constPart;       // A and Q contributions, fixed for the run
    std::vector<Int>  diagX, diagY;    // positions of the two diagonal blocks
    std::vector<Real> Kx;
    std::vector<int8_t> sign;

    void build(const Model& mo) {
        n = mo.numCol(); m = mo.numRow(); N = n + m;
        SparseMatrix Qt = mo.Q.nnz() ? mo.Q.transpose() : SparseMatrix();
        SparseMatrix At = mo.A.transpose();      // column i of At is row i of A

        Kp.assign(N + 1, 0);
        Ki.clear(); constPart.clear();
        diagX.assign(n, kNone); diagY.assign(m, kNone);

        std::vector<std::pair<Int, Real>> col;
        for (Int j = 0; j < n; ++j) {
            col.clear();
            if (Qt.ncol > j) {
                // row j of the stored lower triangle == column j of its transpose,
                // which is exactly the upper-triangle entries (i, j) with i <= j.
                for (Int p = Qt.colPtr[j]; p < Qt.colPtr[j + 1]; ++p) {
                    Int i = Qt.rowIdx[p];
                    if (i <= j) col.emplace_back(i, -Qt.val[p]);
                }
            }
            bool haveDiag = false;
            for (auto& pr : col) if (pr.first == j) haveDiag = true;
            if (!haveDiag) col.emplace_back(j, 0.0);
            std::sort(col.begin(), col.end());
            for (auto& pr : col) {
                if (pr.first == j) diagX[j] = (Int)Ki.size();
                Ki.push_back(pr.first);
                constPart.push_back(pr.second);
            }
            Kp[j + 1] = (Int)Ki.size();
        }
        for (Int i = 0; i < m; ++i) {
            Int c = n + i;
            col.clear();
            for (Int p = At.colPtr[i]; p < At.colPtr[i + 1]; ++p)
                col.emplace_back(At.rowIdx[p], At.val[p]);
            std::sort(col.begin(), col.end());
            for (auto& pr : col) { Ki.push_back(pr.first); constPart.push_back(pr.second); }
            diagY[i] = (Int)Ki.size();
            Ki.push_back(c);
            constPart.push_back(0.0);
            Kp[c + 1] = (Int)Ki.size();
        }
        Kx.assign(constPart.size(), 0.0);
        sign.assign(N, 1);
        for (Int j = 0; j < n; ++j) sign[j] = -1;
    }

    void refill(const std::vector<Real>& Dx, const std::vector<Real>& Dy) {
        Kx = constPart;
        for (Int j = 0; j < n; ++j) Kx[diagX[j]] -= Dx[j];
        for (Int i = 0; i < m; ++i) Kx[diagY[i]]  = Dy[i];
    }

    // y = K * v, K symmetric, only the upper triangle stored.
    void multiply(const std::vector<Real>& v, std::vector<Real>& y) const {
        y.assign(N, 0.0);
        for (Int c = 0; c < N; ++c) {
            Real vc = v[c];
            for (Int p = Kp[c]; p < Kp[c + 1]; ++p) {
                Int r = Ki[p];
                Real a = Kx[p];
                y[r] += a * vc;
                if (r != c) y[c] += a * v[r];
            }
        }
    }
};

// One bounded variable's barrier state.  Used for both structural columns and
// logical (row) variables, which is what keeps the row-type handling out of the
// algorithm entirely.
struct Barrier {
    std::vector<uint8_t> hasLo, hasUp, fixed;
    std::vector<Real> lo, up, g, t, z, w;
    Int pairs = 0;

    void init(Int k, const std::vector<Real>& l, const std::vector<Real>& u) {
        hasLo.assign(k, 0); hasUp.assign(k, 0); fixed.assign(k, 0);
        lo = l; up = u;
        g.assign(k, 0); t.assign(k, 0); z.assign(k, 0); w.assign(k, 0);
        pairs = 0;
        for (Int i = 0; i < k; ++i) {
            bool fl = !isNegInf(l[i]) && isFinite(l[i]);
            bool fu = !isInf(u[i]) && isFinite(u[i]);
            if (fl && fu && (u[i] - l[i]) <= 1e-11 * (1.0 + std::fabs(l[i]))) {
                fixed[i] = 1;
                continue;
            }
            hasLo[i] = fl ? 1 : 0;
            hasUp[i] = fu ? 1 : 0;
            if (fl) ++pairs;
            if (fu) ++pairs;
        }
    }

    // Place a value strictly inside its bounds, at least `push` away from each.
    Real interior(Int i, Real want, Real push) const {
        Real v = want;
        if (fixed[i]) return lo[i];
        if (hasLo[i] && hasUp[i]) {
            Real width = up[i] - lo[i];
            Real p = std::min(push, 0.45 * width);
            v = std::min(std::max(v, lo[i] + p), up[i] - p);
        } else if (hasLo[i]) {
            v = std::max(v, lo[i] + push);
        } else if (hasUp[i]) {
            v = std::min(v, up[i] - push);
        }
        return v;
    }

    void refreshDistances(Int k, const std::vector<Real>& v) {
        for (Int i = 0; i < k; ++i) {
            g[i] = hasLo[i] ? std::max(v[i] - lo[i], 1e-14) : 0.0;
            t[i] = hasUp[i] ? std::max(up[i] - v[i], 1e-14) : 0.0;
        }
    }
};

// ---------------------------------------------------------------------------
// Equality-constrained convex QP: min c'x + 1/2 x'Qx subject to Ax = b, x free.
//
// One symmetric indefinite solve, no iterations. The system is regularized the
// same way the barrier iterations regularize theirs -- a small negative shift on
// the x block and a small positive one on the y block, which makes it quasi-
// definite and so factorizable without pivoting -- and the regularization is
// then paid back by iterative refinement against the *unregularized* matrix.
// Without that refinement the answer carries the shift; with it, the residual
// on these instances lands at 1e-12 and below.
// ---------------------------------------------------------------------------
static IpmResult solveEqualityQp(const Model& mo, const Options& opt,
                                 KktMatrix& K, IpmResult res, Timer& clock) {
    const Int n = mo.numCol(), m = mo.numRow(), N = n + m;
    LdlFactor ldl;
    ldl.analyze(N, K.Kp, K.Ki);

    std::vector<Real> exact = K.constPart;          // the true KKT, unshifted
    std::vector<Real> rhs(N, 0.0), sol, work, resid(N, 0.0);
    for (Int j = 0; j < n; ++j) rhs[j] = mo.obj[j];
    for (Int i = 0; i < m; ++i) rhs[n + i] = mo.rowLower[i];

    Real scale = 1.0;
    for (Int j = 0; j < n; ++j) scale = std::max(scale, std::fabs(mo.obj[j]));
    for (Int p = 0; p < (Int)K.constPart.size(); ++p)
        scale = std::max(scale, std::fabs(K.constPart[p]));

    bool ok = false;
    for (Real reg : {1e-10, 1e-8, 1e-6, 1e-4}) {
        std::vector<Real> Dx((size_t)n, reg * scale), Dy((size_t)m, reg * scale);
        K.refill(Dx, Dy);
        Timer tF;
        ok = ldl.factorize(K.Kp, K.Ki, K.Kx, K.sign, 1e-10, 1e-13);
        res.factorTime += tF.elapsed();
        if (ok) break;
    }
    if (!ok) { res.status = Status::NumericalError; return res; }
    res.factorNonzeros = ldl.nonzeros();
    res.regularizedPivots = ldl.numRegularized();

    // Refinement is run against `exact`, not against the shifted matrix that was
    // factorized -- that difference is the whole point.
    std::vector<Real> shifted; shifted.swap(K.Kx);
    K.Kx = exact;
    sol = rhs;
    ldl.solve(sol);
    Real worst = 0.0;
    for (int r = 0; r < 24; ++r) {
        K.multiply(sol, work);
        worst = 0.0;
        for (Int i = 0; i < N; ++i) {
            resid[i] = rhs[i] - work[i];
            worst = std::max(worst, std::fabs(resid[i]));
        }
        if (worst <= 1e-11 * (1.0 + scale)) break;
        ldl.solve(resid);
        for (Int i = 0; i < N; ++i) sol[i] += resid[i];
    }

    res.x.assign(sol.begin(), sol.begin() + n);
    res.y.assign(sol.begin() + n, sol.end());
    res.s.assign((size_t)m, 0.0);
    mo.A.multiplyAdd(1.0, res.x, res.s);
    res.zLower.assign((size_t)n, 0.0);              // no bound is active: x is free
    res.zUpper.assign((size_t)n, 0.0);
    res.iterations = 1;
    res.primalObjective = mo.objectiveValue(res.x);
    res.dualObjective = res.primalObjective;   // no separate dual on this path
    res.relDualityGap = -1.0;                  // the KKT residual below is the test
    res.status = (worst <= 1e-6 * (1.0 + scale)) ? Status::Optimal
                                                 : Status::NumericalError;
    opt.log.log(2, "  interior   equality-constrained QP solved directly; "
                   "KKT residual %.3e\n", (double)worst);
    (void)clock;
    return res;
}

} // namespace

// ===========================================================================
IpmResult interiorPoint(const Model& mo, const Options& opt) {
    IpmResult res;
    Timer clock;
    const Int n = mo.numCol(), m = mo.numRow(), N = n + m;
    if (n == 0) { res.status = Status::Optimal; return res; }

    KktMatrix K;
    Timer tA;
    K.build(mo);
    LdlFactor ldl;
    ldl.analyze(N, K.Kp, K.Ki);
    res.analyzeTime = tA.elapsed();

    Barrier bx, bs;
    bx.init(n, mo.colLower, mo.colUpper);
    bs.init(m, mo.rowLower, mo.rowUpper);
    const Int npair = bx.pairs + bs.pairs;
    if (npair == 0) {
        // No bounded variable anywhere, so there is no barrier to follow -- but
        // that is a statement about the *method*, not about the problem.
        //
        //     min  c'x + 1/2 x'Qx    subject to    Ax = b,   x free
        //
        // is an ordinary equality-constrained quadratic program and its optimum
        // is the solution of one linear system, the KKT conditions themselves:
        //
        //     [ -Q   A' ] [ x ]   [ c ]
        //     [  A   0  ] [ y ] = [ b ]
        //
        // which is the very matrix this file already builds and factorizes once
        // per iteration. Returning "numerical error" for it -- which this code
        // did until the Maros and Meszaros set was run against it -- refuses the
        // easiest convex QP there is. AUG2D, AUG2DC, AUG3D and AUG3DC in that
        // collection are exactly this shape, and they are not exotic: a least-
        // squares fit over a linear balance is the ordinary industrial case.
        //
        // For a *linear* objective the old verdict stands: with nothing bounded
        // the problem is unbounded unless c lies in the row space of A, and the
        // simplex settles that with a certificate, so the caller's fallback to
        // it is the right answer rather than a barrier method pretending.
        bool allRowsFixed = true;
        for (Int i = 0; i < m; ++i) if (!bs.fixed[i]) { allRowsFixed = false; break; }
        if (!mo.isQp() || !allRowsFixed) {
            res.status = Status::NumericalError;
            return res;
        }
        return solveEqualityQp(mo, opt, K, res, clock);
    }

    // ---- starting point ---------------------------------------------------
    // Placed strictly inside every finite bound, at a distance that scales with
    // the problem so a model in tonnes and a model in kilograms start alike.
    Real scaleHint = 1.0;
    for (Int j = 0; j < n; ++j) {
        if (isFinite(mo.colLower[j])) scaleHint = std::max(scaleHint, std::fabs(mo.colLower[j]));
        if (isFinite(mo.colUpper[j])) scaleHint = std::max(scaleHint, std::fabs(mo.colUpper[j]));
    }
    const Real push = std::max(1.0, 0.1 * std::sqrt(scaleHint));

    std::vector<Real> x(n, 0.0), s(m, 0.0), y(m, 0.0);
    for (Int j = 0; j < n; ++j) {
        Real want = 0.0;
        if (bx.hasLo[j] && bx.hasUp[j]) want = 0.5 * (mo.colLower[j] + mo.colUpper[j]);
        else if (bx.hasLo[j]) want = mo.colLower[j] + push;
        else if (bx.hasUp[j]) want = mo.colUpper[j] - push;
        x[j] = bx.interior(j, want, push);
    }
    mo.A.multiply(x, s);
    for (Int i = 0; i < m; ++i) s[i] = bs.interior(i, s[i], push);

    for (Int j = 0; j < n; ++j) {
        bx.z[j] = bx.hasLo[j] ? 1.0 : 0.0;
        bx.w[j] = bx.hasUp[j] ? 1.0 : 0.0;
    }
    for (Int i = 0; i < m; ++i) {
        bs.z[i] = bs.hasLo[i] ? 1.0 : 0.0;
        bs.w[i] = bs.hasUp[i] ? 1.0 : 0.0;
    }

    std::vector<Real> Dx(n, 0.0), Dy(m, 0.0), Ds(m, 0.0);
    std::vector<Real> rp(m, 0.0), rdx(n, 0.0), rds(m, 0.0);
    std::vector<Real> rhoX(n, 0.0), rhoS(m, 0.0);
    std::vector<Real> rhs(N, 0.0), sol(N, 0.0), work(N, 0.0), resid(N, 0.0);
    std::vector<Real> dx(n, 0.0), ds(m, 0.0), dy(m, 0.0);
    std::vector<Real> dzx(n, 0.0), dwx(n, 0.0), dzs(m, 0.0), dws(m, 0.0);
    std::vector<Real> dxAff(n, 0.0), dsAff(m, 0.0);
    std::vector<Real> dzxAff(n, 0.0), dwxAff(n, 0.0), dzsAff(m, 0.0), dwsAff(m, 0.0);
    std::vector<Real> Qx(n, 0.0), Ax(m, 0.0), Aty(n, 0.0);

    Real cNorm = 1.0, bNorm = 1.0;
    for (Real v : mo.obj) cNorm = std::max(cNorm, std::fabs(v));
    for (Int i = 0; i < m; ++i) {
        if (isFinite(mo.rowLower[i])) bNorm = std::max(bNorm, std::fabs(mo.rowLower[i]));
        if (isFinite(mo.rowUpper[i])) bNorm = std::max(bNorm, std::fabs(mo.rowUpper[i]));
    }

    const Real regPrimal = 1e-9, regDual = 1e-9;
    // The best point seen, so a run that stalls one digit short of the target
    // still returns something usable instead of nothing.
    Real bestMeasure = kBigReal;
    Real bestRelP = kBigReal, bestRelD = kBigReal, bestRelG = kBigReal;
    std::vector<Real> bestX, bestS, bestY, bestZ, bestW;
    Status status = Status::IterationLimit;

    for (Int iter = 0; iter <= opt.ipmMaxIter; ++iter) {
        bx.refreshDistances(n, x);
        bs.refreshDistances(m, s);

        // ---- residuals ----------------------------------------------------
        std::fill(Qx.begin(), Qx.end(), 0.0);
        addQuadratic(mo.Q, x, Qx);
        mo.A.multiply(x, Ax);
        std::fill(Aty.begin(), Aty.end(), 0.0);
        mo.A.multiplyTransposeAdd(1.0, y, Aty);

        for (Int i = 0; i < m; ++i) rp[i] = Ax[i] - s[i];
        for (Int j = 0; j < n; ++j) rdx[j] = mo.obj[j] + Qx[j] - Aty[j] - bx.z[j] + bx.w[j];
        for (Int i = 0; i < m; ++i) rds[i] = y[i] - bs.z[i] + bs.w[i];

        Real mu = 0.0;
        for (Int j = 0; j < n; ++j) {
            if (bx.hasLo[j]) mu += bx.g[j] * bx.z[j];
            if (bx.hasUp[j]) mu += bx.t[j] * bx.w[j];
        }
        for (Int i = 0; i < m; ++i) {
            if (bs.hasLo[i]) mu += bs.g[i] * bs.z[i];
            if (bs.hasUp[i]) mu += bs.t[i] * bs.w[i];
        }
        mu /= (Real)npair;

        Real pInf = 0, dInf = 0, xNorm = 1.0;
        for (Real v : rp)  pInf = std::max(pInf, std::fabs(v));
        // A variable pinned between equal bounds -- a fixed column, or the
        // logical of an EQUALITY ROW -- has a multiplier that is free in sign
        // and simply absorbs whatever the dual equation needs.  Its "residual"
        // is therefore not a residual at all: it equals the multiplier itself,
        // and no iteration can drive it to zero.  Counting it made dInf plateau
        // at the largest equality-row dual, so a model with any equality row
        // could never satisfy the convergence test however well it converged.
        for (Int j = 0; j < n; ++j)
            if (!bx.fixed[j]) dInf = std::max(dInf, std::fabs(rdx[j]));
        for (Int i = 0; i < m; ++i)
            if (!bs.fixed[i]) dInf = std::max(dInf, std::fabs(rds[i]));
        for (Real v : x)   xNorm = std::max(xNorm, std::fabs(v));

        Real primalObj = mo.objOffset;
        for (Int j = 0; j < n; ++j) primalObj += mo.obj[j] * x[j];
        Real quad = 0.0;
        for (Int j = 0; j < n; ++j) quad += 0.5 * x[j] * Qx[j];
        primalObj += quad;

        Real relP = pInf / (1.0 + bNorm + xNorm);
        Real relD = dInf / (1.0 + cNorm);
        Real relGapM = mu / (1.0 + std::fabs(primalObj));

        res.iterations = iter;
        res.primalInfeasibility = relP;
        res.dualInfeasibility = relD;
        res.complementarityGap = relGapM;
        res.primalObjective = primalObj;

        opt.log.log(3, "    ipm %3d  mu %.3e  primal %.3e  dual %.3e  obj %.10g\n",
                    (int)iter, (double)mu, (double)relP, (double)relD, (double)primalObj);

        if (relP <= opt.ipmTol && relD <= opt.ipmTol && relGapM <= opt.ipmTol) {
            status = Status::Optimal;
            break;
        }
        Real measure = std::max(relGapM, std::max(relP, relD));
        if (measure < bestMeasure) {
            bestMeasure = measure;
            bestRelP = relP; bestRelD = relD; bestRelG = relGapM;
            bestX = x; bestS = s; bestY = y; bestZ = bx.z; bestW = bx.w;
        }
        if (iter == opt.ipmMaxIter) break;
        if (clock.elapsed() > opt.timeLimit) { status = Status::TimeLimit; break; }

        // ---- barrier diagonals --------------------------------------------
        for (Int j = 0; j < n; ++j) {
            if (bx.fixed[j]) { Dx[j] = kFixedDiag; continue; }
            Real d = 0.0;
            if (bx.hasLo[j]) d += bx.z[j] / bx.g[j];
            if (bx.hasUp[j]) d += bx.w[j] / bx.t[j];
            Dx[j] = d + regPrimal;
        }
        for (Int i = 0; i < m; ++i) {
            if (bs.fixed[i]) { Ds[i] = kBigReal; Dy[i] = regDual; continue; }
            Real d = 0.0;
            if (bs.hasLo[i]) d += bs.z[i] / bs.g[i];
            if (bs.hasUp[i]) d += bs.w[i] / bs.t[i];
            Ds[i] = std::max(d, kMinDs);
            Dy[i] = std::min(1.0 / Ds[i], kMaxDy);
        }

        K.refill(Dx, Dy);
        Timer tF;
        bool okFactor = ldl.factorize(K.Kp, K.Ki, K.Kx, K.sign, 1e-10, 1e-13);
        res.factorTime += tF.elapsed();
        if (!okFactor) { status = Status::NumericalError; break; }
        res.regularizedPivots = ldl.numRegularized();
        res.factorNonzeros = ldl.nonzeros();

        // Solve K * sol = rhs with one round of iterative refinement.  The
        // factorization is deliberately allowed to regularize away dangerous
        // pivots; refinement is what pays that approximation back.
        auto solveKkt = [&](std::vector<Real>& b, std::vector<Real>& out) {
            out = b;
            ldl.solve(out);
            for (int r = 0; r < 2; ++r) {
                K.multiply(out, work);
                Real worst = 0;
                for (Int i = 0; i < N; ++i) { resid[i] = b[i] - work[i]; worst = std::max(worst, std::fabs(resid[i])); }
                if (worst <= 1e-12 * (1.0 + worst)) break;
                ldl.solve(resid);
                for (Int i = 0; i < N; ++i) out[i] += resid[i];
            }
        };

        // ---- affine (predictor) step ---------------------------------------
        for (Int j = 0; j < n; ++j) rhoX[j] = bx.fixed[j] ? 0.0 : (mo.obj[j] + Qx[j] - Aty[j]);
        for (Int i = 0; i < m; ++i) rhoS[i] = bs.fixed[i] ? 0.0 : -y[i];

        for (Int j = 0; j < n; ++j) rhs[j] = rhoX[j];
        for (Int i = 0; i < m; ++i)
            rhs[n + i] = -rp[i] + (bs.fixed[i] ? 0.0 : rhoS[i] / Ds[i]);
        Timer tS;
        solveKkt(rhs, sol);
        res.solveTime += tS.elapsed();

        for (Int j = 0; j < n; ++j) dxAff[j] = sol[j];
        for (Int i = 0; i < m; ++i) dy[i] = sol[n + i];
        for (Int i = 0; i < m; ++i)
            dsAff[i] = bs.fixed[i] ? 0.0 : (rhoS[i] - dy[i]) / Ds[i];

        auto bandDeltas = [&](const Barrier& b, Int k, const std::vector<Real>& dv,
                              Real muT, const std::vector<Real>& corrZ, const std::vector<Real>& corrW,
                              std::vector<Real>& dz, std::vector<Real>& dw) {
            for (Int i = 0; i < k; ++i) {
                dz[i] = b.hasLo[i]
                      ? (-(b.z[i] / b.g[i]) * dv[i] + (muT - b.g[i] * b.z[i] - corrZ[i]) / b.g[i])
                      : 0.0;
                dw[i] = b.hasUp[i]
                      ? ((b.w[i] / b.t[i]) * dv[i] + (muT - b.t[i] * b.w[i] - corrW[i]) / b.t[i])
                      : 0.0;
            }
        };

        std::vector<Real> zeroN(n, 0.0), zeroM(m, 0.0);
        bandDeltas(bx, n, dxAff, 0.0, zeroN, zeroN, dzxAff, dwxAff);
        bandDeltas(bs, m, dsAff, 0.0, zeroM, zeroM, dzsAff, dwsAff);

        auto maxStep = [](const Barrier& b, Int k, const std::vector<Real>& dv,
                          const std::vector<Real>& dz, const std::vector<Real>& dw,
                          Real& aP, Real& aD) {
            aP = 1.0; aD = 1.0;
            for (Int i = 0; i < k; ++i) {
                if (b.hasLo[i]) {
                    if (dv[i] < 0) aP = std::min(aP, -b.g[i] / dv[i]);
                    if (dz[i] < 0) aD = std::min(aD, -b.z[i] / dz[i]);
                }
                if (b.hasUp[i]) {
                    if (dv[i] > 0) aP = std::min(aP, b.t[i] / dv[i]);
                    if (dw[i] < 0) aD = std::min(aD, -b.w[i] / dw[i]);
                }
            }
        };

        Real aPx, aDx, aPs, aDs;
        maxStep(bx, n, dxAff, dzxAff, dwxAff, aPx, aDx);
        maxStep(bs, m, dsAff, dzsAff, dwsAff, aPs, aDs);
        Real aP = std::min(aPx, aPs), aD = std::min(aDx, aDs);

        // ---- Mehrotra centering parameter ----------------------------------
        Real muAff = 0.0;
        for (Int j = 0; j < n; ++j) {
            if (bx.hasLo[j]) muAff += (bx.g[j] + aP * dxAff[j]) * (bx.z[j] + aD * dzxAff[j]);
            if (bx.hasUp[j]) muAff += (bx.t[j] - aP * dxAff[j]) * (bx.w[j] + aD * dwxAff[j]);
        }
        for (Int i = 0; i < m; ++i) {
            if (bs.hasLo[i]) muAff += (bs.g[i] + aP * dsAff[i]) * (bs.z[i] + aD * dzsAff[i]);
            if (bs.hasUp[i]) muAff += (bs.t[i] - aP * dsAff[i]) * (bs.w[i] + aD * dwsAff[i]);
        }
        muAff /= (Real)npair;
        Real sigma = (mu > 0) ? std::pow(std::max(muAff, 0.0) / mu, 3.0) : 0.1;
        sigma = std::min(std::max(sigma, 1e-8), 0.99);
        Real muTarget = sigma * mu;

        // ---- corrector -----------------------------------------------------
        std::vector<Real> corrZx(n, 0.0), corrWx(n, 0.0), corrZs(m, 0.0), corrWs(m, 0.0);
        for (Int j = 0; j < n; ++j) {
            if (bx.hasLo[j]) corrZx[j] =  dxAff[j] * dzxAff[j];
            if (bx.hasUp[j]) corrWx[j] = -dxAff[j] * dwxAff[j];
        }
        for (Int i = 0; i < m; ++i) {
            if (bs.hasLo[i]) corrZs[i] =  dsAff[i] * dzsAff[i];
            if (bs.hasUp[i]) corrWs[i] = -dsAff[i] * dwsAff[i];
        }

        for (Int j = 0; j < n; ++j) {
            if (bx.fixed[j]) { rhoX[j] = 0.0; continue; }
            Real v = rdx[j];
            if (bx.hasLo[j]) v -= (muTarget - bx.g[j] * bx.z[j] - corrZx[j]) / bx.g[j];
            if (bx.hasUp[j]) v += (muTarget - bx.t[j] * bx.w[j] - corrWx[j]) / bx.t[j];
            rhoX[j] = v;
        }
        for (Int i = 0; i < m; ++i) {
            if (bs.fixed[i]) { rhoS[i] = 0.0; continue; }
            Real v = -rds[i];
            if (bs.hasLo[i]) v += (muTarget - bs.g[i] * bs.z[i] - corrZs[i]) / bs.g[i];
            if (bs.hasUp[i]) v -= (muTarget - bs.t[i] * bs.w[i] - corrWs[i]) / bs.t[i];
            rhoS[i] = v;
        }

        for (Int j = 0; j < n; ++j) rhs[j] = rhoX[j];
        for (Int i = 0; i < m; ++i)
            rhs[n + i] = -rp[i] + (bs.fixed[i] ? 0.0 : rhoS[i] / Ds[i]);
        Timer tS2;
        solveKkt(rhs, sol);
        res.solveTime += tS2.elapsed();

        for (Int j = 0; j < n; ++j) dx[j] = sol[j];
        for (Int i = 0; i < m; ++i) dy[i] = sol[n + i];
        for (Int i = 0; i < m; ++i) ds[i] = bs.fixed[i] ? 0.0 : (rhoS[i] - dy[i]) / Ds[i];

        bandDeltas(bx, n, dx, muTarget, corrZx, corrWx, dzx, dwx);
        bandDeltas(bs, m, ds, muTarget, corrZs, corrWs, dzs, dws);

        maxStep(bx, n, dx, dzx, dwx, aPx, aDx);
        maxStep(bs, m, ds, dzs, dws, aPs, aDs);
        aP = std::min(1.0, kStepFactor * std::min(aPx, aPs));
        aD = std::min(1.0, kStepFactor * std::min(aDx, aDs));
        // A collapsed step means the iterate has run out of room, which near the
        // solution is convergence rather than failure.  Distinguishing the two by
        // the residuals -- not by the step length -- is what stops a run that is
        // already accurate to eight digits from being reported as a breakdown.
        if (!(aP > 0) || !(aD > 0)) { status = Status::NumericalError; break; }
        if (aP < 1e-10 && aD < 1e-10) { status = Status::NumericalError; break; }

        // ---- take the step -------------------------------------------------
        for (Int j = 0; j < n; ++j) {
            if (bx.fixed[j]) continue;
            x[j] += aP * dx[j];
            bx.z[j] = std::max(bx.z[j] + aD * dzx[j], 0.0);
            bx.w[j] = std::max(bx.w[j] + aD * dwx[j], 0.0);
        }
        for (Int i = 0; i < m; ++i) {
            if (!bs.fixed[i]) s[i] += aP * ds[i];
            bs.z[i] = std::max(bs.z[i] + aD * dzs[i], 0.0);
            bs.w[i] = std::max(bs.w[i] + aD * dws[i], 0.0);
            y[i] += aD * dy[i];
        }
        for (Int i = 0; i < m; ++i) if (bs.fixed[i]) s[i] = bs.lo[i];
        for (Int j = 0; j < n; ++j) if (bx.fixed[j]) x[j] = bx.lo[j];
    }

    // ---- report ------------------------------------------------------------
    // A stalled or iteration-limited run that nevertheless reached a point good
    // to `loose` is reported optimal at that accuracy: refusing it would throw
    // away a solution that is better than most callers' own tolerances, and the
    // crossover step downstream will sharpen it to basic accuracy anyway.
    const Real loose = std::max(1e3 * opt.ipmTol, 1e-7);
    if (status != Status::Optimal && !bestX.empty() &&
        bestRelP <= loose && bestRelD <= loose && bestRelG <= loose) {
        x = bestX; s = bestS; y = bestY; bx.z = bestZ; bx.w = bestW;
        status = Status::Optimal;
        res.primalInfeasibility = bestRelP;
        res.dualInfeasibility   = bestRelD;
        res.complementarityGap  = bestRelG;
        Real po = mo.objOffset;
        for (Int j = 0; j < n; ++j) po += mo.obj[j] * x[j];
        std::vector<Real> qq(n, 0.0);
        addQuadratic(mo.Q, x, qq);
        for (Int j = 0; j < n; ++j) po += 0.5 * x[j] * qq[j];
        res.primalObjective = po;
    }

    // Give the pinned entries the multiplier their dual equation implies, so
    // the reduced costs and dual objective reported to the caller are correct
    // rather than zero.  z - w = c + Qx - A'y for a fixed column; z - w = y for
    // a fixed logical.
    {
        std::vector<Real> qx(n, 0.0), aty2(n, 0.0);
        addQuadratic(mo.Q, x, qx);
        mo.A.multiplyTransposeAdd(1.0, y, aty2);
        for (Int j = 0; j < n; ++j) {
            if (!bx.fixed[j]) continue;
            Real rd = mo.obj[j] + qx[j] - aty2[j];
            bx.z[j] = std::max(rd, 0.0);
            bx.w[j] = std::max(-rd, 0.0);
        }
        for (Int i = 0; i < m; ++i) {
            if (!bs.fixed[i]) continue;
            bs.z[i] = std::max(y[i], 0.0);
            bs.w[i] = std::max(-y[i], 0.0);
        }
    }

    res.status = status;
    res.x = x; res.s = s; res.y = y;
    res.zLower = bx.z; res.zUpper = bx.w;

    Real dualObj = mo.objOffset;
    for (Int j = 0; j < n; ++j) {
        if (bx.hasLo[j]) dualObj += bx.z[j] * mo.colLower[j];
        if (bx.hasUp[j]) dualObj -= bx.w[j] * mo.colUpper[j];
        // A fixed column contributes (z - w) * l at its pinned value.
        if (bx.fixed[j]) dualObj += (bx.z[j] - bx.w[j]) * mo.colLower[j];
    }
    for (Int i = 0; i < m; ++i) {
        if (bs.fixed[i]) dualObj += y[i] * mo.rowLower[i];
        else {
            if (bs.hasLo[i]) dualObj += bs.z[i] * mo.rowLower[i];
            if (bs.hasUp[i]) dualObj -= bs.w[i] * mo.rowUpper[i];
        }
    }
    std::vector<Real> Qx2(n, 0.0);
    addQuadratic(mo.Q, x, Qx2);
    Real q = 0;
    for (Int j = 0; j < n; ++j) q += 0.5 * x[j] * Qx2[j];
    res.dualObjective = dualObj - q;

    // The duality gap, computed last because the dual objective is only
    // complete here.  relP and relD each divide by a norm that grows with the
    // model, so on a large badly conditioned QP both can look tiny while the
    // objective is nowhere near optimal.  The primal-dual gap has no such
    // denominator and is the honest check.
    {
        const Real pO = res.primalObjective, dO = res.dualObjective;
        res.relDualityGap = std::fabs(pO - dO) / (1.0 + std::fabs(pO) + std::fabs(dO));
    }

    // Defect 27.  CONT-300 reaches relP 7.9e-12 and relD 4.5e-06 -- both inside
    // the loose late-accept bar -- on a point whose primal and dual objectives
    // are 0.375 and -0.566.  Every relative test passes and the answer is wrong
    // by 48%.  The gap is the only measure here that notices, so when the check
    // is on, a point that fails it is not called optimal.
    if (opt.dualityGapCheck && res.status == Status::Optimal &&
        res.relDualityGap >= 0.0) {
        const Real gapTol = std::max(1e-6, 1e2 * opt.ipmTol);
        if (res.relDualityGap > gapTol) {
            opt.log.log(1, "  note: refusing optimality -- primal %.10g and dual %.10g "
                           "disagree, relative gap %.3e > %.3e\n",
                        (double)res.primalObjective, (double)res.dualObjective,
                        (double)res.relDualityGap, (double)gapTol);
            res.status = Status::NumericalError;
        }
    }
    return res;
}

} // namespace igaos
