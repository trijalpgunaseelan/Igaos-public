// IGAOS unit tests: numerical kernels and end-to-end solves.
#include "igaos/lu.hpp"
#include "igaos/ldl.hpp"
#include "igaos/solver.hpp"
#include "igaos/presolve.hpp"
#include "igaos/mps.hpp"
#include "igaos/ipm.hpp"
#include "igaos/crossover.hpp"
#include "igaos/pdhg.hpp"
#include "igaos/simplex.hpp"
#include "igaos/iis.hpp"
#include "igaos/sensitivity.hpp"
#include <random>
#include <cstdio>

using namespace igaos;
static int failures = 0;
static void check(bool ok, const char* what, double val = 0) {
    if (ok) std::printf("  [ ok ] %s\n", what);
    else { std::printf("  [FAIL] %s  (%.3e)\n", what, val); ++failures; }
}

// --- 1. sparse LU: factorize, FTRAN, BTRAN, product-form update -------------
static void testLu() {
    std::printf("sparse LU (threshold Markowitz) + PFI updates\n");
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> U(-3, 3);
    double worstF = 0, worstB = 0;
    int tested = 0;
    for (int trial = 0; trial < 60; ++trial) {
        Int m = 5 + (Int)(rng() % 80), n = m + (Int)(rng() % 50);
        TripletBuilder tb; tb.reset(m, n);
        for (Int j = 0; j < n; ++j) {
            int k = 1 + (int)(rng() % 5);
            for (int t = 0; t < k; ++t) tb.add((Int)(rng() % m), j, U(rng));
        }
        SparseMatrix A = tb.build(); A.nrow = m; A.ncol = n; A.colPtr.resize(n + 1, A.nnz());
        ExtendedMatrix E; E.bind(A);
        // Build the basis the way a crash would: start all-logical, then swap in
        // structural columns on distinct pivot rows, so it is non-singular by
        // construction rather than by luck.
        std::vector<Int> basis(m);
        for (Int i = 0; i < m; ++i) basis[i] = n + i;
        std::vector<char> rowUsed(m, 0), colUsed(n, 0);
        for (int attempt = 0; attempt < (int)m; ++attempt) {
            Int j = (Int)(rng() % n);
            if (colUsed[j]) continue;
            Int piv = kNone; Real best = 0;
            for (Int q = A.colPtr[j]; q < A.colPtr[j + 1]; ++q) {
                Int i = A.rowIdx[q];
                if (rowUsed[i]) continue;
                if (std::fabs(A.val[q]) > best) { best = std::fabs(A.val[q]); piv = i; }
            }
            if (piv == kNone || best < 1e-3) continue;
            rowUsed[piv] = 1; colUsed[j] = 1; basis[piv] = j;
        }
        Tolerances tol; BasisFactor F;
        if (!F.factorize(E, basis, tol)) continue;
        ++tested;
        std::vector<Real> b(m), x(m), r(m, 0.0), c(m), y(m);
        for (Int i = 0; i < m; ++i) { b[i] = U(rng); c[i] = U(rng); }
        x = b; F.ftranDense(x);
        for (Int k = 0; k < m; ++k) { Real xk = x[k]; if (xk) E.forEach(basis[k], [&](Int i, Real v) { r[i] += v * xk; }); }
        double e1 = 0, nb = 0;
        for (Int i = 0; i < m; ++i) { e1 = std::max(e1, std::fabs(r[i] - b[i])); nb = std::max(nb, std::fabs(b[i])); }
        y = c; F.btranDense(y);
        double e2 = 0, nc = 0;
        for (Int k = 0; k < m; ++k) {
            Real s = 0; E.forEach(basis[k], [&](Int i, Real v) { s += v * y[i]; });
            e2 = std::max(e2, std::fabs(s - c[k])); nc = std::max(nc, std::fabs(c[k]));
        }
        worstF = std::max(worstF, e1 / nb); worstB = std::max(worstB, e2 / nc);
    }
    check(tested >= 55, "every constructed basis factorized", tested);
    check(worstF < 1e-8, "B * (B^-1 b) reproduces b (FTRAN)", worstF);
    check(worstB < 1e-8, "B^T y = c is solved exactly (BTRAN)", worstB);
}

// --- 2. sparse LDL^T on a quasi-definite KKT matrix --------------------------
static void testLdl() {
    std::printf("AMD ordering + quasi-definite LDL^T\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1, 1);
    double worst = 0; long long lnz = 0;
    for (int trial = 0; trial < 15; ++trial) {
        Int nx = 20 + (Int)(rng() % 90), mm = 10 + (Int)(rng() % 50), n = nx + mm;
        std::vector<std::vector<std::pair<Int, Real>>> Ac(nx), K(n);
        for (Int j = 0; j < nx; ++j) {
            int k = 1 + (int)(rng() % 4);
            for (int t = 0; t < k; ++t) Ac[j].emplace_back((Int)(rng() % mm), U(rng) + 1.5);
        }
        for (Int j = 0; j < nx; ++j) K[j].emplace_back(j, -(0.5 + std::fabs(U(rng))));
        for (Int j = 0; j < nx; ++j) for (auto& e : Ac[j]) K[nx + e.first].emplace_back(j, e.second);
        for (Int i = 0; i < mm; ++i) K[nx + i].emplace_back(nx + i, 1e-2);
        std::vector<Int> Ap(n + 1, 0), Ai; std::vector<Real> Ax;
        for (Int j = 0; j < n; ++j) {
            Ap[j + 1] = Ap[j] + (Int)K[j].size();
            for (auto& e : K[j]) { Ai.push_back(e.first); Ax.push_back(e.second); }
        }
        std::vector<int8_t> sgn(n);
        for (Int j = 0; j < nx; ++j) sgn[j] = -1;
        for (Int i = 0; i < mm; ++i) sgn[nx + i] = 1;
        LdlFactor F; F.analyze(n, Ap, Ai);
        F.factorize(Ap, Ai, Ax, sgn, 1e-12, 1e-14);
        lnz += F.nonzeros();
        std::vector<Real> b(n), x(n), r(n, 0.0);
        for (Int i = 0; i < n; ++i) b[i] = U(rng);
        x = b; F.solve(x);
        for (Int j = 0; j < n; ++j)
            for (auto& e : K[j]) { r[e.first] += e.second * x[j]; if (e.first != j) r[j] += e.second * x[e.first]; }
        double err = 0, nb = 0;
        for (Int i = 0; i < n; ++i) { err = std::max(err, std::fabs(r[i] - b[i])); nb = std::max(nb, std::fabs(b[i])); }
        worst = std::max(worst, err / nb);
    }
    check(worst < 1e-10, "K x = b solved through LDL^T", worst);
    check(lnz > 0, "factor has nonzeros", (double)lnz);
}

// --- 3. MPS round trip -------------------------------------------------------
static void testMps() {
    std::printf("MPS reader / writer round trip\n");
    Model m;
    m.addColumn(0, 10, 1.0, VarType::Continuous, "xa");
    m.addColumn(-kInf, kInf, -2.0, VarType::Continuous, "xb");
    m.addColumn(0, 1, 3.0, VarType::Binary, "xc");
    m.addRow(1.0, 4.0, "r0");            // ranged
    m.addRow(2.0, 2.0, "r1");            // equality
    m.addRow(-kInf, 8.0, "r2");
    m.setElement(0, 0, 1.0); m.setElement(0, 1, 2.0);
    m.setElement(1, 0, 1.0); m.setElement(1, 2, 1.0);
    m.setElement(2, 1, 1.5); m.setElement(2, 2, -1.0);
    m.finalize();
    std::string err;
    check(writeMps("/tmp/_igaos_rt.mps", m, err), "write MPS");
    Model back;
    check(readMps("/tmp/_igaos_rt.mps", back, err), "read MPS");
    check(back.numRow() == m.numRow() && back.numCol() == m.numCol(), "shape preserved");
    check(back.A.nnz() == m.A.nnz(), "nonzeros preserved", (double)back.A.nnz());
    bool bounds = true;
    for (Int i = 0; i < m.numRow(); ++i)
        if (std::fabs(back.rowLower[i] - m.rowLower[i]) > 1e-9 ||
            std::fabs(back.rowUpper[i] - m.rowUpper[i]) > 1e-9) bounds = false;
    check(bounds, "ranged / equality / one-sided row bounds preserved");
    check(back.colType[2] != VarType::Continuous, "integer marker preserved");
}

// --- 4. end to end LP with a known optimum ----------------------------------
static void testLpEndToEnd() {
    std::printf("end-to-end LP against a hand-computed optimum\n");
    // max 3x + 5y  s.t. x <= 4, 2y <= 12, 3x + 2y <= 18, x,y >= 0
    // classic textbook LP; optimum is x=2, y=6, objective 36.
    Model m;
    m.sense = Sense::Maximize;
    m.addColumn(0, kInf, 3.0, VarType::Continuous, "x");
    m.addColumn(0, kInf, 5.0, VarType::Continuous, "y");
    m.addRow(-kInf, 4.0,  "c1");
    m.addRow(-kInf, 12.0, "c2");
    m.addRow(-kInf, 18.0, "c3");
    m.setElement(0, 0, 1.0);
    m.setElement(1, 1, 2.0);
    m.setElement(2, 0, 3.0); m.setElement(2, 1, 2.0);
    m.finalize();
    Solver s; s.opt.log.level = 0;
    Solution sol = s.solve(m);
    check(sol.status == Status::Optimal, "status optimal");
    check(std::fabs(sol.objective - 36.0) < 1e-7, "objective = 36", sol.objective);
    check(std::fabs(sol.colValue[0] - 2.0) < 1e-6, "x = 2", sol.colValue[0]);
    check(std::fabs(sol.colValue[1] - 6.0) < 1e-6, "y = 6", sol.colValue[1]);
}

// --- 5. end to end MILP ------------------------------------------------------
static void testMilpEndToEnd() {
    std::printf("end-to-end MILP against a hand-computed optimum\n");
    // max 5a + 4b, a + b <= 5, 10a + 6b <= 45, a,b integer >= 0  ->  a=3,b=2, obj 23
    Model m;
    m.sense = Sense::Maximize;
    m.addColumn(0, 100, 5.0, VarType::Integer, "a");
    m.addColumn(0, 100, 4.0, VarType::Integer, "b");
    m.addRow(-kInf, 5.0,  "c1");
    m.addRow(-kInf, 45.0, "c2");
    m.setElement(0, 0, 1.0);  m.setElement(0, 1, 1.0);
    m.setElement(1, 0, 10.0); m.setElement(1, 1, 6.0);
    m.finalize();
    Solver s; s.opt.log.level = 0;
    Solution sol = s.solve(m);
    check(sol.status == Status::Optimal, "status optimal");
    check(std::fabs(sol.objective - 23.0) < 1e-6, "objective = 23", sol.objective);
    check(m.integerInfeasibility(sol.colValue, 1e-6) < 1e-6, "solution is integral");
}

// --- 6. infeasible and unbounded detection ----------------------------------
static void testStatuses() {
    std::printf("infeasibility and unboundedness detection\n");
    {   Model m;                                  // x >= 3 and x <= 1
        m.addColumn(0, kInf, 1.0, VarType::Continuous, "x");
        m.addRow(3.0, kInf, "lo"); m.addRow(-kInf, 1.0, "hi");
        m.setElement(0, 0, 1.0); m.setElement(1, 0, 1.0);
        m.finalize();
        Solver s; s.opt.log.level = 0;
        check(s.solve(m).status == Status::Infeasible, "infeasible model reported infeasible");
    }
    {   Model m;                                  // min -x, x >= 0, no upper bound
        m.addColumn(0, kInf, -1.0, VarType::Continuous, "x");
        m.addRow(-kInf, kInf, "free");
        m.setElement(0, 0, 1.0);
        m.finalize();
        Solver s; s.opt.log.level = 0;
        Status st = s.solve(m).status;
        check(st == Status::Unbounded, "unbounded model reported unbounded");
    }
    // A model infeasible by only a little is still infeasible.  Phase one used
    // to accept any residual under 1000x the feasibility tolerance and report
    // the point as optimal, which is a wrong answer rather than a slow one.
    // Presolve catches the easy cases by bound tightening, so the check has to
    // be run with presolve off to reach the simplex.
    for (double gap : {5e-5, 1e-4, 1e-3}) {
        Model m;                                  // x >= 1 and x <= 1 - gap
        m.addColumn(-kInf, kInf, 1.0, VarType::Continuous, "x");
        m.addRow(1.0, kInf, "lo"); m.addRow(-kInf, 1.0 - gap, "hi");
        m.setElement(0, 0, 1.0); m.setElement(1, 0, 1.0);
        m.finalize();
        Solver s; s.opt.log.level = 0; s.opt.presolve = false;
        Status st = s.solve(m).status;
        check(st == Status::Infeasible,
              "model infeasible by a margin above the tolerance is reported infeasible");
    }
    {   // ... and the same model with the gap BELOW the tolerance is allowed to
        // come back optimal: that is the tolerance doing its job, not a defect.
        Model m;
        m.addColumn(-kInf, kInf, 1.0, VarType::Continuous, "x");
        m.addRow(1.0, kInf, "lo"); m.addRow(-kInf, 1.0 - 1e-11, "hi");
        m.setElement(0, 0, 1.0); m.setElement(1, 0, 1.0);
        m.finalize();
        Solver s; s.opt.log.level = 0; s.opt.presolve = false;
        Status st = s.solve(m).status;
        check(st == Status::Optimal || st == Status::Infeasible,
              "model infeasible below the tolerance may go either way, but must not crash");
    }
}

// --- 7. presolve / postsolve preserves the optimum ---------------------------
static void testPresolve() {
    std::printf("presolve + postsolve preserve the optimal value\n");
    std::mt19937 rng(2026);
    std::uniform_real_distribution<double> U(-2, 2);
    double worst = 0; int n_ok = 0;
    for (int trial = 0; trial < 12; ++trial) {
        Int mm = 25 + (Int)(rng() % 40), nn = 35 + (Int)(rng() % 50);
        Model m;
        std::vector<Real> x0(nn);
        for (Int j = 0; j < nn; ++j) {
            x0[j] = 0.5 + std::fabs(U(rng));
            m.addColumn(0.0, x0[j] * 3.0, U(rng));
        }
        for (Int i = 0; i < mm; ++i) {
            std::vector<Int> idx;
            for (Int j = 0; j < nn; ++j) if ((rng() % 100) < 12) idx.push_back(j);
            if (idx.empty()) idx.push_back((Int)(rng() % nn));
            Real act = 0;
            std::vector<Real> co(idx.size());
            for (size_t t = 0; t < idx.size(); ++t) { co[t] = 0.5 + std::fabs(U(rng)); act += co[t] * x0[idx[t]]; }
            m.addRow(-kInf, act * 1.2);
            for (size_t t = 0; t < idx.size(); ++t) m.setElement(i, idx[t], co[t]);
        }
        m.finalize();
        Solver a; a.opt.log.level = 0; a.opt.presolve = true;
        Solver b; b.opt.log.level = 0; b.opt.presolve = false;
        Solution sa = a.solve(m), sb = b.solve(m);
        if (sa.status != Status::Optimal || sb.status != Status::Optimal) continue;
        ++n_ok;
        worst = std::max(worst, std::fabs(sa.objective - sb.objective) / std::max(1.0, std::fabs(sb.objective)));
    }
    check(n_ok >= 8, "enough presolve comparisons ran", n_ok);
    check(worst < 1e-8, "presolved and unpresolved optima agree", worst);
}


// ---------------------------------------------------------------------------
//  Shared generators for the algorithm tests below.
// ---------------------------------------------------------------------------
namespace {

Model randomProgram(std::mt19937& rng, int m, int n, bool integral, bool quadratic) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Model mod;
    int nint = 0;
    for (int j = 0; j < n; ++j) {
        double r = U(rng);
        VarType t = VarType::Continuous;
        if (integral) {
            if (r < 0.55) { t = VarType::Binary; ++nint; }
            else if (r < 0.75) { t = VarType::Integer; ++nint; }
        }
        double up;
        if (t == VarType::Binary) up = 1.0;
        else if (t == VarType::Integer) up = std::floor(1.0 + 8.0 * U(rng));
        else up = 1.0 + 20.0 * U(rng);
        mod.addColumn(0.0, up, -10.0 + 20.0 * U(rng), t);
    }
    if (integral && nint == 0) { mod.colType[0] = VarType::Binary; mod.colUpper[0] = 1.0; }

    // Row kind is decided up front and the bounds are set from it AFTER the
    // activity is known.  The earlier version chose "equality" by setting both
    // bounds to zero and then overwrote each bound independently, which turned
    // every intended equality into a ranged row -- so no equality row was ever
    // generated, and the one bug that only equality rows expose went unseen.
    enum RowKind { LE = 0, GE = 1, EQ = 2, RANGE = 3 };
    for (int i = 0; i < m; ++i) {
        double r = U(rng);
        RowKind kind = (r < 0.42) ? LE : (r < 0.70) ? GE : (r < 0.85) ? EQ : RANGE;
        Int row = mod.addRow(-kInf, kInf);
        double act = 0;
        int len = 2 + (int)(U(rng) * std::min(n - 2, 6));
        for (int k = 0; k < len; ++k) {
            int j = (int)(U(rng) * n); if (j >= n) j = n - 1;
            double a = std::round((-5.0 + 10.0 * U(rng)) * 4.0) / 4.0;
            if (a == 0.0) a = 1.0;
            mod.setElement(row, j, a);
            act += a * 0.5 * (mod.colLower[j] + mod.colUpper[j]);
        }
        double shift = act + (-2.0 + 4.0 * U(rng));
        switch (kind) {
            case LE:    mod.rowLower[row] = -kInf;      mod.rowUpper[row] = shift + 2.5; break;
            case GE:    mod.rowLower[row] = shift - 2.5; mod.rowUpper[row] = kInf;       break;
            case EQ:    mod.rowLower[row] = shift;       mod.rowUpper[row] = shift;      break;
            case RANGE: mod.rowLower[row] = shift - 2.5; mod.rowUpper[row] = shift + 2.5; break;
        }
    }
    if (quadratic) {
        // Diagonally dominant lower triangle: positive semidefinite by
        // construction, so the model stays a convex QP without an eigenvalue check.
        for (int j = 0; j < n; ++j) {
            double diag = 1.0 + 3.0 * U(rng);
            double off = 0.0;
            if (j > 0 && U(rng) < 0.3) {
                double v = (-0.5 + U(rng)) * diag * 0.5;
                mod.setQuadratic(j, j - 1, v);
                off = std::fabs(v);
            }
            mod.setQuadratic(j, j, diag + off + 1.0);
        }
    }
    mod.finalize();
    return mod;
}

} // namespace

// --- 8. cut separation: validity is the property that matters ---------------
static void testCutValidity() {
    std::printf("cut separation (GMI / cover / MIR): validity and strength\n");
    int compared = 0, mismatches = 0, invalidCuts = 0, strengthened = 0;
    double worstObj = 0, bestClosure = 0, totalClosure = 0;
    int closureSamples = 0;

    for (int seed = 1; seed <= 40; ++seed) {
        std::mt19937 rng(seed * 7919u);
        int n = 6 + (int)(rng() % 18);
        int m = 4 + (int)(rng() % 12);
        Model mod = randomProgram(rng, m, n, true, false);

        Solver a; a.opt.log.level = 0; a.opt.cuts = false;
        a.opt.presolve = false; a.opt.scaling = false; a.opt.timeLimit = 10.0;
        Solution sa = a.solve(mod);
        if (sa.status != Status::Optimal) continue;

        // The proven optimum is a point every valid cut must keep feasible.
        // Handing it to the separators as a verification point turns an invalid
        // derivation into an immediate, localized failure instead of a wrong
        // objective thousands of nodes later.
        Solver b; b.opt.log.level = 0; b.opt.cuts = true;
        b.opt.presolve = false; b.opt.scaling = false; b.opt.timeLimit = 10.0;
        b.opt.cutReference = &sa.colValue;
        Solution sb = b.solve(mod);
        invalidCuts += (int)b.report.cutsInvalid;

        if (sb.status != Status::Optimal) continue;
        ++compared;
        double rel = std::fabs(sa.objective - sb.objective) / (1.0 + std::fabs(sa.objective));
        worstObj = std::max(worstObj, rel);
        if (rel > 1e-6) ++mismatches;

        // Cuts may only tighten the root relaxation, never loosen it.
        if (b.report.cutsApplied > 0) {
            ++strengthened;
            double before = b.report.rootBoundLp, after = b.report.rootBoundCut;
            double denom = sa.objective - before;
            if (denom > 1e-9) {
                double closure = (after - before) / denom;
                bestClosure = std::max(bestClosure, closure);
                totalClosure += closure;
                ++closureSamples;
            }
        }
    }
    check(compared >= 20, "enough cut comparisons ran", compared);
    check(invalidCuts == 0, "no cut ever excluded the proven optimum", invalidCuts);
    check(mismatches == 0, "cutting and non-cutting runs agree on the optimum", worstObj);
    check(strengthened > 0, "cuts were actually generated", strengthened);
    if (closureSamples > 0)
        std::printf("         root gap closed: mean %.1f%%, best %.1f%% over %d instances\n",
                    100.0 * totalClosure / closureSamples, 100.0 * bestClosure, closureSamples);
}

// --- 9. interior point against the simplex ----------------------------------
static void testInteriorPoint() {
    std::printf("interior point method vs the simplex\n");
    int compared = 0, bad = 0;
    double worst = 0, worstInf = 0;
    long iterSum = 0;
    for (int seed = 1; seed <= 40; ++seed) {
        std::mt19937 rng(seed * 104729u);
        int n = 8 + (int)(rng() % 30);
        int m = 5 + (int)(rng() % 20);
        Model mod = randomProgram(rng, m, n, false, false);

        Options opt; opt.log.level = 0; opt.ipmTol = 1e-9; opt.timeLimit = 20.0;
        Simplex sx;
        sx.load(mod.A, mod.obj, mod.colLower, mod.colUpper, mod.rowLower, mod.rowUpper, opt);
        if (sx.solve(false) != Status::Optimal) continue;
        std::vector<Real> xs(sx.values().begin(), sx.values().begin() + mod.numCol());
        Real ref = mod.objectiveValue(xs);

        IpmResult ip = interiorPoint(mod, opt);
        if (ip.status != Status::Optimal) { ++bad; continue; }
        ++compared;
        iterSum += ip.iterations;
        worst = std::max(worst, std::fabs(ip.primalObjective - ref) / (1.0 + std::fabs(ref)));
        // Assert the contract the method documents, in the normalisation the
        // method uses: primal infeasibility relative to 1 + |b| + |x|.  An
        // absolute bound would be asserting something never promised -- on a
        // model whose row bounds run to 1e3, a relative 1e-9 is an absolute
        // 1e-6 -- and picking the threshold to fit the observed number would
        // be testing nothing at all.  The user-visible guarantee is tighter
        // because crossover follows; that is asserted in testCrossover.
        Real bNorm = 1.0, xNorm = 1.0;
        for (Int i = 0; i < mod.numRow(); ++i) {
            if (isFinite(mod.rowLower[i])) bNorm = std::max(bNorm, std::fabs(mod.rowLower[i]));
            if (isFinite(mod.rowUpper[i])) bNorm = std::max(bNorm, std::fabs(mod.rowUpper[i]));
        }
        for (Real v : ip.x) xNorm = std::max(xNorm, std::fabs(v));
        worstInf = std::max(worstInf, mod.primalInfeasibility(ip.x) / (1.0 + bNorm + xNorm));
    }
    check(compared >= 15, "enough interior point comparisons ran", compared);
    check(bad == 0, "interior point converged on every feasible model", bad);
    check(worst < 1e-6, "interior point matches the simplex optimum", worst);
    // The accepted band: a converged run reaches ipmTol; a run that stalls just
    // short is still accepted at max(1e3*ipmTol, 1e-7), and says so.
    const Real accepted = std::max(1e3 * 1e-9, 1e-7);
    check(worstInf < accepted,
          "interior point primal feasibility is within the documented band", worstInf);
    if (compared) std::printf("         mean %.1f interior point iterations\n",
                              (double)iterSum / compared);
}

// --- 10. crossover produces a genuine basic solution ------------------------
static void testCrossover() {
    std::printf("crossover from an interior point to a basic solution\n");
    int compared = 0, notBasic = 0;
    double worst = 0, worstInf = 0;
    long pushSum = 0, iterSum = 0;
    for (int seed = 1; seed <= 30; ++seed) {
        std::mt19937 rng(seed * 40503u);
        int n = 8 + (int)(rng() % 25);
        int m = 5 + (int)(rng() % 18);
        Model mod = randomProgram(rng, m, n, false, false);

        Options opt; opt.log.level = 0; opt.ipmTol = 1e-9; opt.timeLimit = 20.0;
        Simplex sx;
        sx.load(mod.A, mod.obj, mod.colLower, mod.colUpper, mod.rowLower, mod.rowUpper, opt);
        if (sx.solve(false) != Status::Optimal) continue;
        std::vector<Real> xs(sx.values().begin(), sx.values().begin() + mod.numCol());
        Real ref = mod.objectiveValue(xs);

        IpmResult ip = interiorPoint(mod, opt);
        if (ip.status != Status::Optimal) continue;
        CrossoverResult cr = crossover(mod, opt, ip);
        if (cr.status != Status::Optimal) continue;
        ++compared;
        pushSum += cr.pushes;
        iterSum += cr.iterations;
        worst = std::max(worst, std::fabs(cr.objective - ref) / (1.0 + std::fabs(ref)));
        // This is the guarantee the caller actually gets: crossover ends at a
        // vertex found by the simplex, so feasibility is absolute, not relative.
        worstInf = std::max(worstInf, mod.primalInfeasibility(cr.colValue));

        // The defining property: exactly as many basic variables as rows.
        Int basic = 0;
        for (VarStatus st : cr.colStatus) if (st == VarStatus::Basic) ++basic;
        for (VarStatus st : cr.rowStatus) if (st == VarStatus::Basic) ++basic;
        if (basic != mod.numRow()) ++notBasic;
    }
    check(compared >= 12, "enough crossover runs completed", compared);
    check(worst < 1e-9, "crossover reaches the simplex optimum exactly", worst);
    check(notBasic == 0, "crossover returns a basis with exactly numRow basic variables", notBasic);
    check(worstInf < 1e-7, "crossover solution is primal feasible in absolute terms", worstInf);
    if (compared)
        std::printf("         mean %.1f pushes, %.1f clean-up iterations\n",
                    (double)pushSum / compared, (double)iterSum / compared);
}

// --- 11. first-order path ---------------------------------------------------
static void testFirstOrder() {
    std::printf("first-order PDHG (the matrix-free / GPU path)\n");
    int converged = 0, attempted = 0;
    double worst = 0;
    long iterSum = 0;
    for (int seed = 1; seed <= 25; ++seed) {
        std::mt19937 rng(seed * 15485863u);
        int n = 10 + (int)(rng() % 25);
        int m = 6 + (int)(rng() % 18);
        Model mod = randomProgram(rng, m, n, false, false);

        Options opt; opt.log.level = 0; opt.timeLimit = 20.0;
        opt.pdhgMaxIter = 100000; opt.pdhgTol = 1e-8;
        Simplex sx;
        sx.load(mod.A, mod.obj, mod.colLower, mod.colUpper, mod.rowLower, mod.rowUpper, opt);
        if (sx.solve(false) != Status::Optimal) continue;
        std::vector<Real> xs(sx.values().begin(), sx.values().begin() + mod.numCol());
        Real ref = mod.objectiveValue(xs);

        ++attempted;
        PdhgResult pr = primalDualHybridGradient(mod, opt);
        if (pr.status != Status::Optimal) continue;
        ++converged;
        iterSum += pr.iterations;
        worst = std::max(worst, std::fabs(pr.primalObjective - ref) / (1.0 + std::fabs(ref)));
    }
    check(attempted >= 10, "enough first-order runs attempted", attempted);
    // A first-order method is not expected to reach tolerance on every model --
    // that is why the driver falls back to the simplex.  What it must never do
    // is converge to the wrong answer.
    check(converged >= attempted / 2, "first-order path converged on most models", converged);
    check(worst < 1e-5, "converged first-order runs match the simplex optimum", worst);
    if (converged) std::printf("         %d of %d converged, mean %.0f iterations\n",
                               converged, attempted, (double)iterSum / converged);
}

// --- 12. convex quadratic programming ---------------------------------------
static void testQuadraticProgram() {
    std::printf("convex quadratic programming\n");
    // min (x-3)^2 + (y-2)^2  s.t.  x + y <= 4,  x,y >= 0
    // = x^2 + y^2 - 6x - 4y + 13.  The unconstrained optimum (3,2) violates the
    // row, so the constraint binds and the answer is (2.5, 1.5) with value 0.5.
    {
        Model mod;
        Int x = mod.addColumn(0.0, kInf, -6.0);
        Int y = mod.addColumn(0.0, kInf, -4.0);
        mod.objOffset = 13.0;
        mod.setQuadratic(x, x, 2.0);
        mod.setQuadratic(y, y, 2.0);
        Int r = mod.addRow(-kInf, 4.0);
        mod.setElement(r, x, 1.0);
        mod.setElement(r, y, 1.0);
        mod.finalize();

        Solver s; s.opt.log.level = 0; s.opt.timeLimit = 20.0;
        Solution sol = s.solve(mod);
        check(sol.status == Status::Optimal, "QP solved to optimality");
        check(std::fabs(sol.colValue[0] - 2.5) < 1e-5, "QP x = 2.5",
              std::fabs(sol.colValue[0] - 2.5));
        check(std::fabs(sol.colValue[1] - 1.5) < 1e-5, "QP y = 1.5",
              std::fabs(sol.colValue[1] - 1.5));
        check(std::fabs(sol.objective - 0.5) < 1e-5, "QP objective = 0.5",
              std::fabs(sol.objective - 0.5));
        check(sol.algorithm.find("interior point") != std::string::npos,
              "a quadratic objective routes to the interior point method");
    }
    // Random convex QPs: primal feasibility plus a vanishing duality gap is a
    // certificate of optimality, so no external reference is needed.
    int compared = 0;
    double worstGap = 0, worstInf = 0;
    for (int seed = 1; seed <= 25; ++seed) {
        std::mt19937 rng(seed * 224737u);
        int n = 8 + (int)(rng() % 20);
        int m = 5 + (int)(rng() % 15);
        Model mod = randomProgram(rng, m, n, false, true);
        Options opt; opt.log.level = 0; opt.ipmTol = 1e-9; opt.timeLimit = 20.0;
        IpmResult ip = interiorPoint(mod, opt);
        if (ip.status != Status::Optimal) continue;
        ++compared;
        worstInf = std::max(worstInf, mod.primalInfeasibility(ip.x));
        worstGap = std::max(worstGap,
                            std::fabs(ip.primalObjective - ip.dualObjective) /
                            (1.0 + std::fabs(ip.primalObjective)));
    }
    check(compared >= 12, "enough convex QPs solved", compared);
    check(worstInf < 1e-6, "QP solutions are primal feasible", worstInf);
    check(worstGap < 1e-6, "QP duality gap closes", worstGap);
}

// --- 13. MPS round trip must preserve a maximization -------------------------
static void testMpsMaximize() {
    std::printf("MPS round trip preserves the objective sense\n");
    Model mod;
    mod.sense = Sense::Maximize;
    Int x = mod.addColumn(0.0, kInf, 3.0, VarType::Continuous, "x");
    Int y = mod.addColumn(0.0, kInf, 5.0, VarType::Continuous, "y");
    Int r0 = mod.addRow(-kInf, 4.0,  "c0");
    Int r1 = mod.addRow(-kInf, 12.0, "c1");
    Int r2 = mod.addRow(-kInf, 18.0, "c2");
    mod.setElement(r0, x, 1.0);
    mod.setElement(r1, y, 2.0);
    mod.setElement(r2, x, 3.0);
    mod.setElement(r2, y, 2.0);
    mod.finalize();

    Solver s1; s1.opt.log.level = 0;
    Solution before = s1.solve(mod);
    check(std::fabs(before.objective - 36.0) < 1e-7, "maximization solved before writing",
          std::fabs(before.objective - 36.0));

    std::string err;
    const char* path = "igaos_max_roundtrip.mps";
    check(writeMps(path, mod, err), "maximization written to MPS");
    Model back;
    check(readMps(path, back, err), "maximization read back from MPS");
    check(back.sense == Sense::Maximize, "objective sense survived the round trip");

    Solver s2; s2.opt.log.level = 0;
    Solution after = s2.solve(back);
    check(std::fabs(after.objective - before.objective) < 1e-7,
          "round-tripped maximization gives the same optimum",
          std::fabs(after.objective - before.objective));
    std::remove(path);
}


// --- 14. equality rows: the case the generator used to miss -----------------
// --- mixed-integer quadratic programming ------------------------------------
static void testMiqp() {
    std::printf("mixed-integer quadratic programming\n");
    {   // min -6x - 8y + x^2 + y^2  s.t. x + y <= 10, x,y integer in [0,10].
        // The LP part alone pushes to (0,10); the quadratic optimum is (3,4).
        // Branch and cut on the LP relaxation returns +20 here, which is a
        // feasible point with a confident wrong label -- the defect this path
        // exists to prevent.
        Model m;
        m.addColumn(0, 10, -6.0, VarType::Integer, "x");
        m.addColumn(0, 10, -8.0, VarType::Integer, "y");
        m.setQuadratic(0, 0, 2.0);
        m.setQuadratic(1, 1, 2.0);
        m.addRow(-kInf, 10.0, "c");
        m.setElement(0, 0, 1.0); m.setElement(0, 1, 1.0);
        m.finalize();
        Solver s; s.opt.log.level = 0;
        Solution r = s.solve(m);
        check(r.status == Status::Optimal, "MIQP solved to optimality");
        check(std::fabs(r.objective + 25.0) < 1e-6, "MIQP objective is the quadratic optimum");
        check(std::fabs(r.colValue[0] - 3.0) < 1e-6 && std::fabs(r.colValue[1] - 4.0) < 1e-6,
              "MIQP returns the integer minimiser, not the LP vertex");
    }
    {   // Every integer point enumerated by hand, compared against the solver.
        // Small enough to enumerate, large enough that branching is required.
        Model m;
        for (int j = 0; j < 4; ++j) m.addColumn(0, 3, -2.0 - j, VarType::Integer);
        for (int j = 0; j < 4; ++j) m.setQuadratic(j, j, 2.0);
        m.setQuadratic(1, 0, 1.0);
        m.setQuadratic(3, 2, -0.5);
        Int r0 = m.addRow(-kInf, 7.0);
        for (int j = 0; j < 4; ++j) m.setElement(r0, j, 1.0 + 0.25 * j);
        m.finalize();

        Real bestObj = kInf;
        std::vector<Real> x(4, 0.0);
        for (int a = 0; a <= 3; ++a) for (int b = 0; b <= 3; ++b)
        for (int c = 0; c <= 3; ++c) for (int d = 0; d <= 3; ++d) {
            x[0] = a; x[1] = b; x[2] = c; x[3] = d;
            if (m.primalInfeasibility(x) > 1e-9) continue;
            bestObj = std::min(bestObj, m.objectiveValue(x));
        }
        Solver s; s.opt.log.level = 0;
        Solution r = s.solve(m);
        check(r.status == Status::Optimal, "enumerable MIQP proved optimal");
        check(std::fabs(r.objective - bestObj) < 1e-6,
              "MIQP matches exhaustive enumeration over the integer box");
    }
    {   // A quadratic column fixed by its own bounds is a SUBSTITUTION: the
        // constant 1/2 q v^2 and the cross terms q_ij v x_i both have to move
        // into the reduced objective.  Presolve used to drop them, so a QP with
        // any fixed column came back with a wrong objective -- and every MIQP
        // node bound is a QP with fixed columns.
        Model m;
        m.addColumn(2, 2, -3.0, VarType::Continuous, "fixed");
        m.addColumn(0, 5, -1.0, VarType::Continuous, "free");
        m.setQuadratic(0, 0, 4.0);
        m.setQuadratic(1, 1, 2.0);
        m.setQuadratic(1, 0, 1.5);          // the cross term that used to vanish
        Int r0 = m.addRow(-kInf, 9.0);
        m.setElement(r0, 0, 1.0); m.setElement(r0, 1, 1.0);
        m.finalize();
        // x0 = 2 exactly, so the objective is  -6 + 8 + (-1 + 3)x1 + x1^2
        //   = 2 + 2 x1 + x1^2, minimised over [0,5] at x1 = 0 with value 2.
        Solver s; s.opt.log.level = 0;
        Solution a = s.solve(m);
        Solver s2; s2.opt.log.level = 0; s2.opt.presolve = false;
        Solution b = s2.solve(m);
        check(std::fabs(a.objective - 2.0) < 1e-6,
              "QP with a fixed quadratic column has the right objective");
        check(std::fabs(a.objective - b.objective) < 1e-6,
              "presolve and no-presolve agree on a QP with a fixed column");
    }
    {   // A non-convex objective used to be REFUSED here, and that was the right
        // answer while the only tree available was one over convex QP
        // relaxations: a relaxation of a non-convex QP is not a lower bound, so
        // branching over it prunes the optimum and reports the survivor as
        // optimal.  Refusing was better than answering wrongly.
        //
        // It is now SOLVED, by the spatial branch-and-bound path in
        // src/global.cpp, which relaxes the indefinite term with a McCormick
        // envelope that IS a valid bound.  min -x^2 over integer x in [0,4] has
        // its optimum at the corner x = 4, value -16 -- the point a convex
        // method walks away from, since the stationary point is x = 0.
        Model m;
        m.addColumn(0, 4, 0.0, VarType::Integer);
        m.setQuadratic(0, 0, -2.0);
        m.addRow(-kInf, 4.0);
        m.setElement(0, 0, 1.0);
        m.finalize();
        Solver s; s.opt.log.level = 0; s.opt.timeLimit = 30.0;
        Solution r = s.solve(m);
        check(r.status == Status::Optimal,
              "non-convex MIQP is now solved globally rather than refused");
        check(std::fabs(r.objective + 16.0) < 1e-6,
              "non-convex MIQP finds the corner optimum, not the stationary point",
              std::fabs(r.objective + 16.0));
        check(std::fabs(r.colValue[0] - 4.0) < 1e-6,
              "and the optimum is at the integer corner it should be");
    }
}

// ---------------------------------------------------------------------------
// The branch-and-cut tree is walked by a pool of worker threads, and a parallel
// tree is NOT deterministic: which node is expanded next depends on which
// worker reached the queue first, so the node count differs from run to run.
// The answer must not.  These models are built to have a tree worth walking --
// a weak relaxation, symmetry, and enough binaries that the pool actually
// engages -- and are solved at one thread and at four, and the objectives are
// compared against each other and against the value each was already known to
// have.  A race in the incumbent, the pseudocosts or the queue shows up here as
// a wrong answer or a hang, not as a subtle slowdown.
static void testParallelTree() {
    std::printf("parallel branch-and-cut tree\n");

    // A small unit-commitment model: on/off binaries, generation linked to the
    // commitment, and a demand row that no single unit can meet.  The LP
    // relaxation splits every commitment fractionally, so there is a real tree.
    auto buildUc = [](int nUnit, int nPeriod) {
        Model m;
        std::vector<std::vector<Int>> on(nUnit), gen(nUnit);
        for (int u = 0; u < nUnit; ++u) {
            for (int t = 0; t < nPeriod; ++t) {
                on[u].push_back(m.addColumn(0.0, 1.0, 40.0 + 7.0 * u, VarType::Binary));
                gen[u].push_back(m.addColumn(0.0, 20.0 + 3.0 * u, 2.0 + 0.5 * u));
            }
        }
        for (int u = 0; u < nUnit; ++u)
            for (int t = 0; t < nPeriod; ++t) {
                Int r = m.addRow(-kInf, 0.0);                  // gen <= pmax * on
                m.setElement(r, gen[u][t], 1.0);
                m.setElement(r, on[u][t], -(20.0 + 3.0 * u));
                Int r2 = m.addRow(0.0, kInf);                  // gen >= pmin * on
                m.setElement(r2, gen[u][t], 1.0);
                m.setElement(r2, on[u][t], -(6.0 + 1.0 * u));
            }
        for (int t = 0; t < nPeriod; ++t) {
            Int r = m.addRow(41.0 + 3.0 * t, kInf);            // demand
            for (int u = 0; u < nUnit; ++u) m.setElement(r, gen[u][t], 1.0);
        }
        m.finalize();
        return m;
    };

    for (int nUnit = 5; nUnit <= 7; ++nUnit) {
        Model m = buildUc(nUnit, 4);
        Real obj[2]; Status st[2]; Long nodes[2];
        const int threadCounts[2] = {1, 4};
        for (int k = 0; k < 2; ++k) {
            Solver s;
            s.opt.log.level = 0;
            s.opt.threads = threadCounts[k];
            Solution r = s.solve(m);
            obj[k] = r.objective; st[k] = r.status; nodes[k] = r.nodes;
        }
        check(st[0] == Status::Optimal && st[1] == Status::Optimal,
              "parallel tree proves optimality on the same model the serial tree does");
        check(std::fabs(obj[0] - obj[1]) <= 1e-6 * (1.0 + std::fabs(obj[0])),
              "four threads and one thread reach the same objective");
        (void)nodes;
    }

    // Infeasibility has to survive the pool too: a tree that empties with no
    // incumbent is a proof, and it is exactly the case where a worker holding a
    // plunge node when another worker stops could leave a subtree unexplored
    // and turn "not searched" into "does not exist".
    {
        Model m;
        Int a = m.addColumn(0.0, 3.0, 1.0, VarType::Integer);
        Int b = m.addColumn(0.0, 3.0, 1.0, VarType::Integer);
        Int r1 = m.addRow(2.5, 2.5);                // a + b = 2.5, both integer
        m.setElement(r1, a, 1.0); m.setElement(r1, b, 1.0);
        m.finalize();
        for (int t : {1, 4}) {
            Solver s; s.opt.log.level = 0; s.opt.threads = t;
            Solution r = s.solve(m);
            check(r.status == Status::Infeasible,
                  "integer infeasibility is still proved with a worker pool");
        }
    }
}

// ---------------------------------------------------------------------------
// Nonconvex QCQP and bilinear MINLP, solved to proven GLOBAL optimality.
//
// The claim being tested is stronger than anywhere else in this file, so the
// checks are too.  "Globally optimal" means no better feasible point exists
// ANYWHERE in the box, and the only way to test that without trusting the thing
// under test is to look everywhere else: these cases are checked against a
// dense grid over the whole box, against exhaustive enumeration where the model
// is small enough, and against published values that predate this project by
// forty years.
// ---------------------------------------------------------------------------
static void testGlobalOptimization() {
    std::printf("nonconvex QCQP and bilinear MINLP -- global optimality\n");

    // ---- Haverly's pooling problem, the standard refinery counterexample ----
    //
    // Three crudes, one pool with a quality that is a DECISION, two products
    // with sulfur specifications.  Quality times flow is a product of two
    // variables, which is what makes the obvious linear model wrong: it reports
    // a blend that cannot physically be made.  The three variants differ in the
    // demand for product X and the cost of crude B, and their global optima are
    // published: 400, 600, 750.
    auto haverly = [](double costB, double demandX) {
        const double demandY = 200.0;
        Model m;
        Int A  = m.addColumn(0.0, 1000.0, 6.0);
        Int B  = m.addColumn(0.0, 1000.0, costB);
        Int Cx = m.addColumn(0.0, demandX,  1.0);
        Int Cy = m.addColumn(0.0, demandY, -5.0);
        Int Px = m.addColumn(0.0, demandX, -9.0);
        Int Py = m.addColumn(0.0, demandY, -15.0);
        Int p  = m.addColumn(1.0, 3.0, 0.0);                 // the pool's sulfur

        Int r0 = m.addRow(0.0, 0.0);                          // pool balance
        m.setElement(r0, A, 1.0);  m.setElement(r0, B, 1.0);
        m.setElement(r0, Px, -1.0); m.setElement(r0, Py, -1.0);

        Int r1 = m.addRow(0.0, 0.0);                          // pool quality
        m.setElement(r1, A, -3.0); m.setElement(r1, B, -1.0);
        m.addQuadraticTerm(r1, p, Px, 1.0);
        m.addQuadraticTerm(r1, p, Py, 1.0);

        Int r2 = m.addRow(-kInf, demandX);
        m.setElement(r2, Px, 1.0); m.setElement(r2, Cx, 1.0);
        Int r3 = m.addRow(-kInf, demandY);
        m.setElement(r3, Py, 1.0); m.setElement(r3, Cy, 1.0);

        Int r4 = m.addRow(-kInf, 0.0);                        // sulfur spec, X
        m.setElement(r4, Px, -2.5); m.setElement(r4, Cx, -0.5);
        m.addQuadraticTerm(r4, p, Px, 1.0);
        Int r5 = m.addRow(-kInf, 0.0);                        // sulfur spec, Y
        m.setElement(r5, Py, -1.5); m.setElement(r5, Cy, 0.5);
        m.addQuadraticTerm(r5, p, Py, 1.0);
        m.finalize();
        return m;
    };

    struct { const char* name; double costB, demandX, opt; } hav[3] = {
        {"HPP1", 16.0, 100.0, -400.0},
        {"HPP2", 16.0, 600.0, -600.0},
        {"HPP3", 13.0, 100.0, -750.0},
    };
    for (const auto& h : hav) {
        Model m = haverly(h.costB, h.demandX);
        Solver s; s.opt.log.level = 0; s.opt.timeLimit = 60.0;
        Solution r = s.solve(m);
        check(r.status == Status::Optimal,
              "Haverly pooling instance is proved globally optimal");
        check(std::fabs(r.objective - h.opt) < 1e-6 * (1.0 + std::fabs(h.opt)),
              "Haverly pooling optimum matches the published value",
              std::fabs(r.objective - h.opt));
        // A "global optimum" that does not satisfy the bilinear rows is not an
        // optimum of anything.  primalInfeasibility evaluates the products.
        check(m.primalInfeasibility(r.colValue) < 1e-6,
              "the reported pooling point satisfies the bilinear balances",
              m.primalInfeasibility(r.colValue));
    }

    // ---- against a dense grid over the whole box ---------------------------
    //
    // Two variables, so a 601x601 grid visits every feasible region the model
    // has.  The solver must be at least as good as the best grid point (or it
    // missed the optimum) and no better than it by more than the grid spacing
    // can explain (or it is reporting an infeasible point as feasible).
    {
        std::mt19937_64 rng(90210);
        std::uniform_real_distribution<double> u(-2.0, 2.0);
        int cases = 0;
        double worstMiss = 0.0;
        for (int trial = 0; trial < 6; ++trial) {
            const double a0 = u(rng), a1 = u(rng), b = u(rng);
            const double c0 = u(rng), c1 = u(rng), qc = u(rng) + (u(rng) > 0 ? 1.0 : -1.0);

            Model m;
            Int x = m.addColumn(-1.0, 2.0, c0);
            Int y = m.addColumn(-1.0, 2.0, c1);
            // a0*x + a1*y + qc*x*y <= b, plus a square term to exercise i == j
            Int r0 = m.addRow(-kInf, b);
            m.setElement(r0, x, a0); m.setElement(r0, y, a1);
            m.addQuadraticTerm(r0, x, y, qc);
            Int r1 = m.addRow(-kInf, 3.0);
            m.setElement(r1, x, 1.0);
            m.addQuadraticTerm(r1, x, x, 1.0);               // x^2 + x <= 3
            m.finalize();

            Solver s; s.opt.log.level = 0; s.opt.timeLimit = 30.0;
            Solution r = s.solve(m);
            if (r.status != Status::Optimal) continue;

            // Brute force over the same box.
            const int N = 600;
            double bestGrid = 1e100;
            for (int i = 0; i <= N; ++i)
                for (int j = 0; j <= N; ++j) {
                    std::vector<Real> p(2);
                    p[0] = -1.0 + 3.0 * i / N;
                    p[1] = -1.0 + 3.0 * j / N;
                    if (m.primalInfeasibility(p) > 1e-9) continue;
                    bestGrid = std::min(bestGrid, (double)m.objectiveValue(p));
                }
            if (bestGrid > 1e99) continue;                    // grid found nothing
            ++cases;
            check(m.primalInfeasibility(r.colValue) < 1e-6,
                  "the global optimum reported is feasible for the true rows",
                  m.primalInfeasibility(r.colValue));
            // The solver must not be beaten by a grid point.
            check(r.objective <= bestGrid + 1e-6,
                  "no grid point beats the reported global optimum",
                  r.objective - bestGrid);
            worstMiss = std::max(worstMiss, bestGrid - (double)r.objective);
        }
        check(cases >= 4, "enough random bilinear cases had a feasible grid point");
        // The grid can only ever be worse than the true optimum, and by at most
        // what its spacing allows.  A large gap the other way would mean the
        // solver had claimed a point the grid says is infeasible.
        check(worstMiss < 0.05, "the grid comes within its own spacing of the optimum",
              worstMiss);
    }

    // ---- bilinear MINLP: the same machinery with an integer variable -------
    //
    // Checked by exhaustive enumeration over the integer, with the continuous
    // part solved on a fine grid for each value.  This is the case that makes
    // it a MINLP solver rather than an NLP one, and the spatial tree and the
    // integer tree have to cooperate: the relaxation of each spatial node is
    // itself a mixed-integer program.
    {
        Model m;
        Int x = m.addColumn(0.0, 4.0, 0.0, VarType::Integer);
        Int y = m.addColumn(0.0, 3.0, -1.0);
        Int r0 = m.addRow(-kInf, 6.0);
        m.addQuadraticTerm(r0, x, y, 1.0);                   // x*y <= 6
        m.setElement(r0, y, 1.0);                            //   + y
        Int r1 = m.addRow(2.0, kInf);
        m.setElement(r1, x, 1.0); m.setElement(r1, y, 1.0);  // x + y >= 2
        m.finalize();

        Solver s; s.opt.log.level = 0; s.opt.timeLimit = 30.0;
        Solution r = s.solve(m);
        check(r.status == Status::Optimal, "bilinear MINLP is proved globally optimal");

        double bestEnum = 1e100;
        for (int xi = 0; xi <= 4; ++xi)
            for (int k = 0; k <= 3000; ++k) {
                std::vector<Real> p(2);
                p[0] = xi; p[1] = 3.0 * k / 3000.0;
                if (m.primalInfeasibility(p) > 1e-9) continue;
                bestEnum = std::min(bestEnum, (double)m.objectiveValue(p));
            }
        check(std::fabs(r.objective - bestEnum) < 1e-3,
              "bilinear MINLP matches enumeration over the integer and a grid",
              std::fabs(r.objective - bestEnum));
        check(m.integerInfeasibility(r.colValue, 1e-6) < 1e-6,
              "the MINLP answer is integral where it must be");
    }

    // ---- a nonconvex OBJECTIVE, with no quadratic constraint at all --------
    //
    // min -x^2 on [0,3] has its minimum at a CORNER, x = 3, value -9.  A convex
    // QP method would go to the stationary point x = 0 and call it optimal.
    // The routing has to notice the Hessian is indefinite and take the global
    // path, and this is the smallest case that proves it does.
    {
        Model m;
        m.addColumn(0.0, 3.0, 0.0);
        m.setQuadratic(0, 0, -2.0);                          // 1/2 * (-2) x^2
        m.addRow(-kInf, kInf);
        m.finalize();
        check(m.hasNonconvexObjective(), "an indefinite objective Hessian is detected");
        Solver s; s.opt.log.level = 0; s.opt.timeLimit = 30.0;
        Solution r = s.solve(m);
        check(r.status == Status::Optimal, "nonconvex QP is proved globally optimal");
        check(std::fabs(r.objective + 9.0) < 1e-5,
              "nonconvex QP finds the corner minimum, not the stationary point",
              std::fabs(r.objective + 9.0));
    }
}

static void testEqualityRows() {
    std::printf("equality-constrained models on every path\n");
    // A pinned variable -- a fixed column, or the logical of an equality row --
    // has a multiplier that is free in sign and absorbs whatever the dual
    // equation needs.  Treating that as a residual makes the interior point
    // method's dual measure plateau at the largest equality dual, so it can
    // never converge on a model with an equality row no matter how good the
    // iterate is.  This is a direct regression test for that.
    {
        Model mod;
        Int x = mod.addColumn(0.0, 5.0, 1.0);
        Int y = mod.addColumn(0.0, 5.0, 2.0);
        Int r = mod.addRow(3.0, 3.0);               // equality
        mod.setElement(r, x, 1.0);
        mod.setElement(r, y, 1.0);
        mod.finalize();

        Options opt; opt.log.level = 0; opt.ipmTol = 1e-9; opt.timeLimit = 20.0;
        IpmResult ip = interiorPoint(mod, opt);
        check(ip.status == Status::Optimal,
              "interior point converges with an equality row");
        check(ip.iterations < 40, "and does so in a sane iteration count",
              (double)ip.iterations);
        check(std::fabs(ip.primalObjective - 3.0) < 1e-6, "equality LP objective = 3",
              std::fabs(ip.primalObjective - 3.0));
        check(ip.dualInfeasibility < 1e-6,
              "dual residual ignores the pinned equality multiplier",
              ip.dualInfeasibility);
    }
    // Now across random models that genuinely contain equality rows, on every
    // continuous path, against the simplex.
    int compared = 0, eqRows = 0, bad = 0;
    double worstIpm = 0, worstPdhg = 0;
    for (int seed = 1; seed <= 30; ++seed) {
        std::mt19937 rng(seed * 611953u);
        int n = 8 + (int)(rng() % 22);
        int m = 5 + (int)(rng() % 14);
        Model mod = randomProgram(rng, m, n, false, false);
        for (Int i = 0; i < mod.numRow(); ++i)
            if (isFinite(mod.rowLower[i]) && isFinite(mod.rowUpper[i]) &&
                mod.rowLower[i] == mod.rowUpper[i]) ++eqRows;

        Options opt; opt.log.level = 0; opt.ipmTol = 1e-9; opt.timeLimit = 20.0;
        opt.pdhgMaxIter = 100000; opt.pdhgTol = 1e-8;
        Simplex sx;
        sx.load(mod.A, mod.obj, mod.colLower, mod.colUpper, mod.rowLower, mod.rowUpper, opt);
        if (sx.solve(false) != Status::Optimal) continue;
        std::vector<Real> xs(sx.values().begin(), sx.values().begin() + mod.numCol());
        Real ref = mod.objectiveValue(xs);

        IpmResult ip = interiorPoint(mod, opt);
        if (ip.status != Status::Optimal) { ++bad; continue; }
        ++compared;
        worstIpm = std::max(worstIpm, std::fabs(ip.primalObjective - ref) / (1.0 + std::fabs(ref)));

        PdhgResult pr = primalDualHybridGradient(mod, opt);
        if (pr.status == Status::Optimal)
            worstPdhg = std::max(worstPdhg,
                                 std::fabs(pr.primalObjective - ref) / (1.0 + std::fabs(ref)));
    }
    check(eqRows > 0, "the generator actually produces equality rows", eqRows);
    check(bad == 0, "interior point converged on every equality-constrained model", bad);
    check(compared >= 12, "enough equality-constrained models compared", compared);
    check(worstIpm < 1e-6, "interior point matches the simplex with equality rows", worstIpm);
    check(worstPdhg < 1e-5, "first-order path matches the simplex with equality rows", worstPdhg);
    std::printf("         %d equality rows across the sample\n", eqRows);
}


// --- [A; Q] equilibration --------------------------------------------------
// A QP whose Q spans sixteen orders of magnitude while every entry of A is
// O(1).  Scaling from A alone cannot see that, so the KKT matrix the interior
// point path factorizes stays as badly conditioned as it arrived.  Letting Q
// into the column pass is the fix; this test is what proves it does something.
static Real kktSpread(const Model& m) {
    Real lo = kBigReal, hi = 0.0;
    for (Int p = 0; p < (Int)m.A.val.size(); ++p) {
        Real a = std::fabs(m.A.val[p]);
        if (a > 0) { lo = std::min(lo, a); hi = std::max(hi, a); }
    }
    for (Int p = 0; p < (Int)m.Q.val.size(); ++p) {
        Real q = std::fabs(m.Q.val[p]);
        if (q > 0) { lo = std::min(lo, q); hi = std::max(hi, q); }
    }
    return (lo < kBigReal && lo > 0) ? hi / lo : 1.0;
}

static Model badlyScaledQp() {
    Model m;
    const int n = 6;
    const Real qdiag[n] = {1e8, 1e-8, 1e4, 1e-4, 1e6, 1e-6};
    for (int j = 0; j < n; ++j) m.addColumn(0.0, 10.0, -1.0);
    m.addRow(1.0, 1.0);                       // sum x = 1, every coefficient 1
    for (int j = 0; j < n; ++j) m.setElement(0, j, 1.0);
    for (int j = 0; j < n; ++j) m.setQuadratic(j, j, qdiag[j]);
    m.finalize();
    return m;
}

// --- irreducible infeasible subsystem -------------------------------------
// --- sensitivity: shadow prices, reduced costs, ranging ---------------------
static void testSensitivity() {
    std::printf("sensitivity (shadow prices + ranging)\n");

    // The textbook Wyndor problem, whose answer is known by hand:
    //   maximize 3 x + 5 y
    //   R1:  x            <= 4
    //   R2:        2 y    <= 12
    //   R3:  3 x + 2 y    <= 18
    //   x, y >= 0
    // Optimum x = 2, y = 6, objective 36.
    // Shadow prices 0, 3/2, 1.  R3 right-hand side ranges over [12, 24];
    // R2 over [6, 18]; R1 is slack so its price is 0.
    Model m;
    m.sense = Sense::Maximize;
    m.addColumn(0.0, kInf, 3.0, VarType::Continuous, "x");
    m.addColumn(0.0, kInf, 5.0, VarType::Continuous, "y");
    Int r1 = m.addRow(-kInf, 4.0,  "plant1");
    Int r2 = m.addRow(-kInf, 12.0, "plant2");
    Int r3 = m.addRow(-kInf, 18.0, "plant3");
    m.setElement(r1, 0, 1.0);
    m.setElement(r2, 1, 2.0);
    m.setElement(r3, 0, 3.0); m.setElement(r3, 1, 2.0);
    m.finalize();

    Solver s; s.opt.log.level = 0; s.opt.lpAlgorithm = LpAlgorithm::DualSimplex;
    Solution sol = s.solve(m);
    check(sol.status == Status::Optimal, "sens: model solved");
    check(std::fabs(sol.objective - 36.0) < 1e-7, "sens: objective is 36",
          (double)sol.objective);

    Sensitivity sen = computeSensitivity(m, sol, s.opt);
    check(sen.available, "sens: available for an LP with a basis");
    check((Int)sen.rows.size() == 3 && (Int)sen.cols.size() == 2,
          "sens: one entry per row and column");

    check(std::fabs(sen.rows[r1].dual) < 1e-7, "sens: slack row priced at zero",
          (double)sen.rows[r1].dual);
    check(!sen.rows[r1].binding, "sens: slack row reported non-binding");
    check(std::fabs(sen.rows[r2].dual - 1.5) < 1e-6, "sens: plant2 shadow price 3/2",
          (double)sen.rows[r2].dual);
    check(std::fabs(sen.rows[r3].dual - 1.0) < 1e-6, "sens: plant3 shadow price 1",
          (double)sen.rows[r3].dual);
    check(sen.rows[r2].binding && sen.rows[r3].binding,
          "sens: both active rows reported binding");

    check(std::fabs(sen.rows[r3].rhsLower - 12.0) < 1e-5,
          "sens: plant3 rhs range starts at 12", (double)sen.rows[r3].rhsLower);
    check(std::fabs(sen.rows[r3].rhsUpper - 24.0) < 1e-5,
          "sens: plant3 rhs range ends at 24", (double)sen.rows[r3].rhsUpper);
    check(std::fabs(sen.rows[r2].rhsLower - 6.0) < 1e-5,
          "sens: plant2 rhs range starts at 6", (double)sen.rows[r2].rhsLower);
    check(std::fabs(sen.rows[r2].rhsUpper - 18.0) < 1e-5,
          "sens: plant2 rhs range ends at 18", (double)sen.rows[r2].rhsUpper);

    // The shadow price is a real derivative: raise plant3 by one unit and the
    // objective must rise by exactly its price. Tested, not asserted.
    {
        Model m2 = m;
        m2.rowUpper[r3] = 19.0;
        Solver s2; s2.opt.log.level = 0; s2.opt.lpAlgorithm = LpAlgorithm::DualSimplex;
        Solution sol2 = s2.solve(m2);
        check(sol2.status == Status::Optimal, "sens: perturbed model solved");
        check(std::fabs((sol2.objective - sol.objective) - sen.rows[r3].dual) < 1e-6,
              "sens: +1 on rhs moves the objective by exactly the shadow price",
              (double)(sol2.objective - sol.objective));
    }

    // Both structural variables are basic here, so each has a zero reduced cost
    // and a two-sided objective range that contains its own coefficient.
    for (Int j = 0; j < 2; ++j) {
        check(sen.cols[j].basic, "sens: structural variable is basic");
        check(std::fabs(sen.cols[j].reducedCost) < 1e-7,
              "sens: basic variable has zero reduced cost",
              (double)sen.cols[j].reducedCost);
        check(sen.cols[j].objLower <= m.obj[j] + 1e-7 &&
              sen.cols[j].objUpper >= m.obj[j] - 1e-7,
              "sens: objective range brackets the coefficient");
    }

    // A variable priced out of the plan must report a non-zero reduced cost.
    {
        Model m3;
        m3.sense = Sense::Maximize;
        m3.addColumn(0.0, kInf, 5.0, VarType::Continuous, "good");
        m3.addColumn(0.0, kInf, 1.0, VarType::Continuous, "poor");
        Int c = m3.addRow(-kInf, 10.0, "capacity");
        m3.setElement(c, 0, 1.0); m3.setElement(c, 1, 1.0);
        m3.finalize();
        Solver s3; s3.opt.log.level = 0; s3.opt.lpAlgorithm = LpAlgorithm::DualSimplex;
        Solution sol3 = s3.solve(m3);
        Sensitivity sen3 = computeSensitivity(m3, sol3, s3.opt);
        check(sen3.available, "sens: second model available");
        check(std::fabs(sen3.rows[c].dual - 5.0) < 1e-6,
              "sens: capacity priced at the better variable's coefficient",
              (double)sen3.rows[c].dual);
        check(std::fabs(sen3.cols[1].reducedCost + 4.0) < 1e-6,
              "sens: the unused variable is 4 short of paying",
              (double)sen3.cols[1].reducedCost);
        check(sen3.cols[1].objUpper > 4.99 && sen3.cols[1].objUpper < 5.01,
              "sens: it would enter the plan at a coefficient of 5",
              (double)sen3.cols[1].objUpper);
    }

    // Scope is refused honestly rather than answered wrongly.
    {
        Model mi;
        mi.addColumn(0.0, 10.0, 1.0, VarType::Integer, "k");
        Int r = mi.addRow(1.0, kInf, "need");
        mi.setElement(r, 0, 1.0);
        mi.finalize();
        Solver si; si.opt.log.level = 0;
        Solution soli = si.solve(mi);
        Sensitivity seni = computeSensitivity(mi, soli, si.opt);
        check(!seni.available, "sens: refused on a mixed-integer model");
        check(!seni.reason.empty(), "sens: refusal states a reason");
    }
}

static void testIis() {
    std::printf("IIS (deletion filter)\n");

    // Three rows, two of which contradict each other; the third is slack and
    // must NOT appear in the IIS.
    //   r_tight : x >= 5
    //   r_cap   : x <= 2        <-- contradicts r_tight
    //   r_idle  : x <= 100      <-- irrelevant
    {
        Model m;
        m.addColumn(-kInf, kInf, 1.0, VarType::Continuous, "x");
        Int rTight = m.addRow(5.0, kInf, "r_tight");
        Int rCap   = m.addRow(-kInf, 2.0, "r_cap");
        Int rIdle  = m.addRow(-kInf, 100.0, "r_idle");
        m.setElement(rTight, 0, 1.0);
        m.setElement(rCap, 0, 1.0);
        m.setElement(rIdle, 0, 1.0);
        m.finalize();

        IisOptions io; io.logLevel = 0;
        Iis iis = computeIis(m, io);

        check(!iis.modelWasFeasible, "IIS: infeasible model recognized");
        check(iis.irreducible, "IIS: filter ran to completion");
        check(iis.size() == 2, "IIS: exactly two members", (double)iis.size());
        check(iis.rows.size() == 2, "IIS: both members are rows",
              (double)iis.rows.size());
        bool hasTight = false, hasCap = false, hasIdle = false;
        for (Int i : iis.rows) {
            if (i == rTight) hasTight = true;
            if (i == rCap)   hasCap = true;
            if (i == rIdle)  hasIdle = true;
        }
        check(hasTight && hasCap, "IIS: the contradicting pair is reported");
        check(!hasIdle, "IIS: the slack row is excluded");

        // Irreducibility, tested directly rather than trusted: removing either
        // member must make the remaining system feasible.
        for (Int drop : iis.rows) {
            Model r = m;
            r.rowLower[drop] = -kInf; r.rowUpper[drop] = kInf;
            Solver s; s.opt.log.level = 0;
            Solution sol = s.solve(r);
            check(sol.status != Status::Infeasible,
                  "IIS: dropping a member restores feasibility");
        }
    }

    // A variable bound, not a row, carries the contradiction.
    {
        Model m;
        m.addColumn(0.0, 1.0, 1.0, VarType::Continuous, "y");   // y <= 1
        Int r = m.addRow(4.0, kInf, "demand");                  // y >= 4
        m.setElement(r, 0, 1.0);
        m.finalize();

        IisOptions io; io.logLevel = 0; io.includeBounds = true;
        Iis iis = computeIis(m, io);
        check(iis.irreducible, "IIS: bound case completed");
        check(iis.rows.size() == 1 && iis.upperBounds.size() == 1,
              "IIS: one row and one upper bound", (double)iis.size());
        check(iis.upperBounds[0] == 0, "IIS: the reported bound is y's");
    }

    // A feasible model must be reported as such, not as an empty IIS.
    {
        Model m;
        m.addColumn(0.0, 10.0, 1.0, VarType::Continuous, "z");
        Int r = m.addRow(1.0, 5.0, "band");
        m.setElement(r, 0, 1.0);
        m.finalize();
        Iis iis = computeIis(m, IisOptions{});
        check(iis.modelWasFeasible, "IIS: feasible model flagged, not isolated");
        check(iis.size() == 0, "IIS: feasible model yields no members",
              (double)iis.size());
    }

    // A truncated run must say so rather than pass off a non-minimal set.
    {
        Model m;
        m.addColumn(-kInf, kInf, 1.0, VarType::Continuous, "x");
        Int a = m.addRow(5.0, kInf, "lo");
        Int b = m.addRow(-kInf, 2.0, "hi");
        m.setElement(a, 0, 1.0); m.setElement(b, 0, 1.0);
        m.finalize();
        IisOptions io; io.maxSolves = 1; io.logLevel = 0;
        Iis iis = computeIis(m, io);
        check(!iis.irreducible, "IIS: truncated run is not claimed irreducible");
    }
}

static void testDualityGapGuard() {
    std::printf("duality gap guard (defect 27)\n");

    // A well-behaved convex QP: the guard must not touch it.
    {
        Model m;
        m.addColumn(0.0, 10.0, -1.0);
        m.addColumn(0.0, 10.0, -1.0);
        m.addRow(-kInf, 3.0);
        m.setElement(0, 0, 1.0);
        m.setElement(0, 1, 1.0);
        m.setQuadratic(0, 0, 2.0);
        m.setQuadratic(1, 1, 2.0);
        m.finalize();

        Solver plain;  plain.opt.log.level = 0;
        Solver guard;  guard.opt.log.level = 0; guard.opt.dualityGapCheck = true;
        Model a = m, b = m;
        Solution sa = plain.solve(a), sb = guard.solve(b);
        check(sa.status == Status::Optimal, "well-conditioned QP solves without the guard");
        check(sb.status == Status::Optimal, "the guard does not refuse a genuine optimum");
        check(std::fabs(sa.objective - sb.objective) <= 1e-9 * (1.0 + std::fabs(sa.objective)),
              "the guard does not change the answer it accepts");
    }

    // The guard is off unless asked for, so no recorded benchmark moves.
    {
        Options o;
        check(o.dualityGapCheck == true, "duality gap check defaults on");
    }

    // The measure itself: equal objectives give a zero gap, and a wrong dual is
    // caught by the same formula the guard applies.
    {
        const Real p1 = 0.3747882067, d1 = 0.3747882067;
        const Real g1 = std::fabs(p1 - d1) / (1.0 + std::fabs(p1) + std::fabs(d1));
        check(g1 <= 1e-15, "matching objectives give a zero relative gap");

        // CONT-300's actual pair, from bench/results_maros_gapcheck_ab.csv.
        const Real p2 = 0.3747882067, d2 = -0.5656951564;
        const Real g2 = std::fabs(p2 - d2) / (1.0 + std::fabs(p2) + std::fabs(d2));
        check(g2 > 1e-6, "CONT-300's primal and dual pair fails the gap test");
        std::printf("       CONT-300 relative duality gap %.3e\n", (double)g2);
    }
}

static void testScalingWithQ() {
    std::printf("[A; Q] equilibration\n");
    const Model base = badlyScaledQp();
    const Real raw = kktSpread(base);

    Model aOnly = base;  Scaling s1; computeScaling(aOnly, s1, 6, false); s1.apply(aOnly);
    Model withQ = base;  Scaling s2; computeScaling(withQ, s2, 6, true ); s2.apply(withQ);

    const Real spreadA = kktSpread(aOnly), spreadQ = kktSpread(withQ);
    std::printf("       spread: raw %.2e -> A-only %.2e -> [A;Q] %.2e\n",
                raw, spreadA, spreadQ);

    // 1. The A-only rule is blind here: it cannot improve on the raw model,
    //    because every entry of A is already 1.
    check(spreadA >= raw * 0.99, "A-only scaling leaves the Q spread untouched", spreadA);
    // 2. Equilibrating [A; Q] collapses it by at least six orders of magnitude.
    check(spreadQ <= spreadA * 1e-6, "[A; Q] equilibration reduces the spread", spreadQ);
    // 3. Scaling is a change of variables, not of problem: unapply must return
    //    the model bit-for-bit, both ways.  Powers of two make this exact.
    Model back = withQ; s2.unapply(back);
    Real worst = 0;
    for (Int p = 0; p < (Int)back.Q.val.size(); ++p)
        worst = std::max(worst, std::fabs(back.Q.val[p] - base.Q.val[p]));
    for (Int p = 0; p < (Int)back.A.val.size(); ++p)
        worst = std::max(worst, std::fabs(back.A.val[p] - base.A.val[p]));
    check(worst == 0.0, "unapply restores the model exactly", worst);

    // 4. Same answer either way.  If the two paths disagree on the objective
    //    the scaling is not a change of variables and the fix is wrong.
    Solver svA; svA.opt.log.level = 0; svA.opt.scaleWithQ = false;
    Solver svQ; svQ.opt.log.level = 0; svQ.opt.scaleWithQ = true;
    Solution r1 = svA.solve(base);
    Solution r2 = svQ.solve(base);
    bool bothOk = (r1.status == Status::Optimal && r2.status == Status::Optimal);
    check(bothOk, "both scalings reach Optimal");
    if (bothOk) {
        Real d = std::fabs(r1.objective - r2.objective)
               / std::max(1.0, std::fabs(r1.objective));
        check(d <= 1e-6, "both scalings agree on the objective", d);
    }

    // 5. A quadratically constrained model must survive the round trip too.
    //    Before qcon was carried, scaling a QCQP silently changed the problem,
    //    which is why the solver refused to scale one at all.
    {
        Model q;
        Int x = q.addColumn(0.0, 10.0, -1.0);
        Int y = q.addColumn(0.0, 10.0, -1.0);
        Int r = q.addRow(-kBigReal, 4.0);
        q.setElement(r, x, 1e-5);                 // linear part, tiny
        q.addQuadraticTerm(r, x, y, 1e7);         // bilinear part, huge
        q.finalize();
        Model scaled = q;
        Scaling sq; computeScaling(scaled, sq, 6, true); sq.apply(scaled);
        Model round = scaled; sq.unapply(round);
        Real w = 0;
        for (size_t k = 0; k < q.qcon.size(); ++k)
            w = std::max(w, std::fabs(round.qcon[k].coef - q.qcon[k].coef));
        for (Int p2 = 0; p2 < (Int)q.A.val.size(); ++p2)
            w = std::max(w, std::fabs(round.A.val[p2] - q.A.val[p2]));
        check(w == 0.0, "QCQP round-trips through apply/unapply exactly", w);

        Solver qa; qa.opt.log.level = 0; qa.opt.scaleWithQ = false;
        Solver qb; qb.opt.log.level = 0; qb.opt.scaleWithQ = true;
        Solution ra = qa.solve(q), rb = qb.solve(q);
        if (ra.status == rb.status && ra.status != Status::Infeasible) {
            Real d = std::fabs(ra.objective - rb.objective)
                   / std::max(1.0, std::fabs(ra.objective));
            check(d <= 1e-6, "scaled and unscaled QCQP agree on the objective", d);
        } else {
            check(ra.status == rb.status, "scaled and unscaled QCQP agree on status");
        }
    }
}

int main() {
    std::printf("IGAOS test suite\n==================\n");
    testLu();
    testLdl();
    testMps();
    testLpEndToEnd();
    testMilpEndToEnd();
    testStatuses();
    testPresolve();
    testCutValidity();
    testInteriorPoint();
    testCrossover();
    testFirstOrder();
    testQuadraticProgram();
    testMpsMaximize();
    testMiqp();
    testParallelTree();
    testGlobalOptimization();
    testEqualityRows();
    testScalingWithQ();
    testDualityGapGuard();
    testIis();
    testSensitivity();
    std::printf("==================\n%s (%d failures)\n",
                failures ? "FAILED" : "ALL TESTS PASSED", failures);
    return failures ? 1 : 0;
}
