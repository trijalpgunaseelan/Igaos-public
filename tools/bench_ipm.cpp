// Compare the interior point method against the simplex on random LPs and QPs.
// The simplex is the reference: it is the path that is already validated against
// an independent solver, so any disagreement here is the new code's fault.
#include "igaos/solver.hpp"
#include "igaos/ipm.hpp"
#include "igaos/simplex.hpp"
#include <random>
#include <cstdio>

using namespace igaos;

static Model randomLp(std::mt19937& rng, int m, int n, bool quadratic, bool ranged) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Model mod;
    for (int j = 0; j < n; ++j) {
        double lo = 0.0;
        double up = (U(rng) < 0.25) ? kInf : 1.0 + 20.0 * U(rng);
        if (U(rng) < 0.10) lo = -5.0 * U(rng);
        mod.addColumn(lo, up, -10.0 + 20.0 * U(rng));
    }
    for (int i = 0; i < m; ++i) {
        double r = U(rng);
        double lo, up;
        if (r < 0.40)                 { lo = -kInf; up = 0.0; }
        else if (r < 0.70)            { lo = 0.0;  up = kInf; }
        else if (r < 0.85 && ranged)  { lo = 0.0;  up = 0.0; }   // becomes a range below
        else                          { lo = 0.0;  up = 0.0; }   // equality
        Int row = mod.addRow(lo, up);
        double act = 0;
        int len = 2 + (int)(U(rng) * std::min(n - 2, 8));
        for (int k = 0; k < len; ++k) {
            int j = (int)(U(rng) * n); if (j >= n) j = n - 1;
            double a = std::round((-5.0 + 10.0 * U(rng)) * 4.0) / 4.0;
            if (a == 0.0) a = 1.0;
            mod.setElement(row, j, a);
            act += a * 1.0;
        }
        double shift = act + (-2.0 + 4.0 * U(rng));
        if (isFinite(mod.rowUpper[row]))  mod.rowUpper[row] = shift + 3.0;
        if (!isNegInf(mod.rowLower[row])) mod.rowLower[row] = shift - 3.0;
        if (mod.rowLower[row] == mod.rowUpper[row]) {
            if (ranged && r >= 0.70 && r < 0.85) { mod.rowLower[row] = shift - 1.0; mod.rowUpper[row] = shift + 1.0; }
            else { mod.rowLower[row] = shift; mod.rowUpper[row] = shift; }
        }
    }
    if (quadratic) {
        // A diagonally dominant lower triangle is positive semidefinite by
        // construction, so the model stays a convex QP.
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

int main(int argc, char** argv) {
    int cases = argc > 1 ? atoi(argv[1]) : 120;
    int mode = argc > 2 ? atoi(argv[2]) : 0;   // 0 = LP, 1 = QP
    int fails = 0, compared = 0, notOptimal = 0;
    double worstRel = 0;
    long totIter = 0;

    for (int seed = 1; seed <= cases; ++seed) {
        std::mt19937 rng(seed * 104729u);
        int n = 8 + (int)(rng() % 40);
        int m = 5 + (int)(rng() % 30);
        Model mod = randomLp(rng, m, n, mode == 1, true);

        Options opt;
        opt.log.level = 0;
        opt.ipmTol = 1e-9;
        opt.ipmMaxIter = 200;
        opt.timeLimit = 30.0;

        IpmResult ip = interiorPoint(mod, opt);

        // reference: simplex on the same model (LP only -- the simplex path does
        // not solve QP, so QP is checked by optimality conditions instead)
        if (mode == 0) {
            Simplex sx;
            sx.load(mod.A, mod.obj, mod.colLower, mod.colUpper, mod.rowLower, mod.rowUpper, opt);
            Status st = sx.solve(false);
            if (st != Status::Optimal) continue;
            std::vector<Real> xs(sx.values().begin(), sx.values().begin() + mod.numCol());
            Real refObj = mod.objectiveValue(xs);

            if (ip.status != Status::Optimal) {
                std::printf("NOTOPT seed=%d  ipm=%s after %d iters (p=%.2e d=%.2e gap=%.2e)\n",
                            seed, statusName(ip.status), (int)ip.iterations,
                            ip.primalInfeasibility, ip.dualInfeasibility, ip.complementarityGap);
                ++notOptimal; ++fails;
                continue;
            }
            ++compared;
            totIter += ip.iterations;
            Real rel = std::fabs(ip.primalObjective - refObj) / (1.0 + std::fabs(refObj));
            worstRel = std::max(worstRel, rel);
            if (rel > 1e-6) {
                std::printf("MISMATCH seed=%d  simplex=%.12g  ipm=%.12g  rel=%.3g\n",
                            seed, refObj, ip.primalObjective, rel);
                ++fails;
            }
            Real pinf = mod.primalInfeasibility(ip.x);
            if (pinf > 1e-6) {
                std::printf("PRIMALINF seed=%d  %.3g\n", seed, pinf);
                ++fails;
            }
        } else {
            if (ip.status != Status::Optimal) {
                std::printf("NOTOPT(qp) seed=%d %s iters=%d p=%.2e d=%.2e gap=%.2e\n",
                            seed, statusName(ip.status), (int)ip.iterations,
                            ip.primalInfeasibility, ip.dualInfeasibility, ip.complementarityGap);
                ++notOptimal; ++fails;
                continue;
            }
            ++compared;
            totIter += ip.iterations;
            // Convex QP: a point satisfying primal feasibility and a vanishing
            // duality gap is optimal, so check exactly that.
            Real pinf = mod.primalInfeasibility(ip.x);
            Real gap = std::fabs(ip.primalObjective - ip.dualObjective) /
                       (1.0 + std::fabs(ip.primalObjective));
            worstRel = std::max(worstRel, gap);
            if (pinf > 1e-6 || gap > 1e-6) {
                std::printf("QPFAIL seed=%d  pinf=%.3g  gap=%.3g  (p=%.10g d=%.10g)\n",
                            seed, pinf, gap, ip.primalObjective, ip.dualObjective);
                ++fails;
            }
        }
    }
    std::printf("\n%s: compared %d, failures %d (not-optimal %d), worst relative error %.3g, "
                "mean iterations %.1f\n",
                mode ? "QP" : "LP", compared, fails, notOptimal, worstRel,
                compared ? (double)totIter / compared : 0.0);
    return fails ? 1 : 0;
}
