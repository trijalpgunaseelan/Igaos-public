// Cross-check every continuous path against the simplex on the same models:
// interior point + crossover, first-order PDHG + crossover, and the simplex
// itself.  Whatever the path, the answer must be the same optimum.
#include "igaos/solver.hpp"
#include <random>
#include <cstdio>

using namespace igaos;

static double gPdhgTol = 1e-8;

static Model randomLp(std::mt19937& rng, int m, int n) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Model mod;
    for (int j = 0; j < n; ++j) {
        double lo = 0.0;
        double up = (U(rng) < 0.2) ? kInf : 1.0 + 20.0 * U(rng);
        mod.addColumn(lo, up, -10.0 + 20.0 * U(rng));
    }
    for (int i = 0; i < m; ++i) {
        double r = U(rng);
        double lo, up;
        if (r < 0.45)      { lo = -kInf; up = 0.0; }
        else if (r < 0.80) { lo = 0.0;  up = kInf; }
        else               { lo = 0.0;  up = 0.0; }
        Int row = mod.addRow(lo, up);
        double act = 0;
        int len = 2 + (int)(U(rng) * std::min(n - 2, 8));
        for (int k = 0; k < len; ++k) {
            int j = (int)(U(rng) * n); if (j >= n) j = n - 1;
            double a = std::round((-5.0 + 10.0 * U(rng)) * 4.0) / 4.0;
            if (a == 0.0) a = 1.0;
            mod.setElement(row, j, a);
            act += a;
        }
        double shift = act + (-2.0 + 4.0 * U(rng));
        if (isFinite(mod.rowUpper[row]))  mod.rowUpper[row] = shift + 3.0;
        if (!isNegInf(mod.rowLower[row])) mod.rowLower[row] = shift - 3.0;
        if (mod.rowLower[row] == mod.rowUpper[row]) { mod.rowLower[row] = shift; mod.rowUpper[row] = shift; }
    }
    mod.finalize();
    return mod;
}

struct Outcome { Status st; double obj; double pinf; long long iters; std::string alg; };

static Outcome run(const Model& mod, LpAlgorithm alg, bool crossover) {
    Solver s;
    s.opt.log.level = 0;
    s.opt.lpAlgorithm = alg;
    s.opt.crossover = crossover;
    s.opt.presolve = true;
    s.opt.timeLimit = 30.0;
    s.opt.pdhgMaxIter = 200000;
    s.opt.pdhgTol = gPdhgTol;
    Solution sol = s.solve(mod);
    return {sol.status, sol.objective, mod.primalInfeasibility(sol.colValue),
            sol.iterations, sol.algorithm};
}

int main(int argc, char** argv) {
    int cases = argc > 1 ? atoi(argv[1]) : 60;
    if (argc > 2) gPdhgTol = atof(argv[2]);
    int fails = 0, compared = 0;
    double worstIpm = 0, worstPdhg = 0;
    int pdhgFellBack = 0;

    for (int seed = 1; seed <= cases; ++seed) {
        std::mt19937 rng(seed * 15485863u);
        int n = 10 + (int)(rng() % 40);
        int m = 6 + (int)(rng() % 30);
        Model mod = randomLp(rng, m, n);

        Outcome sx = run(mod, LpAlgorithm::DualSimplex, true);
        if (sx.st != Status::Optimal) continue;
        ++compared;

        Outcome ip = run(mod, LpAlgorithm::InteriorPoint, true);
        Outcome pd = run(mod, LpAlgorithm::PDHG, true);

        auto check = [&](const char* tag, const Outcome& o, double& worst) {
            if (o.st != Status::Optimal) {
                std::printf("NOTOPT[%s] seed=%d %s\n", tag, seed, statusName(o.st));
                ++fails; return;
            }
            double rel = std::fabs(o.obj - sx.obj) / (1.0 + std::fabs(sx.obj));
            worst = std::max(worst, rel);
            if (rel > 1e-7) {
                std::printf("MISMATCH[%s] seed=%d simplex=%.12g %s=%.12g rel=%.3g\n",
                            tag, seed, sx.obj, tag, o.obj, rel);
                ++fails;
            }
            if (o.pinf > 1e-6) {
                std::printf("PRIMALINF[%s] seed=%d %.3g\n", tag, seed, o.pinf);
                ++fails;
            }
        };
        check("ipm", ip, worstIpm);
        check("pdhg", pd, worstPdhg);
        if (pd.alg.find("simplex") != std::string::npos) ++pdhgFellBack;
    }
    std::printf("\ncompared %d models; failures %d\n"
                "  worst relative objective error: interior point %.3g, first-order %.3g\n"
                "  first-order runs that fell back to the simplex: %d\n",
                compared, fails, worstIpm, worstPdhg, pdhgFellBack);
    return fails ? 1 : 0;
}
