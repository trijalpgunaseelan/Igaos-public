// Fuzz harness: solve the same random MILP with cut separation on and off and
// insist the proven optimal objectives agree.  An invalid cut removes feasible
// integer points, so it shows up here as a strictly worse objective on the
// cutting run -- the failure mode that matters most and the one that is
// invisible on a single instance.
#include "igaos/solver.hpp"
#include <random>
#include <cstdio>
#include <cmath>

using namespace igaos;

static Model randomMilp(std::mt19937& rng, int m, int n, int seed) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    Model mod;
    mod.name = "fuzz" + std::to_string(seed);
    int nint = 0;
    for (int j = 0; j < n; ++j) {
        double r = U(rng);
        VarType t = VarType::Continuous;
        if (r < 0.55) { t = VarType::Binary; ++nint; }
        else if (r < 0.75) { t = VarType::Integer; ++nint; }
        double lo = 0.0, up;
        if (t == VarType::Binary) up = 1.0;
        else if (t == VarType::Integer) up = std::floor(1.0 + 8.0 * U(rng));
        else up = 1.0 + 20.0 * U(rng);
        double cost = -10.0 + 20.0 * U(rng);
        mod.addColumn(lo, up, cost, t);
    }
    if (nint == 0) mod.colType[0] = VarType::Binary, mod.colUpper[0] = 1.0;

    // Row kind is decided up front and applied AFTER the activity is known.
    // Setting both bounds to zero to mean "equality" and then overwriting each
    // bound separately silently produced a ranged row every time, which left
    // the separators unfuzzed against equality rows -- exactly the rows whose
    // logical variable is fixed and handled by a special case.
    enum RowKind { LE = 0, GE = 1, EQ = 2, RANGE = 3 };
    for (int i = 0; i < m; ++i) {
        double r = U(rng);
        RowKind kind = (r < 0.45) ? LE : (r < 0.72) ? GE : (r < 0.87) ? EQ : RANGE;
        Int row = mod.addRow(-kInf, kInf);
        double act = 0;
        int len = 2 + (int)(U(rng) * std::min(n - 2, 6));
        for (int k = 0; k < len; ++k) {
            int j = (int)(U(rng) * n);
            if (j >= n) j = n - 1;
            double a = std::round((-5.0 + 10.0 * U(rng)) * 4.0) / 4.0;
            if (a == 0.0) a = 1.0;
            mod.setElement(row, j, a);
            act += a * 0.5 * (mod.colLower[j] + mod.colUpper[j]);
        }
        // Put the right-hand side near the midpoint activity so the row bites
        // without making the model trivially infeasible.
        double shift = act + (-2.0 + 4.0 * U(rng));
        switch (kind) {
            case LE:    mod.rowLower[row] = -kInf;       mod.rowUpper[row] = shift + 2.0; break;
            case GE:    mod.rowLower[row] = shift - 2.0; mod.rowUpper[row] = kInf;        break;
            case EQ:    mod.rowLower[row] = shift;       mod.rowUpper[row] = shift;       break;
            case RANGE: mod.rowLower[row] = shift - 2.0; mod.rowUpper[row] = shift + 2.0; break;
        }
    }
    mod.finalize();
    return mod;
}

static int gWhich = 7;   // bitmask: 1 = gomory, 2 = cover, 4 = mir

static Solution runRef(const Model& mod, const std::vector<Real>* ref, SolveReport& rep) {
    // presolve and scaling off, so the model the separators see is exactly the
    // model the reference point lives in.
    Solver s;
    s.opt.log.level = 0;
    s.opt.cuts = true;
    s.opt.cutGomory = (gWhich & 1) != 0;
    s.opt.cutCover  = (gWhich & 2) != 0;
    s.opt.cutMir    = (gWhich & 4) != 0;
    s.opt.presolve = false;
    s.opt.scaling  = false;
    s.opt.cutReference = ref;
    s.opt.timeLimit = 20.0;
    Solution sol = s.solve(mod);
    rep = s.report;
    return sol;
}

static Solution run(const Model& mod, bool cuts, bool presolve) {
    Solver s;
    s.opt.log.level = 0;
    s.opt.cuts = cuts;
    s.opt.cutGomory = (gWhich & 1) != 0;
    s.opt.cutCover  = (gWhich & 2) != 0;
    s.opt.cutMir    = (gWhich & 4) != 0;
    s.opt.presolve = presolve;
    s.opt.timeLimit = 20.0;
    s.opt.nodeLimit = 200000;
    return s.solve(mod);
}

static void verbose(int seed) {
    std::mt19937 rng(seed * 7919u);
    int n = 6 + (int)(rng() % 20);
    int m = 4 + (int)(rng() % 16);
    Model mod = randomMilp(rng, m, n, seed);
    std::printf("seed %d: %d rows x %d cols, %d integer\n", seed, (int)mod.numRow(),
                (int)mod.numCol(), (int)mod.numInt());
    Solution ref;
    { Solver s0; s0.opt.log.level = 0; s0.opt.cuts = false; s0.opt.presolve = false;
      s0.opt.scaling = false; s0.opt.timeLimit = 60.0; ref = s0.solve(mod); }
    {
        Solver s1; s1.opt.log.level = 0; s1.opt.cuts = true; s1.opt.presolve = false;
        s1.opt.scaling = false; s1.opt.timeLimit = 60.0;
        s1.opt.cutGomory = (gWhich & 1) != 0; s1.opt.cutCover = (gWhich & 2) != 0;
        s1.opt.cutMir = (gWhich & 4) != 0;
        s1.opt.cutReference = &ref.colValue;
        Solution v = s1.solve(mod);
        std::printf("  [verified] obj=%.12g  cutsApplied=%d  cutsInvalid=%d worst=%.3g from=%s\n",
                    v.objective, (int)s1.report.cutsApplied, (int)s1.report.cutsInvalid,
                    s1.report.cutInvalidWorst,
                    s1.report.cutInvalidOrigin ? s1.report.cutInvalidOrigin : "-");
    }
    for (int cuts = 0; cuts < 2; ++cuts) {
        for (int sc = 0; sc < 2; ++sc) {
            Solver s;
            s.opt.log.level = 0;
            s.opt.cuts = cuts != 0;
            s.opt.cutGomory = (gWhich & 1) != 0;
            s.opt.cutCover  = (gWhich & 2) != 0;
            s.opt.cutMir    = (gWhich & 4) != 0;
            s.opt.presolve = false;
            s.opt.scaling  = sc != 0;
            s.opt.timeLimit = 60.0;
            Solution sol = s.solve(mod);
            std::printf("  cuts=%d scaling=%d  status=%-10s obj=%.12g  bound=%.12g gap=%.3g "
                        "nodes=%lld iters=%lld  cutsApplied=%d rounds=%d  pinf=%.3g iinf=%.3g\n",
                        cuts, sc, statusName(sol.status), sol.objective, sol.bestBound,
                        sol.mipGap, (long long)sol.nodes, (long long)sol.iterations,
                        (int)s.report.cutsApplied, (int)s.report.cutRounds,
                        mod.primalInfeasibility(sol.colValue),
                        mod.integerInfeasibility(sol.colValue, 1e-6));
        }
    }
}

int main(int argc, char** argv) {
    int cases = (argc > 1) ? atoi(argv[1]) : 300;
    if (argc > 2) gWhich = atoi(argv[2]);
    if (argc > 3) { verbose(atoi(argv[3])); return 0; }
    int failures = 0, compared = 0, improvedRoot = 0;
    for (int seed = 1; seed <= cases; ++seed) {
        std::mt19937 rng(seed * 7919u);
        int n = 6 + (int)(rng() % 20);
        int m = 4 + (int)(rng() % 16);
        Model mod = randomMilp(rng, m, n, seed);

        for (int pre = 0; pre < 2; ++pre) {
            Solution a = run(mod, false, pre == 1);
            Solution b = run(mod, true,  pre == 1);

            if (a.status == Status::Optimal && b.status == Status::Optimal) {
                ++compared;
                double tol = 1e-6 * (1.0 + std::fabs(a.objective));
                if (std::fabs(a.objective - b.objective) > tol) {
                    std::printf("MISMATCH seed=%d presolve=%d  nocuts=%.12g  cuts=%.12g  (diff %.3g)\n",
                                seed, pre, a.objective, b.objective, a.objective - b.objective);
                    ++failures;
                }
            } else if (a.status == Status::Infeasible && b.status != Status::Infeasible) {
                std::printf("STATUS seed=%d presolve=%d nocuts=INFEASIBLE cuts=%s\n",
                            seed, pre, statusName(b.status));
                ++failures;
            } else if (b.status == Status::Infeasible && a.status != Status::Infeasible) {
                std::printf("STATUS seed=%d presolve=%d nocuts=%s cuts=INFEASIBLE  <-- cut cut off the optimum\n",
                            seed, pre, statusName(a.status));
                ++failures;
            }
            // Feasibility of BOTH answers, so a pre-existing tolerance issue in
            // the solver is not misread as a broken cut.
            auto check = [&](const char* tag, const Solution& sol) {
                if (sol.status != Status::Optimal && sol.status != Status::Feasible) return;
                Real pinf = mod.primalInfeasibility(sol.colValue);
                Real iinf = mod.integerInfeasibility(sol.colValue, 1e-6);
                if (pinf > 1e-6) {
                    std::printf("PRIMALINF[%s] seed=%d presolve=%d  %.3g\n", tag, seed, pre, pinf);
                    ++failures;
                }
                if (iinf > 1e-6) {
                    std::printf("INTINF[%s] seed=%d presolve=%d  %.3g\n", tag, seed, pre, iinf);
                    ++failures;
                }
            };
            check("nocuts", a);
            check("cuts", b);

            // Direct validity assertion: replay separation with the proven
            // optimum as a verification point.  Any cut that excludes it is a
            // derivation bug, reported here instead of showing up as a wrong
            // objective several thousand nodes later.
            if (pre == 0 && a.status == Status::Optimal) {
                SolveReport rep;
                Solution v = runRef(mod, &a.colValue, rep);
                if (rep.cutsInvalid > 0) {
                    std::printf("INVALIDCUT seed=%d  %d cuts excluded the optimum "
                                "(worst %.3g, separator %s)\n",
                                seed, (int)rep.cutsInvalid, rep.cutInvalidWorst,
                                rep.cutInvalidOrigin ? rep.cutInvalidOrigin : "?");
                    ++failures;
                }
                if (v.status == Status::Optimal &&
                    std::fabs(v.objective - a.objective) > 1e-6 * (1.0 + std::fabs(a.objective))) {
                    std::printf("REFMISMATCH seed=%d  nocuts=%.12g  cuts(nopre,noscale)=%.12g\n",
                                seed, a.objective, v.objective);
                    ++failures;
                }
            }
        }
    }
    std::printf("\ncompared %d optimal pairs, %d failures\n", compared, failures);
    return failures ? 1 : 0;
}
