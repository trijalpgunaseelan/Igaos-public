// pooling_bench.cpp : the global solver against multistart local search, on the
// refinery pooling family.
//
//     python3 bench/generate_pooling.py benchmarks/pooling --seeds 3
//     ./build/igaos_pooling_bench benchmarks/pooling
//
// ===========================================================================
//  WHY THIS IS THE CHECK WORTH RUNNING
// ===========================================================================
//  Global optimality is the strongest claim in this repository, and on these
//  instances there is no published value to check it against and no grid small
//  enough to enumerate.  What there IS, is a second solver in the same source
//  tree that shares no code with the first and works on a completely different
//  principle:
//
//     src/global.cpp   spatial branch and bound over McCormick RELAXATIONS.
//                      Never evaluates the nonlinear functions to make a
//                      decision; works entirely with linear outer approximations
//                      and proves a bound.
//
//     src/nlp.cpp      a primal-dual interior point method on the TRUE
//                      nonconvex problem.  Never builds a relaxation; follows
//                      derivatives downhill to a KKT point and has no idea
//                      whether a better one exists.
//
//  Run the second from many random starts and two things must hold, or one of
//  them is wrong:
//
//     1. NO local solution may beat the global one.  A single counterexample
//        would mean the branch and bound pruned a subtree containing the
//        optimum -- the exact failure mode a nonconvex solver has to be trusted
//        not to have.
//     2. The best local solution should REACH the global one on most instances.
//        If it never did, the "global" answer would more likely be a bug than a
//        discovery.
//
//  The spread between the best and the worst local solution is reported too,
//  because that number is the argument for having built the global solver at
//  all: it is what a local method would have cost a planner who trusted it.
// ===========================================================================
#include "igaos/solver.hpp"
#include "igaos/mps.hpp"
#include "igaos/expr.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <random>
#include <string>
#include <vector>

using namespace igaos;

// Translate a quadratically constrained Model into the expression-graph form
// the NLP solver takes.  Same problem, different representation -- and the
// translation is mechanical enough to read at a glance, which matters because a
// cross-check is only worth anything if both sides really are the same problem.
static NlpProblem toNlp(const Model& m) {
    NlpProblem p;
    for (Int j = 0; j < m.numCol(); ++j)
        p.addVariable(m.colLower[j], m.colUpper[j]);
    ExprTape& t = p.tape;

    Int obj = t.constant(m.objOffset);
    for (Int j = 0; j < m.numCol(); ++j)
        if (m.obj[j] != 0.0)
            obj = t.add(obj, t.mul(t.constant(m.obj[j]), t.variable(j)));
    for (Int j = 0; j < m.Q.ncol; ++j)
        for (Int q = m.Q.colPtr[j]; q < m.Q.colPtr[j + 1]; ++q) {
            const Int i = m.Q.rowIdx[q];
            const Real c = (i == j) ? 0.5 * m.Q.val[q] : m.Q.val[q];
            obj = t.add(obj, t.mul(t.constant(c), t.mul(t.variable(i), t.variable(j))));
        }
    p.objective = obj;

    std::vector<std::vector<std::pair<Int, Real>>> rowLin((size_t)m.numRow());
    for (Int j = 0; j < m.numCol(); ++j)
        for (Int q = m.A.colPtr[j]; q < m.A.colPtr[j + 1]; ++q)
            rowLin[(size_t)m.A.rowIdx[q]].emplace_back(j, m.A.val[q]);
    std::vector<std::vector<const Model::QuadTerm*>> rowQ((size_t)m.numRow());
    for (const Model::QuadTerm& qt : m.qcon) rowQ[(size_t)qt.row].push_back(&qt);

    for (Int i = 0; i < m.numRow(); ++i) {
        if (rowLin[(size_t)i].empty() && rowQ[(size_t)i].empty()) continue;
        if (!isFinite(m.rowLower[i]) && !isFinite(m.rowUpper[i])) continue;
        Int e = t.constant(0.0);
        for (const auto& lt : rowLin[(size_t)i])
            e = t.add(e, t.mul(t.constant(lt.second), t.variable(lt.first)));
        for (const Model::QuadTerm* qt : rowQ[(size_t)i])
            e = t.add(e, t.mul(t.constant(qt->coef),
                               t.mul(t.variable(qt->i), t.variable(qt->j))));
        p.addConstraint(e, m.rowLower[i], m.rowUpper[i]);
    }
    return p;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "benchmarks/pooling";
    std::vector<std::string> files;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) {
            std::string nm = e->d_name;
            if (nm.size() > 4 && nm.substr(nm.size() - 4) == ".mps") files.push_back(nm);
        }
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        std::printf("no .mps files in %s\n", dir.c_str());
        std::printf("run:  python3 bench/generate_pooling.py %s --seeds 3\n", dir.c_str());
        return 1;
    }

    std::printf("Refinery pooling: proven global optimum vs multistart local search\n"
                "==================================================================\n"
                "Two solvers, no shared code: spatial branch and bound over McCormick\n"
                "relaxations, against an interior point method on the true nonconvex\n"
                "problem started from %d random points.\n\n", 40);
    std::printf("%-12s %6s %6s %6s | %15s %6s | %15s %15s %5s\n",
                "instance", "rows", "cols", "bilin", "global", "nodes",
                "best local", "worst local", "hits");
    std::printf("--------------------------------------------------------------------"
                "---------------------------------\n");

    int beaten = 0, reached = 0, total = 0;
    double worstSpread = 0.0;
    std::mt19937_64 rng(4242);

    for (const std::string& fn : files) {
        Model m;
        std::string err;
        if (!readMps(dir + "/" + fn, m, err)) { std::printf("  %-12s  %s\n", fn.c_str(), err.c_str()); continue; }

        Solver s;
        s.opt.log.level = 0;
        s.opt.timeLimit = 120.0;
        Solution g = s.solve(m);
        if (g.status != Status::Optimal) {
            std::printf("%-12s  global solve did not prove optimality (%s); skipped\n",
                        fn.c_str(), statusName(g.status));
            continue;
        }

        NlpProblem p = toNlp(m);
        double bestLocal = 1e100, worstLocal = -1e100;
        int hits = 0, converged = 0;
        for (int trial = 0; trial < 40; ++trial) {
            p.start.assign((size_t)m.numCol(), 0.0);
            for (Int j = 0; j < m.numCol(); ++j) {
                const Real lo = m.colLower[j];
                const Real up = isFinite(m.colUpper[j]) ? m.colUpper[j] : lo + 100.0;
                const double u = (double)(rng() % 100000) / 100000.0;
                p.start[(size_t)j] = lo + u * (up - lo);
            }
            NlpOptions o; o.tolerance = 1e-8; o.maxIter = 200; o.timeLimit = 5.0;
            NlpResult r = solveNlp(p, o);
            if (r.status != Status::Optimal || r.primalInf > 1e-6) continue;
            ++converged;
            bestLocal = std::min(bestLocal, (double)r.objective);
            worstLocal = std::max(worstLocal, (double)r.objective);
            if (std::fabs(r.objective - g.objective) < 1e-5 * (1.0 + std::fabs(g.objective)))
                ++hits;
        }
        ++total;
        if (converged == 0) {
            std::printf("%-12s %6d %6d %6d | %15.6f %6lld | %15s\n",
                        fn.c_str(), (int)m.numRow(), (int)m.numCol(), (int)m.qcon.size(),
                        (double)g.objective, (long long)g.nodes, "no local converged");
            continue;
        }
        // The one that must never happen.
        const bool beat = bestLocal < (double)g.objective - 1e-6 * (1.0 + std::fabs(bestLocal));
        if (beat) ++beaten;
        if (hits > 0) ++reached;
        worstSpread = std::max(worstSpread,
                               (worstLocal - bestLocal) / (1.0 + std::fabs(bestLocal)));

        std::printf("%-12s %6d %6d %6d | %15.6f %6lld | %15.6f %15.6f %4d%s\n",
                    fn.c_str(), (int)m.numRow(), (int)m.numCol(), (int)m.qcon.size(),
                    (double)g.objective, (long long)g.nodes,
                    bestLocal, worstLocal, hits,
                    beat ? "   <== LOCAL BEAT GLOBAL" : "");
    }

    std::printf("\n  instances .................................. %d\n", total);
    std::printf("  a local solution BEAT the global optimum ... %d   (must be 0)\n", beaten);
    std::printf("  multistart reached the global optimum ...... %d / %d\n", reached, total);
    std::printf("  worst spread between local solutions ....... %.1f%%\n", 100.0 * worstSpread);
    std::printf("\n  The last line is what the global solver is for: it is how far apart\n"
                "  two local answers to the same blending problem can be, and therefore\n"
                "  how wrong a planner following a local method could be without ever\n"
                "  seeing an error message.\n");
    return beaten == 0 ? 0 : 1;
}
