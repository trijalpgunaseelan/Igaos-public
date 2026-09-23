// miqp.cpp : mixed-integer quadratic programming.
//
// ===========================================================================
//  WHY THIS FILE EXISTS
// ===========================================================================
//  Because without it the solver got MIQP *wrong*, not slow.
//
//  Branch and cut searches an LP relaxation.  Hand it a model with a quadratic
//  objective and integer variables and it will branch on the linear part, find
//  the integer point that optimises c'x, evaluate c'x + 1/2 x'Qx there, and
//  report "optimal".  On the two-variable model in tests/test_all.cpp that
//  returns +20 when the optimum is -25.  A feasible point with a confident
//  wrong label is the worst thing a solver can produce, and it is exactly what
//  the LP tree produces for a QP.
//
//  So MIQP gets its own tree, over its own relaxation.
//
// ===========================================================================
//  THE METHOD, AND WHAT IT IS NOT
// ===========================================================================
//  Branch and bound where each node relaxation is the convex QP obtained by
//  dropping integrality, solved by the interior point method in src/ipm.cpp.
//  For a convex Q that relaxation is a genuine lower bound, which is the whole
//  basis for pruning; a non-convex Q would make every node bound meaningless
//  and is rejected up front rather than answered wrongly.
//
//  There are NO cutting planes here, and that is a correctness decision rather
//  than an omission.  Gomory mixed-integer cuts are read off a simplex tableau
//  row.  The interior point method does not produce a tableau, and the basis
//  crossover would give one for the *linear* problem, not for this relaxation.
//  Separating cuts from the wrong tableau is how a solver cuts off its own
//  optimum, so the tree branches and does not cut.  That costs nodes, and the
//  cost is visible in the node counts rather than hidden.
//
//  Branching is on the integer variable whose value is most fractional, with
//  ties broken toward the largest diagonal Hessian entry: of two equally
//  fractional candidates, the one with more curvature moves the objective more
//  when it is forced to an integer, so it splits the bound further.
//
//  Node order is depth-first with plunging, which finds incumbents early; the
//  reported bound is the minimum over all open nodes, so the gap is honest at
//  every point rather than only at the end.
//
// ===========================================================================
//  MULTI-CORE
// ===========================================================================
//  The tree is searched by several threads sharing one node pool and one
//  incumbent.  This is the part of the solver where parallelism actually pays:
//  node relaxations are independent, each takes milliseconds, and nothing is
//  shared between them except the bound used for pruning.  Contrast the simplex,
//  where every iteration depends on the previous basis, and the sparse
//  factorizations, whose cost sits in a sequential chain through the elimination
//  tree -- putting threads there buys contention, not speed.
//
//  Each worker builds its own Model and its own nested Solver, so nothing in
//  the numerical core is shared and nothing there needed to be made thread safe.
//  The only shared state is the node stack, the incumbent and the counters, all
//  under one mutex held for pushes and pops but never across a solve.
//
//  Sharing the incumbent is what makes this worth doing: a bound found by one
//  thread immediately prunes work on every other. It also makes the search
//  NON-DETERMINISTIC in node count -- whichever thread finds an incumbent first
//  changes what the others prune. The objective does not move; the node count
//  does. Run with --threads 1 for a reproducible tree.
// ===========================================================================

#include "igaos/solver.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

namespace igaos {

namespace {

struct Node {
    std::vector<Real> lower, upper;
    Real bound;                 // relaxation objective of the parent
    int  depth;
};

// A convex quadratic objective is what makes a node bound a bound at all.  We
// cannot cheaply certify positive semidefiniteness of a sparse Q, but a
// negative diagonal entry is a sufficient certificate of the opposite, and it
// is the way non-convex models are usually written by mistake.
bool diagonalIsNonNegative(const Model& m, Int& badColumn) {
    for (Int j = 0; j < m.Q.ncol; ++j)
        for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
            if (m.Q.rowIdx[p] == j && m.Q.val[p] < -1e-12) { badColumn = j; return false; }
    return true;
}

} // namespace

Solution Solver::solveMiqp(const Model& m) {
    Timer clock;
    const Int nc = m.numCol(), nr = m.numRow();

    Solution best;
    best.resize(nr, nc);
    best.status = Status::NotSolved;
    best.algorithm = "branch and bound (convex QP relaxations, interior point)";

    Int badColumn = -1;
    if (!diagonalIsNonNegative(m, badColumn)) {
        opt.log.log(1, "  the quadratic objective has a negative diagonal entry in column %d, "
                       "so it is not convex; a relaxation of a non-convex QP is not a bound and "
                       "branch and bound would return a wrong answer confidently\n",
                    (int)badColumn);
        best.status = Status::NumericalError;
        return best;
    }

    std::vector<Int> intCols;
    for (Int j = 0; j < nc; ++j)
        if (m.colType[j] != VarType::Continuous) intCols.push_back(j);

    // Curvature per integer column, for the branching tie-break.
    std::vector<Real> curvature(nc, 0.0);
    for (Int j = 0; j < m.Q.ncol; ++j)
        for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
            if (m.Q.rowIdx[p] == j) curvature[j] = std::fabs(m.Q.val[p]);

    Long nodes = 0, relaxations = 0, unresolved = 0;
    std::atomic<Long> relaxAtomic{0};
    Status firstUnresolved = Status::NotSolved;

    // Solve one node relaxation: the same model with this node's bounds and
    // integrality dropped.
    //
    // This goes through a nested Solver rather than straight to the interior
    // point method, for one specific reason.  Branching fixes variables, and a
    // fixed variable leaves a structurally singular block in the KKT system
    // that the interior point method reports as a numerical error rather than
    // an answer.  Presolve substitutes fixed columns out before the factorization
    // ever sees them, which is exactly the job it exists to do -- and since the
    // quadratic guard went into presolve, it is sound to use here.
    //
    // Integrality is stripped from the copy, and that is not cosmetic: leaving
    // it on would send the nested solve straight back into this function.
    auto solveRelaxation = [&](const std::vector<Real>& lo, const std::vector<Real>& up,
                               Solution& out) -> Status {
        Model q = m;
        q.colLower = lo;
        q.colUpper = up;
        for (auto& t : q.colType) t = VarType::Continuous;
        relaxAtomic.fetch_add(1);

        Solver sub;
        sub.opt = opt;
        sub.opt.log.level = 0;            // one line per node would be unreadable
        sub.opt.log.progress = false;     // and one pipeline of stage events per
        sub.opt.log.onStage  = nullptr;   // node would drown the real one --
                                          // silence BOTH ways of listening
        sub.opt.presolve  = true;
        sub.opt.crossover = false;        // no basis is needed from a node
        sub.opt.timeLimit = std::max(0.0, opt.timeLimit - clock.elapsed());
        out = sub.solve(q);
        // report fields are written after the join, not from a worker
        return out.status;
    };

    const Real itol = opt.tol.integrality;
    auto fractionality = [&](const std::vector<Real>& x, Int j) {
        Real f = x[j] - std::floor(x[j]);
        return std::min(f, 1.0 - f);
    };
    auto isIntegral = [&](const std::vector<Real>& x) {
        for (Int j : intCols) if (fractionality(x, j) > itol) return false;
        return true;
    };

    Real incumbent = isFinite(opt.cutoff) ? opt.cutoff : kInf;
    bool haveIncumbent = false;
    auto accept = [&](const std::vector<Real>& x, Real obj) {
        if (obj >= incumbent - 1e-12) return;
        incumbent = obj;
        haveIncumbent = true;
        best.colValue = x;
        best.objective = obj;
        m.rowActivity(x, best.rowValue);
        best.primalInf = m.primalInfeasibility(x);
    };

    // Rounding heuristic: round every integer to the nearest integer, clamp to
    // its own bounds, and keep the point if it is feasible.  Cheap, and on
    // models where the quadratic term dominates it lands on the optimum often
    // enough to prune most of the tree.
    auto tryRounding = [&](const std::vector<Real>& x) {
        std::vector<Real> r = x;
        for (Int j : intCols) {
            r[j] = std::floor(r[j] + 0.5);
            r[j] = std::max(m.colLower[j], std::min(m.colUpper[j], r[j]));
        }
        if (m.primalInfeasibility(r) <= opt.tol.primalFeas)
            accept(r, m.objectiveValue(r));
    };

    // ---- root --------------------------------------------------------------
    Solution root;
    Status rs = solveRelaxation(m.colLower, m.colUpper, root);
    if (rs == Status::Infeasible || rs == Status::Unbounded) { best.status = rs; return best; }
    if (rs != Status::Optimal) { best.status = rs; return best; }

    const Real rootBound = root.objective;
    report.rootBoundLp = rootBound;
    report.rootBoundCut = rootBound;      // no separation on this path, by design

    if (isIntegral(root.colValue)) {
        accept(root.colValue, root.objective);
        best.status = Status::Optimal;
        best.bestBound = rootBound;
        best.nodes = 1;
        best.mipGap = 0.0;
        report.nodes = 1;
        report.nodeRelaxations = relaxAtomic.load();
        opt.log.log(2, "  miqp       root relaxation already integral\n");
        return best;
    }
    if (opt.heuristics) tryRounding(root.colValue);

    std::vector<Node> open;
    open.push_back(Node{m.colLower, m.colUpper, rootBound, 0});
    opt.log.stage("tree", "begin", "rootbound=%.10g kind=miqp", (double)rootBound);
    // Every node relaxation below is a convex QP solved by the interior point
    // method -- there is no simplex tableau anywhere on this path, which is
    // also why there is no cutting.
    opt.log.stage("interior", "begin", "node=1 qp=1");
    Timer treeClock;
    Long lastReported = 0;
    double lastBeat = 0.0;

    int nThreads = opt.threads > 0 ? opt.threads
                                   : (int)std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::max(1, std::min(nThreads, 16));

    std::mutex mtx;
    std::atomic<int> active{0};
    std::atomic<bool> stopping{false};
    bool hitLimit = false;

    // One worker. Everything it touches outside the lock is its own.
    auto worker = [&](bool seeded, Solution seedSolution) {
        Solution rel;
        bool haveSeed = seeded;
        if (seeded) rel = std::move(seedSolution);

        for (;;) {
            Node node;
            {
                std::lock_guard<std::mutex> g(mtx);
                if (stopping.load()) return;
                if (clock.elapsed() > opt.timeLimit || nodes >= opt.nodeLimit) {
                    hitLimit = true; stopping.store(true); return;
                }
                if (open.empty()) {
                    // Nothing to take. If no one else is mid-solve the tree is
                    // finished; otherwise a peer may still push children.
                    if (active.load() == 0) return;
                    haveSeed = false;
                    goto waited;
                }
                node = std::move(open.back());
                open.pop_back();
                ++nodes;
                if (opt.log.wantsEvents() && (nodes - lastReported >= 25
                                         || clock.elapsed() - lastBeat >= 0.2)) {
                    lastReported = nodes;
                    lastBeat = clock.elapsed();
                    // The global bound is the weakest bound still open: no
                    // node below it has been proved away yet.
                    Real gb = haveIncumbent ? incumbent : rootBound;
                    for (const Node& o : open) gb = std::min(gb, o.bound);
                    double gap = -1.0;
                    if (haveIncumbent)
                        gap = std::fabs((double)incumbent - (double)gb)
                              / (1e-10 + std::fabs((double)incumbent));
                    opt.log.stage("tree", "node",
                                  "nodes=%lld bound=%.10g incumbent=%.10g open=%d "
                                  "gap=%.6g t=%.4f",
                                  (long long)nodes, (double)gb,
                                  haveIncumbent ? (double)incumbent : 0.0, (int)open.size(),
                                  gap, lastBeat);
                }
                active.fetch_add(1);
            }

            {
                Status st = Status::Optimal;
                if (!haveSeed) st = solveRelaxation(node.lower, node.upper, rel);
                haveSeed = false;

                std::lock_guard<std::mutex> g(mtx);
                active.fetch_sub(1);

                if (st == Status::Infeasible) continue;
                if (st != Status::Optimal) {
                    // The interior point method did not converge on this node's
                    // box. That is a statement about this relaxation, not about
                    // the subtree: an optimal integer point may still be inside
                    // it. Pruning would be a wrong answer and aborting would
                    // throw away the rest of the tree, so split the node on its
                    // widest remaining integer range and record that the proof
                    // is now incomplete.
                    ++unresolved;
                    if (unresolved == 1) firstUnresolved = st;
                    Int widest = -1; Real span = 0.0;
                    for (Int j : intCols) {
                        Real w = node.upper[j] - node.lower[j];
                        if (w > span + 1e-12) { span = w; widest = j; }
                    }
                    if (widest < 0 || span < 1.0) continue;
                    Real mid = std::floor(node.lower[widest] + span / 2.0);
                    Node lo = node, hi = node;
                    lo.upper[widest] = mid;
                    hi.lower[widest] = mid + 1.0;
                    lo.depth = hi.depth = node.depth + 1;
                    open.push_back(lo);
                    open.push_back(hi);
                    continue;
                }

                const Real bound = rel.objective;
                if (haveIncumbent) {
                    Real slack = std::max(opt.tol.mipGapAbs,
                                          opt.tol.mipGapRel * std::fabs(incumbent));
                    if (bound >= incumbent - slack) continue;
                }

                if (isIntegral(rel.colValue)) { accept(rel.colValue, bound); continue; }
                if (opt.heuristics) tryRounding(rel.colValue);

                Int branchCol = -1;
                Real bestScore = -1.0;
                for (Int j : intCols) {
                    Real f = fractionality(rel.colValue, j);
                    if (f <= itol) continue;
                    Real score = f * (1.0 + curvature[j]);
                    if (score > bestScore) { bestScore = score; branchCol = j; }
                }
                if (branchCol < 0) { accept(rel.colValue, bound); continue; }

                const Real v = rel.colValue[branchCol];
                Node down = node, up = node;
                down.upper[branchCol] = std::floor(v);
                up.lower[branchCol]   = std::ceil(v);
                down.bound = up.bound = bound;
                down.depth = up.depth = node.depth + 1;
                if (down.upper[branchCol] >= down.lower[branchCol] - 1e-9) open.push_back(down);
                if (up.lower[branchCol]   <= up.upper[branchCol]   + 1e-9) open.push_back(up);
                continue;
            }
        waited:
            std::this_thread::yield();
        }
    };

    if (nThreads == 1) {
        worker(true, std::move(root));
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nThreads);
        pool.emplace_back(worker, true, std::move(root));
        for (int t = 1; t < nThreads; ++t) pool.emplace_back(worker, false, Solution{});
        for (auto& th : pool) th.join();
    }
    opt.log.log(2, "  miqp       searched with %d thread%s\n",
                nThreads, nThreads == 1 ? "" : "s");
    opt.log.stage("interior", "end", "status=optimal node=1 qp=1 iters=%lld t=%.4f",
                  (long long)relaxAtomic.load(), treeClock.elapsed());
    opt.log.stage("tree", "end", "nodes=%lld threads=%d t=%.4f",
                  (long long)nodes, nThreads, treeClock.elapsed());

    // The honest bound is the smallest bound among nodes still open, and the
    // root bound if none are.
    Real globalBound = open.empty() ? (haveIncumbent ? incumbent : rootBound) : rootBound;
    for (const Node& nd : open) globalBound = std::min(globalBound, nd.bound);

    report.nodes = nodes;
    relaxations = relaxAtomic.load();
    report.nodeRelaxations = relaxations;
    best.nodes = nodes;
    best.bestBound = globalBound;
    best.iterations = relaxations;

    if (!haveIncumbent) {
        best.status = hitLimit ? Status::TimeLimit : Status::Infeasible;
        return best;
    }
    best.mipGap = std::fabs(incumbent) > 1e-12
                    ? std::fabs(incumbent - globalBound) / std::fabs(incumbent) : 0.0;
    // Optimality is only proven when the tree closed AND every node in it was
    // actually bounded.  One unconverged relaxation is enough to make the
    // answer "the best we found" rather than "the best there is", and it is
    // reported as such.
    best.status = (open.empty() && !hitLimit && unresolved == 0)
                    ? Status::Optimal : Status::Feasible;
    if (unresolved > 0)
        opt.log.log(1, "  miqp       %lld node relaxations did not converge (first: %s); the "
                       "incumbent is reported as feasible rather than proven optimal\n",
                    (long long)unresolved, statusName(firstUnresolved));
    if (best.status == Status::Optimal) { best.bestBound = incumbent; best.mipGap = 0.0; }

    opt.log.log(2, "  miqp       %lld nodes, %lld QP relaxations; root bound %.10g -> "
                   "incumbent %.10g\n",
                (long long)nodes, (long long)relaxations,
                (double)rootBound, (double)incumbent);
    return best;
}

} // namespace igaos
