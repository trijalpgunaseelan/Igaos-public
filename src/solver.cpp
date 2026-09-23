#include "igaos/solver.hpp"
#include "igaos/simplex.hpp"
#include "igaos/cuts.hpp"
#include "igaos/ipm.hpp"
#include "igaos/crossover.hpp"
#include "igaos/pdhg.hpp"
#include <queue>
#include <deque>
#include <algorithm>
#include <cstdio>
#include <cstring>
#if defined(__linux__)
#include <unistd.h>
#endif
#include <thread>
#include <mutex>
#include <atomic>

namespace igaos {

// ===========================================================================
//  Continuous driver: picks between the simplex, the interior point method and
//  the first-order path, then makes the result look the same to every caller.
// ===========================================================================
Solution Solver::solveInterior(const Model& m, bool allowFallback) {
    Solution s;
    s.resize(m.numRow(), m.numCol());

    opt.log.stage("interior", "begin", "qp=%d", (int)m.isQp());
    Timer ipmClock;
    IpmResult ip = interiorPoint(m, opt);
    opt.log.stage("interior", "end", "status=%s iters=%d factornnz=%lld t=%.4f",
                  statusName(ip.status), (int)ip.iterations,
                  (long long)ip.factorNonzeros, ipmClock.elapsed());
    report.ipmIterations = ip.iterations;
    opt.log.log(2, "  interior   %d iterations, %lld factor nonzeros, %d regularized pivots\n",
                (int)ip.iterations, (long long)ip.factorNonzeros, (int)ip.regularizedPivots);

    if (ip.status != Status::Optimal) {
        // The interior point method does not produce an infeasibility or
        // unboundedness certificate, so a failure here is not a verdict on the
        // model -- it is a verdict on this path.  The simplex, which does
        // certify both, gets the final word.
        if (allowFallback && !m.isQp()) {
            opt.log.log(2, "  interior point did not converge (%s); falling back to the simplex\n",
                        statusName(ip.status));
            return solveSimplex(m, true);
        }
        s.status = ip.status;
        s.algorithm = "interior point";
        return s;
    }

    if (!m.isQp() && opt.crossover) {
        opt.log.stage("crossover", "begin");
        Timer xClock;
        CrossoverResult cr = crossover(m, opt, ip);
        opt.log.stage("crossover", "end", "status=%s pushes=%d iters=%lld t=%.4f",
                      statusName(cr.status), (int)cr.pushes,
                      (long long)cr.iterations, xClock.elapsed());
        report.crossoverPushes = cr.pushes;
        report.crossoverIterations = cr.iterations;
        if (cr.status == Status::Optimal) {
            opt.log.log(2, "  crossover  %d pushes, %lld simplex iterations to a basic solution\n",
                        (int)cr.pushes, (long long)cr.iterations);
            s.status = Status::Optimal;
            s.colValue = cr.colValue; s.colDual = cr.colDual;
            s.rowValue = cr.rowValue; s.rowDual = cr.rowDual;
            s.colStatus = cr.colStatus; s.rowStatus = cr.rowStatus;
            s.objective = cr.objective;
            s.iterations = ip.iterations + cr.iterations;
            s.primalInf = cr.primalInfeasibility;
            s.dualInf = cr.dualInfeasibility;
            s.algorithm = "interior point + crossover";
            return s;
        }
        opt.log.log(2, "  crossover did not reach a basic optimum (%s); "
                       "reporting the interior point itself\n", statusName(cr.status));
    } else {
        // Two different reasons, and they are worth telling apart: a QP optimum
        // need not sit at a vertex at all, so there is nothing to cross over to.
        opt.log.stage("crossover", "skip", "qp=%d", (int)m.isQp());
    }

    s.status = Status::Optimal;
    s.colValue = ip.x;
    s.rowValue = ip.s;
    s.rowDual  = ip.y;
    for (Int j = 0; j < m.numCol(); ++j) s.colDual[j] = ip.zLower[j] - ip.zUpper[j];
    s.objective = m.objectiveValue(s.colValue);
    s.dualObjective = ip.dualObjective;
    s.iterations = ip.iterations;
    s.primalInf = m.primalInfeasibility(s.colValue);
    s.algorithm = m.isQp() ? "interior point (convex QP)" : "interior point";
    return s;
}

Solution Solver::solveSimplex(const Model& m, bool preferDual) {
    opt.log.stage(preferDual ? "dual" : "primal", "begin");
    Timer stageClock;
    Simplex sx;
    sx.load(m.A, m.obj, m.colLower, m.colUpper, m.rowLower, m.rowUpper, opt);
    Status st = sx.solve(preferDual);
    Solution s;
    s.resize(m.numRow(), m.numCol());
    s.status = st;
    const auto& v = sx.values();
    const auto& d = sx.duals();
    for (Int j = 0; j < m.numCol(); ++j) { s.colValue[j] = v[j]; s.colDual[j] = d[j]; }
    const auto& y = sx.rowDuals();
    for (Int i = 0; i < m.numRow(); ++i) s.rowDual[i] = y[i];
    m.rowActivity(s.colValue, s.rowValue);
    sx.extractStatus(s.colStatus, s.rowStatus);
    s.objective = m.objectiveValue(s.colValue);
    s.iterations = sx.iterations();
    s.primalInf = sx.primalInfeasibility();
    s.dualInf = sx.dualInfeasibility();
    s.algorithm = preferDual ? "dual simplex" : "primal simplex";
    opt.log.stage(preferDual ? "dual" : "primal", "end",
                  "status=%s iters=%lld pinf=%.3e t=%.4f", statusName(st),
                  (long long)s.iterations, (double)s.primalInf, stageClock.elapsed());
    report.conditionEstimate = std::max(report.conditionEstimate, sx.basisConditionEstimate());
    return s;
}

Solution Solver::solveFirstOrder(const Model& m) {
    Solution s;
    s.resize(m.numRow(), m.numCol());
    if (m.isQp()) {
        opt.log.log(1, "  note: the first-order path solves linear objectives only; "
                       "using the interior point method\n");
        return solveInterior(m, true);
    }

    opt.log.stage("pdhg", "begin");
    Timer pdClock;
    PdhgResult pr = primalDualHybridGradient(m, opt);
    opt.log.stage("pdhg", "end", "status=%s iters=%lld restarts=%d t=%.4f",
                  statusName(pr.status), (long long)pr.iterations,
                  (int)pr.restarts, pdClock.elapsed());
    report.pdhgIterations = pr.iterations;
    opt.log.log(2, "  first-order %lld iterations, %d restarts, ||A|| = %.4g, "
                   "primal %.2e dual %.2e gap %.2e\n",
                (long long)pr.iterations, (int)pr.restarts, (double)pr.matrixNorm,
                (double)pr.primalInfeasibility, (double)pr.dualInfeasibility,
                (double)pr.relativeGap);

    if (pr.status != Status::Optimal) {
        // A first-order method that runs out of iterations has produced an
        // approximate point, not a verdict.  The simplex finishes the job.
        opt.log.log(2, "  first-order path did not reach tolerance (%s); "
                       "falling back to the simplex\n", statusName(pr.status));
        return solveSimplex(m, true);
    }

    if (opt.crossover) {
        // Crossover consumes an interior-ish point; PDHG produces one, so the
        // same identification-and-clean-up machinery serves both paths.
        IpmResult bridge;
        bridge.status = Status::Optimal;
        bridge.x = pr.x; bridge.s = pr.s; bridge.y = pr.y;
        opt.log.stage("crossover", "begin");
        Timer xClock;
        CrossoverResult cr = crossover(m, opt, bridge);
        opt.log.stage("crossover", "end", "status=%s pushes=%d iters=%lld t=%.4f",
                      statusName(cr.status), (int)cr.pushes,
                      (long long)cr.iterations, xClock.elapsed());
        report.crossoverPushes = cr.pushes;
        report.crossoverIterations = cr.iterations;
        if (cr.status == Status::Optimal) {
            s.status = Status::Optimal;
            s.colValue = cr.colValue; s.colDual = cr.colDual;
            s.rowValue = cr.rowValue; s.rowDual = cr.rowDual;
            s.colStatus = cr.colStatus; s.rowStatus = cr.rowStatus;
            s.objective = cr.objective;
            s.iterations = pr.iterations + cr.iterations;
            s.primalInf = cr.primalInfeasibility;
            s.dualInf = cr.dualInfeasibility;
            s.algorithm = "first-order PDHG + crossover";
            return s;
        }
    }

    s.status = Status::Optimal;
    s.colValue = pr.x;
    s.rowValue = pr.s;
    s.rowDual  = pr.y;
    s.objective = m.objectiveValue(s.colValue);
    s.dualObjective = pr.dualObjective;
    s.iterations = pr.iterations;
    s.primalInf = m.primalInfeasibility(s.colValue);
    s.algorithm = "first-order PDHG";
    return s;
}

Solution Solver::solveContinuous(const Model& m, bool preferDual) {
    LpAlgorithm alg = opt.lpAlgorithm;
    if (alg == LpAlgorithm::Auto) {
        // A quadratic objective is not a preference, it is a requirement: the
        // simplex implemented here optimizes a linear objective, so handing it a
        // QP would silently return the solution to a different problem.
        alg = m.isQp() ? LpAlgorithm::InteriorPoint
                       : (preferDual ? LpAlgorithm::DualSimplex : LpAlgorithm::PrimalSimplex);
    } else if (m.isQp() && alg != LpAlgorithm::InteriorPoint) {
        opt.log.log(1, "  note: the model has a quadratic objective; using the interior "
                       "point method instead of the requested algorithm\n");
        alg = LpAlgorithm::InteriorPoint;
    }

    switch (alg) {
        case LpAlgorithm::InteriorPoint: return solveInterior(m, true);
        case LpAlgorithm::PDHG:          return solveFirstOrder(m);
        case LpAlgorithm::DualSimplex:   return solveSimplex(m, true);
        case LpAlgorithm::PrimalSimplex: return solveSimplex(m, false);
        default:                         return solveSimplex(m, preferDual);
    }
}

// ===========================================================================
//  Branch and cut: a root cutting-plane loop (Gomory mixed-integer, knapsack
//  cover and complemented MIR) followed by branch and bound with dual-simplex
//  warm starts, pseudocost branching with a reliability phase, hybrid plunge /
//  best-bound node selection, and rounding plus diving heuristics.
// ===========================================================================
namespace {

struct BoundChange { Int col; Real lo, up; };

struct Node {
    std::vector<BoundChange> path;      // bound changes from the root
    Real  bound = -kInf;
    int   depth = 0;
    std::vector<VarStatus> colStat, rowStat;
    bool  hasBasis = false;
    // what produced this node, so its pseudocost can be updated once solved
    Int   branchVar = kNone;
    Real  branchFrac = 0.0;
    bool  branchUp = false;
    Real  parentObj = 0.0;
};

struct NodeCmp {
    bool operator()(const Node* a, const Node* b) const { return a->bound > b->bound; }
};

// Pseudocosts must be learned from *measured* objective degradation per unit of
// fractionality moved, not from the fractionality itself -- seeding them with
// fractionality makes the rule behave like most-fractional branching.
struct Pseudocost {
    Real downSum = 0, upSum = 0;
    Int  downCnt = 0, upCnt = 0;
    Real down() const { return downCnt ? downSum / downCnt : 0.0; }
    Real up()   const { return upCnt   ? upSum   / upCnt   : 0.0; }
    Int  reliability() const { return std::min(downCnt, upCnt); }
};

} // namespace

Solution Solver::solveMip(const Model& m) {
    Timer clock;
    Int nc = m.numCol(), nr = m.numRow();
    std::vector<Int> intCols;
    for (Int j = 0; j < nc; ++j) if (m.colType[j] != VarType::Continuous) intCols.push_back(j);

    // `mm` is the LP relaxation the tree actually searches: the model plus the
    // cut rows.  Every check that touches the *solution* -- feasibility of an
    // incumbent, the reported row activities -- keeps using `m`, so cut rows
    // never leak into what the caller sees.
    Model mm = m;

    Simplex sx;
    sx.load(mm.A, mm.obj, mm.colLower, mm.colUpper, mm.rowLower, mm.rowUpper, opt);

    Solution best;
    best.resize(nr, nc);
    best.status = Status::NotSolved;
    Real incumbent = isFinite(opt.cutoff) ? opt.cutoff : kInf;
    bool haveIncumbent = false;

    // The root relaxation is an ordinary primal simplex solve, and saying so is
    // worth the two lines: without it a watcher sees the mixed-integer lane
    // light up with nothing above it, as though the tree had conjured a bound.
    opt.log.stage("primal", "begin", "root=1");
    Timer rootClock;
    Status rst = sx.solve(false);
    opt.log.stage("primal", "end", "status=%s iters=%lld root=1 t=%.4f",
                  statusName(rst), (long long)sx.iterations(), rootClock.elapsed());
    if (rst == Status::Infeasible) { best.status = Status::Infeasible; return best; }
    if (rst == Status::Unbounded)  { best.status = Status::Unbounded;  return best; }
    if (rst != Status::Optimal)    { best.status = rst; return best; }

    // ---- root cutting-plane loop ------------------------------------------
    CutStats cutStats;
    if (opt.cuts && !intCols.empty() && clock.elapsed() < 0.5 * opt.timeLimit) {
        CutLimits lim;
        lim.referencePoint = opt.cutReference;
        opt.log.stage("cuts", "begin");
        Timer cutClock;
        Status cst = runRootCutLoop(mm, sx, intCols, opt, lim, cutStats);
        opt.log.stage("cuts", "end", "kept=%d gomory=%d cover=%d mir=%d rounds=%d "
                      "before=%.10g after=%.10g t=%.4f",
                      (int)cutStats.applied, (int)cutStats.gomory, (int)cutStats.cover,
                      (int)cutStats.mir, (int)cutStats.rounds,
                      (double)cutStats.rootBefore, (double)cutStats.rootAfter,
                      cutClock.elapsed());
        if (cutStats.rejectedInvalid > 0)
            opt.log.log(1, "  WARNING: %d candidate cuts excluded the verification point "
                           "(worst violation %.3g, first from the %s separator)\n",
                        (int)cutStats.rejectedInvalid, (double)cutStats.worstInvalidViolation,
                        cutStats.firstInvalidOrigin ? cutStats.firstInvalidOrigin : "?");
        if (cst != Status::Optimal) {
            // Cutting must never turn a solvable root into an unsolvable one.
            mm = m;
            sx.load(mm.A, mm.obj, mm.colLower, mm.colUpper, mm.rowLower, mm.rowUpper, opt);
            if (sx.solve(false) != Status::Optimal) { best.status = cst; return best; }
            cutStats.applied = 0;
        } else if (cutStats.applied > 0) {
            opt.log.log(2, "  cuts       %d applied (%d gomory, %d cover, %d mir) in %d rounds; "
                           "%d purged; root bound %.10g -> %.10g\n",
                        (int)cutStats.applied, (int)cutStats.gomory, (int)cutStats.cover,
                        (int)cutStats.mir, (int)cutStats.rounds, (int)cutStats.purged,
                        (double)cutStats.rootBefore, (double)cutStats.rootAfter);
        }
    } else {
        opt.log.stage("cuts", "skip");
    }
    report.cutsApplied  = cutStats.applied - cutStats.purged;
    report.cutRounds    = cutStats.rounds;
    report.cutsGomory   = cutStats.gomory;
    report.cutsCover    = cutStats.cover;
    report.cutsMir      = cutStats.mir;
    report.rootBoundLp  = cutStats.rootBefore;
    report.rootBoundCut = cutStats.rootAfter;
    report.cutsInvalid       = cutStats.rejectedInvalid;
    report.cutInvalidOrigin  = cutStats.firstInvalidOrigin;
    report.cutInvalidWorst   = cutStats.worstInvalidViolation;

    std::vector<Real> rootLower = sx.lower(), rootUpper = sx.upper();
    std::vector<Real> rootX(sx.values().begin(), sx.values().begin() + nc);
    Real rootBound = m.objectiveValue(rootX);

    // A single warm-started re-solve should take tens of iterations, not tens of
    // thousands.  Capping each node keeps one pathological LP from consuming the
    // whole budget; a node that hits the cap is retried cold, and if that also
    // fails the run is reported as not proven rather than silently pruned.
    Long rootIters = sx.iterations();
    const Long nodeIterCap = std::max<Long>(2000, 8 * (Long)mm.numRow());

    // One search context per worker.  Everything a worker touches while it is
    // actually solving a node relaxation lives in here -- its own simplex, its
    // own iteration count, its own record of whether a relaxation on its watch
    // hit the cap.  Nothing in a Ctx is shared with another thread, and that is
    // the whole reason the tree below can be walked by more than one of them.
    struct Ctx {
        Simplex* sx = nullptr;
        Long     iters = 0;
        bool     exact = true;
    };
    Ctx rootCtx; rootCtx.sx = &sx;

    auto resolve = [&](Ctx& c, bool warm) {
        Simplex& s = *c.sx;
        s.resetIterations();
        s.opt.iterationLimit = nodeIterCap;
        Status st = s.solve(warm);
        if (st == Status::IterationLimit) {
            s.resetIterations();
            s.opt.iterationLimit = 4 * nodeIterCap;
            s.setSlackBasis();
            st = s.solve(false);
            if (st == Status::IterationLimit) c.exact = false;
        }
        c.iters += s.iterations();
        s.opt.iterationLimit = opt.iterationLimit;
        return st;
    };

    std::vector<Pseudocost> pc(nc);
    // Nodes are owned by the queue.  DEFECT 24: they used to be owned by a
    // `pool` vector that was never emptied until the solve ended, so memory
    // grew with the number of nodes EXPLORED rather than with the number still
    // open -- about 3.6 KB a node, which is 2 GB after half a million of them
    // and a process the kernel kills before the time limit is reached.  A node
    // is finished the moment its children have copied what they need from it,
    // so it is deleted there; what is left in the queue at the end is deleted
    // once, below.
    // A vector heap rather than std::priority_queue, for one reason: the queue
    // has to be FILTERED, and priority_queue does not let you look inside it.
    // See purgeDominated below.
    NodeCmp nodeCmp;
    std::vector<Node*> open;
    auto pushNode = [&](Node* n) {
        open.push_back(n);
        std::push_heap(open.begin(), open.end(), nodeCmp);
    };
    auto popNode = [&]() {
        std::pop_heap(open.begin(), open.end(), nodeCmp);
        Node* n = open.back();
        open.pop_back();
        return n;
    };

    // Everything below this line that more than one thread can touch is guarded
    // by `mtx`.  `incumbentAtomic` is the one deliberate exception: it is read
    // without the lock to set each node's objective cutoff and to prune, and a
    // stale read there is always a LARGER value than the truth, which prunes
    // less rather than more.  Pruning less is slow; pruning more is wrong.
    std::mutex mtx;
    std::atomic<double> incumbentAtomic{(double)incumbent};
    std::atomic<bool>   haveIncumbentAtomic{false};

    auto accept = [&](const std::vector<Real>& cand, Real obj) {
        std::lock_guard<std::mutex> guard(mtx);
        if (obj >= incumbent - opt.tol.mipGapAbs) return false;
        // The rounding and diving heuristics propose points that are integral by
        // construction but need not satisfy the rows.  This gate is the only
        // thing standing between such a point and the reported answer, so it is
        // held at the solver's own feasibility tolerance rather than a loose
        // multiple of it: a candidate admitted here at 1e-4 is handed back to
        // the caller as an optimal solution that violates its constraints by
        // 1e-4, and nothing downstream will catch it.
        if (m.primalInfeasibility(cand) > 10.0 * opt.tol.primalFeas) return false;
        if (m.integerInfeasibility(cand, opt.tol.integrality) > opt.tol.integrality) return false;
        incumbent = obj;
        haveIncumbent = true;
        incumbentAtomic.store((double)obj);
        haveIncumbentAtomic.store(true);
        best.colValue = cand;
        best.objective = obj;
        best.status = Status::Feasible;
        m.rowActivity(cand, best.rowValue);
        return true;
    };

    auto tryRounding = [&](const std::vector<Real>& x) {
        std::vector<Real> cand(x.begin(), x.begin() + nc);
        for (Int j : intCols) {
            Real v = std::floor(cand[j] + 0.5);
            cand[j] = std::min(std::max(v, m.colLower[j]), m.colUpper[j]);
        }
        accept(cand, m.objectiveValue(cand));
    };

    // Fractional diving with one-level backtracking.
    //
    // Fixing a batch of binaries at once is fast but frequently drives the LP
    // infeasible; fixing one at a time is reliable but slow.  This dive starts
    // with a batch, and on infeasibility rolls the batch back and retries with a
    // smaller one, finally flipping the rounding direction of a single variable
    // before giving up.  On unit-commitment models -- where minimum generation
    // can exceed off-peak demand, so neither round-up nor round-down alone is
    // feasible -- the direction flip is what actually produces the incumbent.
    auto dive = [&](Ctx& c,
                    const std::vector<VarStatus>& cs0, const std::vector<VarStatus>& rs0,
                    const std::vector<Real>& lo0, const std::vector<Real>& up0,
                    int maxRounds, Real upBias, int batchDiv) {
        Simplex& sx = *c.sx;
        struct Fix { Int j; Real lo, up; };
        double budget = std::min(0.15 * opt.timeLimit, 8.0);
        double t0 = clock.elapsed();
        int curDiv = batchDiv;
        std::vector<Fix> batch;

        for (int round = 0; round < maxRounds; ++round) {
            if (clock.elapsed() > opt.timeLimit) break;
            if (clock.elapsed() - t0 > budget) break;

            std::vector<Real> x(sx.values().begin(), sx.values().begin() + nc);
            if (m.integerInfeasibility(x, opt.tol.integrality) <= opt.tol.integrality) {
                accept(x, m.objectiveValue(x));
                break;
            }
            std::vector<std::pair<Real, Int>> frac;
            for (Int j : intCols) {
                Real f = x[j] - std::floor(x[j]);
                Real d = std::min(f, 1.0 - f);
                if (d > opt.tol.integrality) frac.emplace_back(-d, j);
            }
            if (frac.empty()) { tryRounding(x); break; }
            std::sort(frac.begin(), frac.end());
            size_t take = std::max<size_t>(1, frac.size() / (size_t)std::max(1, curDiv));

            batch.clear();
            for (size_t t = 0; t < take; ++t) {
                Int j = frac[t].second;
                Real f = x[j] - std::floor(x[j]);
                Real v = (f > upBias) ? std::ceil(x[j]) : std::floor(x[j]);
                v = std::min(std::max(v, rootLower[j]), rootUpper[j]);
                batch.push_back({j, sx.lower()[j], sx.upper()[j]});
                sx.changeBound(j, v, v);
            }
            Status st = resolve(c, true);
            if (st == Status::Optimal) { curDiv = std::max(batchDiv, curDiv / 2); continue; }

            // roll the batch back
            for (const Fix& f : batch) sx.changeBound(f.j, f.lo, f.up);
            if (resolve(c, true) != Status::Optimal) break;

            if (take > 1) { curDiv *= 4; continue; }        // retry with a smaller batch

            // single variable: try the other direction before giving up
            Int j = batch[0].j;
            std::vector<Real> xr(sx.values().begin(), sx.values().begin() + nc);
            Real f = xr[j] - std::floor(xr[j]);
            Real other = (f > upBias) ? std::floor(xr[j]) : std::ceil(xr[j]);
            other = std::min(std::max(other, rootLower[j]), rootUpper[j]);
            sx.changeBound(j, other, other);
            if (resolve(c, true) != Status::Optimal) {
                sx.changeBound(j, batch[0].lo, batch[0].up);
                break;
            }
            curDiv = batchDiv;
        }
        sx.restoreBounds(lo0, up0);
        sx.setBasis(cs0, rs0);
    };

    Node* root = new Node();
    root->bound = rootBound; root->depth = 0;
    sx.extractStatus(root->colStat, root->rowStat);
    root->hasBasis = true;
    root->parentObj = rootBound;

    tryRounding(rootX);
    if (opt.heuristics) {
        // Nearest rounding first; then progressively more eager round-up dives.
        // On models whose hard constraints are >= (demand balance, spinning
        // reserve, minimum service level) committing extra capacity preserves
        // feasibility, so an up-biased dive finds an incumbent where the
        // nearest-rounding dive drives the LP infeasible.
        const Real biases[3] = {0.5, 0.25, 0.08};
        for (int b = 0; b < 3 && !haveIncumbent; ++b) {
            if (clock.elapsed() > 0.45 * opt.timeLimit) break;
            dive(rootCtx, root->colStat, root->rowStat, rootLower, rootUpper, 400, biases[b], 8);
        }
    }

    pushNode(root);

    // ---- how many threads walk the tree ------------------------------------
    //
    // The branch-and-cut tree is the one place in a MILP solve with real
    // parallelism in it: two nodes in different subtrees share nothing but the
    // incumbent.  Each worker below gets its OWN copy of the cut-augmented
    // relaxation, so a warm-started re-solve on one thread cannot disturb
    // another; the queue, the pseudocosts and the incumbent are the only things
    // behind the lock.
    //
    // A parallel tree is not deterministic.  Which node is expanded next
    // depends on which worker reached the queue first, so the node count and
    // the path the gap takes differ from run to run.  The ANSWER does not, and
    // that is the property the tests check.  A caller who needs a reproducible
    // node count passes --threads 1 and gets exactly the search this code did
    // before the pool existed.
    int nThreads = opt.threads > 0 ? opt.threads
                                   : (int)std::max(1u, std::thread::hardware_concurrency());
    nThreads = std::max(1, std::min(nThreads, 16));
    // Two reasons to stay serial.  Below a handful of integer variables the
    // tree is a few nodes deep and building a second factorization costs more
    // than it saves -- uc_m closes in one node once the cuts are in.  And each
    // worker's relaxation is a full copy of the matrix, so on a very large
    // model the pool is bounded by memory rather than by cores.
    if ((Int)intCols.size() < 8) nThreads = 1;
    if (mm.A.nnz() > 2000000) nThreads = std::min(nThreads, 4);

    opt.log.stage("tree", "begin", "rootbound=%.10g threads=%d",
                  (double)rootBound, nThreads);
    // Every node relaxation below is a warm-started dual simplex re-solve, so
    // the dual box is genuinely lit for as long as the tree is.
    opt.log.stage("dual", "begin", "node=1");
    Timer treeClock;
    Long nodeCount = 0;
    Long lastReported = 0;
    double lastBeat = 0.0;
    bool hitLimit = false;
    bool memoryStop = false;
    std::atomic<int>  active{0};
    std::atomic<bool> stopping{false};

    // The bound of the node each worker is holding right now.  A node in flight
    // has left the queue but has not been proved away, so the global bound has
    // to account for it -- without this, a run that stops on the time limit
    // would report a bound stronger than anything it actually proved.
    std::vector<Real>    inflight((size_t)nThreads, kInf);
    std::vector<Ctx>     ctx((size_t)nThreads);
    std::vector<Simplex> sxPool((size_t)nThreads);

    auto openBound = [&]() {                 // caller holds mtx
        Real gb = open.empty() ? kInf : open.front()->bound;
        for (Real b : inflight) gb = std::min(gb, b);
        return gb;
    };

    // How the memory limit is enforced.
    //
    // Every node in the open list carries a copy of its parent's basis and its
    // own path of bound changes, so it is not small -- but "not small" is not a
    // number, and an estimate built from sizeof() is wrong by a factor of two
    // or three once vector capacity and allocator headers are counted.  On
    // Linux the honest thing is available directly, so the limit is enforced
    // against the process's ACTUAL resident set; the size estimate below is
    // only the fallback for platforms without /proc.
    const size_t nodeBytes = sizeof(Node)
                           + 3 * 32                                   // heap headers
                           + 2 * (size_t)(mm.numCol() + mm.numRow()) * sizeof(VarStatus)
                           + 32 * sizeof(BoundChange);
    // Resolve the automatic setting once, at the top of the tree.
    int memLimitMb = opt.memoryLimitMb;
    if (memLimitMb < 0) {
        memLimitMb = 2048;
#if defined(__linux__)
        if (std::FILE* mi = std::fopen("/proc/meminfo", "r")) {
            char key[64]; long value = 0; char unit[16];
            while (std::fscanf(mi, "%63s %ld %15s", key, &value, unit) >= 2) {
                if (std::strncmp(key, "MemAvailable:", 13) == 0) {
                    long mb = value / 1024;                       // reported in kB
                    memLimitMb = (int)std::min<long>(16384, std::max<long>(512, mb * 3 / 5));
                    break;
                }
            }
            std::fclose(mi);
        }
#endif
        opt.log.log(2, "  tree       memory limit %d MB (automatic; --memory-limit "
                       "overrides, 0 disables)\n", memLimitMb);
    }
    // The fallback ceiling on the open list, for platforms with no /proc.  The
    // factor of three is the gap between the sizeof() estimate above and what
    // the allocator actually charges, measured on gt2.
    const size_t maxOpen = memLimitMb > 0
        ? std::max<size_t>(1000, ((size_t)memLimitMb << 20)
                                 / std::max<size_t>(1, 3 * nodeBytes))
        : (size_t)-1;

    auto overMemoryLimit = [&]() {                    // caller holds mtx
        if (memLimitMb <= 0) return false;
#if defined(__linux__)
        // statm field 2 is the resident set in pages.  Read every few thousand
        // nodes, not every node: it is a file read.
        static thread_local long pageSize = sysconf(_SC_PAGESIZE);
        std::FILE* f = std::fopen("/proc/self/statm", "r");
        if (f) {
            long total = 0, res = 0;
            int got = std::fscanf(f, "%ld %ld", &total, &res);
            std::fclose(f);
            if (got == 2)
                return (size_t)res * (size_t)pageSize > ((size_t)memLimitMb << 20);
        }
#endif
        return open.size() > maxOpen;                 // fallback: the estimate
    };
    Long lastMemCheck = 0;
    bool memoryHit = false;

    // Nodes whose bound has been passed by the incumbent can never produce a
    // better solution.  They are already skipped when popped -- but under
    // best-bound ordering they have the WORST bounds, so they are popped last
    // and sit in the list until then.  On a model the tree cannot close that is
    // most of the memory.  Dropping them when the list gets big is exact: it
    // removes nodes that are provably not worth exploring, and changes nothing
    // about the answer.  Caller holds mtx.
    size_t purgeAt = std::min<size_t>(maxOpen / 2, 200000);
    auto purgeDominated = [&]() {
        if (!haveIncumbent || open.size() < purgeAt) return;
        const Real cut = incumbent - opt.tol.mipGapAbs;
        size_t keep = 0;
        for (size_t i = 0; i < open.size(); ++i) {
            if (open[i]->bound > cut) { delete open[i]; continue; }
            open[keep++] = open[i];
        }
        const size_t dropped = open.size() - keep;
        open.resize(keep);
        std::make_heap(open.begin(), open.end(), nodeCmp);
        if (dropped) opt.log.log(2, "  tree       purged %zu dominated open nodes "
                                    "(%zu remain)\n", dropped, open.size());
        // Only re-purge once the list has grown appreciably again, so a list
        // full of live nodes is not swept on every push.
        purgeAt = std::min(maxOpen, std::max<size_t>(open.size() * 2, 20000));
    };

    ctx[0].sx    = &sx;                      // worker 0 inherits the root basis
    ctx[0].iters = rootCtx.iters;
    ctx[0].exact = rootCtx.exact;
    for (int t = 1; t < nThreads; ++t) {
        sxPool[(size_t)t].load(mm.A, mm.obj, mm.colLower, mm.colUpper,
                               mm.rowLower, mm.rowUpper, opt);
        ctx[(size_t)t].sx = &sxPool[(size_t)t];
    }

    auto worker = [&](int id) {
        Ctx&     c   = ctx[(size_t)id];
        Simplex& sxw = *c.sx;
        Node* plunge = nullptr;
        int   diveDepth = 0;

        // Give the node back: record what this worker is still holding (for
        // the global bound) and free the node itself.  Every path out of the
        // body below goes through here or through the branch block, and both
        // delete the node -- nothing else refers to it once its children have
        // copied its path.
        auto release = [&](Node* done, Real held) {
            {
                std::lock_guard<std::mutex> g(mtx);
                inflight[(size_t)id] = held;
                active.fetch_sub(1);
            }
            delete done;
        };

        for (;;) {
            Node* node  = nullptr;
            Long  myNode = 0;
            bool  idle  = false;
            {
                std::lock_guard<std::mutex> g(mtx);
                // A plunge node has left the queue.  Whenever this worker gives
                // up it has to go back, or the tree can end with `open` empty
                // and an unexplored subtree -- which would be reported as a
                // proof of optimality that was never completed.
                if (stopping.load()) {
                    if (plunge) { pushNode(plunge); plunge = nullptr; }
                    inflight[(size_t)id] = kInf;
                    return;
                }
                if (!plunge && open.empty()) {
                    inflight[(size_t)id] = kInf;
                    if (active.load() == 0) return;   // nobody can push any more
                    idle = true;
                } else if (clock.elapsed() > opt.timeLimit || nodeCount >= opt.nodeLimit
                           || (nodeCount - lastMemCheck >= 2000
                               && (lastMemCheck = nodeCount, memoryHit = overMemoryLimit()))) {
                    hitLimit = true;
                    if (memoryHit && !memoryStop) {
                        memoryStop = true;
                        opt.log.log(1, "  tree       stopping: this solve reached the %d MB "
                                       "memory limit with %zu nodes still open. The "
                                       "incumbent and the proven bound below are correct; "
                                       "optimality is NOT proven. Raise --memory-limit, or "
                                       "accept the bound.\n",
                                    memLimitMb, open.size());
                    }
                    stopping.store(true);
                    if (plunge) { pushNode(plunge); plunge = nullptr; }
                    inflight[(size_t)id] = kInf;
                    return;
                } else {
                    if (plunge) { node = plunge; plunge = nullptr; }
                    else { node = popNode(); diveDepth = 0; }
                    inflight[(size_t)id] = node->bound;
                    myNode = ++nodeCount;

                    // A heartbeat, so a caller watching the solve sees the tree
                    // moving rather than a frozen box.  Whichever comes first,
                    // every 100 nodes or every 0.2 s: node count alone goes
                    // silent for a minute on a model whose nodes are expensive,
                    // and time alone floods a model whose nodes are cheap.
                    if (opt.log.wantsEvents()) {
                        double now = clock.elapsed();
                        if (nodeCount - lastReported >= 100 || now - lastBeat >= 0.2) {
                            lastReported = nodeCount;
                            lastBeat = now;
                            Real gb = openBound();
                            if (gb > 1e29) gb = haveIncumbent ? incumbent : rootBound;
                            double gap = 1e30;
                            if (haveIncumbent && std::isfinite((double)gb))
                                gap = std::fabs((double)incumbent - (double)gb)
                                      / (1e-10 + std::fabs((double)incumbent));
                            opt.log.stage("tree", "node",
                                          "nodes=%lld bound=%.10g incumbent=%.10g open=%d "
                                          "gap=%.6g t=%.4f",
                                          (long long)nodeCount, (double)gb,
                                          haveIncumbent ? (double)incumbent : 0.0,
                                          (int)open.size(), gap > 1e29 ? -1.0 : gap, now);
                        }
                    }
                    active.fetch_add(1);
                }
            }
            if (idle) { std::this_thread::yield(); continue; }

            // ---- everything from here to `release` touches only this worker's
            //      own simplex, except where it takes the lock explicitly.
            const double incNow = incumbentAtomic.load();
            const bool   havNow = haveIncumbentAtomic.load();
            if (node->bound > incNow - opt.tol.mipGapAbs) { release(node, kInf); continue; }

            sxw.restoreBounds(rootLower, rootUpper);
            for (const BoundChange& b : node->path) sxw.changeBound(b.col, b.lo, b.up);
            if (node->hasBasis) sxw.setBasis(node->colStat, node->rowStat);
            // A dual-feasible basis makes every dual iterate a valid bound, so
            // the node can be abandoned the moment it passes the incumbent.
            sxw.setObjectiveCutoff(havNow ? (Real)incNow : kInf);
            Status st = resolve(c, node->hasBasis);
            sxw.setObjectiveCutoff(kInf);
            if (st != Status::Optimal) { release(node, kInf); continue; }

            std::vector<Real> x(sxw.values().begin(), sxw.values().begin() + nc);
            Real lpObj = m.objectiveValue(x);

            Int  branchCol = kNone;
            {
                std::lock_guard<std::mutex> g(mtx);
                // learn the pseudocost from the degradation this branch caused
                if (node->branchVar != kNone) {
                    Real deg = std::max(0.0, lpObj - node->parentObj);
                    Pseudocost& q = pc[node->branchVar];
                    if (node->branchUp) {
                        q.upSum += deg / std::max(1e-6, 1.0 - node->branchFrac); q.upCnt++;
                    } else {
                        q.downSum += deg / std::max(1e-6, node->branchFrac);     q.downCnt++;
                    }
                }
                Real bestScore = -1.0;
                for (Int j : intCols) {
                    Real f = x[j] - std::floor(x[j]);
                    if (f < opt.tol.integrality || f > 1.0 - opt.tol.integrality) continue;
                    Real dn = std::max(pc[j].down() * f, 1e-6);
                    Real up = std::max(pc[j].up() * (1.0 - f), 1e-6);
                    Real score = dn * up;
                    // Unreliable pseudocounts fall back to fractionality, which
                    // is what the reliability phase of reliability branching
                    // replaces.
                    if (pc[j].reliability() < opt.reliability)
                        score = 1e-4 + std::min(f, 1.0 - f);
                    if (score > bestScore) { bestScore = score; branchCol = j; }
                }
            }
            if (lpObj > incumbentAtomic.load() - opt.tol.mipGapAbs) { release(node, kInf); continue; }
            if (branchCol == kNone) { accept(x, lpObj); release(node, kInf); continue; }

            tryRounding(x);
            if (opt.heuristics && !haveIncumbentAtomic.load() && (myNode % 150) == 1) {
                std::vector<VarStatus> cs0, rs0; sxw.extractStatus(cs0, rs0);
                std::vector<Real> lo0 = sxw.lower(), up0 = sxw.upper();
                dive(c, cs0, rs0, lo0, up0, 300,
                     haveIncumbentAtomic.load() ? 0.5 : 0.2, 6);
            }

            Real xv = x[branchCol];
            Real fl = std::floor(xv), ce = std::ceil(xv), frac = xv - fl;
            std::vector<VarStatus> cs, rs;
            sxw.extractStatus(cs, rs);
            const Real bLo = sxw.lower()[branchCol], bUp = sxw.upper()[branchCol];

            Node* down = new Node();
            down->path = node->path;
            down->path.push_back({branchCol, bLo, fl});
            down->bound = lpObj; down->depth = node->depth + 1;
            down->colStat = cs; down->rowStat = rs; down->hasBasis = true;
            down->branchVar = branchCol; down->branchFrac = frac; down->branchUp = false;
            down->parentObj = lpObj;

            Node* up = new Node();
            up->path = node->path;
            up->path.push_back({branchCol, ce, bUp});
            up->bound = lpObj; up->depth = node->depth + 1;
            up->colStat = cs; up->rowStat = rs; up->hasBasis = true;
            up->branchVar = branchCol; up->branchFrac = frac; up->branchUp = true;
            up->parentObj = lpObj;

            Node* preferred = (frac > 0.5) ? up : down;
            Node* other     = (frac > 0.5) ? down : up;
            {
                std::lock_guard<std::mutex> g(mtx);
                pushNode(other);
                if (diveDepth < 24) { plunge = preferred; ++diveDepth; }
                else pushNode(preferred);
                purgeDominated();
                inflight[(size_t)id] = plunge ? plunge->bound : kInf;
                active.fetch_sub(1);
            }
            delete node;          // both children copied its path; nobody else has it
        }
    };

    if (nThreads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threadPool;
        threadPool.reserve((size_t)nThreads);
        for (int t = 0; t < nThreads; ++t) threadPool.emplace_back(worker, t);
        for (std::thread& th : threadPool) th.join();
    }

    bool exact = true;
    Long totalIters = rootIters;
    for (const Ctx& c : ctx) { totalIters += c.iters; exact = exact && c.exact; }

    Real globalBound = openBound();
    if (globalBound > 1e29) globalBound = haveIncumbent ? incumbent : rootBound;
    globalBound = std::max(rootBound, globalBound);

    if (nThreads > 1)
        opt.log.log(2, "  tree       searched with %d threads (%lld nodes)\n",
                    nThreads, (long long)nodeCount);

    opt.log.stage("dual", "end", "status=optimal iters=%lld node=1 t=%.4f",
                  (long long)(totalIters - rootIters), treeClock.elapsed());
    opt.log.stage("tree", "end", "nodes=%lld bound=%.10g incumbent=%.10g iters=%lld "
                  "threads=%d t=%.4f",
                  (long long)nodeCount, (double)globalBound,
                  haveIncumbent ? (double)incumbent : 0.0,
                  (long long)totalIters, nThreads, treeClock.elapsed());

    best.bestBound = globalBound;
    best.nodes = nodeCount;
    best.iterations = totalIters;
    best.algorithm = (report.cutsApplied > 0)
        ? "branch and cut (root GMI/cover/MIR separation, dual simplex warm start, "
          "pseudocost branching, diving)"
        : "branch and bound (dual simplex warm start, pseudocost branching, diving)";
    if (haveIncumbent) {
        // The tree is only *closed* when the queue emptied on its own.  With a
        // worker pool a node in flight is not in the queue, so a worker that
        // gives up pushes its plunge node back before returning -- an empty
        // queue here really does mean nothing is left unexplored.
        best.mipGap = relGap(incumbent, globalBound);
        if (exact && open.empty() && !hitLimit) { best.status = Status::Optimal; best.mipGap = 0.0; }
        else if (exact && best.mipGap <= opt.tol.mipGapRel) best.status = Status::Optimal;
        else best.status = Status::Feasible;
    } else {
        best.status = (open.empty() && !hitLimit) ? Status::Infeasible : Status::TimeLimit;
    }
    report.nodes = nodeCount;
    for (Node* n : open) delete n;
    open.clear();
    return best;
}

// ===========================================================================
Solution Solver::solve(const Model& modelIn) {
    Timer total;
    Model m = modelIn;
    m.finalize();
    m.ensureNames();
    m.validate();

    bool maximize = (m.sense == Sense::Maximize);
    if (maximize) { m.toMaximizationNegated(); m.sense = Sense::Minimize; }

    auto st = m.stats();
    report.origRows = st.nrow; report.origCols = st.ncol;
    report.origNnz = st.nnz;   report.origInt = st.nint;
    report.maxCoefRatio = st.ratio;

    // ---- presolve ----------------------------------------------------------
    opt.log.stage("model", "end", "rows=%d cols=%d nnz=%lld int=%d",
                  (int)m.numRow(), (int)m.numCol(), (long long)m.A.nnz(), (int)m.numInt());
    Timer t0;
    PresolveResult pr;
    Model work;
    bool usedPresolve = false;
    // Presolve and scaling both rewrite the model, and neither of them knows
    // about row quadratics: presolve would substitute out a column that appears
    // in a product and drop the product with it, and scaling would rescale A
    // and leave the quadratic coefficients untouched.  Either one is a SILENT
    // wrong answer -- the model that gets solved is not the model that was
    // handed in -- so a quadratically constrained problem skips both, and says
    // so rather than letting a reader assume they ran.
    const bool quadraticRows = m.isQcqp();
    if (quadraticRows && (opt.presolve || opt.scaling))
        opt.log.log(2, "  note       presolve and scaling are skipped on a quadratically "
                       "constrained model: neither transforms the row quadratics, and a "
                       "reduction that drops them would be a silent wrong answer\n");
    if (opt.presolve && !quadraticRows) {
        opt.log.stage("presolve", "begin");
        presolve(m, opt, pr);
        if (pr.status == Status::Infeasible || pr.status == Status::Unbounded) {
            Solution s; s.resize(m.numRow(), m.numCol());
            s.status = pr.status;
            report.totalTime = total.elapsed();
            return s;
        }
        work = pr.reduced;
        usedPresolve = true;
        report.tightenedBounds = pr.tightenedBounds;
    } else {
        work = m;
        opt.log.stage("presolve", "skip");
    }
    report.presolveTime = t0.elapsed();
    if (opt.presolve && !quadraticRows)
        opt.log.stage("presolve", "end", "rows=%d cols=%d nnz=%lld tightened=%d t=%.4f",
                      (int)(m.numRow() - work.numRow()), (int)(m.numCol() - work.numCol()),
                      (long long)(m.A.nnz() - work.A.nnz()), (int)report.tightenedBounds,
                      t0.elapsed());
    report.presolvedRows = work.numRow();
    report.presolvedCols = work.numCol();
    report.presolvedNnz  = work.A.nnz();

    // ---- scaling -----------------------------------------------------------
    Scaling sc;
    // A quadratically constrained model is scaled only on request: Scaling now
    // carries qcon exactly, but no recorded benchmark was measured that way.
    const bool doScale = opt.scaling && (!quadraticRows || opt.scaleWithQ);
    if (doScale) {
        opt.log.stage("scaling", "begin");
        Timer ts;
        computeScaling(work, sc, 6, opt.scaleWithQ); sc.apply(work);
        opt.log.stage("scaling", "end", "t=%.4f", ts.elapsed());
    } else {
        opt.log.stage("scaling", "skip");
    }

    // ---- solve -------------------------------------------------------------
    Timer t1;
    opt.log.stage("solve", "begin", "kind=%s",
                  (work.isQcqp() || work.hasNonconvexObjective())
                      ? (work.isMip() ? "minlp" : "qcqp")
                      : (work.isMip() ? (work.isQp() ? "miqp" : "milp")
                                      : (work.isQp() ? "qp" : "lp")));
    // A quadratic objective with integer variables needs a tree over QP
    // relaxations, not over LP ones: branching on the linear part and then
    // evaluating the quadratic objective at the answer returns a feasible point
    // with a wrong "optimal" label.  See src/miqp.cpp.
    // Route.  A quadratic CONSTRAINT, or a nonconvex quadratic objective, means
    // no relaxation this solver has is a valid bound unless it is built for the
    // purpose -- so those go to the global path whether or not there are
    // integer variables, and everything else keeps the route it always had.
    const bool needsGlobal = work.isQcqp() || work.hasNonconvexObjective();
    Solution reduced =
        needsGlobal ? solveGlobal(work)
                    : (work.isMip() ? (work.isQp() ? solveMiqp(work) : solveMip(work))
                                    : solveContinuous(work, false));

    // A convex QP that fails numerically, RETRIED UNSCALED.
    //
    // The scaling is a geometric-mean equilibration of A.  Q is then transformed
    // consistently -- D Q D, so the scaled problem has the same optimum -- but Q
    // never gets a say in choosing D.  On a quadratic objective that is the
    // wrong thing to optimize for: the KKT matrix the interior point method
    // factors contains BOTH blocks, and a D that flattens A can leave Q spanning
    // orders of magnitude it did not span before.  The method then stalls with a
    // large primal residual and reports numerical_error, on a problem that is
    // perfectly solvable without any scaling at all.
    //
    // QBANDM is the clean example: 178 primal infeasibility scaled, and
    // 16352.3424715 to seven digits unscaled.  QRECIPE and QGROW7 are the same
    // story.  This retry is the same shape as the interior-point-to-simplex
    // fallback above it -- a failure on one path is a statement about the path,
    // not about the model -- and it costs nothing on the models that converge.
    if (reduced.status == Status::NumericalError && opt.scaling && !work.isMip()
        && work.isQp() && total.elapsed() < opt.timeLimit) {
        opt.log.log(1, "  note: the scaled convex QP did not converge (primal infeasibility "
                       "%.3e); retrying unscaled\n", (double)reduced.primalInf);
        Model unscaled = work;
        sc.unapply(unscaled);
        Options saved = opt;
        opt.scaling = false;
        Solution retry = solveContinuous(unscaled, false);
        opt = saved;
        if (retry.status == Status::Optimal) {
            // The retry solved the UNSCALED model, so its solution is already in
            // the presolved space and must not be unscaled again below.
            sc.active = false;
            reduced = retry;
            opt.log.log(1, "  note: unscaled retry solved it (%.10g, primal infeasibility "
                           "%.3e)\n", (double)retry.objective, (double)retry.primalInf);
        }
    }
    report.solveTime = t1.elapsed();
    opt.log.stage("solve", "end", "status=%s obj=%.10g iters=%lld nodes=%lld t=%.4f",
                  statusName(reduced.status), (double)reduced.objective,
                  (long long)reduced.iterations, (long long)reduced.nodes, t1.elapsed());
    report.simplexIterations = reduced.iterations;
    report.path = reduced.algorithm;

    if (doScale) sc.unscaleSolution(reduced);

    // ---- postsolve ---------------------------------------------------------
    opt.log.stage("postsolve", "begin");
    Solution full;
    if (usedPresolve) postsolve(m, pr, reduced, full);
    else full = reduced;

    // Last resort for a convex QP: the ORIGINAL model, with neither presolve nor
    // scaling.  Both reductions are sound -- they preserve the optimum -- but
    // both change the numbers the interior point method factors, and either can
    // be what tips a badly conditioned quadratic over.  QGROW7 converges without
    // presolve and diverges with it; QSHARE1B's residual drops by an order of
    // magnitude when presolve is turned off.  Trying the untouched problem costs
    // one more solve on a model that has already failed, and it either works or
    // it does not -- what it cannot do is turn a good answer into a bad one,
    // because it is only reached when there is no answer at all.
    if (full.status == Status::NumericalError && m.isQp() && !m.isMip()
        && (usedPresolve || opt.scaling) && total.elapsed() < opt.timeLimit) {
        opt.log.log(1, "  note: retrying the convex QP on the original model, "
                       "without presolve or scaling\n");
        Options saved = opt;
        opt.presolve = false;
        opt.scaling = false;
        Solution direct = solveContinuous(m, false);
        opt = saved;
        if (direct.status == Status::Optimal) {
            opt.log.log(1, "  note: it solved untouched (%.10g, primal infeasibility %.3e)\n",
                        (double)direct.objective, (double)direct.primalInf);
            full = direct;
            report.path = direct.algorithm;
        }
    }

    // ---- cleanup on the original problem -----------------------------------
    // Restores exact duals and reduced costs in the user's own space.  For a
    // correct basis this terminates immediately; it also acts as an independent
    // check that the presolve/postsolve chain preserved optimality.
    Timer t2;
    if (!m.isMip() && !m.isQp() && !m.isQcqp() &&
        (full.status == Status::Optimal || full.status == Status::Feasible)) {
        // Guarded against QP on purpose: the cleanup re-solves with the simplex,
        // which optimizes the linear objective only, so on a quadratic objective
        // it would "improve" the point against the wrong function.  Guarded
        // against a quadratically constrained model for a sharper reason: the
        // simplex would be handed the LINEAR part of every row, so a point it
        // called feasible could violate a bilinear balance by any amount at all,
        // and the check that guards this block measures infeasibility with the
        // same linear-only eyes.  A global answer must not be "improved" by
        // something that cannot see the constraints that made it global.
        Real before = m.objectiveValue(full.colValue);
        Simplex cx;
        Options copt = opt;
        // The cleanup is a verification pass, not the solve: bound it so it can
        // never dominate the run, and charge it against the remaining budget.
        copt.timeLimit = std::min(std::max(0.25, opt.timeLimit - total.elapsed()), 10.0);
        cx.load(m.A, m.obj, m.colLower, m.colUpper, m.rowLower, m.rowUpper, copt);
        cx.setBasis(full.colStatus, full.rowStatus);
        Status cs = cx.solve(true);
        if (cs != Status::Optimal) {          // dual path struggled: retry primal
            cx.setBasis(full.colStatus, full.rowStatus);
            cs = cx.solvePrimal();
        }
        std::vector<Real> xc(cx.values().begin(), cx.values().begin() + m.numCol());
        Real after = m.objectiveValue(xc);
        Real tolObj = 1e-7 * (1.0 + std::fabs(before));
        // The cleanup doubles as an independent check on the presolve/postsolve
        // chain: it may only confirm or improve the postsolved point.  If it
        // comes back worse, the postsolved solution stands and we say so.
        if (cs == Status::Optimal && after <= before + tolObj &&
            cx.primalInfeasibility() <= 1e3 * opt.tol.primalFeas) {
            const auto& v = cx.values(); const auto& d = cx.duals(); const auto& y = cx.rowDuals();
            for (Int j = 0; j < m.numCol(); ++j) { full.colValue[j] = v[j]; full.colDual[j] = d[j]; }
            for (Int i = 0; i < m.numRow(); ++i) full.rowDual[i] = y[i];
            cx.extractStatus(full.colStatus, full.rowStatus);
            m.rowActivity(full.colValue, full.rowValue);
            full.primalInf = cx.primalInfeasibility();
            full.dualInf   = cx.dualInfeasibility();
            full.status    = Status::Optimal;
            report.simplexIterations += cx.iterations();
        } else {
            opt.log.log(2, "  cleanup did not improve on the postsolved point "
                           "(%.10g -> %.10g, %s); keeping postsolved solution\n",
                        (double)before, (double)after, statusName(cs));
        }
    }

    // ---- cleanup on the original problem: mixed-integer case ---------------
    // The tree searches a presolved, scaled model; the incumbent it returns is
    // integral there, but the continuous variables carry whatever error the
    // scaling and postsolve chain introduced.  Fixing the integers at the values
    // the search proved and re-solving the remaining LP on the *original* model
    // restores exact continuous values, and doubles as an independent check:
    // the polish may only confirm or improve the incumbent, never replace it
    // with something worse.
    if (m.isMip() &&
        (full.status == Status::Optimal || full.status == Status::Feasible)) {
        Real before = m.objectiveValue(full.colValue);
        Real beforeInf = m.primalInfeasibility(full.colValue);
        Model pm = m;
        bool integral = true;
        for (Int j = 0; j < pm.numCol(); ++j) {
            if (pm.colType[j] == VarType::Continuous) continue;
            Real v = std::floor(full.colValue[j] + 0.5);
            if (std::fabs(full.colValue[j] - v) > 1e-4) { integral = false; break; }
            v = std::min(std::max(v, m.colLower[j]), m.colUpper[j]);
            pm.colLower[j] = v;
            pm.colUpper[j] = v;
        }
        if (integral) {
            Options copt = opt;
            copt.log.level = 0;
            copt.timeLimit = std::min(std::max(0.25, opt.timeLimit - total.elapsed()), 10.0);
            Simplex px;
            px.load(pm.A, pm.obj, pm.colLower, pm.colUpper, pm.rowLower, pm.rowUpper, copt);
            Status ps = px.solve(false);
            if (ps == Status::Optimal) {
                std::vector<Real> xp(px.values().begin(), px.values().begin() + pm.numCol());
                for (Int j = 0; j < pm.numCol(); ++j)
                    if (pm.colType[j] != VarType::Continuous) xp[j] = pm.colLower[j];
                Real after = m.objectiveValue(xp);
                Real afterInf = m.primalInfeasibility(xp);
                Real tolObj = 1e-7 * (1.0 + std::fabs(before));
                if (after <= before + tolObj && afterInf <= std::max(beforeInf, 10.0 * opt.tol.primalFeas)) {
                    full.colValue = xp;
                    m.rowActivity(full.colValue, full.rowValue);
                    full.primalInf = afterInf;
                    opt.log.log(3, "  polish     integers fixed, continuous re-solved: "
                                   "%.12g -> %.12g, infeasibility %.3g -> %.3g\n",
                                (double)before, (double)after, (double)beforeInf, (double)afterInf);
                } else {
                    opt.log.log(2, "  polish did not improve on the search's incumbent "
                                   "(%.10g -> %.10g); keeping it\n", (double)before, (double)after);
                }
            }
        }
    }
    report.cleanupTime = t2.elapsed();

    full.objective = m.objectiveValue(full.colValue);
    full.bestBound = reduced.bestBound;
    full.mipGap = reduced.mipGap;
    full.nodes = reduced.nodes;
    full.algorithm = reduced.algorithm;
    if (maximize) {
        full.objective = -full.objective;
        full.bestBound = -full.bestBound;
        for (Real& d : full.colDual) d = -d;
        for (Real& y : full.rowDual) y = -y;
    }
    opt.log.stage("postsolve", "end");
    full.solveTime = total.elapsed();
    report.totalTime = full.solveTime;
    opt.log.stage("done", "end", "status=%s obj=%.12g nodes=%lld t=%.4f",
                  statusName(full.status), (double)full.objective,
                  (long long)full.nodes, full.solveTime);
    return full;
}

} // namespace igaos
