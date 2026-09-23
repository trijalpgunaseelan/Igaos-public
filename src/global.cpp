// global.cpp : spatial branch and bound for nonconvex QCQP and bilinear MINLP.
//
// ===========================================================================
//  WHAT THIS SOLVES, AND WHY IT IS THE ONE THAT MATTERS FOR A REFINERY
// ===========================================================================
//  A model whose ROWS contain products of variables:
//
//      min  c'x + 1/2 x'Qx
//      s.t. rl_i <= a_i'x + SUM_t coef_t * x_{a(t)} * x_{b(t)} <= ru_i
//           l <= x <= u,   x_j integral for j in the integer set
//
//  Q may be indefinite.  Nothing here assumes convexity anywhere.
//
//  This is the POOLING problem.  Blend quality is a flow-weighted average, so
//  the quality leaving a tank multiplied by the flow leaving it is a product of
//  two decisions, and a refinery that wants qualities to be decisions rather
//  than fixed inputs is asking for exactly this class.  Every other model in
//  this repository -- refinery_blend included -- linearises that product on a
//  volume basis, which is the standard PLANNING approximation.  This file is
//  what makes it exact.
//
// ===========================================================================
//  THE ALGORITHM, AND WHY EACH PIECE IS THERE
// ===========================================================================
//  A nonconvex problem cannot be pruned by a local solution: a local optimum is
//  not a bound, so a branch-and-bound tree built on one proves nothing.  What is
//  needed is a RELAXATION -- something whose optimum is guaranteed no better
//  than the true one -- and a way to make it tighter as the search narrows.
//
//  1. RELAXATION.  Replace every product x_i*x_j by a new variable w, and
//     constrain w with the McCormick envelope: the convex hull of the graph of
//     x_i*x_j over the current box.  For x in [xl,xu], y in [yl,yu]:
//
//         w >= xl*y + yl*x - xl*yl        w <= xu*y + yl*x - xu*yl
//         w >= xu*y + yu*x - xu*yu        w <= xl*y + yu*x - xl*yu
//
//     The result is LINEAR, so the relaxation is an LP -- or a MILP if the
//     model has integer variables, which is what makes this a MINLP solver and
//     not just an NLP one.  Either way it is solved by the machinery that is
//     already here and already tested, which is the whole reason this file is
//     as short as it is.
//
//  2. THE ENVELOPE IS EXACT AT THE CORNERS AND LOOSE IN THE MIDDLE, and its
//     looseness shrinks with the width of the box -- quadratically.  So:
//
//  3. SPATIAL BRANCHING.  Split the box of a CONTINUOUS variable.  This is the
//     step that has no analogue in a MILP: there is no integrality to violate,
//     the thing being violated is w != x_i*x_j, and halving a variable's range
//     tightens every envelope that variable appears in.  Branch on the term
//     with the largest violation, on whichever of its two variables is wider
//     relative to its original range.
//
//  4. BOUND TIGHTENING.  Feasibility-based (interval propagation over the rows,
//     including the bilinear terms) at every node, and optimality-based (solve
//     the relaxation twice per variable, minimising and maximising it) at the
//     root.  OBBT is expensive and it is worth it: on pooling models it often
//     collapses a box far enough that the tree is a handful of nodes.
//
//  5. INCUMBENTS.  The relaxation's x is a genuine point of the original space;
//     it is simply not feasible while w != x_i*x_j.  Two cheap repairs make it
//     one: evaluate the true rows at x and accept it if it happens to satisfy
//     them, and -- the one that actually works on pooling -- FIX one side of
//     every product at its relaxation value and re-solve, which turns the
//     bilinear model into a linear one whose solution is exactly feasible for
//     the original.  Alternate the fixed side and repeat.
//
//  TERMINATION.  A node is done when its relaxation is infeasible, when its
//  bound is past the incumbent, or when the relaxation's own solution satisfies
//  every product to tolerance -- at which point the bound IS attained and the
//  node is solved exactly.  The tree closing means the answer is GLOBAL, and
//  that word is used here only because the relaxation is valid at every node.
// ===========================================================================
#include "igaos/solver.hpp"
#include "igaos/simplex.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <vector>

namespace igaos {

namespace {

// One distinct product x_i * x_j appearing anywhere in the model, i >= j.
struct Pair {
    Int i, j;
    bool operator<(const Pair& o) const { return i != o.i ? i < o.i : j < o.j; }
};

struct SpatialNode {
    std::vector<Real> lo, up;      // the box, over the ORIGINAL columns
    Real bound = -kInf;
    int  depth = 0;
};

struct NodeCmp {
    bool operator()(const SpatialNode* a, const SpatialNode* b) const {
        return a->bound > b->bound;               // best bound first
    }
};

// The four McCormick inequalities, written as rows of the relaxation.
//
// Each is exact when both variables sit at a corner of the box and loose
// between, which is the fact the whole method rests on: shrink the box and the
// relaxation tightens toward the true product.  For i == j the four collapse to
// three -- two tangents to x^2 at the ends and the secant across them -- and
// the code below produces exactly that without a special case, because the
// duplicate simply repeats.
void addMcCormick(Model& r, Int wCol, Int xCol, Int yCol,
                  Real xl, Real xu, Real yl, Real yu) {
    // w - yl*x - xl*y >= -xl*yl
    Int a = r.addRow(-xl * yl, kInf);
    r.setElement(a, wCol, 1.0);
    r.setElement(a, xCol, -yl);
    r.setElement(a, yCol, -xl);

    // w - yu*x - xu*y >= -xu*yu
    Int b = r.addRow(-xu * yu, kInf);
    r.setElement(b, wCol, 1.0);
    r.setElement(b, xCol, -yu);
    r.setElement(b, yCol, -xu);

    // w - yl*x - xu*y <= -xu*yl
    Int c = r.addRow(-kInf, -xu * yl);
    r.setElement(c, wCol, 1.0);
    r.setElement(c, xCol, -yl);
    r.setElement(c, yCol, -xu);

    // w - yu*x - xl*y <= -xl*yu
    Int d = r.addRow(-kInf, -xl * yu);
    r.setElement(d, wCol, 1.0);
    r.setElement(d, xCol, -yu);
    r.setElement(d, yCol, -xl);
}

// Range of a product over a box: the extremes are always at corners.
void productRange(Real xl, Real xu, Real yl, Real yu, Real& lo, Real& up) {
    const Real c[4] = {xl * yl, xl * yu, xu * yl, xu * yu};
    lo = std::min(std::min(c[0], c[1]), std::min(c[2], c[3]));
    up = std::max(std::max(c[0], c[1]), std::max(c[2], c[3]));
}

// Interval arithmetic on a product, used by the propagator below.
struct Interval {
    Real lo = -kInf, up = kInf;
    bool empty() const { return lo > up + 1e-9; }
};

Interval mulInterval(Real xl, Real xu, Real yl, Real yu) {
    Interval r;
    productRange(xl, xu, yl, yu, r.lo, r.up);
    return r;
}

} // namespace

// ===========================================================================
//  Build the McCormick relaxation of `m` over the box [lo, up].
//
//  The relaxation has the original columns, one auxiliary per distinct product,
//  the original rows with each product replaced by its auxiliary, and the
//  envelope rows.  Integrality is carried through unchanged, so if the original
//  is a MINLP this is a MILP and the existing branch-and-cut solves it.
// ===========================================================================
static Model buildRelaxation(const Model& m,
                             const std::vector<Pair>& pairs,
                             const std::map<Pair, Int>& index,
                             const std::vector<Real>& lo,
                             const std::vector<Real>& up,
                             bool linearizeObjective) {
    Model r;
    r.name = m.name + "_mccormick";
    r.sense = Sense::Minimize;
    r.objOffset = m.objOffset;

    for (Int j = 0; j < m.numCol(); ++j)
        r.addColumn(lo[(size_t)j], up[(size_t)j], m.obj[j], m.colType[j],
                    j < (Int)m.colName.size() ? m.colName[j] : "");

    // One auxiliary per product, bounded by the range of the product itself.
    // Those bounds are not redundant with the envelope rows: they are what lets
    // the propagator reason about w before any row is looked at.
    std::vector<Int> wcol(pairs.size(), kNone);
    for (size_t p = 0; p < pairs.size(); ++p) {
        Real wl, wu;
        productRange(lo[(size_t)pairs[p].i], up[(size_t)pairs[p].i],
                     lo[(size_t)pairs[p].j], up[(size_t)pairs[p].j], wl, wu);
        wcol[p] = r.addColumn(wl, wu, 0.0, VarType::Continuous,
                              "w_" + std::to_string(pairs[p].i) + "_"
                                   + std::to_string(pairs[p].j));
    }

    for (Int i = 0; i < m.numRow(); ++i)
        r.addRow(m.rowLower[i], m.rowUpper[i],
                 i < (Int)m.rowName.size() ? m.rowName[i] : "");
    for (Int j = 0; j < m.numCol(); ++j)
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p)
            r.setElement(m.A.rowIdx[p], j, m.A.val[p]);

    // Each row's quadratic terms become linear terms in the auxiliaries.
    for (const Model::QuadTerm& t : m.qcon) {
        auto it = index.find(Pair{t.i, t.j});
        r.setElement(t.row, wcol[(size_t)it->second], t.coef);
    }

    // The objective's quadratic part.  Kept quadratic when it is convex, since
    // a convex QP relaxation is tighter than its McCormick linearisation and
    // the interior point method solves it directly; linearised when it is not,
    // because a nonconvex QP relaxation is not a bound at all.
    if (m.Q.nnz() > 0) {
        if (!linearizeObjective) {
            for (Int j = 0; j < m.Q.ncol; ++j)
                for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
                    r.setQuadratic(m.Q.rowIdx[p], j, m.Q.val[p]);
        } else {
            for (Int j = 0; j < m.Q.ncol; ++j)
                for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p) {
                    Int i = m.Q.rowIdx[p];
                    // Lower triangle: the diagonal carries 1/2 q x^2, an
                    // off-diagonal entry carries the pair once with no half.
                    Real c = (i == j) ? 0.5 * m.Q.val[p] : m.Q.val[p];
                    auto it = index.find(Pair{std::max(i, j), std::min(i, j)});
                    Int col = wcol[(size_t)it->second];
                    r.obj[col] += c;
                }
        }
    }

    for (size_t p = 0; p < pairs.size(); ++p) {
        const Int xi = pairs[p].i, xj = pairs[p].j;
        addMcCormick(r, wcol[p], xi, xj,
                     lo[(size_t)xi], up[(size_t)xi], lo[(size_t)xj], up[(size_t)xj]);
    }

    r.finalize();
    return r;
}

// ===========================================================================
//  Feasibility-based bound tightening: interval propagation over the rows.
//
//  For a row  rl <= sum a_j x_j + sum coef_t x_a x_b <= ru,  the interval of
//  every term but one bounds what is left over for that one.  Products are
//  handled with interval multiplication and then divided back out only when
//  the divisor interval excludes zero -- dividing by an interval straddling
//  zero gives (-inf, inf), which is true and useless, and doing it anyway is
//  how a propagator ends up producing a bound of -nan.
//
//  Returns false if the box is proved empty.
// ===========================================================================
static bool propagate(const Model& m, std::vector<Real>& lo, std::vector<Real>& up,
                      int rounds, Real tol) {
    const Int n = m.numCol(), nr = m.numRow();
    std::vector<std::vector<const Model::QuadTerm*>> rowQuad((size_t)nr);
    for (const Model::QuadTerm& t : m.qcon) rowQuad[(size_t)t.row].push_back(&t);

    for (int pass = 0; pass < rounds; ++pass) {
        bool moved = false;
        for (Int i = 0; i < nr; ++i) {
            const bool haveL = isFinite(m.rowLower[i]), haveU = isFinite(m.rowUpper[i]);
            if (!haveL && !haveU) continue;

            // Total activity interval of the row.
            Real actLo = 0.0, actUp = 0.0;
            bool infLo = false, infUp = false;
            std::vector<std::pair<Int, Real>> lin;      // (column, coefficient)
            for (Int j = 0; j < n; ++j)
                for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p)
                    if (m.A.rowIdx[p] == i) lin.emplace_back(j, m.A.val[p]);

            auto accumulate = [&](Real l, Real u) {
                if (isNegInf(l)) infLo = true; else actLo += l;
                if (isInf(u))    infUp = true; else actUp += u;
            };
            for (const auto& e : lin) {
                Real a = e.second, l = lo[(size_t)e.first], u = up[(size_t)e.first];
                accumulate(a >= 0 ? a * l : a * u, a >= 0 ? a * u : a * l);
            }
            for (const Model::QuadTerm* t : rowQuad[(size_t)i]) {
                Interval iv = mulInterval(lo[(size_t)t->i], up[(size_t)t->i],
                                          lo[(size_t)t->j], up[(size_t)t->j]);
                Real l = t->coef >= 0 ? t->coef * iv.lo : t->coef * iv.up;
                Real u = t->coef >= 0 ? t->coef * iv.up : t->coef * iv.lo;
                accumulate(l, u);
            }
            if (!infLo && !infUp) {
                if (haveU && actLo > m.rowUpper[i] + tol) return false;
                if (haveL && actUp < m.rowLower[i] - tol) return false;
            }

            // Back out each linear term in turn.
            for (const auto& e : lin) {
                const Int j = e.first;
                const Real a = e.second;
                if (a == 0.0) continue;
                Real tl = a >= 0 ? a * lo[(size_t)j] : a * up[(size_t)j];
                Real tu = a >= 0 ? a * up[(size_t)j] : a * lo[(size_t)j];
                Real restLo = infLo ? -kInf : actLo - tl;
                Real restUp = infUp ?  kInf : actUp - tu;

                Real newLo = -kInf, newUp = kInf;
                if (haveL && isFinite(restUp)) {        // a*x >= rl - restUp
                    Real b = m.rowLower[i] - restUp;
                    if (a > 0) newLo = b / a; else newUp = b / a;
                }
                if (haveU && isFinite(restLo)) {        // a*x <= ru - restLo
                    Real b = m.rowUpper[i] - restLo;
                    if (a > 0) newUp = std::min(newUp, b / a);
                    else       newLo = std::max(newLo, b / a);
                }
                if (isFinite(newLo) && newLo > lo[(size_t)j] + 1e-9) {
                    lo[(size_t)j] = newLo; moved = true;
                }
                if (isFinite(newUp) && newUp < up[(size_t)j] - 1e-9) {
                    up[(size_t)j] = newUp; moved = true;
                }
                if (lo[(size_t)j] > up[(size_t)j] + tol) return false;
            }
        }
        if (!moved) break;
    }
    return true;
}

// ===========================================================================
Solution Solver::solveGlobal(const Model& m) {
    Timer clock;
    const Int n = m.numCol();

    Solution best;
    best.resize(m.numRow(), n);
    best.status = Status::NotSolved;

    // ---- collect the distinct products ------------------------------------
    std::map<Pair, Int> index;
    std::vector<Pair> pairs;
    auto note = [&](Int a, Int b) {
        Pair p{std::max(a, b), std::min(a, b)};
        if (index.find(p) == index.end()) { index[p] = (Int)pairs.size(); pairs.push_back(p); }
    };
    for (const Model::QuadTerm& t : m.qcon) note(t.i, t.j);

    const bool nonconvexObj = m.hasNonconvexObjective();
    if (nonconvexObj)
        for (Int j = 0; j < m.Q.ncol; ++j)
            for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
                note(m.Q.rowIdx[p], j);

    if (pairs.empty()) {                      // nothing nonlinear after all
        return m.isMip() ? (m.isQp() ? solveMiqp(m) : solveMip(m))
                         : solveContinuous(m, false);
    }

    // Every variable in a product must be boxed; Model::validate enforces it,
    // and this is the second line of defence because a McCormick envelope built
    // on an infinite bound is not a relaxation, it is nonsense.
    for (const Pair& p : pairs)
        for (Int c : {p.i, p.j})
            if (!isFinite(m.colLower[c]) || !isFinite(m.colUpper[c])) {
                opt.log.log(0, "  global: column %d appears in a product but is unbounded; "
                               "cannot build a relaxation\n", (int)c);
                best.status = Status::NumericalError;
                return best;
            }

    opt.log.stage("global", "begin", "products=%d nonconvexobj=%d",
                  (int)pairs.size(), (int)nonconvexObj);
    opt.log.log(2, "  global     %d distinct product%s over %d column%s; objective is %s\n",
                (int)pairs.size(), pairs.size() == 1 ? "" : "s",
                (int)n, n == 1 ? "" : "s",
                m.Q.nnz() == 0 ? "linear" : (nonconvexObj ? "nonconvex quadratic"
                                                          : "convex quadratic"));

    // ---- a sub-solve of the relaxation, with this solver's own logging off --
    auto solveRelaxation = [&](const Model& r, double budget, Solution& out) {
        Solver sub;
        sub.opt = opt;
        sub.opt.log.level = 0;
        sub.opt.log.progress = false;
        sub.opt.log.onStage = nullptr;
        sub.opt.timeLimit = std::max(0.05, budget);
        sub.opt.threads = 1;              // the spatial tree is the parallel layer
        out = sub.solve(r);
        return out.status;
    };

    std::vector<Real> rootLo(m.colLower), rootUp(m.colUpper);
    if (!propagate(m, rootLo, rootUp, 8, opt.tol.primalFeas)) {
        opt.log.log(2, "  global     bound propagation proved the box empty at the root\n");
        best.status = Status::Infeasible;
        opt.log.stage("global", "end", "status=infeasible nodes=0");
        return best;
    }

    Real incumbent = isFinite(opt.cutoff) ? opt.cutoff : kInf;
    bool haveIncumbent = false;

    // ---- accept a candidate, but only if it is TRULY feasible --------------
    //
    // This gate is the whole difference between a global solver and a confident
    // wrong answer.  The relaxation's x satisfies the ENVELOPES; the original
    // rows contain the products themselves, and Model::rowActivity evaluates
    // those.  A point that passes here satisfies the original model.
    // How infeasible a point may be and still count as a solution.  Ten times
    // the solver's own feasibility tolerance -- the same multiple the
    // mixed-integer path uses, and deliberately NOT the looser gate that would
    // let a point through on the strength of the products alone.
    const Real feasTol = 10.0 * opt.tol.primalFeas;

    auto accept = [&](const std::vector<Real>& x) {
        if (m.primalInfeasibility(x) > feasTol) return false;
        if (m.integerInfeasibility(x, opt.tol.integrality) > opt.tol.integrality) return false;
        Real v = m.objectiveValue(x);
        if (v >= incumbent - opt.tol.mipGapAbs) return false;
        incumbent = v;
        haveIncumbent = true;
        best.colValue = x;
        best.objective = v;
        best.status = Status::Feasible;
        m.rowActivity(x, best.rowValue);
        return true;
    };

    // ---- the repair that actually finds pooling incumbents -----------------
    //
    // Fix one variable of every product at its relaxation value.  Every product
    // becomes linear, the model becomes an LP (or MILP), and its solution is
    // EXACTLY feasible for the original -- because the fixed side really is a
    // constant now, not an approximation of one.  Alternate which side is
    // fixed, because fixing the flows and optimising the qualities finds a
    // different point from fixing the qualities and optimising the flows, and
    // on a pooling model one of the two is usually much better.
    auto repair = [&](const std::vector<Real>& xRel, double budget) {
        for (int side = 0; side < 2; ++side) {
            Model f;
            f.name = m.name + "_fixed";
            f.objOffset = m.objOffset;
            for (Int j = 0; j < n; ++j)
                f.addColumn(m.colLower[j], m.colUpper[j], m.obj[j], m.colType[j]);
            for (Int i = 0; i < m.numRow(); ++i) f.addRow(m.rowLower[i], m.rowUpper[i]);
            for (Int j = 0; j < n; ++j)
                for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p)
                    f.setElement(m.A.rowIdx[p], j, m.A.val[p]);
            if (m.Q.nnz() > 0 && !nonconvexObj)
                for (Int j = 0; j < m.Q.ncol; ++j)
                    for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
                        f.setQuadratic(m.Q.rowIdx[p], j, m.Q.val[p]);

            // Which variables get pinned: one side of each product.
            std::vector<uint8_t> pin((size_t)n, 0);
            for (const Model::QuadTerm& t : m.qcon) pin[(size_t)(side == 0 ? t.i : t.j)] = 1;
            for (const Model::QuadTerm& t : m.qcon) {
                Int fixed = (side == 0 ? t.i : t.j), free = (side == 0 ? t.j : t.i);
                if (t.i == t.j) { pin[(size_t)t.i] = 1; continue; }
                f.setElement(t.row, free, t.coef * xRel[(size_t)fixed]);
            }
            // A square term pins its own variable, so it contributes a constant
            // that has to move to the right-hand side rather than to a column.
            for (const Model::QuadTerm& t : m.qcon)
                if (t.i == t.j) {
                    Real c = t.coef * xRel[(size_t)t.i] * xRel[(size_t)t.i];
                    if (isFinite(f.rowLower[t.row])) f.rowLower[t.row] -= c;
                    if (isFinite(f.rowUpper[t.row])) f.rowUpper[t.row] -= c;
                }
            for (Int j = 0; j < n; ++j)
                if (pin[(size_t)j]) {
                    Real v = std::min(std::max(xRel[(size_t)j], m.colLower[j]), m.colUpper[j]);
                    f.colLower[j] = f.colUpper[j] = v;
                }
            f.finalize();

            Solution s;
            if (solveRelaxation(f, budget, s) == Status::Optimal) accept(s.colValue);
        }
    };

    // ---- root relaxation ---------------------------------------------------
    Model rootRelax = buildRelaxation(m, pairs, index, rootLo, rootUp, nonconvexObj);
    Solution rootSol;
    opt.log.stage("global", "root");
    Status rs = solveRelaxation(rootRelax, std::min(0.25 * opt.timeLimit, 60.0), rootSol);
    if (rs == Status::Infeasible) {
        best.status = Status::Infeasible;
        opt.log.stage("global", "end", "status=infeasible nodes=1");
        return best;
    }
    if (rs != Status::Optimal && rs != Status::Feasible) {
        opt.log.log(1, "  global: the root relaxation did not solve (%s); no bound is "
                       "available and nothing below it can be trusted\n", statusName(rs));
        best.status = rs;
        opt.log.stage("global", "end", "status=%s nodes=1", statusName(rs));
        return best;
    }
    const Real rootBound = rootSol.objective;
    report.rootBoundLp = rootBound;

    {   // an incumbent before the tree starts, so the first nodes can be pruned
        std::vector<Real> x(rootSol.colValue.begin(), rootSol.colValue.begin() + n);
        accept(x);
        if (opt.heuristics) repair(x, std::min(0.05 * opt.timeLimit, 5.0));
    }

    // ---- optimality-based bound tightening, at the root only ---------------
    //
    // Minimise and maximise each product variable subject to the relaxation.
    // This is 2k LPs and it is the single most effective thing in the file on
    // pooling models: the McCormick envelope's looseness is quadratic in the
    // box width, so halving a range quarters the gap it contributes.
    if (opt.heuristics && clock.elapsed() < 0.4 * opt.timeLimit) {
        std::vector<Int> obbtCols;
        for (const Pair& p : pairs) { obbtCols.push_back(p.i); obbtCols.push_back(p.j); }
        std::sort(obbtCols.begin(), obbtCols.end());
        obbtCols.erase(std::unique(obbtCols.begin(), obbtCols.end()), obbtCols.end());

        int tightened = 0;
        const double budget = std::min(0.25 * opt.timeLimit, 30.0);
        const double t0 = clock.elapsed();
        for (Int c : obbtCols) {
            if (clock.elapsed() - t0 > budget) break;
            for (int dir = 0; dir < 2; ++dir) {
                Model probe = buildRelaxation(m, pairs, index, rootLo, rootUp, nonconvexObj);
                for (Real& o : probe.obj) o = 0.0;
                probe.Q = SparseMatrix();
                probe.obj[c] = (dir == 0 ? 1.0 : -1.0);
                probe.finalize();
                Solution s;
                if (solveRelaxation(probe, 2.0, s) != Status::Optimal) continue;
                Real v = s.colValue[(size_t)c];
                if (dir == 0 && v > rootLo[(size_t)c] + 1e-7) {
                    rootLo[(size_t)c] = v - 1e-9; ++tightened;
                } else if (dir == 1 && v < rootUp[(size_t)c] - 1e-7) {
                    rootUp[(size_t)c] = v + 1e-9; ++tightened;
                }
            }
        }
        if (tightened > 0) {
            propagate(m, rootLo, rootUp, 8, opt.tol.primalFeas);
            opt.log.log(2, "  global     optimality-based tightening moved %d bound%s\n",
                        tightened, tightened == 1 ? "" : "s");
        }
    }

    // ---- the spatial tree --------------------------------------------------
    std::priority_queue<SpatialNode*, std::vector<SpatialNode*>, NodeCmp> open;
    {
        SpatialNode* root = new SpatialNode();
        root->lo = rootLo; root->up = rootUp; root->bound = rootBound; root->depth = 0;
        open.push(root);
    }

    Long nodes = 0;
    Real globalBound = rootBound;
    bool exhausted = true;
    const Real prodTol = 1e-6;
    double lastBeat = 0.0;
    Long lastReported = 0;

    while (!open.empty()) {
        if (clock.elapsed() > opt.timeLimit || nodes >= opt.nodeLimit) {
            exhausted = false; break;
        }
        SpatialNode* node = open.top(); open.pop();
        globalBound = node->bound;
        ++nodes;

        if (opt.log.wantsEvents()) {
            double now = clock.elapsed();
            if (nodes - lastReported >= 25 || now - lastBeat >= 0.2) {
                lastReported = nodes; lastBeat = now;
                double gap = -1.0;
                if (haveIncumbent)
                    gap = std::fabs((double)incumbent - (double)globalBound)
                          / (1e-10 + std::fabs((double)incumbent));
                opt.log.stage("global", "node",
                              "nodes=%lld bound=%.10g incumbent=%.10g open=%d gap=%.6g t=%.4f",
                              (long long)nodes, (double)globalBound,
                              haveIncumbent ? (double)incumbent : 0.0,
                              (int)open.size(), gap, now);
            }
        }

        if (haveIncumbent && node->bound > incumbent - opt.tol.mipGapAbs) { delete node; continue; }

        Model relax = buildRelaxation(m, pairs, index, node->lo, node->up, nonconvexObj);
        Solution sol;
        Status st = solveRelaxation(relax, std::max(0.5, opt.timeLimit - clock.elapsed()), sol);
        if (st != Status::Optimal) {
            if (st == Status::Infeasible) { delete node; continue; }
            // Anything else is a failure to BOUND this subtree, and a subtree
            // with no bound cannot be discarded.  Recording that the proof is
            // incomplete is the only correct response; pruning here would be a
            // wrong answer wearing the word "global".
            exhausted = false;
            delete node;
            continue;
        }
        if (haveIncumbent && sol.objective > incumbent - opt.tol.mipGapAbs) { delete node; continue; }

        std::vector<Real> x(sol.colValue.begin(), sol.colValue.begin() + n);

        // How badly does the relaxation still lie about the products?
        Real worst = 0.0;
        size_t worstPair = 0;
        for (size_t p = 0; p < pairs.size(); ++p) {
            Int wc = n + (Int)p;
            Real viol = std::fabs(sol.colValue[(size_t)wc]
                                  - x[(size_t)pairs[p].i] * x[(size_t)pairs[p].j]);
            Real scale = 1.0 + std::fabs(sol.colValue[(size_t)wc]);
            if (viol / scale > worst) { worst = viol / scale; worstPair = p; }
        }

        accept(x);
        if (opt.heuristics && (nodes % 10) == 1)
            repair(x, std::min(0.02 * opt.timeLimit, 2.0));

        // ---- is this node finished? ----------------------------------------
        //
        // The test is not "are the products close" but the thing that sentence
        // is shorthand for: the relaxation's own point must be FEASIBLE for the
        // original model, and its TRUE objective must equal the bound the
        // relaxation just proved.  When both hold, the bound is attained inside
        // this box and nothing better can be in it.
        //
        // Testing the products instead was the first version and it was wrong.
        // A product can agree to 1e-6 while the row it sits in, multiplied by a
        // coefficient of a few hundred, is off by 1e-4 -- and the node would
        // close on a point that is not a solution, discarding a subtree that
        // might hold the real optimum.  A random bilinear instance caught it at
        // 2.8e-05, which is exactly the size of miss this structure produces.
        if (worst <= prodTol) {
            const Real trueInf = m.primalInfeasibility(x);
            const Real trueObj = m.objectiveValue(x);
            const bool attained =
                trueInf <= feasTol &&
                std::fabs(trueObj - sol.objective)
                    <= 1e-6 * (1.0 + std::fabs((double)trueObj));
            if (attained) { delete node; continue; }
            // Otherwise fall through and branch.  The products look converged
            // and the point is not one, so the box is still hiding something.
        }

        // ---- spatial branching ---------------------------------------------
        const Pair& bp = pairs[worstPair];
        Int branchCol = bp.i;
        if (bp.i != bp.j) {
            // Split whichever variable is wider RELATIVE to its original range:
            // an absolute comparison would always pick the variable measured in
            // the larger unit, which on a refinery model means always splitting
            // flows and never qualities.
            auto relWidth = [&](Int c) {
                Real full = m.colUpper[c] - m.colLower[c];
                return full > 1e-12 ? (node->up[(size_t)c] - node->lo[(size_t)c]) / full : 0.0;
            };
            branchCol = relWidth(bp.i) >= relWidth(bp.j) ? bp.i : bp.j;
        }

        const Real bl = node->lo[(size_t)branchCol], bu = node->up[(size_t)branchCol];
        if (bu - bl < 1e-9) {
            // The box is a point in this variable and the product is still
            // violated: no split can help, so the subtree cannot be closed.
            exhausted = false;
            delete node;
            continue;
        }
        // Split near the relaxation's own value, pulled toward the midpoint so
        // a point sitting exactly on a bound cannot produce an empty child.
        Real cut = 0.7 * x[(size_t)branchCol] + 0.3 * (0.5 * (bl + bu));
        const Real margin = 0.05 * (bu - bl);
        cut = std::min(std::max(cut, bl + margin), bu - margin);

        for (int side = 0; side < 2; ++side) {
            SpatialNode* child = new SpatialNode();
            child->lo = node->lo; child->up = node->up;
            if (side == 0) child->up[(size_t)branchCol] = cut;
            else           child->lo[(size_t)branchCol] = cut;
            child->depth = node->depth + 1;
            child->bound = sol.objective;                 // parent's bound is valid
            if (!propagate(m, child->lo, child->up, 4, opt.tol.primalFeas)) {
                delete child; continue;                   // box proved empty
            }
            open.push(child);
        }
        delete node;
    }

    // The honest bound is the weakest one still open.
    if (!open.empty()) {
        Real gb = kInf;
        std::vector<SpatialNode*> drain;
        while (!open.empty()) { drain.push_back(open.top()); open.pop(); }
        for (SpatialNode* nd : drain) { gb = std::min(gb, nd->bound); delete nd; }
        globalBound = std::max(rootBound, gb);
    } else if (haveIncumbent && exhausted) {
        globalBound = incumbent;
    }

    report.nodes = nodes;
    report.rootBoundCut = globalBound;
    best.nodes = nodes;
    best.bestBound = globalBound;
    best.algorithm = "spatial branch and bound (McCormick relaxations, "
                     "interval and optimality-based bound tightening)";

    if (!haveIncumbent) {
        best.status = exhausted ? Status::Infeasible : Status::TimeLimit;
    } else {
        best.mipGap = std::fabs(incumbent) > 1e-12
                        ? std::fabs(incumbent - globalBound) / std::fabs(incumbent) : 0.0;
        // "Optimal" here means GLOBALLY optimal, and it is claimed only when the
        // tree emptied on its own with every node bounded.  A run that stopped
        // on a limit, or that met a node it could not bound, reports Feasible --
        // the answer may well be the global optimum and this solver did not
        // prove it, which is a different sentence.
        if (exhausted && best.mipGap <= opt.tol.mipGapRel) { best.status = Status::Optimal;
                                                             best.bestBound = incumbent;
                                                             best.mipGap = 0.0; }
        else best.status = Status::Feasible;
    }
    best.primalInf = m.primalInfeasibility(best.colValue);

    opt.log.stage("global", "end", "status=%s nodes=%lld bound=%.10g incumbent=%.10g t=%.4f",
                  statusName(best.status), (long long)nodes, (double)globalBound,
                  haveIncumbent ? (double)incumbent : 0.0, clock.elapsed());
    opt.log.log(2, "  global     %lld node%s; root bound %.10g -> %.10g, incumbent %.10g\n",
                (long long)nodes, nodes == 1 ? "" : "s",
                (double)rootBound, (double)globalBound,
                haveIncumbent ? (double)incumbent : 0.0);
    return best;
}

} // namespace igaos
