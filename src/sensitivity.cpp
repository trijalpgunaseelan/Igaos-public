#include "igaos/sensitivity.hpp"
#include "igaos/simplex.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace igaos {
namespace {

constexpr Real kBig = 1e29;   // report threshold for "unbounded range"

inline bool finiteBound(Real v) { return v > -kInf && v < kInf; }

}  // namespace

Sensitivity computeSensitivity(const Model& model, const Solution& sol,
                               const Options& optIn) {
    Sensitivity out;

    // ---- scope gates, each with its own reason ------------------------------
    if (sol.status != Status::Optimal) {
        out.reason = "the solve did not finish optimal, so there is no optimal basis to range over";
        return out;
    }
    if (model.isMip()) {
        out.reason = "mixed-integer model -- LP duality does not carry over to the integer problem";
        return out;
    }
    if (model.isQp() || model.isQcqp()) {
        out.reason = "quadratic model -- the ranges here assume a linear objective";
        return out;
    }
    const Int m = model.numRow(), n = model.numCol();
    if ((Int)sol.colStatus.size() != n || (Int)sol.rowStatus.size() != m) {
        out.reason = "no basis in the solution -- run the simplex path, or add crossover "
                     "after an interior-point or first-order solve";
        return out;
    }

    // ---- rebuild the optimal basis -----------------------------------------
    // Internally the solver always minimizes; a maximization model is solved on
    // the negated objective, so duals and reduced costs come back with the
    // opposite sign and are flipped once, at the end, for reporting.
    const Real senseFlip = (model.sense == Sense::Maximize) ? -1.0 : 1.0;
    std::vector<Real> cost(model.obj);
    if (model.sense == Sense::Maximize)
        for (Real& c : cost) c = -c;

    Simplex sx;
    sx.opt = optIn;
    sx.opt.log.level = 0;
    sx.load(model.A, cost, model.colLower, model.colUpper,
            model.rowLower, model.rowUpper, sx.opt);
    sx.setBasis(sol.colStatus, sol.rowStatus);
    // Zero or near-zero iterations: the basis handed in is already optimal.
    // This is what makes the duals and the tableau available, and it also
    // refuses to report anything if the basis turns out not to be optimal.
    const Status st = sx.solve(false);
    if (st != Status::Optimal) {
        out.reason = "the supplied basis did not re-solve to optimal, so its duals are not trustworthy";
        return out;
    }

    const std::vector<Real>& x      = sx.values();      // extended: n structural + m logical
    const std::vector<Real>& dj     = sx.duals();       // reduced costs, extended
    const std::vector<Real>& y      = sx.rowDuals();
    const std::vector<Int>&  basis  = sx.basis();       // position -> extended index
    const std::vector<Real>& lo     = sx.lower();
    const std::vector<Real>& up     = sx.upper();
    const std::vector<VarStatus>& stat = sx.statuses();

    const Real tol = optIn.tol.primalFeas;

    out.rows.assign(m, Sensitivity::RowInfo{});
    out.cols.assign(n, Sensitivity::ColInfo{});

    // ---- right-hand-side ranging, row by row --------------------------------
    //
    // The logical variable of row i is extended index n+i and its value IS the
    // row activity. When it is nonbasic the row is binding and its bound is the
    // right-hand side that matters. Moving that bound by delta moves the basic
    // solution along B^-1 a_{n+i}; the range is the largest delta either way
    // that keeps every basic variable inside its own bounds.
    SparseVector col;
    col.resize(m);
    for (Int i = 0; i < m; ++i) {
        Sensitivity::RowInfo& r = out.rows[i];
        const Int e = n + i;
        r.activity = x[e];
        r.lower = model.rowLower[i];
        r.upper = model.rowUpper[i];
        r.dual = senseFlip * y[i];
        r.binding = (stat[e] != VarStatus::Basic);

        if (!r.binding) {
            // Slack row: the price is zero and the right-hand side can move
            // until the slack runs out.
            r.rhsLower = finiteBound(r.lower) ? r.lower : -kInf;
            r.rhsUpper = finiteBound(r.upper) ? r.upper :  kInf;
            continue;
        }

        const Real rhs = x[e];          // the bound it is sitting on
        col.clear();
        sx.tableauColumn(e, col);

        Real dMin = -kInf, dMax = kInf;   // admissible delta on the bound
        for (Int p = 0; p < m; ++p) {
            const Real a = col[p];
            if (std::fabs(a) <= optIn.tol.zero) continue;
            const Int  b  = basis[p];
            const Real xb = x[b];
            const Real lb = lo[b], ub = up[b];
            // x_B(delta) = x_B - delta * a  must stay within [lb, ub]
            if (a > 0) {
                if (finiteBound(lb)) dMax = std::min(dMax, (xb - lb) / a);
                if (finiteBound(ub)) dMin = std::max(dMin, (xb - ub) / a);
            } else {
                if (finiteBound(lb)) dMin = std::max(dMin, (xb - lb) / a);
                if (finiteBound(ub)) dMax = std::min(dMax, (xb - ub) / a);
            }
        }
        if (dMin > 0) dMin = 0;           // numerical safety: the range contains 0
        if (dMax < 0) dMax = 0;

        r.rhsLower = (dMin <= -kBig) ? -kInf : rhs + dMin;
        r.rhsUpper = (dMax >=  kBig) ?  kInf : rhs + dMax;
        r.degenerate = (std::fabs(dMin) <= tol) || (std::fabs(dMax) <= tol);
        if (r.degenerate) ++out.degenerateRows;
    }

    // ---- objective ranging, column by column --------------------------------
    //
    // Nonbasic j: the plan stays optimal while the reduced cost keeps its sign,
    // so c_j may move by exactly its reduced cost in one direction and without
    // limit in the other.
    //
    // Basic j at position p: changing c_j by delta changes every nonbasic
    // reduced cost d_k by -delta * alpha_k, where alpha is row p of B^-1 A. The
    // range is the largest delta that preserves every d_k's sign condition.
    SparseVector trow, rho;
    trow.resize(n + m);
    rho.resize(m);

    std::vector<Int> posOf(n + m, kNone);
    for (Int p = 0; p < m; ++p) posOf[basis[p]] = p;

    for (Int j = 0; j < n; ++j) {
        Sensitivity::ColInfo& c = out.cols[j];
        c.value = x[j];
        c.objCoef = model.obj[j];
        c.reducedCost = senseFlip * dj[j];
        c.basic = (stat[j] == VarStatus::Basic);

        if (!c.basic) {
            // Sign conventions are stated in the minimization space, then the
            // resulting interval is mapped back to the user's sense.
            const Real d = dj[j];
            Real loC, upC;
            if (stat[j] == VarStatus::AtUpper) {
                // optimality needs d <= 0, so c_j may rise by -d
                loC = -kInf; upC = cost[j] - d;
            } else {
                // at lower / at zero / fixed: needs d >= 0, so c_j may fall by d
                loC = cost[j] - d; upC = kInf;
            }
            if (model.sense == Sense::Maximize) {
                const Real a = (upC >= kBig) ? -kInf : -upC;
                const Real b = (loC <= -kBig) ? kInf : -loC;
                c.objLower = a; c.objUpper = b;
            } else {
                c.objLower = (loC <= -kBig) ? -kInf : loC;
                c.objUpper = (upC >=  kBig) ?  kInf : upC;
            }
            c.degenerate = (std::fabs(d) <= tol);
            if (c.degenerate) ++out.degenerateCols;
            continue;
        }

        const Int p = posOf[j];
        trow.clear(); rho.clear();
        sx.tableauRow(p, trow, rho);

        Real dMin = -kInf, dMax = kInf;
        for (Int k : trow.idx) {
            if (k == j) continue;
            if (stat[k] == VarStatus::Basic) continue;
            if (stat[k] == VarStatus::Fixed) continue;   // a fixed nonbasic constrains nothing
            const Real a = trow[k];
            if (std::fabs(a) <= optIn.tol.zero) continue;
            const Real d = dj[k];
            if (stat[k] == VarStatus::AtUpper) {
                // need d - delta*a <= 0
                if (a > 0) dMin = std::max(dMin, d / a);
                else       dMax = std::min(dMax, d / a);
            } else {
                // need d - delta*a >= 0
                if (a > 0) dMax = std::min(dMax, d / a);
                else       dMin = std::max(dMin, d / a);
            }
        }
        if (dMin > 0) dMin = 0;
        if (dMax < 0) dMax = 0;

        Real loC = (dMin <= -kBig) ? -kInf : cost[j] + dMin;
        Real upC = (dMax >=  kBig) ?  kInf : cost[j] + dMax;
        if (model.sense == Sense::Maximize) {
            const Real a = (upC >=  kBig || upC == kInf) ? -kInf : -upC;
            const Real b = (loC <= -kBig || loC == -kInf) ?  kInf : -loC;
            c.objLower = a; c.objUpper = b;
        } else {
            c.objLower = loC; c.objUpper = upC;
        }
        c.degenerate = (std::fabs(dMin) <= tol) || (std::fabs(dMax) <= tol);
        if (c.degenerate) ++out.degenerateCols;
    }

    out.available = true;
    return out;
}

std::string formatSensitivity(const Model& model, const Sensitivity& s, Int maxLines) {
    std::string o;
    char buf[400];

    if (!s.available) {
        o += "sensitivity unavailable: " + s.reason + "\n";
        return o;
    }

    auto num = [](Real v) {
        char b[40];
        if (v >= kBig)  { std::snprintf(b, sizeof b, "%12s", "+inf"); return std::string(b); }
        if (v <= -kBig) { std::snprintf(b, sizeof b, "%12s", "-inf"); return std::string(b); }
        std::snprintf(b, sizeof b, "%12.6g", (double)v);
        return std::string(b);
    };
    auto rowName = [&](Int i) {
        return (i < (Int)model.rowName.size() && !model.rowName[i].empty())
                   ? model.rowName[i] : ("R" + std::to_string(i));
    };
    auto colName = [&](Int j) {
        return (j < (Int)model.colName.size() && !model.colName[j].empty())
                   ? model.colName[j] : ("C" + std::to_string(j));
    };

    // Rows, most valuable constraint first.
    std::vector<Int> ri(s.rows.size());
    std::iota(ri.begin(), ri.end(), 0);
    std::sort(ri.begin(), ri.end(), [&](Int a, Int b) {
        return std::fabs(s.rows[a].dual) > std::fabs(s.rows[b].dual);
    });

    o += "\nSHADOW PRICES -- what one more unit of each binding constraint is worth\n";
    o += "  constraint                    shadow price      activity     rhs valid from"
         "            to\n";
    Int shown = 0, binding = 0;
    for (Int i : ri) {
        const Sensitivity::RowInfo& r = s.rows[i];
        if (!r.binding) continue;
        ++binding;
        if (maxLines > 0 && shown >= maxLines) continue;
        ++shown;
        std::snprintf(buf, sizeof buf, "  %-26s %s  %s  %s  %s%s\n",
                      rowName(i).substr(0, 26).c_str(),
                      num(r.dual).c_str(), num(r.activity).c_str(),
                      num(r.rhsLower).c_str(), num(r.rhsUpper).c_str(),
                      r.degenerate ? "   [degenerate]" : "");
        o += buf;
    }
    if (binding == 0) o += "  (no binding constraints -- every row has slack)\n";
    else if (maxLines > 0 && binding > shown) {
        std::snprintf(buf, sizeof buf, "  ... %d more binding constraints\n", (int)(binding - shown));
        o += buf;
    }

    // Columns at a bound, largest reduced cost first.
    std::vector<Int> ci(s.cols.size());
    std::iota(ci.begin(), ci.end(), 0);
    std::sort(ci.begin(), ci.end(), [&](Int a, Int b) {
        return std::fabs(s.cols[a].reducedCost) > std::fabs(s.cols[b].reducedCost);
    });

    o += "\nREDUCED COSTS -- how much each unused variable's coefficient must improve\n";
    o += "  variable                     reduced cost         value     obj valid from"
         "            to\n";
    shown = 0;
    Int atBound = 0;
    for (Int j : ci) {
        const Sensitivity::ColInfo& c = s.cols[j];
        if (c.basic) continue;
        if (std::fabs(c.reducedCost) <= 1e-12) continue;
        ++atBound;
        if (maxLines > 0 && shown >= maxLines) continue;
        ++shown;
        std::snprintf(buf, sizeof buf, "  %-26s %s  %s  %s  %s\n",
                      colName(j).substr(0, 26).c_str(),
                      num(c.reducedCost).c_str(), num(c.value).c_str(),
                      num(c.objLower).c_str(), num(c.objUpper).c_str());
        o += buf;
    }
    if (atBound == 0) o += "  (every variable is either basic or has a zero reduced cost)\n";
    else if (maxLines > 0 && atBound > shown) {
        std::snprintf(buf, sizeof buf, "  ... %d more variables at a bound\n", (int)(atBound - shown));
        o += buf;
    }

    if (s.degenerateRows || s.degenerateCols) {
        std::snprintf(buf, sizeof buf,
            "\n  note: %d row range(s) and %d column range(s) have a zero-width side.\n"
            "  The optimum is degenerate there, so the price is one of several valid\n"
            "  values and the range should be read as a caution, not a guarantee.\n",
            (int)s.degenerateRows, (int)s.degenerateCols);
        o += buf;
    }
    return o;
}

}  // namespace igaos
