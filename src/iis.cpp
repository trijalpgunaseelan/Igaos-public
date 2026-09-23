#include "igaos/iis.hpp"
#include "igaos/solver.hpp"
#include <chrono>
#include <cstdio>
#include <string>

namespace igaos {
namespace {

using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// A working copy of the system.  Relaxing a member means widening its bound to
// infinity; the filter never deletes rows or columns, so every index in the
// working copy keeps meaning the same thing as in the original model.
struct Work {
    Model m;
    explicit Work(const Model& src) : m(src) {
        // Integrality is relaxed: see the header for why.
        for (VarType& t : m.colType) t = VarType::Continuous;
    }
};

bool infeasible(const Model& m, const IisOptions& iopt) {
    Solver s;
    s.opt.log.level = iopt.logLevel;
    s.opt.timeLimit = iopt.innerTimeLimit;
    Solution sol = s.solve(m);
    // Only a definite Infeasible counts.  A time limit or a numerical error is
    // NOT evidence of infeasibility, and treating it as such is how a filter
    // silently returns a subsystem that is not infeasible at all.
    return sol.status == Status::Infeasible;
}

enum class Kind { Row, Lower, Upper };

struct Candidate {
    Kind kind;
    Int  index;
};

}  // namespace

Iis computeIis(const Model& model, const IisOptions& iopt) {
    const Clock::time_point t0 = Clock::now();
    Iis out;

    Work w(model);
    if (!infeasible(w.m, iopt)) {
        out.solves = 1;
        out.time = since(t0);
        out.modelWasFeasible = true;
        return out;
    }
    long solves = 1;

    // Candidate list: every finite-bounded row, then every finite variable
    // bound.  A row or bound that is already infinite constrains nothing and
    // can never be part of an IIS, so it is not a candidate.
    std::vector<Candidate> cands;
    for (Int i = 0; i < model.numRow(); ++i)
        if (isFinite(model.rowLower[i]) || isFinite(model.rowUpper[i]))
            cands.push_back({Kind::Row, i});
    if (iopt.includeBounds) {
        for (Int j = 0; j < model.numCol(); ++j) {
            if (isFinite(model.colLower[j])) cands.push_back({Kind::Lower, j});
            if (isFinite(model.colUpper[j])) cands.push_back({Kind::Upper, j});
        }
    }

    std::vector<bool> keep(cands.size(), true);
    bool complete = true;

    for (size_t k = 0; k < cands.size(); ++k) {
        if (iopt.maxSolves > 0 && solves >= iopt.maxSolves) { complete = false; break; }
        if (iopt.timeLimit > 0 && since(t0) >= iopt.timeLimit) { complete = false; break; }

        const Candidate c = cands[k];
        // Save, relax, test.
        Real savedLo = 0, savedUp = 0;
        switch (c.kind) {
            case Kind::Row:
                savedLo = w.m.rowLower[c.index]; savedUp = w.m.rowUpper[c.index];
                w.m.rowLower[c.index] = -kInf;   w.m.rowUpper[c.index] = kInf;
                break;
            case Kind::Lower:
                savedLo = w.m.colLower[c.index];
                w.m.colLower[c.index] = -kInf;
                break;
            case Kind::Upper:
                savedUp = w.m.colUpper[c.index];
                w.m.colUpper[c.index] = kInf;
                break;
        }

        const bool stillInfeasible = infeasible(w.m, iopt);
        ++solves;

        if (stillInfeasible) {
            // Not needed for the contradiction.  Leave it relaxed permanently:
            // that is what shrinks the system as the pass proceeds.
            keep[k] = false;
        } else {
            // Load-bearing: restore it and keep it in the subsystem.
            switch (c.kind) {
                case Kind::Row:
                    w.m.rowLower[c.index] = savedLo; w.m.rowUpper[c.index] = savedUp; break;
                case Kind::Lower:
                    w.m.colLower[c.index] = savedLo; break;
                case Kind::Upper:
                    w.m.colUpper[c.index] = savedUp; break;
            }
        }
    }

    for (size_t k = 0; k < cands.size(); ++k) {
        if (!keep[k]) continue;
        switch (cands[k].kind) {
            case Kind::Row:   out.rows.push_back(cands[k].index); break;
            case Kind::Lower: out.lowerBounds.push_back(cands[k].index); break;
            case Kind::Upper: out.upperBounds.push_back(cands[k].index); break;
        }
    }

    out.irreducible = complete;
    out.solves = solves;
    out.time = since(t0);
    return out;
}

std::string formatIis(const Model& model, const Iis& iis) {
    std::string s;
    char buf[512];

    if (iis.modelWasFeasible)
        return "model is not infeasible -- nothing to isolate\n";

    std::snprintf(buf, sizeof buf,
                  "IIS: %d members (%zu rows, %zu bounds) from %ld LP solves in %.3fs -- %s\n",
                  (int)iis.size(), iis.rows.size(),
                  iis.lowerBounds.size() + iis.upperBounds.size(),
                  iis.solves, iis.time,
                  iis.irreducible ? "irreducible"
                                  : "TRUNCATED, infeasible but not proven minimal");
    s += buf;

    for (Int i : iis.rows) {
        const std::string nm =
            (i < (Int)model.rowName.size() && !model.rowName[i].empty())
                ? model.rowName[i] : ("R" + std::to_string(i));
        std::string lo = isNegInf(model.rowLower[i]) ? "-inf" : std::to_string(model.rowLower[i]);
        std::string up = isInf(model.rowUpper[i]) ? "+inf" : std::to_string(model.rowUpper[i]);
        std::snprintf(buf, sizeof buf, "  row    %-24s  %s <= row <= %s\n",
                      nm.c_str(), lo.c_str(), up.c_str());
        s += buf;
    }
    auto emitBound = [&](Int j, bool lower) {
        const std::string nm =
            (j < (Int)model.colName.size() && !model.colName[j].empty())
                ? model.colName[j] : ("C" + std::to_string(j));
        const Real v = lower ? model.colLower[j] : model.colUpper[j];
        std::snprintf(buf, sizeof buf, "  bound  %-24s  %s %s %.10g\n",
                      nm.c_str(), lower ? "x >=" : "x <=", "", (double)v);
        s += buf;
    };
    for (Int j : iis.lowerBounds) emitBound(j, true);
    for (Int j : iis.upperBounds) emitBound(j, false);
    return s;
}

}  // namespace igaos
