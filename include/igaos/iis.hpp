// iis.hpp : irreducible infeasible subsystem.
//
// When a model is infeasible, a Farkas dual ray PROVES it -- but a proof is not
// an answer to the question a planner actually asks, which is "which of my
// constraints are fighting?".  An IIS answers that: a subset of the rows and
// variable bounds that is infeasible, and from which the removal of ANY single
// member makes it feasible.  Minimal, so nothing in it is noise.
//
// The method is the classical deletion filter.  Start from the whole system and
// walk it once; at each member, relax that member and re-solve.  If the system
// is still infeasible without it, the member was not needed and is dropped for
// good.  If relaxing it makes the system feasible, it is load-bearing and stays.
// After one pass every surviving member is load-bearing, which is exactly the
// irreducibility condition.
//
// Cost is one LP solve per candidate, so the filter is bounded by
// IisOptions::maxSolves and reports honestly when it stopped early: a truncated
// run yields an infeasible subsystem that is SMALL but not proven irreducible,
// and `irreducible` says which of the two you have.
#pragma once
#include "igaos/model.hpp"

namespace igaos {

struct IisOptions {
    // Include variable bounds as candidates, not only rows.  A refinery model
    // is often infeasible because of a capacity bound rather than a balance
    // row, and omitting bounds would hide that.
    bool includeBounds = true;

    // Upper bound on the LP solves the filter may spend.  <= 0 means unlimited.
    // Each candidate costs one solve, so this is roughly the candidate budget.
    long maxSolves = 0;

    // Wall-clock ceiling for the whole filter, seconds.  <= 0 means unlimited.
    double timeLimit = 0.0;

    // Per-solve time limit handed to the inner solver.
    double innerTimeLimit = 10.0;

    int logLevel = 0;           // inner solves are silent by default
};

struct Iis {
    // Indices into the ORIGINAL model.
    std::vector<Int> rows;
    std::vector<Int> lowerBounds;   // columns whose LOWER bound is in the IIS
    std::vector<Int> upperBounds;   // columns whose UPPER bound is in the IIS

    // True when the filter ran to completion, so every member is load-bearing
    // and the subsystem is genuinely irreducible.  False when maxSolves or
    // timeLimit stopped it: the subsystem is still infeasible, just not proven
    // minimal.  Never report a truncated result as an IIS without saying so.
    bool irreducible = false;

    // Set when the input model was NOT infeasible, in which case there is
    // nothing to isolate and every vector above is empty.
    bool modelWasFeasible = false;

    long solves = 0;            // LP solves the filter spent
    double time = 0.0;

    Int size() const {
        return (Int)(rows.size() + lowerBounds.size() + upperBounds.size());
    }
};

// Compute an IIS of `model`.  The model is treated as a continuous system: any
// integrality is relaxed, because an IIS over the LP relaxation is what makes
// the diagnosis actionable and an integer-infeasible system needs a different
// question asked of it.  `model` is not modified.
Iis computeIis(const Model& model, const IisOptions& iopt = IisOptions{});

// One line per member, in the model's own row and column names.
std::string formatIis(const Model& model, const Iis& iis);

}  // namespace igaos
