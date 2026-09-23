// sensitivity.hpp : shadow prices, reduced costs and ranging for a solved LP.
//
// A solved LP answers "what should I do". Sensitivity answers the question a
// planner asks next, which is "what is it worth to change my mind":
//
//   * the SHADOW PRICE of a row -- how much the objective moves per unit of
//     right-hand side, i.e. what one more barrel of capacity is worth;
//   * the RANGE over which that price holds, because a shadow price quoted
//     without its range is worse than no number at all: it is only valid while
//     the optimal basis survives, and outside that range it is simply wrong;
//   * the REDUCED COST of a variable at a bound -- how much its objective
//     coefficient would have to improve before using it pays;
//   * the OBJECTIVE RANGE over which the current plan stays optimal, which is
//     the honest answer to "how far can the crude price move before I have to
//     re-blend".
//
// Everything here is computed from the final simplex basis: the duals come
// from B^-T c_B and the ranges from the tableau column B^-1 a_j and tableau
// row of B^-1 A. Nothing is estimated, sampled or simulated.
//
// SCOPE, stated plainly. This is linear-programming duality. It applies to an
// LP solved to optimality by the simplex path. It does NOT apply to:
//   * a mixed-integer model -- the LP dual of a MIP has no such interpretation,
//     and a "shadow price" read off the final node is not a shadow price of the
//     integer problem;
//   * a quadratic objective -- the ranges below assume the objective is linear;
//   * a solution from the interior-point or first-order path without crossover,
//     because there is no basis to range over.
// In each of those cases `available` is false and `reason` says which one.
//
// DEGENERACY. At a degenerate optimum a basic variable sits on its bound, so
// some range collapses to zero width and the shadow price is one of several
// valid values. That is a property of the model, not an error, and the report
// marks it rather than hiding it.
#pragma once
#include "igaos/model.hpp"

namespace igaos {

struct Sensitivity {
    struct RowInfo {
        Real dual = 0.0;          // shadow price, in the user's objective sense
        Real activity = 0.0;      // row activity at the optimum
        Real lower = 0.0, upper = 0.0;   // the row's own bounds
        // Range over which the BINDING right-hand side may move with the
        // current basis, and therefore the shadow price, staying valid.
        Real rhsLower = 0.0, rhsUpper = 0.0;
        bool binding = false;     // the row is active (its logical is nonbasic)
        bool degenerate = false;  // a zero-width side: price is not unique
    };
    struct ColInfo {
        Real value = 0.0;
        Real reducedCost = 0.0;   // in the user's objective sense
        Real objCoef = 0.0;
        // Range over which this objective coefficient may move with the current
        // plan staying optimal.
        Real objLower = 0.0, objUpper = 0.0;
        bool basic = false;
        bool degenerate = false;
    };

    std::vector<RowInfo> rows;
    std::vector<ColInfo> cols;

    bool available = false;
    std::string reason;           // why not, when `available` is false
    Int  degenerateRows = 0, degenerateCols = 0;
};

// Compute sensitivity for `model` from the basis carried in `sol`.
// `sol` must come from a simplex (or crossed-over) solve of this same model.
Sensitivity computeSensitivity(const Model& model, const Solution& sol,
                               const Options& opt = Options{});

// Human-readable report. `maxLines` caps each table (0 = no cap); rows and
// columns are ordered by |shadow price| and |reduced cost| so the lines that
// matter come first.
std::string formatSensitivity(const Model& model, const Sensitivity& s,
                              Int maxLines = 25);

}  // namespace igaos
