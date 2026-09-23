// presolve.hpp : problem reductions, scaling, and the postsolve stack.
//
// Presolve is only useful if the reduced solution can be mapped back, so every
// reduction records what it needs for postsolve.  IGAOS takes a deliberately
// conservative route on duals: postsolve reconstructs a *complete basis* in the
// original space, and the driver then runs a cleanup solve of the original
// problem starting from it.  That cleanup normally costs zero iterations, and
// it makes the reported duals and reduced costs exact by construction rather
// than by a chain of per-reduction dual formulas.
#pragma once
#include "igaos/model.hpp"

namespace igaos {

// ---------------------------------------------------------------------------
struct Scaling {
    std::vector<Real> rowScale, colScale;   // A' = diag(rowScale) A diag(colScale)
    bool active = false;
    void apply(Model& m) const;
    // Exactly reverses apply().  Every scale factor is rounded to a power of
    // two, so the multiply and the divide are both exact and the model comes
    // back bit-for-bit -- which is what makes an unscaled retry a retry on the
    // SAME problem rather than on a rounded copy of it.
    void unapply(Model& m) const;
    void unscaleSolution(Solution& s) const;
};

// Iterated geometric-mean equilibration followed by rounding each factor to a
// power of two, so scaling itself introduces no rounding error.  Integer
// columns are left unscaled -- scaling them would destroy integrality.
// equilibrateQ: also let the entries of Q take part in the COLUMN pass.  The
// interior-point path factorizes [[Q + D, A^T], [A, -E]], so on a quadratic
// model a column scaling derived from A alone is blind to the block that
// actually sets the conditioning.  Q is scaled colScale[i]*colScale[j], so it
// belongs in the column statistics only; the row pass is unchanged.
//
// Off by default: every figure recorded in bench/results_* was measured with
// the A-only rule and must stay reproducible.
void computeScaling(const Model& m, Scaling& s, int passes = 6,
                    bool equilibrateQ = false);

// ---------------------------------------------------------------------------
struct PresolveResult {
    Model reduced;
    Status status = Status::NotSolved;      // Infeasible if detected during presolve

    std::vector<Int>  colMap;               // reduced col -> original col
    std::vector<Int>  rowMap;               // reduced row -> original row
    std::vector<Int>  origColToReduced;     // original col -> reduced col or kNone
    std::vector<Int>  origRowToReduced;     // original row -> reduced row or kNone
    std::vector<Real> fixedValue;           // value for removed columns
    std::vector<uint8_t> colRemoved, rowRemoved;

    // statistics reported in the log and the benchmark table
    Int removedCols = 0, removedRows = 0, tightenedBounds = 0, removedNnz = 0;
};

void presolve(const Model& m, const Options& opt, PresolveResult& r);

// Rebuild a full original-space solution (values + a complete basis) from the
// reduced solution.
void postsolve(const Model& orig, const PresolveResult& r,
               const Solution& reduced, Solution& full);

} // namespace igaos
