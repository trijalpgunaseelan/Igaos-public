// cuts.hpp : cutting-plane separation, turning branch and bound into
// branch and cut.
//
// Three separators, all of which produce **globally valid** inequalities:
//
//   Gomory mixed-integer (GMI)  from a fractional row of the simplex tableau.
//                               Note "mixed-integer", not "fractional": the
//                               fractional Gomory cut is only valid when every
//                               variable in the row is integer, so it cannot be
//                               used on the mixed models this solver targets.
//   Knapsack cover              from a row whose support is entirely binary,
//                               with the standard extension step.
//   Complemented MIR (c-MIR)    from a single row, after substituting each
//                               variable against its nearer bound and trying a
//                               family of divisors.
//
// Separation happens at the root only, and that is a deliberate consequence of
// the validity requirement above rather than a shortcut.  A GMI cut read off a
// tableau row at a node is derived from that node's *locally* tightened bounds,
// so it is valid only in that subtree; keeping it globally would cut off
// feasible integer points elsewhere in the tree.  Root separation with the
// global bounds gives cuts that are sound everywhere, which is what lets the
// pool simply become extra rows of the model for the whole search.
#pragma once
#include "igaos/model.hpp"
#include "igaos/simplex.hpp"

namespace igaos {

// A cut is always stored in the ">=" orientation over structural columns:
//        sum_t val[t] * x[idx[t]]  >=  rhs
struct Cut {
    std::vector<Int>  idx;              // structural column indices, ascending
    std::vector<Real> val;
    Real        rhs       = 0.0;
    Real        violation = 0.0;        // rhs - a'x* at the point that produced it
    Real        efficacy  = 0.0;        // violation / ||val||_2
    const char* origin    = "cut";

    Int  nnz() const { return (Int)idx.size(); }
    Real activity(const std::vector<Real>& x) const {
        Real s = 0;
        for (size_t t = 0; t < idx.size(); ++t) s += val[t] * x[idx[t]];
        return s;
    }
};

struct CutStats {
    Int rounds = 0;
    Int gomory = 0, cover = 0, mir = 0;
    Int rejectedDense = 0, rejectedWeak = 0, rejectedUnstable = 0, rejectedDuplicate = 0;
    Int rejectedInvalid = 0;                  // excluded the verification point
    const char* firstInvalidOrigin = nullptr;
    Real worstInvalidViolation = 0.0;
    Int applied = 0, purged = 0;
    Real rootBefore = 0.0, rootAfter = 0.0;   // root LP bound, before and after
};

// Numerical acceptance limits.  A cut that fails any of these is worse than no
// cut at all: it either destroys the conditioning of the basis or fills in the
// factorization for a bound improvement that does not survive rounding.
struct CutLimits {
    Real minEfficacy   = 1e-5;    // violation / ||a||_2
    Real maxDynamism   = 1e6;     // max|a_j| / min|a_j| within one cut
    Real maxAbsCoef    = 1e7;     // reject a cut carrying a huge raw coefficient
    Real maxTableauNorm= 1e7;     // reject a GMI row read off an ill-conditioned basis
    Real minCoef       = 1e-11;   // coefficients below this are dropped
    Real minFrac       = 0.01;    // fractionality window for GMI / MIR
    Real maxFrac       = 0.99;
    // Every cut is relaxed by this relative amount before it enters the model.
    // A cut is only ever an approximation of an exact halfspace once it has been
    // through floating-point arithmetic, and shaving the right-hand side by a
    // hair costs nothing in strength while removing the possibility of clipping
    // the true optimum by rounding error.
    Real safetyRelax   = 1e-9;
    Int  maxNnzAbs     = 0;       // 0 = derive from the column count
    Real maxNnzFrac    = 0.35;    // reject cuts denser than this share of columns
    Int  maxPerRound   = 200;
    Int  maxTotal      = 2000;

    // Verification hook.  When set, every candidate cut is tested against a
    // point that is known to be feasible for the original mixed-integer problem
    // (typically a proven optimum from a reference run).  A valid cut can never
    // exclude such a point, so any cut that does is a bug in the derivation --
    // this turns a silent wrong answer into an immediate, localized failure.
    // Left null in normal solves; the regression suite and the fuzz harness set it.
    const std::vector<Real>* referencePoint = nullptr;

    Int maxNnz(Int ncol) const {
        Int byFrac = (Int)(maxNnzFrac * (Real)ncol) + 1;
        Int cap = maxNnzAbs > 0 ? maxNnzAbs : std::max<Int>(100, byFrac);
        return std::max<Int>(1, std::min(cap, ncol));
    }
};

// ---------------------------------------------------------------------------
// Pool with duplicate suppression.  Cuts are normalized to max|coefficient| = 1
// before hashing so that two derivations of the same halfspace collapse.
// ---------------------------------------------------------------------------
class CutPool {
public:
    // Column bounds are needed to drop a negligible coefficient *safely*: a
    // ">=" cut only stays valid under a dropped term if the right-hand side is
    // relaxed by that term's largest possible contribution, which is a function
    // of the variable's bounds.  Without them the pool keeps every term.
    CutPool(Int ncol, const CutLimits& lim,
            const std::vector<Real>* colLower = nullptr,
            const std::vector<Real>* colUpper = nullptr)
        : ncol_(ncol), lim_(lim), lower_(colLower), upper_(colUpper) {}

    // Merges, normalizes, screens and stores. Returns true if the cut was kept.
    bool offer(Cut c, CutStats& st);

    const std::vector<Cut>& cuts() const { return cuts_; }
    Int  size() const { return (Int)cuts_.size(); }
    void keepOnly(const std::vector<uint8_t>& keep);

private:
    Int        ncol_;
    CutLimits  lim_;
    const std::vector<Real>* lower_ = nullptr;
    const std::vector<Real>* upper_ = nullptr;
    std::vector<Cut>      cuts_;
    std::vector<uint64_t> keys_;
};

// ---------------------------------------------------------------------------
// Separators.  Each returns the number of cuts accepted into the pool.
// `x` is the current LP relaxation solution over structural columns.
// ---------------------------------------------------------------------------
Int separateGomory(Simplex& sx, const Model& m, const std::vector<uint8_t>& isIntCol,
                   const CutLimits& lim, CutPool& pool, CutStats& st, Int maxCuts);

Int separateCover(const Model& m, const std::vector<Real>& x,
                  const CutLimits& lim, CutPool& pool, CutStats& st, Int maxCuts);

Int separateMir(const Model& m, const std::vector<Real>& x,
                const CutLimits& lim, CutPool& pool, CutStats& st, Int maxCuts);

// Append cuts[from..end) to the model as new rows (rhs <= a'x <= +inf).
void appendCutRows(Model& m, const std::vector<Cut>& cuts, Int from);

// ---------------------------------------------------------------------------
// The root cut loop.  On entry `sx` must be loaded with `mm` and solved to
// optimality.  On return `mm` carries the surviving cut rows, `sx` is loaded
// with the augmented model and re-solved, and `st` records what happened.
// Returns the status of the final root solve.
// ---------------------------------------------------------------------------
Status runRootCutLoop(Model& mm, Simplex& sx, const std::vector<Int>& intCols,
                      const Options& opt, const CutLimits& lim, CutStats& st);

} // namespace igaos
