// simplex.hpp : bounded-variable revised simplex, primal and dual.
//
// The LP is held in the computational form
//        [ A  -I ] [x; s] = 0,     l <= [x; s] <= u
// so every row is a ranged row and the logical variables carry the row bounds.
// Variables 0..n-1 are structural, n..n+m-1 logical.
//
// Primal iterations use Devex pricing with reduced costs updated through the
// pricing row; dual iterations use exact dual steepest edge weights with the
// Forrest-Goldfarb update.  Both ratio tests are Harris two-pass tests with a
// bound-flipping stage, which is what keeps degenerate industrial models
// (refinery pooling, unit commitment) from stalling.
#pragma once
#include "igaos/model.hpp"
#include "igaos/lu.hpp"

namespace igaos {

class Simplex {
public:
    // ---- setup -------------------------------------------------------------
    void load(const SparseMatrix& A,
              const std::vector<Real>& obj,
              const std::vector<Real>& colLower, const std::vector<Real>& colUpper,
              const std::vector<Real>& rowLower, const std::vector<Real>& rowUpper,
              const Options& opt);

    void setSlackBasis();
    void crashBasis();                        // triangular crash
    void setBasis(const std::vector<VarStatus>& colStat,
                  const std::vector<VarStatus>& rowStat);

    // ---- solve ---------------------------------------------------------------
    Status solve(bool preferDual);
    Status solvePrimal();
    Status solveDual();

    // ---- results -------------------------------------------------------------
    Real objective() const;
    const std::vector<Real>& values() const { return value_; }
    const std::vector<Real>& duals()  const { return dual_; }
    const std::vector<Real>& rowDuals() const { return y_; }
    const std::vector<Int>&  basis()  const { return basis_; }
    const std::vector<VarStatus>& statuses() const { return status_; }
    Long iterations() const { return iter_; }
    void resetIterations() { iter_ = 0; }
    Int  numRow() const { return m_; }
    Int  numCol() const { return n_; }
    Real primalInfeasibility() const { return primalInf_; }
    Real dualInfeasibility() const { return dualInf_; }
    Real basisConditionEstimate() const { return factor_.conditionEstimate(); }

    void extractStatus(std::vector<VarStatus>& colStat, std::vector<VarStatus>& rowStat) const;

    // ---- warm-start support for branch and cut --------------------------------
    void changeBound(Int extIdx, Real lo, Real up);
    void restoreBounds(const std::vector<Real>& lo, const std::vector<Real>& up);
    const std::vector<Real>& lower() const { return lower_; }
    const std::vector<Real>& upper() const { return upper_; }
    void setObjectiveCutoff(Real c) { cutoff_ = c; }

    // Row p of the simplex tableau (B^{-1}A restricted to nonbasics), used by
    // the Gomory cut generator.  `rowOut` is indexed by extended variable.
    void tableauRow(Int p, SparseVector& rowOut, SparseVector& rhoOut);
    // B^{-1} a_q, indexed by basis position.
    void tableauColumn(Int q, SparseVector& colOut);

    Options opt;

private:
    // problem
    Int m_ = 0, n_ = 0, nTot_ = 0;
    SparseMatrix A_;
    SparseMatrix At_;      // A^T in CSC == A in CSR: column i is row i of A
    ExtendedMatrix E_;
    std::vector<Real> cost_, lower_, upper_;

    // state
    std::vector<Int>       basis_;      // position -> extended index
    std::vector<Int>       inBasis_;    // extended index -> position or kNone
    std::vector<VarStatus> status_;
    std::vector<Real>      value_;      // all extended variables
    std::vector<Real>      xB_;         // basic values by position
    std::vector<Real>      dual_;       // reduced costs, all extended
    std::vector<Real>      y_;          // row duals
    std::vector<Real>      dseWeight_;  // dual steepest edge weights per position
    std::vector<Real>      devex_;      // primal devex reference weights
    std::vector<Real>      costShift_;  // dual phase-1 cost perturbation

    BasisFactor factor_;
    Long iter_ = 0;
    Int  sinceRefactor_ = 0;
    bool hasUserBasis_ = false;
    bool lastRefactorRepaired_ = false;   // basis positions were swapped for logicals
    Real primalInf_ = 0, dualInf_ = 0;
    Real cutoff_ = kInf;
    int  phase_ = 2;
    Timer timer_;

    // scratch
    SparseVector alpha_, rho_, row_, tau_;
    std::vector<Real> dwork_, scratchM_, scratchM2_;
    std::vector<Int>  cand_;

    // helpers
    void   allocate();
    bool   refactorize();
    void   computeBasicValues();
    void   computeDuals();
    Real   varLower(Int k) const { return lower_[k]; }
    Real   varUpper(Int k) const { return upper_[k]; }
    void   setNonbasicToBound(Int k);
    void   sanitizeNonbasic(Int k);   // repair a status against the current bounds
    Real   nonbasicValue(Int k) const;
    void   updatePrimalInfeasibility();
    Real   computeDualInfeasibility();

    Int    pricePrimal(int phase, const std::vector<Real>& d) const;
    bool   ratioTestPrimal(Int q, Int dir, const SparseVector& alpha, int phase,
                           Int& leavePos, Real& theta, bool& boundFlip, Real& leaveToUpper);
    Int    priceDual(Real& delta) const;
    bool   ratioTestDual(const SparseVector& row, Real delta, Int leavePos,
                         Int& enterIdx, Real& alphaPq, std::vector<Int>& flips);

    void   buildPhaseOneCost(std::vector<Real>& c1) const;
    void   makeDualFeasible();
    void   removeCostShifts();
    void   updateDevex(Int q, Int leavePos, const SparseVector& row, Real alphaPq);
    void   updateDse(Int leavePos, const SparseVector& alpha, Real alphaPq);
};

} // namespace igaos
