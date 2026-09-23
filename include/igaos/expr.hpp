// expr.hpp : expression graphs with exact first and second derivatives.
//
// ===========================================================================
//  WHY THIS EXISTS, AND WHY IT IS NOT AN "AD LIBRARY"
// ===========================================================================
//  A nonlinear program needs three things from every function in it: its value,
//  its gradient, and its Hessian.  Finite differences give the first two badly
//  and the third uselessly -- the error in a differenced Hessian is O(sqrt(eps))
//  per entry, which is 1e-8 on a quantity the interior point method needs to
//  factor, and the resulting matrix is not even symmetric.  So the derivatives
//  here are EXACT: same arithmetic, same rounding, no step size to choose.
//
//  The mechanism is a tape.  An expression is built once into a flat array of
//  nodes, and every later evaluation walks that array:
//
//    value            one forward pass
//    gradient         one forward pass, then one reverse pass  (reverse mode:
//                     the whole gradient costs a small multiple of one value,
//                     independent of how many variables there are)
//    Hessian-vector   forward-over-reverse -- the forward pass carries a
//                     tangent alongside each value and the reverse pass carries
//                     a tangent alongside each adjoint, so H*p comes out of one
//                     sweep with no differencing anywhere
//
//  A full sparse Hessian is then |S| Hessian-vector products over the
//  expression's own dependency set S, which is what makes this affordable on a
//  process model: each constraint touches a handful of variables even when the
//  model has thousands.
//
//  It is deliberately small.  The earlier design note in
//  docs/EXTENSION-NLP-MINLP.md argued for a callback interface instead, on the
//  grounds that "writing an AD engine is a separate project".  That was true of
//  a general AD engine and false of this: a fixed set of thirteen operations,
//  each with a known first and second derivative, is a table -- and the table
//  is the file.  What it does NOT do is differentiate arbitrary user code,
//  handle non-smooth functions, or detect that a subexpression repeats.
// ===========================================================================
#pragma once
#include "igaos/common.hpp"
#include <vector>

namespace igaos {

enum class Op : uint8_t {
    Const,      // value
    Var,        // index
    Add, Sub, Mul, Div,
    Neg,
    Pow,        // a ^ value    (value is a real exponent, not a node)
    Exp, Log, Sqrt, Sin, Cos,
    Square      // a * a, kept separate because it is by far the most common
};

// One tape shared by the objective and every constraint of a problem.  Nodes
// are append-only and reference earlier nodes only, so a forward pass is a
// single sweep from 0 upward and a reverse pass a single sweep downward.
class ExprTape {
public:
    struct Node {
        Op   op = Op::Const;
        Int  a = kNone, b = kNone;
        Real value = 0.0;          // constant, exponent, or variable index
    };

    Int numNode() const { return (Int)node_.size(); }
    Int numVar()  const { return nvar_; }
    void setNumVar(Int n) { nvar_ = std::max(nvar_, n); }

    // ---- construction ------------------------------------------------------
    Int constant(Real v);
    Int variable(Int j);
    Int add(Int a, Int b);
    Int sub(Int a, Int b);
    Int mul(Int a, Int b);
    Int div(Int a, Int b);
    Int neg(Int a);
    Int pow(Int a, Real p);
    Int square(Int a);
    Int exp(Int a);
    Int log(Int a);
    Int sqrt(Int a);
    Int sin(Int a);
    Int cos(Int a);
    // Convenience: sum_k coef_k * x_{col_k} + c, as one subtree.
    Int linear(const std::vector<std::pair<Int, Real>>& terms, Real c = 0.0);

    // ---- evaluation --------------------------------------------------------
    Real value(Int root, const std::vector<Real>& x) const;

    // Adds  scale * grad(root)  into g.  g must be sized numVar().
    void gradient(Int root, const std::vector<Real>& x, std::vector<Real>& g,
                  Real scale = 1.0) const;

    // Adds  scale * Hessian(root) * p  into hv.  Exact; no differencing.
    void hessianVector(Int root, const std::vector<Real>& x,
                       const std::vector<Real>& p, std::vector<Real>& hv,
                       Real scale = 1.0) const;

    // The variables `root` actually depends on, ascending and deduplicated.
    // This is the sparsity pattern, and it is what keeps the Hessian cheap.
    void dependencies(Int root, std::vector<Int>& out) const;

    // True when the subtree is linear in the variables -- no products of two
    // non-constant subtrees and no transcendental of one.  A linear constraint
    // contributes nothing to the Hessian of the Lagrangian, and knowing that in
    // advance removes it from the work entirely.
    bool isLinear(Int root) const;

private:
    Int push(Op op, Int a, Int b, Real v);
    std::vector<Node> node_;
    Int nvar_ = 0;
    // Scratch reused across calls so an evaluation allocates nothing.
    mutable std::vector<Real> v_, dv_, adj_, dadj_;
    mutable std::vector<uint8_t> live_;
};

// ---------------------------------------------------------------------------
// A nonlinear program:
//
//      min  f(x)   s.t.  cl <= c(x) <= cu,   l <= x <= u
//
// Bounds are separate from constraints on purpose: the interior point method
// treats a simple bound with a barrier term and a general constraint with a
// slack, and conflating them costs a row and a multiplier each.
// ---------------------------------------------------------------------------
struct NlpProblem {
    ExprTape tape;
    Int  objective = kNone;             // node index of f
    std::vector<Int>  constraint;       // node index of each c_i
    std::vector<Real> conLower, conUpper;
    std::vector<Real> lower, upper;     // variable bounds
    std::vector<Real> start;            // initial point; midpoint if empty
    std::vector<std::string> varName, conName;

    Int numVar() const { return (Int)lower.size(); }
    Int numCon() const { return (Int)constraint.size(); }

    Int addVariable(Real lo, Real up, const std::string& nm = "");
    Int addConstraint(Int node, Real lo, Real up, const std::string& nm = "");
};

struct NlpResult {
    Status status = Status::NotSolved;
    Real   objective = 0.0;
    std::vector<Real> x;
    std::vector<Real> conValue;
    std::vector<Real> conDual;          // multipliers for the general constraints
    Int    iterations = 0;
    Real   primalInf = 0.0;             // max constraint / bound violation
    Real   dualInf = 0.0;               // max violation of stationarity
    Real   complementarity = 0.0;
    // A LOCAL solution.  Stated in the struct because the word matters: unless
    // the problem is convex, "converged" means a KKT point was reached, not
    // that no better one exists.
    bool   localOnly = true;
};

struct NlpOptions {
    Real tolerance   = 1e-8;
    Int  maxIter     = 300;
    Real timeLimit   = 1e30;
    int  verbosity   = 0;
    Real muInit      = 0.1;
    Real boundPush   = 1e-2;      // how far inside its bounds a start is pushed
};

NlpResult solveNlp(const NlpProblem& p, const NlpOptions& opt = NlpOptions());

// ---------------------------------------------------------------------------
// MINLP: the same problem with some variables required integral.
//
// Branch and bound over NLP relaxations.  Read the guarantee carefully, because
// it is weaker than the one src/global.cpp gives and the difference is the
// whole point:
//
//   * If every constraint and the objective are CONVEX, the NLP relaxation of a
//     node is a valid lower bound, the search is exhaustive, and the answer is
//     the global optimum.  This solver cannot verify convexity for a general
//     expression, so it does not claim it.
//   * If they are not, a local NLP solution is NOT a bound.  A subtree pruned
//     against it may have contained the optimum.  The answer is then the best
//     point found, which is a useful thing and not a proof.
//
// So `status` is Feasible, never Optimal, and `provenGlobal` is false.  A model
// whose nonlinearity is quadratic or bilinear should go to Solver::solve
// instead, which routes it to the spatial branch and bound and DOES prove
// global optimality.
// ---------------------------------------------------------------------------
struct MinlpResult : NlpResult {
    Long nodes = 0;
    bool provenGlobal = false;
};

MinlpResult solveMinlp(const NlpProblem& p, const std::vector<Int>& integerVars,
                       const NlpOptions& opt = NlpOptions());

} // namespace igaos
