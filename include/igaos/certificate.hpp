#pragma once
// ===========================================================================
//  Certificates: making a wrong answer detectable by something that is not
//  this solver.
// ===========================================================================
//
//  Twenty-one defects have been found in this solver so far.  Four of them were
//  in code that passed its own tests, and two of those returned a WRONG ANSWER
//  rather than an error -- a feasible-looking number with a confident label on
//  it.  In a refinery that is the failure that costs money, because a bad
//  schedule is executed before anyone discovers it was bad.
//
//  Testing reduces how often that happens.  It cannot make it detectable.
//
//  A certificate can.  The theorem below is elementary and that is exactly why
//  it is useful: it holds for ANY vector y whatsoever, so a checker never has
//  to reproduce, trust, or even understand how the solver arrived at one.
//
//      For  min c'x  subject to  rl <= Ax <= ru,  cl <= x <= cu,
//      write  z = c - A'y.  Then for every y,
//
//          L(y) = SUM_j ( z_j >= 0 ? z_j*cl_j : z_j*cu_j )
//               + SUM_i ( y_i >= 0 ? y_i*rl_i : y_i*ru_i )
//
//      is a lower bound on the optimal objective.
//
//      Proof: c'x = (c - A'y)'x + y'(Ax) = z'x + y'(Ax), and each term is
//      minimised over its own box independently by the expression above.
//
//  So a certificate is just a pair (x, y).  A checker verifies that x is
//  feasible, computes L(y), and if L(y) equals c'x then x is optimal -- since
//  c'x >= optimum >= L(y) = c'x.  No convention about dual signs, no basis, no
//  simplex tableau, nothing that could be misunderstood in the same way twice.
//
//  The same object proves infeasibility.  Take c = 0, so z = -A'y.  If L(y) > 0
//  then every feasible x would give 0 = c'x >= L(y) > 0, which is impossible;
//  therefore no feasible x exists.  One structure, two theorems.
//
//  Values are written as C99 hexadecimal floats.  That is not decoration: it
//  makes the bits exact, so the checker can convert them to rationals and do
//  every arithmetic operation in exact arithmetic.  A checker that uses no
//  floating point cannot be fooled by the rounding that fooled the solver.
//
//  tools/verify_certificate.py is such a checker.  It shares no line of code
//  with this project -- its own MPS reader, its own arithmetic -- so agreement
//  between the two is evidence rather than tautology.
//
//  WHAT THIS DOES NOT YET DO: a mixed-integer certificate needs the branching
//  tree as well, so that a checker can confirm the search covered the space.
//  For a MIP this writes the incumbent, which certifies FEASIBILITY and an
//  upper bound exactly, plus the root dual bound.  Proving MIP optimality end
//  to end is the next piece of work, and it is named as such rather than
//  implied.
// ===========================================================================

#include "igaos/model.hpp"
#include <string>

namespace igaos {

// Write a certificate for `sol` against `model`.  Returns false and sets `err`
// if the file cannot be written.  Refuses to write for statuses that carry no
// provable claim (NotSolved, TimeLimit without an incumbent, and so on).
bool writeCertificate(const std::string& path, const Model& model,
                      const Solution& sol, std::string& err);

} // namespace igaos
