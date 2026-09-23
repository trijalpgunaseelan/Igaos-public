// nlp.cpp : primal-dual interior point method for smooth nonlinear programs.
//
// ===========================================================================
//      min  f(x)    s.t.  cl <= c(x) <= cu,   l <= x <= u
// ===========================================================================
//
//  Slacks turn the ranged constraints into equalities with bounded variables:
//
//      min f(x)  s.t.  c(x) - s = 0,  cl <= s <= cu,  l <= x <= u
//
//  and then every inequality in the problem is a simple bound, handled by a
//  logarithmic barrier.  Writing it this way is not cosmetic -- it means the
//  Newton system has exactly one shape no matter what mix of equalities,
//  ranges and one-sided rows the model has, which is the same reason the LP
//  side of this project puts every row in ranged form.
//
//  THE NEWTON SYSTEM.  Eliminating the bound multipliers through their
//  complementarity equations, and then eliminating ds (its block is diagonal),
//  leaves
//
//      [ -(W + Sx)   A' ] [ dx ]   [  rx                    ]
//      [    A       1/Ss ] [ dy ] = [ -(c - s) - rs/Ss       ]
//
//  which is QUASI-DEFINITE -- negative definite above, positive definite below.
//  That is the same matrix shape src/ipm.cpp hands to src/ldl.cpp for linear
//  and quadratic programs, and it is factorized by the same code with the same
//  AMD ordering and the same dynamic regularization.  The design note in
//  docs/EXTENSION-NLP-MINLP.md predicted exactly this: "a nonlinear objective
//  changes exactly one block -- Q becomes the Hessian at the current point".
//  W is that block, and nothing below it needed rewriting.
//
//  WHAT ACTUALLY MAKES IT WORK, and it is not the factorization:
//
//  1. INERTIA CORRECTION.  The Newton direction is only a descent direction
//     when the KKT matrix has n negative and m positive eigenvalues.  If it
//     does not -- which happens whenever W is indefinite, i.e. routinely -- a
//     multiple of the identity is added to the (1,1) block until it does.  The
//     factorization reports how many pivots it had to regularize, and a nonzero
//     count is exactly the signal that the inertia is wrong.
//
//  2. THE FILTER LINE SEARCH.  A merit function has to trade constraint
//     violation against objective with a penalty parameter nobody can choose
//     well.  A filter refuses that trade: a trial point is acceptable if it
//     improves EITHER the violation or the objective relative to every pair
//     already in the filter.  This is the part the design note called "the real
//     work", and it was right.
//
//  WHAT THIS DOES NOT DO.  It finds a LOCAL solution.  For a convex problem
//  that is the global one; for anything else it is a KKT point and no more, and
//  NlpResult::localOnly says so on every result it returns.  Global optimality
//  for the bilinear class is src/global.cpp, which is a different algorithm for
//  a reason.
// ===========================================================================
#include "igaos/expr.hpp"
#include "igaos/ldl.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <vector>

namespace igaos {

namespace {

struct Kkt {
    // Upper-triangle CSC of the (n+m) square KKT matrix, pattern fixed for the
    // whole solve so the symbolic analysis is done once.
    std::vector<Int>  Ap, Ai;
    std::vector<Real> Ax;
    std::vector<int8_t> sign;
    // Where each structural entry lives, so values can be refilled without
    // rebuilding the pattern.
    std::map<std::pair<Int, Int>, Int> slot;   // (row, col) -> index into Ax
};

inline Real safeDiv(Real a, Real b) { return a / std::max(b, 1e-300); }

} // namespace

NlpResult solveNlp(const NlpProblem& p, const NlpOptions& opt) {
    Timer clock;
    const Int n = p.numVar(), m = p.numCon();
    NlpResult res;
    res.x.assign((size_t)n, 0.0);
    res.conValue.assign((size_t)m, 0.0);
    res.conDual.assign((size_t)m, 0.0);
    if (n == 0) { res.status = Status::Optimal; return res; }

    const ExprTape& T = p.tape;

    // ---- sparsity, once ---------------------------------------------------
    std::vector<std::vector<Int>> conDeps((size_t)m);
    std::vector<uint8_t> conLinear((size_t)m, 0);
    for (Int i = 0; i < m; ++i) {
        T.dependencies(p.constraint[(size_t)i], conDeps[(size_t)i]);
        conLinear[(size_t)i] = T.isLinear(p.constraint[(size_t)i]) ? 1 : 0;
    }
    std::vector<Int> objDeps;
    T.dependencies(p.objective, objDeps);
    const bool objLinear = T.isLinear(p.objective);

    // Hessian pattern: the dense lower triangle of each nonlinear expression's
    // own dependency set, unioned, plus the full diagonal (which the barrier
    // fills whether or not the functions do).
    std::vector<std::pair<Int, Int>> hessPattern;
    {
        std::map<std::pair<Int, Int>, char> seen;
        auto blockOf = [&](const std::vector<Int>& s) {
            for (size_t a = 0; a < s.size(); ++a)
                for (size_t b = 0; b <= a; ++b)
                    seen[{s[a], s[b]}] = 1;                     // (row >= col)
        };
        if (!objLinear) blockOf(objDeps);
        for (Int i = 0; i < m; ++i) if (!conLinear[(size_t)i]) blockOf(conDeps[(size_t)i]);
        for (Int j = 0; j < n; ++j) seen[{j, j}] = 1;
        for (const auto& e : seen) hessPattern.push_back(e.first);
    }

    // ---- KKT pattern, once ------------------------------------------------
    Kkt K;
    {
        std::map<Int, std::vector<Int>> colRows;               // upper triangle
        auto put = [&](Int r, Int c) {
            if (r > c) std::swap(r, c);
            colRows[c].push_back(r);
        };
        for (const auto& e : hessPattern) put(e.first, e.second);
        for (Int i = 0; i < m; ++i) {
            for (Int j : conDeps[(size_t)i]) put(j, n + i);
            put(n + i, n + i);
        }
        K.Ap.assign((size_t)(n + m) + 1, 0);
        for (Int c = 0; c < n + m; ++c) {
            auto& v = colRows[c];
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            K.Ap[(size_t)c + 1] = K.Ap[(size_t)c] + (Int)v.size();
            for (Int r : v) { K.slot[{r, c}] = (Int)K.Ai.size(); K.Ai.push_back(r); }
        }
        K.Ax.assign(K.Ai.size(), 0.0);
        K.sign.assign((size_t)(n + m), 0);
        for (Int j = 0; j < n; ++j)     K.sign[(size_t)j] = -1;   // -(W + Sx)
        for (Int i = 0; i < m; ++i)     K.sign[(size_t)(n + i)] = +1;
    }
    LdlFactor ldl;
    ldl.analyze(n + m, K.Ap, K.Ai);

    // ---- starting point ---------------------------------------------------
    //
    // Every variable is pushed strictly inside its bounds: a barrier term is
    // infinite on the boundary, so a start that sits on one has no gradient to
    // work with.  The push is relative to the range so it means the same thing
    // on a variable measured in tonnes as on one measured in fractions.
    std::vector<Real> x((size_t)n), s((size_t)m), y((size_t)m, 0.0);
    auto pushInside = [&](Real v, Real lo, Real up) {
        const bool hl = isFinite(lo), hu = isFinite(up);
        if (!hl && !hu) return v;
        Real w = (hl && hu) ? std::max(1e-8, opt.boundPush * (up - lo)) : opt.boundPush;
        if (hl && hu && up - lo < 1e-12) return 0.5 * (lo + up);
        if (hl) v = std::max(v, lo + w);
        if (hu) v = std::min(v, up - w);
        if (hl && hu && v <= lo) v = 0.5 * (lo + up);
        return v;
    };
    for (Int j = 0; j < n; ++j) {
        Real v = (Int)p.start.size() == n ? p.start[(size_t)j]
               : (isFinite(p.lower[(size_t)j]) && isFinite(p.upper[(size_t)j])
                      ? 0.5 * (p.lower[(size_t)j] + p.upper[(size_t)j])
                      : (isFinite(p.lower[(size_t)j]) ? p.lower[(size_t)j] + 1.0
                      : (isFinite(p.upper[(size_t)j]) ? p.upper[(size_t)j] - 1.0 : 0.0)));
        x[(size_t)j] = pushInside(v, p.lower[(size_t)j], p.upper[(size_t)j]);
    }
    // An EQUALITY constraint's slack has a degenerate box: cl == cu, so there is
    // no interior to be in and no barrier to apply.  Treating it like any other
    // bounded slack puts the point exactly on both of its own bounds, makes the
    // barrier infinite, and makes the (2,2) block of the KKT matrix zero -- at
    // which point the factorization regularizes a pivot, the inertia test reads
    // that as a wrong inertia, and the correction loop climbs to 1e40 and gives
    // up.  That is precisely how HS6, HS7 and HS71 failed at iteration zero on
    // the first version of this file, and all three have an equality.
    //
    // So an equality's slack is FIXED, contributes no barrier term and no bound
    // multipliers, and its row of the condensed system carries 1/sigma = 0 --
    // which is the correct statement A dx = -(c - s), not an approximation of
    // one.
    std::vector<uint8_t> isEq((size_t)m, 0);
    for (Int i = 0; i < m; ++i) {
        const bool eq = isFinite(p.conLower[(size_t)i]) && isFinite(p.conUpper[(size_t)i]) &&
                        p.conUpper[(size_t)i] - p.conLower[(size_t)i] < 1e-12;
        isEq[(size_t)i] = eq ? 1 : 0;
        Real c = T.value(p.constraint[(size_t)i], x);
        s[(size_t)i] = eq ? p.conLower[(size_t)i]
                          : pushInside(c, p.conLower[(size_t)i], p.conUpper[(size_t)i]);
    }

    // Bound multipliers, one pair per variable and per slack.  Only the finite
    // sides carry one; an infinite bound has no barrier term and no multiplier.
    auto hasLo = [&](Real v) { return isFinite(v); };
    std::vector<Real> zL((size_t)n, 0.0), zU((size_t)n, 0.0);
    std::vector<Real> wL((size_t)m, 0.0), wU((size_t)m, 0.0);
    Real mu = opt.muInit;
    for (Int j = 0; j < n; ++j) {
        if (hasLo(p.lower[(size_t)j])) zL[(size_t)j] = 1.0;
        if (hasLo(p.upper[(size_t)j])) zU[(size_t)j] = 1.0;
    }
    for (Int i = 0; i < m; ++i) {
        if (isEq[(size_t)i]) continue;                     // no bound multipliers
        if (hasLo(p.conLower[(size_t)i])) wL[(size_t)i] = 1.0;
        if (hasLo(p.conUpper[(size_t)i])) wU[(size_t)i] = 1.0;
    }

    // ---- workspaces --------------------------------------------------------
    std::vector<Real> grad((size_t)n), cval((size_t)m), seed((size_t)n), hv((size_t)n);
    std::vector<Real> rhs((size_t)(n + m)), dx((size_t)n), dyh((size_t)m);
    std::vector<Real> hess(hessPattern.size(), 0.0);
    std::vector<std::vector<Real>> jac((size_t)m);

    auto evalAll = [&](const std::vector<Real>& xx) {
        std::fill(grad.begin(), grad.end(), 0.0);
        T.gradient(p.objective, xx, grad);
        for (Int i = 0; i < m; ++i) {
            cval[(size_t)i] = T.value(p.constraint[(size_t)i], xx);
            jac[(size_t)i].assign(conDeps[(size_t)i].size(), 0.0);
            std::vector<Real> g((size_t)n, 0.0);
            T.gradient(p.constraint[(size_t)i], xx, g);
            for (size_t k = 0; k < conDeps[(size_t)i].size(); ++k)
                jac[(size_t)i][k] = g[(size_t)conDeps[(size_t)i][k]];
        }
    };

    // Hessian of the Lagrangian, by exact Hessian-vector products on unit seeds
    // restricted to each expression's own variables.
    auto buildHessian = [&](const std::vector<Real>& xx, const std::vector<Real>& yy) {
        std::map<std::pair<Int, Int>, Real> acc;
        auto addBlock = [&](Int root, const std::vector<Int>& deps, Real scale) {
            for (Int col : deps) {
                std::fill(seed.begin(), seed.end(), 0.0);
                std::fill(hv.begin(), hv.end(), 0.0);
                seed[(size_t)col] = 1.0;
                T.hessianVector(root, xx, seed, hv, scale);
                for (Int row : deps)
                    if (row >= col && hv[(size_t)row] != 0.0)
                        acc[{row, col}] += hv[(size_t)row];
            }
        };
        if (!objLinear) addBlock(p.objective, objDeps, 1.0);
        for (Int i = 0; i < m; ++i)
            if (!conLinear[(size_t)i] && yy[(size_t)i] != 0.0)
                addBlock(p.constraint[(size_t)i], conDeps[(size_t)i], yy[(size_t)i]);
        for (size_t k = 0; k < hessPattern.size(); ++k) {
            auto it = acc.find(hessPattern[k]);
            hess[k] = (it == acc.end()) ? 0.0 : it->second;
        }
    };

    auto barrier = [&](const std::vector<Real>& xx, const std::vector<Real>& ss, Real m_) {
        Real phi = T.value(p.objective, xx);
        for (Int j = 0; j < n; ++j) {
            if (isFinite(p.lower[(size_t)j])) phi -= m_ * std::log(std::max(xx[(size_t)j] - p.lower[(size_t)j], 1e-300));
            if (isFinite(p.upper[(size_t)j])) phi -= m_ * std::log(std::max(p.upper[(size_t)j] - xx[(size_t)j], 1e-300));
        }
        for (Int i = 0; i < m; ++i) {
            if (isEq[(size_t)i]) continue;
            if (isFinite(p.conLower[(size_t)i])) phi -= m_ * std::log(std::max(ss[(size_t)i] - p.conLower[(size_t)i], 1e-300));
            if (isFinite(p.conUpper[(size_t)i])) phi -= m_ * std::log(std::max(p.conUpper[(size_t)i] - ss[(size_t)i], 1e-300));
        }
        return phi;
    };
    auto theta = [&](const std::vector<Real>& xx, const std::vector<Real>& ss) {
        Real t = 0.0;
        for (Int i = 0; i < m; ++i)
            t += std::fabs(T.value(p.constraint[(size_t)i], xx) - ss[(size_t)i]);
        return t;
    };

    // ---- the filter --------------------------------------------------------
    std::vector<std::pair<Real, Real>> filter;      // (theta, phi) pairs
    const Real gammaTheta = 1e-5, gammaPhi = 1e-5;
    const Real sTheta = 1.1, sPhi = 2.3, etaPhi = 1e-4;
    Real theta0 = std::max(theta(x, s), 1e-8);
    const Real thetaMax = 1e4 * std::max(1.0, theta0);
    const Real thetaMin = 1e-4 * std::max(1.0, theta0);

    Real deltaW = 0.0, deltaWLast = 0.0;
    Int iter = 0;
    Status status = Status::IterationLimit;

    for (iter = 0; iter < opt.maxIter; ++iter) {
        if (clock.elapsed() > opt.timeLimit) { status = Status::TimeLimit; break; }
        evalAll(x);

        // ---- KKT residuals -------------------------------------------------
        std::vector<Real> rx((size_t)n, 0.0);
        for (Int j = 0; j < n; ++j) rx[(size_t)j] = grad[(size_t)j] - zL[(size_t)j] + zU[(size_t)j];
        for (Int i = 0; i < m; ++i)
            for (size_t k = 0; k < conDeps[(size_t)i].size(); ++k)
                rx[(size_t)conDeps[(size_t)i][k]] += y[(size_t)i] * jac[(size_t)i][k];

        Real dualInf = 0.0, primInf = 0.0, compl_ = 0.0;
        for (Int j = 0; j < n; ++j) dualInf = std::max(dualInf, std::fabs(rx[(size_t)j]));
        for (Int i = 0; i < m; ++i) {
            if (!isEq[(size_t)i])
                dualInf = std::max(dualInf, std::fabs(-y[(size_t)i] - wL[(size_t)i] + wU[(size_t)i]));
            primInf = std::max(primInf, std::fabs(cval[(size_t)i] - s[(size_t)i]));
        }
        for (Int j = 0; j < n; ++j) {
            if (isFinite(p.lower[(size_t)j]))
                compl_ = std::max(compl_, std::fabs((x[(size_t)j] - p.lower[(size_t)j]) * zL[(size_t)j]));
            if (isFinite(p.upper[(size_t)j]))
                compl_ = std::max(compl_, std::fabs((p.upper[(size_t)j] - x[(size_t)j]) * zU[(size_t)j]));
        }
        for (Int i = 0; i < m; ++i) {
            if (isEq[(size_t)i]) continue;
            if (isFinite(p.conLower[(size_t)i]))
                compl_ = std::max(compl_, std::fabs((s[(size_t)i] - p.conLower[(size_t)i]) * wL[(size_t)i]));
            if (isFinite(p.conUpper[(size_t)i]))
                compl_ = std::max(compl_, std::fabs((p.conUpper[(size_t)i] - s[(size_t)i]) * wU[(size_t)i]));
        }

        // Scaled stationarity, so a problem with large multipliers is not held
        // to an absolute standard it can never meet.
        Real sumMult = 0.0;
        for (Int i = 0; i < m; ++i) sumMult += std::fabs(y[(size_t)i]);
        for (Int j = 0; j < n; ++j) sumMult += zL[(size_t)j] + zU[(size_t)j];
        const Real sd = std::max(1.0, sumMult / std::max<Int>(1, n + m) / 100.0);

        if (std::max(dualInf / sd, std::max(primInf, compl_)) <= opt.tolerance) {
            status = Status::Optimal; break;
        }
        if (std::max(dualInf / sd, std::max(primInf, compl_ - mu)) <= 10.0 * mu) {
            mu = std::max(opt.tolerance / 10.0, std::min(0.2 * mu, std::pow(mu, 1.5)));
            filter.clear();
            continue;
        }

        // ---- barrier gradient and the Sigma diagonals ----------------------
        std::vector<Real> phiX((size_t)n), sigX((size_t)n, 0.0);
        for (Int j = 0; j < n; ++j) {
            Real g = grad[(size_t)j], sg = 0.0;
            if (isFinite(p.lower[(size_t)j])) {
                Real d = std::max(x[(size_t)j] - p.lower[(size_t)j], 1e-12);
                g -= mu / d; sg += zL[(size_t)j] / d;
            }
            if (isFinite(p.upper[(size_t)j])) {
                Real d = std::max(p.upper[(size_t)j] - x[(size_t)j], 1e-12);
                g += mu / d; sg += zU[(size_t)j] / d;
            }
            phiX[(size_t)j] = g; sigX[(size_t)j] = sg;
        }
        // invSigS is 1/sigma for an inequality and ZERO for an equality, so the
        // equality's row of the condensed system is exactly A dx = -(c - s).
        // The small deltaC added below is what keeps that zero pivot factorable;
        // it shrinks with mu so the equality is satisfied to the tolerance being
        // asked for rather than to a fixed fudge.
        std::vector<Real> phiS((size_t)m, 0.0), invSigS((size_t)m, 0.0);
        for (Int i = 0; i < m; ++i) {
            if (isEq[(size_t)i]) { phiS[(size_t)i] = 0.0; invSigS[(size_t)i] = 0.0; continue; }
            Real g = 0.0, sg = 0.0;
            if (isFinite(p.conLower[(size_t)i])) {
                Real d = std::max(s[(size_t)i] - p.conLower[(size_t)i], 1e-12);
                g -= mu / d; sg += wL[(size_t)i] / d;
            }
            if (isFinite(p.conUpper[(size_t)i])) {
                Real d = std::max(p.conUpper[(size_t)i] - s[(size_t)i], 1e-12);
                g += mu / d; sg += wU[(size_t)i] / d;
            }
            phiS[(size_t)i] = g;
            invSigS[(size_t)i] = 1.0 / std::max(sg, 1e-12);
        }
        const Real deltaCbase = 1e-8 * std::pow(std::max(mu, 1e-12), 0.25);

        buildHessian(x, y);

        // ---- assemble, factor, correct the inertia -------------------------
        //
        // The direction is a descent direction only when the factorization has
        // the right inertia.  A regularized pivot is the factorization telling
        // us it did not, so deltaW climbs until the count comes back zero.
        bool factored = false;
        const Real deltaC = deltaCbase;
        deltaW = 0.0;
        for (int attempt = 0; attempt < 40; ++attempt) {
            std::fill(K.Ax.begin(), K.Ax.end(), 0.0);
            for (size_t k = 0; k < hessPattern.size(); ++k) {
                Int r = hessPattern[k].first, c = hessPattern[k].second;
                Int rr = std::min(r, c), cc = std::max(r, c);
                K.Ax[(size_t)K.slot[{rr, cc}]] -= hess[k];        // -(W)
            }
            for (Int j = 0; j < n; ++j)
                K.Ax[(size_t)K.slot[{j, j}]] -= (sigX[(size_t)j] + deltaW);
            for (Int i = 0; i < m; ++i) {
                for (size_t k = 0; k < conDeps[(size_t)i].size(); ++k)
                    K.Ax[(size_t)K.slot[{conDeps[(size_t)i][k], n + i}]] +=
                        jac[(size_t)i][k];
                K.Ax[(size_t)K.slot[{n + i, n + i}]] = invSigS[(size_t)i] + deltaC;
            }
            ldl.factorize(K.Ap, K.Ai, K.Ax, K.sign, 1e-10, 1e-11);
            if (ldl.numRegularized() == 0) { factored = true; break; }
            deltaW = (deltaW == 0.0) ? (deltaWLast > 0.0 ? std::max(1e-20, deltaWLast / 3.0)
                                                         : 1e-4)
                                     : deltaW * (deltaWLast > 0.0 ? 8.0 : 100.0);
            if (deltaW > 1e40) break;
        }
        // If the correction loop ran out, the last factorization is heavily
        // regularized and its direction is probably poor -- but a poor direction
        // is the line search's problem, not a reason to abandon the solve.  It
        // will be rejected, mu will be loosened, and the method gets another
        // chance.  Only a stall at the smallest mu ends the run.
        (void)factored;
        if (deltaW > 0.0) deltaWLast = deltaW;

        // ---- right-hand side and solve -------------------------------------
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (Int j = 0; j < n; ++j) rhs[(size_t)j] = phiX[(size_t)j];
        for (Int i = 0; i < m; ++i)
            for (size_t k = 0; k < conDeps[(size_t)i].size(); ++k)
                rhs[(size_t)conDeps[(size_t)i][k]] += y[(size_t)i] * jac[(size_t)i][k];
        for (Int i = 0; i < m; ++i) {
            const Real rs = phiS[(size_t)i] - y[(size_t)i];
            rhs[(size_t)(n + i)] = -(cval[(size_t)i] - s[(size_t)i])
                                 - (isEq[(size_t)i] ? 0.0 : rs * invSigS[(size_t)i]);
        }
        ldl.solve(rhs);
        for (Int j = 0; j < n; ++j) dx[(size_t)j] = rhs[(size_t)j];
        for (Int i = 0; i < m; ++i) dyh[(size_t)i] = rhs[(size_t)(n + i)];

        std::vector<Real> dy((size_t)m), ds((size_t)m);
        for (Int i = 0; i < m; ++i) {
            dy[(size_t)i] = -dyh[(size_t)i];
            const Real rs = phiS[(size_t)i] - y[(size_t)i];
            // An equality's slack does not move: it is pinned at the constraint
            // value and the whole step is taken by x.
            ds[(size_t)i] = isEq[(size_t)i] ? 0.0 : (dy[(size_t)i] - rs) * invSigS[(size_t)i];
        }

        // ---- bound multiplier steps ---------------------------------------
        std::vector<Real> dzL((size_t)n, 0.0), dzU((size_t)n, 0.0);
        std::vector<Real> dwL((size_t)m, 0.0), dwU((size_t)m, 0.0);
        for (Int j = 0; j < n; ++j) {
            if (isFinite(p.lower[(size_t)j])) {
                Real d = std::max(x[(size_t)j] - p.lower[(size_t)j], 1e-12);
                dzL[(size_t)j] = mu / d - zL[(size_t)j] - zL[(size_t)j] * dx[(size_t)j] / d;
            }
            if (isFinite(p.upper[(size_t)j])) {
                Real d = std::max(p.upper[(size_t)j] - x[(size_t)j], 1e-12);
                dzU[(size_t)j] = mu / d - zU[(size_t)j] + zU[(size_t)j] * dx[(size_t)j] / d;
            }
        }
        for (Int i = 0; i < m; ++i) {
            if (isEq[(size_t)i]) continue;
            if (isFinite(p.conLower[(size_t)i])) {
                Real d = std::max(s[(size_t)i] - p.conLower[(size_t)i], 1e-12);
                dwL[(size_t)i] = mu / d - wL[(size_t)i] - wL[(size_t)i] * ds[(size_t)i] / d;
            }
            if (isFinite(p.conUpper[(size_t)i])) {
                Real d = std::max(p.conUpper[(size_t)i] - s[(size_t)i], 1e-12);
                dwU[(size_t)i] = mu / d - wU[(size_t)i] + wU[(size_t)i] * ds[(size_t)i] / d;
            }
        }

        // ---- fraction to the boundary --------------------------------------
        const Real tau = std::max(0.99, 1.0 - mu);
        Real aMax = 1.0;
        auto capPrimal = [&](Real v, Real d, Real lo, Real up) {
            if (d < 0 && isFinite(lo)) aMax = std::min(aMax, -tau * (v - lo) / d);
            if (d > 0 && isFinite(up)) aMax = std::min(aMax,  tau * (up - v) / d);
        };
        for (Int j = 0; j < n; ++j)
            capPrimal(x[(size_t)j], dx[(size_t)j], p.lower[(size_t)j], p.upper[(size_t)j]);
        for (Int i = 0; i < m; ++i)
            if (!isEq[(size_t)i])
                capPrimal(s[(size_t)i], ds[(size_t)i], p.conLower[(size_t)i], p.conUpper[(size_t)i]);
        Real aDual = 1.0;
        auto capDual = [&](Real v, Real d) { if (d < 0) aDual = std::min(aDual, -tau * v / d); };
        for (Int j = 0; j < n; ++j) { capDual(zL[(size_t)j], dzL[(size_t)j]);
                                      capDual(zU[(size_t)j], dzU[(size_t)j]); }
        for (Int i = 0; i < m; ++i)
            if (!isEq[(size_t)i]) { capDual(wL[(size_t)i], dwL[(size_t)i]);
                                    capDual(wU[(size_t)i], dwU[(size_t)i]); }
        aMax = std::max(aMax, 1e-12);
        aDual = std::max(aDual, 1e-12);

        // ---- filter line search --------------------------------------------
        const Real th0 = theta(x, s), ph0 = barrier(x, s, mu);
        Real dphi = 0.0;
        for (Int j = 0; j < n; ++j) dphi += phiX[(size_t)j] * dx[(size_t)j];
        for (Int i = 0; i < m; ++i) dphi += phiS[(size_t)i] * ds[(size_t)i];

        Real alpha = aMax;
        bool accepted = false;
        std::vector<Real> xt((size_t)n), st((size_t)m);
        for (int ls = 0; ls < 30; ++ls) {
            for (Int j = 0; j < n; ++j) xt[(size_t)j] = x[(size_t)j] + alpha * dx[(size_t)j];
            for (Int i = 0; i < m; ++i) st[(size_t)i] = s[(size_t)i] + alpha * ds[(size_t)i];
            const Real th = theta(xt, st), ph = barrier(xt, st, mu);
            if (!std::isfinite(th) || !std::isfinite(ph)) { alpha *= 0.5; continue; }
            if (th > thetaMax) { alpha *= 0.5; continue; }

            bool blocked = false;
            for (const auto& f : filter)
                if (th >= f.first - 1e-12 && ph >= f.second - 1e-12) { blocked = true; break; }
            if (blocked) { alpha *= 0.5; continue; }

            // The switching condition: when the step promises a real objective
            // decrease and the violation is already small, demand Armijo on the
            // objective.  Otherwise accept progress on either measure.
            const bool switching = (dphi < 0.0) &&
                (alpha * std::pow(-dphi, sPhi) > 1.0 * std::pow(th0, sTheta)) &&
                (th0 <= thetaMin);
            bool ok;
            if (switching) ok = (ph <= ph0 + etaPhi * alpha * dphi);
            else ok = (th <= (1.0 - gammaTheta) * th0) || (ph <= ph0 - gammaPhi * th0);

            if (ok) {
                if (!switching) filter.emplace_back((1.0 - gammaTheta) * th0,
                                                    ph0 - gammaPhi * th0);
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if (!accepted) {
            // No acceptable step.  Loosening the barrier is the standard escape
            // and it usually works; if mu is already at its floor the method has
            // genuinely stalled and says so rather than iterating pointlessly.
            if (mu <= opt.tolerance / 9.0) { status = Status::NumericalError; break; }
            mu = std::max(opt.tolerance / 10.0, 0.1 * mu);
            filter.clear();
            continue;
        }

        // ---- accept ---------------------------------------------------------
        for (Int j = 0; j < n; ++j) x[(size_t)j] = xt[(size_t)j];
        for (Int i = 0; i < m; ++i) s[(size_t)i] = st[(size_t)i];
        for (Int i = 0; i < m; ++i) y[(size_t)i] += aDual * dy[(size_t)i];
        // Multipliers are kept in a box around mu/distance.  Without this they
        // drift by orders of magnitude on a badly scaled problem and take the
        // Sigma diagonals -- and therefore the whole KKT matrix -- with them.
        auto clampMult = [&](Real& z, Real dz, Real dist) {
            z += aDual * dz;
            const Real lo = mu / (1e3 * std::max(dist, 1e-12));
            const Real hi = 1e3 * mu / std::max(dist, 1e-12);
            z = std::min(std::max(z, std::min(lo, hi)), std::max(lo, hi));
            z = std::max(z, 1e-12);
        };
        for (Int j = 0; j < n; ++j) {
            if (isFinite(p.lower[(size_t)j]))
                clampMult(zL[(size_t)j], dzL[(size_t)j], x[(size_t)j] - p.lower[(size_t)j]);
            if (isFinite(p.upper[(size_t)j]))
                clampMult(zU[(size_t)j], dzU[(size_t)j], p.upper[(size_t)j] - x[(size_t)j]);
        }
        for (Int i = 0; i < m; ++i) {
            if (isEq[(size_t)i]) continue;
            if (isFinite(p.conLower[(size_t)i]))
                clampMult(wL[(size_t)i], dwL[(size_t)i], s[(size_t)i] - p.conLower[(size_t)i]);
            if (isFinite(p.conUpper[(size_t)i]))
                clampMult(wU[(size_t)i], dwU[(size_t)i], p.conUpper[(size_t)i] - s[(size_t)i]);
        }

        if (opt.verbosity >= 2)
            std::printf("   nlp %3d  f %14.8g  theta %10.3e  mu %8.2e  alpha %6.4f  "
                        "dw %8.2e\n", (int)iter, (double)T.value(p.objective, x),
                        (double)theta(x, s), (double)mu, (double)alpha, (double)deltaW);
    }

    // ---- report ------------------------------------------------------------
    res.status = status;
    res.x = x;
    res.iterations = iter;
    res.objective = T.value(p.objective, x);
    res.localOnly = true;
    Real pi = 0.0;
    for (Int i = 0; i < m; ++i) {
        res.conValue[(size_t)i] = T.value(p.constraint[(size_t)i], x);
        res.conDual[(size_t)i] = y[(size_t)i];
        pi = std::max(pi, p.conLower[(size_t)i] - res.conValue[(size_t)i]);
        pi = std::max(pi, res.conValue[(size_t)i] - p.conUpper[(size_t)i]);
    }
    for (Int j = 0; j < n; ++j) {
        pi = std::max(pi, p.lower[(size_t)j] - x[(size_t)j]);
        pi = std::max(pi, x[(size_t)j] - p.upper[(size_t)j]);
    }
    res.primalInf = std::max(pi, 0.0);
    return res;
}

// ===========================================================================
//  MINLP by branch and bound over NLP relaxations.
//
//  The tree itself is ordinary: solve the relaxation, pick a fractional integer
//  variable, split its box, repeat.  What is NOT ordinary is what the bound is
//  worth.  For a convex model the relaxation's optimum is a genuine lower bound
//  and this search is a proof; for a nonconvex one it is a local solution and
//  therefore no bound at all, and a node pruned against it may have held the
//  answer.  Since convexity of a general expression graph is not something this
//  code can decide, it assumes the worse case and never reports Optimal.
//
//  Saying that plainly costs a stronger-sounding claim and buys the only thing
//  that matters: nobody reading the status is misled about what was proved.
// ===========================================================================
MinlpResult solveMinlp(const NlpProblem& p, const std::vector<Int>& integerVars,
                       const NlpOptions& opt) {
    Timer clock;
    MinlpResult best;
    best.status = Status::NotSolved;
    best.localOnly = true;
    best.provenGlobal = false;

    const Int n = p.numVar();
    if (integerVars.empty()) {
        NlpResult r = solveNlp(p, opt);
        static_cast<NlpResult&>(best) = r;
        best.nodes = 1;
        return best;
    }

    struct Node { std::vector<Real> lo, up; Real bound; int depth; };
    std::vector<Node> open;
    open.push_back(Node{p.lower, p.upper, -kInf, 0});

    Real incumbent = kInf;
    bool have = false;
    Long nodes = 0;
    bool exhausted = true;

    while (!open.empty()) {
        if (clock.elapsed() > opt.timeLimit || nodes > 20000) { exhausted = false; break; }
        // Depth first: an NLP relaxation has no warm start to inherit, so the
        // only thing a node gains from its parent is a tighter box -- and going
        // deep gets to an integral point, and therefore to an incumbent that
        // prunes, soonest.
        Node node = open.back();
        open.pop_back();
        ++nodes;
        if (have && node.bound > incumbent - 1e-9) continue;

        NlpProblem sub = p;
        sub.lower = node.lo;
        sub.upper = node.up;
        NlpOptions so = opt;
        so.timeLimit = std::max(0.05, opt.timeLimit - clock.elapsed());
        NlpResult r = solveNlp(sub, so);
        if (r.status != Status::Optimal || r.primalInf > 1e-6) {
            // Could not bound this node.  Not a proof of anything, so the run
            // stops being able to claim it searched everywhere.
            if (r.status != Status::Infeasible) exhausted = false;
            continue;
        }
        if (have && r.objective > incumbent - 1e-9) continue;

        // most fractional integer variable
        Int branch = kNone;
        Real worst = 1e-6;
        for (Int j : integerVars) {
            const Real v = r.x[(size_t)j];
            const Real f = std::fabs(v - std::floor(v + 0.5));
            if (f > worst) { worst = f; branch = j; }
        }
        if (branch == kNone) {
            if (!have || r.objective < incumbent - 1e-12) {
                incumbent = r.objective;
                have = true;
                static_cast<NlpResult&>(best) = r;
                best.status = Status::Feasible;
            }
            continue;
        }
        if (node.depth > 60) { exhausted = false; continue; }

        const Real v = r.x[(size_t)branch];
        Node lo = node, hi = node;
        lo.up[(size_t)branch] = std::floor(v);
        hi.lo[(size_t)branch] = std::ceil(v);
        lo.bound = hi.bound = r.objective;
        lo.depth = hi.depth = node.depth + 1;
        if (lo.up[(size_t)branch] >= lo.lo[(size_t)branch] - 1e-9) open.push_back(lo);
        if (hi.lo[(size_t)branch] <= hi.up[(size_t)branch] + 1e-9) open.push_back(hi);
    }

    best.nodes = nodes;
    best.provenGlobal = false;          // see the header comment; never claimed
    if (!have) best.status = exhausted ? Status::Infeasible : Status::TimeLimit;
    (void)n;
    return best;
}

} // namespace igaos
