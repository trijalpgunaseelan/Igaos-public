// nlp_check.cpp : the expression graph against finite differences, and the NLP
// solver against the Hock-Schittkowski collection.
//
//     cd build && ctest -R nlp --output-on-failure
//
// ===========================================================================
//  TWO CHECKS, AND THE FIRST ONE IS THE FOUNDATION
// ===========================================================================
//  Everything the interior point method does rests on the derivatives being
//  right.  A wrong gradient does not crash: it produces a search direction that
//  is merely unhelpful, the line search shortens the step, and the method
//  crawls to something that is not a solution and reports it confidently.  So
//  the first half of this file differentiates random expressions and compares
//  against central differences -- which are a poor way to GET a derivative and
//  a perfectly good way to CHECK one, since the two share no code at all.
//
//  The second half is Hock and Schittkowski's collection (1981), the standard
//  test set for nonlinear programming, with the published optimal objective of
//  each problem written next to it.  These are small and they are nasty: HS5
//  has a trigonometric objective, HS7 an equality constraint with a quartic in
//  it, HS36 and HS37 have trilinear objectives, HS71 is the four-variable model
//  every interior point paper uses as its worked example.
//
//  Anything that does not converge is REPORTED, not dropped.  A test set is
//  only evidence if the failures are counted with the successes.
// ===========================================================================
#include "igaos/expr.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace igaos;

static int failures = 0;
static void check(bool ok, const char* what, double detail = 0.0) {
    if (ok) std::printf("  [ ok ] %s\n", what);
    else { std::printf("  [FAIL] %s  (%.6e)\n", what, detail); ++failures; }
}

// ===========================================================================
//  Part 1: derivatives against central differences.
// ===========================================================================
static void testDerivatives() {
    std::printf("expression graph -- exact derivatives against central differences\n");

    std::mt19937_64 rng(31337);
    std::uniform_real_distribution<double> u(0.4, 1.6);
    double worstG = 0.0, worstH = 0.0;
    int cases = 0;

    for (int trial = 0; trial < 40; ++trial) {
        ExprTape t;
        const Int n = 4;
        Int v[4];
        for (Int j = 0; j < n; ++j) v[(size_t)j] = t.variable(j);

        // A random expression that uses every operation the tape has.  The
        // variables are kept near 1 so log, sqrt and negative powers are all
        // defined -- the point here is the derivative table, not domain
        // handling, and mixing the two would make a failure ambiguous.
        Int e = t.constant(u(rng));
        for (int k = 0; k < 6; ++k) {
            Int a = v[(size_t)(rng() % 4)], b = v[(size_t)(rng() % 4)];
            Int piece;
            switch (rng() % 9) {
                case 0: piece = t.mul(a, b); break;
                case 1: piece = t.div(a, t.add(b, t.constant(2.0))); break;
                case 2: piece = t.exp(t.mul(t.constant(0.3), a)); break;
                case 3: piece = t.log(t.add(a, t.constant(2.0))); break;
                case 4: piece = t.sqrt(t.add(t.square(a), t.constant(1.0))); break;
                case 5: piece = t.sin(t.add(a, b)); break;
                case 6: piece = t.cos(t.mul(a, t.constant(0.7))); break;
                case 7: piece = t.pow(t.add(a, t.constant(1.5)), 2.5); break;
                default: piece = t.square(t.sub(a, b)); break;
            }
            e = t.add(e, t.mul(t.constant(u(rng)), piece));
        }

        std::vector<Real> x((size_t)n);
        for (Int j = 0; j < n; ++j) x[(size_t)j] = u(rng);

        // gradient
        std::vector<Real> g((size_t)n, 0.0);
        t.gradient(e, x, g);
        for (Int j = 0; j < n; ++j) {
            const Real h = 1e-6 * std::max(1.0, std::fabs(x[(size_t)j]));
            std::vector<Real> xp = x, xm = x;
            xp[(size_t)j] += h; xm[(size_t)j] -= h;
            const Real fd = (t.value(e, xp) - t.value(e, xm)) / (2.0 * h);
            worstG = std::max(worstG, std::fabs(fd - g[(size_t)j])
                                      / (1.0 + std::fabs(fd)));
        }

        // Hessian: H*p against the difference of gradients along p.
        std::vector<Real> p((size_t)n);
        for (Int j = 0; j < n; ++j) p[(size_t)j] = u(rng) - 1.0;
        std::vector<Real> hv((size_t)n, 0.0);
        t.hessianVector(e, x, p, hv);
        {
            const Real h = 1e-6;
            std::vector<Real> xp((size_t)n), xm((size_t)n), gp((size_t)n, 0.0), gm((size_t)n, 0.0);
            for (Int j = 0; j < n; ++j) {
                xp[(size_t)j] = x[(size_t)j] + h * p[(size_t)j];
                xm[(size_t)j] = x[(size_t)j] - h * p[(size_t)j];
            }
            t.gradient(e, xp, gp);
            t.gradient(e, xm, gm);
            for (Int j = 0; j < n; ++j) {
                const Real fd = (gp[(size_t)j] - gm[(size_t)j]) / (2.0 * h);
                worstH = std::max(worstH, std::fabs(fd - hv[(size_t)j])
                                          / (1.0 + std::fabs(fd)));
            }
        }
        ++cases;
    }
    // Central differences are accurate to about h^2 plus rounding/h, which at
    // h = 1e-6 is a few parts in 1e-9.  Anything larger is the derivative
    // table being wrong, not the differencing being imprecise.
    check(worstG < 1e-7, "gradients match central differences on every operation", worstG);
    check(worstH < 1e-5, "Hessian-vector products match differenced gradients", worstH);
    std::printf("         %d random expressions; worst gradient %.2e, worst Hessian %.2e\n",
                cases, worstG, worstH);

    // Linearity detection, which decides what enters the Hessian at all.
    {
        ExprTape t;
        Int a = t.variable(0), b = t.variable(1);
        check(t.isLinear(t.add(t.mul(t.constant(3.0), a), b)),
              "a linear combination is recognised as linear");
        check(!t.isLinear(t.mul(a, b)), "a product of two variables is not linear");
        check(!t.isLinear(t.exp(a)), "a transcendental of a variable is not linear");
        check(t.isLinear(t.div(a, t.constant(2.0))), "division by a constant stays linear");
    }
}

// ===========================================================================
//  Part 2: Hock-Schittkowski.
// ===========================================================================
struct HsCase {
    const char* name;
    double published;
    NlpProblem (*build)();
};

// HS1: Rosenbrock with one bound.  f* = 0 at (1,1).
static NlpProblem hs1() {
    NlpProblem p;
    p.addVariable(-kInf, kInf); p.addVariable(-1.5, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    Int r = t.sub(x2, t.square(x1));
    p.objective = t.add(t.mul(t.constant(100.0), t.square(r)),
                        t.square(t.sub(t.constant(1.0), x1)));
    p.start = {-2.0, 1.0};
    return p;
}

// HS3.  f* = 0.
static NlpProblem hs3() {
    NlpProblem p;
    p.addVariable(-kInf, kInf); p.addVariable(0.0, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    p.objective = t.add(x2, t.mul(t.constant(1e-5), t.square(t.sub(x2, x1))));
    p.start = {10.0, 1.0};
    return p;
}

// HS4: (x1+1)^3/3 + x2, x1 >= 1, x2 >= 0.  f* = 8/3.
static NlpProblem hs4() {
    NlpProblem p;
    p.addVariable(1.0, kInf); p.addVariable(0.0, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    p.objective = t.add(t.mul(t.constant(1.0 / 3.0),
                              t.pow(t.add(x1, t.constant(1.0)), 3.0)), x2);
    p.start = {1.125, 0.125};
    return p;
}

// HS5: a trigonometric objective.  f* = -1.9132229.
static NlpProblem hs5() {
    NlpProblem p;
    p.addVariable(-1.5, 4.0); p.addVariable(-3.0, 3.0);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    p.objective = t.add(t.add(t.sin(t.add(x1, x2)), t.square(t.sub(x1, x2))),
                        t.add(t.mul(t.constant(-1.5), x1),
                              t.add(t.mul(t.constant(2.5), x2), t.constant(1.0))));
    p.start = {0.0, 0.0};
    return p;
}

// HS6: equality constraint 10(x2 - x1^2) = 0.  f* = 0.
static NlpProblem hs6() {
    NlpProblem p;
    p.addVariable(-kInf, kInf); p.addVariable(-kInf, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    p.objective = t.square(t.sub(t.constant(1.0), x1));
    p.addConstraint(t.mul(t.constant(10.0), t.sub(x2, t.square(x1))), 0.0, 0.0);
    p.start = {-1.2, 1.0};
    return p;
}

// HS7: log(1+x1^2) - x2, with (1+x1^2)^2 + x2^2 = 4.  f* = -sqrt(3).
static NlpProblem hs7() {
    NlpProblem p;
    p.addVariable(-kInf, kInf); p.addVariable(-kInf, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    Int q = t.add(t.constant(1.0), t.square(x1));
    p.objective = t.sub(t.log(q), x2);
    p.addConstraint(t.add(t.square(q), t.square(x2)), 4.0, 4.0);
    p.start = {2.0, 2.0};
    return p;
}

// HS12: 25 - 4x1^2 - x2^2 >= 0.  f* = -30.
static NlpProblem hs12() {
    NlpProblem p;
    p.addVariable(-kInf, kInf); p.addVariable(-kInf, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    p.objective = t.add(t.sub(t.add(t.mul(t.constant(0.5), t.square(x1)), t.square(x2)),
                              t.mul(x1, x2)),
                        t.add(t.mul(t.constant(-7.0), x1), t.mul(t.constant(-7.0), x2)));
    p.addConstraint(t.sub(t.constant(25.0),
                          t.add(t.mul(t.constant(4.0), t.square(x1)), t.square(x2))),
                    0.0, kInf);
    p.start = {0.0, 0.0};
    return p;
}

// HS22.  f* = 1.
static NlpProblem hs22() {
    NlpProblem p;
    p.addVariable(-kInf, kInf); p.addVariable(-kInf, kInf);
    ExprTape& t = p.tape;
    Int x1 = t.variable(0), x2 = t.variable(1);
    p.objective = t.add(t.square(t.sub(x1, t.constant(2.0))),
                        t.square(t.sub(x2, t.constant(1.0))));
    p.addConstraint(t.sub(t.constant(2.0), t.add(x1, x2)), 0.0, kInf);
    p.addConstraint(t.sub(x2, t.square(x1)), 0.0, kInf);
    p.start = {2.0, 2.0};
    return p;
}

// HS35, Beale's problem.  f* = 1/9.
static NlpProblem hs35() {
    NlpProblem p;
    for (int k = 0; k < 3; ++k) p.addVariable(0.0, kInf);
    ExprTape& t = p.tape;
    Int a = t.variable(0), b = t.variable(1), c = t.variable(2);
    Int f = t.constant(9.0);
    f = t.add(f, t.mul(t.constant(-8.0), a));
    f = t.add(f, t.mul(t.constant(-6.0), b));
    f = t.add(f, t.mul(t.constant(-4.0), c));
    f = t.add(f, t.mul(t.constant(2.0), t.square(a)));
    f = t.add(f, t.mul(t.constant(2.0), t.square(b)));
    f = t.add(f, t.square(c));
    f = t.add(f, t.mul(t.constant(2.0), t.mul(a, b)));
    f = t.add(f, t.mul(t.constant(2.0), t.mul(a, c)));
    p.objective = f;
    p.addConstraint(t.sub(t.constant(3.0),
                          t.add(a, t.add(b, t.mul(t.constant(2.0), c)))), 0.0, kInf);
    p.start = {0.5, 0.5, 0.5};
    return p;
}

// HS36: -x1 x2 x3 with a linear constraint.  f* = -3300.
static NlpProblem hs36() {
    NlpProblem p;
    p.addVariable(0.0, 20.0); p.addVariable(0.0, 11.0); p.addVariable(0.0, 42.0);
    ExprTape& t = p.tape;
    Int a = t.variable(0), b = t.variable(1), c = t.variable(2);
    p.objective = t.neg(t.mul(a, t.mul(b, c)));
    p.addConstraint(t.sub(t.constant(72.0),
                          t.add(a, t.add(t.mul(t.constant(2.0), b),
                                         t.mul(t.constant(2.0), c)))), 0.0, kInf);
    p.start = {10.0, 10.0, 10.0};
    return p;
}

// HS71, the worked example of the interior point literature.  f* = 17.0140173.
static NlpProblem hs71() {
    NlpProblem p;
    for (int k = 0; k < 4; ++k) p.addVariable(1.0, 5.0);
    ExprTape& t = p.tape;
    Int a = t.variable(0), b = t.variable(1), c = t.variable(2), d = t.variable(3);
    p.objective = t.add(t.mul(t.mul(a, d), t.add(a, t.add(b, c))), c);
    p.addConstraint(t.mul(t.mul(a, b), t.mul(c, d)), 25.0, kInf);
    p.addConstraint(t.add(t.add(t.square(a), t.square(b)),
                          t.add(t.square(c), t.square(d))), 40.0, 40.0);
    p.start = {1.0, 5.0, 5.0, 1.0};
    return p;
}

// HS76.  f* = -4.681818181.
static NlpProblem hs76() {
    NlpProblem p;
    for (int k = 0; k < 4; ++k) p.addVariable(0.0, kInf);
    ExprTape& t = p.tape;
    Int a = t.variable(0), b = t.variable(1), c = t.variable(2), d = t.variable(3);
    Int f = t.square(a);
    f = t.add(f, t.mul(t.constant(0.5), t.square(b)));
    f = t.add(f, t.square(c));
    f = t.add(f, t.mul(t.constant(0.5), t.square(d)));
    f = t.sub(f, t.mul(a, c));
    f = t.add(f, t.mul(c, d));
    f = t.add(f, t.mul(t.constant(-1.0), a));
    f = t.add(f, t.mul(t.constant(-3.0), b));
    f = t.add(f, c);
    f = t.add(f, t.mul(t.constant(-1.0), d));
    p.objective = f;
    p.addConstraint(t.sub(t.constant(5.0),
                          t.add(a, t.add(t.mul(t.constant(2.0), b),
                                t.add(c, d)))), 0.0, kInf);
    p.addConstraint(t.sub(t.constant(4.0),
                          t.add(t.mul(t.constant(3.0), a),
                                t.add(b, t.add(t.mul(t.constant(2.0), c),
                                               t.mul(t.constant(-1.0), d))))), 0.0, kInf);
    p.addConstraint(t.sub(t.add(b, t.mul(t.constant(4.0), c)), t.constant(1.5)),
                    0.0, kInf);
    p.start = {0.5, 0.5, 0.5, 0.5};
    return p;
}

static void testHockSchittkowski() {
    std::printf("\nHock-Schittkowski, against the published optimal objectives\n");
    const HsCase cases[] = {
        {"HS1",   0.0,           hs1},
        {"HS3",   0.0,           hs3},
        {"HS4",   8.0 / 3.0,     hs4},
        {"HS5",  -1.91322295,    hs5},
        {"HS6",   0.0,           hs6},
        {"HS7",  -1.73205081,    hs7},
        {"HS12", -30.0,          hs12},
        {"HS22",  1.0,           hs22},
        {"HS35",  1.0 / 9.0,     hs35},
        {"HS36", -3300.0,        hs36},
        {"HS71", 17.0140173,     hs71},
        {"HS76", -4.681818181,   hs76},
    };

    int solved = 0, matched = 0;
    const int total = (int)(sizeof(cases) / sizeof(cases[0]));
    std::printf("  %-6s %14s %16s %10s %6s %11s\n",
                "prob", "published", "igaos", "status", "iters", "primal inf");
    for (const HsCase& hc : cases) {
        NlpProblem p = hc.build();
        NlpOptions o; o.tolerance = 1e-8; o.maxIter = 300; o.timeLimit = 30.0;
        NlpResult r = solveNlp(p, o);
        const bool conv = (r.status == Status::Optimal) && r.primalInf < 1e-6;
        const double err = std::fabs(r.objective - hc.published)
                           / (1.0 + std::fabs(hc.published));
        if (conv) ++solved;
        const bool ok = conv && err < 1e-5;
        if (ok) ++matched;
        std::printf("  %-6s %14.8g %16.9g %10s %6d %11.2e %s\n",
                    hc.name, hc.published, (double)r.objective, statusName(r.status),
                    (int)r.iterations, (double)r.primalInf,
                    ok ? "" : (conv ? "  <- WRONG OPTIMUM" : "  <- did not converge"));
    }
    std::printf("\n  converged .................. %d / %d\n", solved, total);
    std::printf("  matched the published value  %d / %d\n", matched, total);
    check(matched >= 10, "at least ten of the twelve HS problems match their published optimum",
          (double)matched);
}

// ===========================================================================
//  Part 3: MINLP -- a smooth nonlinear model with integer variables.
// ===========================================================================
static void testMinlp() {
    std::printf("\nMINLP -- branch and bound over NLP relaxations\n");

    // min (x - 1.7)^2 + (y - 2.3)^2 + 0.1*exp(z)  s.t. x + y + z >= 4,
    // with x, y integral in [0,5] and z continuous in [0,3].
    //
    // Small enough to enumerate exactly: for each integer pair, the remaining
    // one-dimensional problem is solved on a fine grid.  That is what the
    // branch and bound is checked against.
    NlpProblem p;
    p.addVariable(0.0, 5.0); p.addVariable(0.0, 5.0); p.addVariable(0.0, 3.0);
    ExprTape& t = p.tape;
    Int x = t.variable(0), y = t.variable(1), z = t.variable(2);
    p.objective = t.add(t.add(t.square(t.sub(x, t.constant(1.7))),
                              t.square(t.sub(y, t.constant(2.3)))),
                        t.mul(t.constant(0.1), t.exp(z)));
    p.addConstraint(t.add(x, t.add(y, z)), 4.0, kInf);
    p.start = {1.0, 1.0, 1.0};

    NlpOptions o; o.tolerance = 1e-9; o.timeLimit = 30.0;
    MinlpResult r = solveMinlp(p, {0, 1}, o);

    double bestEnum = 1e100;
    for (int xi = 0; xi <= 5; ++xi)
        for (int yi = 0; yi <= 5; ++yi)
            for (int k = 0; k <= 6000; ++k) {
                const double zz = 3.0 * k / 6000.0;
                if (xi + yi + zz < 4.0 - 1e-12) continue;
                const double f = (xi - 1.7) * (xi - 1.7) + (yi - 2.3) * (yi - 2.3)
                               + 0.1 * std::exp(zz);
                bestEnum = std::min(bestEnum, f);
            }

    std::printf("  enumeration %.9g   branch and bound %.9g   nodes %lld   status %s\n",
                bestEnum, (double)r.objective, (long long)r.nodes, statusName(r.status));
    check(r.status == Status::Feasible, "MINLP returns a feasible point");
    check(std::fabs(r.objective - bestEnum) < 1e-4,
          "MINLP matches exhaustive enumeration over the integers",
          std::fabs(r.objective - bestEnum));
    check(!r.provenGlobal,
          "and does NOT claim to have proved global optimality, because it has not");
}

int main() {
    std::printf("IGAOS nonlinear programming\n"
                "===========================\n");
    testDerivatives();
    testHockSchittkowski();
    testMinlp();
    std::printf("===========================\n%s (%d failures)\n",
                failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
