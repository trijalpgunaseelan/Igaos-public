# NLP and MINLP

Problem statement 26119 asks for LP, MILP and QP **as the initial focus**, with

> a modular architecture that can later be extended to Mixed-Integer Quadratic
> Programming (MIQP), Nonlinear Programming (NLP) and Mixed-Integer Nonlinear
> Programming (MINLP).

All three are now built. This document used to be a plan; it is now a record of
what was built, what the plan got right, and the one thing it got wrong.

Measured results are in **[bench/results_nlp.txt](../bench/results_nlp.txt)**.
The short version:

| | |
|---|---|
| Haverly pooling problems vs published global optima | **3 / 3** exact |
| Hock–Schittkowski vs published optima | **12 / 12** |
| Global optimum vs 40-start local search, 12 instances | **0** local solutions beat it; **12 / 12** reached |
| Exact derivatives vs central differences | 1.3e-09 worst — the *differencing* error |

---

## The two solvers, and why there are two

The most important thing on this page is that "nonlinear" names two different
problems with two different guarantees, and conflating them is how a solver ends
up lying.

### `src/global.cpp` — nonconvex QCQP and bilinear MINLP, **proven global**

For a model whose *rows* contain products of variables:

```
min  c'x + ½x'Qx
s.t. rl ≤ a_i'x + Σ coef · x_a · x_b ≤ ru,   l ≤ x ≤ u,   x_j integral
```

`Q` may be indefinite. Nothing assumes convexity anywhere.

Spatial branch and bound. Each product `x_i·x_j` is replaced by a variable `w`
constrained by its **McCormick envelope** — the convex hull of the product's
graph over the current box — which makes the relaxation *linear*, and therefore
a **valid lower bound**, and therefore something a tree can prune against. The
envelope is exact at the corners of the box and loose in the middle, and its
looseness shrinks quadratically with the box width, so **branching splits the
range of a continuous variable**. That step has no analogue in a mixed-integer
tree: nothing is fractional, what is violated is `w ≠ x_i·x_j`.

Because the relaxation is an LP — or a MILP when the model has integer variables
— it is solved by the branch-and-cut code that was already here and already
tested. Integer variables and continuous nonconvexity are searched by one tree,
which is what makes this a MINLP solver rather than an NLP one.

Also: interval propagation at every node, optimality-based bound tightening at
the root, and an incumbent heuristic that fixes one side of every product and
re-solves — which on a pooling model is exactly the classical alternating
heuristic, and is what actually finds feasible points.

### `src/nlp.cpp` — general smooth NLP, **local only**

```
min f(x)  s.t.  cl ≤ c(x) ≤ cu,  l ≤ x ≤ u
```

with `f` and `c` built from `+ − × ÷`, powers, `exp`, `log`, `sqrt`, `sin`,
`cos`. Primal-dual interior point: slacks turn every ranged row into an equality
with bounded variables, a log barrier handles the bounds, and the Newton system
condenses to

```
[ −(W + Σx)   A'  ] [dx]   [ rx              ]
[    A       1/Σs ] [dy] = [ −(c−s) − rs/Σs  ]
```

which is quasi-definite — the same shape `src/ipm.cpp` hands to `src/ldl.cpp`
for linear and quadratic programs. What makes it *work* is not the
factorization: it is the **inertia correction** (a Newton direction is a descent
direction only when the KKT matrix has the right inertia) and the **filter line
search** (which refuses to trade constraint violation against objective through
a penalty parameter nobody can choose well).

`NlpResult::localOnly` is **true on every result this returns.** For a convex
problem a KKT point is the global optimum; for anything else it is a KKT point
and nothing more. `bench/results_nlp.txt` shows what that costs: on Haverly 1,
started from 40 random points, this method reports a converged optimal solution
with zero constraint violation and the **wrong answer** in 10 of them.

`solveMinlp` adds branch and bound over NLP relaxations for integer variables.
Its status is `Feasible`, never `Optimal`, and `provenGlobal` is `false` — for a
nonconvex model a local solution is not a bound, so a pruned subtree may have
held the optimum, and this code cannot decide convexity of an arbitrary
expression graph.

---

## What the architecture actually carried

The three predictions this document made in its planning form all held.

**1. The interior point method solves the augmented KKT system.** The plan said
"a nonlinear objective changes exactly one block: `Q` becomes `∇²f(x)`". That is
precisely what happened — `W` above is that block, and `src/ldl.cpp` factors it
with the same AMD ordering and the same dynamic regularization it uses for LP
and QP. Had the normal equations been used instead, this would have been a
rewrite.

**2. Symbolic and numeric factorization are separate.** The KKT *pattern* is
computed once from the constraint sparsity and the Hessian pattern; only the
*values* change per iteration. `LdlFactor::analyze` is called once per solve.

**3. Branch and bound is generic over the relaxation.** `src/miqp.cpp`
demonstrated it; `src/global.cpp` relies on it completely — a spatial node's
relaxation is handed to `Solver::solve` and comes back as an LP or a MILP result
without the tree knowing which.

## What the plan got wrong

It said automatic differentiation was "deliberately **not** on that list …
writing an AD engine is a separate project", and proposed a callback interface
taking gradients and Hessians from the caller.

That was wrong, and it is worth being precise about why, because the reasoning
sounded prudent. A *general* AD engine is a separate project. What an NLP solver
needs is not general: a fixed set of thirteen operations, each with a known
first and second derivative, is a **table** — and the table is `src/expr.cpp`,
about 250 lines. Reverse mode for gradients, forward-over-reverse for
Hessian-vector products, both exact.

The callback design would also have been worse. Hand-written Hessians are the
most common source of silent wrongness in nonlinear modelling, and "the caller
supplies them" moves that failure somewhere nobody tests.

Similarly, the plan ranked convex MINLP by outer approximation *first* by
industrial value, and bilinear/pooling support third. That ordering was wrong
for this customer. Pooling is the problem MRPL actually has, outer approximation
does not cover it, and the bilinear route reuses more of the existing machinery
— the whole spatial solver is about 600 lines *because* the relaxation is an LP.

---

## Where this matters for MRPL

Crude blending with quality constraints **is** the pooling problem: blend quality
is a flow-weighted average, so `quality × flow` is a product of two decisions.
Every other model in this repository — `refinery_blend` included — linearises
that by fixing the qualities. That is the standard *planning* approximation and
what an LP-based refinery planner runs, so it keeps its place.

It is also, in general, wrong, and Haverly's counterexample exists to show how
wrong. `igaos -m haverly1`:

```
rootlp = -500        what the linear relaxation says
final  = -400        the true global optimum
```

A planner acting on the 500 schedules a blend that cannot be made. And no single
fixed pool quality fixes it — 1%, 2% and 3% each give the right answer on some
instances and the wrong one on the rest, because a constant cannot stand in for
something the optimizer is meant to be choosing.

`bench/generate_pooling.py` builds this family, `igaos -m pool_m` runs one, and
the three Haverly instances ship with it.

---

## What is still not built

- **No file format for a general NLP.** `QCMATRIX` in MPS carries quadratic
  constraints, so the whole pooling class round-trips through a file and runs
  from the command line. An `exp` or `log` constraint has to be built through
  the C++ API.
- **The spatial tree is serial.** The mixed-integer tree beside it is a worker
  pool; this one is not.
- **No convexity detection**, so a convex MINLP is not recognised as one and
  does not get the stronger guarantee it would deserve.
- **No second-order correction and no restoration phase** in the filter line
  search. When no trial step is acceptable this loosens the barrier instead,
  which usually works and is not the same thing.
- **Scale is unmeasured.** The largest pooling instance here is 21 rows with 48
  bilinear terms. A spatial tree grows with the number of products, and nothing
  in this repository says what happens at refinery scale. That is the honest
  boundary, and the next thing worth measuring.
- **Presolve and scaling are skipped** on quadratically constrained models —
  neither transforms the row quadratics, and a reduction that dropped them would
  be a silent wrong answer. So a QCQP gets no presolve at all, which is a real
  cost on a large one.
