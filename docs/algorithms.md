# Algorithms

What each component actually does, and — more usefully — why the obvious
alternative was rejected. Most of these are decisions where the textbook answer
and the answer that works on industrial models differ.

---

## 1. The computational form

Everything is held as

```
    minimize    c'x + ½ x'Qx + offset
    subject to  A x - s = 0
                lx ≤ x ≤ ux ,   ls ≤ s ≤ us
```

A row is a *bounded logical variable*, not a sense. `≤`, `≥`, `=`, ranged and
free rows are all the same object with different bounds. This is not
cosmetic: it means the simplex, the interior point method and PDHG each have one
row case to reason about instead of five, and it is why adding cut rows,
changing bounds at a branch-and-bound node, and warm starting are all the same
operation on the same structure.

It also removes the need for artificial variables. The logical basis (every `s`
basic) always exists and is always non-singular, so phase 1 starts from it and
minimises primal infeasibility directly.

---

## 2. Sparse LU: threshold Markowitz, not partial pivoting

Partial pivoting picks the largest element in the column. It is the right answer
for dense matrices and the wrong answer for sparse ones, because it ignores fill
entirely — and on an LP basis, fill is the dominant cost.

Threshold Markowitz picks the pivot minimising the Markowitz count
`(rᵢ − 1)(cⱼ − 1)` — the number of entries the elimination step could create —
subject to a *relative* stability floor `|aᵢⱼ| ≥ τ · maxᵢ|aᵢⱼ|`. The threshold
buys stability; the Markowitz count buys sparsity; the two are traded off
explicitly rather than one being sacrificed silently.

Basis changes are absorbed by product-form (eta) updates, with refactorization
triggered by measured accuracy loss rather than a fixed count alone.

---

## 3. Degeneracy: Harris and perturbation, not Bland's rule

The textbook answer to degeneracy is Bland's rule, which guarantees termination.
It is the wrong tool here, and the reason is that **cycling is not the problem
industrial models have**. Cycling — returning to a previously visited basis — is
vanishingly rare in floating-point arithmetic. What actually happens is
*stalling*: thousands of consecutive degenerate pivots with zero objective
progress. Bland's rule does not fix stalling; it makes it worse, because it
discards the pricing information that would otherwise find a productive edge.

What is implemented instead:

- **Harris two-pass ratio test.** The first pass computes the maximum step under
  bounds relaxed by a small tolerance; the second picks, among the candidates
  that pass, the one with the largest pivot magnitude. This buys numerical
  stability *and* frequently a longer step than the textbook ratio test allows.
- **Bound flipping.** A boxed variable that would block the step can be flipped
  to its opposite bound instead of leaving the basis, letting the step continue.
- **Bound perturbation** as the anti-stalling device, removed at the end.
- **Cost shifting** for dual phase 1.

---

## 4. Interior point: augmented KKT, not normal equations

Each Newton step solves

```
    [ -(Q + Dx)   A'  ] [ dx ]   [ rx ]
    [    A        Dy  ] [ dy ] = [ ry ]
```

The alternative is to eliminate `dx` and solve the normal equations
`A (Q + Dx)⁻¹ A' dy = r`. For an LP that is often faster: `Dx` is diagonal, so
its inverse is diagonal and `A D⁻¹ A'` is usually sparse.

**It cannot serve QP.** `(Q + Dx)⁻¹` is dense whenever `Q` is not diagonal, so
`A (Q + Dx)⁻¹ A'` is dense — for any `Q` with off-diagonal structure, which is
every interesting QP. A solver built on normal equations needs a second kernel
for QP, with its own ordering, its own failure modes and its own bugs.

The augmented form avoids that because `Q` enters the (1,1) block *additively*
and changes nothing structural. One factorization routine, one ordering, one set
of numerical failure modes, LP and QP alike.

The second reason is **quasi-definiteness**. With `Dx ≥ 0` and `Dy > 0` the
matrix is quasi-definite, which means an `LDLᵀ` factorization exists for *every*
symmetric permutation. So the ordering can be chosen purely to reduce fill — AMD
on the pattern, computed once — and stability is recovered afterwards by dynamic
regularization plus iterative refinement, rather than by numerical pivoting that
would destroy the ordering. Symbolic analysis runs once for the whole solve.

Row and variable bound patterns are all handled through the same barrier state:
both bounds, lower only, upper only, free, and fixed. Equality rows are the
fixed case, where `Dy` collapses to the regularization term and `ds = 0`.

---

## 5. Crossover

An interior point converges to the relative interior of the optimal face. That
point is optimal, but it is not a vertex, so it carries no basis — and without a
basis there is nothing for branch and cut to warm start from and no exact
identification of active constraints.

The implementation ranks every variable, structural and logical alike, by how
far the interior point leaves it from its nearest bound (relatively, so a
variable on `[0, 10⁶]` and one on `[0, 1]` compare fairly), takes the `m` most
interior as the candidate basis, pushes the rest onto their nearer bound, and
hands the assignment to the simplex, which repairs linear dependence by swapping
in logicals and re-optimizes.

This is identification-and-clean-up, not a Megiddo-style strongly polynomial
push. What it guarantees is the part that matters downstream: on return the
solution is basic, primal and dual feasible, and optimal to the *simplex's*
tolerances rather than the barrier's. On the test set the clean-up costs a mean
of 0.0 simplex iterations — the identification is usually already exact.

---

## 6. First-order PDHG, and why *this* is the GPU path

Dualizing the row-box indicator gives the saddle point problem

```
    L(x, y) = c'x + y'Ax − σ_C(y),      σ_C(y) = sup_{s ∈ C} y's
```

and PDHG alternates

```
    x⁺ = proj_[l,u] ( x − τ (c + A'y) )
    y⁺ = prox_{σ·σ_C} ( y + σ A (2x⁺ − x) )
```

For a box `C`, Moreau's identity gives `prox_{a·σ_C}(v) = v − a·proj_C(v/a)` in
closed form. So one iteration is **exactly one `Ax`, one `A'y`, and elementwise
work**. No factorization. No elimination tree. No sequential dependency chain.

That is the entire GPU argument, and it is a claim about this method, not about
the solver as a whole:

- Sparse LU and sparse Cholesky are dominated by a long, irregular dependency
  chain through the elimination tree, with little work per node. A GPU spends
  its time waiting on dependencies. Published sparse-direct GPU results are
  dominated by dense supernodes, which industrial LP bases do not have.
- SpMV is bandwidth-bound and perfectly parallel across rows. That is what GPU
  memory systems are built for.

Restarts to the running average and primal weight adaptation are what make PDHG
practical rather than merely convergent. The weight is updated from how far each
block has travelled *since the previous restart*; comparing a single primal step
against `‖y‖` instead makes the weight drift monotonically until it pins at its
cap, and a pinned weight is a stalled method — `τ` shrinks to nothing and the
primal iterate stops moving. (That was a real bug during development, and the
symptom was convergence stalling at ~1e-4 with `ω` at 10⁴.)

First-order methods converge slowly in the final digits. The driver therefore
falls back to the simplex when PDHG does not reach tolerance, and runs crossover
when it does.

---

## 7. Cut separation

### Gomory *mixed-integer*, not *fractional*

A tableau row for a basic variable reads, once every nonbasic variable is
rewritten as its non-negative distance `t` from the bound it sits at,

```
    x_B + Σ ᾱⱼ tⱼ = b ,   tⱼ ≥ 0
```

and with `f₀ = frac(b)` the GMI inequality is

```
    Σ_{j integer} φ(ᾱⱼ) tⱼ + Σ_{j continuous} ψ(ᾱⱼ) tⱼ ≥ 1

    φ(a) = f_a / f₀             if f_a ≤ f₀,  else (1 − f_a)/(1 − f₀)
    ψ(a) = a / f₀               if a ≥ 0,     else −a/(1 − f₀)
```

The *fractional* Gomory cut has only the `φ` branch and is valid **only if every
variable in the row is integer**. On a mixed-integer model it cuts off feasible
points. This distinction is the single easiest way to build a solver that
returns confident wrong answers, which is why the separator is named for it.

Two validity conditions are enforced that are easy to miss:

- A **free nonbasic** variable sitting at zero has no non-negative `t` form, so
  a row that touches one is skipped rather than fudged.
- An integer variable resting on a **fractional bound** (which presolve's bound
  tightening can produce) has non-integral `t`, so the `φ` branch does not
  apply. Such a variable is downgraded to the continuous branch, which is valid
  for any real `t ≥ 0`.

### Knapsack cover

For a row whose support is entirely binary, complement the negative
coefficients, greedily find a cover `C` with `Σ_C w > cap` minimising
`Σ_C (1 − z)`, make it minimal, extend it with every item at least as heavy as
the heaviest cover member, and emit `Σ_E y ≤ |C| − 1`.

### Complemented MIR

Substitute each variable against the bound it currently sits nearer to so all
variables are non-negative; drop continuous terms with positive coefficients
(a valid relaxation, since they only push the left-hand side up); collect the
rest into a slack; scale by `1/δ` for a family of candidate divisors — MIR is
not scale-invariant, and the right divisor is what turns a weak row into a
strong cut — and apply

```
    Σ ( ⌊aⱼ⌋ + (fⱼ − f₀)⁺ / (1 − f₀) ) x'ⱼ − s/(1 − f₀) ≤ ⌊b⌋
```

then undo the substitution. The two substitution cases move the constant to the
right-hand side with **opposite signs**; getting that wrong produces cuts that
are violated by the optimum on roughly one model in a hundred, which is exactly
often enough to survive casual testing.

### Root-only, and why

Separation happens at the root and nowhere else. That is a validity decision,
not a shortcut. A GMI cut read off a tableau row *at a node* is derived from
that node's locally tightened bounds, so it is sound only inside that subtree;
keeping it globally would cut off feasible integer points elsewhere in the tree.
Root separation with the global bounds gives cuts that are sound everywhere,
which is what lets the pool simply become extra rows of the model for the whole
search.

### Pool hygiene

A cut that is numerically bad is worse than no cut: it destroys the conditioning
of every basis that includes it, and it is paid for again at every node. The
pool therefore:

- merges repeated columns (a separator accumulating into a dense workspace can
  emit the same column twice when a logical substitution cancels a structural
  coefficient exactly — halving the coefficient and invalidating the cut);
- normalises to `max|a| = 1` and rejects on absolute coefficient size, dynamism
  `max|a|/min|a|`, density, and efficacy `violation / ‖a‖₂`;
- compensates the right-hand side when it drops a negligible coefficient, since
  dropping a term from a `≥` inequality otherwise *strengthens* it;
- shaves the right-hand side by a relative `1e-9` so floating-point noise in the
  derivation cannot make a cut tighter than the exact halfspace it models;
- suppresses duplicates by hashing the normalised halfspace;
- purges cuts that end up slack at the root before the tree starts.

### Verifying validity

`CutLimits::referencePoint` accepts a point known to be feasible for the
mixed-integer problem — typically a proven optimum from a reference run. Every
candidate cut is tested against it, and a cut that excludes it is rejected and
counted. A valid cut can never exclude a feasible point, so this converts the
worst failure mode in the whole solver — a silent wrong answer thousands of
nodes later — into an immediate, localized, attributable failure. The regression
suite and the fuzz harness both switch it on.

---

## 8. Branch and cut

- **Dual simplex warm starts** at every node. Changing one bound leaves the
  parent basis dual feasible, which is exactly the situation the dual simplex is
  built for; a cold start would throw that away.
- **Pseudocost branching** learned from *measured* objective degradation per
  unit of fractionality moved. Seeding pseudocosts with fractionality instead
  makes the rule behave like most-fractional branching, which is barely better
  than random.
- **Reliability phase**: until a variable has enough observations, fall back to
  fractionality rather than trusting a pseudocost estimated from one sample.
- **Plunge then best-bound**: depth-first to find incumbents, best-bound to
  prove optimality.
- **Heuristics**: nearest rounding, then progressively more up-biased fractional
  dives with one-level backtracking. On unit-commitment models — where minimum
  generation can exceed off-peak demand, so neither round-up nor round-down
  alone is feasible — the direction flip is what actually produces the incumbent.
- **Node iteration cap**: a warm-started re-solve should take tens of iterations.
  A node that hits the cap is retried cold, and if that also fails the run is
  reported as not proven rather than silently pruned.
- **Acceptance gate**: heuristic candidates are integral by construction but need
  not satisfy the rows. The gate is held at the solver's own feasibility
  tolerance, not a loose multiple of it — a candidate admitted at `1e-4` is
  handed back to the caller as an *optimal* solution violating its constraints
  by `1e-4`, and nothing downstream catches it.
- **Polish**: after postsolve, the integers are fixed at the values the search
  proved and the remaining LP is re-solved on the *original* model, so the
  continuous values reported are exact rather than carrying scaling and
  postsolve error.

---

## 9. Presolve and the cleanup solve

Presolve removes empty and fixed columns, empty rows, singleton rows, forcing
and redundant rows, and tightens bounds by constraint propagation with integer
rounding. Postsolve reconstructs primal values *and a complete basis*.

The cleanup step then re-solves the **original** problem from that basis. For a
correct chain it terminates immediately at zero iterations, and it does two
things: it makes the reported duals and reduced costs exact in the user's own
space, independent of how aggressive presolve was, and it acts as an independent
check that the reduction chain preserved optimality. It may only confirm or
improve the postsolved point; if it comes back worse, the postsolved solution
stands and the log says so.
