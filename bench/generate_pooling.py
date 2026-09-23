#!/usr/bin/env python3
"""
Refinery pooling instances -- the crude blending problem with pool qualities as
DECISIONS rather than as fixed inputs.

    python3 bench/generate_pooling.py outdir [--seeds 3]

WHY THIS IS A SEPARATE FAMILY FROM refinery_blend
--------------------------------------------------
`bench/generate.py: refinery_blend` is the same physical network solved on a
volume basis: every pool is given a FIXED average quality, computed once from
the crudes that feed it, and the model is then linear.  That is the standard
planning approximation and it is what an LP-based refinery planner actually
runs, so it earns its place.

It is also, in general, wrong.  The quality leaving a pool is the flow-weighted
average of what went in, so quality x flow is a product of two decisions:

    sum_q  p[k,t] * g[k,q]  =  sum_c  prop[c,t] * f[c,k]

with p (the pool's quality) and g (what leaves it) both free.  Fixing p turns a
nonconvex problem into a linear one whose optimum may not be achievable at all
-- Haverly's 1978 counterexample exists precisely to show a linearised blending
model reporting a blend that cannot be made.  tools/pooling_check.cpp solves the
three Haverly instances and prints, next to each, what the same network says
when the pool quality is pinned at 1%, 2% and 3%: no single choice reproduces
all three answers, because no single choice can.

So these instances are the honest form of the problem, and they are solved to
PROVEN GLOBAL optimality by the spatial branch and bound in src/global.cpp.
They are also much harder, which is the trade being made and the reason both
families exist.

FORMAT
------
Written as MPS with QCMATRIX sections -- the CPLEX and Gurobi convention for
quadratic constraints, one section per row, off-diagonal terms split
symmetrically into two halves.  bench/mpsread.py and src/mps.cpp both read it.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import LP, INF


class PoolLP(LP):
    """An LP plus bilinear terms inside rows."""

    def __init__(self, name):
        super().__init__(name)
        self.q = []                      # (row, col_i, col_j, coefficient)

    def quad(self, row, i, j, v):
        if v != 0.0:
            self.q.append((row, i, j, v))

    def nquad(self):
        return len(self.q)

    def write(self, path):
        super().write(path)
        if not self.q:
            return path
        with open(path) as f:
            text = f.read()
        byrow = {}
        for (r, i, j, v) in self.q:
            byrow.setdefault(r, []).append((i, j, v))
        lines = []
        for r in sorted(byrow):
            lines.append(f"QCMATRIX  {self.rows[r][0]}")
            for (i, j, v) in byrow[r]:
                ni, nj = self.cols[i][0], self.cols[j][0]
                if i == j:
                    lines.append(f"    {ni}  {nj}  {float(v):.12g}")
                else:
                    # Symmetric halves: a reader that sums both triangles gets
                    # the coefficient back exactly.  Writing one triangle would
                    # halve every cross term, which in a quality balance is a
                    # silent factor of two on the thing being blended.
                    lines.append(f"    {ni}  {nj}  {float(v) * 0.5:.12g}")
                    lines.append(f"    {nj}  {ni}  {float(v) * 0.5:.12g}")
        text = text.replace("ENDATA", "\n".join(lines) + "\nENDATA")
        with open(path, "w") as f:
            f.write(text)
        return path


def pooling(seed=0, n_crude=4, n_pool=1, n_prod=2, n_prop=1, bypass=True):
    """Crude -> pool -> product, with pool qualities free.

    Every variable that appears in a product must be boxed, and both boxes here
    are physical rather than invented: a flow cannot exceed the demand it serves,
    and a pool's quality cannot leave the range of the crudes that can reach it.
    That second bound is what makes the McCormick relaxation tight enough to be
    useful -- a pool quality boxed at [0, 100] would relax to almost nothing.
    """
    rng = np.random.default_rng(seed + 7700)
    p = PoolLP(f"POOL{seed}")

    prop = rng.uniform(0.8, 3.2, (n_crude, n_prop))       # crude qualities
    # CHEAP CRUDE IS BAD CRUDE.  Cost falls as the contaminant rises, which is
    # the actual economics of a refinery feedstock slate and the entire reason
    # blending is a decision at all.  The first version of this generator drew
    # cost independently of quality, and every instance then solved at the ROOT
    # in one node: with no tension between price and specification the optimum
    # sits at a corner of the box, and a McCormick envelope is EXACT at its
    # corners.  The instances were easy for a reason that had nothing to do with
    # the solver, which makes them worthless as a test of it.
    cost = 22.0 - 4.5 * prop.mean(axis=1) + rng.uniform(-0.6, 0.6, n_crude)
    avail = rng.uniform(80.0, 260.0, n_crude)
    price = rng.uniform(14.0, 24.0, n_prod)
    dem = rng.uniform(60.0, 180.0, n_prod)
    # A specification tight enough that no single crude meets it cheaply: below
    # the median quality, so the cheap high-contaminant crudes have to be cut
    # with expensive clean ones and the ratio is what is being optimised.
    spec = np.array([[float(np.percentile(prop[:, t], 35)) for t in range(n_prop)]
                     for _ in range(n_prod)])

    f = {}                                                # crude -> pool
    for c in range(n_crude):
        for k in range(n_pool):
            f[(c, k)] = p.col(f"F{c}_{k}", 0.0, float(avail[c]), float(cost[c]))
    g = {}                                                # pool -> product
    for k in range(n_pool):
        for q in range(n_prod):
            g[(k, q)] = p.col(f"G{k}_{q}", 0.0, float(dem[q]), float(-price[q]))
    # Only the CLEANEST crudes get a direct route to the products.  Everything
    # else must pass through a pool, which is what forces the blend decision to
    # be made in the pool and keeps the bilinear terms active at the optimum.
    #
    # Giving every crude a bypass arc -- the first version of this generator --
    # lets the model meet every specification by direct blending and never use
    # the pool at all, so the products are inactive and the whole instance
    # solves at the root.  It is exactly Haverly's structure that makes the
    # problem hard: crudes A and B can reach the products ONLY through the pool,
    # and C is the one that bypasses it.
    order = np.argsort(prop.mean(axis=1))                 # cleanest first
    bypassSet = set(int(c) for c in order[:max(1, n_crude // 3)]) if bypass else set()
    d = {}                                                # crude -> product direct
    for c in sorted(bypassSet):
        for q in range(n_prod):
            d[(c, q)] = p.col(f"D{c}_{q}", 0.0, float(dem[q]),
                              float(cost[c] - price[q]))
    pq = {}                                               # the pool qualities
    for k in range(n_pool):
        for t in range(n_prop):
            lo, up = float(prop[:, t].min()), float(prop[:, t].max())
            pq[(k, t)] = p.col(f"P{k}_{t}", lo, up, 0.0)

    for c in range(n_crude):                              # availability
        r = p.row(f"AV{c}", -INF, float(avail[c]))
        for k in range(n_pool):
            p.add(r, f[(c, k)], 1.0)
        for q in range(n_prod):
            if (c, q) in d:
                p.add(r, d[(c, q)], 1.0)

    for k in range(n_pool):                               # pool material balance
        r = p.row(f"PB{k}", 0.0, 0.0)
        for c in range(n_crude):
            p.add(r, f[(c, k)], 1.0)
        for q in range(n_prod):
            p.add(r, g[(k, q)], -1.0)

    for k in range(n_pool):                               # pool QUALITY balance
        for t in range(n_prop):
            r = p.row(f"PQ{k}_{t}", 0.0, 0.0)
            for q in range(n_prod):
                p.quad(r, pq[(k, t)], g[(k, q)], 1.0)     # <- the nonconvexity
            for c in range(n_crude):
                p.add(r, f[(c, k)], float(-prop[c, t]))

    for q in range(n_prod):                               # demand
        r = p.row(f"DM{q}", float(dem[q] * 0.35), float(dem[q]))
        for k in range(n_pool):
            p.add(r, g[(k, q)], 1.0)
        for c in range(n_crude):
            if (c, q) in d:
                p.add(r, d[(c, q)], 1.0)

    for q in range(n_prod):                               # product specification
        for t in range(n_prop):
            r = p.row(f"SP{q}_{t}", -INF, 0.0)
            for k in range(n_pool):
                p.quad(r, pq[(k, t)], g[(k, q)], 1.0)     # <- and again
                p.add(r, g[(k, q)], float(-spec[q, t]))
            for c in range(n_crude):
                if (c, q) in d:
                    p.add(r, d[(c, q)], float(prop[c, t] - spec[q, t]))
    return p


def haverly(variant=1):
    """Haverly's 1978 pooling problem, the standard counterexample.

    Three crudes, one pool whose sulfur is a decision, two products with sulfur
    specifications.  Crudes A and B reach the products ONLY through the pool; C
    bypasses it.  Published global optima, as minimised negative profit:

        HPP1  cost(B) 16, demand X 100  ->  -400
        HPP2  cost(B) 16, demand X 600  ->  -600
        HPP3  cost(B) 13, demand X 100  ->  -750

    Included in this family because the generated instances above turn out to
    have a single local optimum each -- which makes them a fair test of
    correctness and no test at all of why global optimization is needed.  These
    three have several, and a local method lands on the wrong one depending on
    where it starts.  That is the entire argument, and it deserves an instance
    rather than a paragraph.
    """
    costB = 13.0 if variant == 3 else 16.0
    demX = 600.0 if variant == 2 else 100.0
    demY = 200.0
    p = PoolLP(f"HAVERLY{variant}")
    A  = p.col("A", 0.0, 1000.0, 6.0)
    B  = p.col("B", 0.0, 1000.0, costB)
    Cx = p.col("Cx", 0.0, demX, 1.0)
    Cy = p.col("Cy", 0.0, demY, -5.0)
    Px = p.col("Px", 0.0, demX, -9.0)
    Py = p.col("Py", 0.0, demY, -15.0)
    S  = p.col("poolS", 1.0, 3.0, 0.0)

    r = p.row("poolbal", 0.0, 0.0)
    p.add(r, A, 1.0); p.add(r, B, 1.0); p.add(r, Px, -1.0); p.add(r, Py, -1.0)

    r = p.row("poolqual", 0.0, 0.0)
    p.add(r, A, -3.0); p.add(r, B, -1.0)
    p.quad(r, S, Px, 1.0); p.quad(r, S, Py, 1.0)

    r = p.row("demX", -INF, demX); p.add(r, Px, 1.0); p.add(r, Cx, 1.0)
    r = p.row("demY", -INF, demY); p.add(r, Py, 1.0); p.add(r, Cy, 1.0)

    r = p.row("specX", -INF, 0.0)
    p.add(r, Px, -2.5); p.add(r, Cx, -0.5); p.quad(r, S, Px, 1.0)
    r = p.row("specY", -INF, 0.0)
    p.add(r, Py, -1.5); p.add(r, Cy, 0.5); p.quad(r, S, Py, 1.0)
    return p


SUITE_POOL = [
    ("pool_s", lambda s: pooling(s, 4, 1, 2, 1), "Refinery pooling"),
    ("pool_m", lambda s: pooling(s, 5, 2, 3, 1), "Refinery pooling"),
    ("pool_l", lambda s: pooling(s, 6, 2, 3, 2), "Refinery pooling"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for v in (1, 2, 3):
        m = haverly(v)
        path = os.path.join(a.outdir, f"haverly{v}.mps")
        m.write(path)
        print(f"haverly{v:<4} {'Haverly (published optimum)':<20} {len(m.rows):>5} rows "
              f"{len(m.cols):>5} cols {m.nnz():>7} nnz {m.nquad():>5} bilinear terms")
    for tag, fn, family in SUITE_POOL:
        for s in range(a.seeds):
            m = fn(s)
            path = os.path.join(a.outdir, f"{tag}_{s}.mps")
            m.write(path)
            print(f"{tag}_{s:<3} {family:<20} {len(m.rows):>5} rows {len(m.cols):>5} cols "
                  f"{m.nnz():>7} nnz {m.nquad():>5} bilinear terms")


if __name__ == "__main__":
    main()
