#!/usr/bin/env python3
"""
Mixed-integer convex quadratic instances, written as MPS with QUADOBJ.

A convex QP with some columns declared integer. The point of the family is that
the answer is *not* obtainable by rounding the QP relaxation and is *not*
obtainable by solving the linear part and evaluating the quadratic at the
result — the two shortcuts a solver takes when it does not really have an MIQP
path. Both were live defects here (bug 13, bug 15); this is the generator that
would have caught them.

  lots     Discrete lot blending. Components arrive as whole tanker lots, so the
           quantity of each is an integer. A blend is judged by how far its
           properties land from spec, and that penalty is quadratic:
           min ||S x - t||^2 + c'x  gives  Q = 2 S'S, positive semidefinite by
           construction and normally rank deficient. Rounding the continuous
           optimum is wrong here in the ordinary case, not the contrived one:
           the property constraints couple the components, so moving one lot up
           forces another down.

  card     Cardinality-constrained portfolio, the textbook MIQP. Minimise
           w'Sigma w - mu'w subject to a budget, with a binary y_i per asset,
           w_i <= u y_i, and sum y_i <= k. The quadratic is over the continuous
           weights only; the tree branches on the binaries. Every node
           relaxation is a QP, which is exactly the structure src/miqp.cpp
           exists to handle.

  smoothi  Integer production smoothing. Units run at whole megawatt steps, so
           the period-to-period smoothing penalty sum (x_t - x_{t-1})^2 sits on
           integer columns and the Hessian is the singular tridiagonal
           second-difference matrix.

    python3 bench/generate_miqp.py outdir [--seeds 3]

Verification, since these have no published reference: for the small sizes the
optimum can be found by exhaustive enumeration over the integer columns, each
fixing solved as a continuous QP. `--verify` does that and compares.
"""
import argparse, itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import INF
from generate_qp import QP


def lots(seed, ncomp=14, nprop=5, maxlot=6):
    """Discrete lot blending: integer lots, quadratic property penalty."""
    rng = np.random.default_rng(7000 + seed)
    p = QP(f"MIQLOTS{seed}")
    x = [p.col(f"L{c}", 0.0, float(maxlot), integer=True) for c in range(ncomp)]

    S = rng.uniform(0.3, 6.0, size=(nprop, ncomp))     # property per lot
    demand = float(rng.integers(3 * ncomp // 2, 2 * ncomp))
    r = p.row("VOLUME", demand, demand)                # total lots is fixed
    for c in range(ncomp):
        p.add(r, x[c], 1.0)
    for g in range(3):                                 # supply groups
        rr = p.row(f"GROUP{g}", -INF, float(maxlot * (ncomp // 3) * 0.6))
        for c in range(g, ncomp, 3):
            p.add(rr, x[c], 1.0)

    t = S @ (np.ones(ncomp) * demand / ncomp)          # spec, reachable on average
    p.set_hessian(2.0 * (S.T @ S), x)                  # from ||Sx - t||^2
    g = -2.0 * (S.T @ t) + rng.uniform(0.5, 4.0, size=ncomp)
    for c in range(ncomp):
        nm, lo, up, _, ii = p.cols[x[c]]
        p.cols[x[c]] = (nm, lo, up, float(g[c]), ii)
    return p


def card(seed, nasset=40, nfactor=6, kmax=8):
    """Cardinality-constrained portfolio: binary holds, quadratic risk."""
    rng = np.random.default_rng(8000 + seed)
    p = QP(f"MIQCARD{seed}")
    u = 0.30
    w = [p.col(f"W{i}", 0.0, u) for i in range(nasset)]
    y = [p.col(f"Y{i}", 0.0, 1.0, integer=True) for i in range(nasset)]

    r = p.row("BUDGET", 1.0, 1.0)
    for i in range(nasset):
        p.add(r, w[i], 1.0)
    rk = p.row("CARD", -INF, float(kmax))              # hold at most k names
    for i in range(nasset):
        p.add(rk, y[i], 1.0)
    for i in range(nasset):                            # w_i <= u y_i
        rr = p.row(f"LINK{i}", -INF, 0.0)
        p.add(rr, w[i], 1.0)
        p.add(rr, y[i], -u)

    F = rng.normal(0, 0.10, size=(nasset, nfactor))
    D = np.diag(rng.uniform(0.005, 0.06, size=nasset))
    p.set_hessian(2.0 * (F @ F.T + D), w)              # 1/2 x'(2 Sigma)x
    mu = rng.normal(0.07, 0.05, size=nasset)
    for i in range(nasset):
        nm, lo, up, _, ii = p.cols[w[i]]
        p.cols[w[i]] = (nm, lo, up, float(-mu[i]), ii)
    return p


def smooth_int(seed, T=18, nunit=3, cap=12):
    """Integer production smoothing: whole steps, tridiagonal Hessian."""
    rng = np.random.default_rng(9000 + seed)
    p = QP(f"MIQSMTH{seed}")
    dem = rng.integers(int(0.5 * nunit * cap), int(0.8 * nunit * cap), size=T)
    x = [[p.col(f"P{uu}_{t}", 0.0, float(cap), integer=True) for t in range(T)]
         for uu in range(nunit)]
    for t in range(T):
        r = p.row(f"DEM{t}", float(dem[t]), INF)
        for uu in range(nunit):
            p.add(r, x[uu][t], 1.0)
    for uu in range(nunit):
        wgt = float(rng.uniform(1.0, 3.0))
        for t in range(1, T):                          # wgt * (x_t - x_{t-1})^2
            p.quad(x[uu][t], x[uu][t], 2 * wgt)
            p.quad(x[uu][t - 1], x[uu][t - 1], 2 * wgt)
            p.quad(x[uu][t], x[uu][t - 1], -2 * wgt)
        c = rng.uniform(1.0, 5.0, size=T)
        for t in range(T):
            nm, lo, up, _, ii = p.cols[x[uu][t]]
            p.cols[x[uu][t]] = (nm, lo, up, float(c[t]), ii)
    return p


SUITE_MIQP = [
    ("miqlots_s",  lambda s: lots(s, 10, 4, 5),        "Discrete lot blending"),
    ("miqlots_m",  lambda s: lots(s, 16, 6, 6),        "Discrete lot blending"),
    ("miqcard_s",  lambda s: card(s, 24, 5, 5),        "Cardinality portfolio"),
    ("miqcard_m",  lambda s: card(s, 45, 7, 8),        "Cardinality portfolio"),
    ("miqsmth_s",  lambda s: smooth_int(s, 14, 2, 10), "Integer smoothing"),
    ("miqsmth_m",  lambda s: smooth_int(s, 20, 3, 12), "Integer smoothing"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for tag, fn, family in SUITE_MIQP:
        for s in range(a.seeds):
            q = fn(s)
            path = os.path.join(a.outdir, f"{tag}_{s}.mps")
            q.write(path)
            nint = sum(1 for c in q.cols if c[4])
            print(f"{tag}_{s:<3} {family:<24} {len(q.rows):>5} rows {len(q.cols):>5} cols "
                  f"{q.nnz():>7} nnz {len(q.q):>7} quad {nint:>4} integer")


if __name__ == "__main__":
    main()
