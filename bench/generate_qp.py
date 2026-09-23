#!/usr/bin/env python3
"""
Convex quadratic programming instances, written as MPS with a QUADOBJ section.

Three families, each the quadratic form an industrial planner actually writes:

  blendfit   Refinery blend property tracking. A blend is judged by how far its
             properties land from target, and the penalty is quadratic because
             being twice as far off is more than twice as bad. Minimising
             ||Sx - t||^2 over the blending polytope gives Q = 2 S'S, which is
             positive semidefinite by construction and usually rank deficient —
             the case a solver that assumes a positive definite Hessian gets
             wrong.

  smooth     Production smoothing. Running a unit up and down costs money, so
             the objective carries a quadratic penalty on the period-to-period
             change, sum (x_t - x_{t-1})^2. That Hessian is the tridiagonal
             second-difference matrix: singular, and every off-diagonal
             negative.

  risk       Portfolio / feed-selection risk. Minimise x'Sigma x - mu'x subject
             to a budget and position limits, with Sigma built as a factor model
             F F' + D, which is how a covariance matrix is actually estimated.

Every objective is c'x + 1/2 x'Qx, with Q symmetric and written as its lower
triangle — the MPS QUADOBJ convention, and the one the solver uses internally.

    python3 bench/generate_qp.py outdir [--seeds 3]
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import LP, INF


class QP(LP):
    """An LP plus a lower-triangular quadratic objective term."""

    def __init__(self, name):
        super().__init__(name)
        self.q = {}                       # (i, j) -> value, i >= j

    def quad(self, i, j, v):
        if v == 0.0:
            return
        if i < j:
            i, j = j, i
        self.q[(i, j)] = self.q.get((i, j), 0.0) + v

    def set_hessian(self, Q, cols):
        """Take a dense symmetric Q over `cols` and store its lower triangle.

        The objective is c'x + 1/2 x'Qx, so the lower-triangle entry for a pair
        (a, b) is the symmetric average of Q[a,b] and Q[b,a]; the diagonal goes
        in as it stands.
        """
        for a in range(len(cols)):
            for b in range(a + 1):
                v = Q[a, b] if a == b else (Q[a, b] + Q[b, a]) / 2.0
                if abs(v) > 1e-14:
                    self.quad(cols[a], cols[b], v)

    def write(self, path):
        super().write(path)
        if not self.q:
            return
        with open(path) as f:
            text = f.read()
        bycol = {}
        for (i, j), v in self.q.items():
            bycol.setdefault(j, []).append((i, v))       # column j, row i, i >= j
        lines = ["QUADOBJ"]
        for j in sorted(bycol):
            for i, v in sorted(bycol[j]):
                lines.append(f"    {self.cols[j][0]}  {self.cols[i][0]}  {float(v):.12g}")
        text = text.replace("ENDATA", "\n".join(lines) + "\nENDATA")
        with open(path, "w") as f:
            f.write(text)


# ---------------------------------------------------------------------------

def blend_fit(seed, ncomp=40, nprop=8, nblend=3):
    """Least-squares property tracking over a blending polytope."""
    rng = np.random.default_rng(1000 + seed)
    p = QP(f"QBLEND{seed}")
    x = [[p.col(f"X{b}_{c}", 0.0, 1.0) for c in range(ncomp)] for b in range(nblend)]

    S = rng.uniform(0.2, 8.0, size=(nprop, ncomp))          # component properties
    for b in range(nblend):
        r = p.row(f"SUM{b}", 1.0, 1.0)                       # fractions sum to one
        for c in range(ncomp):
            p.add(r, x[b][c], 1.0)
    for c in range(ncomp):                                   # shared availability
        r = p.row(f"AVAIL{c}", -INF, 1.4)
        for b in range(nblend):
            p.add(r, x[b][c], 1.0)

    for b in range(nblend):
        t = S @ rng.dirichlet(np.ones(ncomp))                # a reachable target
        Q = 2.0 * (S.T @ S)                                  # from ||Sx - t||^2
        p.set_hessian(Q, x[b])
        g = -2.0 * (S.T @ t)
        for c in range(ncomp):
            nm, lo, up, _, ii = p.cols[x[b][c]]
            p.cols[x[b][c]] = (nm, lo, up, float(g[c]), ii)
    return p


def smooth(seed, T=60, nunit=4):
    """Production smoothing: quadratic penalty on period-to-period change."""
    rng = np.random.default_rng(2000 + seed)
    p = QP(f"QSMOOTH{seed}")
    cap = rng.uniform(60, 120, size=nunit)
    dem = rng.uniform(80, 160, size=T)
    x = [[p.col(f"P{u}_{t}", 0.0, float(cap[u])) for t in range(T)] for u in range(nunit)]

    for t in range(T):                                       # meet demand each period
        r = p.row(f"DEM{t}", float(dem[t]), INF)
        for u in range(nunit):
            p.add(r, x[u][t], 1.0)

    for u in range(nunit):
        w = float(rng.uniform(0.5, 3.0))
        Q = np.zeros((T, T))
        for t in range(1, T):                                # w * (x_t - x_{t-1})^2
            Q[t, t] += 2 * w; Q[t - 1, t - 1] += 2 * w
            Q[t, t - 1] -= 2 * w; Q[t - 1, t] -= 2 * w
        for a in range(T):
            for b in range(a + 1):
                v = Q[a, b] if a == b else (Q[a, b] + Q[b, a]) / 2.0
                if abs(v) > 1e-14:
                    p.quad(x[u][a], x[u][b], v)
        c = rng.uniform(1.0, 4.0, size=T)
        for t in range(T):
            nm, lo, up, _, ii = p.cols[x[u][t]]
            p.cols[x[u][t]] = (nm, lo, up, float(c[t]), ii)
    return p


def risk(seed, nasset=120, nfactor=10):
    """Factor-model risk minimisation with a budget and position limits."""
    rng = np.random.default_rng(3000 + seed)
    p = QP(f"QRISK{seed}")
    x = [p.col(f"W{i}", 0.0, 0.25) for i in range(nasset)]

    r = p.row("BUDGET", 1.0, 1.0)
    for i in range(nasset):
        p.add(r, x[i], 1.0)
    for g in range(5):                                       # sector caps
        rr = p.row(f"SECTOR{g}", -INF, 0.45)
        for i in range(g, nasset, 5):
            p.add(rr, x[i], 1.0)

    F = rng.normal(0, 0.09, size=(nasset, nfactor))
    D = np.diag(rng.uniform(0.004, 0.05, size=nasset))
    Sigma = F @ F.T + D                                      # PSD by construction
    p.set_hessian(2.0 * Sigma, x)                            # 1/2 x'(2 Sigma)x = x'Sigma x
    mu = rng.normal(0.06, 0.04, size=nasset)
    for i in range(nasset):
        nm, lo, up, _, ii = p.cols[x[i]]
        p.cols[x[i]] = (nm, lo, up, float(-mu[i]), ii)
    return p


def process_rto(seed, nunit=8, nmv=6, ncv=5, nutil=3):
    """Steady-state process optimization -- the real-time-optimization layer that
    sits above a refinery's regulatory control and below the planner.

    Each process unit has a handful of manipulated variables (severity, reflux,
    feed rate, catalyst circulation) and a handful of controlled ones (a product
    property, a bed temperature, a column loading).  Around the current operating
    point the plant is described by its gain matrix -- the linearised steady
    state a plant test produces -- so

        y_u - G_u x_u = y_u^0

    is a linear equality, and the optimizer picks a new operating point subject
    to it.  Three terms go in the objective, and all three are quadratic for a
    physical reason rather than for convenience:

      * deviation of a controlled variable from target, squared, because being
        twice as far off spec is worse than twice as bad;
      * a move penalty on the manipulated variables, which is what stops the
        RTO handing the regulatory layer a step it cannot make;
      * utility consumption.  Compressor and pump power rise roughly with the
        square of throughput, so the energy term really is quadratic.

    The Hessian is deliberately rank deficient: a share of the controlled
    variables sit in a spec band but carry no target, so they appear in the
    constraints and not in Q at all.  That is the case a solver assuming a
    positive definite Q gets wrong.  The utility headers couple every unit, so
    the model does not decompose into `nunit` small ones.

    Feasible by construction: x = 0 is the linearisation point itself, which
    satisfies every spec band and every header limit."""
    rng = np.random.default_rng(4000 + seed)
    p = QP(f"QRTO{seed}")

    x, y = [], []
    for u in range(nunit):
        x.append([p.col(f"X{u}_{j}", -1.0, 1.0) for j in range(nmv)])   # scaled moves
        y.append([p.col(f"Y{u}_{i}", -1.2, 1.2) for i in range(ncv)])   # spec bands

    for u in range(nunit):                                   # y - G x = y0
        G = rng.normal(0.0, 0.9, size=(ncv, nmv))
        y0 = rng.uniform(-0.3, 0.3, size=ncv)
        for i in range(ncv):
            r = p.row(f"G{u}_{i}", float(y0[i]), float(y0[i]))
            p.add(r, y[u][i], 1.0)
            for j in range(nmv):
                p.add(r, x[u][j], -float(G[i, j]))

    pw = []                                                  # utility headers
    for k in range(nutil):
        w = p.col(f"PW{k}", 0.0, INF)
        pw.append(w)
        r = p.row(f"UTIL{k}", 0.0, 0.0)                      # P = sum a_j x_j
        p.add(r, w, 1.0)
        for u in range(nunit):
            for j in range(nmv):
                if rng.random() < 0.55:
                    p.add(r, x[u][j], -float(rng.uniform(0.05, 0.6)))
        rr = p.row(f"UCAP{k}", -INF, float(rng.uniform(1.5, 4.0)))
        p.add(rr, w, 1.0)

    def obj(col, v):
        nm, lo, up, _, ii = p.cols[col]
        p.cols[col] = (nm, lo, up, float(v), ii)

    for u in range(nunit):
        for i in range(ncv):
            if rng.random() < 0.4:
                continue                                     # spec only, no target
            wgt = float(rng.uniform(1.0, 6.0)); tgt = float(rng.uniform(-0.5, 0.5))
            p.quad(y[u][i], y[u][i], 2.0 * wgt)              # wgt * (y - tgt)^2
            obj(y[u][i], -2.0 * wgt * tgt)
        for j in range(nmv):
            p.quad(x[u][j], x[u][j], 2.0 * float(rng.uniform(0.05, 0.8)))
            obj(x[u][j], rng.uniform(-1.2, 1.2))
    for k in range(nutil):
        p.quad(pw[k], pw[k], 2.0 * float(rng.uniform(0.6, 2.4)))   # kappa * P^2
        obj(pw[k], rng.uniform(0.3, 1.5))
    return p


SUITE_QP = [
    ("qblend_s",  lambda s: blend_fit(s, 24, 6, 2),  "Blend property tracking"),
    ("qblend_m",  lambda s: blend_fit(s, 40, 8, 3),  "Blend property tracking"),
    ("qsmooth_s", lambda s: smooth(s, 40, 3),        "Production smoothing"),
    ("qsmooth_m", lambda s: smooth(s, 80, 4),        "Production smoothing"),
    ("qrisk_s",   lambda s: risk(s, 60, 8),          "Factor-model risk"),
    ("qrisk_m",   lambda s: risk(s, 150, 12),        "Factor-model risk"),
    ("qrto_s",    lambda s: process_rto(s, 6, 5, 4, 2),   "Process optimization (RTO)"),
    ("qrto_m",    lambda s: process_rto(s, 14, 8, 6, 3),  "Process optimization (RTO)"),
    ("qrto_l",    lambda s: process_rto(s, 150, 12, 10, 5), "Process optimization (RTO)"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    for tag, fn, family in SUITE_QP:
        for s in range(a.seeds):
            q = fn(s)
            path = os.path.join(a.outdir, f"{tag}_{s}.mps")
            q.write(path)
            print(f"{tag}_{s:<3} {family:<26} {len(q.rows):>5} rows {len(q.cols):>5} cols "
                  f"{q.nnz():>7} nnz {len(q.q):>7} quad")


if __name__ == "__main__":
    main()
