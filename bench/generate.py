#!/usr/bin/env python3
"""
IGAOS benchmark instance generators.

Builds MPS instances representative of the industrial problem classes named in
the problem statement -- refinery crude blending, multi-period refinery
production planning, unit commitment / power dispatch, and multi-echelon supply
chain flow -- plus two families constructed specifically to stress numerical
robustness: massively degenerate transportation models and blending models with
coefficient magnitudes spanning eight orders.

Every generator returns a model in the ranged-row form
    rl <= A x <= ru,  l <= x <= u
and is constructed around a known feasible point so the instances are feasible
by construction.
"""
import numpy as np, os, sys

INF = 1e30

class LP:
    def __init__(self, name):
        self.name = name
        self.cols = []          # (name, lo, up, obj, is_int)
        self.rows = []          # (name, lo, up)
        self.ent  = {}          # (i,j) -> value
    def col(self, nm, lo=0.0, up=INF, obj=0.0, integer=False):
        self.cols.append((nm, lo, up, obj, integer)); return len(self.cols)-1
    def row(self, nm, lo=-INF, up=INF):
        self.rows.append((nm, lo, up)); return len(self.rows)-1
    def add(self, i, j, v):
        if v == 0: return
        self.ent[(i,j)] = self.ent.get((i,j), 0.0) + v
    def nnz(self): return len(self.ent)
    def write(self, path):
        bycol = {}
        for (i,j),v in self.ent.items():
            bycol.setdefault(j, []).append((i,v))
        L = [f"NAME          {self.name[:8]}", "ROWS", " N  COST"]
        for i,(nm,lo,up) in enumerate(self.rows):
            lf, uf = lo > -1e29, up < 1e29
            if lf and uf and lo == up: k = 'E'
            elif uf: k = 'L'
            elif lf: k = 'G'
            else:    k = 'N'
            L.append(f" {k}  {nm}")
        L.append("COLUMNS")
        inint = False; mk = 0
        for j,(nm,lo,up,obj,ii) in enumerate(self.cols):
            if ii and not inint: L.append(f"    MARKER{mk}  'MARKER'  'INTORG'"); mk+=1; inint=True
            if not ii and inint: L.append(f"    MARKER{mk}  'MARKER'  'INTEND'"); mk+=1; inint=False
            if obj != 0: L.append(f"    {nm}  COST  {float(obj):.12g}")
            for i,v in sorted(bycol.get(j, [])):
                L.append(f"    {nm}  {self.rows[i][0]}  {float(v):.12g}")
        if inint: L.append(f"    MARKER{mk}  'MARKER'  'INTEND'")
        L.append("RHS")
        for i,(nm,lo,up) in enumerate(self.rows):
            lf, uf = lo > -1e29, up < 1e29
            if lf or uf:
                b = up if uf else lo
                L.append(f"    RHS  {nm}  {float(b):.12g}")
        L.append("RANGES")
        for i,(nm,lo,up) in enumerate(self.rows):
            if lo > -1e29 and up < 1e29 and lo != up:
                L.append(f"    RNG  {nm}  {float(up-lo):.12g}")
        L.append("BOUNDS")
        for j,(nm,lo,up,obj,ii) in enumerate(self.cols):
            if lo == 0 and up >= 1e29 and not ii: continue
            if lo <= -1e29 and up >= 1e29: L.append(f" FR BND  {nm}"); continue
            if lo == up: L.append(f" FX BND  {nm}  {float(lo):.12g}"); continue
            if lo <= -1e29: L.append(f" MI BND  {nm}")
            elif lo != 0:   L.append(f" LO BND  {nm}  {float(lo):.12g}")
            if up < 1e29:   L.append(f" UP BND  {nm}  {float(up):.12g}")
        L.append("ENDATA")
        with open(path, "w") as f: f.write("\n".join(L) + "\n")
        return path

# ---------------------------------------------------------------------------
def refinery_blend(seed=0, n_crude=40, n_pool=12, n_prod=14, n_prop=6):
    """Crude blending through intermediate pools into finished products with
    quality specifications.  Linear (volume-basis) property blending, which is
    the standard LP relaxation of the pooling problem used in refinery planning."""
    rng = np.random.default_rng(seed)
    p = LP(f"BLEND{seed}")
    prop = rng.uniform(0.1, 3.0, (n_crude, n_prop))     # crude qualities
    cost = rng.uniform(40, 95, n_crude)
    avail = rng.uniform(200, 1500, n_crude)
    demand = rng.uniform(80, 400, n_prod)
    price = rng.uniform(100, 190, n_prod)

    # variables
    f = {}   # crude -> pool
    for c in range(n_crude):
        for k in range(n_pool):
            if rng.random() < 0.35:
                f[(c,k)] = p.col(f"F{c}_{k}", 0.0, avail[c], cost[c])
    g = {}   # pool -> product
    for k in range(n_pool):
        for q in range(n_prod):
            if rng.random() < 0.5:
                g[(k,q)] = p.col(f"G{k}_{q}", 0.0, INF, -price[q])
    # ensure connectivity
    for c in range(n_crude):
        if not any((c,k) in f for k in range(n_pool)):
            k = int(rng.integers(n_pool)); f[(c,k)] = p.col(f"F{c}_{k}", 0.0, avail[c], cost[c])
    for q in range(n_prod):
        if not any((k,q) in g for k in range(n_pool)):
            k = int(rng.integers(n_pool)); g[(k,q)] = p.col(f"G{k}_{q}", 0.0, INF, -price[q])

    # crude availability
    for c in range(n_crude):
        r = p.row(f"AV{c}", -INF, avail[c])
        for k in range(n_pool):
            if (c,k) in f: p.add(r, f[(c,k)], 1.0)
    # pool balance
    for k in range(n_pool):
        r = p.row(f"PB{k}", 0.0, 0.0)
        for c in range(n_crude):
            if (c,k) in f: p.add(r, f[(c,k)], 1.0)
        for q in range(n_prod):
            if (k,q) in g: p.add(r, g[(k,q)], -1.0)
    # product demand
    for q in range(n_prod):
        r = p.row(f"DM{q}", demand[q]*0.55, demand[q]*2.4)
        for k in range(n_pool):
            if (k,q) in g: p.add(r, g[(k,q)], 1.0)
    # quality specs: sum_k (poolprop_k * g) <= spec * sum_k g  -- linearised on
    # the pool-average property, which is the standard planning approximation
    poolprop = np.zeros((n_pool, n_prop))
    for k in range(n_pool):
        src = [c for c in range(n_crude) if (c,k) in f]
        poolprop[k] = prop[src].mean(axis=0) if src else prop.mean(axis=0)
    for q in range(n_prod):
        conn = [k for k in range(n_pool) if (k,q) in g]
        if not conn: continue
        for t in range(n_prop):
            # Set each specification against the pools actually feeding this
            # product, so at least one qualifying blend always exists: the
            # instance is hard but feasible by construction.
            spec = float(np.percentile(poolprop[conn, t], 65))
            r = p.row(f"Q{q}_{t}", -INF, 0.0)
            for k in conn: p.add(r, g[(k,q)], float(poolprop[k,t] - spec))
    return p

# ---------------------------------------------------------------------------
def production_planning(seed=0, n_unit=25, n_prod=18, n_period=24):
    """Multi-period refinery/process production planning: unit capacities,
    yields, inventory balance and time-varying demand.  Staircase structure,
    which is where a triangular crash basis and hyper-sparse pricing pay off."""
    rng = np.random.default_rng(seed + 500)
    p = LP(f"PROD{seed}")
    yld = rng.uniform(0.15, 0.95, (n_unit, n_prod)) * (rng.random((n_unit, n_prod)) < 0.4)
    for u in range(n_unit):
        if not yld[u].any(): yld[u, int(rng.integers(n_prod))] = 0.7
    cap = rng.uniform(60, 260, n_unit)
    rcost = rng.uniform(4, 22, n_unit)
    hcost = rng.uniform(0.4, 2.5, n_prod)
    dem = rng.uniform(8, 60, (n_prod, n_period))

    run = np.empty((n_unit, n_period), dtype=int)
    inv = np.empty((n_prod, n_period), dtype=int)
    sal = np.empty((n_prod, n_period), dtype=int)
    for u in range(n_unit):
        for t in range(n_period):
            run[u,t] = p.col(f"R{u}_{t}", 0.0, cap[u], rcost[u])
    for q in range(n_prod):
        for t in range(n_period):
            inv[q,t] = p.col(f"I{q}_{t}", 0.0, 900.0, hcost[q])
            sal[q,t] = p.col(f"S{q}_{t}", 0.0, INF, -rng.uniform(30, 75))
    for q in range(n_prod):
        for t in range(n_period):
            r = p.row(f"BAL{q}_{t}", 0.0, 0.0)
            for u in range(n_unit):
                if yld[u,q] != 0: p.add(r, run[u,t], float(yld[u,q]))
            if t > 0: p.add(r, inv[q,t-1], 1.0)
            p.add(r, inv[q,t], -1.0)
            p.add(r, sal[q,t], -1.0)
        for t in range(n_period):
            r = p.row(f"DEM{q}_{t}", -INF, float(dem[q,t]))
            p.add(r, sal[q,t], 1.0)
    for t in range(n_period):
        r = p.row(f"CAP{t}", -INF, float(cap.sum() * 0.72))
        for u in range(n_unit): p.add(r, run[u,t], 1.0)
    return p

# ---------------------------------------------------------------------------
def unit_commitment(seed=0, n_gen=14, n_period=24):
    """Power system unit commitment: on/off binaries, generation limits linked
    to commitment, ramp limits, minimum up/down time and a demand balance.
    Weak LP relaxation and heavy symmetry -- the hard MILP class."""
    rng = np.random.default_rng(seed + 900)
    p = LP(f"UC{seed}")
    pmin = rng.uniform(8, 40, n_gen)
    pmax = pmin + rng.uniform(40, 160, n_gen)
    ccost = rng.uniform(120, 700, n_gen)
    gcost = rng.uniform(9, 42, n_gen)
    ramp = rng.uniform(25, 90, n_gen)
    peak = pmax.sum() * 0.62
    dem = peak * (0.62 + 0.34*np.sin(np.linspace(0, 2*np.pi, n_period)) + 0.05*rng.random(n_period))

    on = np.empty((n_gen, n_period), dtype=int)
    gp = np.empty((n_gen, n_period), dtype=int)
    su = np.empty((n_gen, n_period), dtype=int)
    for u in range(n_gen):
        for t in range(n_period):
            on[u,t] = p.col(f"U{u}_{t}", 0.0, 1.0, 0.0, integer=True)
            gp[u,t] = p.col(f"P{u}_{t}", 0.0, float(pmax[u]), float(gcost[u]))
            su[u,t] = p.col(f"V{u}_{t}", 0.0, 1.0, float(ccost[u]), integer=True)
    for u in range(n_gen):
        for t in range(n_period):
            r = p.row(f"MIN{u}_{t}", 0.0, INF)          # p >= pmin * on
            p.add(r, gp[u,t], 1.0); p.add(r, on[u,t], -float(pmin[u]))
            r = p.row(f"MAX{u}_{t}", -INF, 0.0)         # p <= pmax * on
            p.add(r, gp[u,t], 1.0); p.add(r, on[u,t], -float(pmax[u]))
            if t > 0:
                r = p.row(f"SU{u}_{t}", -INF, 0.0)      # on_t - on_{t-1} <= v_t
                p.add(r, on[u,t], 1.0); p.add(r, on[u,t-1], -1.0); p.add(r, su[u,t], -1.0)
                r = p.row(f"RU{u}_{t}", -INF, float(ramp[u]))
                p.add(r, gp[u,t], 1.0); p.add(r, gp[u,t-1], -1.0)
                r = p.row(f"RD{u}_{t}", -INF, float(ramp[u]))
                p.add(r, gp[u,t-1], 1.0); p.add(r, gp[u,t], -1.0)
    for t in range(n_period):
        r = p.row(f"BAL{t}", float(dem[t]), INF)
        for u in range(n_gen): p.add(r, gp[u,t], 1.0)
        r = p.row(f"RES{t}", float(dem[t]*1.12), INF)   # spinning reserve
        for u in range(n_gen): p.add(r, on[u,t], float(pmax[u]))
    return p

# ---------------------------------------------------------------------------
def supply_chain(seed=0, n_src=30, n_hub=18, n_dst=45):
    """Two-echelon minimum cost flow.  Transportation polytopes are massively
    primal degenerate, which is exactly the stalling regime the ratio test and
    the anti-cycling strategy have to survive."""
    rng = np.random.default_rng(seed + 1300)
    p = LP(f"SC{seed}")
    sup = rng.uniform(50, 300, n_src)
    dem = rng.uniform(20, 120, n_dst)
    dem *= (sup.sum()*0.8) / dem.sum()
    a = {}; b = {}
    for s in range(n_src):
        for h in range(n_hub):
            if rng.random() < 0.45: a[(s,h)] = p.col(f"A{s}_{h}", 0.0, INF, float(rng.uniform(1,14)))
    for h in range(n_hub):
        for d in range(n_dst):
            if rng.random() < 0.45: b[(h,d)] = p.col(f"B{h}_{d}", 0.0, INF, float(rng.uniform(1,14)))
    for s in range(n_src):
        if not any((s,h) in a for h in range(n_hub)):
            h = int(rng.integers(n_hub)); a[(s,h)] = p.col(f"A{s}_{h}", 0.0, INF, float(rng.uniform(1,14)))
    for d in range(n_dst):
        if not any((h,d) in b for h in range(n_hub)):
            h = int(rng.integers(n_hub)); b[(h,d)] = p.col(f"B{h}_{d}", 0.0, INF, float(rng.uniform(1,14)))
    for h in range(n_hub):
        if not any((s,h) in a for s in range(n_src)):
            s = int(rng.integers(n_src)); a[(s,h)] = p.col(f"A{s}_{h}", 0.0, INF, float(rng.uniform(1,14)))
        if not any((h,d) in b for d in range(n_dst)):
            d = int(rng.integers(n_dst)); b[(h,d)] = p.col(f"B{h}_{d}", 0.0, INF, float(rng.uniform(1,14)))
    for s in range(n_src):
        r = p.row(f"SUP{s}", -INF, float(sup[s]))
        for h in range(n_hub):
            if (s,h) in a: p.add(r, a[(s,h)], 1.0)
    for h in range(n_hub):
        r = p.row(f"HUB{h}", 0.0, 0.0)
        for s in range(n_src):
            if (s,h) in a: p.add(r, a[(s,h)], 1.0)
        for d in range(n_dst):
            if (h,d) in b: p.add(r, b[(h,d)], -1.0)
    for d in range(n_dst):
        r = p.row(f"DEM{d}", float(dem[d]), INF)
        for h in range(n_hub):
            if (h,d) in b: p.add(r, b[(h,d)], 1.0)
    return p

# ---------------------------------------------------------------------------
def ill_conditioned_blend(seed=0, n=180, m=140, spread=8):
    """Blending-style LP whose coefficients span `spread` orders of magnitude --
    trace metals in ppm alongside volumes in kilotonnes.  Unscaled this has a
    condition number around 10^spread; it is the direct test of the scaling and
    threshold-pivoting requirement in the problem statement."""
    rng = np.random.default_rng(seed + 2100)
    p = LP(f"ILL{seed}")
    x0 = rng.uniform(0.5, 4.0, n)
    cols = [p.col(f"X{j}", 0.0, float(x0[j]*3), float(rng.uniform(-6, 6))) for j in range(n)]
    for i in range(m):
        mag = 10.0 ** rng.uniform(-spread/2, spread/2)
        idx = rng.choice(n, size=int(rng.integers(3, 9)), replace=False)
        coef = rng.uniform(0.5, 2.0, len(idx)) * mag
        act = float(np.dot(coef, x0[idx]))
        if i % 3 == 0:   r = p.row(f"R{i}", act, act)
        elif i % 3 == 1: r = p.row(f"R{i}", -INF, act*1.15)
        else:            r = p.row(f"R{i}", act*0.85, INF)
        for k, j in enumerate(idx): p.add(r, cols[j], float(coef[k]))
    return p

# ---------------------------------------------------------------------------
def degenerate_transport(seed=0, n_src=45, n_dst=45):
    """Balanced transportation problem with many equal supplies and demands, so
    a large fraction of basic variables sit at zero: heavy primal degeneracy."""
    rng = np.random.default_rng(seed + 3300)
    p = LP(f"DEG{seed}")
    base = 100.0
    sup = np.full(n_src, base); dem = np.full(n_dst, base * n_src / n_dst)
    x = {}
    for s in range(n_src):
        for d in range(n_dst):
            x[(s,d)] = p.col(f"X{s}_{d}", 0.0, INF, float(rng.integers(1, 6)))
    for s in range(n_src):
        r = p.row(f"S{s}", float(sup[s]), float(sup[s]))
        for d in range(n_dst): p.add(r, x[(s,d)], 1.0)
    for d in range(n_dst):
        r = p.row(f"D{d}", float(dem[d]), float(dem[d]))
        for s in range(n_src): p.add(r, x[(s,d)], 1.0)
    return p

# ---------------------------------------------------------------------------
def refinery_schedule(seed=0, n_unit=6, n_mode=3, n_period=12, n_crude=3,
                      n_inter=5, n_prod=4, min_run=2):
    """Discrete-time refinery scheduling: which operating MODE each process unit
    runs in each time slot, how much it charges, and what sits in the tanks
    between them.

    This is the scheduling problem that sits under the planning LP.  The planner
    (`refinery_blend`, `production_planning`) decides volumes over a month; the
    scheduler decides the order and the switches over a week, and the switches
    are what make it combinatorial:

      * one mode per unit per slot -- max-gasoline, max-distillate, off -- as a
        set-partitioning row, so the relaxation splits fractionally across modes
        and has to be branched;
      * a changeover binary that fires when the mode changes, carrying a real
        cost, because swinging a cat cracker is hours of off-spec product;
      * a minimum run length, so a mode entered has to be held -- the same
        structure as minimum up-time in unit commitment, and just as weak in the
        relaxation;
      * semi-continuous charge rates: a unit is off, or it runs at or above its
        turndown, never between;
      * tank inventory balances with finite capacity linking every slot to the
        next, which is what stops the slots decoupling into T independent
        problems.

    Materials flow crude -> intermediates -> products.  Tier-one units charge a
    crude, tier-two units charge an intermediate, and the yield vector depends
    on the mode: that dependence is the whole point, and it is what a fixed-yield
    planning LP cannot express.

    Feasible by construction: mode 0 is `off` for every unit, and unmet demand
    carries a shortfall variable at a penalty price, so the all-idle schedule is
    always feasible and the model is bounded.  A refinery scheduler writes the
    same shortfall term, for the same reason."""
    rng = np.random.default_rng(seed + 4700)
    p = LP(f"RSCH{seed}")
    U, M, T = n_unit, n_mode, n_period
    NC, NI, NP = n_crude, n_inter, n_prod
    K = NC + NI + NP                      # materials, in that order
    crude = list(range(NC))
    inter = list(range(NC, NC + NI))
    prod  = list(range(NC + NI, K))

    # --- topology: first half of the units charge crude, the rest intermediates
    tier1 = list(range(0, max(1, U // 2)))
    feed  = {}
    for u in range(U):
        feed[u] = int(rng.choice(crude)) if u in tier1 else int(rng.choice(inter))

    # --- per-unit, per-mode data.  Mode 0 is OFF: zero rate, zero yield.
    fmin = np.zeros((U, M)); fmax = np.zeros((U, M))
    ocost = np.zeros((U, M)); chg = np.zeros((U, M))
    yld = np.zeros((U, M, K))
    for u in range(U):
        cap = float(rng.uniform(90, 260))
        for m in range(1, M):
            fmax[u, m] = cap * float(rng.uniform(0.75, 1.0))
            fmin[u, m] = fmax[u, m] * float(rng.uniform(0.35, 0.55))   # turndown
            ocost[u, m] = float(rng.uniform(3.0, 11.0))
            chg[u, m] = float(rng.uniform(400, 2600))
            out = inter if u in tier1 else prod
            share = rng.dirichlet(np.ones(len(out)) * 0.8)
            total = float(rng.uniform(0.94, 0.99))                     # process loss
            for i, k in enumerate(out):
                yld[u, m, k] = float(share[i]) * total

    pcost = rng.uniform(48, 88, NC)               # crude purchase price
    price = rng.uniform(115, 205, NP)             # product netback
    hold  = rng.uniform(0.15, 0.9, K)             # tank holding cost
    tcap  = rng.uniform(300, 1400, K)             # tank capacity
    dem   = rng.uniform(40, 170, (NP, T))         # per-slot product lifting
    penal = float(price.max() * 3.0)              # shortfall penalty

    # --- variables
    y  = np.empty((U, M, T), dtype=int)           # mode selection   (binary)
    z  = np.empty((U, M, T), dtype=int)           # switch INTO mode (binary)
    fm = np.empty((U, M, T), dtype=int)           # charge rate in that mode
    for u in range(U):
        for m in range(M):
            for t in range(T):
                y[u,m,t] = p.col(f"Y{u}_{m}_{t}", 0.0, 1.0, 0.0, integer=True)
                z[u,m,t] = p.col(f"Z{u}_{m}_{t}", 0.0, 1.0, float(chg[u,m]), integer=True)
                fm[u,m,t] = p.col(f"F{u}_{m}_{t}", 0.0, float(fmax[u,m]), float(ocost[u,m]))
    inv = np.empty((K, T), dtype=int)
    for k in range(K):
        for t in range(T):
            inv[k,t] = p.col(f"I{k}_{t}", 0.0, float(tcap[k]), float(hold[k]))
    buy = np.empty((NC, T), dtype=int)
    for c in range(NC):
        for t in range(T):
            buy[c,t] = p.col(f"B{c}_{t}", 0.0, INF, float(pcost[c]))
    sell = np.empty((NP, T), dtype=int); short = np.empty((NP, T), dtype=int)
    for q in range(NP):
        for t in range(T):
            sell[q,t]  = p.col(f"S{q}_{t}", 0.0, float(dem[q,t]), float(-price[q]))
            short[q,t] = p.col(f"H{q}_{t}", 0.0, float(dem[q,t]), penal)

    # --- one mode per unit per slot
    for u in range(U):
        for t in range(T):
            r = p.row(f"ASG{u}_{t}", 1.0, 1.0)
            for m in range(M): p.add(r, y[u,m,t], 1.0)

    # --- semi-continuous charge: fmin*y <= f <= fmax*y
    for u in range(U):
        for m in range(M):
            for t in range(T):
                r = p.row(f"UB{u}_{m}_{t}", -INF, 0.0)
                p.add(r, fm[u,m,t], 1.0); p.add(r, y[u,m,t], -float(fmax[u,m]))
                if fmin[u,m] > 0:
                    r = p.row(f"LB{u}_{m}_{t}", 0.0, INF)
                    p.add(r, fm[u,m,t], 1.0); p.add(r, y[u,m,t], -float(fmin[u,m]))

    # --- changeover: z_t >= y_t - y_{t-1}.  Every unit starts the horizon idle,
    #     so entering any running mode in slot 0 is itself a changeover.
    for u in range(U):
        for m in range(M):
            for t in range(T):
                r = p.row(f"CH{u}_{m}_{t}", -INF, 0.0)
                p.add(r, y[u,m,t], 1.0); p.add(r, z[u,m,t], -1.0)
                if t > 0: p.add(r, y[u,m,t-1], -1.0)
                elif m == 0: p.rows[r] = (f"CH{u}_{m}_{t}", -INF, 1.0)   # idle at t=-1

    # --- minimum run length: a mode entered is held for min_run slots
    if min_run > 1:
        for u in range(U):
            for m in range(1, M):
                for t in range(T):
                    win = range(t, min(t + min_run, T))
                    n = len(list(win))
                    if n < 2: continue
                    r = p.row(f"MR{u}_{m}_{t}", 0.0, INF)
                    for tau in win: p.add(r, y[u,m,tau], 1.0)
                    p.add(r, z[u,m,t], -float(n))

    # --- tank balance:  I_t - I_{t-1} = in - out
    for k in range(K):
        for t in range(T):
            r = p.row(f"BAL{k}_{t}", 0.0, 0.0)
            p.add(r, inv[k,t], -1.0)
            if t > 0: p.add(r, inv[k,t-1], 1.0)
            if k < NC: p.add(r, buy[k,t], 1.0)
            for u in range(U):
                for m in range(1, M):
                    if yld[u,m,k] != 0.0: p.add(r, fm[u,m,t], float(yld[u,m,k]))
                    if feed[u] == k:      p.add(r, fm[u,m,t], -1.0)
            if k >= NC + NI: p.add(r, sell[k - NC - NI, t], -1.0)

    # --- lifting: every slot's demand is either sold or short
    for q in range(NP):
        for t in range(T):
            r = p.row(f"DEM{q}_{t}", float(dem[q,t]), float(dem[q,t]))
            p.add(r, sell[q,t], 1.0); p.add(r, short[q,t], 1.0)
    return p

# ---------------------------------------------------------------------------
SUITE = [
    ("blend_s",   lambda s: refinery_blend(s, 40, 12, 14, 6),        "Refinery crude blending",  "LP"),
    ("blend_m",   lambda s: refinery_blend(s, 120, 30, 40, 8),       "Refinery crude blending",  "LP"),
    ("blend_l",   lambda s: refinery_blend(s, 300, 60, 90, 10),      "Refinery crude blending",  "LP"),
    ("prod_s",    lambda s: production_planning(s, 15, 12, 12),      "Production planning",      "LP"),
    ("prod_m",    lambda s: production_planning(s, 25, 18, 24),      "Production planning",      "LP"),
    ("prod_l",    lambda s: production_planning(s, 40, 30, 36),      "Production planning",      "LP"),
    ("chain_s",   lambda s: supply_chain(s, 30, 18, 45),             "Supply chain flow",        "LP"),
    ("chain_m",   lambda s: supply_chain(s, 70, 40, 110),            "Supply chain flow",        "LP"),
    ("chain_l",   lambda s: supply_chain(s, 140, 70, 220),           "Supply chain flow",        "LP"),
    ("degen_s",   lambda s: degenerate_transport(s, 30, 30),         "Degenerate transport",     "LP"),
    ("degen_m",   lambda s: degenerate_transport(s, 60, 60),         "Degenerate transport",     "LP"),
    ("illcond_s", lambda s: ill_conditioned_blend(s, 180, 140, 8),   "Ill-conditioned blending", "LP"),
    ("illcond_m", lambda s: ill_conditioned_blend(s, 420, 340, 8),   "Ill-conditioned blending", "LP"),
    ("uc_s",      lambda s: unit_commitment(s, 6, 12),               "Unit commitment",          "MILP"),
    ("uc_m",      lambda s: unit_commitment(s, 10, 16),              "Unit commitment",          "MILP"),
    ("uc_l",      lambda s: unit_commitment(s, 14, 24),              "Unit commitment",          "MILP"),
    ("sched_s",   lambda s: refinery_schedule(s, 4, 3,  8, 2, 4, 3, 2), "Refinery scheduling",   "MILP"),
    ("sched_m",   lambda s: refinery_schedule(s, 6, 3, 12, 3, 5, 4, 2), "Refinery scheduling",   "MILP"),
    ("sched_l",   lambda s: refinery_schedule(s, 8, 4, 16, 3, 6, 5, 3), "Refinery scheduling",   "MILP"),
]

def main(outdir="instances", seeds=(0,)):
    os.makedirs(outdir, exist_ok=True)
    manifest = []
    for tag, fn, family, kind in SUITE:
        for s in seeds:
            p = fn(s)
            name = f"{tag}_{s}"
            path = os.path.join(outdir, name + ".mps")
            p.write(path)
            nint = sum(1 for c in p.cols if c[4])
            manifest.append(dict(name=name, path=path, family=family, kind=kind,
                                 rows=len(p.rows), cols=len(p.cols), nnz=p.nnz(), nint=nint))
            print(f"{name:14s} {family:26s} {kind:4s} rows={len(p.rows):6d} cols={len(p.cols):6d} nnz={p.nnz():7d} int={nint}")
    import json
    with open(os.path.join(outdir, "manifest.json"), "w") as f: json.dump(manifest, f, indent=1)
    return manifest

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "instances")
