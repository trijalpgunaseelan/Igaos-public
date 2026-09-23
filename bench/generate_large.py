#!/usr/bin/env python3
"""
Industrial-scale instances: hundreds of thousands of variables, millions of
nonzeros.

The problem statement asks for models with "thousands to millions of variables
and constraints". The families in bench/generate.py are sized to run a full
benchmark sweep in minutes; these are sized to answer a different question --
does the thing stand up at industrial scale, and where does the time actually
go when it does.

Two families, both multi-period, because that is what makes industrial models
large: the same network repeated over a planning horizon, tied together by
inventory.

  chain    Multi-echelon supply chain over T periods: plants -> hubs -> demand
           points, with inventory carried at the hubs. Sparse and very wide.

  refinery Multi-period refinery production planning: units, modes, blending
           and product inventory over T periods. Denser rows than `chain`, and
           the shape MRPL's own scheduling models take.

    python3 bench/generate_large.py out.mps --family chain --periods 60 --scale 3

Writing a two-million-nonzero MPS file takes longer than solving it, so the
generator reports its own time separately.
"""
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import LP, INF


def chain(periods, scale, seed=0):
    """Plants -> hubs -> demand, with inventory across periods."""
    rng = np.random.default_rng(seed)
    nplant, nhub, ndem = 20 * scale, 40 * scale, 120 * scale
    p = LP(f"CHAIN{periods}x{scale}")

    # arcs are fixed across periods; the sparsity comes from each hub serving
    # only a slice of the demand points, which is how real networks look
    hub_of = [rng.choice(nhub, size=3, replace=False) for _ in range(ndem)]
    plant_of = [rng.choice(nplant, size=2, replace=False) for _ in range(nhub)]

    hubs_of_plant = [[] for _ in range(nplant)]
    for h in range(nhub):
        for pl in plant_of[h]:
            hubs_of_plant[pl].append(h)
    hubs_serving = [[] for _ in range(nhub)]
    for d in range(ndem):
        for h in hub_of[d]:
            hubs_serving[h].append(d)

    ship_ph, ship_hd, inv = {}, {}, {}
    for t in range(periods):
        for h in range(nhub):
            for pl in plant_of[h]:
                ship_ph[(t, pl, h)] = p.col(f"P{t}_{pl}_{h}", 0.0, 400.0,
                                            float(rng.uniform(1.0, 4.0)))
            inv[(t, h)] = p.col(f"I{t}_{h}", 0.0, 900.0, float(rng.uniform(0.1, 0.6)))
        for d in range(ndem):
            for h in hub_of[d]:
                ship_hd[(t, h, d)] = p.col(f"H{t}_{h}_{d}", 0.0, 300.0,
                                           float(rng.uniform(1.0, 5.0)))

    for t in range(periods):
        for pl in range(nplant):                    # plant capacity
            r = p.row(f"CAP{t}_{pl}", -INF, float(rng.uniform(500, 900)))
            for h in hubs_of_plant[pl]:
                p.add(r, ship_ph[(t, pl, h)], 1.0)
        for h in range(nhub):                       # hub balance with inventory
            r = p.row(f"BAL{t}_{h}", 0.0, 0.0)
            for pl in plant_of[h]:
                p.add(r, ship_ph[(t, pl, h)], 1.0)
            if t > 0:
                p.add(r, inv[(t - 1, h)], 1.0)
            p.add(r, inv[(t, h)], -1.0)
            for d in hubs_serving[h]:
                p.add(r, ship_hd[(t, h, d)], -1.0)
        for d in range(ndem):                       # demand
            r = p.row(f"DEM{t}_{d}", float(rng.uniform(20, 60)), INF)
            for h in hub_of[d]:
                p.add(r, ship_hd[(t, h, d)], 1.0)
    return p


def refinery(periods, scale, seed=0):
    """Multi-period crude processing, unit modes, blending and inventory."""
    rng = np.random.default_rng(100 + seed)
    ncrude, nunit, nmode, nprod = 12 * scale, 8 * scale, 4, 10 * scale
    p = LP(f"REFIN{periods}x{scale}")

    yields = rng.uniform(0.02, 0.35, size=(nunit, nmode, nprod))
    buy, run, blend, inv = {}, {}, {}, {}
    for t in range(periods):
        for c in range(ncrude):
            buy[(t, c)] = p.col(f"B{t}_{c}", 0.0, 600.0, float(rng.uniform(30, 70)))
        for u in range(nunit):
            for md in range(nmode):
                run[(t, u, md)] = p.col(f"R{t}_{u}_{md}", 0.0, 500.0,
                                        float(rng.uniform(1, 6)))
        for k in range(nprod):
            blend[(t, k)] = p.col(f"S{t}_{k}", 0.0, INF, -float(rng.uniform(80, 150)))
            inv[(t, k)] = p.col(f"V{t}_{k}", 0.0, 400.0, float(rng.uniform(0.2, 0.9)))

    for t in range(periods):
        r = p.row(f"CRUDE{t}", -INF, float(rng.uniform(1500, 2600)))
        for c in range(ncrude):
            p.add(r, buy[(t, c)], 1.0)
        for u in range(nunit):                       # unit capacity across modes
            r = p.row(f"UCAP{t}_{u}", -INF, float(rng.uniform(300, 600)))
            for md in range(nmode):
                p.add(r, run[(t, u, md)], 1.0)
        r = p.row(f"FEED{t}", 0.0, 0.0)              # crude in = throughput out
        for c in range(ncrude):
            p.add(r, buy[(t, c)], 1.0)
        for u in range(nunit):
            for md in range(nmode):
                p.add(r, run[(t, u, md)], -1.0)
        for k in range(nprod):                       # product balance + inventory
            r = p.row(f"PBAL{t}_{k}", 0.0, 0.0)
            for u in range(nunit):
                for md in range(nmode):
                    y = yields[u, md, k]
                    if y > 0.08:                      # keep the row sparse
                        p.add(r, run[(t, u, md)], float(y))
            if t > 0:
                p.add(r, inv[(t - 1, k)], 1.0)
            p.add(r, inv[(t, k)], -1.0)
            p.add(r, blend[(t, k)], -1.0)
            rr = p.row(f"DMD{t}_{k}", -INF, float(rng.uniform(60, 160)))
            p.add(rr, blend[(t, k)], 1.0)
    return p


FAMILIES = {"chain": chain, "refinery": refinery}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--family", choices=sorted(FAMILIES), default="chain")
    ap.add_argument("--periods", type=int, default=60)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    t0 = time.time()
    p = FAMILIES[a.family](a.periods, a.scale, a.seed)
    build = time.time() - t0
    t1 = time.time()
    p.write(a.output)
    write = time.time() - t1

    print(f"{a.family} · {a.periods} periods · scale {a.scale}")
    print(f"  rows      {len(p.rows):>12,}")
    print(f"  columns   {len(p.cols):>12,}")
    print(f"  nonzeros  {p.nnz():>12,}")
    print(f"  file      {os.path.getsize(a.output) / 1e6:>9.1f} MB   "
          f"built in {build:.1f} s, written in {write:.1f} s")


if __name__ == "__main__":
    main()
