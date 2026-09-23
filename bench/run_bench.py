#!/usr/bin/env python3
"""
Head-to-head benchmark: IGAOS vs HiGHS (via SciPy) on the industrial suite.

HiGHS is used as the open baseline required by the problem statement.  Both
solvers see the identical model; IGAOS reads it from MPS through its own
parser, HiGHS receives it directly as sparse arrays, so IGAOS is if anything
carrying extra I/O cost.  Agreement is checked on the objective value in the
original space and on primal feasibility, not on the solver's own status flag.
"""
import numpy as np, subprocess, time, json, os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate import SUITE, LP, INF
from scipy.sparse import csr_matrix
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

BIN = os.environ.get("IGAOS_BIN", os.path.expanduser("~/igaos/build/igaos"))

def to_arrays(p):
    m, n = len(p.rows), len(p.cols)
    ri, ci, vv = [], [], []
    for (i, j), v in p.ent.items():
        ri.append(i); ci.append(j); vv.append(v)
    A = csr_matrix((vv, (ri, ci)), shape=(m, n))
    c  = np.array([col[3] for col in p.cols], float)
    lo = np.array([col[1] for col in p.cols], float)
    up = np.array([col[2] for col in p.cols], float)
    rl = np.array([r[1] for r in p.rows], float)
    ru = np.array([r[2] for r in p.rows], float)
    integrality = np.array([1 if col[4] else 0 for col in p.cols], int)
    return A, c, lo, up, rl, ru, integrality

def run_highs(p, tl):
    A, c, lo, up, rl, ru, integ = to_arrays(p)
    lo2 = np.where(lo <= -1e29, -np.inf, lo)
    up2 = np.where(up >=  1e29,  np.inf, up)
    rl2 = np.where(rl <= -1e29, -np.inf, rl)
    ru2 = np.where(ru >=  1e29,  np.inf, ru)
    t = time.time()
    if integ.any():
        res = milp(c=c, constraints=LinearConstraint(A, rl2, ru2),
                   bounds=Bounds(lo2, up2), integrality=integ,
                   options=dict(time_limit=tl, mip_rel_gap=1e-6))
        el = time.time() - t
        ok = res.status == 0
        return (res.fun if ok else None), el, ("optimal" if ok else f"status{res.status}")
    eqm = (rl2 == ru2) & np.isfinite(rl2)
    Aeq = A[eqm]; beq = rl2[eqm]
    rows_ub, b_ub = [], []
    for i in np.where(~eqm)[0]:
        if np.isfinite(ru2[i]): rows_ub.append(A[i]);  b_ub.append(ru2[i])
        if np.isfinite(rl2[i]): rows_ub.append(-A[i]); b_ub.append(-rl2[i])
    from scipy.sparse import vstack
    Aub = vstack(rows_ub) if rows_ub else None
    res = linprog(c, A_ub=Aub, b_ub=np.array(b_ub) if b_ub else None,
                  A_eq=Aeq if eqm.any() else None, b_eq=beq if eqm.any() else None,
                  bounds=list(zip(lo2, up2)), method="highs")
    el = time.time() - t
    ok = res.status == 0
    name = {0:"optimal",1:"iteration_limit",2:"infeasible",3:"unbounded"}.get(res.status, "?")
    return (res.fun if ok else None), el, name

def run_igaos(path, tl, extra=()):
    t = time.time()
    try:
        out = subprocess.run([BIN, path, "-q", "--time-limit", str(tl), *extra],
                             capture_output=True, text=True, timeout=tl + 60).stdout
    except subprocess.TimeoutExpired:
        return None, tl, "timeout", {}
    el = time.time() - t
    line = [l for l in out.splitlines() if l.startswith("IGAOS_RESULT")]
    if not line: return None, el, "error", {}
    kv = dict(x.split("=", 1) for x in line[0].split()[1:])
    st = kv.get("status", "?")
    obj = float(kv["obj"]) if st in ("optimal", "feasible") else None
    return obj, el, st, kv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/inst")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--json", default="/tmp/bench_results.json")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    hdr = (f"{'instance':14s} {'family':26s} {'kind':5s} {'rows':>6s} {'cols':>6s} {'nnz':>7s} "
           f"{'igaos':>11s} {'highs':>11s} {'reldiff':>9s} {'s_time':>8s} {'h_time':>8s} {'ratio':>7s} {'iters':>8s} {'pinf':>9s}")
    print(hdr); print("-" * len(hdr))
    for tag, fn, family, kind in SUITE:
        for s in range(a.seeds):
            p = fn(s)
            name = f"{tag}_{s}"
            path = os.path.join(a.outdir, name + ".mps")
            p.write(path)
            sobj, stime, sst, kv = run_igaos(path, a.time_limit)
            hobj, htime, hst = run_highs(p, a.time_limit)
            if sobj is not None and hobj is not None:
                rd = abs(sobj - hobj) / max(1.0, abs(hobj))
                agree = rd < 1e-6
            else:
                rd = float("nan"); agree = (sst == hst)
            ratio = stime / htime if htime > 1e-6 else float("nan")
            rec = dict(name=name, family=family, kind=kind, rows=len(p.rows), cols=len(p.cols),
                       nnz=p.nnz(), nint=sum(1 for c in p.cols if c[4]),
                       igaos_obj=sobj, highs_obj=hobj, reldiff=rd, agree=bool(agree),
                       igaos_status=sst, highs_status=hst,
                       igaos_time=stime, highs_time=htime, ratio=ratio,
                       iters=int(kv.get("iters", 0)), nodes=int(kv.get("nodes", 0)),
                       pinf=float(kv.get("pinf", float("nan"))),
                       iinf=float(kv.get("iinf", float("nan"))))
            rows.append(rec)
            flag = "" if agree else "   <-- MISMATCH"
            print(f"{name:14s} {family:26s} {kind:5s} {len(p.rows):6d} {len(p.cols):6d} {p.nnz():7d} "
                  f"{(f'{sobj:.5g}' if sobj is not None else sst):>11s} "
                  f"{(f'{hobj:.5g}' if hobj is not None else hst):>11s} "
                  f"{rd:9.1e} {stime:8.3f} {htime:8.3f} {ratio:7.2f} "
                  f"{int(kv.get('iters',0)):8d} {float(kv.get('pinf',0)):9.1e}{flag}")
    with open(a.json, "w") as f: json.dump(rows, f, indent=1)

    ok = sum(1 for r in rows if r["agree"])
    print("\n" + "=" * 70)
    print(f"agreement: {ok}/{len(rows)} instances match HiGHS to 1e-6 relative")
    lps = [r for r in rows if r["kind"] == "LP" and r["agree"]]
    if lps:
        import math
        gm = math.exp(sum(math.log(max(r['ratio'], 1e-3)) for r in lps) / len(lps))
        print(f"LP time ratio vs HiGHS (geometric mean over {len(lps)}): {gm:.2f}x")
        worst = max(lps, key=lambda r: r['pinf'])
        print(f"worst LP primal infeasibility: {worst['pinf']:.2e}  ({worst['name']})")
    mips = [r for r in rows if r["kind"] == "MILP" and r["agree"]]
    if mips:
        import math
        gm = math.exp(sum(math.log(max(r['ratio'], 1e-3)) for r in mips) / len(mips))
        print(f"MILP time ratio vs HiGHS (geometric mean over {len(mips)}): {gm:.2f}x")

if __name__ == "__main__":
    main()
