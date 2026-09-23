#!/usr/bin/env python3
"""
IGAOS against the commercial solvers the problem statement names.

Until now every comparison in this project was against open source -- HiGHS and
OSQP -- because we had no commercial licence.  Two of the three solvers PS 26119
names ship a free, size-limited licence that is the real product, not a
re-implementation:

    CPLEX Community Edition   pip install cplex     1000 rows x 1000 columns
    Gurobi restricted licence pip install gurobipy  2000 rows x 2000 columns

Those limits are the whole caveat and they are stated in the output rather than
buried: everything here is a SMALL model, which is exactly where a mature
commercial solver's advantages are smallest.  Nothing in this file supports a
claim about how IGAOS behaves at industrial scale against CPLEX or Gurobi.  It
supports one claim only, and it is the claim that was missing: the answers
agree.

    python3 bench/commercial.py --dir benchmarks/standard
    python3 bench/commercial.py --dir demo/models --time-limit 30

Writes results_commercial.csv and prints a summary.
"""
import argparse, csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpsread

CPLEX_ROWS = CPLEX_COLS = 1000
GUROBI_ROWS = GUROBI_COLS = 2000


# --------------------------------------------------------------------- IGAOS
def run_igaos(path, binary, tl):
    t0 = time.time()
    try:
        p = subprocess.run([binary, path, "-q", "--no-live", "--time-limit", str(tl)],
                           capture_output=True, text=True, timeout=tl + 60)
    except subprocess.TimeoutExpired:
        return dict(status="hard_timeout", obj=None, time=tl)
    wall = time.time() - t0
    m = re.search(r"IGAOS_RESULT status=(\S+) obj=(\S+)", p.stdout)
    if not m:
        return dict(status="no_result", obj=None, time=wall)
    return dict(status=m.group(1), obj=float(m.group(2)), time=wall)


# ------------------------------------------------------------------- helpers
def load(path):
    return mpsread.read_mps(path)


def is_min(p):
    """mpsread reports the sense as the string 'min' or 'max'."""
    return str(p.sense).lower().startswith("min")


# --------------------------------------------------------------------- CPLEX
def run_cplex(p, tl):
    import cplex
    from cplex.exceptions import CplexError
    c = cplex.Cplex()
    c.set_log_stream(None); c.set_results_stream(None)
    c.set_warning_stream(None); c.set_error_stream(None)
    c.parameters.timelimit.set(tl)
    # Both solvers stop at a default relative MIP gap of 1e-4. Leaving that in
    # place makes them report a suboptimal incumbent as "optimal" and turns an
    # apples-to-apples comparison into an argument about default settings.
    c.parameters.mip.tolerances.mipgap.set(1e-9)
    c.parameters.mip.tolerances.absmipgap.set(1e-9)
    c.objective.set_sense(c.objective.sense.minimize if is_min(p)
                          else c.objective.sense.maximize)
    n = p.ncol
    types = "".join("I" if p.integrality[j] else "C" for j in range(n))
    lb = [float(v) if v > -1e29 else -cplex.infinity for v in p.lb]
    ub = [float(v) if v < 1e29 else cplex.infinity for v in p.ub]
    if set(types) == {"C"}:
        c.variables.add(obj=[float(v) for v in p.c], lb=lb, ub=ub)
    else:
        c.variables.add(obj=[float(v) for v in p.c], lb=lb, ub=ub, types=types)

    A = p.A.tocsr()
    rows, senses, rhs, rng = [], [], [], []
    for i in range(p.nrow):
        s, e = A.indptr[i], A.indptr[i + 1]
        rows.append(cplex.SparsePair(ind=[int(k) for k in A.indices[s:e]],
                                     val=[float(v) for v in A.data[s:e]]))
        lo, hi = p.rl[i], p.ru[i]
        if lo > -1e29 and hi < 1e29:
            if lo == hi: senses.append("E"); rhs.append(float(lo)); rng.append(0.0)
            else:        senses.append("R"); rhs.append(float(lo)); rng.append(float(hi - lo))
        elif hi < 1e29:  senses.append("L"); rhs.append(float(hi)); rng.append(0.0)
        elif lo > -1e29: senses.append("G"); rhs.append(float(lo)); rng.append(0.0)
        else:            senses.append("L"); rhs.append(cplex.infinity); rng.append(0.0)
    c.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, range_values=rng)

    t0 = time.time()
    try:
        c.solve()
    except CplexError as e:
        return dict(status="error:" + str(e)[:40], obj=None, time=time.time() - t0)
    wall = time.time() - t0
    st = c.solution.get_status_string()
    try:
        obj = c.solution.get_objective_value() + p.c0
    except Exception:
        obj = None
    return dict(status=st, obj=obj, time=wall)


# -------------------------------------------------------------------- Gurobi
def run_gurobi(p, tl):
    import gurobipy as gp
    from gurobipy import GRB
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
    m.setParam("TimeLimit", tl)
    m.setParam("MIPGap", 1e-9)
    m.setParam("MIPGapAbs", 1e-9)
    lb = [float(v) if v > -1e29 else -GRB.INFINITY for v in p.lb]
    ub = [float(v) if v < 1e29 else GRB.INFINITY for v in p.ub]
    vt = [GRB.INTEGER if p.integrality[j] else GRB.CONTINUOUS for j in range(p.ncol)]
    x = m.addVars(p.ncol, lb=lb, ub=ub, vtype=vt)
    m.setObjective(gp.quicksum(float(p.c[j]) * x[j] for j in range(p.ncol)
                               if p.c[j] != 0.0),
                   GRB.MINIMIZE if is_min(p) else GRB.MAXIMIZE)
    A = p.A.tocsr()
    for i in range(p.nrow):
        s, e = A.indptr[i], A.indptr[i + 1]
        if e == s: continue
        expr = gp.quicksum(float(A.data[k]) * x[int(A.indices[k])] for k in range(s, e))
        lo, hi = p.rl[i], p.ru[i]
        if lo > -1e29 and hi < 1e29 and lo == hi: m.addConstr(expr == float(lo))
        else:
            if hi < 1e29:  m.addConstr(expr <= float(hi))
            if lo > -1e29: m.addConstr(expr >= float(lo))
    t0 = time.time()
    m.optimize()
    wall = time.time() - t0
    names = {GRB.OPTIMAL: "optimal", GRB.INFEASIBLE: "infeasible",
             GRB.UNBOUNDED: "unbounded", GRB.TIME_LIMIT: "time_limit",
             GRB.INF_OR_UNBD: "inf_or_unbounded"}
    st = names.get(m.Status, "status_%d" % m.Status)
    obj = None
    if m.SolCount > 0:
        obj = m.ObjVal + p.c0
    return dict(status=st, obj=obj, time=wall)


# ---------------------------------------------------------------------- main
def agree(a, b, tol=1e-6):
    if a is None or b is None: return None
    return abs(a - b) <= tol * (1.0 + max(abs(a), abs(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--bin", default=os.environ.get("IGAOS_BIN", "./build/igaos"))
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results_commercial.csv"))
    args = ap.parse_args()

    have_cplex = have_gurobi = True
    try: import cplex          # noqa: F401
    except Exception: have_cplex = False
    try: import gurobipy       # noqa: F401
    except Exception: have_gurobi = False
    if not (have_cplex or have_gurobi):
        print("Neither CPLEX nor Gurobi is importable. Install one:")
        print("    pip install cplex        # Community Edition, 1000x1000")
        print("    pip install gurobipy     # restricted licence, 2000x2000")
        return 1

    files = sorted(f for f in os.listdir(args.dir)
                   if f.lower().endswith((".mps", ".qps")))
    rows = []
    print("\nIGAOS against the solvers PS 26119 names.")
    print("Both licences are size limited, so every instance below is small --")
    print("which is where a commercial solver's advantage is smallest. That is a")
    print("statement about the licence, not about the solvers.")
    # Printed, not assumed. Left at their 1e-4 defaults both commercial solvers
    # report a suboptimal incumbent as "optimal" and this table shows false
    # disagreements -- which is exactly what happened on the first run. Anyone
    # reading a results file should be able to see the comparison was fair
    # without reading the source.
    print("\nAll three solvers run at a relative MIP gap of 1e-9. At the 1e-4")
    print("default, CPLEX and Gurobi stop early and the disagreements below")
    print("would be about settings rather than about answers.\n")
    hdr = f"{'instance':14s} {'r x c':>13s} {'int':>5s} | {'IGAOS':>14s} {'t':>7s} | {'CPLEX':>14s} {'t':>7s} | {'Gurobi':>14s} {'t':>7s} | verdict"
    print(hdr); print("-" * len(hdr))

    for f in files:
        path = os.path.join(args.dir, f)
        try:
            p = load(path)
        except Exception as e:
            print(f"{f[:-4]:14s}  unreadable: {str(e)[:40]}")
            continue
        nint = int(sum(1 for t in p.integrality if t))
        fits_c = have_cplex and p.nrow <= CPLEX_ROWS and p.ncol <= CPLEX_COLS
        fits_g = have_gurobi and p.nrow <= GUROBI_ROWS and p.ncol <= GUROBI_COLS
        if not (fits_c or fits_g):
            continue

        ig = run_igaos(path, args.bin, args.time_limit)
        cp = run_cplex(p, args.time_limit) if fits_c else dict(status="-", obj=None, time=None)
        gu = run_gurobi(p, args.time_limit) if fits_g else dict(status="-", obj=None, time=None)

        verdicts = []
        for other, name in ((cp, "cplex"), (gu, "gurobi")):
            if other["obj"] is None or ig["obj"] is None: continue
            if ig["status"] not in ("optimal", "feasible"): continue
            verdicts.append("agree " + name if agree(ig["obj"], other["obj"])
                            else "DIFFERS " + name)
        verdict = "; ".join(verdicts) if verdicts else "not comparable"

        def fmt(d):
            o = "-" if d["obj"] is None else f"{d['obj']:.8g}"
            t = "-" if d["time"] is None else f"{d['time']:.2f}s"
            return o, t
        io, it = fmt(ig); co, ct = fmt(cp); go, gt = fmt(gu)
        print(f"{f[:-4]:14s} {p.nrow:6d}x{p.ncol:<6d} {nint:5d} | "
              f"{io:>14s} {it:>7s} | {co:>14s} {ct:>7s} | {go:>14s} {gt:>7s} | {verdict}")

        rows.append(dict(name=f[:-4], rows=p.nrow, cols=p.ncol, integers=nint,
                         igaos_status=ig["status"], igaos_obj=ig["obj"], igaos_time=ig["time"],
                         cplex_status=cp["status"], cplex_obj=cp["obj"], cplex_time=cp["time"],
                         gurobi_status=gu["status"], gurobi_obj=gu["obj"], gurobi_time=gu["time"],
                         verdict=verdict))

    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    nc = sum(1 for r in rows if "agree cplex" in r["verdict"])
    ng = sum(1 for r in rows if "agree gurobi" in r["verdict"])
    bad = [r["name"] for r in rows if "DIFFERS" in r["verdict"]]
    print()
    print(f"  agree with CPLEX ....... {nc}")
    print(f"  agree with Gurobi ...... {ng}")
    print(f"  disagreements .......... {len(bad)}" + (("  " + ", ".join(bad)) if bad else ""))
    print(f"\n  written to {args.out}")
    print("\n  Read this as a correctness result, not a performance one. The free")
    print("  licences cap the models at a size where every solver here is fast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
