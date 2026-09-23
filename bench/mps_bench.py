#!/usr/bin/env python3
"""
Run IGAOS and HiGHS over a directory of MPS files and compare them.

Unlike bench/run_bench.py, which builds its models in memory, this harness
reads real benchmark files off disk — MIPLIB, Netlib, QPLIB or anything else in
MPS form — and hands the same file to both solvers. The Python side parses the
file with bench/mpsread.py, written independently of the C++ reader, so a
disagreement between the two readers shows up as a disagreement in the results
rather than hiding inside a shared parser.

    IGAOS_BIN=./build/igaos python3 bench/mps_bench.py --dir /path/to/instances

HiGHS is reached through scipy.optimize.milp, which uses it for both linear and
mixed-integer models.
"""
import argparse, csv, glob, os, re, subprocess, sys, time

RESULT = re.compile(r"^IGAOS_RESULT\s+(.*)$", re.M)


def igaos(binary, path, tl):
    argv = [binary, path, "-q", "--time-limit", "%g" % tl]
    t0 = time.time()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=tl + 120)
    except subprocess.TimeoutExpired:
        return {"status": "hard-timeout", "time": time.time() - t0}
    m = RESULT.search(cp.stdout or "")
    if not m:
        return {"status": "error", "time": time.time() - t0}
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:    d[k] = float(v)
            except ValueError: d[k] = v
    return d


def highs(path, tl):
    from mpsread import read_mps
    from scipy.optimize import milp, LinearConstraint, Bounds
    import numpy as np
    q = read_mps(path)
    c = q.c if q.sense == "min" else -q.c
    t0 = time.time()
    r = milp(c=c, constraints=LinearConstraint(q.A, q.rl, q.ru),
             bounds=Bounds(q.lb, q.ub), integrality=q.integrality,
             options=dict(time_limit=tl))
    el = time.time() - t0
    if r.status == 0:
        obj = r.fun + q.c0 if q.sense == "min" else -(r.fun) + q.c0
        return {"status": "optimal", "obj": obj, "time": el, "problem": q}
    st = {1: "time_limit", 2: "infeasible", 3: "unbounded"}.get(r.status, "other")
    return {"status": st, "time": el, "problem": q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default="mps_bench_results.csv")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    binary = os.environ.get("IGAOS_BIN", "./build/igaos")
    files = sorted(glob.glob(os.path.join(a.dir, "*.mps")))
    print(f"{len(files)} instances · time limit {a.time_limit:g} s\n")
    print(f"{'instance':<14}{'rows':>7}{'cols':>7}{'int':>6}{'nnz':>8}  "
          f"{'IGAOS':>12}{'objective':>18}  {'HiGHS':>10}{'objective':>18}"
          f"{'rel diff':>10}{'t_igaos':>9}{'t_highs':>9}  verdict")
    print("-" * 142)

    rows, agree, disagree = [], 0, 0
    for path in files:
        n = os.path.splitext(os.path.basename(path))[0]
        h = highs(path, a.time_limit)
        q = h.get("problem")
        g = igaos(binary, path, a.time_limit)

        rel, verdict = None, ""
        if g.get("status") == "optimal" and h.get("status") == "optimal":
            rel = abs(g["obj"] - h["obj"]) / (1.0 + abs(h["obj"]))
            if rel <= a.tol: agree, verdict = agree + 1, ""
            else:            disagree, verdict = disagree + 1, "OBJECTIVE MISMATCH"
        elif g.get("status") == h.get("status"):
            agree, verdict = agree + 1, "(both %s)" % g.get("status")
        else:
            verdict = "status differs"

        print(f"{n:<14}{q.nrow:>7}{q.ncol:>7}{int(q.integrality.sum()):>6}{q.A.nnz:>8}  "
              f"{g.get('status','?'):>12}"
              f"{(('%.10g' % g['obj']) if isinstance(g.get('obj'),float) else '—'):>18}  "
              f"{h.get('status','?'):>10}"
              f"{(('%.10g' % h['obj']) if isinstance(h.get('obj'),float) else '—'):>18}"
              f"{(('%.1e' % rel) if rel is not None else '—'):>10}"
              f"{g.get('time',0):>9.2f}{h.get('time',0):>9.2f}  {verdict}")
        sys.stdout.flush()
        rows.append(dict(name=n, rows=q.nrow, cols=q.ncol,
                         integers=int(q.integrality.sum()), nnz=q.A.nnz,
                         igaos_status=g.get("status"), igaos_obj=g.get("obj"),
                         highs_status=h.get("status"), highs_obj=h.get("obj"),
                         rel_diff=rel, igaos_time=g.get("time"), highs_time=h.get("time"),
                         verdict=verdict or "agree"))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 70)
    print(f"agreeing with HiGHS ........ {agree} / {len(rows)}")
    if disagree:
        print(f"OBJECTIVE MISMATCHES ....... {disagree}")
    both = [r for r in rows if r["rel_diff"] is not None]
    if both:
        worst = max(both, key=lambda r: r["rel_diff"])
        print(f"worst relative difference .. {worst['rel_diff']:.2e}  ({worst['name']})")
        ti = sum(r["igaos_time"] or 0 for r in both)
        th = sum(r["highs_time"] or 0 for r in both)
        print(f"total time ................. IGAOS {ti:.1f} s · HiGHS {th:.1f} s")
    print(f"\nwritten to {a.out}")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
