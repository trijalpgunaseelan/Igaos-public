#!/usr/bin/env python3
"""
Compare the IGAOS convex-QP path against OSQP on the same MPS files.

Quadratic programming is the third class the problem statement names, and until
this harness existed it was only ever checked against IGAOS's own tests — which
is no check at all. OSQP is an independent, widely used convex QP solver with a
completely different method (operator splitting / ADMM against an interior point
method), so agreement between the two is real evidence.

    python3 bench/generate_qp.py benchmarks/qp --seeds 3
    IGAOS_BIN=./build/igaos python3 bench/qp_bench.py --dir benchmarks/qp

A note on what "agreement" means here. OSQP is a first-order method: it reaches
moderate accuracy quickly and high accuracy slowly, and its own documentation
says so. It is run below at eps_abs = eps_rel = 1e-9 with polishing on, which
gets it close, but a disagreement in the seventh digit is as likely to be OSQP
as IGAOS. The comparison is therefore reported at 1e-6 relative, and the raw
objectives are printed so the reader can judge.
"""
import argparse, csv, glob, os, re, subprocess, sys, time
import numpy as np

RESULT = re.compile(r"^IGAOS_RESULT\s+(.*)$", re.M)


def igaos(binary, path, tl, extra=()):
    t0 = time.time()
    try:
        cp = subprocess.run([binary, path, "-q", "--time-limit", "%g" % tl] + list(extra),
                            capture_output=True, text=True, timeout=tl + 60)
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


def osqp_solve(q, tl):
    import osqp
    import scipy.sparse as sp
    n = q.ncol
    P = q.Q if q.Q is not None else sp.csc_matrix((n, n))
    P = sp.triu(sp.csc_matrix(P), format="csc")          # OSQP wants the upper triangle
    # variable bounds become rows, which is how OSQP takes them
    A = sp.vstack([sp.csc_matrix(q.A), sp.eye(n, format="csc")], format="csc")
    lo = np.concatenate([q.rl, q.lb])
    hi = np.concatenate([q.ru, q.ub])
    m = osqp.OSQP()
    m.setup(P=P, q=q.c, A=A, l=lo, u=hi, verbose=False,
            eps_abs=1e-9, eps_rel=1e-9, max_iter=400000, polish=True,
            time_limit=tl)
    t0 = time.time()
    r = m.solve()
    el = time.time() - t0
    st = str(r.info.status)
    if "solved" not in st.lower():
        return {"status": st, "time": el}
    x = np.asarray(r.x)
    obj = float(q.c @ x + 0.5 * x @ (q.Q @ x) if q.Q is not None else q.c @ x) + q.c0
    return {"status": "optimal", "obj": obj, "time": el, "x": x,
            "polished": bool(getattr(r.info, "status_polish", 0) == 1)}


def violation(q, x):
    ax = q.A @ x
    v = max(np.max(np.maximum(q.rl - ax, 0.0), initial=0.0),
            np.max(np.maximum(ax - q.ru, 0.0), initial=0.0),
            np.max(np.maximum(q.lb - x, 0.0), initial=0.0),
            np.max(np.maximum(x - q.ub, 0.0), initial=0.0))
    return float(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--out", default="qp_results.csv")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to the solver verbatim")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mpsread import read_mps

    binary = os.environ.get("IGAOS_BIN", "./build/igaos")
    files = sorted(sum((glob.glob(os.path.join(a.dir, e))
                        for e in ("*.mps", "*.MPS", "*.qps", "*.QPS")), []))
    print(f"{len(files)} convex QP instances · reference OSQP at 1e-9 with polishing\n")
    print(f"{'instance':<14}{'rows':>6}{'cols':>6}{'quad nnz':>10}  "
          f"{'IGAOS':>10}{'objective':>18}  {'OSQP':>10}{'objective':>18}"
          f"{'rel diff':>10}{'t_igaos':>9}{'t_osqp':>9}  verdict")
    print("-" * 138)

    rows, agree, disagree = [], 0, 0
    for path in files:
        n = os.path.splitext(os.path.basename(path))[0]
        q = read_mps(path)
        g = igaos(binary, path, a.time_limit, a.extra_args)
        h = osqp_solve(q, a.time_limit)

        rel, verdict = None, ""
        if g.get("status") == "optimal" and h.get("status") == "optimal":
            rel = abs(g["obj"] - h["obj"]) / (1.0 + abs(h["obj"]))
            if rel <= a.tol: agree, verdict = agree + 1, ""
            else:            disagree, verdict = disagree + 1, "OBJECTIVE MISMATCH"
        else:
            verdict = "status differs (%s / %s)" % (g.get("status"), h.get("status"))

        print(f"{n:<14}{q.nrow:>6}{q.ncol:>6}{(q.Q.nnz if q.Q is not None else 0):>10}  "
              f"{g.get('status','?'):>10}"
              f"{(('%.10g' % g['obj']) if isinstance(g.get('obj'),float) else '—'):>18}  "
              f"{h.get('status','?')[:10]:>10}"
              f"{(('%.10g' % h['obj']) if isinstance(h.get('obj'),float) else '—'):>18}"
              f"{(('%.1e' % rel) if rel is not None else '—'):>10}"
              f"{g.get('time',0):>9.2f}{h.get('time',0):>9.2f}  {verdict}")
        sys.stdout.flush()
        rows.append(dict(name=n, rows=q.nrow, cols=q.ncol,
                         quad_nnz=(q.Q.nnz if q.Q is not None else 0),
                         igaos_status=g.get("status"), igaos_obj=g.get("obj"),
                         igaos_pinf=g.get("pinf"),
                         osqp_status=h.get("status"), osqp_obj=h.get("obj"),
                         osqp_violation=(violation(q, h["x"]) if "x" in h else None),
                         rel_diff=rel, igaos_time=g.get("time"), osqp_time=h.get("time"),
                         verdict=verdict or "agree"))

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    both = [r for r in rows if r["rel_diff"] is not None]
    print("\n" + "=" * 70)
    print(f"agreeing with OSQP ......... {agree} / {len(rows)}")
    if disagree:
        print(f"OBJECTIVE MISMATCHES ....... {disagree}")
    if both:
        worst = max(both, key=lambda r: r["rel_diff"])
        print(f"worst relative difference .. {worst['rel_diff']:.2e}  ({worst['name']})")
        wp = max((r for r in rows if r["igaos_pinf"] is not None),
                 key=lambda r: r["igaos_pinf"])
        print(f"worst IGAOS infeasibility .. {wp['igaos_pinf']:.2e}  ({wp['name']})")
        wo = [r for r in rows if r["osqp_violation"] is not None]
        if wo:
            w2 = max(wo, key=lambda r: r["osqp_violation"])
            print(f"worst OSQP  infeasibility .. {w2['osqp_violation']:.2e}  ({w2['name']})")
        ti = sum(r["igaos_time"] or 0 for r in both)
        to = sum(r["osqp_time"] or 0 for r in both)
        print(f"total time ................. IGAOS {ti:.2f} s · OSQP {to:.2f} s")
    print(f"\nwritten to {a.out}")
    return 1 if disagree else 0


if __name__ == "__main__":
    sys.exit(main())
