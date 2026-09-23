#!/usr/bin/env python3
"""
Run IGAOS over the Netlib LP test set and check every objective against a
reference value.

The Netlib collection is the standard correctness test for a linear programming
code: 114 feasible models, many of them deliberately degenerate or badly scaled,
with optimal objectives that have been agreed on for decades. Passing it is not
a performance claim, it is the entry ticket.

    python3 bench/netlib.py --dir /path/to/netlib --ref netlib_reference.csv

The reference CSV needs two columns, `name` and `pobj`. The values shipped with
this repository were produced by Gurobi 10 at a 1e-8 tolerance.
"""
import argparse, csv, os, re, subprocess, sys, time

RESULT = re.compile(r"^IGAOS_RESULT\s+(.*)$", re.M)

# One reference value in the mirrored CSV disagrees with the value Netlib itself
# publishes, and with what HiGHS returns when handed the same file. Rather than
# edit the downloaded reference, the correction is applied here in the open.
#
#   forplan: the CSV says -1163.915769.  The PROBLEM SUMMARY TABLE in
#   https://www.netlib.org/lp/data/readme gives -6.6421873953E+02, and HiGHS,
#   fed the same MPS file through bench/mpsread.py, returns -664.2189613.
#   Two independent sources against one, so the CSV entry is the outlier.
KNOWN_REFERENCE_CORRECTIONS = {
    "forplan": (-664.21873953, "netlib.org/lp/data/readme; corroborated by HiGHS"),
}


def parse(out):
    m = RESULT.search(out)
    if not m:
        return None
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
    return d


def run(binary, path, tl, algorithm=None, extra=()):
    argv = [binary, path, "-q", "--time-limit", "%g" % tl]
    if algorithm:
        argv += ["--algorithm", algorithm]
    argv += list(extra)
    t0 = time.time()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=tl + 120)
    except subprocess.TimeoutExpired:
        return {"status": "hard-timeout", "time": time.time() - t0}
    d = parse(cp.stdout)
    if d is None:
        head = (cp.stderr or cp.stdout or "").strip().splitlines()
        return {"status": "error", "time": time.time() - t0,
                "msg": head[0][:70] if head else "no output"}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory of .mps instances")
    ap.add_argument("--ref", required=True, help="CSV with name,pobj")
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--tol", type=float, default=1e-6, help="relative objective tolerance")
    ap.add_argument("--algorithm", default=None)
    ap.add_argument("--out", default="netlib_results.csv")
    ap.add_argument("--only", default=None, help="comma-separated subset")
    ap.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to the solver verbatim")
    a = ap.parse_args()

    binary = os.environ.get("IGAOS_BIN", "./build/igaos")
    ref = {r["name"]: float(r["pobj"]) for r in csv.DictReader(open(a.ref))}
    for n, (v, why) in KNOWN_REFERENCE_CORRECTIONS.items():
        if n in ref and abs(ref[n] - v) > 1e-6 * (1 + abs(v)):
            print(f"note: reference for {n} corrected from {ref[n]:.10g} to {v:.10g}  ({why})")
            ref[n] = v
    names = sorted(ref)
    if a.only:
        want = set(a.only.split(","))
        names = [n for n in names if n in want]

    print(f"{len(names)} instances · time limit {a.time_limit:g} s · tolerance {a.tol:g} relative\n")
    print(f"{'instance':<12}{'rows':>7}{'cols':>8}{'status':>12}"
          f"{'objective':>20}{'reference':>20}{'rel diff':>11}{'iters':>9}{'time':>9}   verdict")
    print("-" * 122)

    rows, matched, missed, unsolved = [], 0, 0, []
    for n in names:
        path = os.path.join(a.dir, n + ".mps")
        if not os.path.exists(path):
            print(f"{n:<12}{'':>7}{'':>8}{'missing':>12}")
            continue
        d = run(binary, path, a.time_limit, a.algorithm, a.extra_args)
        st = d.get("status", "?")
        obj, r = d.get("obj"), ref[n]
        rel = abs(obj - r) / (1.0 + abs(r)) if isinstance(obj, float) else float("inf")
        if st == "optimal" and rel <= a.tol:
            verdict, matched = "match", matched + 1
        elif st == "optimal":
            verdict, missed = "OBJECTIVE MISMATCH", missed + 1
        else:
            verdict = "not solved"
            unsolved.append(n)
        rows.append(dict(name=n, status=st, objective=obj, reference=r, rel_diff=rel,
                         iterations=d.get("iters"), time=d.get("time", d.get("time", 0)),
                         primal_infeas=d.get("pinf"), verdict=verdict))
        print(f"{n:<12}{'':>7}{'':>8}{st:>12}"
              f"{(('%.10g' % obj) if isinstance(obj,float) else '—'):>20}"
              f"{'%.10g' % r:>20}{('%.1e' % rel) if rel < 1e30 else '—':>11}"
              f"{(('%d' % d['iters']) if 'iters' in d else '—'):>9}"
              f"{(('%.2f' % d.get('time',0))):>9}   "
              f"{'' if verdict=='match' else verdict}")
        sys.stdout.flush()

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    solved = [r for r in rows if r["status"] == "optimal"]
    print("\n" + "=" * 70)
    print(f"solved to optimality ......... {len(solved)} / {len(rows)}")
    print(f"objective matches reference .. {matched} / {len(rows)}")
    if missed:
        print(f"OBJECTIVE MISMATCHES ......... {missed}")
    if unsolved:
        print(f"not solved ................... {len(unsolved)}: {', '.join(unsolved)}")
    if solved:
        worst = max(solved, key=lambda r: r["rel_diff"])
        wpinf = max(solved, key=lambda r: r["primal_infeas"] or 0)
        tot = sum(r["time"] or 0 for r in solved)
        print(f"worst relative difference .... {worst['rel_diff']:.2e}  ({worst['name']})")
        print(f"worst primal infeasibility ... {wpinf['primal_infeas']:.2e}  ({wpinf['name']})")
        print(f"total solve time ............. {tot:.1f} s over {len(solved)} instances")
    print(f"\nwritten to {a.out}")
    return 1 if (missed or unsolved) else 0


if __name__ == "__main__":
    sys.exit(main())
