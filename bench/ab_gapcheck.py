#!/usr/bin/env python3
"""
A/B the duality-gap guard (defect 27) across the Maros-Meszaros set.

Arm A: default.  Arm B: --duality-gap-check.  The guard can only ever turn an
`optimal` into a `numerical_error`, so the only question this answers is how
many honest optima it costs to stop reporting the dishonest ones.
"""
import csv, glob, os, re, subprocess, sys, time
B = os.environ.get("IGAOS_BIN", "./build_audit/igaos")
TL = 15.0
def run(p, extra):
    try:
        cp = subprocess.run([B, p, "-q", "--time-limit", str(TL)] + extra,
                            capture_output=True, text=True, timeout=TL + 90)
        o = cp.stdout + cp.stderr
    except subprocess.TimeoutExpired:
        return ("timeout", "")
    st = re.search(r"status=(\S+)", o); ob = re.search(r"obj=(\S+)", o)
    return (st.group(1) if st else "?", ob.group(1) if ob else "")
rows = []
for f in sorted(glob.glob("benchmarks/qp_maros/*.QPS")):
    n = os.path.basename(f)[:-4]
    a = run(f, []); b = run(f, ["--duality-gap-check"])
    rows.append((n, a[0], a[1], b[0], b[1]))
    print(rows[-1], flush=True)
with open("bench/results_maros_gapcheck_ab.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["instance", "base_status", "base_obj", "gapcheck_status", "gapcheck_obj"])
    w.writerows(rows)
na = sum(1 for r in rows if r[1] == "optimal")
nb = sum(1 for r in rows if r[3] == "optimal")
lost = [r[0] for r in rows if r[1] == "optimal" and r[3] != "optimal"]
print("BASE %d / %d   GAPCHECK %d / %d   refused %d" % (na, len(rows), nb, len(rows), len(lost)))
print("REFUSED:", ", ".join(lost))
