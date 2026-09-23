#!/usr/bin/env python3
"""
Measure the mixed-integer convex QP family.

The deck carried a "200 / 200 mixed-integer QP, proven optimal" figure that no
file in bench/ supported -- it had never been run. This is the harness that
produces the number, so the claim is a measurement or it is not made.

    python3 bench/generate_miqp.py benchmarks/miqp --seeds 40
    IGAOS_BIN=./build/igaos python3 bench/miqp_bench.py --dir benchmarks/miqp
"""
import argparse, csv, glob, os, re, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--time-limit", type=float, default=60.0)
ap.add_argument("--out", default="bench/results_miqp.csv")
a = ap.parse_args()
B = os.environ.get("IGAOS_BIN", "./build/igaos")

rows = []
for p in sorted(glob.glob(os.path.join(a.dir, "*.mps"))):
    t0 = time.time()
    try:
        cp = subprocess.run([B, p, "-q", "--time-limit", "%g" % a.time_limit],
                            capture_output=True, text=True, timeout=a.time_limit + 60)
        o = cp.stdout + cp.stderr
    except subprocess.TimeoutExpired:
        o = "status=timeout"
    el = time.time() - t0
    g = lambda k: (re.search(k + r"=(\S+)", o) or [None, ""])[1]
    rows.append(dict(name=os.path.basename(p), status=g("status"), obj=g("obj"),
                     gap=g("gap"), nodes=g("nodes"), time="%.3f" % el))
    print(rows[-1], flush=True)

with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
n = sum(1 for r in rows if r["status"] == "optimal")
print("PROVEN OPTIMAL %d / %d" % (n, len(rows)))
