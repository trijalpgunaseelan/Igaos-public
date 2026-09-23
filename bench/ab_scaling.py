#!/usr/bin/env python3
"""
A/B the [A; Q] equilibration against the A-only rule.

    IGAOS_BIN=./build/igaos python3 bench/ab_scaling.py \
        ./build/igaos benchmarks/qp_maros 15 bench/results_maros_scaleq_ab.csv

Runs every instance TWICE in the same process -- once with the default scaling
and once with --scale-with-q -- so the two arms differ by exactly one flag and
nothing else.  Records the status, the objective and the wall time for each.

Measured 20 September 2026 on the 138 Maros-Meszaros convex QPs, 15 s limit:
gained HUES-MOD, HUESTIS and QCAPRI; lost QSCFXM3; net 108 -> 110.  Five of the
107 both-solved instances disagree by more than 1e-6, CONT-300 by 18 percent --
see HANDOFF-REPORT section 16.6, defect 27.
"""

import csv, glob, os, re, subprocess, sys, time
RESULT = re.compile(r"^IGAOS_RESULT\s+(.*)$", re.M)
BIN, DIR, TL, OUT = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
def run(path, extra):
    t0 = time.time()
    try:
        cp = subprocess.run([BIN, path, "-q", "--time-limit", "%g" % TL] + extra,
                            capture_output=True, text=True, timeout=TL + 30)
    except subprocess.TimeoutExpired:
        return {"status": "hard-timeout", "obj": "", "time": time.time() - t0}
    m = RESULT.search(cp.stdout or ""); d = {}
    if m:
        for tok in m.group(1).split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                d[k] = v
    return {"status": d.get("status", "error"), "obj": d.get("obj", ""),
            "time": round(time.time() - t0, 3)}
rows = []
files = sorted(glob.glob(os.path.join(DIR, "*")))
for i, p in enumerate(files, 1):
    a = run(p, []); b = run(p, ["--scale-with-q"])
    rows.append({"name": os.path.basename(p),
                 "base_status": a["status"], "base_obj": a["obj"], "base_time": a["time"],
                 "q_status": b["status"], "q_obj": b["obj"], "q_time": b["time"]})
    print(f"{i:3d}/{len(files)} {os.path.basename(p):<16} {a['status']:<16} {b['status']}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\ndone")
