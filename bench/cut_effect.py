#!/usr/bin/env python3
"""Measure what cut separation actually bought, instance by instance.

Runs the same mixed-integer instances twice -- once with the root cutting-plane
loop disabled and once with it on -- and reports the proven status, the gap, the
node count and the wall time for each.  The headline number a solver quotes for
cuts is meaningless without the paired run it is compared against, so this is
the paired run.

Usage:
    IGAOS_BIN=../build/igaos python3 cut_effect.py /tmp/inst/uc_*.mps
"""
import json
import os
import subprocess
import sys
import time

BIN = os.environ.get("IGAOS_BIN", os.path.expanduser("~/igaos/build/igaos"))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", "120"))


def run(path, cuts):
    args = [BIN, path, "-q", "--time-limit", str(TIME_LIMIT)]
    if not cuts:
        args.append("--no-cuts")
    t = time.time()
    try:
        out = subprocess.run(args, capture_output=True, text=True,
                             timeout=TIME_LIMIT + 60).stdout
    except subprocess.TimeoutExpired:
        return dict(status="timeout", time=TIME_LIMIT + 60)
    elapsed = time.time() - t
    line = [l for l in out.splitlines() if l.startswith("IGAOS_RESULT")]
    if not line:
        return dict(status="error", time=elapsed)
    kv = dict(x.split("=", 1) for x in line[0].split()[1:])
    obj = float(kv["obj"])
    bound = float(kv["bound"])
    gap = abs(obj - bound) / max(1.0, abs(obj))
    return dict(status=kv.get("status", "?"), obj=obj, bound=bound, gap=gap,
                nodes=int(kv.get("nodes", 0)), iters=int(kv.get("iters", 0)),
                cuts=int(kv.get("cuts", 0)), cutrounds=int(kv.get("cutrounds", 0)),
                rootlp=float(kv.get("rootlp", 0.0)),
                rootcut=float(kv.get("rootcut", 0.0)),
                time=elapsed)


def main(paths):
    rows = []
    hdr = (f"{'instance':12s} | {'no cuts: status':>16s} {'gap':>9s} {'nodes':>9s} {'time':>8s}"
           f" | {'with cuts: status':>18s} {'gap':>9s} {'nodes':>9s} {'time':>8s} {'cuts':>5s}"
           f" {'root closed':>12s}")
    print(hdr)
    print("-" * len(hdr))
    proven_off = proven_on = 0
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        off = run(path, cuts=False)
        on = run(path, cuts=True)
        proven_off += off.get("status") == "optimal"
        proven_on += on.get("status") == "optimal"

        closed = ""
        if on.get("cuts", 0) > 0 and on.get("obj") is not None:
            denom = on["obj"] - on["rootlp"]
            if abs(denom) > 1e-9:
                closed = f"{100.0 * (on['rootcut'] - on['rootlp']) / denom:10.1f}%"
        print(f"{name:12s} | {off.get('status','?'):>16s} {off.get('gap',float('nan')):9.2%}"
              f" {off.get('nodes',0):9d} {off.get('time',0):8.2f}"
              f" | {on.get('status','?'):>18s} {on.get('gap',float('nan')):9.2%}"
              f" {on.get('nodes',0):9d} {on.get('time',0):8.2f} {on.get('cuts',0):5d}"
              f" {closed:>12s}")
        rows.append(dict(name=name, without_cuts=off, with_cuts=on))

    print()
    print(f"proven optimal without cuts: {proven_off}/{len(paths)}")
    print(f"proven optimal with cuts:    {proven_on}/{len(paths)}")
    with open(os.environ.get("JSON_OUT", "/tmp/cut_effect.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
