#!/usr/bin/env python3
"""
Run IGAOS over the Netlib *infeasible* LP collection.

This set exists because the interesting failure is not "slow", it is "confident
and wrong". A solver that returns `optimal` on a model with no feasible point
has handed a planner a schedule that cannot be executed. The 28 instances here
are all provably infeasible, several of them by margins small enough that a
loose feasibility tolerance will swallow them.

    IGAOS_BIN=./build/igaos python3 bench/infeasible.py --dir benchmarks/netlib_infeasible
"""
import argparse, glob, os, re, subprocess, sys, time

RESULT = re.compile(r"^IGAOS_RESULT\s+(.*)$", re.M)


def run(binary, path, tl):
    try:
        cp = subprocess.run([binary, path, "-q", "--time-limit", "%g" % tl],
                            capture_output=True, text=True, timeout=tl + 60)
    except subprocess.TimeoutExpired:
        return {"status": "hard-timeout"}
    m = RESULT.search(cp.stdout or "")
    if not m:
        return {"status": "error"}
    d = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:    d[k] = float(v)
            except ValueError: d[k] = v
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--time-limit", type=float, default=150.0)
    a = ap.parse_args()
    binary = os.environ.get("IGAOS_BIN", "./build/igaos")

    files = sorted(glob.glob(os.path.join(a.dir, "*.mps")))
    print(f"{len(files)} provably infeasible instances · time limit {a.time_limit:g} s\n")
    print(f"{'instance':<14}{'reported':>14}{'time':>9}{'residual':>12}   note")
    print("-" * 78)

    right, wrong, other = 0, [], []
    for p in files:
        n = os.path.splitext(os.path.basename(p))[0]
        d = run(binary, p, a.time_limit)
        st = d.get("status", "?")
        note = ""
        if st == "infeasible":
            right += 1
        elif st == "optimal":
            wrong.append(n)
            note = "WRONG — reports an optimum for a model with no feasible point"
        else:
            other.append(n)
            note = "did not conclude"
        print(f"{n:<14}{st:>14}{d.get('time',0):>9.2f}"
              f"{(('%.2e' % d['pinf']) if 'pinf' in d else '—'):>12}   {note}")
        sys.stdout.flush()

    print("\n" + "=" * 70)
    print(f"correctly reported infeasible .. {right} / {len(files)}")
    if wrong:
        print(f"reported optimal (WRONG) ....... {len(wrong)}: {', '.join(wrong)}")
    if other:
        print(f"inconclusive ................... {len(other)}: {', '.join(other)}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
