#!/usr/bin/env python3
"""
Rebuild demo/models/ and its manifest.

The console ships its instances rather than generating them, so it works on a
machine with no numpy. This script is how they are made; it needs numpy, the
console does not.

    python3 demo/make_models.py

The set is chosen to cover every branch of the pipeline diagram, and to span
running times on purpose: a pipeline that always finishes in sixty milliseconds
teaches you nothing about where a solver spends its time. Three of these run
long enough to watch — the seconds column says which.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "models")
sys.path.insert(0, os.path.join(ROOT, "bench"))

from generate import (refinery_blend, production_planning, unit_commitment,
                      supply_chain, ill_conditioned_blend, degenerate_transport,
                      refinery_schedule)
from generate_qp import risk, smooth, process_rto
from generate_miqp import lots, card
from generate_pooling import haverly, pooling

# id, builder, kind, display name, blurb, roughly how long it runs and why
SET = [
    ("illcond_s", lambda: ill_conditioned_blend(0, 180, 140, 8), "LP",
     "Ill-conditioned blending",
     "Coefficients spanning eight orders of magnitude — built to break a solver's numerics.",
     "instant"),
    ("blend_s", lambda: refinery_blend(0, 40, 12, 14, 6), "LP",
     "Refinery crude blending",
     "A small refinery choosing how much of each crude to buy and how to route it "
     "through the units, subject to product quality specs.",
     "instant"),
    ("degen_s", lambda: degenerate_transport(0, 30, 30), "LP",
     "Degenerate transport",
     "A transportation model with massive primal degeneracy — built to stall a naive ratio test.",
     "instant"),
    ("chain_m", lambda: supply_chain(0, 70, 40, 110), "LP",
     "Supply chain flow",
     "Multi-echelon flow from sources through hubs to demand points.",
     "instant"),
    ("blend_m", lambda: refinery_blend(0, 120, 30, 40, 8), "LP",
     "Refinery crude blending — planner size",
     "The same problem an order of magnitude larger — the size a planner actually runs.",
     "instant"),
    ("prod_m", lambda: production_planning(0, 25, 18, 24), "LP",
     "Production planning",
     "Multi-period production, inventory and backlog across a plant network.",
     "instant"),
    ("prod_l", lambda: production_planning(0, 40, 30, 36), "LP",
     "Production planning — full horizon",
     "Forty units, thirty products, thirty-six periods. Around thirty thousand simplex "
     "iterations: long enough to watch the primal box sit lit while the log scrolls.",
     "seconds"),

    ("qsmooth_m", lambda: smooth(0, 80, 4), "QP",
     "Production smoothing",
     "Running a unit up and down costs money, so the objective penalises the "
     "period-to-period change quadratically. The Hessian is the singular tridiagonal "
     "second-difference matrix.",
     "instant"),
    ("qrisk_m", lambda: risk(0, 150, 12), "QP",
     "Factor-model risk",
     "Minimise x'Σx − μ'x over a budget and position limits, Σ built as a factor model. "
     "Takes the interior-point path and then crossover, because normal equations go dense "
     "the moment Q is not diagonal.",
     "instant"),

    ("uc_s", lambda: unit_commitment(0, 6, 12), "MILP",
     "Unit commitment — half day",
     "Which generators to switch on, and how hard to run them. Binary on/off decisions.",
     "instant"),
    ("uc_m", lambda: unit_commitment(0, 10, 16), "MILP",
     "Unit commitment — the cuts demo",
     "Solve it, then untick cut separation and solve it again. Same objective; one node "
     "against fifty-one. This is the model to show.",
     "instant"),
    ("uc_l", lambda: unit_commitment(1, 14, 24), "MILP",
     "Unit commitment — a full day, 14 units",
     "Cuts close part of the gap and the tree does the rest: a few thousand nodes, proven "
     "optimal. The branch-and-bound box stays lit and the bound and incumbent close on "
     "each other while you watch.",
     "a minute"),

    ("sched_s", lambda: refinery_schedule(0, 4, 3, 8, 2, 4, 3, 2), "MILP",
     "Refinery scheduling — a shift",
     "Which mode each unit runs in each slot, what it charges, and what sits in the tanks "
     "between. Mode changes cost money, a mode entered has to be held, and a unit is off "
     "or above turndown — never between. This is the layer under the planning LP.",
     "instant"),
    ("sched_m", lambda: refinery_schedule(0, 6, 3, 12, 3, 5, 4, 2), "MILP",
     "Refinery scheduling — a week",
     "Six units, three modes, twelve slots, 432 binaries. Yields depend on the mode, which "
     "is exactly what a fixed-yield planning LP cannot express. CPLEX and Gurobi both "
     "confirm the answer; watch how much of the gap the cuts take before the tree starts.",
     "seconds"),
    ("qrto_m", lambda: process_rto(0, 14, 8, 6, 3), "QP",
     "Process optimization (RTO)",
     "The real-time-optimization layer: move each unit's operating point along its measured "
     "gain matrix to hit product targets at least utility cost. Compressor power really "
     "does go as the square of throughput. A third of the controlled variables carry a spec "
     "band but no target, so the Hessian is singular on purpose.",
     "instant"),

    ("haverly1", lambda: haverly(1), "QCQP",
     "Haverly pooling — the counterexample",
     "Three crudes, one pool whose sulfur is a DECISION, two products with sulfur specs. "
     "Quality times flow is a product of two variables, so the model is nonconvex and the "
     "obvious linear version of it reports a blend that cannot be made. Solved to proven "
     "global optimality; the published answer is 400 and this is the instance that shows "
     "why a local method is not enough — started from random points it lands on 100 a "
     "quarter of the time.",
     "instant"),
    ("pool_m", lambda: pooling(0, 5, 2, 3, 1), "QCQP",
     "Refinery pooling — two pools",
     "The same physics at planner scale: five crudes, two pools with free qualities, three "
     "products. Cheap crude is high-sulfur crude, so the blend ratio is the whole decision. "
     "Watch the spatial branch-and-bound box: it splits a CONTINUOUS variable's range, which "
     "is the step a mixed-integer tree has no analogue for.",
     "instant"),

    ("miqcard_s", lambda: card(0, 24, 5, 5), "MIQP",
     "Cardinality-constrained portfolio",
     "Quadratic risk over continuous weights, binaries deciding which names are held at "
     "all, at most five. Every node relaxation is a QP — there is no simplex tableau to "
     "read a Gomory cut off, which is why this path has no cut box.",
     "instant"),
    ("miqlots_m", lambda: lots(0, 16, 6, 6), "MIQP",
     "Discrete lot blending",
     "Components arrive as whole tanker lots, and the blend is judged by squared deviation "
     "from spec. Rounding the continuous optimum gives the wrong answer; so does "
     "optimising the linear part and evaluating the quadratic there — that second one was "
     "a live bug here until an instance like this caught it.",
     "instant"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    for mid, build, kind, name, blurb, runtime in SET:
        p = build()
        path = os.path.join(OUT, mid + ".mps")
        p.write(path)
        nint = sum(1 for c in p.cols if c[4])
        manifest.append({
            "id": mid, "name": name, "blurb": blurb, "kind": kind,
            "rows": len(p.rows), "cols": len(p.cols), "nnz": p.nnz(),
            "nint": nint, "nquad": len(getattr(p, "q", {})),
            "runtime": runtime, "bytes": os.path.getsize(path),
        })
        print(f"{mid:<12} {kind:<5} {len(p.rows):>6} rows {len(p.cols):>6} cols "
              f"{p.nnz():>8} nnz {nint:>5} int {len(getattr(p,'q',{})):>6} quad  {runtime}")
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwrote {len(manifest)} models and manifest.json into demo/models/")


if __name__ == "__main__":
    main()
