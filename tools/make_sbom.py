#!/usr/bin/env python3
"""Regenerate sbom.json (CycloneDX 1.5).

    python3 tools/make_sbom.py

WHY A SOFTWARE BILL OF MATERIALS FOR A PROJECT WITH NO DEPENDENCIES

Because that is the interesting thing to state, and stating it in prose means
somebody has to believe a README.  An SBOM with an empty `components` array is
the same claim in a format a procurement, compliance or vulnerability-management
process can read without asking anyone.  "Are we exposed to CVE-2025-nnnn in
libfoo?" stops being a research task: libfoo is not in the file, at any depth.

The claim it makes is a SUPPLY-CHAIN claim and nothing more.  Zero dependencies
means no transitive CVE surface; it does not mean no bugs, and the properties
below say so rather than letting a reader infer otherwise.  Memory safety is the
job of the sanitizers, the fuzzers and CodeQL -- see SECURITY.md.

The source hash is what stops this file drifting into fiction.  It covers every
first-party source file, so a stale SBOM is detectable rather than merely wrong:

    python3 tools/make_sbom.py && git diff --exit-code sbom.json
"""
import datetime
import hashlib
import json
import os

ROOTS = ["src", "include", "apps", "tools", "tests", "cuda"]
EXTS = (".cpp", ".hpp", ".h", ".c", ".cu")


def source_digest():
    files = []
    for r in ROOTS:
        for dirpath, _, names in os.walk(r):
            for n in sorted(names):
                if n.endswith(EXTS):
                    files.append(os.path.join(dirpath, n))
    files.sort()
    h = hashlib.sha256()
    lines = 0
    for f in files:
        b = open(f, "rb").read()
        h.update(f.encode())
        h.update(b)
        lines += b.count(b"\n")
    return files, lines, h.hexdigest()


def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    files, lines, digest = source_digest()

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc)
                                 .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lifecycles": [{"phase": "build"}],
            "component": {
                "type": "library",
                "bom-ref": "pkg:generic/igaos@0.2.0",
                "name": "igaos",
                "version": "0.2.0",
                "description": "Indigenous GPU-Accelerated Optimization Solver -- "
                               "LP, MILP, QP, MIQP, NLP and MINLP, written from the "
                               "mathematical foundation up in C++17 with no "
                               "third-party numerical library.",
                "scope": "required",
                "licenses": [{"license": {"name": "See LICENSE -- team-owned, with a "
                                                  "perpetual royalty-free grant to "
                                                  "MRPL, ONGC, MoE and AICTE"}}],
                "hashes": [{"alg": "SHA-256", "content": digest}],
            },
            "properties": [
                {"name": "igaos:thirdPartyRuntimeDependencies", "value": "0"},
                {"name": "igaos:thirdPartyBuildDependencies", "value": "0"},
                {"name": "igaos:buildRequirements",
                 "value": "C++17 compiler; CMake >= 3.16; OpenMP optional"},
                {"name": "igaos:transitiveCveSurface",
                 "value": "none -- no third-party library is linked at any layer, so "
                          "no CVE in a dependency can apply. This is a supply-chain "
                          "property and explicitly NOT a memory-safety claim; see "
                          "SECURITY.md."},
                {"name": "igaos:linkedOptimizationLibraries",
                 "value": "none -- not CPLEX, Gurobi, Xpress, COIN-OR, HiGHS, GLPK, "
                          "SCIP, OSQP, BLAS, LAPACK, SuiteSparse or Eigen"},
                {"name": "igaos:optionalToolchainComponents",
                 "value": "OpenMP (compiler runtime, optional, degrades to serial); "
                          "CUDA toolkit (compile-only, never linked into the CPU "
                          "build, kernels never executed on a device)"},
                {"name": "igaos:benchmarkOnlyPythonPackages",
                 "value": "osqp, cplex, gurobipy, highspy -- used ONLY by scripts "
                          "under bench/ to produce reference numbers. Not imported "
                          "by the solver, the library, the CLI or the Python "
                          "bindings, and not required to build or run any of them."},
                {"name": "igaos:firstPartySourceFiles", "value": str(len(files))},
                {"name": "igaos:firstPartySourceLines", "value": str(lines)},
                {"name": "igaos:firstPartySourceSha256", "value": digest},
            ],
        },
        # Empty, and that emptiness IS the assertion. In CycloneDX an empty
        # components array means "this software has no dependencies".
        "components": [],
        "dependencies": [{"ref": "pkg:generic/igaos@0.2.0", "dependsOn": []}],
    }

    with open("sbom.json", "w") as f:
        f.write(json.dumps(sbom, indent=2) + "\n")
    print(f"sbom.json: {len(files)} first-party source files, {lines} lines")
    print(f"tree sha256: {digest}")


if __name__ == "__main__":
    main()
