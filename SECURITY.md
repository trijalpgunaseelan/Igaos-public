# Security

IGAOS is a solver that runs inside industrial planning systems and reads model
files it did not write. This document says what that means for its security, what
is done about it, and — the part most of these documents leave out — what is not.

Nothing below is a promise about code nobody has attacked. It is a description of
the controls, and of the gaps between them.

---

## Reporting a vulnerability

Use **GitHub's private vulnerability reporting** on this repository:
*Security → Report a vulnerability*. It opens a channel visible only to the
maintainers, and it needs no address from you.

There is deliberately **no email address in this file**. An address nobody watches
is worse than no address, because it converts a report into silence and the
reporter into someone who has already tried. If this project later has a mailbox
somebody actually reads, it belongs here; until then the GitHub channel is the
honest answer.

Please include the model file or input that triggers it. A crash on a file we
cannot reproduce is a rumour.

**What to expect.** This is a student team's project, not a vendor with a
follow-the-sun rota. Expect acknowledgement in days, not hours. If a report is
serious and the team goes quiet, publish — a silent maintainer is not a reason to
leave users exposed.

---

## The property that matters most, stated precisely

**No third-party optimization or numerical library is linked at any layer.** The
sparse factorizations, orderings, simplex, interior point, cut separation,
presolve, branch-and-cut, spatial branch-and-bound and automatic differentiation
are all written here. A C++17 compiler is the whole build requirement.

For security this has one large consequence and one honest limit.

*The consequence.* There is no transitive dependency tree, so there is no
transitive CVE surface. When a CVE lands in a BLAS, a graph library, a JSON
parser or a compression library, the question "are we affected" has an answer that
takes no research: we do not link it. `sbom.json` states this formally, in
CycloneDX, so a procurement or compliance process can check it mechanically rather
than take our word.

*The limit.* Zero dependencies does not mean fewer bugs. It means the bugs are
**ours** rather than someone else's, and nobody else is looking for them. A CVE in
a widely used library is found because thousands of people depend on it. A bug
here is found because we went looking. That is the entire reason for the controls
below, and it is why "no dependencies" is a supply-chain claim and never a
memory-safety claim.

---

## Attack surface

| Surface | Reachable by | Notes |
|---|---|---|
| **`readMps()`** — MPS files | anyone who can put a file where the solver reads it | **The main one.** See below. |
| C API (`include/igaos/igaos.h`) | the embedding process | Same trust domain as the caller; pointer arguments are checked for NULL, not for validity. |
| Python bindings | the embedding process | `ctypes` over the same C ABI. `IGAOS_LIBRARY` names the shared object to load — an attacker who can set that environment variable can load anything, which is a property of `ctypes`, not of this code. |
| CLI (`apps/igaos_cli.cpp`) | whoever runs it | Arguments come from the operator. |
| **`demo/server.py`** | any process — **and any web page** — that can reach the loopback port | **Has a known unfixed weakness.** See "Known gaps". |
| CUDA kernels | nothing, today | Never executed on a device. Compiled to SASS and checked against the CPU path by an emulator. |

### The MPS reader

`readMps()` is the only function in the project that consumes bytes nobody here
produced. An MPS file arrives from a refinery planner, a vendor export, a shared
drive or an upload form, and nothing upstream validates it.

It is written to refuse a hostile file rather than trust it:

- **Non-finite values are a parse error, not a value.** `nan`, `NaN`, `inf`,
  `1e999` are all valid input to `std::stod`, and until this was fixed all of them
  reached the constraint matrix intact. A NaN coefficient is the worst possible
  corruption of a solver: every comparison against NaN is false, so the
  feasibility test does not fail, the ratio test does not fail, and the run ends
  reporting `status=optimal` on a model whose answer is undefined. Measured, on a
  two-variable model whose honest optimum is −12: the pre-fix reader accepted
  `nan` as a coefficient and the solver reported **`status=optimal obj=-4`**. No
  warning, no non-zero exit, a number indistinguishable from a real one.
  Skipping the bad field is not a fix either — that silently solves a *different*
  model than the file describes. The whole file is refused.
- **Every unbounded quantity in the format has a ceiling**, declared in
  `MpsLimits` (`include/igaos/mps.hpp`): file bytes, line bytes, name bytes, rows,
  columns, nonzeros, quadratic terms. Defaults sit three orders of magnitude above
  the largest model this project has ever solved, so a legitimate file never meets
  one. Callers that know the shape of what they should be reading — a service
  accepting uploads certainly does — should tighten them.
- **`Int` is `int32_t`, and every narrowing cast from `size_t` is checked before
  it happens**, not after. Past `INT32_MAX` columns an unchecked cast wraps
  negative and every subsequent indexed write is out of bounds.
- **A Fortran `D` exponent is normalized rather than truncated.** `1.5D+02` used to
  parse as `1.5` because `std::stod` stops at the `D` without complaining.

---

## Controls

Everything here runs in CI on every push and pull request
(`.github/workflows/ci.yml`).

| Control | What it covers |
|---|---|
| **AddressSanitizer + UndefinedBehaviorSanitizer** | The whole ctest suite, including the MPS fuzzer, so tens of thousands of malformed files go through ASan and UBSan on every change. |
| **`tools/fuzz_mps.cpp`** | The MPS reader against hostile input. Runs as a ctest (seeded, 20k cases, under a second) and as a long CI campaign across five seeds plus a coverage-guided libFuzzer run. |
| **`tools/fuzz_cuts.cpp`** | Cut separation: does a valid cut ever remove an optimal point. A different question about a different surface. |
| **Warnings as errors** | `-Wall -Wextra -Wpedantic -Werror` on the whole tree. |
| **CodeQL** (`security-extended`) | Static analysis. Non-blocking while the repository is private — CodeQL is free on public repositories and needs Advanced Security on private ones. |
| **Compiler and linker hardening** | See below. |
| **Binary verification** | A CI job reads the *linked binary* with `checksec` and fails if full RELRO, stack canary, NX or PIE is missing. A mitigation that is configured but not linked is worse than none, because it is believed. |

### What the fuzzer actually checks

Not only "does it crash" — sanitizers cover that. The harness checks that any file
which **parses successfully** produces a model satisfying the invariants the rest
of the solver assumes: no NaN in the objective, bounds, matrix or quadratic terms;
`colPtr` monotone and of length `ncol+1`; every row index in range; every quadratic
term pointing at a row and columns that exist.

That is the interesting failure mode. A crash is loud. A file that parses into a
quietly wrong model produces a confident number that looks exactly like every
other number the solver has ever produced.

### Compiler and linker hardening — and an honest note about it

`CMakeLists.txt` enables, when the toolchain supports them (each is probed, never
assumed): `-fstack-protector-strong`, `-fstack-clash-protection`,
`-fcf-protection=full` (x86) or `-mbranch-protection=standard` (aarch64),
`-ftrivial-auto-var-init=zero`, `-D_FORTIFY_SOURCE=2`, `-Wformat-security`, and at
link time full RELRO, `BIND_NOW`, non-executable stack, separate code, and PIE.
`-DIGAOS_HARDENING=OFF` turns the block off for benchmarking, so a measurement is
never quietly a measurement of the mitigations.

**Measured, and worth saying plainly: on Ubuntu's GCC 13 most of these are already
the distribution default.** A build with `-DIGAOS_HARDENING=OFF` on that toolchain
still comes out with full RELRO, a stack canary, NX, PIE and CET. So on the CI
runner this block adds `-ftrivial-auto-var-init=zero` and little else.

Its value is not that it hardened the Ubuntu build. It is that the mitigations
stop depending on which distribution happened to compile it. Upstream GCC, many
embedded and vendor toolchains, and MSVC do **not** default to these, and a
refinery deployment is exactly where an unusual toolchain turns up. The CI job
that inspects the linked binary is what keeps the claim true rather than assumed.

### `_FORTIFY_SOURCE` and sanitizers

`_FORTIFY_SOURCE` is suppressed automatically in Debug builds and whenever a
sanitizer appears in the flags — the two interpose on the same calls and the
collision surfaces as an ASan false positive. This is handled in the build, not
left to whoever configures it.

---

## Supply chain

- **No runtime dependencies.** `sbom.json` (CycloneDX 1.5) states it in a form a
  tool can read. Regenerate with `python3 tools/make_sbom.py`.
- **Build dependencies:** a C++17 compiler and CMake ≥ 3.16. OpenMP is optional.
- **CI actions are not yet pinned to commit SHAs.** `uses: actions/checkout@v4`
  names a *tag*, and a tag is a mutable pointer its owner can move — the route by
  which the `tj-actions/changed-files` compromise reached tens of thousands of
  repositories in March 2025. `tools/pin_actions.sh` resolves and rewrites every
  pin in one command on a machine with GitHub access; the `supply-chain` CI job
  reports what is still unpinned. **This is the largest open supply-chain item and
  it is deliberately visible rather than quietly absent.** Once pinned, flip that
  job's `continue-on-error` to `false`.
- **`GITHUB_TOKEN` is scoped to `contents: read`** at the workflow level; only the
  CodeQL job asks for more, and only for `security-events: write`. Every checkout
  uses `persist-credentials: false`, so the job's credentials are not left in
  `.git/config` for later steps to read.
- **Python packages in CI are pinned to a compatible release series** (`~=12.0`).
  A fully hash-pinned requirements file is better and is the next step:
  `pip-compile --generate-hashes`.

---

## Known gaps

These are open. They are listed because a security document that lists only what
was fixed is marketing.

1. **`demo/server.py` has no origin or `Host` validation.** `GET /api/stream`
   accepts a complete MPS model in the query string and spawns the solver.
   A plain `GET` is not preflighted, so **any web page the operator visits can
   drive the console** with an `<img>` or a `no-cors` fetch, and DNS rebinding
   defeats the loopback bind. `threads` is taken from the request and passed to
   `--threads` without an upper bound; `/api/env` returns absolute filesystem
   paths. **Treat the console as a demonstration to stand in front of, on a
   machine whose browser is not being used for anything else. Do not run it on a
   host that matters, and do not expose the port.** Fixing it needs a one-time
   token in the URL, an `Origin`/`Host` check, a clamp on `threads`, a concurrency
   cap and security headers.
2. **`MpsLimits` bounds allocation; it is not a sandbox.** A file inside the
   ceilings can still be enormous. A service accepting uploads should tighten the
   limits, run the solve in a separate process, and impose its own memory and time
   limits from outside.
3. **The line-length ceiling is enforced after the line is read.** The whole-file
   ceiling bounds the worst case; the line ceiling rejects the file. Truly
   incremental bounding would need the reader restructured around chunked reads.
4. **Duplicate row and column names are last-one-wins, silently.** Legal MPS files
   do not do this; a hostile one might, and the resulting model is well-formed but
   not the file the author of the file had in mind.
5. **No formal verification, no external audit, no third-party pentest.** The
   controls above are what a student team can run. They are not an assurance
   argument.
6. **The GPU path has never executed on a device.** Its arithmetic is verified
   against the CPU path by a warp-level emulator, which is a correctness control,
   not a security one.
7. **Certificates are not produced for two paths** — the branch-and-bound tree
   emits no dual multipliers, and infeasibility found inside presolve produces no
   Farkas ray. The checker reports both as unprovable rather than passing them
   silently, which is the right behaviour, but it means a MIP result carries less
   independent evidence than an LP result.

---

## Reproducing any of this

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cd build && ctest --output-on-failure          # includes the MPS fuzzer

# the long fuzz campaign
./build/igaos_fuzz_mps 100000 1

# under the sanitizers, as CI runs it
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1"
cmake --build build-asan --parallel
cd build-asan && ctest --output-on-failure

# confirm the mitigations are in the shipped binary, not just in the flags
checksec --file=build/igaos

# what is still unpinned in CI
bash tools/pin_actions.sh --check
```
