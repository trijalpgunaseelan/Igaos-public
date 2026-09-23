# Threat model

A threat model is only useful if it names a deployment. This one names three,
because IGAOS is used in three shapes and they do not share a boundary.

Companion to `SECURITY.md`, which lists the controls and the open gaps. This
document is about *who* and *where*, not *what was fixed*.

---

## 1. What is being protected

| Asset | Why it matters |
|---|---|
| **The answer** | A refinery blend, a unit-commitment schedule, a crude purchase. Wrong by 2% is money; wrong and labelled `optimal` is a decision nobody reviews. |
| **The model** | A production plan encodes yields, capacities, contract prices and margins. The *file* is often more commercially sensitive than the answer. |
| **The host** | The solver runs inside planning systems, which sit on networks that reach process historians and, eventually, control systems. |
| **The claim of sovereignty** | The reason to adopt this over CPLEX is that it can be inspected. A compromised build or an unpinned CI action makes the shipped binary something nobody inspected. |

**Integrity of the answer is the top asset, above confidentiality and above
availability.** A solver that refuses to run is a bad afternoon. A solver that
returns a confident wrong number is a bad quarter, and nothing downstream is built
to catch it. Every design choice below follows from that ordering — most visibly,
a corrupt input file is *refused* rather than repaired, skipped or best-guessed.

---

## 2. Deployment shapes and their boundaries

### A. Embedded in a planning application — the real one

```
  planner's model file --> [ planning app ] --> libigaos --> answer --> decision
        ^                        ^                  ^
        |                        |                  |
   TRUST BOUNDARY 1         same process       TRUST BOUNDARY 2
   the file is untrusted    (no boundary)      the answer is trusted downstream
```

- **Boundary 1** is the only one inside our control: `readMps()` and the C API.
- The library shares an address space with its caller and there is **no boundary
  between them**. A caller that passes a malformed `SparseMatrix` through the C++
  API can corrupt memory, and no amount of checking in the library changes that.
  The C ABI checks pointers for NULL; it does not and cannot validate them.
- **Boundary 2** is outside our control and is where the integrity asset lives.
  Whatever reads the answer treats it as authoritative.

### B. Command line, operator-driven

The operator supplies arguments and the file. The threat is a **malicious or
corrupt model file** from a shared drive, a vendor, or an email. The operator is
not the adversary; the file is.

### C. `demo/server.py` — a demonstration, not a deployment

```
  browser --HTTP--> 127.0.0.1:8420 --> subprocess: igaos <tmpfile>
     ^
   ANY WEB PAGE THE OPERATOR HAS OPEN IS ALSO ON THIS SIDE
```

The loopback bind looks like a boundary and is not one. Every page in the
operator's browser can issue requests to `127.0.0.1:8420`, and a plain `GET` is
not preflighted. See `SECURITY.md`, "Known gaps", item 1. **This shape is not
deployable and is not intended to be.**

---

## 3. Adversaries

| # | Adversary | Capability | Goal |
|---|---|---|---|
| A1 | **Whoever authored the model file** | Full control of the bytes reaching `readMps()`. The realistic and most important adversary. | Crash the planning run; or, far worse, make it return a wrong answer that looks right. |
| A2 | **A web page the operator has open** | Requests to loopback, no read access to responses. | Drive `demo/server.py`; consume the machine. |
| A3 | **Someone on the planning network** | Reach the host, not the process. | Read model files at rest; not addressed here — that is the host's job. |
| A4 | **A compromised CI dependency** | Arbitrary code inside a build, with `GITHUB_TOKEN`'s scopes. | Alter the shipped binary. The sovereignty asset. |
| A5 | **A malicious caller of the library** | Already in the address space. | Out of scope: same trust domain, no boundary to cross. |

**Explicitly out of scope:** the host operating system, the network, physical
access, the operator's credentials, and anyone who already executes code in the
embedding process. Those are real, and they are not this project's boundary to
defend.

---

## 4. Threats against the top asset, and what answers each

The interesting column is the last one.

| Threat | Mechanism | Control | Residual |
|---|---|---|---|
| **Poisoned coefficient** | `nan` / `inf` in a numeric field. Every comparison against NaN is false, so no feasibility or ratio test fails and the run reports `optimal`. **Measured pre-fix: `status=optimal obj=-4` on a model whose honest optimum is −12.** | Non-finite literals are a parse error; the file is refused, not repaired. Fuzzer asserts no NaN reaches any model field. | A finite but absurd coefficient (`1e29`) is legal MPS and is accepted. Conditioning, not corruption — the solver reports its own numerical trouble. |
| **Silently different model** | A bad field skipped rather than refused: the file says the constraint has a term, the solve behaves as though it does not. | The whole file is refused. Skipping was the earlier behaviour and was itself the bug. | Duplicate names are still last-one-wins (gap 4). |
| **Truncated exponent** | `1.5D+02` parsed as `1.5` — a Fortran exponent `std::stod` stops at without complaining. Off by 100x. | `D`/`d` normalized to `E` before parsing. | — |
| **Memory exhaustion** | A small file declaring vast structure, or repeated rows/columns. | `MpsLimits` ceilings on file, line, name, rows, columns, nonzeros, quadratic terms. | Ceilings are generous by design; a file inside them can still be large. Run untrusted solves in their own process with external limits. |
| **Index wrap to out-of-bounds write** | `Int` is `int32_t`; an unchecked `size_t` to `Int` cast past `INT32_MAX` wraps negative and every indexed write after it is out of bounds. | Every narrowing cast is bounds-checked **before** it happens, plus an invariant re-checked at point of use. | — |
| **Memory-safety bug we have not found** | Anywhere in 9,000 lines of hand-written numerical C++. | ASan + UBSan over the whole suite; two fuzzers; `-Werror`; CodeQL; exploit mitigations verified in the linked binary. | **This is the standing residual risk and it does not go to zero.** No formal verification, no external audit. See `SECURITY.md`, gap 5. |
| **Altered shipped binary** | A moved tag on a CI action (A4). | Scoped `GITHUB_TOKEN`, `persist-credentials: false`, a job reporting unpinned actions. | **Actions are not yet SHA-pinned.** Largest open supply-chain item; `tools/pin_actions.sh` closes it. |
| **Console driven by a web page** | A2 against shape C. | None. | **Open.** Do not run the console on a host that matters. |

---

## 5. Assumptions

State them, because a threat model whose assumptions are implicit is a threat
model that is wrong somewhere nobody is looking.

1. The **compiler and standard library are trusted.** Not verified here.
2. The **operator is not the adversary.** Whoever runs the CLI may already read
   the files it reads.
3. The **embedding application is in the same trust domain.** The library defends
   against its *input*, not against its *caller*.
4. **Files at rest are the host's problem.** Confidentiality of a model on disk is
   not addressed by anything in this repository.
5. **Availability is the lowest-ranked asset.** Where refusing and continuing
   conflict, this code refuses. A planning run that stops is visible; a planning
   run that returns a wrong number is not.

---

## 6. What would change this document

- Making `demo/server.py` a real service — a different model entirely, with
  authentication, per-request isolation and rate limiting as the minimum.
- Executing the CUDA path on a device: a new component, a new driver dependency,
  and the first third-party runtime this project would link.
- Accepting models over a network rather than from a file.
- Any third-party library entering the build. The zero-dependency property is
  load-bearing for section 3's A4 row, and losing it is a change to the threat
  model, not only to the build.
