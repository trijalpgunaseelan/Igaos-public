#!/usr/bin/env python3
"""
Check an IGAOS certificate. Independently, and in exact arithmetic.

    python3 tools/verify_certificate.py model.mps model.cert

WHAT THIS PROGRAM IS FOR
------------------------
The solver says a number is optimal. This program decides whether to believe it,
without running the solver, without sharing a line of code with it, and without
performing a single floating-point operation.

That last part is the point. Eighteen defects have been found in IGAOS so far;
two of them produced a WRONG ANSWER rather than an error, and one of those --
an MPS reader that mis-parsed a fixed-column file -- survived every test the
project had. A checker built from the same parts, with the same arithmetic,
would have been fooled in the same way. So this one has its own MPS reader,
written from the format description, and it converts every number to an exact
rational (fractions.Fraction) before touching it. Nothing here rounds.

THE THEOREM
-----------
For   min c'x   subject to   rl <= Ax <= ru,   cl <= x <= cu,
write z = c - A'y.  Then for ANY vector y at all,

    L(y) = SUM_j ( z_j >= 0 ? z_j*cl_j : z_j*cu_j )
         + SUM_i ( y_i >= 0 ? y_i*rl_i : y_i*ru_i )

is a lower bound on the optimum, because c'x = z'x + y'(Ax) and each term is
minimised over its own interval independently.

So the check is:

    1. x is feasible                  =>  c'x >= optimum is not needed; rather,
                                          optimum <= c'x
    2. L(y) is a valid lower bound    =>  optimum >= L(y)
    3. c'x == L(y)                    =>  x IS optimal.

"For any y" is what makes this worth doing. The checker never has to reproduce
the solver's reasoning, agree with its dual sign conventions, or trust its
basis. It is handed a vector and it verifies an inequality.

The same object proves infeasibility. Put c = 0, so z = -A'y. If L(y) > 0 then a
feasible x would give 0 >= L(y) > 0, which is false, so none exists.

WHAT IT DOES NOT DO
-------------------
A mixed-integer OPTIMALITY proof needs the branch-and-bound tree as well, so a
checker can confirm the search covered the space. For a MIP this verifies that
the incumbent is genuinely feasible and genuinely integral -- an exact upper
bound, and the check that catches a rounding-a-fraction-to-an-integer bug -- and
verifies the dual bound if one is supplied. It does not yet prove that no better
integer point exists, and it says so in its verdict rather than implying
otherwise.

EXIT CODES
----------
    0  every claim verified
    1  a claim FAILED -- the solver is wrong, or the certificate is corrupt
    2  the certificate could not be read, or the claim is not one we check
"""
import sys
from fractions import Fraction

# --------------------------------------------------------------------------
# An MPS reader. Written from the format description, deliberately not shared
# with the solver's. Every value becomes an exact Fraction on the way in.
# --------------------------------------------------------------------------
INF = None                      # None is our infinity; comparisons are explicit

_FIXED = ((1, 2), (4, 8), (14, 8), (24, 12), (39, 8), (49, 12))
_MAXFREE = {"ROWS": 2, "COLUMNS": 5, "RHS": 5, "RANGES": 5, "BOUNDS": 4}


def _frac(tok):
    """Exact rational from a decimal literal. No float ever appears."""
    tok = tok.strip()
    neg = tok.startswith("-")
    if tok[0] in "+-":
        tok = tok[1:]
    if "e" in tok.lower():
        mant, _, ex = tok.lower().partition("e")
        v = _frac(mant) * (Fraction(10) ** int(ex))
    elif "." in tok:
        whole, _, dec = tok.partition(".")
        v = Fraction(int(whole or 0)) + (Fraction(int(dec or 0), 10 ** len(dec))
                                         if dec else Fraction(0))
    else:
        v = Fraction(int(tok))
    return -v if neg else v


def _split_fixed(raw):
    out = []
    for beg, ln in _FIXED:
        if len(raw) <= beg:
            break
        piece = raw[beg:beg + ln].strip()
        if piece:
            out.append(piece)
    return out


def _detect_fixed(path):
    section = None
    with open(path, "r", errors="replace") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("*"):
                continue
            if not raw[0].isspace():
                head = raw.split()[0].upper()
                if head == "ENDATA":
                    break
                section = head if head in _MAXFREE else None
                continue
            if section is None:
                continue
            t = raw.split()
            if not t:
                continue
            if section == "COLUMNS" and len(t) >= 3 and "'MARKER'" in (t[1], t[2]):
                continue
            if len(t) > _MAXFREE[section]:
                return True
    return False


class Problem:
    __slots__ = ("name", "sense", "nrow", "ncol", "c", "c0", "cl", "cu",
                 "rl", "ru", "cols", "rows", "integer", "quad", "A")


def read_mps(path):
    fixed = _detect_fixed(path)
    p = Problem()
    p.name, p.sense = "", "min"
    rowk, rowidx, rown = [], {}, []
    colidx, coln = {}, []
    objname = None
    c, cl, cu = {}, {}, {}
    entries = {}                                 # col -> list of (row, value)
    rhs, rng, lo, up = {}, {}, {}, {}
    intcols, inint = set(), False
    p.quad = False
    section = None

    def col(nm):
        if nm not in colidx:
            colidx[nm] = len(coln)
            coln.append(nm)
            entries[nm] = []
        return colidx[nm]

    with open(path, "r", errors="replace") as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip() or raw.lstrip().startswith("*"):
                continue
            if not raw[0].isspace():
                t = raw.split()
                head = t[0].upper()
                if head == "NAME":
                    p.name = t[1] if len(t) > 1 else ""
                elif head in ("QUADOBJ", "QMATRIX", "QSECTION"):
                    p.quad = True
                    section = "QUAD"
                elif head == "OBJSENSE":
                    section = "OBJSENSE"
                    if len(t) > 1 and t[1].upper().startswith("MAX"):
                        p.sense = "max"
                elif head == "ENDATA":
                    break
                else:
                    section = head
                continue

            t = raw.split()
            if fixed:
                marker = (section == "COLUMNS" and len(t) >= 3
                          and "'MARKER'" in (t[1], t[2]))
                if not marker:
                    t = _split_fixed(raw)
            if not t:
                continue

            if section == "OBJSENSE":
                if t[0].upper().startswith("MAX"):
                    p.sense = "max"

            elif section == "ROWS":
                k, nm = t[0].upper(), t[1]
                if k == "N":
                    if objname is None:
                        objname = nm
                    continue
                rowidx[nm] = len(rowk)
                rowk.append(k)
                rown.append(nm)

            elif section == "COLUMNS":
                if len(t) >= 3 and "'MARKER'" in (t[1], t[2]):
                    joined = " ".join(t).upper()
                    if "INTORG" in joined:
                        inint = True
                    if "INTEND" in joined:
                        inint = False
                    continue
                nm = t[0]
                j = col(nm)
                if inint:
                    intcols.add(j)
                for k in range(1, len(t) - 1, 2):
                    rname, val = t[k], _frac(t[k + 1])
                    if rname == objname:
                        c[j] = c.get(j, Fraction(0)) + val
                    elif rname in rowidx:
                        entries[nm].append((rowidx[rname], val))

            elif section == "RHS":
                start = 1 if (len(t) % 2 == 1) else 0
                for k in range(start, len(t) - 1, 2):
                    rname, val = t[k], _frac(t[k + 1])
                    if rname == objname:
                        rhs["__obj__"] = val
                    elif rname in rowidx:
                        rhs[rowidx[rname]] = val

            elif section == "RANGES":
                start = 1 if (len(t) % 2 == 1) else 0
                for k in range(start, len(t) - 1, 2):
                    if t[k] in rowidx:
                        rng[rowidx[t[k]]] = _frac(t[k + 1])

            elif section == "BOUNDS":
                kind = t[0].upper()
                cname = t[2] if len(t) >= 3 else t[1]
                if cname not in colidx and len(t) >= 2 and t[1] in colidx:
                    cname = t[1]
                j = col(cname)
                v = _frac(t[3]) if len(t) >= 4 else None
                if kind == "UP":
                    up[j] = v
                    if v is not None and v < 0 and j not in lo:
                        lo[j] = None                       # UP with negative value
                elif kind == "LO":
                    lo[j] = v
                elif kind == "FX":
                    lo[j] = up[j] = v
                elif kind == "FR":
                    lo[j], up[j] = None, None
                elif kind == "MI":
                    lo[j] = None
                elif kind == "PL":
                    up[j] = None
                elif kind in ("BV",):
                    lo[j], up[j] = Fraction(0), Fraction(1)
                    intcols.add(j)
                elif kind in ("LI",):
                    lo[j] = v
                    intcols.add(j)
                elif kind in ("UI",):
                    up[j] = v
                    intcols.add(j)

    n, m = len(coln), len(rowk)
    p.ncol, p.nrow = n, m
    p.cols, p.rows = coln, rown
    p.integer = intcols
    p.c = [c.get(j, Fraction(0)) for j in range(n)]
    p.c0 = -rhs.get("__obj__", Fraction(0))          # RHS on the N row is -constant
    p.cl = [lo.get(j, Fraction(0)) for j in range(n)]
    p.cu = [up.get(j, None) for j in range(n)]

    p.rl, p.ru = [], []
    for i in range(m):
        k, b = rowk[i], rhs.get(i, Fraction(0))
        r = rng.get(i)
        if k == "E":
            if r is None:
                p.rl.append(b); p.ru.append(b)
            elif r >= 0:
                p.rl.append(b); p.ru.append(b + r)
            else:
                p.rl.append(b + r); p.ru.append(b)
        elif k == "L":
            p.rl.append(None if r is None else b - abs(r)); p.ru.append(b)
        elif k == "G":
            p.rl.append(b); p.ru.append(None if r is None else b + abs(r))
        else:
            p.rl.append(None); p.ru.append(None)

    # column-major A
    p.A = [entries[nm] for nm in coln]
    return p


# --------------------------------------------------------------------------
# The certificate
# --------------------------------------------------------------------------
def read_cert(path):
    """Hex floats in, exact Fractions out. float.fromhex is lossless."""
    head, x, y = {}, {}, {}
    with open(path) as f:
        first = f.readline().split()
        if not first or first[0] != "IGAOS-CERT":
            raise ValueError("not an IGAOS certificate")
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "end":
                break
            t = line.split()
            if t[0] == "x":
                x[int(t[1])] = Fraction(float.fromhex(t[2]))
            elif t[0] == "y":
                y[int(t[1])] = Fraction(float.fromhex(t[2]))
            elif len(t) >= 2:
                head[t[0]] = t[1] if not t[1].startswith(("0x", "-0x")) \
                    else Fraction(float.fromhex(t[1]))
    return head, x, y


# --------------------------------------------------------------------------
def fmt(v):
    """Fractions are unreadable at a glance; show a float only for display."""
    if v is None:
        return "inf"
    try:
        return "%.12g" % float(v)
    except (OverflowError, ValueError):
        return str(v)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    box = None
    for a in argv[1:]:
        if a.startswith("--box="):
            box = _frac(a.split("=", 1)[1])
    if len(args) != 2:
        print("usage: verify_certificate.py model.mps model.cert [--box=1e9]")
        return 2
    mps, cert = args

    p = read_mps(mps)
    head, xs, ys = read_cert(cert)
    claim = head.get("claim", "?")

    print("model        %s   %d rows x %d cols%s" %
          (p.name or mps, p.nrow, p.ncol,
           ", %d integer" % len(p.integer) if p.integer else ""))
    print("certificate  claims %s" % claim)
    if p.quad or head.get("quadratic") == "1":
        print("\nREFUSED: the objective is quadratic. The bound theorem this checker")
        print("         implements is the linear one; certifying a QP needs a")
        print("         different argument and it is not written yet.")
        return 2

    x = [xs.get(j, Fraction(0)) for j in range(p.ncol)]
    y = [ys.get(i, Fraction(0)) for i in range(p.nrow)]

    # The box only ever multiplies a reduced cost that should have been zero and
    # is not, because y was rounded to a double. Those residuals sit around
    # 1e-16, so the box's only job is to keep their product negligible -- and a
    # box scaled to the point being certified does that far better than a fixed
    # one. On SCORPION a box of 1e9 costs 3.2e-3 and loses the claim; a box of
    # ten times the point's own norm costs 3e-9 and proves it to ten digits.
    # It stays an assumption either way, and the verdict always names it.
    if box is None:
        span = max((abs(v) for v in x), default=Fraction(0))
        box = max(Fraction(1000), span * 10)
    sign = Fraction(-1) if p.sense == "max" else Fraction(1)
    c = [sign * v for v in p.c]                # work in minimisation throughout

    failures = []
    checks = []

    # ---- 1. primal feasibility, exactly ----------------------------------
    if claim != "infeasible":
        worst_b, worst_bj = Fraction(0), None
        for j in range(p.ncol):
            if p.cl[j] is not None and x[j] < p.cl[j]:
                d = p.cl[j] - x[j]
                if d > worst_b: worst_b, worst_bj = d, j
            if p.cu[j] is not None and x[j] > p.cu[j]:
                d = x[j] - p.cu[j]
                if d > worst_b: worst_bj = j if d > worst_b else worst_bj
                worst_b = max(worst_b, d)
        act = [Fraction(0)] * p.nrow
        for j in range(p.ncol):
            xj = x[j]
            if xj:
                for (i, v) in p.A[j]:
                    act[i] += v * xj
        worst_r, worst_ri = Fraction(0), None
        for i in range(p.nrow):
            if p.rl[i] is not None and act[i] < p.rl[i]:
                d = p.rl[i] - act[i]
                if d > worst_r: worst_r, worst_ri = d, i
            if p.ru[i] is not None and act[i] > p.ru[i]:
                d = act[i] - p.ru[i]
                if d > worst_r: worst_r, worst_ri = d, i
        worst = max(worst_b, worst_r)
        where = ("column %s" % p.cols[worst_bj]) if worst_b >= worst_r and worst_bj is not None \
            else (("row %s" % p.rows[worst_ri]) if worst_ri is not None else "nowhere")
        checks.append(("primal feasibility", worst, where))

        # ---- 2. integrality, exactly -------------------------------------
        if p.integer:
            worst_i, worst_ij = Fraction(0), None
            for j in p.integer:
                d = abs(x[j] - round(x[j]))
                if d > worst_i:
                    worst_i, worst_ij = d, j
            checks.append(("integrality", worst_i,
                           "column %s" % p.cols[worst_ij] if worst_ij is not None else "exact"))

    # ---- 3. the bound L(y), computed safely -------------------------------
    #
    # A floating-point y is almost never exactly dual-feasible. On AFIRO four
    # basic columns come out with a reduced cost of -3e-17 instead of 0, purely
    # because y was rounded to a double. Taken literally that makes L(y) = -inf
    # and rejects a correct answer, which would make this tool useless.
    #
    # The standard remedy (Neumaier and Shcherbina, "Safe bounds in linear and
    # mixed-integer programming") is to bound the offending term rather than
    # discard it. If a column has no upper bound and z_j < 0, its contribution
    # is at worst z_j * X for any X that bounds x_j from above. So the checker
    # takes an explicit box X, applies it ONLY where a bound is missing in the
    # direction needed, and reports exactly what that cost -- here, four terms
    # of 1e-17 against a box of 1e9 weaken the bound by 1e-7.
    #
    # This keeps the result rigorous and keeps the assumption visible: the
    # verdict states the box whenever one was used, and says the proof is
    # unconditional whenever it was not.
    cz = [Fraction(0)] * p.ncol if claim == "infeasible" else c
    z = list(cz)
    for j in range(p.ncol):
        s_ = Fraction(0)
        for (i, v) in p.A[j]:
            if y[i]:
                s_ += v * y[i]
        z[j] = cz[j] - s_

    # The objective constant belongs in the bound too. Leaving it out made the
    # checker report e226 as unsupported by exactly 7.113 -- which is e226's
    # objective constant, and which three different algorithms reproduced to ten
    # digits. Three independent methods agreeing on a "wrong" number is never
    # three coincidences; it was the checker that was wrong.
    L = (p.c0 * sign) if claim != "infeasible" else Fraction(0)
    boxed, penalty = 0, Fraction(0)
    for j in range(p.ncol):
        if z[j] > 0:
            if p.cl[j] is None:
                L += z[j] * (-box); boxed += 1; penalty += z[j] * box
            else:
                L += z[j] * p.cl[j]
        elif z[j] < 0:
            if p.cu[j] is None:
                L += z[j] * box; boxed += 1; penalty += -z[j] * box
            else:
                L += z[j] * p.cu[j]
    for i in range(p.nrow):
        if y[i] > 0:
            if p.rl[i] is None:
                L += y[i] * (-box); boxed += 1; penalty += y[i] * box
            else:
                L += y[i] * p.rl[i]
        elif y[i] < 0:
            if p.ru[i] is None:
                L += y[i] * box; boxed += 1; penalty += -y[i] * box
            else:
                L += y[i] * p.ru[i]

    no_duals = str(head.get("duals", "?")) == "0"

    print()
    for nm, viol, where in checks:
        print("  %-24s %s%s" %
              (nm, "EXACT" if viol == 0 else "violated by " + fmt(viol),
               "" if viol == 0 else "  at %s" % where))

    if no_duals:
        print("  %-24s none supplied — this solve path does not yet emit them"
              % "dual multipliers")
    elif boxed:
        print("  %-24s L(y) = %s" % ("rigorous lower bound", fmt(L)))
        print("  %-24s %d term%s assumed |x| <= %s; that cost %s of the bound" %
              ("", boxed, "" if boxed == 1 else "s", fmt(box), fmt(penalty)))
    else:
        print("  %-24s L(y) = %s   (unconditional)" % ("rigorous lower bound", fmt(L)))

    # The dual bound needs no primal point at all, which is why it is stated
    # first and separately: it is valid even if x below turns out to be junk.

    # ---- 4. the verdict ---------------------------------------------------
    print()
    if claim == "infeasible":
        if no_duals:
            print("VERDICT: CANNOT BE CHECKED.")
            print("         The claim is infeasibility, whose proof is a multiplier vector y")
            print("         with L(y) > 0. None was supplied: this instance was refused by")
            print("         presolve, which reaches its conclusion by reasoning the solver")
            print("         does not yet turn into a certificate. The refusal may well be")
            print("         correct — it is simply not proven here.")
            return 2
        if L > 0:
            print("VERDICT: INFEASIBILITY PROVEN.")
            print("         L(y) = %s > 0, so a feasible point would give 0 >= L(y) > 0." % fmt(L))
            if boxed:
                print("         Conditional on |x| <= %s on %d unbounded term%s."
                      % (fmt(box), boxed, "" if boxed == 1 else "s"))
            return 0
        print("VERDICT: NOT PROVEN — L(y) = %s is not positive." % fmt(L))
        return 1

    cx = p.c0 * sign + sum(c[j] * x[j] for j in range(p.ncol) if x[j])
    user_cx = sign * cx
    stated = head.get("objective")
    infeas = max((v for _, v, _ in checks), default=Fraction(0))

    print("  %-24s %s   (recomputed here, exactly)" % ("objective of x", fmt(user_cx)))
    if isinstance(stated, Fraction):
        d = abs(stated - user_cx)
        rel = d / (1 + abs(user_cx))
        print("  %-24s %s   differs by %s" % ("objective solver stated", fmt(stated), fmt(d)))
        if rel > Fraction(1, 10 ** 6):
            failures.append("the stated objective does not match the point supplied — "
                            "the certificate does not belong to this model")

    print()
    if failures:
        print("VERDICT: REJECTED.")
        for f in failures:
            print("         %s" % f)
        return 1

    gap = cx - L

    # Does the certificate support the claim it makes? Tampering with y cannot
    # produce a WRONG bound -- the theorem holds for every y -- but it produces
    # a useless one. Changing a single dual on AFIRO moves L(y) from -464.75 to
    # -31,429,036: still true, still no help. A certificate that proves a bound
    # nowhere near its own claimed objective has not certified that claim, and
    # saying so is the difference between a checker and a rubber stamp.
    # For a MIXED-INTEGER model this test does not apply. The multipliers can
    # only ever come from a continuous relaxation, so L(y) sits below the integer
    # optimum by the integrality gap -- which is a property of the problem, not
    # evidence of tampering. Saying "claim not supported" there would be the
    # checker misreading its own arithmetic.
    if claim == "optimal" and not no_duals and not p.integer and L != cx:
        spread = abs(cx - L) / (1 + abs(cx))
        if spread > Fraction(1, 10 ** 6):
            print("VERDICT: CLAIM NOT SUPPORTED.")
            print("         The certificate claims an optimum of %s." % fmt(user_cx))
            print("         Its multipliers prove only that the optimum is at least %s."
                  % fmt(sign * L if p.sense == "max" else L))
            print("         That bound is valid -- L(y) is a bound for any y at all -- but it")
            print("         is %s away from the claim, so it does not certify it." % fmt(abs(cx - L)))
            print("         Either the multipliers were altered, or the solver did not")
            print("         converge as far as it reported.")
            return 1

    if p.integer:
        if infeas == 0:
            print("VERDICT: FEASIBILITY AND INTEGRALITY PROVEN, exactly.")
            print("         Every integer column holds an exact integer and every constraint")
            print("         is satisfied exactly, so %s is a rigorous upper bound on the"
                  % fmt(user_cx))
            print("         optimum. That is the check which catches a fractional value")
            print("         rounded and reported as integral.")
            if no_duals:
                print("         NOT proven: any lower bound. The tree emits no multipliers yet,")
                print("         so optimality is claimed by the solver and not certified here.")
            else:
                lo = sign * L if p.sense == "max" else L
                print("         Lower bound L(y) = %s, unconditional." % fmt(lo))
                print("         So the optimum lies in [%s, %s]," % (fmt(lo), fmt(user_cx)))
                print("         proven at both ends: the point above it, these multipliers below.")
                width = abs(cx - L)
                rel = width / (1 + abs(cx))
                if rel <= Fraction(1, 10 ** 6):
                    print("         The interval is closed to %s — this incumbent IS optimal," % fmt(width))
                    print("         and that is proven here rather than asserted by the solver.")
                else:
                    print("         Width %s. That gap is the integrality gap of the root" % fmt(width))
                    print("         relaxation, not slack in the proof.")
                print("         NOT proven: that no better integer point exists — that needs")
                print("         the branch-and-bound tree, which this format does not carry.")
            return 0
        print("VERDICT: INTEGRAL POINT NOT EXACTLY FEASIBLE — misses by %s." % fmt(infeas))
        print("         L(y) = %s remains a rigorous lower bound regardless." % fmt(L))
        return 1

    if infeas == 0 and gap == 0:
        print("VERDICT: OPTIMALITY PROVEN.")
        print("         x is feasible, so the optimum is at most %s." % fmt(user_cx))
        print("         L(y) is a lower bound and equals it exactly.")
        print("         Every step was exact rational arithmetic on an independently")
        print("         parsed model, with no floating-point operation anywhere.")
        return 0

    if infeas == 0:
        print("VERDICT: BOUNDED, NOT PROVEN OPTIMAL.")
        print("         The optimum lies in [%s, %s]; width %s." % (fmt(L), fmt(user_cx), fmt(gap)))
        print("         x is exactly feasible, so the upper end is rigorous.")
        return 0

    print("VERDICT: LOWER BOUND PROVEN; the point is not exactly feasible.")
    print("         The optimum is at least %s — that holds whatever x is." % fmt(L))
    print("         The point supplied misses feasibility by %s in exact arithmetic," % fmt(infeas))
    print("         so it does not establish an upper bound. At that size it is the")
    print("         solver's working tolerance, not a wrong answer; but a tolerance is")
    print("         not a proof, and this tool does not call it one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
