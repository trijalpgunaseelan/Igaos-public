#!/usr/bin/env python3
"""
A small MPS reader, used only by the benchmark harnesses so that a reference
solver can be handed the same file IGAOS was given.

This is deliberately independent of the solver's own C++ reader — if the two
disagree about what a file means, the comparison is worthless, so the Python
side is written from the format description rather than from the C++ source.

Returns a Problem in the ranged-row form used throughout the bench harness:

    minimize   c'x + c0        subject to   rl <= Ax <= ru,   l <= x <= u

with `integrality[j]` set for columns inside MARKER INTORG/INTEND.
"""
import numpy as np
from dataclasses import dataclass, field

INF = np.inf


@dataclass
class Problem:
    name: str = ""
    sense: str = "min"
    c: np.ndarray = None
    c0: float = 0.0
    A: object = None                      # scipy.sparse.csr_matrix
    rl: np.ndarray = None
    ru: np.ndarray = None
    lb: np.ndarray = None
    ub: np.ndarray = None
    integrality: np.ndarray = None
    Q: object = None                      # symmetric Hessian; objective is c'x + 1/2 x'Qx
    rownames: list = field(default_factory=list)
    colnames: list = field(default_factory=list)

    @property
    def nrow(self): return len(self.rownames)

    @property
    def ncol(self): return len(self.colnames)


# Fixed-column MPS field positions, 0-based [start, length].
_FIXED = ((1, 2), (4, 8), (14, 8), (24, 12), (39, 8), (49, 12))
_MAXFREE = {"ROWS": 2, "COLUMNS": 5, "RHS": 5, "RANGES": 5, "BOUNDS": 4, "QUADOBJ": 3}


def _split_fixed(raw):
    out = []
    for beg, ln in _FIXED:
        if len(raw) <= beg:
            break
        piece = raw[beg:beg + ln].strip()
        if piece:
            out.append(piece)
    return out


def _is_fixed_format(path):
    """True when any data line carries more fields than its section allows.

    A name with a space in it is legal in fixed-column MPS -- QFORPLAN in the
    Maros and Meszaros set has a column called "DEDO3 11" -- and splitting such
    a line on whitespace silently builds a different model. The verdict is taken
    over the whole file, because only the COLUMNS lines overflow; an RHS line
    for a spaced row name still splits into a legal number of fields.
    """
    section = None
    with open(path, "r", errors="replace") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("*"):
                continue
            if not raw[0].isspace():
                head = raw.split()[0].upper()
                if head in ("QUADOBJ", "QMATRIX", "QSECTION"):
                    head = "QUADOBJ"
                section = head if head in _MAXFREE else None
                if head == "ENDATA":
                    break
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


def read_mps(path):
    from scipy.sparse import coo_matrix

    fixed = _is_fixed_format(path)

    section = None
    objname = None
    freerows = set()
    rowkind, rowindex, rownames = {}, {}, []
    colindex, colnames = {}, []
    obj = {}
    entries = []                                   # (i, j, v)
    rhs, ranges = {}, {}
    lb, ub = {}, {}
    integer_cols, in_int_marker = set(), False
    quad = []                                      # (col_i, col_j, value), lower triangle
    sense = "min"
    name = ""
    objconst = 0.0

    def col(nm):
        if nm not in colindex:
            colindex[nm] = len(colnames); colnames.append(nm)
            lb.setdefault(nm, 0.0)
        return colindex[nm]

    with open(path, "r", errors="replace") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("*"):
                continue
            if not raw[0].isspace():                       # section header
                t = raw.split()
                head = t[0].upper()
                if head == "NAME":
                    name = t[1] if len(t) > 1 else ""
                elif head == "OBJSENSE":
                    section = "OBJSENSE"
                    if len(t) > 1:
                        sense = "max" if t[1].upper().startswith("MAX") else "min"
                elif head in ("QUADOBJ", "QMATRIX", "QSECTION"):
                    section = "QUADOBJ"
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
                sense = "max" if t[0].upper().startswith("MAX") else "min"

            elif section == "ROWS":
                kind, nm = t[0].upper(), t[1]
                if kind == "N":
                    if objname is None:
                        objname = nm
                    else:
                        freerows.add(nm)               # extra N rows are ignored
                else:
                    rowindex[nm] = len(rownames); rownames.append(nm); rowkind[nm] = kind

            elif section == "COLUMNS":
                if len(t) >= 3 and t[1].upper() == "'MARKER'":
                    tag = " ".join(t).upper()
                    if "INTORG" in tag: in_int_marker = True
                    elif "INTEND" in tag: in_int_marker = False
                    continue
                cname = t[0]
                j = col(cname)
                if in_int_marker:
                    integer_cols.add(cname)
                for k in range(1, len(t) - 1, 2):
                    rname, val = t[k], float(t[k + 1])
                    if rname == objname:
                        obj[j] = obj.get(j, 0.0) + val
                    elif rname in rowindex:
                        entries.append((rowindex[rname], j, val))
                    elif rname in freerows:
                        pass
                    else:
                        raise ValueError(f"{path}: unknown row {rname!r} in COLUMNS")

            elif section == "RHS":
                # the first token is the RHS vector's name, which nobody uses;
                # a file may also omit it, so detect that by looking at token 0
                start = 0 if (t[0] == objname or t[0] in rowindex) else 1
                for k in range(start, len(t) - 1, 2):
                    rname, val = t[k], float(t[k + 1])
                    if rname == objname:
                        objconst = -val                # MPS convention: negated
                    elif rname in rowindex:
                        rhs[rname] = val
                    elif rname not in freerows:
                        raise ValueError(f"{path}: unknown row {rname!r} in RHS")

            elif section == "RANGES":
                start = 0 if t[0] in rowindex else 1
                for k in range(start, len(t) - 1, 2):
                    if t[k] in rowindex:
                        ranges[t[k]] = float(t[k + 1])

            elif section == "QUADOBJ":
                # `col_j  col_i  value`, the lower triangle of Q
                for k in range(1, len(t) - 1, 2):
                    cj, ci, v = t[0], t[k], float(t[k + 1])
                    quad.append((col(ci), col(cj), v))

            elif section == "BOUNDS":
                bt = t[0].upper()
                # BOUNDS lines are `type setname column [value]`, but the set
                # name is optional in some files.
                if len(t) >= 3 and t[2] in colindex:
                    cname, val = t[2], (float(t[3]) if len(t) > 3 else None)
                elif len(t) >= 2 and t[1] in colindex:
                    cname, val = t[1], (float(t[2]) if len(t) > 2 else None)
                elif len(t) >= 3:
                    cname, val = t[2], (float(t[3]) if len(t) > 3 else None)
                else:
                    continue
                col(cname)
                if bt == "UP":
                    ub[cname] = val
                    # classic convention: a negative upper bound on a column
                    # still at its default zero lower bound frees the lower bound
                    if val is not None and val < 0 and lb.get(cname, 0.0) == 0.0:
                        lb[cname] = -INF
                elif bt == "LO": lb[cname] = val
                elif bt == "FX": lb[cname] = ub[cname] = val
                elif bt == "FR": lb[cname] = -INF; ub[cname] = INF
                elif bt == "MI": lb[cname] = -INF
                elif bt == "PL": ub[cname] = INF
                elif bt == "BV": lb[cname] = 0.0; ub[cname] = 1.0; integer_cols.add(cname)
                elif bt == "LI": lb[cname] = val; integer_cols.add(cname)
                elif bt == "UI": ub[cname] = val; integer_cols.add(cname)
                else:
                    raise ValueError(f"{path}: unsupported bound type {bt!r}")

    n, m = len(colnames), len(rownames)
    c = np.zeros(n)
    for j, v in obj.items():
        c[j] = v

    rl = np.full(m, -INF); ru = np.full(m, INF)
    for nm, i in rowindex.items():
        b = rhs.get(nm, 0.0)
        k = rowkind[nm]
        if k == "L":   rl[i], ru[i] = -INF, b
        elif k == "G": rl[i], ru[i] = b, INF
        else:          rl[i] = ru[i] = b
    for nm, r in ranges.items():
        i, k, b = rowindex[nm], rowkind[nm], rhs.get(nm, 0.0)
        if k == "L":   rl[i] = b - abs(r)
        elif k == "G": ru[i] = b + abs(r)
        else:                                        # E row
            if r >= 0: rl[i], ru[i] = b, b + r
            else:      rl[i], ru[i] = b + r, b

    L = np.array([lb.get(nm, 0.0) if lb.get(nm, 0.0) is not None else 0.0 for nm in colnames],
                 dtype=float)
    U = np.array([ub.get(nm, INF) if ub.get(nm, INF) is not None else INF for nm in colnames],
                 dtype=float)
    integ = np.array([1 if nm in integer_cols else 0 for nm in colnames], dtype=int)

    Qm = None
    if quad:
        from scipy.sparse import coo_matrix as _coo
        n_now = len(colnames)
        ii = [a for a, b, v in quad]; jj = [b for a, b, v in quad]
        vv = [v for a, b, v in quad]
        # stored lower triangle -> full symmetric matrix.
        # (named Ltri, not L: L is already the lower-bound vector below)
        Ltri = _coo((vv, (ii, jj)), shape=(n_now, n_now)).tocsr()
        Qm = (Ltri + Ltri.T).tocsr()
        d = Ltri.diagonal()
        Qm.setdiag(Qm.diagonal() - d)              # the diagonal was counted twice
        Qm.eliminate_zeros()

    if entries:
        i, j, v = zip(*entries)
        A = coo_matrix((v, (i, j)), shape=(m, n)).tocsr()
    else:
        from scipy.sparse import csr_matrix
        A = csr_matrix((m, n))
    A.sum_duplicates()

    if Qm is not None and Qm.shape[0] != len(colnames):
        from scipy.sparse import csr_matrix as _csr
        Qm = _csr((Qm.data, Qm.indices, Qm.indptr), shape=(len(colnames), len(colnames)))

    return Problem(name=name, sense=sense, c=c, c0=objconst, A=A, rl=rl, ru=ru,
                   lb=L, ub=U, integrality=integ, Q=Qm,
                   rownames=rownames, colnames=colnames)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        q = read_mps(p)
        print(f"{q.name:<10} {q.nrow:>7} rows {q.ncol:>7} cols {q.A.nnz:>9} nnz "
              f"{int(q.integrality.sum()):>6} integer "
              f"{(q.Q.nnz if q.Q is not None else 0):>8} quad  "
              f"sense={q.sense} const={q.c0:g}")
