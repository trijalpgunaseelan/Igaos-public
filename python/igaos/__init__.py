"""IGAOS -- Indigenous GPU-Accelerated Optimization Solver.

Linear, mixed-integer and convex quadratic programming, from Python, with no
third-party solver underneath and no build step to install.

    >>> from igaos import Model, INF
    >>> m = Model(sense="maximize")
    >>> x = m.add_variable(0, INF, cost=3.0, name="x")
    >>> y = m.add_variable(0, INF, cost=5.0, name="y")
    >>> m.add_constraint({x: 1.0}, upper=4.0)
    >>> m.add_constraint({y: 2.0}, upper=12.0)
    >>> m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)
    >>> result = m.solve()
    >>> round(result.objective, 6)
    36.0
    >>> [round(v, 6) for v in result.x]
    [2.0, 6.0]

Design notes
------------
The binding is a thin, honest wrapper over the C ABI: it does not build an
expression algebra, and it does not pretend to be a modelling language.  A
model is variables, a sparse matrix and bounds, which is what the solver
actually consumes.  For anything larger than a few thousand nonzeros, use
:meth:`Model.load`, which hands the whole matrix over in one call instead of
one Python-to-C round trip per coefficient.
"""

from __future__ import annotations

import ctypes
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ._library import LibraryNotFound, bind, load_library

__all__ = [
    "Model", "Result", "Status", "VarType", "INF",
    "IgaosError", "LibraryNotFound", "version",
]

INF = 1e30

_OK = 0
_ERROR_MESSAGES = {
    -1: "null handle or buffer",
    -2: "index out of range",
    -3: "unknown parameter name",
    -4: "parameter value rejected",
    -5: "file could not be read or written",
    -6: "result requested before solve()",
    -7: "internal error",
}


class IgaosError(RuntimeError):
    """A call into the solver failed.  Carries the library's own message."""


class Status:
    """Solve outcomes, matching the ``IGAOS_STATUS_*`` constants."""
    NOT_SOLVED = 0
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    ITERATION_LIMIT = 4
    TIME_LIMIT = 5
    NODE_LIMIT = 6
    NUMERICAL_ERROR = 7
    INTERRUPTED = 8
    FEASIBLE = 9          #: incumbent found, optimality not proven


class VarType:
    CONTINUOUS = 0
    INTEGER = 1
    BINARY = 2


_VTYPE_ALIASES = {
    "c": VarType.CONTINUOUS, "continuous": VarType.CONTINUOUS,
    "i": VarType.INTEGER, "integer": VarType.INTEGER, "int": VarType.INTEGER,
    "b": VarType.BINARY, "binary": VarType.BINARY, "bin": VarType.BINARY,
}

_lib = None


def _library():
    global _lib
    if _lib is None:
        _lib = bind(load_library())
    return _lib


def version() -> str:
    """Version string of the loaded shared library."""
    return _library().igaos_version().decode()


def _as_vtype(vtype) -> int:
    if isinstance(vtype, int):
        return vtype
    key = str(vtype).strip().lower()
    if key not in _VTYPE_ALIASES:
        raise ValueError(
            "variable type must be one of 'continuous', 'integer', 'binary' "
            "(got %r)" % (vtype,))
    return _VTYPE_ALIASES[key]


def _double_array(values: Optional[Sequence[float]], n: int, default: float):
    arr = (ctypes.c_double * n)()
    if values is None:
        for i in range(n):
            arr[i] = default
    else:
        if len(values) != n:
            raise ValueError("expected %d values, got %d" % (n, len(values)))
        for i, v in enumerate(values):
            arr[i] = float(v)
    return arr


class Result:
    """What a solve produced.  Plain data; safe to keep after the model changes."""

    __slots__ = ("status", "status_name", "objective", "best_bound", "mip_gap",
                 "iterations", "nodes", "solve_time", "algorithm",
                 "x", "reduced_costs", "row_activity", "row_duals",
                 "cuts_applied", "cuts_gomory", "cuts_cover", "cuts_mir",
                 "cut_rounds", "root_bound_before_cuts", "root_bound_after_cuts")

    def __init__(self, **kw):
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    @property
    def optimal(self) -> bool:
        return self.status == Status.OPTIMAL

    @property
    def feasible(self) -> bool:
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    def __repr__(self) -> str:
        return ("Result(status=%s, objective=%.10g, iterations=%d, nodes=%d, "
                "time=%.3fs, algorithm=%r)"
                % (self.status_name, self.objective or 0.0, self.iterations or 0,
                   self.nodes or 0, self.solve_time or 0.0, self.algorithm))


class Model:
    """A linear, mixed-integer or convex quadratic program.

    Parameters
    ----------
    sense:
        ``"minimize"`` (default) or ``"maximize"``.
    name:
        Ignored by the solver; kept for the caller's convenience.
    """

    def __init__(self, sense: str = "minimize", name: str = ""):
        self._lib = _library()
        self._handle = self._lib.igaos_create()
        if not self._handle:
            raise IgaosError("could not create a solver handle")
        self.name = name
        self._num_col = 0
        self._num_row = 0
        self._var_names: List[str] = []
        self._row_names: List[str] = []
        self._solved = False
        self.sense = sense

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self) -> None:
        """Release the underlying handle.  Safe to call more than once."""
        if getattr(self, "_handle", None):
            self._lib.igaos_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- error plumbing ----------------------------------------------------
    def _check(self, code: int, what: str) -> int:
        if code >= _OK:
            return code
        detail = self._lib.igaos_last_error(self._handle)
        detail = detail.decode() if detail else _ERROR_MESSAGES.get(code, "error %d" % code)
        raise IgaosError("%s: %s" % (what, detail))

    # -- structure ---------------------------------------------------------
    @property
    def sense(self) -> str:
        return self._sense

    @sense.setter
    def sense(self, value: str) -> None:
        key = str(value).strip().lower()
        if key.startswith("max"):
            self._check(self._lib.igaos_set_sense(self._handle, -1), "set sense")
            self._sense = "maximize"
        elif key.startswith("min"):
            self._check(self._lib.igaos_set_sense(self._handle, 1), "set sense")
            self._sense = "minimize"
        else:
            raise ValueError("sense must be 'minimize' or 'maximize', got %r" % (value,))
        self._solved = False

    def add_variable(self, lower: float = 0.0, upper: float = INF, cost: float = 0.0,
                     vtype="continuous", name: str = "") -> int:
        """Add one column and return its index."""
        idx = self._check(
            self._lib.igaos_add_column(self._handle, float(lower), float(upper),
                                       float(cost), _as_vtype(vtype),
                                       name.encode() if name else None),
            "add variable")
        self._var_names.append(name or ("x%d" % idx))
        self._num_col += 1
        self._solved = False
        return idx

    def add_binary(self, cost: float = 0.0, name: str = "") -> int:
        return self.add_variable(0.0, 1.0, cost, VarType.BINARY, name)

    def add_integer(self, lower: float = 0.0, upper: float = INF,
                    cost: float = 0.0, name: str = "") -> int:
        return self.add_variable(lower, upper, cost, VarType.INTEGER, name)

    def add_constraint(self, coefficients: Mapping[int, float],
                       lower: float = -INF, upper: float = INF,
                       name: str = "") -> int:
        """Add one row ``lower <= sum(coefficients) <= upper`` and return its index.

        A range with both bounds finite is a ranged row; equal bounds make an
        equality.  There is no separate "sense" argument because the solver has
        exactly one row form.
        """
        row = self._check(
            self._lib.igaos_add_row(self._handle, float(lower), float(upper),
                                    name.encode() if name else None),
            "add constraint")
        for col, value in coefficients.items():
            if value == 0.0:
                continue
            self._check(self._lib.igaos_set_element(self._handle, int(row), int(col),
                                                    float(value)),
                        "set element (%d, %d)" % (row, col))
        self._row_names.append(name or ("r%d" % row))
        self._num_row += 1
        self._solved = False
        return row

    def set_quadratic(self, i: int, j: int, value: float) -> None:
        """Set one entry of the lower triangle of Q; the objective is c'x + 1/2 x'Qx.

        Q must be positive semidefinite -- the solver assumes convexity and does
        not check it, because checking would cost an eigenvalue computation on
        every solve.  A model with any Q entry is solved by the interior point
        method.
        """
        if j > i:
            i, j = j, i
        self._check(self._lib.igaos_set_quadratic(self._handle, int(i), int(j), float(value)),
                    "set quadratic entry")
        self._solved = False

    @property
    def objective_offset(self) -> float:
        return self._offset

    @objective_offset.setter
    def objective_offset(self, value: float) -> None:
        self._check(self._lib.igaos_set_objective_offset(self._handle, float(value)),
                    "set objective offset")
        self._offset = float(value)
        self._solved = False

    _offset = 0.0

    # -- bulk load ---------------------------------------------------------
    def load(self, num_rows: int, num_cols: int,
             colptr: Sequence[int], rowidx: Sequence[int], values: Sequence[float],
             obj: Optional[Sequence[float]] = None,
             col_lower: Optional[Sequence[float]] = None,
             col_upper: Optional[Sequence[float]] = None,
             row_lower: Optional[Sequence[float]] = None,
             row_upper: Optional[Sequence[float]] = None,
             col_types: Optional[Sequence[int]] = None) -> None:
        """Replace the model with a matrix given in compressed sparse column form.

        One call instead of one per coefficient.  On a model with 45,000
        nonzeros that is the difference between a noticeable pause and no pause
        at all, because each :meth:`add_constraint` coefficient costs a full
        Python-to-C transition.
        """
        nnz = len(values)
        if len(rowidx) != nnz:
            raise ValueError("rowidx and values must have the same length")
        if len(colptr) != num_cols + 1:
            raise ValueError("colptr must have num_cols + 1 entries")

        cp = (ctypes.c_int * (num_cols + 1))(*[int(v) for v in colptr])
        ri = (ctypes.c_int * max(nnz, 1))(*[int(v) for v in rowidx]) if nnz else None
        va = (ctypes.c_double * max(nnz, 1))(*[float(v) for v in values]) if nnz else None
        ob = _double_array(obj, num_cols, 0.0)
        cl = _double_array(col_lower, num_cols, 0.0)
        cu = _double_array(col_upper, num_cols, INF)
        rl = _double_array(row_lower, num_rows, -INF)
        ru = _double_array(row_upper, num_rows, INF)
        ct = None
        if col_types is not None:
            ct = (ctypes.c_int * num_cols)(*[_as_vtype(t) for t in col_types])

        self._check(self._lib.igaos_load(self._handle, int(num_rows), int(num_cols),
                                         cp, ri, va, ob, cl, cu, rl, ru, ct), "bulk load")
        self._num_col = num_cols
        self._num_row = num_rows
        self._var_names = ["x%d" % j for j in range(num_cols)]
        self._row_names = ["r%d" % i for i in range(num_rows)]
        self._solved = False

    # -- file interchange --------------------------------------------------
    @classmethod
    def from_mps(cls, path: str) -> "Model":
        """Read an MPS file.  MPS is the universal interchange format, so a model
        written for another solver loads here unchanged."""
        m = cls()
        m.read_mps(path)
        return m

    def read_mps(self, path: str) -> None:
        self._check(self._lib.igaos_read_mps(self._handle, str(path).encode()), "read MPS")
        self._num_col = self._lib.igaos_get_num_cols(self._handle)
        self._num_row = self._lib.igaos_get_num_rows(self._handle)
        self._var_names = ["x%d" % j for j in range(self._num_col)]
        self._row_names = ["r%d" % i for i in range(self._num_row)]
        self._solved = False

    def write_mps(self, path: str) -> None:
        self._check(self._lib.igaos_write_mps(self._handle, str(path).encode()), "write MPS")

    # -- parameters --------------------------------------------------------
    def set_param(self, name: str, value) -> None:
        """Set one solver parameter.  Integers and booleans go to the integer
        table, floats to the double table; see igaos.h for the full list."""
        if isinstance(value, bool) or isinstance(value, int):
            code = self._lib.igaos_set_int_param(self._handle, name.encode(), int(value))
            if code == -3:   # not an integer parameter: try the double table
                code = self._lib.igaos_set_double_param(self._handle, name.encode(), float(value))
        else:
            code = self._lib.igaos_set_double_param(self._handle, name.encode(), float(value))
            if code == -3:
                code = self._lib.igaos_set_int_param(self._handle, name.encode(), int(value))
        self._check(code, "set parameter %r" % name)

    def get_param(self, name: str):
        iv = ctypes.c_longlong()
        if self._lib.igaos_get_int_param(self._handle, name.encode(), ctypes.byref(iv)) == _OK:
            return int(iv.value)
        dv = ctypes.c_double()
        if self._lib.igaos_get_double_param(self._handle, name.encode(), ctypes.byref(dv)) == _OK:
            return float(dv.value)
        raise IgaosError("unknown parameter %r" % name)

    # -- solve -------------------------------------------------------------
    def solve(self, **params) -> Result:
        """Solve, optionally setting parameters for this call.

            m.solve(time_limit=60.0, verbosity=1, cuts=True)
        """
        for key, value in params.items():
            self.set_param(key, value)
        self._check(self._lib.igaos_solve(self._handle), "solve")
        self._solved = True
        return self.result()

    def result(self) -> Result:
        if not self._solved:
            raise IgaosError("solve() has not been called")
        lib, h = self._lib, self._handle
        ncol = lib.igaos_get_num_cols(h)
        nrow = lib.igaos_get_num_rows(h)

        def fetch(fn, n):
            if n == 0:
                return []
            buf = (ctypes.c_double * n)()
            if fn(h, buf) != _OK:
                return []
            return list(buf)

        applied = ctypes.c_int(); gomory = ctypes.c_int(); cover = ctypes.c_int()
        mir = ctypes.c_int(); rounds = ctypes.c_int()
        lib.igaos_get_cut_counts(h, ctypes.byref(applied), ctypes.byref(gomory),
                                 ctypes.byref(cover), ctypes.byref(mir), ctypes.byref(rounds))

        status = lib.igaos_get_status(h)
        return Result(
            status=status,
            status_name=lib.igaos_status_string(status).decode(),
            objective=lib.igaos_get_objective(h),
            best_bound=lib.igaos_get_best_bound(h),
            mip_gap=lib.igaos_get_mip_gap(h),
            iterations=lib.igaos_get_iterations(h),
            nodes=lib.igaos_get_nodes(h),
            solve_time=lib.igaos_get_solve_time(h),
            algorithm=(lib.igaos_get_algorithm(h) or b"").decode(),
            x=fetch(lib.igaos_get_solution, ncol),
            reduced_costs=fetch(lib.igaos_get_reduced_costs, ncol),
            row_activity=fetch(lib.igaos_get_row_activity, nrow),
            row_duals=fetch(lib.igaos_get_row_duals, nrow),
            cuts_applied=applied.value, cuts_gomory=gomory.value,
            cuts_cover=cover.value, cuts_mir=mir.value, cut_rounds=rounds.value,
            root_bound_before_cuts=lib.igaos_get_root_bound_before_cuts(h),
            root_bound_after_cuts=lib.igaos_get_root_bound_after_cuts(h),
        )

    # -- introspection -----------------------------------------------------
    @property
    def num_variables(self) -> int:
        return self._lib.igaos_get_num_cols(self._handle)

    @property
    def num_constraints(self) -> int:
        return self._lib.igaos_get_num_rows(self._handle)

    def __repr__(self) -> str:
        return "Model(%s, %d variables, %d constraints)" % (
            self._sense, self.num_variables, self.num_constraints)
