"""Locating and binding the IGAOS shared library.

This module is the only place that knows about ctypes.  Everything above it
works with Python objects, so the binding layer can be swapped (for a C
extension, or cffi) without touching the public API.

Deliberately dependency-free: ctypes ships with CPython, so ``pip install``
needs no compiler, no numpy, and no build step.  The cost is that array
transfers go through ctypes buffers rather than the buffer protocol, which is
fine at model-building sizes and is why :meth:`Model.load` exists for the case
where it is not.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Optional

__all__ = ["load_library", "LibraryNotFound"]


class LibraryNotFound(RuntimeError):
    """Raised when the IGAOS shared library cannot be located."""


def _candidate_names() -> list:
    if sys.platform == "darwin":
        return ["libigaos.dylib", "libigaos.0.dylib"]
    if os.name == "nt":
        return ["igaos.dll", "libigaos.dll"]
    return ["libigaos.so", "libigaos.so.0"]


def _candidate_directories() -> list:
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = [
        here,                                              # shipped inside the wheel
        os.path.join(here, "lib"),
        os.path.abspath(os.path.join(here, "..", "..")),   # a source checkout
    ]
    # A build tree sitting next to the source, which is where it lands during
    # development and in CI.
    for build in ("build", "build-release", "cmake-build-release"):
        dirs.append(os.path.abspath(os.path.join(here, "..", "..", build)))
    env = os.environ.get("IGAOS_LIBRARY_PATH")
    if env:
        dirs.insert(0, env)
    return dirs


def load_library(path: Optional[str] = None) -> ctypes.CDLL:
    """Load libigaos and return the ``ctypes`` handle.

    Search order: an explicit *path*, then ``$IGAOS_LIBRARY`` naming the file
    directly, then the directories in :func:`_candidate_directories`, then the
    platform loader's own search path.
    """
    tried = []

    explicit = path or os.environ.get("IGAOS_LIBRARY")
    if explicit:
        tried.append(explicit)
        if os.path.exists(explicit):
            return ctypes.CDLL(explicit)

    for directory in _candidate_directories():
        for name in _candidate_names():
            candidate = os.path.join(directory, name)
            tried.append(candidate)
            if os.path.exists(candidate):
                return ctypes.CDLL(candidate)

    found = ctypes.util.find_library("igaos")
    if found:
        return ctypes.CDLL(found)
    tried.append("(system library path)")

    raise LibraryNotFound(
        "could not find the IGAOS shared library.\n"
        "Build it with:\n"
        "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build\n"
        "then either install it, or point IGAOS_LIBRARY at the built file:\n"
        "    export IGAOS_LIBRARY=/path/to/libigaos.so\n"
        "Looked in:\n  " + "\n  ".join(tried)
    )


_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_longlong_p = ctypes.POINTER(ctypes.c_longlong)


def bind(lib: ctypes.CDLL) -> ctypes.CDLL:
    """Attach argument and return types.

    Without this every call would default to ``int`` returns and unchecked
    arguments, which on a 64-bit platform silently truncates every pointer and
    every double the library hands back.
    """
    handle = ctypes.c_void_p

    sig = {
        "igaos_version": ([], ctypes.c_char_p),
        "igaos_version_numbers": ([_c_int_p, _c_int_p, _c_int_p], None),
        "igaos_create": ([], handle),
        "igaos_destroy": ([handle], None),
        "igaos_add_column": ([handle, ctypes.c_double, ctypes.c_double,
                              ctypes.c_double, ctypes.c_int, ctypes.c_char_p], ctypes.c_int),
        "igaos_add_row": ([handle, ctypes.c_double, ctypes.c_double,
                           ctypes.c_char_p], ctypes.c_int),
        "igaos_set_element": ([handle, ctypes.c_int, ctypes.c_int, ctypes.c_double], ctypes.c_int),
        "igaos_set_quadratic": ([handle, ctypes.c_int, ctypes.c_int, ctypes.c_double], ctypes.c_int),
        "igaos_set_sense": ([handle, ctypes.c_int], ctypes.c_int),
        "igaos_set_objective_offset": ([handle, ctypes.c_double], ctypes.c_int),
        "igaos_load": ([handle, ctypes.c_int, ctypes.c_int, _c_int_p, _c_int_p,
                        _c_double_p, _c_double_p, _c_double_p, _c_double_p,
                        _c_double_p, _c_double_p, _c_int_p], ctypes.c_int),
        "igaos_read_mps": ([handle, ctypes.c_char_p], ctypes.c_int),
        "igaos_write_mps": ([handle, ctypes.c_char_p], ctypes.c_int),
        "igaos_set_int_param": ([handle, ctypes.c_char_p, ctypes.c_longlong], ctypes.c_int),
        "igaos_get_int_param": ([handle, ctypes.c_char_p, _c_longlong_p], ctypes.c_int),
        "igaos_set_double_param": ([handle, ctypes.c_char_p, ctypes.c_double], ctypes.c_int),
        "igaos_get_double_param": ([handle, ctypes.c_char_p, _c_double_p], ctypes.c_int),
        "igaos_solve": ([handle], ctypes.c_int),
        "igaos_get_status": ([handle], ctypes.c_int),
        "igaos_status_string": ([ctypes.c_int], ctypes.c_char_p),
        "igaos_get_objective": ([handle], ctypes.c_double),
        "igaos_get_best_bound": ([handle], ctypes.c_double),
        "igaos_get_mip_gap": ([handle], ctypes.c_double),
        "igaos_get_solve_time": ([handle], ctypes.c_double),
        "igaos_get_iterations": ([handle], ctypes.c_longlong),
        "igaos_get_nodes": ([handle], ctypes.c_longlong),
        "igaos_get_num_rows": ([handle], ctypes.c_int),
        "igaos_get_num_cols": ([handle], ctypes.c_int),
        "igaos_get_solution": ([handle, _c_double_p], ctypes.c_int),
        "igaos_get_reduced_costs": ([handle, _c_double_p], ctypes.c_int),
        "igaos_get_row_activity": ([handle, _c_double_p], ctypes.c_int),
        "igaos_get_row_duals": ([handle, _c_double_p], ctypes.c_int),
        "igaos_get_cut_counts": ([handle, _c_int_p, _c_int_p, _c_int_p,
                                  _c_int_p, _c_int_p], ctypes.c_int),
        "igaos_get_root_bound_before_cuts": ([handle], ctypes.c_double),
        "igaos_get_root_bound_after_cuts": ([handle], ctypes.c_double),
        "igaos_get_algorithm": ([handle], ctypes.c_char_p),
        "igaos_last_error": ([handle], ctypes.c_char_p),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib
