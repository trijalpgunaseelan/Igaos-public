#!/usr/bin/env bash
# IGAOS — runs every section of RUNNING.md end to end and prints the output.
#
#   ./run_demo.sh          build, test, solve, cuts demo, four paths, Python, C
#   ./run_demo.sh --bench  the above plus the HiGHS head-to-head (~8 min, needs SciPy)
#   ./run_demo.sh --web    build if needed, then open the demo console in a browser
#
# Works on macOS (Apple Clang) and Linux (GCC/Clang). Nothing to install.

set -u
cd "$(dirname "$0")"

BENCH=0
[ "${1:-}" = "--bench" ] && BENCH=1

# --- a build/ directory created somewhere else -------------------------------
# CMake bakes absolute paths into build/CMakeCache.txt. Copy a tree between
# machines -- or between a synced folder and the machine it is synced to -- and
# the cache still points at the old path, so configure fails with "does not
# match the source used to generate cache". Nobody should have to read a
# troubleshooting section for that: check it and start clean.
stale_build() {
    [ -f build/CMakeCache.txt ] || return 1
    _here=$(pwd -P)
    _src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt | head -1)
    _bin=$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' build/CMakeCache.txt | head -1)
    [ "$_src" = "$_here" ] && [ "$_bin" = "$_here/build" ] && return 1
    return 0
}
drop_stale_build() {
    if stale_build; then
        echo "  build/ was configured for a different path -- starting it clean"
        rm -rf build
    fi
}


if [ "${1:-}" = "--web" ]; then
    drop_stale_build
    if [ ! -x build/igaos ] || ! ./build/igaos --version >/dev/null 2>&1; then
        echo "building the solver first…"
        cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /tmp/igaos_cmake.log 2>&1 \
            && cmake --build build -j > /tmp/igaos_build.log 2>&1 \
            || { echo "build failed — see /tmp/igaos_build.log"; exit 1; }
    fi
    exec python3 demo/server.py
fi

case "$(uname -s)" in
    Darwin) LIB=build/libigaos.dylib; LIBVAR=DYLD_LIBRARY_PATH ;;
    *)      LIB=build/libigaos.so;    LIBVAR=LD_LIBRARY_PATH   ;;
esac

FAIL=0
step() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
ok()   { printf '  \033[32m[ok]\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=1; }

# ---------------------------------------------------------------- 2. build

drop_stale_build

step "Build (RUNNING.md section 2)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /tmp/igaos_cmake.log 2>&1 \
    || { bad "cmake configure failed — see /tmp/igaos_cmake.log"; tail -20 /tmp/igaos_cmake.log; exit 1; }
grep -q "Could NOT find OpenMP" /tmp/igaos_cmake.log \
    && echo "  note: no OpenMP (expected with Apple Clang) — building single-threaded"
cmake --build build -j > /tmp/igaos_build.log 2>&1 \
    || { bad "build failed — see /tmp/igaos_build.log"; tail -30 /tmp/igaos_build.log; exit 1; }
W=$(grep -ci warning /tmp/igaos_build.log)
[ "$W" -eq 0 ] && ok "built warning-free" || echo "  $W warning lines in /tmp/igaos_build.log"

step "Test suites (RUNNING.md section 2)"
( cd build && ctest --output-on-failure ) || bad "ctest"
./build/igaos_tests    2>&1 | tail -2
./build/igaos_capi_test 2>&1 | tail -2

# ------------------------------------------------------- 3. generate models

step "Generate instances (RUNNING.md section 3)"
if python3 -c "import numpy" 2>/dev/null; then
    python3 - <<'PY' || exit 1
import sys; sys.path.insert(0, 'bench')
from generate import SUITE
for tag, fn, family, kind in SUITE:
    if tag.startswith('blend_s'): fn(0).write('example_lp.mps')
    if tag.startswith('uc_m'):    fn(0).write('example_milp.mps')
print("  wrote example_lp.mps (refinery blending, LP)")
print("  wrote example_milp.mps (unit commitment, MILP)")
PY
else
    cp demo/models/blend_s.mps example_lp.mps
    cp demo/models/uc_m.mps    example_milp.mps
    echo "  numpy not installed — using the copies shipped in demo/models/ instead"
    echo "  (identical files; the generators need numpy, the solver needs nothing)"
fi

# ------------------------------------------------------------------ 4. solve

step "Solve an LP (RUNNING.md section 4)"
./build/igaos example_lp.mps -v

# --------------------------------------------------------------- 5. the demo

step "Cut separation on and off — the demo worth showing (section 5)"
echo "--- cuts ON ---"
./build/igaos example_milp.mps -v  | grep -E "cuts|nodes|objective|iterations|time"
echo "--- cuts OFF ---"
./build/igaos example_milp.mps -v --no-cuts | grep -E "nodes|objective|iterations|time"
echo
echo "  Same objective either way. The node count is the point: the root cuts"
echo "  close the gap, so the tree finishes at the root instead of branching."
echo "  Counts differ by a few percent on ARM vs x86 — see RUNNING.md section 5."

# --------------------------------------------------------- 6. the four paths

step "The four solve paths (RUNNING.md section 6)"
for a in primal dual interior pdhg; do
    printf '  %-9s ' "$a:"
    ./build/igaos example_lp.mps -q --algorithm $a
done
echo
./build/igaos example_lp.mps -v --algorithm interior 2>&1 | grep -E "^  (interior|crossover)"

# ----------------------------------------------------------------- 7. Python

step "Python bindings (RUNNING.md section 7)"
export PYTHONPATH=$PWD/python
export IGAOS_LIBRARY=$PWD/$LIB
python3 -c "
from igaos import Model, INF
m = Model(sense='maximize')
x = m.add_variable(0, INF, cost=3.0)
y = m.add_variable(0, INF, cost=5.0)
m.add_constraint({x: 1.0}, upper=4.0)
m.add_constraint({y: 2.0}, upper=12.0)
m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)
r = m.solve(); print(' ', r); print('  x =', r.x)
" || bad "python quickstart"
python3 -m unittest discover -s python/tests 2>&1 | tail -3

# ---------------------------------------------------------------------- 8. C

step "C ABI against the shared library (RUNNING.md section 8)"
cat > /tmp/igaos_demo.c <<'EOF'
#include <stdio.h>
#include "igaos/igaos.h"
int main(void) {
    igaos_model* m = igaos_create();
    igaos_set_sense(m, IGAOS_MAXIMIZE);
    int x = igaos_add_column(m, 0.0, IGAOS_INFINITY, 3.0, IGAOS_CONTINUOUS, "x");
    int y = igaos_add_column(m, 0.0, IGAOS_INFINITY, 5.0, IGAOS_CONTINUOUS, "y");
    int r0 = igaos_add_row(m, -IGAOS_INFINITY, 4.0, "c0");
    igaos_set_element(m, r0, x, 1.0);
    int r1 = igaos_add_row(m, -IGAOS_INFINITY, 12.0, "c1");
    igaos_set_element(m, r1, y, 2.0);
    int r2 = igaos_add_row(m, -IGAOS_INFINITY, 18.0, "c2");
    igaos_set_element(m, r2, x, 3.0); igaos_set_element(m, r2, y, 2.0);
    igaos_solve(m);
    printf("  status=%s objective=%g\n",
           igaos_status_string(igaos_get_status(m)), igaos_get_objective(m));
    igaos_destroy(m);
    return 0;
}
EOF
cc /tmp/igaos_demo.c -Iinclude -Lbuild -ligaos -o /tmp/igaos_demo \
    && env $LIBVAR=build /tmp/igaos_demo || bad "C example"

# ------------------------------------------------ 8b. mixed-integer quadratic

step "Mixed-integer quadratic programming (RUNNING.md section 10)"
cat > /tmp/igaos_miqp.mps <<'EOF'
NAME          MIQP1
ROWS
 N  COST
 L  C1
COLUMNS
    MARKER0  'MARKER'  'INTORG'
    X  COST  -6.0
    X  C1  1.0
    Y  COST  -8.0
    Y  C1  1.0
    MARKER1  'MARKER'  'INTEND'
RHS
    RHS  C1  10.0
BOUNDS
 UP BND  X  10.0
 UP BND  Y  10.0
QUADOBJ
    X  X  2.0
    Y  Y  2.0
ENDATA
EOF
./build/igaos /tmp/igaos_miqp.mps -v | grep -E "algorithm|status|objective|nodes"
echo
echo "  The optimum is -25 at (3,4). Branch and cut on the LP relaxation returns"
echo "  +20 -- it optimises the linear part to (0,10) and evaluates the quadratic"
echo "  objective there. That is why this path exists."

# ------------------------------------------------------- 8c. CUDA compilation

step "CUDA kernels: compile check (RUNNING.md section 11)"
if python3 -c "import ctypes,glob,site,os,sys
sys.exit(0 if any(glob.glob(os.path.join(sp,'nvidia','cuda_nvrtc','lib','libnvrtc.so*')) for sp in site.getsitepackages()) else 1)" 2>/dev/null; then
    python3 tools/cuda_compile_check.py cuda/pdhg_kernels.cu | sed 's/^/  /'
else
    echo "  skipped -- NVRTC not installed. To run it:"
    echo "    pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12"
    echo "  It compiles the kernels to SASS for sm_75/80/90 and needs no GPU."
fi

# ------------------------------------------------- 8d. checkable certificates

step "Certificates: proving the answer, in exact arithmetic (RUNNING.md section 12)"
./build/igaos example_lp.mps -q --certificate /tmp/igaos_demo.cert | head -1
python3 tools/verify_certificate.py example_lp.mps /tmp/igaos_demo.cert | sed 's/^/  /'
echo
echo "  Now corrupt one multiplier and check it again:"
python3 - <<'PYC'
out = []
done = False
for l in open('/tmp/igaos_demo.cert').read().splitlines():
    if l.startswith('y ') and not done:
        t = l.split(); t[2] = (float.fromhex(t[2]) * 1.05).hex()
        l = ' '.join(t); done = True
    out.append(l)
open('/tmp/igaos_demo_bad.cert', 'w').write('\n'.join(out) + '\n')
PYC
python3 tools/verify_certificate.py example_lp.mps /tmp/igaos_demo_bad.cert | sed 's/^/  /'
if python3 tools/verify_certificate.py example_lp.mps /tmp/igaos_demo_bad.cert >/dev/null 2>&1; then
    bad "the checker accepted a tampered certificate"
else
    ok "the checker rejected the tampered certificate"
fi

# --------------------------------------------------- 9. cut-validity fuzzing

step "Cut-validity fuzzing (RUNNING.md section 9)"
OMP=""
"${CXX:-c++}" -fopenmp -x c++ -E - < /dev/null > /dev/null 2>&1 && OMP="-fopenmp"
"${CXX:-c++}" -std=c++17 -O2 -Iinclude tools/fuzz_cuts.cpp build/libigaos_core.a $OMP \
    -o /tmp/igaos_fuzz 2>/tmp/igaos_fuzz_build.log \
    && /tmp/igaos_fuzz 600 7 | sed 's/^/  /' \
    || { bad "fuzz build failed"; tail -10 /tmp/igaos_fuzz_build.log; }
echo "  Expected: compared 578 optimal pairs, 0 failures"

# ------------------------------------------------------------ 9b. benchmark

if [ "$BENCH" -eq 1 ]; then
    step "Head-to-head against HiGHS (RUNNING.md section 9) — this takes ~8 minutes"
    if python3 -c "import scipy" 2>/dev/null; then
        IGAOS_BIN=$PWD/build/igaos python3 -u bench/run_bench.py --seeds 3 --time-limit 120
        step "What cut separation bought, as a paired run"
        IGAOS_BIN=$PWD/build/igaos python3 -u bench/cut_effect.py /tmp/inst/uc_*.mps
    else
        bad "SciPy not importable — see RUNNING.md section 1 for the virtualenv"
    fi
else
    echo
    echo "  (skipped the HiGHS benchmark — re-run as ./run_demo.sh --bench)"
fi

step "Done"
[ "$FAIL" -eq 0 ] && ok "every section completed" || bad "something above failed"
echo
echo "  For a demo with a screen behind it:  ./run_demo.sh --web"
echo "  Solver Studio at http://127.0.0.1:8420 -- the pipeline as cards that light"
echo "  up phase by phase, the full solver log, and the branch-and-bound gap"
echo "  closing live. Same binary, driven as a subprocess."
exit $FAIL
