#!/usr/bin/env bash
# ===========================================================================
#  start.sh — clone, run this, done.
#
#  This is the one command a new collaborator needs. It checks the toolchain,
#  builds the solver, runs the test suite, and opens Solver Studio in a
#  browser.
#
#      ./start.sh              build, test, open the console      (~2 minutes)
#      ./start.sh --no-web     build and test only, no browser
#      ./start.sh --quick      skip the tests, just build and open
#      ./start.sh --check      only check the toolchain and report
#
#  There is nothing to install first beyond a C++17 compiler and CMake, and
#  nothing to configure. If a prerequisite is missing this script says which
#  one and how to get it, rather than failing somewhere deep in a build log.
#
#  Full manual: RUNNING.md.  Repository map: CONTRIBUTING.md.
# ===========================================================================
set -u
cd "$(dirname "$0")"

MODE="${1:-}"
bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad()   { printf '  \033[31m✗\033[0m %s\n' "$*"; }
note()  { printf '    %s\n' "$*"; }

bold "IGAOS — Indigenous GPU-Accelerated Optimization Solver"
echo "SIH 2026, problem statement 26119 (MRPL)"
echo

# ------------------------------------------------------------------ 1. tools
bold "Checking what you have"
MISSING=0

if command -v cmake >/dev/null 2>&1; then
    ok "cmake        $(cmake --version | head -1 | awk '{print $3}')"
else
    bad "cmake        not found"
    note "macOS:         brew install cmake"
    note "Debian/Ubuntu: sudo apt install cmake build-essential"
    MISSING=1
fi

CXX_FOUND=""
for c in "${CXX:-}" c++ g++ clang++; do
    [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { CXX_FOUND="$c"; break; }
done
if [ -n "$CXX_FOUND" ]; then
    ok "C++ compiler $($CXX_FOUND --version 2>/dev/null | head -1)"
else
    bad "C++ compiler not found"
    note "macOS:         xcode-select --install"
    note "Debian/Ubuntu: sudo apt install build-essential"
    MISSING=1
fi

if command -v python3 >/dev/null 2>&1; then
    ok "python3      $(python3 --version 2>&1 | awk '{print $2}')   (for the console)"
else
    bad "python3      not found — the solver will still build, the console will not run"
    note "macOS:         brew install python3"
    note "Debian/Ubuntu: sudo apt install python3"
fi

if command -v nvcc >/dev/null 2>&1; then
    ok "nvcc         present — the CUDA kernels can be compiled (see RUNNING.md §11)"
else
    note "no nvcc — expected on most machines. Nothing here needs a GPU to run."
fi

echo
if [ "$MISSING" = "1" ]; then
    bad "Install the missing tools above, then run this again."
    exit 1
fi
[ "$MODE" = "--check" ] && { ok "Toolchain is complete."; exit 0; }

# ------------------------------------------- 2. a build/ from another machine
# CMake bakes absolute paths into build/CMakeCache.txt. Copy this tree between
# machines -- or unzip it somewhere new -- and configure fails with "does not
# match the source used to generate cache". Nobody should have to read a
# troubleshooting section for that.
if [ -f build/CMakeCache.txt ]; then
    _here=$(pwd -P)
    _src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt | head -1)
    if [ "$_src" != "$_here" ]; then
        echo "build/ was configured somewhere else — starting it clean"
        rm -rf build
    fi
fi

# ------------------------------------------------------------------ 3. build
echo
bold "Building"
echo "  this takes a minute or two the first time, and seconds after that"
if ! cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /tmp/igaos_cmake.log 2>&1; then
    bad "cmake configure failed"
    tail -20 /tmp/igaos_cmake.log
    exit 1
fi
grep -q "Could NOT find OpenMP" /tmp/igaos_cmake.log \
    && note "no OpenMP (normal with Apple Clang) — building single-threaded"

if ! cmake --build build -j > /tmp/igaos_build.log 2>&1; then
    bad "build failed — the last lines of /tmp/igaos_build.log:"
    tail -30 /tmp/igaos_build.log
    exit 1
fi
W=$(grep -ci warning /tmp/igaos_build.log || true)
if [ "$W" -eq 0 ]; then ok "built warning-free"; else ok "built ($W warning lines in /tmp/igaos_build.log)"; fi
ok "solver at ./build/igaos  ($(./build/igaos --version 2>/dev/null | head -1))"

# ------------------------------------------------------------------ 4. tests
if [ "$MODE" != "--quick" ]; then
    echo
    bold "Running the test suite"
    if ( cd build && ctest --output-on-failure ) > /tmp/igaos_ctest.log 2>&1; then
        ok "$(grep -E '[0-9]+% tests passed' /tmp/igaos_ctest.log | head -1)"
    else
        bad "tests failed — /tmp/igaos_ctest.log"
        tail -30 /tmp/igaos_ctest.log
        exit 1
    fi
    ./build/igaos_tests     2>&1 | tail -1 | sed 's/^/  /'
    ./build/igaos_capi_test 2>&1 | tail -1 | sed 's/^/  /'
fi

# ------------------------------------------------------------- 5. prove it runs
# The two example models are not committed -- .gitignore keeps scratch .mps out
# of the repository root -- so a fresh clone makes them from the bundled
# instances, exactly as RUNNING.md section 3 does.
[ -f example_lp.mps ]   || cp demo/models/blend_s.mps example_lp.mps   2>/dev/null
[ -f example_milp.mps ] || cp demo/models/uc_m.mps    example_milp.mps 2>/dev/null

echo
bold "Solving something"
echo "  example_lp.mps — a refinery blending model"
if [ -f example_lp.mps ]; then
    ./build/igaos example_lp.mps 2>&1 | tail -6 | sed 's/^/  /'
else
    bad "demo/models is missing — try a clean clone"
fi

# ------------------------------------------------------------------ 6. console
if [ "$MODE" = "--no-web" ]; then
    echo
    bold "Ready."
    echo "  Open the console any time with:   ./start.sh        or  python3 demo/server.py"
    echo "  The full walkthrough is in:       RUNNING.md"
    echo "  Where everything lives:           CONTRIBUTING.md"
    exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo
    bold "Built and tested. The console needs python3, which is not installed."
    echo "  The command line works now:  ./build/igaos example_milp.mps -v"
    exit 0
fi

echo
bold "Opening Solver Studio"
echo "  A browser tab will open at http://127.0.0.1:8420"
echo "  Pick a model on the left, press Solve, and watch the flow light up."
echo "  Ctrl-C here stops it."
echo
exec python3 demo/server.py
