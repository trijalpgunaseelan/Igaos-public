#!/bin/bash
# ===========================================================================
#  IGAOS - build and test the engine
#
#  Double-click this file in Finder. The first time, macOS may say it is from
#  an unidentified developer: right-click it once and choose Open. After that
#  a double-click is enough.
# ===========================================================================
cd "$(dirname "$0")" || exit 1
printf '\033[1m  IGAOS - build and test the engine  \033[0m\n\n'
# ---------------------------------------------------------------------------
# Shared preamble for the IGAOS double-click launchers.
# Sourced, never run on its own.
# ---------------------------------------------------------------------------
hold() { printf '\n\nPress Return to close this window. '; read -r _; exit "${1:-0}"; }

# Homebrew and the Xcode tools are not always on a GUI-launched PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.homebrew/bin:$HOME/.local/bin:$PATH"

here="$(cd "$(dirname "$0")" && pwd)"
find_tree() {
    # "$here" first: in the public repository the solver IS the repository root,
    # so the launchers sit beside CMakeLists.txt rather than one level above it.
    for c in "$here" "$here/igaos" "$here/IGAOS-SIH26119/igaos" "$here/../igaos" \
             "$here/../IGAOS-SIH26119/igaos"; do
        [ -f "$c/CMakeLists.txt" ] && { cd "$c" && pwd && return 0; }
    done
    return 1
}
TREE="$(find_tree)" || {
    echo "Could not find the solver."
    echo "Keep this file in the same folder as the igaos/ directory."
    hold 1
}
cd "$TREE" || exit 1

need_cxx() {
    command -v c++ >/dev/null 2>&1 || {
        echo "No C++ compiler found. Install Apple's command line tools:"
        echo
        echo "    xcode-select --install"
        echo
        echo "then double-click this again."
        hold 1
    }
    command -v cmake >/dev/null 2>&1 || {
        echo "cmake not found.  Install it with:"
        echo
        echo "    brew install cmake"
        echo
        hold 1
    }
}

# CMake bakes absolute paths into build/CMakeCache.txt. Move or unzip this tree
# somewhere new and configure fails with "does not match the source used to
# generate cache". Nobody should have to read a troubleshooting note for that.
drop_stale_build() {
    [ -f build/CMakeCache.txt ] || return 0
    _src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' build/CMakeCache.txt | head -1)
    [ "$_src" = "$(pwd -P)" ] || { echo "  build/ was configured elsewhere - starting clean"; rm -rf build; }
}

# A binary that EXISTS is not the same as a binary that is CURRENT. Without
# this check, editing a source file and double-clicking a launcher runs the old
# solver and reports success -- the worst kind of failure, because it is silent.
sources_newer_than_binary() {
    [ -x build/igaos ] || return 0
    _hit=$(find CMakeLists.txt src include apps tests cuda \
                \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \
                   -o -name '*.cu' -o -name '*.cuh' -o -name 'CMakeLists.txt' \) \
                -newer build/igaos -print -quit 2>/dev/null)
    [ -n "$_hit" ]
}

build_if_needed() {
    need_cxx
    drop_stale_build
    if [ -x build/igaos ] && ./build/igaos --version >/dev/null 2>&1; then
        if sources_newer_than_binary; then
            echo "  sources have changed since the last build - rebuilding"
        else
            return 0
        fi
    fi
    echo "Building the solver (about a minute the first time)..."
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /tmp/igaos_cmake.log 2>&1 \
        || { echo "cmake configure failed - see /tmp/igaos_cmake.log"; tail -15 /tmp/igaos_cmake.log; hold 1; }
    cmake --build build -j > /tmp/igaos_build.log 2>&1 \
        || { echo "build failed - see /tmp/igaos_build.log"; tail -25 /tmp/igaos_build.log; hold 1; }
    echo "  built: $(./build/igaos --version)"
}

echo "This builds the solver from source and runs every test suite."
echo "Nothing is downloaded and nothing is installed system-wide."
echo
build_if_needed
echo
echo "Running the test suites..."
( cd build && ctest --output-on-failure ) || { echo "TESTS FAILED"; hold 1; }
./build/igaos_tests     2>&1 | tail -1
./build/igaos_capi_test 2>&1 | tail -1
echo
echo "Solving an example to prove it runs..."
[ -f example_lp.mps ] || cp demo/models/blend_s.mps example_lp.mps 2>/dev/null
./build/igaos example_lp.mps --no-live 2>&1 | tail -8
echo
printf '\033[1m  The engine is built and tested. The three front ends are:\033[0m\n'
echo "     2 - Interactive Console.command    the solver as a terminal application"
echo "     3 - Browser Console.command        the same thing in a browser"
echo "     4 - Batch Command Line.command     flags in, answers out"
echo "     5 - Backend Service.command        install it, or serve it over HTTP"
hold 0
