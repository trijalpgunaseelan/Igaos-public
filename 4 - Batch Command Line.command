#!/bin/bash
# ===========================================================================
#  IGAOS - batch command line
#
#  Double-click this file in Finder. The first time, macOS may say it is from
#  an unidentified developer: right-click it once and choose Open. After that
#  a double-click is enough.
# ===========================================================================
cd "$(dirname "$0")" || exit 1
printf '\033[1m  IGAOS - batch command line  \033[0m\n\n'
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

echo "The batch interface: flags in, answers out. This runs the four"
echo "demonstrations worth showing, then leaves you a shell prompt."
echo
build_if_needed
echo
printf '\033[1m=== The model library ===\033[0m\n'
./build/igaos --models
printf '\033[1m=== Cut separation on, then off - the same answer, 1 node against 51 ===\033[0m\n'
./build/igaos -m uc_m --compare cuts --no-live
printf '\033[1m=== Four continuous methods, one model ===\033[0m\n'
./build/igaos -m blend_s --compare paths --no-live
printf '\033[1m=== The solve pipeline as a chart ===\033[0m\n'
./build/igaos -m uc_m --flow --no-live -q
printf '\033[1m=== A certificate, checked by a separate program in exact arithmetic ===\033[0m\n'
./build/igaos -m blend_s -q --no-live --certificate /tmp/igaos_demo.cert >/dev/null 2>&1
python3 tools/verify_certificate.py demo/models/blend_s.mps /tmp/igaos_demo.cert 2>&1 | tail -12
echo
printf '\033[1m  Every command above is in RUNNING.md. Try your own:\033[0m\n'
echo "     ./build/igaos --models"
echo "     ./build/igaos -m uc_l --chart"
echo "     ./build/igaos yourmodel.mps --algorithm interior --solution out.sol"
echo "     ./build/igaos --help"
echo
echo "Opening a shell in the solver directory. Type exit to close it."
exec "${SHELL:-/bin/bash}" -l
