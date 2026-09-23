#!/usr/bin/env bash
# Build IGAOS and install it as a command you can run from anywhere.
#
#   ./install.sh                 # into ~/.local  (no sudo, add ~/.local/bin to PATH)
#   ./install.sh /usr/local      # system-wide    (needs sudo)
#   ./install.sh --uninstall     # remove what was installed
#
# Installs: bin/igaos, lib/libigaos.so (.dylib), lib/libigaos_core.a,
# include/igaos/*.hpp and a CMake package so other projects can
# `find_package(igaos)`.

set -eu
cd "$(dirname "$0")"

PREFIX="${1:-$HOME/.local}"
MANIFEST="build/install_manifest.txt"

if [ "${1:-}" = "--uninstall" ]; then
    [ -f "$MANIFEST" ] || { echo "no install manifest at $MANIFEST — nothing to undo"; exit 1; }
    while IFS= read -r f; do [ -e "$f" ] && rm -f "$f" && echo "removed $f"; done < "$MANIFEST"
    exit 0
fi

command -v cmake >/dev/null 2>&1 || {
    echo "cmake not found."
    echo "  macOS: brew install cmake     Debian/Ubuntu: sudo apt install cmake"
    exit 1
}

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
if stale_build; then
    echo "  build/ was configured for a different path -- starting it clean"
    rm -rf build
fi

echo "building (Release) …"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" >/dev/null
cmake --build build -j >/dev/null

echo "running the test suites …"
( cd build && ctest --output-on-failure >/dev/null ) || { echo "tests FAILED — not installing"; exit 1; }

echo "installing into $PREFIX …"
cmake --install build >/dev/null

cat <<EOF

Installed:
  $PREFIX/bin/igaos                 command-line solver
  $PREFIX/lib/libigaos*             shared and static libraries (stable C ABI)
  $PREFIX/include/igaos/            headers
  $PREFIX/lib/cmake/igaos/          find_package(igaos) support

Check it:
  $PREFIX/bin/igaos --version
EOF

case ":$PATH:" in
    *":$PREFIX/bin:"*) ;;
    *) echo
       echo "$PREFIX/bin is not on your PATH. Add it:"
       echo "  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.zshrc && exec zsh" ;;
esac

echo
echo "Python bindings (optional, no compiler needed):"
echo "  pip install ./python"
echo "  export IGAOS_LIBRARY=$PREFIX/lib/libigaos.so   # .dylib on macOS"
