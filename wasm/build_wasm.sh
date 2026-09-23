#!/usr/bin/env bash
# Build IGAOS for WebAssembly (wasm32-wasi). Produces igaos.wasm.
#
# Reproduces the binary behind Solver Studio. Same solver source as the native
# CLI; only the front end and the threading differ.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"     # repo root
SDK="${WASI_SDK:-/opt/wasi-sdk}"             # https://github.com/WebAssembly/wasi-sdk/releases
SYSROOT="$SDK/share/wasi-sysroot"
SHIM="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-igaos.wasm}"
OBJ="$(mktemp -d)"

# The shim directory must precede the sysroot so its <thread>/<mutex>/
# <condition_variable> win: WASI's libc++ is built without thread support, and
# these make std::thread run its callable inline. The tree becomes serial and
# deterministic -- the same search `--threads 1` gives natively.
FLAGS=(--target=wasm32-wasi --sysroot="$SYSROOT" -std=c++17 -O2 -DNDEBUG
       -fno-exceptions -I"$SHIM/include" -I"$ROOT/include")

mkdir -p "$SHIM/include"
cp "$SHIM/shim_thread.hpp"             "$SHIM/include/thread"
cp "$SHIM/shim_mutex.hpp"              "$SHIM/include/mutex"
cp "$SHIM/shim_condition_variable.hpp" "$SHIM/include/condition_variable"

# capi.cpp is skipped: the C ABI is exception-based and the page does not use it.
for f in "$ROOT"/src/*.cpp; do
  b="$(basename "$f" .cpp)"; [ "$b" = capi ] && continue
  "$SDK/bin/clang++" "${FLAGS[@]}" -c "$f" -o "$OBJ/$b.o"
done
"$SDK/bin/clang++" "${FLAGS[@]}" -c "$SHIM/wasm_main.cpp" -o "$OBJ/main.o"
"$SDK/bin/clang"   --target=wasm32-wasi --sysroot="$SYSROOT" -O2 -c "$SHIM/stub.c" -o "$OBJ/stub.o"
"$SDK/bin/clang++" --target=wasm32-wasi --sysroot="$SYSROOT" -O2 -fno-exceptions "$OBJ"/*.o -o "$OUT"
echo "built $OUT ($(du -h "$OUT" | cut -f1))"
