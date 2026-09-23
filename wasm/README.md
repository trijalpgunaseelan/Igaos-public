# IGAOS on WebAssembly

The solver compiled for `wasm32-wasi` and run in a browser tab. Same C++17
sources as the native CLI — `src/` is untouched by this directory except for two
changes described below, which are in the repo's own files.

    WASI_SDK=/opt/wasi-sdk ./wasm/build_wasm.sh igaos.wasm

## What had to change, and why

**1. `<thread>`, `<mutex>`, `<condition_variable>` shims.** WASI's libc++ is
built without thread support, so `#include <thread>` is a hard error. The shims
here provide the same API single-threaded: `std::thread` runs its callable
inline on construction, `join()` is a no-op, `hardware_concurrency()` returns 1.
The branch-and-cut tree therefore runs **serially and deterministically** — the
same search a native build performs with `--threads 1`.

**2. Exceptions.** WASI's libc++abi has no `__cxa_throw`, so the build is
`-fno-exceptions`. Two files needed real edits:

- `src/mps.cpp` — `std::stod` replaced with `strtod`, which reports the same
  two failures (not-a-number, out-of-range) through `errno` instead of throwing.
  The poisoned-value detection is unchanged.
- `src/model.cpp` — `validate()`'s eight `throw std::runtime_error` calls print
  the identical message to stderr and `exit(3)`. Same message, same stop.

`src/capi.cpp` is excluded from this build: the C ABI is exception-based and the
page calls the solver directly, not through it.

**3. `__cxa_thread_atexit`** (`stub.c`) — `thread_local` destructor
registration, absent from the threadless libc. The module runs one solve and
exits, so nothing outlives the call.

## What is NOT different

The MPS reader, presolve, scaling, revised simplex, interior point, crossover,
cut separation and the branch-and-bound search are the project's own code,
unmodified. `blend_m` returns `-2135858.97349` here, on native macOS and on
native Linux — the same value to every digit printed.

CUDA is absent from this build. It has never run on a device and no GPU claim
is made anywhere on the page.

## Front end

`wasm_main.cpp` replaces `apps/igaos_cli.cpp` for this target (the CLI drives a
terminal through `termios`, which WebAssembly has no equivalent for). It reads
the model from the WASI filesystem, solves, and writes one JSON object to
stdout. The page supplies a minimal WASI preview1 implementation in JavaScript
and runs the module inside a Web Worker so a long MILP never freezes the tab.
