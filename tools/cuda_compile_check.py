#!/usr/bin/env python3
"""
Compile the CUDA kernels, all the way to machine code, without a GPU.

Compiling and running are two different things, and the difference matters for
an honest claim. A GPU is needed to *run* a kernel. It is not needed to compile
one: NVRTC turns CUDA C++ into PTX, and ptxas turns PTX into the SASS the
hardware actually executes, register allocation and all. Both run happily on a
machine with no NVIDIA device in it.

So this harness closes the gap between "we wrote CUDA" and "we compiled CUDA".
It does not close the gap to "we ran CUDA", and nothing here should be read as
if it did.

    pip install nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
    python3 tools/cuda_compile_check.py cuda/pdhg_kernels.cu

Exit code 0 means every target architecture compiled to PTX and assembled to a
cubin with no errors and no warnings.
"""
import argparse, ctypes, os, subprocess, sys, glob, tempfile

DEFAULT_ARCHS = ["75", "80", "90"]          # Turing, Ampere, Hopper


def find(pattern, *roots):
    for r in roots:
        hits = sorted(glob.glob(os.path.join(r, pattern)))
        if hits:
            return hits[-1]
    return None


def site_nvidia():
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        p = os.path.join(sp, "nvidia")
        if os.path.isdir(p):
            return p
    return None


def load_nvrtc():
    nv = site_nvidia()
    cands = []
    if nv:
        cands.append(os.path.join(nv, "cuda_nvrtc", "lib"))
    cands += ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]
    lib = find("libnvrtc.so*", *cands)
    if lib is None:
        sys.exit("libnvrtc not found — pip install nvidia-cuda-nvrtc-cu12")
    return ctypes.CDLL(lib), lib


def cuda_include_dirs():
    nv = site_nvidia()
    dirs = []
    if nv:
        for sub in ("cuda_runtime", "cuda_cccl", "cuda_nvcc"):
            d = os.path.join(nv, sub, "include")
            if os.path.isdir(d):
                dirs.append(d)
            d2 = os.path.join(nv, sub, "include", "cuda", "std")
            if os.path.isdir(d2):
                dirs.append(os.path.join(nv, sub, "include"))
    for d in ("/usr/local/cuda/include",):
        if os.path.isdir(d):
            dirs.append(d)
    return sorted(set(dirs))


def compile_to_ptx(nvrtc, src, name, arch, extra_includes):
    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(),
                                  0, None, None)
    if rc != 0:
        sys.exit(f"nvrtcCreateProgram failed: {rc}")

    opts = [f"--gpu-architecture=compute_{arch}".encode(),
            b"--std=c++17",
            b"-default-device",
            b"--restrict"]
    for d in extra_includes:
        opts.append(("-I" + d).encode())
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = nvrtc.nvrtcCompileProgram(prog, len(opts), arr)

    size = ctypes.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(prog, log)
    log = log.value.decode(errors="replace").strip()

    if rc != 0:
        return None, log

    nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size))
    ptx = ctypes.create_string_buffer(size.value)
    nvrtc.nvrtcGetPTX(prog, ptx)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return ptx.value.decode(errors="replace"), log


def assemble(ptx, arch, verbose=True):
    """PTX -> SASS with ptxas. This is the real device back end."""
    nv = site_nvidia()
    ptxas = None
    if nv:
        ptxas = find("ptxas", os.path.join(nv, "cuda_nvcc", "bin"))
    ptxas = ptxas or find("ptxas", "/usr/local/cuda/bin") or "ptxas"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "k.ptx")
        c = os.path.join(td, "k.cubin")
        open(p, "w").write(ptx)
        argv = [ptxas, "-arch", f"sm_{arch}", "-O3", p, "-o", c]
        if verbose:
            argv.append("-v")
        r = subprocess.run(argv, capture_output=True, text=True)
        size = os.path.getsize(c) if os.path.exists(c) else 0
        return r.returncode, (r.stderr or "").strip(), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="cuda/pdhg_kernels.cu")
    ap.add_argument("--arch", action="append", default=None,
                    help="compute capability, e.g. 80 (repeatable)")
    a = ap.parse_args()
    archs = a.arch or DEFAULT_ARCHS

    src = open(a.source).read()
    nvrtc, libpath = load_nvrtc()
    major, minor = ctypes.c_int(), ctypes.c_int()
    nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    inc = cuda_include_dirs()

    print(f"source   {a.source}  ({len(src.splitlines())} lines)")
    print(f"NVRTC    {major.value}.{minor.value}   {libpath}")
    print(f"includes {', '.join(inc) or '(none)'}")
    print()

    failed = False
    for arch in archs:
        ptx, log = compile_to_ptx(nvrtc, src, os.path.basename(a.source), arch, inc)
        if ptx is None:
            failed = True
            print(f"sm_{arch}: COMPILE FAILED")
            print("\n".join("    " + l for l in log.splitlines()[:40]))
            continue
        rc, perr, size = assemble(ptx, arch)
        kernels = ptx.count(".visible .entry")
        regs = [l.strip() for l in perr.splitlines() if "registers" in l]
        status = "ok" if rc == 0 else "PTXAS FAILED"
        print(f"sm_{arch}: {status}   {kernels} kernels   "
              f"{len(ptx.splitlines()):>5} PTX lines   {size:>6} B cubin")
        for r in regs:
            print(f"          {r}")
        if log:
            print("\n".join("    warning: " + l for l in log.splitlines()[:10]))
            failed = True
        if rc != 0:
            failed = True
            print("\n".join("    " + l for l in perr.splitlines()[:20]))

    print()
    if failed:
        print("RESULT: at least one target did not compile cleanly.")
        return 1
    print(f"RESULT: compiles clean to SASS for sm_{', sm_'.join(archs)}.")
    print("        This proves the kernels are valid CUDA and that the device")
    print("        back end can allocate registers for them. It proves nothing")
    print("        about what they compute.")
    print()
    print("        What they compute is checked separately, and also without a")
    print("        GPU: tools/cuda_emulate.cpp executes these same kernel")
    print("        bodies on the CPU and compares them against src/pdhg.cpp")
    print("        and against the simplex. It runs under ctest as")
    print("        `cuda_kernels`. What still needs a real device: any timing")
    print("        number, the launch configuration, memory coalescing, and")
    print("        the host launcher at the bottom of the .cu.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
