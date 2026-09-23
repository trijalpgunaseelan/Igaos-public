// pdhg_kernels.cu : the GPU side of the first-order path.
//
// ===========================================================================
//  BUILD AND VERIFICATION STATUS -- READ THIS FIRST
// ===========================================================================
//  COMPILED: yes.  EXECUTED: no.  Those are different claims and the difference
//  is the whole point of this block.
//
//  The kernels below compile to PTX with NVRTC 12.9 and assemble to SASS with
//  ptxas for sm_75, sm_80 and sm_90, with no errors and no warnings.  Neither
//  step needs a GPU, so it runs in CI:
//
//      python3 tools/cuda_compile_check.py cuda/pdhg_kernels.cu
//
//  Doing that for the first time found two real defects that had been sitting
//  here unseen: a missing <utility> include for std::swap, and host code mixed
//  into a translation unit nothing had ever separated.  Which is the argument
//  for compiling code you cannot yet run.
//
//  What is still NOT verified, and cannot be here: what these kernels COMPUTE.
//  Register allocation succeeding says nothing about whether spmvCsr sums the
//  right products.  There is no test run, no benchmark number, and no
//  correctness claim for the device path -- only the CPU implementation in
//  src/pdhg.cpp, which is tested, and which this file mirrors line for line.
//
//  The host launcher at the bottom is not covered either: NVRTC compiles device
//  code only and stops before it.
//
//  It is excluded from the default build.  Enable it with -DIGAOS_ENABLE_CUDA=ON
//  once a real device is available, and expect to spend time on it: code that
//  has never run is code that has never run.
//
//  Verify in this order when a GPU is available:
//    1. spmvCsr against the CPU SparseMatrix::multiply on random matrices
//    2. one full PDHG iteration against src/pdhg.cpp, bit-for-bit in double
//    3. whole solves against the CPU path on the benchmark suite
//    4. only then measure speedup, against the CPU path on the same instances
//
// ===========================================================================
//  WHY THIS IS THE KERNEL THAT BELONGS ON A GPU
// ===========================================================================
//  A PDHG iteration is: one A'y, one Ax, and a handful of elementwise vector
//  operations.  There is no factorization, no elimination tree, no fill, and no
//  sequential dependency chain -- every row of the SpMV is independent, and the
//  whole iteration is memory-bandwidth bound.  That is the workload GPUs are
//  built for.
//
//  Sparse LU and sparse Cholesky are the opposite: their cost is dominated by a
//  long, irregular dependency chain through the elimination tree, with little
//  work at each node.  Porting *those* to a GPU is the thing that does not pay,
//  and it is why this solver's GPU story is the first-order path specifically
//  and not "the solver, on a GPU".
//
//  Two matrices are kept on the device, A in CSR for Ax and A' in CSR for A'y.
//  Storing the transpose explicitly costs one extra copy of the nonzeros and
//  buys coalesced reads in both directions; the alternative, an atomic-scatter
//  CSC product, serializes on contended columns.
// ===========================================================================

// NVRTC compiles device code without the host standard library, so the two
// fixed-width types this file needs are declared directly when it is the
// compiler.  tools/cuda_compile_check.py drives exactly that path, which is how
// this file gets compiled at all on a machine with no GPU in it.
#if defined(IGAOS_CUDA_EMULATE)
// Third build of this file, and the only one that RUNS it: cuda/emulate.hpp
// defines the CUDA built-ins below in ordinary C++ so tools/cuda_emulate.cpp
// can execute these kernel bodies on a CPU and check what they compute against
// the solver.  See that header for what the emulation does and does not cover.
#include "emulate.hpp"
#include <cstdint>
#elif defined(__CUDACC_RTC__)
using int32_t = int;
using int64_t = long long;
#else
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <utility>        // std::swap in launchIteration -- this was missing, and
                          // the file had never been compiled, so nobody knew
#endif

namespace igaos {
namespace cuda {

using Real = double;
using Int  = int32_t;

// The host side represents an infinite bound as this sentinel rather than as a
// floating-point infinity (igaos/common.hpp: kInf = 1e30, and isInf(v) is
// v >= kInf).  The kernels have to use the same convention or a free row is a
// bounded one on the device.
__device__ const Real kInfSentinel = 1.0e30;

// ---------------------------------------------------------------------------
// One warp per row.  Rows in LP constraint matrices are short (single digits to
// low hundreds of nonzeros), so a warp-per-row split keeps the reads coalesced
// without the launch overhead of a block per row.  For matrices with a heavy
// row-length tail, split by nonzero count instead -- a merge-based SpMV -- but
// measure before adding that complexity.
// ---------------------------------------------------------------------------
__global__ void spmvCsr(Int nrow,
                        const Int* __restrict__ rowPtr,
                        const Int* __restrict__ colIdx,
                        const Real* __restrict__ val,
                        const Real* __restrict__ x,
                        Real* __restrict__ y)
{
    const int warpsPerBlock = blockDim.x / 32;
    const int warpInBlock   = threadIdx.x / 32;
    const int lane          = threadIdx.x % 32;
    Int row = blockIdx.x * warpsPerBlock + warpInBlock;
    if (row >= nrow) return;

    Int start = rowPtr[row], end = rowPtr[row + 1];
    Real sum = 0.0;
    for (Int p = start + lane; p < end; p += 32) sum += val[p] * x[colIdx[p]];

    // Warp reduction. __shfl_down_sync is exact in double; the summation order
    // differs from the CPU loop, so agreement between the two paths is to
    // rounding, not bitwise -- check with a tolerance, never with ==.
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) y[row] = sum;
}

// ---------------------------------------------------------------------------
// x+ = proj_[l,u]( x - tau * (c + A'y) )
//
// The clamp tests the infinity sentinel the way src/pdhg.cpp's clampTo does,
// rather than calling fmin/fmax against +/-1e30 directly.  It looks like the
// same thing and is not: a value larger in magnitude than the sentinel is left
// alone by the CPU and pulled to +/-1e30 by the naive clamp, so the two paths
// would disagree exactly where the iterate is worst behaved.  Both branches
// here are predicated, not divergent -- the cost is a select, not a warp split.
// ---------------------------------------------------------------------------
__global__ void primalStep(Int n, Real tau,
                           const Real* __restrict__ c,
                           const Real* __restrict__ aty,
                           const Real* __restrict__ lower,
                           const Real* __restrict__ upper,
                           const Real* __restrict__ x,
                           Real* __restrict__ xNew,
                           Real* __restrict__ extrapolated)
{
    Int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    Real xj = x[j];
    Real v  = xj - tau * (c[j] + aty[j]);
    const Real lo = lower[j], up = upper[j];
    if (lo > -kInfSentinel && v < lo) v = lo;
    if (up <  kInfSentinel && v > up) v = up;
    xNew[j] = v;
    // 2x+ - x, fused here so the extrapolated vector never needs its own pass
    // over memory -- on a bandwidth-bound kernel that is a real saving.
    extrapolated[j] = 2.0 * v - xj;
}

// ---------------------------------------------------------------------------
// y+ = prox_{sigma * sigma_C}( y + sigma * A xbar )
//    = v - sigma * proj_C(v / sigma),   v = y + sigma * A xbar
// Moreau's identity, exactly as in the CPU implementation -- including the
// exact-zero branch, which is the part that was missing.
//
// DEFECT 23.  This kernel used to evaluate `v - sigma * (v / sigma)`
// unconditionally.  When v/sigma lands strictly inside the row box the
// projection is the identity and the true answer is zero; subtracting two
// nearly equal numbers instead leaves noise of order 1e-16*|v| whose SIGN is
// not reproducible.  A dual component of the wrong sign on a row with no bound
// on that side makes the support function infinite, and that is enough to blow
// up the entire dual measure -- which is exactly what happened on the CPU path
// once, and is why the comment in src/pdhg.cpp exists.  The kernel claimed to
// mirror the CPU implementation and did not.
//
// It was found by tools/cuda_emulate.cpp, which executes these kernel bodies on
// a CPU and compares them against src/pdhg.cpp.  No GPU was involved in finding
// it, and none was needed: "we have no device" had been quietly excusing never
// checking what these kernels COMPUTE.
// ---------------------------------------------------------------------------
__global__ void dualStep(Int m, Real sigma,
                         const Real* __restrict__ axbar,
                         const Real* __restrict__ rowLower,
                         const Real* __restrict__ rowUpper,
                         const Real* __restrict__ y,
                         Real* __restrict__ yNew)
{
    Int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    Real v = y[i] + sigma * axbar[i];
    const Real t = v / sigma;
    const Real lo = rowLower[i], up = rowUpper[i];
    Real p = t;
    if (lo > -kInfSentinel && p < lo) p = lo;
    if (up <  kInfSentinel && p > up) p = up;
    yNew[i] = (p == t) ? 0.0 : v - sigma * p;
}

// ---------------------------------------------------------------------------
// Running sums for the restart average.  Kept on the device so the averages
// never cross the PCIe bus; only the convergence check does, and that runs
// every few dozen iterations rather than every iteration.
// ---------------------------------------------------------------------------
__global__ void accumulate(Int k, const Real* __restrict__ v, Real* __restrict__ sum) {
    Int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) sum[i] += v[i];
}

__global__ void axpby(Int k, Real a, const Real* __restrict__ x,
                      Real b, const Real* __restrict__ y, Real* __restrict__ out) {
    Int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) out[i] = a * x[i] + b * y[i];
}

// ---------------------------------------------------------------------------
// Everything below is HOST code: device memory handles and the launch sequence.
// NVRTC compiles device code only and does not accept the <<<>>> launch syntax,
// so tools/cuda_compile_check.py -- which is how this file gets compiled on a
// machine with no GPU -- sees the kernels above and stops here.  That means the
// kernels are compiler-verified and this launcher is not; it is fifteen lines of
// glue, but fifteen unverified lines, and worth saying so.
// ---------------------------------------------------------------------------
#if !defined(__CUDACC_RTC__) && !defined(IGAOS_CUDA_EMULATE)

// Device-side problem data.  Allocated once per solve: the matrices never
// change during PDHG, which is precisely why the method suits a device with
// slow host transfers.
struct DeviceProblem {
    Int n = 0, m = 0;
    Int  *aRowPtr = nullptr, *aColIdx = nullptr;      // A in CSR (m x n)
    Real *aVal = nullptr;
    Int  *atRowPtr = nullptr, *atColIdx = nullptr;    // A' in CSR (n x m)
    Real *atVal = nullptr;
    Real *c = nullptr, *lower = nullptr, *upper = nullptr;
    Real *rowLower = nullptr, *rowUpper = nullptr;
    Real *x = nullptr, *xNew = nullptr, *extrapolated = nullptr;
    Real *y = nullptr, *yNew = nullptr;
    Real *aty = nullptr, *axbar = nullptr;
    Real *xSum = nullptr, *ySum = nullptr;
};

inline void launchIteration(DeviceProblem& d, Real tau, Real sigma, cudaStream_t stream) {
    const int block = 256;
    const int warpsPerBlock = block / 32;

    // A'y  -- n rows of the transpose
    spmvCsr<<<(d.n + warpsPerBlock - 1) / warpsPerBlock, block, 0, stream>>>(
        d.n, d.atRowPtr, d.atColIdx, d.atVal, d.y, d.aty);

    primalStep<<<(d.n + block - 1) / block, block, 0, stream>>>(
        d.n, tau, d.c, d.aty, d.lower, d.upper, d.x, d.xNew, d.extrapolated);

    // A xbar -- m rows of A
    spmvCsr<<<(d.m + warpsPerBlock - 1) / warpsPerBlock, block, 0, stream>>>(
        d.m, d.aRowPtr, d.aColIdx, d.aVal, d.extrapolated, d.axbar);

    dualStep<<<(d.m + block - 1) / block, block, 0, stream>>>(
        d.m, sigma, d.axbar, d.rowLower, d.rowUpper, d.y, d.yNew);

    accumulate<<<(d.n + block - 1) / block, block, 0, stream>>>(d.n, d.xNew, d.xSum);
    accumulate<<<(d.m + block - 1) / block, block, 0, stream>>>(d.m, d.yNew, d.ySum);

    std::swap(d.x, d.xNew);
    std::swap(d.y, d.yNew);
}

#endif  // !__CUDACC_RTC__ && !IGAOS_CUDA_EMULATE

} // namespace cuda
} // namespace igaos
