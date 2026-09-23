// cuda_emulate.cpp : execute the CUDA kernels on a CPU and check what they
// compute against the solver.
//
//     cd build && ctest -R cuda_kernels --output-on-failure
//
//  or, standalone:
//     g++ -std=c++17 -O2 -Iinclude -Icuda -fopenmp tools/cuda_emulate.cpp
//         build/libigaos_core.a -o cuda_emulate && ./cuda_emulate
//
// ===========================================================================
//  WHY THIS EXISTS
// ===========================================================================
//  The problem statement asks for GPU acceleration.  This project was built
//  without an NVIDIA device, so it can make no speedup claim at all and says so
//  everywhere.  But "we have no GPU" was also being used, silently, to excuse
//  never checking whether the kernels compute the right numbers -- and that
//  part needs no GPU.  Compiling proves the register allocator was happy.  It
//  proves nothing about whether spmvCsr sums the right products.
//
//  So the kernel bodies are included here, verbatim, through cuda/emulate.hpp,
//  and executed.  Their results are compared against code that shares nothing
//  with them: the sparse matrix product in src/sparse.cpp, the CPU first-order
//  method in src/pdhg.cpp, and -- for the end-to-end test -- the simplex, which
//  is a completely different algorithm.
//
//  The first run of this program found defect 23.  See the dual-step test.
//
//  What this does NOT establish: any performance number, that the launch
//  configuration is valid on real hardware, that memory access is coalesced,
//  or that the host launcher works.  Those need a device.  This establishes
//  that the arithmetic is right, which is the prerequisite for all of them.
// ===========================================================================

#define IGAOS_CUDA_EMULATE 1
#include "pdhg_kernels.cu"          // the kernels themselves, executed below

#include "igaos/model.hpp"
#include "igaos/solver.hpp"
#include "igaos/sparse.hpp"
#include "igaos/pdhg.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

using namespace igaos;
using igaos::cudaemu::launchKernel;
using igaos::cudaemu::launchWarpKernel;

static int failures = 0;
static void check(bool ok, const char* what, double detail = 0.0) {
    if (ok) std::printf("  [ ok ] %s\n", what);
    else { std::printf("  [FAIL] %s  (%.6e)\n", what, detail); ++failures; }
}

// --------------------------------------------------------------------------
// The launch shapes below are the ones cuda/pdhg_kernels.cu's own
// launchIteration uses: block 256, one warp per row for the SpMV, one thread
// per element for everything else.  Emulating a different shape would test a
// program nobody is going to run.
// --------------------------------------------------------------------------
static const unsigned kBlock = 256;

static void spmv(Int nrow, const std::vector<Int>& rowPtr, const std::vector<Int>& colIdx,
                 const std::vector<Real>& val, const std::vector<Real>& x,
                 std::vector<Real>& y) {
    const unsigned warpsPerBlock = kBlock / 32;
    const unsigned grid = ((unsigned)nrow + warpsPerBlock - 1) / warpsPerBlock;
    launchWarpKernel(grid, kBlock, [&] {
        igaos::cuda::spmvCsr(nrow, rowPtr.data(), colIdx.data(), val.data(),
                             x.data(), y.data());
    });
}

// A CSR view of a SparseMatrix, built through the transpose so the conversion
// itself is the library's, not this file's.
struct Csr {
    std::vector<Int> rowPtr, colIdx;
    std::vector<Real> val;
    Int nrow = 0, ncol = 0;
};
static Csr toCsr(const SparseMatrix& A) {
    Csr c;
    c.nrow = A.nrow; c.ncol = A.ncol;
    SparseMatrix At = A.transpose();          // CSC of A' == CSR of A
    c.rowPtr.assign(At.colPtr.begin(), At.colPtr.end());
    c.colIdx.assign(At.rowIdx.begin(), At.rowIdx.end());
    c.val.assign(At.val.begin(), At.val.end());
    return c;
}

// ===========================================================================
static void testSpmv() {
    std::printf("spmvCsr against the CPU sparse product\n");
    std::mt19937_64 rng(20260828);
    Real worst = 0.0;
    int cases = 0;

    // Shapes chosen for what they break: rows shorter than a warp, rows longer
    // than a warp (so the strided loop runs more than once), empty rows (so the
    // reduction sums nothing), and a row count that is not a multiple of the
    // warps per block (so the tail block has idle warps).
    const int shapes[][3] = {{1, 1, 1}, {7, 5, 3}, {33, 40, 4}, {64, 64, 12},
                             {97, 61, 40}, {150, 200, 3}, {257, 129, 80}};
    for (const auto& sh : shapes) {
        const Int m = sh[0], n = sh[1];
        const int maxPerRow = sh[2];
        SparseMatrix A; A.nrow = m; A.ncol = n;
        std::vector<std::vector<std::pair<Int, Real>>> byCol((size_t)n);
        std::uniform_real_distribution<double> u(-3.0, 3.0);
        for (Int i = 0; i < m; ++i) {
            int k = (i % 5 == 0) ? 0 : (int)(rng() % (uint64_t)maxPerRow) + 1;  // some empty rows
            std::vector<Int> cols;
            for (int t = 0; t < k; ++t) cols.push_back((Int)(rng() % (uint64_t)n));
            std::sort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            for (Int j : cols) byCol[(size_t)j].push_back({i, u(rng)});
        }
        A.colPtr.assign((size_t)n + 1, 0);
        for (Int j = 0; j < n; ++j) A.colPtr[(size_t)j + 1] = A.colPtr[(size_t)j]
                                                            + (Int)byCol[(size_t)j].size();
        A.rowIdx.resize((size_t)A.colPtr[(size_t)n]);
        A.val.resize((size_t)A.colPtr[(size_t)n]);
        for (Int j = 0; j < n; ++j) {
            Int p = A.colPtr[(size_t)j];
            for (auto& e : byCol[(size_t)j]) { A.rowIdx[(size_t)p] = e.first;
                                              A.val[(size_t)p] = e.second; ++p; }
        }

        std::vector<Real> x((size_t)n), gpu((size_t)m, 0.0), cpu((size_t)m, 0.0);
        for (Real& v : x) v = u(rng);
        Csr c = toCsr(A);
        spmv(m, c.rowPtr, c.colIdx, c.val, x, gpu);
        A.multiply(x, cpu);                                  // independent CPU code
        for (Int i = 0; i < m; ++i)
            worst = std::max(worst, std::fabs(gpu[(size_t)i] - cpu[(size_t)i])
                                    / (1.0 + std::fabs(cpu[(size_t)i])));
        ++cases;
    }
    // The warp reduction sums in a different order from the CPU loop, so the
    // agreement is to rounding, never bitwise.  Anything above this is a bug,
    // not an ordering difference.
    check(worst < 1e-12, "spmvCsr matches SparseMatrix::multiply on every shape", worst);
    std::printf("         %d shapes, worst relative difference %.3e\n", cases, worst);
}

// ===========================================================================
static void testPrimalStep() {
    std::printf("primalStep against the CPU primal update\n");
    const Int n = 500;
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> u(-5.0, 5.0);
    std::vector<Real> c((size_t)n), aty((size_t)n), lo((size_t)n), up((size_t)n),
                      x((size_t)n), xNew((size_t)n, 0.0), extra((size_t)n, 0.0);
    for (Int j = 0; j < n; ++j) {
        c[(size_t)j] = u(rng); aty[(size_t)j] = u(rng); x[(size_t)j] = u(rng);
        int kind = (int)(rng() % 4);
        lo[(size_t)j] = (kind == 0 || kind == 3) ? -kInf : -1.0;   // free and half-open
        up[(size_t)j] = (kind == 1 || kind == 3) ?  kInf :  1.0;   // bounds included
    }
    const Real tau = 0.37;
    launchKernel(((unsigned)n + kBlock - 1) / kBlock, kBlock, [&] {
        igaos::cuda::primalStep(n, tau, c.data(), aty.data(), lo.data(), up.data(),
                                x.data(), xNew.data(), extra.data());
    });

    Real worstX = 0, worstE = 0;
    for (Int j = 0; j < n; ++j) {
        // Exactly the expression in src/pdhg.cpp's primal step.
        Real v = x[(size_t)j] - tau * (c[(size_t)j] + aty[(size_t)j]);
        if (!isNegInf(lo[(size_t)j]) && v < lo[(size_t)j]) v = lo[(size_t)j];
        if (!isInf(up[(size_t)j])    && v > up[(size_t)j]) v = up[(size_t)j];
        worstX = std::max(worstX, std::fabs(xNew[(size_t)j] - v));
        worstE = std::max(worstE, std::fabs(extra[(size_t)j] - (2.0 * v - x[(size_t)j])));
    }
    check(worstX == 0.0, "primalStep projects onto the same box the CPU does", worstX);
    check(worstE == 0.0, "primalStep's fused extrapolation is 2x+ - x", worstE);
}

// ===========================================================================
static void testDualStep() {
    std::printf("dualStep against the CPU dual update\n");
    const Int m = 400;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u(-4.0, 4.0);
    std::vector<Real> axbar((size_t)m), rlo((size_t)m), rup((size_t)m),
                      y((size_t)m), yNew((size_t)m, 0.0);
    int freeRows = 0;
    for (Int i = 0; i < m; ++i) {
        axbar[(size_t)i] = u(rng); y[(size_t)i] = u(rng);
        int kind = (int)(rng() % 4);
        // kind 3 is a row with no bound on either side.  Rare in a real model
        // and the reason this test exists: see below.
        rlo[(size_t)i] = (kind == 0 || kind == 3) ? -kInf : -2.0;
        rup[(size_t)i] = (kind == 1 || kind == 3) ?  kInf :  2.0;
        if (kind == 3) ++freeRows;
    }
    const Real sigma = 0.21;
    launchKernel(((unsigned)m + kBlock - 1) / kBlock, kBlock, [&] {
        igaos::cuda::dualStep(m, sigma, axbar.data(), rlo.data(), rup.data(),
                              y.data(), yNew.data());
    });

    // The CPU version, from src/pdhg.cpp, INCLUDING the exact-zero branch:
    //
    //     Real t = vi / sigma;
    //     Real proj = clampTo(t, rowLower[i], rowUpper[i]);
    //     v[i] = (proj == t) ? 0.0 : vi - sigma * proj;
    //
    // That branch is not cosmetic.  When the projection is the identity the
    // exact answer is zero, and evaluating it as vi - sigma*(vi/sigma) instead
    // leaves rounding noise of order 1e-16*|vi| whose SIGN is not reproducible.
    // A dual component of the wrong sign on a row with no bound on that side
    // makes the support function infinite, which is how a single rounding bit
    // used to blow up the whole dual measure on the CPU path.  The comment in
    // src/pdhg.cpp records that; the kernel did not have the branch.
    Real worst = 0.0, worstFree = 0.0;
    for (Int i = 0; i < m; ++i) {
        Real vi = y[(size_t)i] + sigma * axbar[(size_t)i];
        Real t = vi / sigma;
        Real proj = t;
        if (!isNegInf(rlo[(size_t)i]) && proj < rlo[(size_t)i]) proj = rlo[(size_t)i];
        if (!isInf(rup[(size_t)i])    && proj > rup[(size_t)i]) proj = rup[(size_t)i];
        Real want = (proj == t) ? 0.0 : vi - sigma * proj;
        Real diff = std::fabs(yNew[(size_t)i] - want);
        worst = std::max(worst, diff);
        if (isNegInf(rlo[(size_t)i]) && isInf(rup[(size_t)i]))
            worstFree = std::max(worstFree, diff);
    }
    check(worst == 0.0, "dualStep matches the CPU Moreau step exactly", worst);
    std::printf("         %d of %d rows unbounded on both sides; worst difference "
                "there %.3e\n", freeRows, (int)m, worstFree);
}

// ===========================================================================
static void testAccumulateAxpby() {
    std::printf("accumulate and axpby\n");
    const Int k = 300;
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(-2.0, 2.0);
    std::vector<Real> v((size_t)k), sum((size_t)k), sum0, xv((size_t)k), yv((size_t)k),
                      out((size_t)k, 0.0);
    for (Int i = 0; i < k; ++i) { v[(size_t)i] = u(rng); sum[(size_t)i] = u(rng);
                                  xv[(size_t)i] = u(rng); yv[(size_t)i] = u(rng); }
    sum0 = sum;
    launchKernel(((unsigned)k + kBlock - 1) / kBlock, kBlock, [&] {
        igaos::cuda::accumulate(k, v.data(), sum.data());
    });
    Real w1 = 0;
    for (Int i = 0; i < k; ++i)
        w1 = std::max(w1, std::fabs(sum[(size_t)i] - (sum0[(size_t)i] + v[(size_t)i])));
    check(w1 == 0.0, "accumulate adds the vector into the running sum", w1);

    const Real a = 1.7, b = -0.4;
    launchKernel(((unsigned)k + kBlock - 1) / kBlock, kBlock, [&] {
        igaos::cuda::axpby(k, a, xv.data(), b, yv.data(), out.data());
    });
    Real w2 = 0;
    for (Int i = 0; i < k; ++i)
        w2 = std::max(w2, std::fabs(out[(size_t)i]
                                    - (a * xv[(size_t)i] + b * yv[(size_t)i])));
    check(w2 == 0.0, "axpby computes a*x + b*y", w2);
}

// ===========================================================================
// End to end: a PDHG loop built out of nothing but the emulated kernels, run
// against the simplex.  If any kernel's arithmetic were wrong the iteration
// would not be a valid primal-dual method and this would not converge to the
// LP optimum -- which is why this is the test that matters most.
// ===========================================================================
static void testWholeSolve() {
    std::printf("a PDHG loop assembled only from the kernels, against the simplex\n");

    std::mt19937_64 rng(1234);
    std::uniform_real_distribution<double> u(0.5, 4.0);
    int solved = 0;
    Real worst = 0.0;

    for (int inst = 0; inst < 4; ++inst) {
        const Int n = 12 + 6 * inst, m = 8 + 4 * inst;
        Model mo;
        for (Int j = 0; j < n; ++j) mo.addColumn(0.0, 6.0, u(rng));
        for (Int i = 0; i < m; ++i) {
            Int r = mo.addRow(u(rng) * 3.0, kInf);          // A x >= b, x in [0,6]
            for (Int j = 0; j < n; ++j)
                if ((int)(rng() % 3) == 0) mo.setElement(r, j, u(rng));
        }
        mo.finalize();

        Solver s; s.opt.log.level = 0;
        Solution ref = s.solve(mo);
        if (ref.status != Status::Optimal) continue;

        // Same step sizes as src/pdhg.cpp: tau = sigma = 1 / ||A||.
        const Real normA = std::max(spectralNormEstimate(mo.A), 1e-12);
        const Real tau = 1.0 / normA, sigma = 1.0 / normA;

        Csr ca = toCsr(mo.A);                    // A   in CSR, for A xbar
        Csr cat = toCsr(mo.A.transpose());       // A'  in CSR, for A'y

        std::vector<Real> x((size_t)n, 0.0), xNew((size_t)n, 0.0), extra((size_t)n, 0.0),
                          y((size_t)m, 0.0), yNew((size_t)m, 0.0),
                          aty((size_t)n, 0.0), axbar((size_t)m, 0.0),
                          xSum((size_t)n, 0.0), ySum((size_t)m, 0.0);

        const int iters = 4000;
        for (int it = 0; it < iters; ++it) {
            spmv(n, cat.rowPtr, cat.colIdx, cat.val, y, aty);           // A'y
            launchKernel(((unsigned)n + kBlock - 1) / kBlock, kBlock, [&] {
                igaos::cuda::primalStep(n, tau, mo.obj.data(), aty.data(),
                                        mo.colLower.data(), mo.colUpper.data(),
                                        x.data(), xNew.data(), extra.data());
            });
            spmv(m, ca.rowPtr, ca.colIdx, ca.val, extra, axbar);        // A xbar
            launchKernel(((unsigned)m + kBlock - 1) / kBlock, kBlock, [&] {
                igaos::cuda::dualStep(m, sigma, axbar.data(),
                                      mo.rowLower.data(), mo.rowUpper.data(),
                                      y.data(), yNew.data());
            });
            launchKernel(((unsigned)n + kBlock - 1) / kBlock, kBlock, [&] {
                igaos::cuda::accumulate(n, xNew.data(), xSum.data());
            });
            launchKernel(((unsigned)m + kBlock - 1) / kBlock, kBlock, [&] {
                igaos::cuda::accumulate(m, yNew.data(), ySum.data());
            });
            x.swap(xNew);
            y.swap(yNew);
        }

        // The ergodic average is the iterate a first-order method converges in.
        std::vector<Real> xAvg((size_t)n);
        for (Int j = 0; j < n; ++j) xAvg[(size_t)j] = xSum[(size_t)j] / (Real)iters;
        Real obj = mo.objectiveValue(xAvg);
        Real rel = std::fabs(obj - ref.objective) / (1.0 + std::fabs(ref.objective));
        Real pinf = mo.primalInfeasibility(xAvg);
        std::printf("         instance %d: %d x %d, kernels %.9g vs simplex %.9g "
                    "(rel %.2e, primal infeas %.2e)\n",
                    inst, (int)m, (int)n, (double)obj, (double)ref.objective,
                    (double)rel, (double)pinf);
        worst = std::max(worst, rel);
        ++solved;
    }
    // 4000 iterations of an unaccelerated first-order method is three digits,
    // not twelve.  The tolerance says "this is the same optimum", which is the
    // claim; a kernel with a sign error or a dropped term misses it by orders
    // of magnitude, not by 1e-3.
    check(solved == 4, "every instance had a simplex reference to compare against");
    check(worst < 5e-3, "the kernel-only PDHG loop reaches the simplex optimum", worst);
}

int main() {
    std::printf("IGAOS CUDA kernel arithmetic, executed on the CPU\n"
                "=================================================\n"
                "No GPU is involved.  This checks WHAT the kernels compute, not\n"
                "how fast they would compute it -- see the header of this file.\n\n");
    testSpmv();
    testPrimalStep();
    testDualStep();
    testAccumulateAxpby();
    testWholeSolve();
    std::printf("=================================================\n%s (%d failures)\n",
                failures ? "FAILED" : "ALL KERNEL CHECKS PASSED", failures);
    return failures ? 1 : 0;
}
