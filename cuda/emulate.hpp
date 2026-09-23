// emulate.hpp : run the CUDA kernels in pdhg_kernels.cu on an ordinary CPU.
//
// ===========================================================================
//  WHAT THIS IS FOR
// ===========================================================================
//  The kernels compile (tools/cuda_compile_check.py drives NVRTC and ptxas, and
//  neither needs a device).  Compiling says nothing about what they COMPUTE.
//  This header closes that gap without a GPU: it defines the handful of CUDA
//  built-ins the kernels use, so `pdhg_kernels.cu` can be included into an
//  ordinary C++ translation unit and the kernel bodies -- the actual text, not
//  a re-implementation of it -- executed and checked against the CPU solver.
//
//  What it does verify: the arithmetic and the indexing.  Every lane, every
//  round of the warp reduction, every clamp, every fused expression.
//
//  What it does NOT verify, and cannot: occupancy, coalescing, launch
//  configuration limits, stream semantics, real hardware rounding of anything
//  the compiler is allowed to contract, or the host launcher at the bottom of
//  the .cu.  "The arithmetic is right" is a smaller claim than "the GPU path
//  works", and this file only supports the smaller one.
//
// ===========================================================================
//  HOW A WARP IS EMULATED WITHOUT THREADS
// ===========================================================================
//  __shfl_down_sync exchanges a value between lanes of a warp mid-function, so
//  a single-threaded emulator cannot simply run lane 0 to completion, then lane
//  1: when lane 0 asks for lane 16's value, lane 16 has not run yet.  Threads or
//  fibres would solve it and both cost more than this is worth.
//
//  Instead the whole 32-lane sweep is run REPEATEDLY to a fixed point.  Each
//  shuffle is numbered by the round it occurs in (a per-lane counter, reset at
//  the start of every lane's run) and writes its argument into a table indexed
//  by (round, lane); a read of a lane that has not run yet in this sweep sees
//  the previous sweep's entry.  Round 0's inputs depend on no shuffle, so sweep
//  0 fixes round 0; sweep 1 then fixes round 1, and so on.  After as many
//  sweeps as there are rounds the table is stationary and the last sweep's
//  writes are the true result.  The loop below detects that stationarity rather
//  than assuming a round count, so it stays correct if the reduction changes.
//
//  This is exact, not approximate: at the fixed point every lane read the value
//  the corresponding lane actually produced, in the same round.  The cost is
//  running each kernel body six times instead of once, which for a verification
//  tool is free.
// ===========================================================================
#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

// The kernels are ordinary functions here.
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

namespace igaos {
namespace cudaemu {

struct Dim3 { unsigned x = 1, y = 1, z = 1; };

// The launch coordinates the kernel body reads.  Single-threaded emulation, so
// plain globals rather than thread_local: nothing here is re-entrant and
// pretending otherwise would only hide that.
inline Dim3& threadIdxRef() { static Dim3 d; return d; }
inline Dim3& blockIdxRef()  { static Dim3 d; return d; }
inline Dim3& blockDimRef()  { static Dim3 d; return d; }
inline Dim3& gridDimRef()   { static Dim3 d; return d; }

// One warp's shuffle table: value[round][lane], with a matching filled flag so
// "not yet written" is distinguishable from "written as zero".
struct WarpState {
    std::vector<std::vector<double>> value;
    std::vector<std::vector<char>>   filled;
    int  lane   = 0;
    int  round  = 0;
    bool changed = false;

    void beginSweep() { changed = false; }
    void beginLane(int l) { lane = l; round = 0; }

    double shuffleDown(double var, unsigned delta) {
        const int r = round++;
        if ((int)value.size() <= r) {
            value.resize((size_t)r + 1, std::vector<double>(32, 0.0));
            filled.resize((size_t)r + 1, std::vector<char>(32, 0));
        }
        if (!filled[(size_t)r][(size_t)lane] || value[(size_t)r][(size_t)lane] != var) {
            // A NaN never compares equal to itself, so compare the bits rather
            // than the values: otherwise a NaN in the reduction would keep the
            // sweep loop reporting "changed" forever.
            double old = value[(size_t)r][(size_t)lane];
            bool same = filled[(size_t)r][(size_t)lane] &&
                        std::memcmp(&old, &var, sizeof(double)) == 0;
            if (!same) changed = true;
            value[(size_t)r][(size_t)lane] = var;
            filled[(size_t)r][(size_t)lane] = 1;
        }
        const unsigned src = (unsigned)lane + delta;
        // CUDA semantics: a source lane outside the warp leaves the value
        // unchanged, which is what makes the butterfly reduction correct for a
        // partially populated warp.
        if (src >= 32u) return var;
        return filled[(size_t)r][src] ? value[(size_t)r][src] : 0.0;
    }
};

// The warps of the block currently being emulated, indexed by warp-in-block.
inline std::vector<WarpState>& warps() { static std::vector<WarpState> w; return w; }
inline int& currentWarp() { static int w = 0; return w; }

inline double shflDownSync(unsigned, double var, unsigned delta) {
    return warps()[(size_t)currentWarp()].shuffleDown(var, delta);
}

// ---------------------------------------------------------------------------
// launchWarpKernel: for kernels that use __shfl_down_sync.  Runs every thread
// of every block, sweeping each block to the fixed point described above.
// ---------------------------------------------------------------------------
template <typename Body>
void launchWarpKernel(unsigned gridDim_, unsigned blockDim_, Body body) {
    const int warpsPerBlock = (int)blockDim_ / 32;
    blockDimRef() = Dim3{blockDim_, 1, 1};
    gridDimRef()  = Dim3{gridDim_, 1, 1};
    for (unsigned b = 0; b < gridDim_; ++b) {
        blockIdxRef() = Dim3{b, 1, 1};
        warps().assign((size_t)(warpsPerBlock > 0 ? warpsPerBlock : 1), WarpState{});
        for (int sweep = 0; sweep < 64; ++sweep) {
            for (WarpState& w : warps()) w.beginSweep();
            for (unsigned t = 0; t < blockDim_; ++t) {
                currentWarp() = (int)t / 32;
                warps()[(size_t)currentWarp()].beginLane((int)t % 32);
                threadIdxRef() = Dim3{t, 1, 1};
                body();
            }
            bool moved = false;
            for (const WarpState& w : warps()) moved = moved || w.changed;
            if (!moved) break;                      // table stationary: done
        }
    }
}

// ---------------------------------------------------------------------------
// launchKernel: for the elementwise kernels, which have no cross-lane traffic,
// so one pass over the threads is the whole story.
// ---------------------------------------------------------------------------
template <typename Body>
void launchKernel(unsigned gridDim_, unsigned blockDim_, Body body) {
    blockDimRef() = Dim3{blockDim_, 1, 1};
    gridDimRef()  = Dim3{gridDim_, 1, 1};
    for (unsigned b = 0; b < gridDim_; ++b) {
        blockIdxRef() = Dim3{b, 1, 1};
        for (unsigned t = 0; t < blockDim_; ++t) {
            threadIdxRef() = Dim3{t, 1, 1};
            body();
        }
    }
}

} // namespace cudaemu
} // namespace igaos

#include <cstring>

// The names the kernel bodies use, bound to the emulator's state.
#define threadIdx (::igaos::cudaemu::threadIdxRef())
#define blockIdx  (::igaos::cudaemu::blockIdxRef())
#define blockDim  (::igaos::cudaemu::blockDimRef())
#define gridDim   (::igaos::cudaemu::gridDimRef())
#define __shfl_down_sync(mask, var, delta) \
    (::igaos::cudaemu::shflDownSync((mask), (var), (delta)))
