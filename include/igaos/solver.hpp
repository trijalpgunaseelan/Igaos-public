// solver.hpp : the top-level driver.
//
//   Model -> presolve -> scale -> solve -> unscale -> postsolve -> cleanup
//
// The cleanup step re-solves the *original* problem starting from the basis
// postsolve reconstructed.  It normally costs zero iterations and makes the
// reported primal values, duals and reduced costs exact in the user's own
// space, independent of how aggressive presolve was.
#pragma once
#include "igaos/model.hpp"
#include "igaos/presolve.hpp"

namespace igaos {

struct SolveReport {
    Int    origRows = 0, origCols = 0, origNnz = 0, origInt = 0;
    Int    presolvedRows = 0, presolvedCols = 0, presolvedNnz = 0;
    Int    tightenedBounds = 0;
    double presolveTime = 0, solveTime = 0, cleanupTime = 0, totalTime = 0;
    Long   simplexIterations = 0, nodes = 0;
    Real   conditionEstimate = 1.0;
    Real   maxCoefRatio = 1.0;
    std::string path;                 // which algorithm actually ran

    // cutting planes (MIP only)
    Int    cutsApplied = 0, cutRounds = 0;
    Int    cutsGomory = 0, cutsCover = 0, cutsMir = 0;
    Real   rootBoundLp = 0.0;         // root LP bound before separation
    Real   rootBoundCut = 0.0;        // root LP bound after separation
    // Non-zero only when Options::cutReference was supplied: candidate cuts
    // that would have excluded the verification point.  Must be 0 on a correct
    // solver; the regression suite asserts exactly that.
    Int    cutsInvalid = 0;
    const char* cutInvalidOrigin = nullptr;
    Real   cutInvalidWorst = 0.0;

    // interior point / first-order paths
    Int    ipmIterations = 0;
    Long   nodeRelaxations = 0;       // MIQP: interior point solves in the tree
    Long   pdhgIterations = 0;
    Int    crossoverPushes = 0;
    Long   crossoverIterations = 0;
};

class Solver {
public:
    Options opt;
    SolveReport report;

    Solution solve(const Model& model);

private:
    Solution solveContinuous(const Model& m, bool preferDual);
    Solution solveSimplex(const Model& m, bool preferDual);
    Solution solveInterior(const Model& m, bool allowFallback);
    Solution solveFirstOrder(const Model& m);
    Solution solveMip(const Model& m);
    Solution solveMiqp(const Model& m);
    // Nonconvex QCQP and bilinear MINLP, solved to proven GLOBAL optimality by
    // spatial branch and bound over McCormick relaxations.  See src/global.cpp.
    Solution solveGlobal(const Model& m);
};

} // namespace igaos
