// igaos_cli : command line driver.
//   igaos model.mps [options]
#include "igaos/solver.hpp"
#include "igaos/mps.hpp"
#include "igaos/certificate.hpp"
#include "igaos/iis.hpp"
#include "igaos/sensitivity.hpp"
#include "pipeline_view.hpp"
#include "cli_extras.hpp"
#include "tui.hpp"
#include <cstring>
#include <cstdio>
#include <cstdlib>

using namespace igaos;

static void usage() {
    std::printf(
"IGAOS %s -- Indigenous GPU-Accelerated Optimization Solver\n"
"\n"
"usage: igaos <model.mps> [options]\n"
"       igaos -m <name>  [options]      solve a model from the bundled library\n"
"       igaos                           the interactive console (also: --tui, -i)\n"
"\n"
"  -i, --tui           interactive console: pick a model, toggle options, solve,\n"
"                      and browse the pipeline, log, timeline, chart and tables\n"
"  --models            list the bundled model library and exit\n"
"  -m NAME             solve a library model by name (a unique prefix will do)\n"
"  --compare cuts      solve twice, with and without separation, and tabulate\n"
"  --compare paths     solve four times, one per continuous method, and tabulate\n"
"  --chart             draw dual bound against incumbent (mixed integer only)\n"
"  --explain [STAGE]   what each stage of the pipeline does, and exit\n"
"\n"
"  --time-limit S      wall clock limit in seconds        (default none)\n"
"  --iter-limit N      simplex iteration limit\n"
"  --threads N        worker threads for the branch-and-cut tree (0 = all cores, 1 = serial)\n"
        "  --node-limit N      branch and bound node limit\n"
        "  --memory-limit MB   stop the tree when the process reaches this much\n"
        "                      memory and report the incumbent, rather than being\n"
        "                      killed with no answer.  Default is automatic: 60%% of\n"
        "                      what the machine reports free.  0 disables it\n"
        "  --threads N         worker threads for the branch-and-cut and branch-and-bound\n"
        "                      trees, MILP and MIQP alike (0 = all cores, 1 = serial and\n"
        "                      reproducible node counts)\n"
        "  --progress          emit IGAOS_STAGE events as each phase starts and finishes\n"
        "  --live              draw the solve pipeline as it runs; on by default at a\n"
        "                      terminal, and off when the output is piped or -v is used\n"
        "  --no-live           never draw it\n"
"  --no-presolve       disable presolve\n"
"  --no-scaling        disable scaling\n"
"  --scale-with-q      equilibrate [A; Q] together, not A alone (QP)\n"
"  --no-duality-gap-check  accept optimality without the primal-dual gap test\n"
"  --no-crash          disable the triangular crash basis\n"
"  --dual              start from the dual simplex\n"
"  --algorithm A       auto | primal | dual | interior | pdhg   (default auto)\n"
"                      a quadratic objective always uses interior\n"
"  --no-crossover      keep the interior point itself, do not make it basic\n"
"  --no-cuts           disable cutting-plane separation\n"
"  --cut-rounds N      root separation rounds                (default 12)\n"
"  --no-gomory         disable Gomory mixed-integer cuts\n"
"  --no-cover          disable knapsack cover cuts\n"
"  --no-mir            disable complemented MIR cuts\n"
"  --no-heuristics     disable rounding and diving heuristics\n"
"  --ipm-tol R         interior point tolerance              (default 1e-8)\n"
"  --pdhg-tol R        first-order tolerance                 (default 1e-8)\n"
"  --pdhg-iters N      first-order iteration limit\n"
"  --cutoff R          prune nodes worse than this objective\n"
"  --gap R             relative MIP gap tolerance         (default 1e-6)\n"
"  --feas-tol R        primal feasibility tolerance       (default 1e-7)\n"
"  --opt-tol R         dual feasibility tolerance         (default 1e-7)\n"
"  --refactor N        simplex updates between refactorizations\n"
"  --certificate FILE  write a checkable proof of the answer\n"
"  --iis               on an infeasible model, isolate the conflicting\n"
"                      constraints (irreducible infeasible subsystem)\n"
"  --iis-time-limit S  ceiling on the whole IIS search, seconds\n"
"  --sensitivity       shadow prices, reduced costs and the ranges over\n"
"                      which they hold (LP only)\n"
"  --solution FILE     write the solution file\n"
"  --write-mps FILE    write the model back out (round-trip check)\n"
"  -v, -vv, -vvv       verbosity\n"
"  -q                  silent\n"
"  --version           print the version and exit\n"
"  -h, --help          this message\n", "0.2.0");
}

int main(int argc, char** argv) {
    // No arguments at a terminal means "show me the solver", not "print help".
    // The interactive console is the front door; --help is still one flag away,
    // and a pipe or a script still gets the usage text.
    if (argc < 2) {
#if !defined(_WIN32)
        if (IGAOS_ISATTY(IGAOS_FILENO(stdout)) && IGAOS_ISATTY(0)) {
            Options o;
            return Tui(o).run();
        }
#endif
        usage();
        return 1;
    }
    std::string path, solFile, mpsOut, certFile;
    bool wantIis = false;
    bool wantSens = false;
    double iisTimeLimit = 0.0;
    Options opt;
    bool preferDual = false;
    int liveFlag = -1;                       // -1 decide, 0 off, 1 on
    bool wantChart = false, wantFlow = false;
    std::string compareMode, libModel;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "--version") { std::printf("IGAOS 0.2.0\n"); return 0; }
        else if (a == "--models")  return listModels(stdout);
#if !defined(_WIN32)
        else if (a == "--tui" || a == "-i") { Options o = opt; return Tui(o).run(); }
#endif
        else if (a == "--explain") {
            std::string which = (i + 1 < argc && argv[i + 1][0] != '-') ? argv[++i] : "";
            PipelineView::explain(stdout, which);
            std::printf("\n");
            return 0;
        }
        else if (a == "-m")        libModel = next();
        else if (a == "--compare") compareMode = next();
        else if (a == "--chart")   wantChart = true;
        else if (a == "--flow")    wantFlow  = true;
        else if (a == "--algorithm") {
            std::string v = next();
            if      (v == "auto")     opt.lpAlgorithm = LpAlgorithm::Auto;
            else if (v == "primal")   opt.lpAlgorithm = LpAlgorithm::PrimalSimplex;
            else if (v == "dual")     opt.lpAlgorithm = LpAlgorithm::DualSimplex;
            else if (v == "interior") opt.lpAlgorithm = LpAlgorithm::InteriorPoint;
            else if (v == "pdhg")     opt.lpAlgorithm = LpAlgorithm::PDHG;
            else { std::fprintf(stderr, "unknown algorithm '%s'\n", v.c_str()); return 1; }
        }
        else if (a == "--no-crossover")  opt.crossover = false;
        else if (a == "--no-cuts")       opt.cuts = false;
        else if (a == "--cut-rounds")    opt.cutRoundsRoot = std::stoi(next());
        else if (a == "--no-gomory")     opt.cutGomory = false;
        else if (a == "--no-cover")      opt.cutCover = false;
        else if (a == "--no-mir")        opt.cutMir = false;
        else if (a == "--no-heuristics") opt.heuristics = false;
        else if (a == "--ipm-tol")       opt.ipmTol = std::stod(next());
        else if (a == "--pdhg-tol")      opt.pdhgTol = std::stod(next());
        else if (a == "--pdhg-iters")    opt.pdhgMaxIter = std::stoll(next());
        else if (a == "--cutoff")        opt.cutoff = std::stod(next());
        else if (a == "--time-limit")  opt.timeLimit = std::stod(next());
        else if (a == "--iter-limit")  opt.iterationLimit = std::stoll(next());
        else if (a == "--node-limit")  opt.nodeLimit = std::stoll(next());
        else if (a == "--memory-limit") opt.memoryLimitMb = std::stoi(next());
        else if (a == "--threads")     opt.threads = std::stoi(next());
        else if (a == "--progress")    opt.log.progress = true;
        else if (a == "--live")        liveFlag = 1;
        else if (a == "--no-live")     liveFlag = 0;
        else if (a == "--no-presolve") opt.presolve = false;
        else if (a == "--no-scaling")  opt.scaling = false;
        else if (a == "--scale-with-q") opt.scaleWithQ = true;
        else if (a == "--duality-gap-check") opt.dualityGapCheck = true;
        else if (a == "--no-duality-gap-check") opt.dualityGapCheck = false;
        else if (a == "--no-crash")    opt.crash = false;
        else if (a == "--dual")        preferDual = true;
        else if (a == "--gap")         opt.tol.mipGapRel = std::stod(next());
        else if (a == "--feas-tol")    opt.tol.primalFeas = std::stod(next());
        else if (a == "--opt-tol")     opt.tol.dualFeas = std::stod(next());
        else if (a == "--refactor")    opt.refactorFreq = std::stoi(next());
        else if (a == "--solution")    solFile = next();
        else if (a == "--certificate") certFile = next();
        else if (a == "--iis") wantIis = true;
        else if (a == "--sensitivity") wantSens = true;
        else if (a == "--iis-time-limit") iisTimeLimit = std::stod(next());
        else if (a == "--write-mps")   mpsOut = next();
        else if (a == "-q")            opt.log.level = 0;
        else if (a == "-v")            opt.log.level = 2;
        else if (a == "-vv")           opt.log.level = 3;
        else if (a == "-vvv")          opt.log.level = 4;
        else if (!a.empty() && a[0] == '-') { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
        else path = a;
    }
    if (!libModel.empty()) {
        std::string e;
        path = resolveModel(libModel, e);
        if (path.empty()) { std::fprintf(stderr, "igaos: %s\n", e.c_str()); return 1; }
    }
    if (path.empty()) { usage(); return 1; }
    (void)preferDual;

    Model model;
    std::string err;
    Timer readTimer;
    if (!readMps(path, model, err)) { std::fprintf(stderr, "igaos: %s\n", err.c_str()); return 2; }
    double readTime = readTimer.elapsed();

    auto st = model.stats();
    opt.log.log(1,
        "IGAOS 0.2.0\n"
        "  model      %s\n"
        "  rows       %d   (%d equality, %d ranged, %d free)\n"
        "  columns    %d   (%d integer, %d binary)\n"
        "  nonzeros   %d   density %.4f%%   max/min |a| %.2e\n"
        "  read       %.3f s\n",
        model.name.c_str(), st.nrow, st.nEqRows, st.nRangeRows, st.nFreeRows,
        st.ncol, st.nint, st.nbin, st.nnz, 100.0 * st.density, st.ratio, readTime);

    if (!mpsOut.empty() && !writeMps(mpsOut, model, err))
        std::fprintf(stderr, "igaos: %s\n", err.c_str());

    // ---------------------------------------------------------- live view --
    // On by default at a terminal, because that is where somebody is watching.
    // Off when the output is piped -- a harness wants IGAOS_RESULT, not a
    // repainting box -- and off at -v and above, where the solver's own log is
    // already printing lines that a repaint would fight with. Both are
    // overridable; NO_COLOR is honoured.
    const bool atTty  = IGAOS_ISATTY(IGAOS_FILENO(stdout)) != 0;
    const bool live   = (liveFlag == 1) ||
                        (liveFlag == -1 && atTty && opt.log.level == 1 && !opt.log.progress);
    const bool colour = atTty && std::getenv("NO_COLOR") == nullptr;
    PipelineView view(stdout, colour);

    // ------------------------------------------------------ comparison modes
    // Same two comparisons the console offers, run in process rather than by
    // shelling out, and driving the same pipeline so each leg is watchable.
    if (!compareMode.empty()) {
        PipelineView* v = live ? &view : nullptr;
        if      (compareMode == "cuts")  return compareCuts (stdout, model, opt, v);
        else if (compareMode == "paths") return comparePaths(stdout, model, opt, v);
        std::fprintf(stderr, "igaos: --compare takes 'cuts' or 'paths'\n");
        return 1;
    }

    // --chart needs the same event stream the picture is drawn from, so the hook
    // goes on for either; the view simply records and stays silent when only the
    // chart was asked for.
    if (live || wantChart || wantFlow) {
        view.setQuiet(!live);
        if (live) std::printf("\n");
        opt.log.onStage = [&view](const char* n, const char* s, const char* d) {
            view.event(n, s, d);
        };
    }

    Solver solver;
    solver.opt = opt;
    Solution sol = solver.solve(model);
    if (live) { view.finish(); std::printf("\n"); }
    if (wantFlow) {
        std::vector<std::string> f = view.flowLines(78, 40, colour);
        if (f.empty())
            std::printf("  the terminal is too narrow for the chart -- try --live\n\n");
        else {
            std::printf("  %ssolve pipeline%s\n", colour ? "\x1b[1m" : "", colour ? "\x1b[0m" : "");
            for (const std::string& l : f) std::printf("  %s\n", l.c_str());
            std::printf("\n");
        }
    }
    if (wantChart && !view.drawConvergence())
        std::printf("  nothing to chart: a dual bound and an incumbent only move "
                    "apart in a branch and bound run.\n\n");
    const SolveReport& rep = solver.report;

    opt.log.log(1,
        "  presolve   %d rows, %d cols, %d nonzeros removed; %d bounds tightened (%.3f s)\n"
        "  algorithm  %s\n",
        rep.origRows - rep.presolvedRows, rep.origCols - rep.presolvedCols,
        rep.origNnz - rep.presolvedNnz, rep.tightenedBounds, rep.presolveTime,
        sol.algorithm.c_str());

    if (rep.cutsApplied > 0) {
        double denom = rep.rootBoundCut - rep.rootBoundLp;
        opt.log.log(1,
        "  cuts       %d kept (%d gomory, %d cover, %d mir) over %d rounds;\n"
        "             root bound %.10g -> %.10g  (%+.6g)\n",
        rep.cutsApplied, rep.cutsGomory, rep.cutsCover, rep.cutsMir, rep.cutRounds,
        (double)rep.rootBoundLp, (double)rep.rootBoundCut, denom);
    }
    if (rep.ipmIterations > 0)
        opt.log.log(1, "  interior   %d iterations", rep.ipmIterations);
    if (rep.crossoverIterations > 0 || rep.crossoverPushes > 0)
        opt.log.log(1, ";  crossover %d pushes, %lld iterations",
                    rep.crossoverPushes, (long long)rep.crossoverIterations);
    if (rep.ipmIterations > 0) opt.log.log(1, "\n");
    if (rep.pdhgIterations > 0)
        opt.log.log(1, "  first-order %lld iterations\n", (long long)rep.pdhgIterations);

    Real pinf = model.primalInfeasibility(sol.colValue);
    Real iinf = model.integerInfeasibility(sol.colValue, opt.tol.integrality);

    opt.log.log(1,
        "\nRESULT\n"
        "  status         %s\n"
        "  objective      %.12g\n", statusName(sol.status), (double)sol.objective);
    if (model.isMip())
        opt.log.log(1,
        "  best bound     %.12g\n"
        "  gap            %.4g%%\n"
        "  nodes          %lld\n",
        (double)sol.bestBound, (double)(100.0 * sol.mipGap), (long long)sol.nodes);
    opt.log.log(1,
        "  iterations     %lld\n"
        "  primal infeas  %.3e\n"
        "  integer infeas %.3e\n"
        "  dual infeas    %.3e\n"
        "  basis cond est %.3e\n"
        "  time           %.3f s  (presolve %.3f, solve %.3f, cleanup %.3f)\n",
        (long long)rep.simplexIterations, (double)pinf, (double)iinf, (double)sol.dualInf,
        (double)rep.conditionEstimate,
        rep.totalTime, rep.presolveTime, rep.solveTime, rep.cleanupTime);

    if (!solFile.empty() && !writeSolution(solFile, model, sol, err))
        std::fprintf(stderr, "igaos: %s\n", err.c_str());

    if (wantSens) {
        Sensitivity sen = computeSensitivity(model, sol, opt);
        std::printf("%s", formatSensitivity(model, sen).c_str());
    }

    if (wantIis) {
        if (sol.status != Status::Infeasible) {
            std::printf("\n  --iis: model is %s, not infeasible -- nothing to isolate\n",
                        statusName(sol.status));
        } else {
            IisOptions io;
            io.logLevel = 0;
            io.timeLimit = iisTimeLimit;
            Iis iis = computeIis(model, io);
            std::printf("\n%s", formatIis(model, iis).c_str());
        }
    }

    if (!certFile.empty()) {
        // A mixed-integer certificate used to carry the incumbent and nothing
        // else, because the branch-and-bound tree emits no dual multipliers --
        // so it proved feasibility and integrality but no bound at all.
        //
        // The theorem behind the certificate holds for ANY vector y, so the
        // duals of the LP RELAXATION of the original model are a legitimate
        // choice: L(y) computed from them is a rigorous lower bound on the
        // relaxation, and therefore on the mixed-integer optimum. Solving that
        // relaxation once more costs one LP and turns "no bound certified" into
        // a proven interval around the answer.
        //
        // It is the ROOT bound, not the tree's. It does not prove the incumbent
        // optimal, and the checker is told exactly that.
        if (model.isMip() && sol.status != Status::Infeasible) {
            Model relaxed = model;
            for (VarType& t : relaxed.colType) t = VarType::Continuous;
            Options ropt = opt;
            ropt.log.level = 0;
            ropt.log.progress = false;
            ropt.log.onStage = nullptr;
            Solver rsolver;
            rsolver.opt = ropt;
            Solution rsol = rsolver.solve(relaxed);
            if (rsol.status == Status::Optimal &&
                rsol.rowDual.size() == (size_t)model.numRow()) {
                sol.rowDual = rsol.rowDual;
                opt.log.log(1, "  certificate: root LP relaxation solved for dual "
                               "multipliers (bound %.12g)\n", (double)rsol.objective);
            }
        }
        if (writeCertificate(certFile, model, sol, err))
            std::printf("  certificate written to %s -- check it with\n"
                        "    python3 tools/verify_certificate.py <model> %s\n",
                        certFile.c_str(), certFile.c_str());
        else
            std::fprintf(stderr, "igaos: %s\n", err.c_str());
    }

    // machine-readable one-liner for the benchmark harness
    std::printf("IGAOS_RESULT status=%s obj=%.12g bound=%.12g iters=%lld nodes=%lld "
                "pinf=%.3e iinf=%.3e dinf=%.3e time=%.4f cuts=%d cutrounds=%d "
                "rootlp=%.12g rootcut=%.12g\n",
                statusName(sol.status), (double)sol.objective, (double)sol.bestBound,
                (long long)rep.simplexIterations, (long long)sol.nodes,
                (double)pinf, (double)iinf, (double)sol.dualInf, rep.totalTime,
                (int)rep.cutsApplied, (int)rep.cutRounds,
                (double)rep.rootBoundLp, (double)rep.rootBoundCut);
    return sol.status == Status::Optimal ? 0 : 10;
}
