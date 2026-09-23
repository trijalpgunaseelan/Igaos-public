// wasm_main.cpp : WebAssembly entry point for IGAOS Solver Studio.
//
// The native CLI (apps/igaos_cli.cpp) drives a terminal UI through termios,
// which WebAssembly has no equivalent for. This driver keeps the same solver
// and the same MPS reader and replaces only the front end: it reads a model
// from the WASI filesystem, solves it, and writes one JSON object to stdout
// for the page to parse.
//
// Nothing about the numerics differs from the native build. The one real
// difference is the tree: this binary is linked against a single-threaded
// <thread> shim, so branch-and-bound runs serially and is deterministic --
// the same search a native build gives with `--threads 1`.

#include <igaos/model.hpp>
#include <igaos/mps.hpp>
#include <igaos/solver.hpp>
#include <igaos/common.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace igaos;

namespace {

// Minimal JSON string escaping -- model names and error text come from a file
// the visitor supplied, so they are not trusted to be JSON-safe.
std::string esc(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

// A finite double, or JSON null. Infinity and NaN are not valid JSON numbers
// and an unbounded objective is exactly the case that produces them.
std::string num(double v) {
    if (v != v) return "null";                       // NaN
    if (v >= 1e300 || v <= -1e300) return "null";    // +/- inf and kInf
    char b[40];
    std::snprintf(b, sizeof b, "%.12g", v);
    return b;
}

int usage() {
    std::fprintf(stderr, "usage: igaos <model.mps> [--time-limit S] [--no-cuts] "
                         "[--no-presolve] [--no-scaling] [--max-nodes N]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage();

    const std::string path = argv[1];
    double timeLimit = 20.0;
    bool   cuts = true, presolve = true, scaling = true;
    long long nodeLimit = 0;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--no-cuts")     cuts = false;
        else if (a == "--no-presolve") presolve = false;
        else if (a == "--no-scaling")  scaling = false;
        else if (a == "--time-limit" && i + 1 < argc) timeLimit = std::atof(argv[++i]);
        else if (a == "--max-nodes"  && i + 1 < argc) nodeLimit = std::atoll(argv[++i]);
        else return usage();
    }

    Model model;
    std::string err;
    if (!readMps(path, model, err)) {
        std::printf("{\"ok\":false,\"error\":\"%s\"}\n", esc(err).c_str());
        return 1;
    }

    // Shape of the model as read, before presolve touches it.
    const Int  rows = model.numRow();
    const Int  cols = model.numCol();
    const Int  ints = model.numInt();
    const Int  nnz  = model.A.nnz();
    model.ensureNames();

    Solver solver;
    solver.opt.timeLimit     = timeLimit;
    solver.opt.cuts          = cuts;
    solver.opt.presolve      = presolve;
    solver.opt.scaling       = scaling;
    solver.opt.threads       = 1;     // the shim is serial; say so explicitly
    solver.opt.memoryLimitMb = 0;     // the browser tab owns the memory limit
    if (nodeLimit > 0) solver.opt.nodeLimit = (Long)nodeLimit;

    Solution sol = solver.solve(model);
    const SolveReport& r = solver.report;

    std::string out = "{\"ok\":true";
    out += ",\"status\":\"";   out += statusName(sol.status); out += "\"";
    out += ",\"statusCode\":"; out += std::to_string((int)sol.status);
    out += ",\"objective\":";  out += num(sol.objective);
    out += ",\"bestBound\":";  out += num(sol.bestBound);
    out += ",\"mipGap\":";     out += num(sol.mipGap);
    out += ",\"iterations\":"; out += std::to_string((long long)sol.iterations);
    out += ",\"nodes\":";      out += std::to_string((long long)sol.nodes);
    out += ",\"solveTime\":";  out += num(sol.solveTime);
    out += ",\"totalTime\":";  out += num(r.totalTime);
    out += ",\"primalInf\":";  out += num(sol.primalInf);
    out += ",\"dualInf\":";    out += num(sol.dualInf);
    out += ",\"algorithm\":\"";out += esc(sol.algorithm); out += "\"";
    out += ",\"path\":\"";     out += esc(r.path);        out += "\"";

    out += ",\"rows\":";       out += std::to_string((long long)rows);
    out += ",\"cols\":";       out += std::to_string((long long)cols);
    out += ",\"integers\":";   out += std::to_string((long long)ints);
    out += ",\"nnz\":";        out += std::to_string((long long)nnz);

    out += ",\"presolvedRows\":"; out += std::to_string((long long)r.presolvedRows);
    out += ",\"presolvedCols\":"; out += std::to_string((long long)r.presolvedCols);
    out += ",\"presolveTime\":";  out += num(r.presolveTime);

    out += ",\"cutsApplied\":";   out += std::to_string((long long)r.cutsApplied);
    out += ",\"cutRounds\":";     out += std::to_string((long long)r.cutRounds);
    out += ",\"cutsGomory\":";    out += std::to_string((long long)r.cutsGomory);
    out += ",\"cutsCover\":";     out += std::to_string((long long)r.cutsCover);
    out += ",\"cutsMir\":";       out += std::to_string((long long)r.cutsMir);
    out += ",\"rootBoundLp\":";   out += num(r.rootBoundLp);
    out += ",\"rootBoundCut\":";  out += num(r.rootBoundCut);
    out += ",\"ipmIterations\":"; out += std::to_string((long long)r.ipmIterations);
    out += ",\"simplexIterations\":"; out += std::to_string((long long)r.simplexIterations);

    // The first 200 structural values, enough for the page to show a solution
    // vector without shipping a megabyte of JSON for a large model.
    const Int show = cols < 200 ? cols : 200;
    out += ",\"varsShown\":"; out += std::to_string((long long)show);
    out += ",\"vars\":[";
    for (Int j = 0; j < show; ++j) {
        if (j) out += ",";
        out += "{\"n\":\"" + esc(model.colName[(size_t)j]) + "\",\"v\":" + num(sol.colValue[(size_t)j])
             + ",\"i\":" + (model.colType[(size_t)j] != VarType::Continuous ? "1" : "0") + "}";
    }
    out += "]}";

    std::printf("%s\n", out.c_str());
    return 0;
}
