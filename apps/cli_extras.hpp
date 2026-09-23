#pragma once
// ===========================================================================
//  cli_extras.hpp -- the parts of the console the command line was missing.
//
//  Solver Studio does four things beyond drawing the pipeline: it offers a
//  library of models so you do not have to know a path, it solves the same
//  model twice with and without cut separation, it solves it four times with
//  each continuous method, and it explains what each stage is. None of that
//  is a property of a browser. It is here, in the interface the problem
//  statement actually asks for.
//
//  The comparison runs are done IN PROCESS -- the model is read once and
//  handed to a fresh Solver for each leg. The console shells out to the
//  binary because it has to; this does not.
// ===========================================================================

#include "igaos/solver.hpp"
#include "igaos/mps.hpp"
#include "pipeline_view.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>

#if !defined(_WIN32)
#  include <dirent.h>
#endif

namespace igaos {

// -------------------------------------------------------------- the library
// The fourteen instances that ship with the console live in demo/models. Find
// that directory from wherever the user happens to be standing: the repository
// root, inside build/, or one level down.
inline bool isDir(const std::string& p) {
    struct stat s;
    return ::stat(p.c_str(), &s) == 0 && (s.st_mode & S_IFDIR);
}
inline bool isFile(const std::string& p) {
    struct stat s;
    return ::stat(p.c_str(), &s) == 0 && (s.st_mode & S_IFREG);
}
inline std::string modelsDir() {
    static const char* candidates[] = {
        "demo/models", "../demo/models", "../../demo/models", "./models"
    };
    for (const char* c : candidates) if (isDir(c)) return c;
    return std::string();
}

struct LibModel { std::string tag, path; Int rows = 0, cols = 0, ints = 0; Long nnz = 0; };

inline std::vector<LibModel> library() {
    std::vector<LibModel> out;
    const std::string dir = modelsDir();
    if (dir.empty()) return out;
#if !defined(_WIN32)
    DIR* d = ::opendir(dir.c_str());
    if (!d) return out;
    while (struct dirent* e = ::readdir(d)) {
        std::string n = e->d_name;
        if (n.size() < 5 || n.substr(n.size() - 4) != ".mps") continue;
        LibModel m;
        m.tag  = n.substr(0, n.size() - 4);
        m.path = dir + "/" + n;
        out.push_back(m);
    }
    ::closedir(d);
#endif
    std::sort(out.begin(), out.end(),
              [](const LibModel& a, const LibModel& b) { return a.tag < b.tag; });
    return out;
}

// Resolve "-m uc_m" to a path. An exact tag wins; otherwise a unique prefix.
inline std::string resolveModel(const std::string& want, std::string& err) {
    if (isFile(want)) return want;
    std::vector<LibModel> lib = library();
    if (lib.empty()) {
        err = "no model library found -- run from the repository root, or give a path";
        return std::string();
    }
    for (const LibModel& m : lib) if (m.tag == want) return m.path;
    std::vector<const LibModel*> hits;
    for (const LibModel& m : lib)
        if (m.tag.compare(0, want.size(), want) == 0) hits.push_back(&m);
    if (hits.size() == 1) return hits[0]->path;
    if (hits.empty()) err = "no model called '" + want + "'. Try --models.";
    else {
        err = "'" + want + "' matches";
        for (const LibModel* h : hits) err += " " + h->tag;
    }
    return std::string();
}

inline int listModels(FILE* out) {
    std::vector<LibModel> lib = library();
    if (lib.empty()) {
        std::fprintf(out, "No model library found. These ship in demo/models; run this\n"
                          "from the repository root, or pass an MPS path directly.\n");
        return 1;
    }
    std::fprintf(out, "\n  %-16s %9s %9s %11s %9s   %s\n",
                 "name", "rows", "columns", "nonzeros", "integer", "kind");
    std::fprintf(out, "  %s\n", std::string(72, '-').c_str());
    for (LibModel& m : lib) {
        Model mod; std::string err;
        if (!readMps(m.path, mod, err)) continue;
        Model::Stats st = mod.stats();
        // A quadratic CONSTRAINT outranks everything else in this label: it is
        // what decides the solve path, and calling a pooling model "LP" because
        // its objective happens to be linear would be actively misleading.
        const char* kind =
            (mod.isQcqp() || mod.hasNonconvexObjective())
                ? (mod.isMip() ? "MINLP" : "QCQP")
                : (mod.isQp() ? (mod.isMip() ? "MIQP" : "QP")
                              : (mod.isMip() ? "MILP" : "LP"));
        std::fprintf(out, "  %-16s %9d %9d %11lld %9d   %s\n",
                     m.tag.c_str(), st.nrow, st.ncol, (long long)st.nnz, st.nint, kind);
    }
    std::fprintf(out, "\n  solve one with:   igaos -m <name>\n"
                      "  compare with:     igaos -m <name> --compare cuts\n"
                      "                    igaos -m <name> --compare paths\n\n");
    return 0;
}

// ------------------------------------------------------------------ tables
struct Col { std::string head; std::vector<std::string> cells; bool right = true; };

inline void printTable(FILE* out, std::vector<Col>& cols) {
    std::vector<size_t> w(cols.size());
    for (size_t i = 0; i < cols.size(); ++i) {
        w[i] = cols[i].head.size();
        for (const std::string& c : cols[i].cells) w[i] = std::max(w[i], c.size());
    }
    std::fprintf(out, "  ");
    for (size_t i = 0; i < cols.size(); ++i)
        std::fprintf(out, "%-*s  ", (int)w[i], cols[i].head.c_str());
    std::fprintf(out, "\n  ");
    for (size_t i = 0; i < cols.size(); ++i)
        std::fprintf(out, "%s  ", std::string(w[i], '-').c_str());
    std::fputc('\n', out);
    const size_t rows = cols.empty() ? 0 : cols[0].cells.size();
    for (size_t r = 0; r < rows; ++r) {
        std::fprintf(out, "  ");
        for (size_t i = 0; i < cols.size(); ++i) {
            const std::string& c = cols[i].cells[r];
            if (cols[i].right && i) std::fprintf(out, "%*s  ", (int)w[i], c.c_str());
            else                    std::fprintf(out, "%-*s  ", (int)w[i], c.c_str());
        }
        std::fputc('\n', out);
    }
}

inline std::string num(double v, int prec = 6) {
    char b[64]; std::snprintf(b, sizeof b, "%.*g", prec, v); return b;
}
inline std::string num(long long v) {
    char b[48]; std::snprintf(b, sizeof b, "%lld", v);
    std::string s = b, o; int c = 0;
    const bool neg = !s.empty() && s[0] == '-'; if (neg) s.erase(0, 1);
    for (int i = (int)s.size() - 1; i >= 0; --i) {
        o.insert(o.begin(), s[(size_t)i]);
        if (++c % 3 == 0 && i > 0) o.insert(o.begin(), ',');
    }
    return (neg ? "-" : "") + o;
}
inline std::string tsec(double t) {
    char b[48];
    if (t < 1.0) std::snprintf(b, sizeof b, "%.0f ms", t * 1e3);
    else         std::snprintf(b, sizeof b, "%.2f s", t);
    return b;
}

// One leg of a comparison: solve, remember what mattered.
struct Leg {
    std::string label;
    Status status = Status::NotSolved;
    Real obj = 0, bound = 0, rootLp = 0, rootCut = 0;
    Long iters = 0, nodes = 0;
    Int  cuts = 0;
    double time = 0;
    bool fellBack = false;
};

inline Leg runLeg(const Model& model, Options opt, const std::string& label,
                  PipelineView* view) {
    Leg L; L.label = label;
    if (view) {
        view->startLeg(label);
        opt.log.onStage = [view](const char* n, const char* s, const char* d) {
            view->event(n, s, d);
        };
    }
    opt.log.level = 0;                      // the table is the output, not a log
    Solver solver; solver.opt = opt;
    Solution sol = solver.solve(model);
    if (view) view->finish();
    const SolveReport& r = solver.report;
    L.status = sol.status;   L.obj    = sol.objective; L.bound = sol.bestBound;
    L.iters  = r.simplexIterations;
    if (r.ipmIterations  > 0) L.iters = r.ipmIterations;
    if (r.pdhgIterations > 0) L.iters = r.pdhgIterations;
    L.nodes = sol.nodes; L.cuts = r.cutsApplied; L.time = r.totalTime;
    L.rootLp = r.rootBoundLp; L.rootCut = r.rootBoundCut;
    return L;
}

// ------------------------------------------------------- cuts on / cuts off
inline int compareCuts(FILE* out, const Model& model, Options opt, PipelineView* view) {
    if (!model.isMip()) {
        std::fprintf(out, "\n  This model has no integer columns, so there is nothing to\n"
                          "  separate. Try a mixed-integer one: igaos --models\n\n");
        return 1;
    }
    Options on = opt;  on.cuts = true;
    Options off = opt; off.cuts = false;
    Leg a = runLeg(model, on,  "leg 1 of 2 -- cut separation on",  view);
    Leg b = runLeg(model, off, "leg 2 of 2 -- cut separation off", view);

    std::vector<Col> t(3);
    t[0].head = "measure";       t[0].right = false;
    t[1].head = "with cuts";     t[2].head = "without";
    auto row = [&](const char* k, const std::string& x, const std::string& y) {
        t[0].cells.push_back(k); t[1].cells.push_back(x); t[2].cells.push_back(y);
    };
    row("status",             statusName(a.status),   statusName(b.status));
    row("objective",          num(a.obj, 12),         num(b.obj, 12));
    row("nodes",              num((long long)a.nodes), num((long long)b.nodes));
    row("simplex iterations", num((long long)a.iters), num((long long)b.iters));
    row("time",               tsec(a.time),           tsec(b.time));
    row("cuts kept",          num((long long)a.cuts), num((long long)b.cuts));
    row("root LP bound",      num(a.rootLp, 10),      "-");
    row("bound after cuts",   num(a.rootCut, 10),     "-");
    std::fprintf(out, "\n");
    printTable(out, t);

    const bool same = std::fabs(a.obj - b.obj) <= 1e-6 * (1.0 + std::fabs(a.obj));
    if (same)
        std::fprintf(out, "\n  Both runs return %s. The difference is how much work it\n"
                          "  took to prove it -- a cut that changed the answer would be a\n"
                          "  defect, not an optimisation.\n\n", num(a.obj, 12).c_str());
    else
        std::fprintf(out, "\n  The two objectives DIFFER. Check whether either hit a limit;\n"
                          "  if neither did, that is an invalid cut and a wrong answer.\n\n");
    return 0;
}

// --------------------------------------------------- the four continuous paths
inline int comparePaths(FILE* out, const Model& model, Options opt, PipelineView* view) {
    struct Entry { LpAlgorithm alg; const char* name; };
    static const Entry entries[] = {
        {LpAlgorithm::PrimalSimplex, "primal simplex"},
        {LpAlgorithm::DualSimplex,   "dual simplex"},
        {LpAlgorithm::InteriorPoint, "interior point"},
        {LpAlgorithm::PDHG,          "first-order (PDHG)"},
    };
    std::vector<Leg> legs;
    int n = 0;
    for (const Entry& e : entries) {
        Options o = opt; o.lpAlgorithm = e.alg;
        char lab[96];
        std::snprintf(lab, sizeof lab, "leg %d of 4 -- %s", ++n, e.name);
        Leg L = runLeg(model, o, lab, view);
        L.label = e.name;
        legs.push_back(L);
    }

    std::vector<Col> t(5);
    t[0].head = "method"; t[0].right = false;
    t[1].head = "status"; t[2].head = "objective"; t[3].head = "iterations"; t[4].head = "time";
    for (const Leg& L : legs) {
        t[0].cells.push_back(L.label);
        t[1].cells.push_back(statusName(L.status));
        t[2].cells.push_back(num(L.obj, 12));
        t[3].cells.push_back(num((long long)L.iters));
        t[4].cells.push_back(tsec(L.time));
    }
    std::fprintf(out, "\n");
    printTable(out, t);

    double lo = 0, hi = 0; bool first = true;
    for (const Leg& L : legs) {
        if (L.status != Status::Optimal && L.status != Status::Feasible) continue;
        if (first) { lo = hi = L.obj; first = false; }
        lo = std::min(lo, (double)L.obj); hi = std::max(hi, (double)L.obj);
    }
    const double spread = hi - lo;
    if (first) std::fprintf(out, "\n  No leg reached a provable answer.\n\n");
    else if (spread <= 1e-6 * (1.0 + std::fabs(lo)))
        std::fprintf(out, "\n  Four separate solves, four different algorithms, one answer:\n"
                          "  %s -- agreeing to %s.\n\n", num(lo, 12).c_str(),
                     spread == 0 ? "every printed digit" : num(spread, 3).c_str());
    else
        std::fprintf(out, "\n  The four disagree by %s. On a proven optimum that is a\n"
                          "  defect, not a tolerance.\n\n", num(spread, 3).c_str());
    return 0;
}

} // namespace igaos
