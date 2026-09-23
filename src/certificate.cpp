#include "igaos/certificate.hpp"
#include <cstdio>
#include <cmath>

namespace igaos {

namespace {

// C99 %a: the exact bits, in a form a checker can read back without loss.
// "%.17g" would round-trip on most implementations; "%a" cannot fail to.
void putValue(std::FILE* f, const char* tag, Int idx, Real v) {
    std::fprintf(f, "%s %d %a\n", tag, (int)idx, (double)v);
}

} // namespace

bool writeCertificate(const std::string& path, const Model& m,
                      const Solution& sol, std::string& err) {
    const bool haveClaim =
        sol.status == Status::Optimal ||
        sol.status == Status::Infeasible ||
        (sol.status == Status::Feasible && m.isMip());
    if (!haveClaim) {
        err = "no certifiable claim for status " + std::string(statusName(sol.status));
        return false;
    }

    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) { err = "cannot write " + path; return false; }

    const Int n = m.numCol(), nr = m.numRow();

    std::fprintf(f, "IGAOS-CERT 1\n");
    std::fprintf(f, "# A certificate is the pair (x, y).  See include/igaos/certificate.hpp\n");
    std::fprintf(f, "# for the one-line theorem it rests on, and tools/verify_certificate.py\n");
    std::fprintf(f, "# for an independent checker that reads it in exact arithmetic.\n");
    std::fprintf(f, "# Entries not listed are zero.  Values are C99 hex floats: exact bits.\n");
    std::fprintf(f, "problem %s\n", m.name.c_str());
    std::fprintf(f, "sense %s\n", m.sense == Sense::Maximize ? "maximize" : "minimize");
    std::fprintf(f, "rows %d\ncols %d\n", (int)nr, (int)n);
    std::fprintf(f, "integer %d\n", (int)m.numInt());
    std::fprintf(f, "quadratic %d\n", m.isQp() ? 1 : 0);

    switch (sol.status) {
    case Status::Optimal:
        std::fprintf(f, "claim optimal\n");
        std::fprintf(f, "objective %a\n", (double)sol.objective);
        break;
    case Status::Infeasible:
        // c is taken as zero for the infeasibility theorem; the checker knows.
        std::fprintf(f, "claim infeasible\n");
        break;
    default:
        std::fprintf(f, "claim feasible\n");
        std::fprintf(f, "objective %a\n", (double)sol.objective);
        std::fprintf(f, "bound %a\n", (double)sol.bestBound);
        break;
    }

    if (sol.status != Status::Infeasible) {
        std::fprintf(f, "# primal point\n");
        for (Int j = 0; j < n && j < (Int)sol.colValue.size(); ++j)
            if (sol.colValue[j] != 0.0) putValue(f, "x", j, sol.colValue[j]);
    }

    // Say how many multipliers there are, so a checker can tell "all zero
    // because the problem says so" from "all zero because this path does not
    // produce them yet". The mixed-integer tree and presolve-detected
    // infeasibility are both currently the second case, and a certificate that
    // hid that would be worse than no certificate.
    Int nz = 0;
    for (Int i = 0; i < nr && i < (Int)sol.rowDual.size(); ++i)
        if (sol.rowDual[i] != 0.0) ++nz;
    std::fprintf(f, "duals %d\n", (int)nz);
    if (nz == 0)
        std::fprintf(f, "# NOTE: this path does not yet produce dual multipliers, so the\n"
                        "# bound below cannot be certified. What follows certifies the\n"
                        "# primal point only.\n");
    std::fprintf(f, "# dual multipliers, one per row\n");
    for (Int i = 0; i < nr && i < (Int)sol.rowDual.size(); ++i)
        if (sol.rowDual[i] != 0.0) putValue(f, "y", i, sol.rowDual[i]);

    std::fprintf(f, "end\n");
    if (std::fclose(f) != 0) { err = "error closing " + path; return false; }
    return true;
}

} // namespace igaos
