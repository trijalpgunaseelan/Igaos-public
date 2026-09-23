// fuzz_mps.cpp -- the MPS reader against hostile input.
//
// WHY THIS EXISTS SEPARATELY FROM fuzz_cuts.cpp
// --------------------------------------------
// tools/fuzz_cuts.cpp fuzzes cut separation: it asks whether a valid cut ever
// removes an optimal point.  That is a question about MATHEMATICS, and the
// inputs it uses are always well-formed models this program generated itself.
//
// This harness asks a different question about a different attack surface.
// readMps() is the only function in the project that consumes a byte stream
// nobody here produced.  An MPS file arrives from a refinery planner, a vendor
// export, a shared drive or an upload form, and nothing upstream validates it.
// So the question here is not "is the answer right" but "what does a malformed
// or malicious file make this process do".
//
// TWO FAILURE MODES, AND THE SECOND IS THE INTERESTING ONE
// --------------------------------------------------------
//  1. It crashes.  ASan/UBSan catch that, and CI runs both.
//  2. It returns TRUE and hands back a model that is quietly wrong -- a NaN
//     coefficient, a colPtr that does not describe the matrix it indexes, a row
//     index past the end of the row array.  Nothing crashes.  The solver runs.
//     It reports OPTIMAL.  The number is meaningless and looks exactly like
//     every other number the solver has ever produced.
//
// checkModel() below is the answer to (2): every file that parses successfully
// has its model checked against the invariants the rest of the solver assumes,
// so a silent corruption fails the run instead of becoming a result.
//
// BUILD AND RUN
//
//   Standalone (any compiler), deterministic, this is what CI runs:
//       cmake --build build --target igaos_fuzz_mps
//       ./build/igaos_fuzz_mps 20000 1
//
//   libFuzzer (clang), for coverage-guided runs and a corpus:
//       clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined
//           -DIGAOS_LIBFUZZER -Iinclude tools/fuzz_mps.cpp src/*.cpp -fopenmp -o fuzz_mps
//       (one line; split here only so it fits the margin)
//       ./fuzz_mps corpus/ -max_total_time=300

#include "igaos/mps.hpp"
#include "igaos/model.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace igaos;

// ---------------------------------------------------------------------------
// The invariants every consumer of a Model takes for granted.  If readMps
// returns true, all of these hold, or the file should have been rejected.
static bool checkModel(const Model& m, std::string& why) {
    const std::int64_t n = (std::int64_t)m.obj.size();
    const std::int64_t r = (std::int64_t)m.rowLower.size();

    auto bad = [&](const std::string& s) { why = s; return false; };

    // --- shapes agree with each other -------------------------------------
    if ((std::int64_t)m.colLower.size() != n) return bad("colLower size != obj size");
    if ((std::int64_t)m.colUpper.size() != n) return bad("colUpper size != obj size");
    if ((std::int64_t)m.colType.size()  != n) return bad("colType size != obj size");
    if ((std::int64_t)m.rowUpper.size() != r) return bad("rowUpper size != rowLower size");
    if (!m.colName.empty() && (std::int64_t)m.colName.size() != n) return bad("colName size != obj size");
    if (!m.rowName.empty() && (std::int64_t)m.rowName.size() != r) return bad("rowName size != rowLower size");
    if ((std::int64_t)m.A.ncol != n) return bad("A.ncol != obj size");
    if ((std::int64_t)m.A.nrow != r) return bad("A.nrow != rowLower size");

    // --- no NaN anywhere ---------------------------------------------------
    // This is the check the "nan" literal used to walk straight past.
    for (std::int64_t j = 0; j < n; ++j) {
        if (std::isnan(m.obj[(size_t)j]))      return bad("NaN in objective");
        if (std::isnan(m.colLower[(size_t)j])) return bad("NaN in column lower bound");
        if (std::isnan(m.colUpper[(size_t)j])) return bad("NaN in column upper bound");
    }
    for (std::int64_t i = 0; i < r; ++i) {
        if (std::isnan(m.rowLower[(size_t)i])) return bad("NaN in row lower bound");
        if (std::isnan(m.rowUpper[(size_t)i])) return bad("NaN in row upper bound");
    }
    if (std::isnan(m.objOffset)) return bad("NaN in objective offset");

    // --- CSC structure is a matrix, not just three vectors -----------------
    auto csc = [&](const SparseMatrix& A, const char* tag, std::int64_t rowBound) {
        if ((std::int64_t)A.colPtr.size() != (std::int64_t)A.ncol + 1)
            return bad(std::string(tag) + ": colPtr size != ncol+1");
        if (A.colPtr.front() != 0) return bad(std::string(tag) + ": colPtr[0] != 0");
        for (size_t k = 1; k < A.colPtr.size(); ++k)
            if (A.colPtr[k] < A.colPtr[k - 1])
                return bad(std::string(tag) + ": colPtr not monotone");
        const Int nz = A.colPtr.back();
        if (nz < 0) return bad(std::string(tag) + ": negative nnz");
        if ((std::int64_t)A.rowIdx.size() < (std::int64_t)nz)
            return bad(std::string(tag) + ": rowIdx shorter than nnz");
        if ((std::int64_t)A.val.size() < (std::int64_t)nz)
            return bad(std::string(tag) + ": val shorter than nnz");
        for (Int p = 0; p < nz; ++p) {
            if (A.rowIdx[(size_t)p] < 0 || (std::int64_t)A.rowIdx[(size_t)p] >= rowBound)
                return bad(std::string(tag) + ": row index out of range");
            if (std::isnan(A.val[(size_t)p]))
                return bad(std::string(tag) + ": NaN coefficient");
        }
        return true;
    };
    if (!csc(m.A, "A", r)) return false;
    if (m.Q.ncol != 0 && !csc(m.Q, "Q", n)) return false;

    // --- quadratic row terms point at rows and columns that exist ----------
    for (const Model::QuadTerm& q : m.qcon) {
        if (q.row < 0 || (std::int64_t)q.row >= r) return bad("qcon row out of range");
        if (q.i   < 0 || (std::int64_t)q.i   >= n) return bad("qcon i out of range");
        if (q.j   < 0 || (std::int64_t)q.j   >= n) return bad("qcon j out of range");
        if (std::isnan(q.coef)) return bad("NaN in qcon coefficient");
    }
    return true;
}

// ---------------------------------------------------------------------------
static std::string scratchPath() {
    const char* t = std::getenv("TMPDIR");
    std::string dir = (t && *t) ? std::string(t) : std::string("/tmp");
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    static int counter = 0;
    std::string p = dir + "/igaos_fuzz_mps_" + std::to_string(counter++) + ".mps";
    std::ofstream probe(p);
    if (!probe) return "igaos_fuzz_mps_scratch.mps";      // fall back to cwd
    return p;
}

// Returns false only when an invariant was violated -- a rejected file and a
// cleanly parsed file are both passes.
static bool runOne(const char* data, size_t n, std::string& why) {
    static const std::string path = scratchPath();
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        if (!f) { why = "cannot open scratch file"; return false; }
        f.write(data, (std::streamsize)n);
    }
    Model m;
    std::string err;
    if (!readMps(path, m, err)) return true;              // rejection is success
    return checkModel(m, why);
}

#ifdef IGAOS_LIBFUZZER
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    std::string why;
    if (!runOne(reinterpret_cast<const char*>(data), size, why)) {
        std::fprintf(stderr, "INVARIANT VIOLATED: %s\n", why.c_str());
        std::abort();
    }
    return 0;
}
#else

// ---------------------------------------------------------------------------
// Standalone driver.  Deterministic: the same seed reproduces the same run, so
// a CI failure is replayable from the two numbers in the log.
namespace {

std::uint64_t rngState = 1;
std::uint32_t next() {                                    // xorshift64*
    rngState ^= rngState >> 12; rngState ^= rngState << 25; rngState ^= rngState >> 27;
    return (std::uint32_t)((rngState * 2685821657736338717ull) >> 32);
}
std::uint32_t below(std::uint32_t k) { return k ? next() % k : 0; }

const char* kSeeds[] = {
    // A minimal well-formed LP.
    "NAME t\nROWS\n N cost\n L c1\nCOLUMNS\n x cost 1.0 c1 2.0\nRHS\n r c1 4.0\n"
    "BOUNDS\n UP b x 10.0\nENDATA\n",
    // Ranges, an equality row, an integer marker.
    "NAME t2\nROWS\n N obj\n E e1\n G g1\nCOLUMNS\n MARKER M1 'MARKER' 'INTORG'\n"
    " y obj 3.0 e1 1.0\n MARKER M2 'MARKER' 'INTEND'\n z obj -1.0 g1 1.0 e1 2.0\n"
    "RHS\n r e1 5.0 g1 1.0\nRANGES\n rg e1 2.0\nBOUNDS\n BV b y\n FR b z\nENDATA\n",
    // Quadratic objective and a quadratically constrained row.
    "NAME q\nROWS\n N obj\n L c1\nCOLUMNS\n x obj 1.0 c1 1.0\n w obj 1.0 c1 1.0\n"
    "RHS\n r c1 3.0\nQUADOBJ\n x x 2.0\n w w 2.0\nQCMATRIX c1\n x w 0.5\n w x 0.5\nENDATA\n",
    // Free-form with a spaced name, which forces the fixed-column reader.
    "NAME  f\nROWS\n N  obj\n L  DEDO3 1R\nCOLUMNS\n    DEDO3 11  obj       1.0"
    "        DEDO3 1R  2.0\nRHS\n    r         DEDO3 1R  9.0\nENDATA\n",
    "NAME\nROWS\nCOLUMNS\nENDATA\n",
    "",
};
const size_t kNumSeeds = sizeof(kSeeds) / sizeof(kSeeds[0]);

// The tokens that made this harness worth writing.  Each one is legal input to
// std::stod and each one used to reach the constraint matrix intact.
const char* kHostile[] = {
    "nan", "NaN", "NAN", "nan(0x1)", "inf", "-inf", "INF", "Infinity", "-Infinity",
    "1e999", "-1e999", "1e-999", "0x1p3", "1.5D+02", "1d400",
    "2147483648", "-2147483649", "4294967296", "9223372036854775808",
    "1e30", "1e31", "-1e31", "", " ", "\t", "--1", "1..2", "+-3",
};
const size_t kNumHostile = sizeof(kHostile) / sizeof(kHostile[0]);

void mutate(std::string& s) {
    switch (below(8)) {
        case 0: {                                          // flip a byte
            if (s.empty()) break;
            s[below((std::uint32_t)s.size())] = (char)below(256);
            break;
        }
        case 1: {                                          // truncate
            if (s.empty()) break;
            s.resize(below((std::uint32_t)s.size()));
            break;
        }
        case 2: {                                          // splice a hostile token in
            const char* h = kHostile[below((std::uint32_t)kNumHostile)];
            size_t at = s.empty() ? 0 : below((std::uint32_t)s.size());
            s.insert(at, h);
            break;
        }
        case 3: {                                          // replace a number with a hostile one
            size_t at = s.find_first_of("0123456789");
            if (at == std::string::npos) break;
            size_t end = s.find_first_not_of("0123456789.eE+-", at);
            if (end == std::string::npos) end = s.size();
            s.replace(at, end - at, kHostile[below((std::uint32_t)kNumHostile)]);
            break;
        }
        case 4: {                                          // duplicate a line many times
            size_t nl = s.find('\n');
            if (nl == std::string::npos) break;
            std::string line = s.substr(0, nl + 1);
            std::string out;
            std::uint32_t reps = 1 + below(400);
            out.reserve(line.size() * reps + s.size());
            for (std::uint32_t k = 0; k < reps; ++k) out += line;
            s = out + s;
            break;
        }
        case 5: {                                          // one very long token
            size_t at = s.empty() ? 0 : below((std::uint32_t)s.size());
            s.insert(at, std::string(1 + below(20000), 'A'));
            break;
        }
        case 6: {                                          // drop a line
            size_t a = s.empty() ? 0 : below((std::uint32_t)s.size());
            size_t b = s.find('\n', a);
            s.erase(a, (b == std::string::npos ? s.size() : b + 1) - a);
            break;
        }
        default: {                                         // strip all newlines
            std::string out;
            for (char c : s) if (c != '\n') out.push_back(c);
            s = out;
            break;
        }
    }
}

// Files aimed straight at the defects this harness was written for.  These run
// first, every time, whatever the seed -- they are regression tests wearing a
// fuzzer's clothes.
int targeted() {
    struct Case { const char* what; std::string body; };
    std::vector<Case> cases;
    const char* head = "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n";
    for (size_t k = 0; k < kNumHostile; ++k) {
        cases.push_back({kHostile[k],
            std::string(head) + " x obj " + kHostile[k] + " c1 " + kHostile[k] +
            "\nRHS\n r c1 " + kHostile[k] + "\nBOUNDS\n UP b x " + kHostile[k] + "\nENDATA\n"});
    }
    cases.push_back({"quadobj nan", "NAME q\nROWS\n N obj\n L c1\nCOLUMNS\n x obj 1 c1 1\n"
                                    "RHS\n r c1 3\nQUADOBJ\n x x nan\nENDATA\n"});
    cases.push_back({"qcmatrix unknown row", "NAME q\nROWS\n N obj\n L c1\nCOLUMNS\n x obj 1 c1 1\n"
                                             "QCMATRIX nosuchrow\n x x 1.0\nENDATA\n"});
    cases.push_back({"no endata", "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n x obj 1 c1 1\n"});
    cases.push_back({"section only", "COLUMNS\n"});
    {   // Embedded NUL bytes: std::getline stops at '\n', not at '\0', so these
        // travel into the tokenizer inside a std::string whose size disagrees
        // with strlen().  Built rather than written as a sized literal, so the
        // length can never drift out of step with the text.
        std::string nul = "NAME t\nROWS\n N obj";
        nul.push_back('\0');
        nul.push_back('\0');
        nul += "\n L c1\nCOLUMNS\n x obj 1 c1 1\nENDATA\n";
        cases.push_back({"nul bytes", nul});
    }
    cases.push_back({"fortran exponent", "NAME t\nROWS\n N obj\n L c1\nCOLUMNS\n x obj 1.5D+02 c1 1\n"
                                         "RHS\n r c1 3\nENDATA\n"});

    int failures = 0;
    for (const Case& c : cases) {
        std::string why;
        if (!runOne(c.body.data(), c.body.size(), why)) {
            std::fprintf(stderr, "  FAIL  targeted [%s]: %s\n", c.what, why.c_str());
            ++failures;
        }
    }
    std::printf("  targeted cases ......... %zu run, %d failed\n", cases.size(), failures);
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    const long iters = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 20000;
    const unsigned long seed = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1;
    rngState = seed ? seed : 1;

    std::printf("igaos_fuzz_mps -- MPS reader against hostile input\n");
    std::printf("  iterations %ld, seed %lu\n", iters, seed);

    int failures = targeted();

    for (long k = 0; k < iters; ++k) {
        std::string s = kSeeds[below((std::uint32_t)kNumSeeds)];
        const std::uint32_t rounds = 1 + below(4);
        for (std::uint32_t t = 0; t < rounds; ++t) mutate(s);
        if (s.size() > (4u << 20)) s.resize(4u << 20);     // keep the harness itself bounded
        std::string why;
        if (!runOne(s.data(), s.size(), why)) {
            std::fprintf(stderr, "  FAIL  iteration %ld (seed %lu): %s\n", k, seed, why.c_str());
            std::fprintf(stderr, "  ---- input ----\n%.2000s\n  ---------------\n", s.c_str());
            if (++failures > 20) break;
        }
    }

    std::printf("  mutation cases ......... %ld run\n", iters);
    std::printf("%s\n", failures ? "FUZZ FAILED" : "ALL CHECKS PASSED");
    return failures ? 1 : 0;
}
#endif
