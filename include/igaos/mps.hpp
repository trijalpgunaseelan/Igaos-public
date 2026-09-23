// mps.hpp : MPS (fixed and free form) and LP-format readers/writers.
// Handles RANGES, BOUNDS (all types), integer MARKERs, OBJSENSE, and the
// QUADOBJ / QMATRIX sections used by QPLIB-style quadratic instances.
//
// AN MPS FILE IS UNTRUSTED INPUT.
// It arrives from a planner, a vendor tool, a shared drive or a web form, and
// nothing upstream of this reader validates it.  The reader is therefore
// written to fail on a hostile file rather than to trust it, and MpsLimits
// below is the contract: every unbounded quantity in the format has a ceiling,
// and every ceiling is a caller-visible number rather than whatever the
// machine happens to run out of first.  See SECURITY.md and
// docs/THREAT-MODEL.md.
#pragma once
#include "igaos/model.hpp"
#include <cstddef>
#include <cstdint>

namespace igaos {

// Ceilings applied while reading an MPS file.  The defaults are far above any
// instance in Netlib, MIPLIB, Maros-Meszaros or QPLIB -- the largest model this
// project has solved is 403,200 x 1,075,200 with 2,150,080 nonzeros, three
// orders of magnitude inside every limit here -- so a legitimate file never
// meets one.  They exist so that a malicious file meets a defined error
// instead of the allocator.
//
// Tighten them when the caller knows the shape of what it should be reading; a
// service accepting uploads should.
struct MpsLimits {
    // Whole-file ceiling, checked before a byte is parsed.  4 GiB.
    std::uint64_t maxFileBytes  = 4ull << 30;
    // A single line.  The format's own fixed-form record is 61 columns; 1 MiB
    // is generous for free-form writers and still bounds one allocation.
    std::size_t   maxLineBytes  = 1u << 20;
    // A single row, column or set name.
    std::size_t   maxNameBytes  = 8192;
    // Structural ceilings.  Int is int32_t, so these also keep every index and
    // every colPtr[n]+1 arithmetic well inside the type -- the narrowing casts
    // from size_t to Int in the reader are checked against these before they
    // happen rather than after.
    std::int64_t  maxRows       = 50000000;      // 5e7
    std::int64_t  maxCols       = 50000000;      // 5e7
    std::int64_t  maxNonzeros   = 400000000;     // 4e8, and 4e8 < INT32_MAX
    std::int64_t  maxQuadTerms  = 400000000;
};

// Reads an MPS file.  Returns false and sets err on any malformed file, on any
// file that exceeds limits, and on any file containing a non-finite
// coefficient.  On failure model is not usable and must not be read.
bool readMps(const std::string& path, Model& model, std::string& err);
bool readMps(const std::string& path, Model& model, std::string& err,
             const MpsLimits& limits);

bool writeMps(const std::string& path, const Model& model, std::string& err);
bool writeSolution(const std::string& path, const Model& m, const Solution& s, std::string& err);

} // namespace igaos
