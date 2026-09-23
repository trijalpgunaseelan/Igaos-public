#include "igaos/mps.hpp"
#include <map>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>

namespace igaos {
namespace {

inline void splitTokens(const std::string& line, std::vector<std::string>& out) {
    out.clear();
    const char* p = line.c_str();
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p) break;
        const char* s = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') ++p;
        out.emplace_back(s, p - s);
    }
}

inline bool isSectionLine(const std::string& l) {
    return !l.empty() && l[0] != ' ' && l[0] != '\t' && l[0] != '*';
}

// ---------------------------------------------------------------------------
// Fixed-column MPS.
//
// The format predates free-form text: its fields live at fixed character
// positions, and a name may therefore contain spaces. Real files in the
// canonical collections do exactly that -- QFORPLAN in the Maros and Meszaros
// set has columns called "DEDO3 11" and rows called "DEDO3 1R" -- and splitting
// such a line on whitespace silently produces a different model. This solver
// read that file as INFEASIBLE, which is the worst way to be wrong: a confident
// verdict on a problem it had misread.
//
//   field    1      2       3       4        5       6
//   columns 2-3    5-12   15-22   25-36    40-47   50-61
//
// Free-form tokenizing stays the default, because most modern writers emit it
// and it tolerates names longer than eight characters. The fixed reader is used
// only when the free-form split yields more fields than the section can hold,
// which is an unambiguous signal that a name contained a space.
inline void splitFixed(const std::string& line, std::vector<std::string>& out) {
    static const int beg[6] = { 1,  4, 14, 24, 39, 49 };
    static const int len[6] = { 2,  8,  8, 12,  8, 12 };
    out.clear();
    for (int f = 0; f < 6; ++f) {
        if ((int)line.size() <= beg[f]) break;
        std::string t = line.substr(beg[f], std::min<size_t>(len[f], line.size() - beg[f]));
        size_t a = t.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) continue;
        size_t b = t.find_last_not_of(" \t\r\n");
        out.push_back(t.substr(a, b - a + 1));
    }
}

// The largest number of fields a data line in each section can legitimately
// carry when it is free-form. Anything above this means a name held a space.
//
// This is a whole-file verdict, not a per-line one, and that matters. An RHS
// line for a row called "DEDO3 1R" splits into four fields, which is a legal
// count -- the spaced name hides inside it. Only the COLUMNS lines give the
// format away by overflowing. So the file is scanned once for any overflowing
// line anywhere, and if one is found every data line is read by position.
inline int maxFreeFields(int sec) {
    switch (sec) {
        case 2:  return 2;    // ROWS
        case 3:  return 5;    // COLUMNS  (name, row, val, row, val)
        case 4:  return 5;    // RHS
        case 5:  return 5;    // RANGES
        case 6:  return 4;    // BOUNDS
        case 7:  return 3;    // QUADOBJ
        case 8:  return 3;    // QCMATRIX
        default: return 1 << 30;
    }
}

// Parses one numeric field.
//
// TWO THINGS THIS REJECTS THAT std::stod ACCEPTS, AND WHY.
//
// "nan", "NAN", "nan(0x1)", "inf", "infinity" are all valid input to std::stod
// and all of them used to reach the constraint matrix.  A NaN coefficient is
// the worst possible corruption of a solver: every comparison against NaN is
// false, so the feasibility test does not fail, the ratio test does not fail,
// and the run ends by reporting OPTIMAL on a model whose answer is undefined.
// It is the exact failure mode this reader already guards against elsewhere --
// a confident verdict on a problem it had misread -- so a non-finite literal is
// a hard parse error, not a value.
//
// This costs nothing on real files.  MPS predates IEEE 754 naming and no writer
// in Netlib, MIPLIB, Maros-Meszaros or QPLIB emits either token; a file meaning
// "infinite bound" writes 1e30 or 1e31, which is finite here and is turned into
// an infinity by isInf() downstream exactly as before.
//
// Second: a Fortran-style exponent -- 1.5D+02 -- is normalized to 1.5E+02
// rather than silently truncated to 1.5 at the D.  Old MPS writers emit it and
// std::stod stops at the D without complaining, which turns 150 into 1.5.
// Two different failures, and only one of them is survivable.
//
//   ok = false, poisoned = false
//       The token is not a number at all.  This is ORDINARY: an RHS or RANGES
//       line may lead with a set name, and the reader identifies the set name
//       by failing to parse it.  The caller skips the field, as it always has.
//
//   poisoned = true
//       The token IS a number and the number cannot be used -- "nan", "inf",
//       "1e999".  This is NOT survivable and must not be skipped.  Skipping it
//       drops a coefficient and solves a DIFFERENT MODEL than the file
//       describes, then reports the answer as optimal: the file said the
//       constraint had a term and the solve behaved as though it did not.
//       Corruption that changes the answer has to stop the read, so the caller
//       turns this into a parse error and the whole file is refused.
inline Real parseReal(const std::string& t, bool& ok, bool& poisoned) {
    ok = false;
    if (t.empty() || t.size() > 512) return 0.0;
    std::string b;
    b.reserve(t.size());
    for (char c : t) b.push_back((c == 'D' || c == 'd') ? 'E' : c);
    Real v = 0.0;
    try {
        size_t pos = 0;
        v = std::stod(b, &pos);
    } catch (const std::out_of_range&) {
        poisoned = true;                    // 1e999: a magnitude no double holds
        return 0.0;
    } catch (...) {
        return 0.0;                         // not a number; a name, most likely
    }
    if (!std::isfinite(v)) { poisoned = true; return 0.0; }   // nan, inf
    ok = true;
    return v;
}

} // namespace

// ---------------------------------------------------------------------------
bool readMps(const std::string& path, Model& model, std::string& err) {
    return readMps(path, model, err, MpsLimits());
}

bool readMps(const std::string& path, Model& model, std::string& err,
             const MpsLimits& lim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { err = "cannot open " + path; return false; }

    // Whole-file ceiling first, before a byte is parsed.  Every allocation
    // below is bounded by this one even where a finer limit only rejects the
    // file after the read that tripped it.
    in.seekg(0, std::ios::end);
    std::streamoff fileBytes = in.tellg();
    if (fileBytes < 0) { err = "cannot size " + path; return false; }
    if ((std::uint64_t)fileBytes > lim.maxFileBytes) {
        err = "file exceeds maxFileBytes (" + std::to_string((std::uint64_t)fileBytes) +
              " > " + std::to_string(lim.maxFileBytes) + ")";
        return false;
    }
    in.seekg(0, std::ios::beg);

    // Set by any limit breach anywhere below, including inside the lambdas.
    // Checked after every section so a hostile file stops at the first ceiling
    // it touches rather than at the last.
    std::string limitErr;
    auto over = [&](const char* what, std::int64_t have, std::int64_t cap) {
        if (limitErr.empty())
            limitErr = std::string("input exceeds ") + what + " (" +
                       std::to_string(have) + " > " + std::to_string(cap) + ")";
    };

    enum Sec { NONE, NAME_, ROWS, COLUMNS, RHS, RANGES, BOUNDS, QUAD, QCON, OBJSENSE_, ENDATA };
    Sec sec = NONE;

    std::unordered_map<std::string, Int>  rowIndex, colIndex;
    std::vector<char>  rowKind;                  // 'N','L','G','E'
    std::vector<std::string> rowNames;
    Int objRow = kNone;

    struct Entry { Int row, col; Real val; };
    std::vector<Entry> entries, qentries;
    struct QconEntry { std::string row; Int i, j; Real val; };
    std::vector<QconEntry> qconEntries;
    std::string qconRow;
    std::vector<Real> rhsVals, rngVals;
    std::vector<Real> objCoef;
    std::vector<Real> lbs, ubs;
    std::vector<char> isInt;
    std::vector<char> lbSet, ubSet;
    std::vector<std::string> colNames;
    Real objConst = 0.0;
    Sense sense = Sense::Minimize;
    bool intBlock = false;
    std::string modelName = "mps_model";

    std::string line;
    std::vector<std::string> tk;
    Int lineNo = 0;

    // ---- pass one: free-form or fixed-column? -----------------------------
    bool useFixed = false;
    {
        int probeSec = NONE;
        std::vector<std::string> ptk;
        while (std::getline(in, line)) {
            if (line.size() > lim.maxLineBytes) {
                err = "line exceeds maxLineBytes (" + std::to_string(line.size()) +
                      " > " + std::to_string(lim.maxLineBytes) + ")";
                return false;
            }
            if (line.empty() || line[0] == '*') continue;
            if (isSectionLine(line)) {
                splitTokens(line, ptk);
                if (ptk.empty()) continue;
                std::string s2 = ptk[0];
                for (char& c : s2) c = (char)std::toupper((unsigned char)c);
                if      (s2 == "ROWS")    probeSec = ROWS;
                else if (s2 == "COLUMNS") probeSec = COLUMNS;
                else if (s2 == "RHS")     probeSec = RHS;
                else if (s2 == "RANGES")  probeSec = RANGES;
                else if (s2 == "BOUNDS")  probeSec = BOUNDS;
                else if (s2 == "QUADOBJ" || s2 == "QMATRIX" || s2 == "QSECTION") probeSec = QUAD;
                else if (s2 == "QCMATRIX" || s2 == "QCONSTRAINT") probeSec = QCON;
                else if (s2 == "ENDATA")  break;
                else probeSec = NONE;
                continue;
            }
            splitTokens(line, ptk);
            if (ptk.empty()) continue;
            if (probeSec == COLUMNS && ptk.size() >= 3 &&
                (ptk[1] == "'MARKER'" || ptk[2] == "'MARKER'")) continue;
            if ((int)ptk.size() > maxFreeFields(probeSec)) { useFixed = true; break; }
        }
        in.clear();
        in.seekg(0);
    }

    auto ensureCol = [&](const std::string& nm) -> Int {
        auto it = colIndex.find(nm);
        if (it != colIndex.end()) return it->second;
        if (nm.size() > lim.maxNameBytes) {
            over("maxNameBytes", (std::int64_t)nm.size(), (std::int64_t)lim.maxNameBytes);
            return kNone;
        }
        // The cast below is the one that matters: Int is int32_t and
        // colNames.size() is a size_t.  Past INT32_MAX columns it wraps
        // negative and every later objCoef[j], lbs[j], isInt[j] is an
        // out-of-bounds write.  maxCols is checked BEFORE the cast, not after.
        if ((std::int64_t)colNames.size() >= lim.maxCols) {
            over("maxCols", (std::int64_t)colNames.size() + 1, lim.maxCols);
            return kNone;
        }
        Int j = (Int)colNames.size();
        colIndex[nm] = j;
        colNames.push_back(nm);
        objCoef.push_back(0.0);
        lbs.push_back(0.0); ubs.push_back(kInf);
        lbSet.push_back(0); ubSet.push_back(0);
        isInt.push_back(intBlock ? 1 : 0);
        return j;
    };

    while (std::getline(in, line)) {
        ++lineNo;
        if (!limitErr.empty()) { err = limitErr + " at line " + std::to_string(lineNo); return false; }
        if (line.size() > lim.maxLineBytes) {
            err = "line " + std::to_string(lineNo) + " exceeds maxLineBytes (" +
                  std::to_string(line.size()) + " > " + std::to_string(lim.maxLineBytes) + ")";
            return false;
        }
        if (line.empty() || line[0] == '*') continue;
        if (isSectionLine(line)) {
            splitTokens(line, tk);
            if (tk.empty()) continue;
            std::string s = tk[0];
            for (char& c : s) c = (char)std::toupper((unsigned char)c);
            if      (s == "NAME")     { sec = NAME_; if (tk.size() > 1) modelName = tk[1]; }
            else if (s == "ROWS")     sec = ROWS;
            else if (s == "COLUMNS")  sec = COLUMNS;
            else if (s == "RHS")      sec = RHS;
            else if (s == "RANGES")   sec = RANGES;
            else if (s == "BOUNDS")   sec = BOUNDS;
            else if (s == "QUADOBJ" || s == "QMATRIX" || s == "QSECTION") sec = QUAD;
            // QCMATRIX names the row it belongs to on its own header line, so a
            // file can carry one section per quadratically constrained row.
            // This is the CPLEX and Gurobi convention and it is what makes a
            // pooling model expressible as a file rather than only in code.
            else if (s == "QCMATRIX" || s == "QCONSTRAINT") {
                sec = QCON;
                qconRow = tk.size() > 1 ? tk[1] : std::string();
            }
            else if (s == "OBJSENSE") {
                sec = OBJSENSE_;
                if (tk.size() > 1) {
                    std::string v = tk[1];
                    for (char& c : v) c = (char)std::toupper((unsigned char)c);
                    if (v == "MAX" || v == "MAXIMIZE") sense = Sense::Maximize;
                }
            }
            else if (s == "ENDATA")   { sec = ENDATA; break; }
            else sec = NONE;
            continue;
        }

        if (useFixed) {
            splitTokens(line, tk);
            bool marker = (sec == COLUMNS && tk.size() >= 3 &&
                           (tk[1] == "'MARKER'" || tk[2] == "'MARKER'"));
            if (!marker) splitFixed(line, tk);
        } else {
            splitTokens(line, tk);
        }
        if (tk.empty()) continue;
        bool ok = true;
        bool poisoned = false;

        switch (sec) {
        case OBJSENSE_: {
            std::string v = tk[0];
            for (char& c : v) c = (char)std::toupper((unsigned char)c);
            if (v == "MAX" || v == "MAXIMIZE") sense = Sense::Maximize;
            break;
        }
        case ROWS: {
            if (tk.size() < 2) { err = "malformed ROWS line " + std::to_string(lineNo); return false; }
            char k = (char)std::toupper((unsigned char)tk[0][0]);
            const std::string& nm = tk[1];
            if (k == 'N' && objRow == kNone) {
                objRow = -2;                       // objective, not a constraint
                rowIndex[nm] = kNone;
                continue;
            }
            if (nm.size() > lim.maxNameBytes) {
                err = "row name at line " + std::to_string(lineNo) + " exceeds maxNameBytes";
                return false;
            }
            if ((std::int64_t)rowKind.size() >= lim.maxRows) {   // before the narrowing cast
                err = "input exceeds maxRows (" + std::to_string(lim.maxRows) + ")";
                return false;
            }
            Int i = (Int)rowKind.size();
            rowIndex[nm] = i;
            rowKind.push_back(k);
            rowNames.push_back(nm);
            rhsVals.push_back(0.0);
            rngVals.push_back(0.0);
            break;
        }
        case COLUMNS: {
            if (tk.size() >= 3 && tk[1] == "'MARKER'") {
                std::string v = tk.size() > 2 ? tk[2] : "";
                if (v.find("INTORG") != std::string::npos) intBlock = true;
                if (v.find("INTEND") != std::string::npos) intBlock = false;
                continue;
            }
            if (tk.size() >= 3 &&
                (tk[2] == "'MARKER'" || tk[1] == "MARKER")) {
                for (const auto& t : tk) {
                    if (t.find("INTORG") != std::string::npos) intBlock = true;
                    if (t.find("INTEND") != std::string::npos) intBlock = false;
                }
                continue;
            }
            if (tk.size() < 3) break;
            Int j = ensureCol(tk[0]);
            if (j == kNone) break;                 // limit hit; caught at the top of the loop
            if (intBlock) isInt[j] = 1;
            for (size_t t = 1; t + 1 < tk.size(); t += 2) {
                auto it = rowIndex.find(tk[t]);
                if (it == rowIndex.end()) continue;
                Real v = parseReal(tk[t + 1], ok, poisoned);
                if (!ok) continue;
                if (it->second == kNone) { objCoef[j] += v; continue; }
                if ((std::int64_t)entries.size() >= lim.maxNonzeros) {
                    over("maxNonzeros", (std::int64_t)entries.size() + 1, lim.maxNonzeros);
                    break;
                }
                entries.push_back({it->second, j, v});
            }
            break;
        }
        case RHS: {
            size_t start = (tk.size() % 2 == 1) ? 1 : 0;   // optional RHS set name
            if (rowIndex.count(tk[0])) start = 0;
            for (size_t t = start; t + 1 < tk.size(); t += 2) {
                auto it = rowIndex.find(tk[t]);
                if (it == rowIndex.end()) continue;
                Real v = parseReal(tk[t + 1], ok, poisoned);
                if (!ok) continue;
                if (it->second == kNone) objConst = -v;    // MPS convention
                else rhsVals[it->second] = v;
            }
            break;
        }
        case RANGES: {
            size_t start = (tk.size() % 2 == 1) ? 1 : 0;
            if (rowIndex.count(tk[0])) start = 0;
            for (size_t t = start; t + 1 < tk.size(); t += 2) {
                auto it = rowIndex.find(tk[t]);
                if (it == rowIndex.end() || it->second == kNone) continue;
                Real v = parseReal(tk[t + 1], ok, poisoned);
                if (ok) rngVals[it->second] = v;
            }
            break;
        }
        case BOUNDS: {
            if (tk.size() < 2) break;
            std::string type = tk[0];
            for (char& c : type) c = (char)std::toupper((unsigned char)c);
            // Layout: TYPE  BNDNAME  COLNAME  [VALUE]
            size_t ci = 2, vi = 3;
            if (!colIndex.count(tk.size() > 2 ? tk[2] : "") && colIndex.count(tk[1])) { ci = 1; vi = 2; }
            if (ci >= tk.size()) break;
            Int j = ensureCol(tk[ci]);
            if (j == kNone) break;
            Real v = 0.0;
            if (vi < tk.size()) v = parseReal(tk[vi], ok, poisoned);
            if (type == "UP")      { ubs[j] = v; ubSet[j] = 1; if (v < 0 && !lbSet[j]) lbs[j] = -kInf; }
            else if (type == "LO") { lbs[j] = v; lbSet[j] = 1; }
            else if (type == "FX") { lbs[j] = ubs[j] = v; lbSet[j] = ubSet[j] = 1; }
            else if (type == "FR") { lbs[j] = -kInf; ubs[j] = kInf; lbSet[j] = ubSet[j] = 1; }
            else if (type == "MI") { lbs[j] = -kInf; lbSet[j] = 1; }
            else if (type == "PL") { ubs[j] =  kInf; ubSet[j] = 1; }
            else if (type == "BV") { lbs[j] = 0; ubs[j] = 1; isInt[j] = 1; lbSet[j] = ubSet[j] = 1; }
            else if (type == "LI") { lbs[j] = v; isInt[j] = 1; lbSet[j] = 1; }
            else if (type == "UI") { ubs[j] = v; isInt[j] = 1; ubSet[j] = 1; }
            break;
        }
        case QUAD: {
            if (tk.size() < 3) break;
            Int i = ensureCol(tk[0]), j = ensureCol(tk[1]);
            if (i == kNone || j == kNone) break;
            if ((std::int64_t)qentries.size() >= lim.maxQuadTerms) {
                over("maxQuadTerms", (std::int64_t)qentries.size() + 1, lim.maxQuadTerms);
                break;
            }
            Real v = parseReal(tk[2], ok, poisoned);
            if (ok) qentries.push_back({i, j, v});
            break;
        }
        case QCON: {
            if (tk.size() < 3 || qconRow.empty()) break;
            Int i = ensureCol(tk[0]), j = ensureCol(tk[1]);
            if (i == kNone || j == kNone) break;
            if ((std::int64_t)qconEntries.size() >= lim.maxQuadTerms) {
                over("maxQuadTerms", (std::int64_t)qconEntries.size() + 1, lim.maxQuadTerms);
                break;
            }
            Real v = parseReal(tk[2], ok, poisoned);
            // The section is written symmetrically -- (i,j) and (j,i) each carry
            // half the coefficient -- so entries are SUMMED rather than
            // deduplicated.  Reading only one triangle would halve every
            // off-diagonal term in the file, which is a silent factor of two in
            // a quality balance.
            if (ok) qconEntries.push_back({qconRow, i, j, v});
            break;
        }
        default: break;
        }

        // A non-finite or unrepresentable literal anywhere on the line refuses
        // the whole file.  Deliberately after the switch rather than inside
        // each section, so no section can be added later that forgets it.
        if (poisoned) {
            err = "non-finite or unrepresentable numeric field at line " +
                  std::to_string(lineNo) +
                  " -- NaN and infinity are corruption, not values; refusing " + path;
            return false;
        }
    }

    if (!limitErr.empty()) { err = limitErr; return false; }
    if (colNames.empty()) { err = "no columns found in " + path; return false; }

    // Belt and braces on the two narrowing casts below.  Both counters were
    // capped as they grew; this asserts the invariant at the point of use, so a
    // future edit that adds a growth path without a cap fails here loudly
    // rather than wrapping a size_t into a negative Int.
    if ((std::int64_t)colNames.size() > lim.maxCols ||
        (std::int64_t)rowKind.size()  > lim.maxRows ||
        colNames.size() > (size_t)std::numeric_limits<Int>::max() ||
        rowKind.size()  > (size_t)std::numeric_limits<Int>::max()) {
        err = "model dimensions exceed the Int index range";
        return false;
    }

    // Assemble the model in ranged-row form.
    model = Model();
    model.name = modelName;
    // Keep the file's own sense instead of folding a maximization into a
    // negated minimization.  The solver handles Sense::Maximize natively, and
    // normalizing here would flip the sign of every objective value the caller
    // reads back -- so a model that went out through writeMps would come back
    // reporting the negative of its own optimum.
    model.sense = sense;
    Int n = (Int)colNames.size(), m = (Int)rowKind.size();
    model.obj.resize(n); model.colLower.resize(n); model.colUpper.resize(n);
    model.colType.resize(n); model.colName = colNames;
    const Real sgn = 1.0;
    for (Int j = 0; j < n; ++j) {
        model.obj[j] = sgn * objCoef[j];
        // MPS default for an integer column with no explicit bounds is [0,1]
        // in the original standard but [0,inf) in most modern readers; follow
        // the modern convention, which MIPLIB instances assume.
        model.colLower[j] = lbs[j];
        model.colUpper[j] = ubs[j];
        model.colType[j]  = isInt[j] ? VarType::Integer : VarType::Continuous;
        if (isInt[j] && isFinite(lbs[j]) && isFinite(ubs[j]) &&
            lbs[j] >= -1e-9 && ubs[j] <= 1.0 + 1e-9)
            model.colType[j] = VarType::Binary;
    }
    model.objOffset = sgn * objConst;
    model.rowLower.resize(m); model.rowUpper.resize(m); model.rowName = rowNames;
    for (Int i = 0; i < m; ++i) {
        Real b = rhsVals[i], r = rngVals[i];
        switch (rowKind[i]) {
        case 'L': model.rowLower[i] = -kInf; model.rowUpper[i] = b; break;
        case 'G': model.rowLower[i] = b;     model.rowUpper[i] = kInf; break;
        case 'E': model.rowLower[i] = b;     model.rowUpper[i] = b; break;
        default:  model.rowLower[i] = -kInf; model.rowUpper[i] = kInf; break;
        }
        if (r != 0.0) {                                   // RANGES semantics
            Real a = std::fabs(r);
            switch (rowKind[i]) {
            case 'L': model.rowLower[i] = b - a; break;
            case 'G': model.rowUpper[i] = b + a; break;
            case 'E': if (r > 0) { model.rowLower[i] = b; model.rowUpper[i] = b + a; }
                      else       { model.rowLower[i] = b - a; model.rowUpper[i] = b; }
                      break;
            default: break;
            }
        }
    }
    for (const Entry& e : entries) model.setElement(e.row, e.col, e.val);
    for (const Entry& e : qentries) model.setQuadratic(e.row, e.col, sgn * e.val);
    // Row quadratics.  The row is named, not numbered, so it is looked up here
    // rather than while parsing -- a QCMATRIX section may legally appear before
    // the row it refers to has been seen in COLUMNS.
    if (!qconEntries.empty()) {
        std::map<std::string, Int> rowIndex;
        for (Int i = 0; i < (Int)rowNames.size(); ++i) rowIndex[rowNames[(size_t)i]] = i;
        for (const QconEntry& e : qconEntries) {
            auto it = rowIndex.find(e.row);
            if (it == rowIndex.end()) {
                err = "QCMATRIX names row '" + e.row + "', which is not in ROWS";
                return false;
            }
            model.addQuadraticTerm(it->second, e.i, e.j, e.val);
        }
    }
    model.finalize();
    model.A.nrow = m; model.A.ncol = n;
    if ((Int)model.A.colPtr.size() != n + 1) model.A.colPtr.resize(n + 1, model.A.nnz());
    return true;
}

// ---------------------------------------------------------------------------
bool writeMps(const std::string& path, const Model& mIn, std::string& err) {
    Model m = mIn;
    m.ensureNames();
    std::ofstream f(path);
    if (!f) { err = "cannot write " + path; return false; }
    f.setf(std::ios::scientific); f.precision(12);
    f << "NAME          " << m.name << "\n";
    // Without this section a maximization is indistinguishable from a
    // minimization with the same coefficients, and the round trip silently
    // returns the wrong optimum.
    if (m.sense == Sense::Maximize) f << "OBJSENSE\n    MAX\n";
    f << "ROWS\n N  COST\n";
    for (Int i = 0; i < m.numRow(); ++i) {
        bool lf = isFinite(m.rowLower[i]), uf = isFinite(m.rowUpper[i]);
        char k = 'N';
        if (lf && uf) k = (m.rowLower[i] == m.rowUpper[i]) ? 'E' : 'L';
        else if (uf)  k = 'L';
        else if (lf)  k = 'G';
        f << " " << k << "  " << m.rowName[i] << "\n";
    }
    f << "COLUMNS\n";
    bool inInt = false; int mk = 0;
    for (Int j = 0; j < m.numCol(); ++j) {
        bool wantInt = m.colType[j] != VarType::Continuous;
        if (wantInt && !inInt) { f << "    MARKER" << mk++ << "  'MARKER'  'INTORG'\n"; inInt = true; }
        if (!wantInt && inInt) { f << "    MARKER" << mk++ << "  'MARKER'  'INTEND'\n"; inInt = false; }
        if (m.obj[j] != 0.0) f << "    " << m.colName[j] << "  COST  " << m.obj[j] << "\n";
        for (Int p = m.A.colPtr[j]; p < m.A.colPtr[j + 1]; ++p)
            f << "    " << m.colName[j] << "  " << m.rowName[m.A.rowIdx[p]]
              << "  " << m.A.val[p] << "\n";
    }
    if (inInt) f << "    MARKER" << mk++ << "  'MARKER'  'INTEND'\n";
    f << "RHS\n";
    for (Int i = 0; i < m.numRow(); ++i) {
        bool lf = isFinite(m.rowLower[i]), uf = isFinite(m.rowUpper[i]);
        Real b = uf ? m.rowUpper[i] : (lf ? m.rowLower[i] : 0.0);
        if (lf || uf) f << "    RHS  " << m.rowName[i] << "  " << b << "\n";
    }
    f << "RANGES\n";
    for (Int i = 0; i < m.numRow(); ++i)
        if (isFinite(m.rowLower[i]) && isFinite(m.rowUpper[i]) &&
            m.rowLower[i] != m.rowUpper[i])
            f << "    RNG  " << m.rowName[i] << "  " << (m.rowUpper[i] - m.rowLower[i]) << "\n";
    f << "BOUNDS\n";
    for (Int j = 0; j < m.numCol(); ++j) {
        Real l = m.colLower[j], u = m.colUpper[j];
        if (l == 0.0 && isInf(u)) continue;
        if (isNegInf(l) && isInf(u)) { f << " FR BND  " << m.colName[j] << "\n"; continue; }
        if (l == u) { f << " FX BND  " << m.colName[j] << "  " << l << "\n"; continue; }
        if (isNegInf(l)) f << " MI BND  " << m.colName[j] << "\n";
        else if (l != 0.0) f << " LO BND  " << m.colName[j] << "  " << l << "\n";
        if (isFinite(u)) f << " UP BND  " << m.colName[j] << "  " << u << "\n";
    }
    if (m.Q.nnz() > 0) {
        f << "QUADOBJ\n";
        for (Int j = 0; j < m.Q.ncol; ++j)
            for (Int p = m.Q.colPtr[j]; p < m.Q.colPtr[j + 1]; ++p)
                f << "    " << m.colName[m.Q.rowIdx[p]] << "  " << m.colName[j]
                  << "  " << m.Q.val[p] << "\n";
    }
    if (!m.qcon.empty()) {
        // One section per row, written SYMMETRICALLY: an off-diagonal term
        // appears as two half-coefficients, which is what a reader following
        // the same convention will sum back to the original.  Writing one
        // triangle would round-trip to half the model.
        std::map<Int, std::vector<const Model::QuadTerm*>> byRow;
        for (const Model::QuadTerm& t : m.qcon) byRow[t.row].push_back(&t);
        for (const auto& e : byRow) {
            f << "QCMATRIX  " << m.rowName[e.first] << "\n";
            for (const Model::QuadTerm* t : e.second) {
                if (t->i == t->j) {
                    f << "    " << m.colName[t->i] << "  " << m.colName[t->j]
                      << "  " << t->coef << "\n";
                } else {
                    f << "    " << m.colName[t->i] << "  " << m.colName[t->j]
                      << "  " << (0.5 * t->coef) << "\n";
                    f << "    " << m.colName[t->j] << "  " << m.colName[t->i]
                      << "  " << (0.5 * t->coef) << "\n";
                }
            }
        }
    }
    f << "ENDATA\n";
    return true;
}

bool writeSolution(const std::string& path, const Model& m, const Solution& s, std::string& err) {
    std::ofstream f(path);
    if (!f) { err = "cannot write " + path; return false; }
    f.precision(12);
    f << "# IGAOS solution file\n";
    f << "# status      " << statusName(s.status) << "\n";
    f << "# objective   " << s.objective << "\n";
    f << "# bound       " << s.bestBound << "\n";
    f << "# iterations  " << s.iterations << "\n";
    f << "# nodes       " << s.nodes << "\n";
    f << "# time        " << s.solveTime << "\n";
    f << "# algorithm   " << s.algorithm << "\n";
    f << "\n# columns: name value reduced_cost\n";
    for (Int j = 0; j < m.numCol() && j < (Int)s.colValue.size(); ++j)
        f << (j < (Int)m.colName.size() ? m.colName[j] : ("x" + std::to_string(j)))
          << "  " << s.colValue[j] << "  " << s.colDual[j] << "\n";
    f << "\n# rows: name activity dual\n";
    for (Int i = 0; i < m.numRow() && i < (Int)s.rowValue.size(); ++i)
        f << (i < (Int)m.rowName.size() ? m.rowName[i] : ("r" + std::to_string(i)))
          << "  " << s.rowValue[i] << "  " << s.rowDual[i] << "\n";
    return true;
}

} // namespace igaos
