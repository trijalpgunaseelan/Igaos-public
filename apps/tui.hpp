#pragma once
// ===========================================================================
//  tui.hpp -- Solver Studio, in a terminal.
//
//  The browser console is not a browser feature. Picking a model from a list,
//  ticking cut separation off and running again, watching the pipeline light
//  up, clicking a stage to read what it does, flipping to the log or the
//  convergence plot -- all of that is an INTERFACE, and a terminal is an
//  interface. PS 26119 asks for a command line and says a graphical interface
//  is not required; it does not ask the command line to be worse.
//
//  So this is the same console, driven by keys instead of a mouse, over the
//  same event stream (Logger::onStage) that feeds the browser. One solver, one
//  protocol, three front ends: batch flags, this, and the web page.
//
//  Run it with `igaos` and no arguments, or `igaos --tui`.
//
//  It refuses to start unless both stdin and stdout are terminals, because a
//  full-screen application writing into a pipe is a corrupted log file.
// ===========================================================================

#include "igaos/solver.hpp"
#include "igaos/mps.hpp"
#include "pipeline_view.hpp"
#include "cli_extras.hpp"
#include "igaos/certificate.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <string>
#include <vector>
#include <deque>
#include <algorithm>

#if !defined(_WIN32)
#  include <termios.h>
#  include <unistd.h>
#  include <sys/ioctl.h>
#endif

namespace igaos {

#if !defined(_WIN32)

// --------------------------------------------------------------- terminal --
// Raw mode, alternate screen, hidden cursor -- and every one of them put back,
// including when the process is killed. A tool that leaves your terminal
// unusable is not finished.
class RawTerminal {
public:
    RawTerminal() {
        ::tcgetattr(STDIN_FILENO, &saved_);
        termios raw = saved_;
        raw.c_lflag &= ~(unsigned)(ECHO | ICANON);
        raw.c_cc[VMIN]  = 1;
        raw.c_cc[VTIME] = 0;
        ::tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
        std::fputs("\x1b[?1049h\x1b[?25l", stdout);   // alt screen, hide cursor
        std::fflush(stdout);
        active_ = this;
        prevInt_  = std::signal(SIGINT,  onSignal);
        prevTerm_ = std::signal(SIGTERM, onSignal);
    }
    ~RawTerminal() { restore(); active_ = nullptr; }
    void restore() {
        if (done_) return;
        done_ = true;
        std::fputs("\x1b[?25h\x1b[?1049l", stdout);   // cursor back, main screen
        std::fflush(stdout);
        ::tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved_);
        std::signal(SIGINT,  prevInt_);
        std::signal(SIGTERM, prevTerm_);
    }
private:
    static void onSignal(int sig) {
        if (active_) active_->restore();
        std::signal(sig, SIG_DFL);
        ::raise(sig);
    }
    termios saved_{};
    bool done_ = false;
    void (*prevInt_)(int)  = nullptr;
    void (*prevTerm_)(int) = nullptr;
    static RawTerminal* active_;
};
inline RawTerminal* RawTerminal::active_ = nullptr;

// ------------------------------------------------------------------- the app
class Tui {
public:
    explicit Tui(Options base) : opt_(base), view_(stdout, false) {
        lib_ = library();
        view_.setQuiet(true);          // this class owns every pixel on screen
        opt_.log.level = 0;                 // the panes are the output
        // A console with no Stop key must not be able to hang. Every run here
        // is bounded; [ and ] change it. The batch flags keep their own
        // behaviour, where an unbounded run is what a harness wants.
        if (opt_.timeLimit <= 0 || opt_.timeLimit > 1e29) opt_.timeLimit = 20.0;
    }

    int run() {
        if (!::isatty(STDIN_FILENO) || !::isatty(STDOUT_FILENO)) {
            std::fprintf(stderr,
                "igaos: the interactive console needs a terminal on both stdin and\n"
                "       stdout. For a script use the flags instead:\n"
                "         igaos --models\n"
                "         igaos -m <name> --compare cuts\n"
                "         igaos -m <name> --compare paths\n");
            return 1;
        }
        if (lib_.empty())
            note_ = "No model library found -- run from the repository root to get one.";

        RawTerminal term;
        term_ = &term;
        draw();
        for (;;) {
            const int k = key();
            if (k == 'q' || k == 3 /* ctrl-C */) break;
            handle(k);
            draw();
        }
        term.restore();
        term_ = nullptr;
        std::printf("\n");
        return 0;
    }

private:
    // ----------------------------------------------------------------- input
    static int key() {
        unsigned char c = 0;
        if (::read(STDIN_FILENO, &c, 1) != 1) return 'q';
        if (c != 27) return c;
        unsigned char a = 0, b = 0;
        if (::read(STDIN_FILENO, &a, 1) != 1) return 27;
        if (a != '[') return 27;
        if (::read(STDIN_FILENO, &b, 1) != 1) return 27;
        switch (b) {                              // arrows -> vi keys
            case 'A': return 'k';
            case 'B': return 'j';
            case 'C': return 'l';
            case 'D': return 'h';
            default:  return 27;
        }
    }
    static void size(int& rows, int& cols) {
        winsize w{};
        if (::ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0 && w.ws_col > 20) {
            rows = w.ws_row; cols = w.ws_col;
        } else { rows = 30; cols = 100; }
    }

    void handle(int k) {
        switch (k) {
            case 'j': if (!lib_.empty()) sel_ = (sel_ + 1) % (int)lib_.size(); break;
            case 'k': if (!lib_.empty()) sel_ = (sel_ + (int)lib_.size() - 1)
                                                % (int)lib_.size(); break;
            case '\r': case '\n': solveOne(); break;
            case 'c': opt_.cuts      = !opt_.cuts;      break;
            case 'p': opt_.presolve  = !opt_.presolve;  break;
            case 's': opt_.scaling   = !opt_.scaling;   break;
            case 'x': opt_.crossover = !opt_.crossover; break;
            case 'a': cycleAlgorithm(); break;
            case '[': opt_.timeLimit = std::max(1.0, opt_.timeLimit / 2); break;
            case ']': opt_.timeLimit = std::min(600.0, opt_.timeLimit * 2); break;
            case 'C': compare(true);  break;            // cuts on/off
            case 'P': compare(false); break;            // four paths
            case '1': pane_ = PIPELINE; break;
            case '2': pane_ = CONSOLE;  break;
            case '3': pane_ = TIMELINE; break;
            case '4': pane_ = CHART;    break;
            case '5': pane_ = TABLE;    break;
            case '?': pane_ = HELP;     break;
            case 'e': pane_ = STAGES;   break;
            case 'f': flow_ = !flow_; pane_ = PIPELINE; break;
            case '6': pane_ = SOLUTION; break;
            case 'v': verify();  break;
            case 'S': sweep();   break;
            case '\t': pane_ = (Pane)((pane_ + 1) % PANE_COUNT); break;
            default: break;
        }
    }
    void cycleAlgorithm() {
        static const LpAlgorithm order[] = {
            LpAlgorithm::Auto, LpAlgorithm::PrimalSimplex, LpAlgorithm::DualSimplex,
            LpAlgorithm::InteriorPoint, LpAlgorithm::PDHG };
        for (size_t i = 0; i < 5; ++i)
            if (opt_.lpAlgorithm == order[i]) { opt_.lpAlgorithm = order[(i + 1) % 5]; return; }
        opt_.lpAlgorithm = LpAlgorithm::Auto;
    }
    static const char* algName(LpAlgorithm a) {
        switch (a) {
            case LpAlgorithm::PrimalSimplex: return "primal simplex";
            case LpAlgorithm::DualSimplex:   return "dual simplex";
            case LpAlgorithm::InteriorPoint: return "interior point";
            case LpAlgorithm::PDHG:          return "first-order (PDHG)";
            default:                         return "let the solver choose";
        }
    }

    // ----------------------------------------------------------------- solve
    struct StageRow { std::string id, title, badge, metric; int colour = 0; };

    void resetRun() {
        rows_.clear(); console_.clear(); curve_.clear(); table_.clear();
        haveResult_ = false; note_.clear(); spans_.clear();
    }

    // Same rules the console and the batch view use, kept in one place here so
    // the three cannot drift: exactly one continuous method runs, the rest are
    // OTHER PATH, and a stage switched off is SKIPPED.
    void onEvent(const char* name, const char* state, const char* detail) {
        view_.event(name, state, detail);
        const std::string n = name, s = state, d = detail ? detail : "";
        char line[320];
        double t = clock_.elapsed();
        if (s == "begin")      std::snprintf(line, sizeof line, "%-9s %s", n.c_str(), "begin");
        else if (s == "skip")  std::snprintf(line, sizeof line, "%-9s %s", n.c_str(), "skipped");
        else                   std::snprintf(line, sizeof line, "%-9s %s %s",
                                             n.c_str(), s.c_str(), d.c_str());
        char stamp[400];
        std::snprintf(stamp, sizeof stamp, "%7.3f  %s", t, line);
        console_.push_back(stamp);
        if (console_.size() > 4000) console_.pop_front();

        if (n == "tree" && s == "node") {
            CurvePt p;
            p.t = clock_.elapsed();
            p.bound = std::atof(PipelineView::field(d, "bound").c_str());
            const std::string inc = PipelineView::field(d, "incumbent");
            p.hasInc = !inc.empty() && std::atof(inc.c_str()) != 0.0;
            p.inc = p.hasInc ? std::atof(inc.c_str()) : 0.0;
            curve_.push_back(p);
        }
        // Repaint while the solve runs, but not faster than a person can read.
        // The rows have to be pulled across on every event, not only at the
        // end: without that the pipeline pane stays empty for the whole run and
        // the live view is live in name only.
        if (t - lastPaint_ > 0.06) {
            lastPaint_ = t;
            snapshot(view_);
            draw();
        }
    }

    void solveOne() {
        if (lib_.empty()) { note_ = "No models to solve."; return; }
        Model model; std::string err;
        if (!readMps(lib_[(size_t)sel_].path, model, err)) { note_ = err; return; }

        resetRun();
        view_.resetAll();
        clock_.reset(); lastPaint_ = 0;
        running_ = true; pane_ = PIPELINE;

        Options o = opt_;
        o.log.level = 0;
        o.log.onStage = [this](const char* a, const char* b, const char* c) {
            onEvent(a, b, c);
        };
        Solver solver; solver.opt = o;
        sol_ = solver.solve(model);
        rep_ = solver.report;
        model_ = model.stats();
        modelName_ = lib_[(size_t)sel_].tag;
        haveResult_ = true;
        running_ = false;
        snapshot(view_);
    }

    void compare(bool cuts) {
        if (lib_.empty()) { note_ = "No models to compare."; return; }
        Model model; std::string err;
        if (!readMps(lib_[(size_t)sel_].path, model, err)) { note_ = err; return; }
        if (cuts && !model.isMip()) {
            note_ = "That model has no integer columns -- nothing to separate. "
                    "Pick a MILP or MIQP.";
            return;
        }
        resetRun();
        running_ = true; pane_ = PIPELINE;
        clock_.reset(); lastPaint_ = 0;

        struct Leg2 { std::string label; Solution s; SolveReport r; };
        std::vector<Leg2> legs;
        std::vector<std::pair<std::string, Options>> plan;
        if (cuts) {
            Options a = opt_; a.cuts = true;  plan.push_back({"cut separation on",  a});
            Options b = opt_; b.cuts = false; plan.push_back({"cut separation off", b});
        } else {
            const LpAlgorithm algs[] = {LpAlgorithm::PrimalSimplex, LpAlgorithm::DualSimplex,
                                        LpAlgorithm::InteriorPoint, LpAlgorithm::PDHG};
            for (LpAlgorithm a : algs) {
                Options o = opt_; o.lpAlgorithm = a;
                plan.push_back({algName(a), o});
            }
        }
        view_.resetAll();
        for (size_t i = 0; i < plan.size(); ++i) {
            legLabel_ = "leg " + std::to_string(i + 1) + " of "
                      + std::to_string(plan.size()) + " -- " + plan[i].first;
            view_.startLeg("");
            Options o = plan[i].second;
            o.log.level = 0;
            o.log.onStage = [this](const char* a, const char* b, const char* c) {
                onEvent(a, b, c);
            };
            Solver s; s.opt = o;
            Leg2 L; L.label = plan[i].first;
            L.s = s.solve(model); L.r = s.report;
            legs.push_back(L);
            snapshot(view_);
            draw();
        }
        legLabel_.clear();
        model_ = model.stats();
        modelName_ = lib_[(size_t)sel_].tag;
        sol_ = legs.back().s; rep_ = legs.back().r;
        haveResult_ = true; running_ = false;

        // Build the comparison table the console shows.
        table_.clear();
        if (cuts) {
            const Leg2& a = legs[0]; const Leg2& b = legs[1];
            table_.push_back({"measure", "with cuts", "without"});
            table_.push_back({"status", statusName(a.s.status), statusName(b.s.status)});
            table_.push_back({"objective", num((double)a.s.objective, 12),
                                           num((double)b.s.objective, 12)});
            table_.push_back({"nodes", num((long long)a.s.nodes), num((long long)b.s.nodes)});
            table_.push_back({"iterations", num((long long)a.r.simplexIterations),
                                            num((long long)b.r.simplexIterations)});
            table_.push_back({"time", tsec(a.r.totalTime), tsec(b.r.totalTime)});
            table_.push_back({"cuts kept", num((long long)a.r.cutsApplied),
                                           num((long long)b.r.cutsApplied)});
            table_.push_back({"root LP bound", num((double)a.r.rootBoundLp, 10), "-"});
            table_.push_back({"after cuts", num((double)a.r.rootBoundCut, 10), "-"});
            const bool same = std::fabs(a.s.objective - b.s.objective)
                              <= 1e-6 * (1.0 + std::fabs((double)a.s.objective));
            note_ = same ? "Both runs return the same objective. The difference is how much "
                           "work it took to prove it."
                         : "The objectives DIFFER -- check for a limit, or an invalid cut.";
        } else {
            table_.push_back({"method", "status", "objective", "iterations", "time"});
            double lo = 0, hi = 0; bool first = true;
            for (const Leg2& L : legs) {
                long long it = L.r.simplexIterations;
                if (L.r.ipmIterations  > 0) it = L.r.ipmIterations;
                if (L.r.pdhgIterations > 0) it = L.r.pdhgIterations;
                table_.push_back({L.label, statusName(L.s.status),
                                  num((double)L.s.objective, 12), num(it),
                                  tsec(L.r.totalTime)});
                if (L.s.status == Status::Optimal || L.s.status == Status::Feasible) {
                    if (first) { lo = hi = L.s.objective; first = false; }
                    lo = std::min(lo, (double)L.s.objective);
                    hi = std::max(hi, (double)L.s.objective);
                }
            }
            note_ = first ? "No leg reached a provable answer."
                          : (hi - lo <= 1e-6 * (1.0 + std::fabs(lo))
                             ? "Four algorithms sharing no code path, one answer: "
                               + num(lo, 12) + "."
                             : "The four disagree by " + num(hi - lo, 3)
                               + " -- on a proven optimum that is a defect.");
        }
        pane_ = TABLE;
    }

    // ----------------------------------------------------- certificates ----
    // The point of the certificate is that something which is NOT this solver
    // checks the answer. So this writes the file and then runs the independent
    // checker as a separate process, in exact rational arithmetic, and shows
    // its verdict verbatim. Doing the check in here would defeat the argument.
    void verify() {
        if (!haveResult_) { note_ = "Solve something first (Enter), then press v."; return; }
        Model model; std::string err;
        const std::string mps = lib_[(size_t)sel_].path;
        if (!readMps(mps, model, err)) { note_ = err; return; }

        const std::string cert = "/tmp/igaos_tui.cert";
        if (!writeCertificate(cert, model, sol_, err)) { note_ = err; return; }

        std::string checker;
        for (const char* c : {"tools/verify_certificate.py", "../tools/verify_certificate.py",
                              "../../tools/verify_certificate.py"})
            if (isFile(c)) { checker = c; break; }
        if (checker.empty()) {
            note_ = "Certificate written to " + cert
                  + " -- the checker is in tools/verify_certificate.py; run from the repo root.";
            return;
        }
        const std::string cmd = "python3 " + checker + " " + mps + " " + cert + " 2>&1";
        verdict_.clear();
        if (FILE* pp = ::popen(cmd.c_str(), "r")) {
            char line[512];
            while (std::fgets(line, sizeof line, pp)) {
                std::string s = line;
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
                verdict_.push_back(s);
                if (verdict_.size() > 400) break;
            }
            ::pclose(pp);
        }
        if (verdict_.empty()) verdict_.push_back("the checker produced no output -- is python3 present?");
        verdict_.insert(verdict_.begin(), "$ " + cmd);
        pane_ = VERDICT;
        note_ = "Checked by a separate program with its own MPS reader and no floating point.";
    }

    // ------------------------------------------------------------ sweep ----
    // Every bundled model, one table. Breadth in one keypress, and it is the
    // honest kind of breadth: whatever does not solve says so.
    void sweep() {
        if (lib_.empty()) { note_ = "No models to sweep."; return; }
        table_.clear();
        table_.push_back({"model", "kind", "rows", "cols", "status", "objective", "nodes", "time"});
        running_ = true;
        int ok = 0;
        for (size_t i = 0; i < lib_.size(); ++i) {
            legLabel_ = "sweeping " + std::to_string(i + 1) + " of "
                      + std::to_string(lib_.size()) + " -- " + lib_[i].tag;
            pane_ = TABLE; draw();
            Model m; std::string e;
            if (!readMps(lib_[i].path, m, e)) {
                table_.push_back({lib_[i].tag, "-", "-", "-", "unreadable", "-", "-", "-"});
                continue;
            }
            Options o = opt_;
            o.log.level = 0; o.log.onStage = nullptr;
            o.timeLimit = std::min(opt_.timeLimit, 10.0);
            Solver s; s.opt = o;
            Solution sol = s.solve(m);
            const char* kind = m.isQp() ? (m.isMip() ? "MIQP" : "QP")
                                        : (m.isMip() ? "MILP" : "LP");
            Model::Stats st = m.stats();
            table_.push_back({lib_[i].tag, kind, num((long long)st.nrow),
                              num((long long)st.ncol), statusName(sol.status),
                              num((double)sol.objective, 10),
                              num((long long)sol.nodes), tsec(s.report.totalTime)});
            if (sol.status == Status::Optimal) ++ok;
        }
        legLabel_.clear(); running_ = false; pane_ = TABLE;
        note_ = std::to_string(ok) + " of " + std::to_string(lib_.size())
              + " proven optimal within " + tsec(std::min(opt_.timeLimit, 10.0))
              + " each. Anything else is reported as it came out.";
    }

    void solutionPane(std::vector<std::string>& v, size_t w, int h) {
        if (!haveResult_) {
            v.push_back("\x1b[90mNothing solved yet -- press Enter.\x1b[0m");
            return;
        }
        char b[256];
        std::snprintf(b, sizeof b, "status %s   objective %s   %lld nonzero of %d columns",
                      statusName(sol_.status), num((double)sol_.objective, 12).c_str(),
                      (long long)std::count_if(sol_.colValue.begin(), sol_.colValue.end(),
                                               [](Real x) { return x != 0.0; }),
                      (int)sol_.colValue.size());
        v.push_back(std::string("\x1b[90m") + b + "\x1b[0m");
        v.push_back("");
        int shown = 0;
        const int room = std::max(4, h - 8);
        for (size_t j = 0; j < sol_.colValue.size(); ++j) {
            if (sol_.colValue[j] == 0.0) continue;
            if (shown >= room) {
                std::snprintf(b, sizeof b, "\x1b[90m... and more; --solution FILE writes them all\x1b[0m");
                v.push_back(b);
                break;
            }
            std::snprintf(b, sizeof b, "  x[%-6d] = %s", (int)j,
                          num((double)sol_.colValue[j], 10).c_str());
            v.push_back(clip(b, w));
            ++shown;
        }
        if (shown == 0) v.push_back("\x1b[90mEvery variable is zero.\x1b[0m");
    }

    void verdictPane(std::vector<std::string>& v, size_t w, int h) {
        if (verdict_.empty()) {
            v.push_back("\x1b[90mPress v after a solve to write a certificate and run\x1b[0m");
            v.push_back("\x1b[90mthe independent checker over it.\x1b[0m");
            return;
        }
        const int room = std::max(4, h - 4);
        const size_t from = verdict_.size() > (size_t)room ? verdict_.size() - (size_t)room : 0;
        for (size_t i = from; i < verdict_.size(); ++i) {
            const std::string& s = verdict_[i];
            const bool good = s.find("PROVEN") != std::string::npos
                           || s.find("VERIFIED") != std::string::npos;
            const bool bad  = s.find("REJECT") != std::string::npos
                           || s.find("NOT SUPPORTED") != std::string::npos;
            v.push_back(std::string(good ? "\x1b[32m" : bad ? "\x1b[31m" : "\x1b[90m")
                        + clip(s, w) + "\x1b[0m");
        }
    }

    // Copy the pipeline's rows out so the panes can draw them after the run.
    void snapshot(const PipelineView& v) {
        rows_.clear();
        spans_ = v.timings();
        for (const auto& r : v.snapshot()) {
            StageRow s;
            s.id = r.id; s.title = r.title; s.badge = r.badge; s.metric = r.metric;
            s.colour = r.colour;
            rows_.push_back(s);
        }
    }

    // ------------------------------------------------------------------ draw
    enum Pane { PIPELINE, CONSOLE, TIMELINE, CHART, TABLE, SOLUTION, VERDICT,
                STAGES, HELP, PANE_COUNT };

    void put(const std::string& s) { buf_ += s; }
    void putf(const char* fmt, ...) {
        char b[1024];
        va_list ap; va_start(ap, fmt);
        std::vsnprintf(b, sizeof b, fmt, ap);
        va_end(ap);
        buf_ += b;
    }
    static std::string clip(const std::string& s, size_t w) {
        return s.size() <= w ? s : s.substr(0, w > 1 ? w - 1 : 0) + "~";
    }

    void draw() {
        int rows = 30, cols = 100;
        size(rows, cols);
        const int left = 28;
        const int right = std::max(30, cols - left - 3);
        buf_.clear();
        put("\x1b[H\x1b[2J");

        // ------------------------------------------------------------ header
        putf("\x1b[1m  IGAOS SOLVER STUDIO\x1b[0m  \x1b[90m0.2.0 - PS 26119 - press ? for keys\x1b[0m\n");
        putf("  \x1b[90m%s\x1b[0m\n", std::string((size_t)std::min(cols - 4, 110), '-').c_str());

        std::vector<std::string> L = leftColumn();
        std::vector<std::string> R = rightColumn((size_t)right, rows - 6);
        const size_t n = std::max(L.size(), R.size());
        for (size_t i = 0; i < n && (int)i < rows - 5; ++i) {
            const std::string a = i < L.size() ? L[i] : std::string();
            const std::string b = i < R.size() ? R[i] : std::string();
            // A left cell wider than its column would push the divider across
            // the screen and shear the whole layout. Never pad by a negative.
            const int pad = std::max(1, left - visible(a));
            putf("  %s%*s\x1b[90m|\x1b[0m %s\n", a.c_str(), pad, "", b.c_str());
        }

        // ------------------------------------------------------------ footer
        putf("  \x1b[90m%s\x1b[0m\n", std::string((size_t)std::min(cols - 4, 110), '-').c_str());
        if (!note_.empty())
            putf("  \x1b[33m%s\x1b[0m\n", clip(note_, (size_t)cols - 6).c_str());
        putf("  \x1b[90m"
             "j/k model  enter solve  c/p/s/x options  a algorithm  "
             "C cuts on-off  P four paths\x1b[0m\n");
        putf("  \x1b[90m"
             "1 pipeline  2 console  3 timeline  4 chart  5 table  e stages  "
             "? help  q quit\x1b[0m");
        std::fputs(buf_.c_str(), stdout);
        std::fflush(stdout);
    }

    // Count printable width, ignoring the escape sequences we inserted.
    static int visible(const std::string& s) {
        int n = 0;
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i] == '\x1b') { while (i < s.size() && s[i] != 'm') ++i; continue; }
            if ((unsigned char)s[i] < 0x80 || ((unsigned char)s[i] & 0xC0) != 0x80) ++n;
        }
        return n;
    }

    std::vector<std::string> leftColumn() {
        std::vector<std::string> v;
        v.push_back("\x1b[1mMODEL LIBRARY\x1b[0m");
        for (size_t i = 0; i < lib_.size(); ++i) {
            const bool on = (int)i == sel_;
            char b[160];
            std::snprintf(b, sizeof b, "%s%-14s %s%s",
                          on ? "\x1b[7m> " : "  ", clip(lib_[i].tag, 14).c_str(),
                          kindOf(lib_[i]).c_str(), on ? "\x1b[0m" : "");
            v.push_back(b);
        }
        v.push_back("");
        v.push_back("\x1b[1mOPTIONS\x1b[0m");
        auto tick = [](bool b) { return b ? "[x]" : "[ ]"; };
        v.push_back(std::string("  ") + tick(opt_.cuts)      + " cut separation   (c)");
        v.push_back(std::string("  ") + tick(opt_.presolve)  + " presolve         (p)");
        v.push_back(std::string("  ") + tick(opt_.scaling)   + " scaling          (s)");
        v.push_back(std::string("  ") + tick(opt_.crossover) + " crossover        (x)");
        v.push_back("");
        v.push_back("  algorithm            (a)");
        v.push_back(std::string("    \x1b[36m") + clip(algName(opt_.lpAlgorithm), 21) + "\x1b[0m");
        {
            char b[96];
            std::snprintf(b, sizeof b, "  time limit  \x1b[36m%gs\x1b[0m   ([ ])",
                          opt_.timeLimit);
            v.push_back(b);
        }
        if (haveResult_) {
            v.push_back("");
            v.push_back("\x1b[1mRESULT\x1b[0m");
            v.push_back("  status     " + std::string(statusName(sol_.status)));
            v.push_back("  objective  " + num((double)sol_.objective, 12));
            if (sol_.nodes > 0) v.push_back("  nodes      " + num((long long)sol_.nodes));
            v.push_back("  iterations " + num((long long)rep_.simplexIterations));
            v.push_back("  time       " + tsec(rep_.totalTime));
        }
        return v;
    }
    std::string kindOf(const LibModel& m) const {
        const std::string t = m.tag;
        if (t.compare(0, 3, "miq") == 0) return "\x1b[90mMIQP\x1b[0m";
        if (t.compare(0, 1, "q")   == 0) return "\x1b[90mQP\x1b[0m";
        if (t.compare(0, 2, "uc")  == 0) return "\x1b[90mMILP\x1b[0m";
        return "\x1b[90mLP\x1b[0m";
    }

    std::vector<std::string> rightColumn(size_t w, int h) {
        std::vector<std::string> v;
        const char* title[PANE_COUNT] = {"SOLVE PIPELINE", "EXECUTION CONSOLE",
                                         "WHERE THE TIME WENT", "BOUND vs INCUMBENT",
                                         "COMPARISON", "SOLUTION", "CERTIFICATE CHECK",
                                         "WHAT EACH STAGE IS", "KEYS"};
        std::string head = std::string("\x1b[1m") + title[pane_] + "\x1b[0m";
        if (!modelName_.empty())
            head += "   \x1b[90m" + modelName_ + "  " + std::to_string(model_.nrow) + "x"
                  + std::to_string(model_.ncol) + "\x1b[0m";
        if (running_) head += "  \x1b[36m* solving\x1b[0m";
        if (!legLabel_.empty()) head += "  \x1b[36m" + legLabel_ + "\x1b[0m";
        v.push_back(head);
        v.push_back("");

        switch (pane_) {
            case PIPELINE:  pipelinePane(v, w, h);        break;
            case CONSOLE:   consolePane(v, w, h);         break;
            case TIMELINE:  timelinePane(v, w);           break;
            case CHART:     chartPane(v, w, h);           break;
            case TABLE:     tablePane(v, w);              break;
            case SOLUTION:  solutionPane(v, w, h);        break;
            case VERDICT:   verdictPane(v, w, h);         break;
            case STAGES:    stagesPane(v, w, h);          break;
            default:        helpPane(v);                  break;
        }
        return v;
    }

    void pipelinePane(std::vector<std::string>& v, size_t w, int h) {
        // The chart when it fits, the list when it does not. A chart that wraps
        // is worse than a list that does not.
        if (flow_ && !rows_.empty()) {
            std::vector<std::string> f = view_.flowLines((int)w, h, true);
            if (!f.empty()) {
                for (const std::string& s : f) v.push_back(s);
                v.push_back("");
                v.push_back("\x1b[90mf for the compact list\x1b[0m");
                return;
            }
        }
        if (rows_.empty()) {
            v.push_back("\x1b[90mPress Enter to solve the selected model.\x1b[0m");
            v.push_back("");
            v.push_back("\x1b[90mExactly one of the four continuous methods runs per\x1b[0m");
            v.push_back("\x1b[90msolve. The other three read OTHER PATH: idle, not\x1b[0m");
            v.push_back("\x1b[90mmissing. Press P to run all four in turn.\x1b[0m");
            return;
        }
        for (const StageRow& r : rows_) {
            const char* c = r.colour == 1 ? "\x1b[36m" : r.colour == 2 ? "\x1b[32m"
                          : r.colour == 3 ? "\x1b[31m" : "\x1b[90m";
            char b[512];
            std::snprintf(b, sizeof b, "%s%-20s %-10s\x1b[0m \x1b[90m%s\x1b[0m",
                          c, clip(r.title, 20).c_str(), r.badge.c_str(),
                          clip(r.metric, w > 34 ? w - 33 : 10).c_str());
            v.push_back(b);
        }
    }
    void consolePane(std::vector<std::string>& v, size_t w, int h) {
        const int room = std::max(4, h - 4);
        const size_t from = console_.size() > (size_t)room ? console_.size() - (size_t)room : 0;
        for (size_t i = from; i < console_.size(); ++i)
            v.push_back("\x1b[90m" + clip(console_[i], w) + "\x1b[0m");
        if (console_.empty()) v.push_back("\x1b[90mNothing yet -- press Enter to solve.\x1b[0m");
    }
    void timelinePane(std::vector<std::string>& v, size_t w) {
        if (spans_.empty()) { v.push_back("\x1b[90mNothing timed yet.\x1b[0m"); return; }
        double total = 0;
        for (const auto& s : spans_) total += s.second;
        if (total <= 0) { v.push_back("\x1b[90mToo fast to measure.\x1b[0m"); return; }
        const int barw = (int)std::min<size_t>(w > 40 ? w - 34 : 12, 46);
        for (const auto& s : spans_) {
            const int k = std::max(1, (int)(barw * (s.second / total) + 0.5));
            char b[512];
            std::snprintf(b, sizeof b, "%-20s \x1b[32m%s\x1b[0m%*s \x1b[90m%s\x1b[0m",
                          clip(s.first, 20).c_str(), std::string((size_t)k, '=').c_str(),
                          barw - k, "", tsec(s.second).c_str());
            v.push_back(b);
        }
    }
    void chartPane(std::vector<std::string>& v, size_t w, int h) {
        if (curve_.size() < 3) {
            v.push_back("\x1b[90mA dual bound and an incumbent only move apart in a\x1b[0m");
            v.push_back("\x1b[90mbranch and bound run. Solve a MILP -- try uc_l with\x1b[0m");
            v.push_back("\x1b[90mcut separation off.\x1b[0m");
            return;
        }
        const int width  = (int)std::min<size_t>(w > 20 ? w - 16 : 20, 70);
        const int height = std::max(6, std::min(h - 6, 16));
        double lo = curve_[0].bound, hi = curve_[0].bound, t1 = 0;
        for (const CurvePt& p : curve_) {
            lo = std::min(lo, p.bound); hi = std::max(hi, p.bound);
            if (p.hasInc) { lo = std::min(lo, p.inc); hi = std::max(hi, p.inc); }
            t1 = std::max(t1, p.t);
        }
        if (!(hi > lo)) hi = lo + 1;
        if (t1 <= 0) t1 = 1;
        std::vector<std::string> g((size_t)height, std::string((size_t)width, ' '));
        auto plot = [&](double t, double val, char ch) {
            int x = (int)((t / t1) * (width - 1) + 0.5);
            int y = (int)((1.0 - (val - lo) / (hi - lo)) * (height - 1) + 0.5);
            x = std::max(0, std::min(width - 1, x));
            y = std::max(0, std::min(height - 1, y));
            g[(size_t)y][(size_t)x] = ch;
        };
        for (const CurvePt& p : curve_) {
            plot(p.t, p.bound, 'o');
            if (p.hasInc) plot(p.t, p.inc, '*');
        }
        v.push_back("\x1b[90mo dual bound   * incumbent\x1b[0m");
        for (size_t i = 0; i < g.size(); ++i) {
            char lab[32] = "";
            if (i == 0)              std::snprintf(lab, sizeof lab, "%.6g", hi);
            if (i + 1 == g.size())   std::snprintf(lab, sizeof lab, "%.6g", lo);
            char b[512];
            std::snprintf(b, sizeof b, "%11s |%s", lab, g[i].c_str());
            v.push_back(b);
        }
        char b[512];
        std::snprintf(b, sizeof b, "%11s +%s", "", std::string((size_t)width, '-').c_str());
        v.push_back(b);
        std::snprintf(b, sizeof b, "%11s  0 s%*s%s", "", width - 8, "", tsec(t1).c_str());
        v.push_back(b);
    }
    void tablePane(std::vector<std::string>& v, size_t w) {
        if (table_.empty()) {
            v.push_back("\x1b[90mPress C to solve with and without cut separation,\x1b[0m");
            v.push_back("\x1b[90mor P to solve with all four continuous methods.\x1b[0m");
            return;
        }
        std::vector<size_t> cw(table_[0].size(), 0);
        for (const auto& r : table_)
            for (size_t i = 0; i < r.size() && i < cw.size(); ++i)
                cw[i] = std::max(cw[i], r[i].size());
        for (size_t r = 0; r < table_.size(); ++r) {
            std::string line;
            for (size_t i = 0; i < table_[r].size() && i < cw.size(); ++i) {
                char b[160];
                std::snprintf(b, sizeof b, i ? "%*s  " : "%-*s  ",
                              (int)cw[i], table_[r][i].c_str());
                line += b;
            }
            if (r == 0) v.push_back("\x1b[1m" + clip(line, w) + "\x1b[0m");
            else        v.push_back(clip(line, w));
            if (r == 0) {
                std::string rule;
                for (size_t i = 0; i < cw.size(); ++i) rule += std::string(cw[i], '-') + "  ";
                v.push_back("\x1b[90m" + clip(rule, w) + "\x1b[0m");
            }
        }
    }
    void stagesPane(std::vector<std::string>& v, size_t w, int h) {
        // The console's focus panel, one stage at a time; j/k move the model
        // list, so this scrolls with the pane key instead. Show the stage that
        // is currently selected in the pipeline, defaulting to the first.
        std::string txt = PipelineView::explainText(stageSel_);
        size_t pos = 0;
        while (pos < txt.size() && (int)v.size() < h) {
            size_t nl = txt.find('\n', pos);
            v.push_back(clip(txt.substr(pos, nl - pos), w));
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        v.push_back("");
        v.push_back("\x1b[90mpress e again for the next stage\x1b[0m");
        stageSel_ = (stageSel_ + 1) % PipelineView::stageCount();
    }
    void helpPane(std::vector<std::string>& v) {
        static const char* keys[][2] = {
            {"j / k  or arrows", "move through the model library"},
            {"Enter",            "solve the selected model"},
            {"c p s x",          "toggle cuts, presolve, scaling, crossover"},
            {"a",                "cycle the continuous algorithm"},
            {"[  ]",             "halve or double the time limit"},
            {"C",                "solve twice: cut separation on, then off"},
            {"P",                "solve four times, one per continuous method"},
            {"1",                "the solve pipeline"},
            {"2",                "the execution console, one line per event"},
            {"3",                "where the time went"},
            {"4",                "dual bound against incumbent"},
            {"5",                "the comparison table"},
            {"6",                "the solution: every nonzero variable"},
            {"f",                "flow chart or compact list"},
            {"v",                "write a certificate and run the independent checker"},
            {"S",                "sweep every model in the library into one table"},
            {"e",                "what each stage of the pipeline is"},
            {"Tab",              "next pane"},
            {"q",                "quit"},
        };
        for (const auto& k : keys) {
            char b[256];
            std::snprintf(b, sizeof b, "  \x1b[36m%-18s\x1b[0m \x1b[90m%s\x1b[0m", k[0], k[1]);
            v.push_back(b);
        }
        v.push_back("");
        v.push_back("\x1b[90mSKIPPED means you switched a stage off. OTHER PATH means it\x1b[0m");
        v.push_back("\x1b[90mworks and this model went another way -- exactly one of the\x1b[0m");
        v.push_back("\x1b[90mfour continuous methods runs per solve.\x1b[0m");
    }

    struct CurvePt { double t = 0, bound = 0, inc = 0; bool hasInc = false; };

    Options opt_;
    std::vector<LibModel> lib_;
    int sel_ = 0, stageSel_ = 0;
    Pane pane_ = PIPELINE;
    bool running_ = false, haveResult_ = false, flow_ = true;
    std::string note_, buf_, modelName_, legLabel_;
    std::vector<StageRow> rows_;
    std::deque<std::string> console_;
    std::vector<CurvePt> curve_;
    std::vector<std::vector<std::string>> table_;
    std::vector<std::string> verdict_;
    std::vector<std::pair<std::string, double>> spans_;
    Solution sol_;
    SolveReport rep_;
    Model::Stats model_{};
    PipelineView view_;
    RawTerminal* term_ = nullptr;
    Timer clock_;
    double lastPaint_ = 0;
};

#endif // !_WIN32

} // namespace igaos
