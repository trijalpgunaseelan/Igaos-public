#!/usr/bin/env python3
"""
IGAOS solver console — a live view of the pipeline, not a form with a result.

    python3 demo/server.py            # then open http://127.0.0.1:8420

Standard library only: no Flask, no npm, no build step.

WHAT MAKES THIS LIVE RATHER THAN ANIMATED
-----------------------------------------
The solver is run with --progress, which makes it emit one line per pipeline
event at the moment the event happens, flushed:

    IGAOS_STAGE presolve begin
    IGAOS_STAGE presolve end rows=3 cols=0 nnz=3 tightened=173 t=0.0004
    IGAOS_STAGE tree node nodes=250 bound=76208.28 incumbent=76267.64 open=31

This server reads those lines as they arrive and forwards them to the browser
over Server-Sent Events, together with every line of the human log. Nothing on
the page is on a timer and nothing is replayed from a recording: a box lights up
because the solver just entered that phase, and the node counter moves because
the tree moved. If the solver stalls, the page stalls with it — which is the
honest behaviour and, when you are looking for a stall, the useful one.

Binds to 127.0.0.1 only. A console to stand in front of, not a service.
"""
import json, os, re, shutil, subprocess, sys, tempfile, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(HERE, "models")
MAX_UPLOAD = 8 << 20
HARD_TIME_LIMIT = 600.0


def find_binary():
    env = os.environ.get("IGAOS_BIN")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    for c in ("build/igaos", "build/Release/igaos", "igaos"):
        p = os.path.join(ROOT, c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return shutil.which("igaos")


BIN = find_binary()
RESULT_RE = re.compile(r"^IGAOS_RESULT\s+(.*)$")
STAGE_RE = re.compile(r"^IGAOS_STAGE\s+(\S+)\s+(\S+)\s*(.*)$")
# The basis condition estimate is printed in the human RESULT block and is
# NOT in the IGAOS_RESULT line, so it has to be picked up separately. It is
# the number the ill-conditioned demo model exists to show.
COND_RE = re.compile(r"^\s*basis cond est\s+(\S+)\s*$")


def kv(text):
    out = {}
    for tok in text.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        try:
            out[k] = float(v) if re.fullmatch(r"[-+0-9.eE]+", v) else v
        except ValueError:
            out[k] = v
    return out


def model_path(model_id):
    if not re.fullmatch(r"[A-Za-z0-9_]+", model_id or ""):
        return None
    p = os.path.realpath(os.path.join(MODELS, model_id + ".mps"))
    return p if p.startswith(os.path.realpath(MODELS) + os.sep) and os.path.isfile(p) else None


def build_argv(path, opts):
    argv = [BIN, path, "-vv", "--progress"]
    alg = opts.get("algorithm")
    if alg and alg != "auto":
        argv += ["--algorithm", alg]
    if not opts.get("cuts", True):
        argv.append("--no-cuts")
    if not opts.get("presolve", True):
        argv.append("--no-presolve")
    if not opts.get("scaling", True):
        argv.append("--no-scaling")
    if not opts.get("crossover", True):
        argv.append("--no-crossover")
    if opts.get("threads"):
        argv += ["--threads", str(int(opts["threads"]))]
    tl = max(0.1, min(HARD_TIME_LIMIT, float(opts.get("timeLimit", 60))))
    argv += ["--time-limit", "%g" % tl]
    return argv, tl



# --------------------------------------------------------------- diagnostics
# Two questions a planner asks that a plain objective does not answer:
#   "why can't this be built?"      -> irreducible infeasible subsystem
#   "what is this constraint worth?" -> shadow prices with their valid ranges
# Both are computed by the solver itself; this only parses what it prints.

IIS_HDR = re.compile(r"^IIS: (\d+) members \((\d+) rows, (\d+) bounds\) "
                     r"from (\d+) LP solves in ([\d.]+)s -- (.+)$", re.M)
IIS_ROW = re.compile(r"^  row    (\S+)\s+(\S+) <= row <= (\S+)\s*$", re.M)
IIS_BND = re.compile(r"^  bound  (\S+)\s+x (>=|<=)\s+(\S+)\s*$", re.M)


def parse_iis(text):
    h = IIS_HDR.search(text)
    if not h:
        return None
    return {
        "members": int(h.group(1)), "rows": int(h.group(2)),
        "bounds": int(h.group(3)), "solves": int(h.group(4)),
        "seconds": float(h.group(5)),
        "irreducible": h.group(6).startswith("irreducible"),
        "note": h.group(6),
        "rowList": [{"name": m.group(1), "lower": m.group(2), "upper": m.group(3)}
                    for m in IIS_ROW.finditer(text)],
        "boundList": [{"name": m.group(1), "sense": m.group(2), "value": m.group(3)}
                      for m in IIS_BND.finditer(text)],
    }


def _num(tok):
    """Infinities become None: JSON has no way to spell them, and json.dumps
    would emit the bare word Infinity, which every browser's JSON.parse
    rejects.  The UI renders None as "no limit", which is what it means."""
    try:
        v = float(tok)
    except ValueError:
        return None
    return v if -1e29 < v < 1e29 else None


def parse_sensitivity(text):
    """Both tables are fixed-column; a row is a name followed by four numbers."""
    rows, cols, where = [], [], None
    for line in text.splitlines():
        if line.startswith("SHADOW PRICES"):
            where = "rows"; continue
        if line.startswith("REDUCED COSTS"):
            where = "cols"; continue
        if where is None or not line.startswith("  "):
            continue
        if line.strip().startswith(("constraint", "variable", "(", "...", "note:")):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        name = parts[0]
        toks = parts[1:5]
        nums = [_num(t) for t in toks]
        # A row is well-formed when every token is numeric-or-infinite; only a
        # token that is neither disqualifies it.
        if any(nums[i] is None and toks[i] not in ("+inf", "-inf") for i in range(4)):
            continue
        degenerate = "[degenerate]" in line
        if where == "rows":
            rows.append({"name": name, "dual": nums[0], "activity": nums[1],
                         "rhsLower": nums[2], "rhsUpper": nums[3],
                         "degenerate": degenerate})
        else:
            cols.append({"name": name, "reducedCost": nums[0], "value": nums[1],
                         "objLower": nums[2], "objUpper": nums[3]})
    if not rows and not cols:
        return None
    return {"rows": rows, "cols": cols}


def run_diagnose(path, opts):
    """Sensitivity on a solved LP, IIS on an infeasible one. One extra run."""
    if BIN is None:
        return {"error": "solver binary not found"}
    tl = max(0.1, min(HARD_TIME_LIMIT, float(opts.get("timeLimit", 60))))
    argv = [BIN, path, "--sensitivity", "--iis",
            "--iis-time-limit", "%g" % min(tl, 20.0), "--time-limit", "%g" % tl]
    try:
        p = subprocess.run(argv, capture_output=True, text=True,
                           timeout=tl + 60)
    except subprocess.TimeoutExpired:
        return {"error": "diagnostics timed out"}
    text = (p.stdout or "") + (p.stderr or "")
    # An older binary predates these flags and says so on stderr.
    if "unknown option" in text or "unrecognized" in text.lower():
        return {"error": "this build predates --sensitivity / --iis; rebuild the solver",
                "stale": True}
    res = kv(text)
    out = {"status": res.get("status"), "objective": res.get("obj"),
           "raw": text[-8000:]}
    iis = parse_iis(text)
    sen = parse_sensitivity(text)
    if iis:
        out["iis"] = iis
    if sen:
        out["sensitivity"] = sen
    if not iis and not sen:
        out["note"] = ("no diagnostics for this model: sensitivity is linear-programming "
                       "duality and does not apply to a mixed-integer or quadratic model, "
                       "and an IIS only exists for an infeasible one")
    return out



# ------------------------------------------------------------- certificates
# The solver writes a proof of its own answer; a SEPARATE program, which shares
# no code with it, re-checks that proof in exact rational arithmetic.  The point
# of the panel is that the verdict is not the solver's opinion of itself.

VERDICT = re.compile(r"^VERDICT: (.+)$", re.M)
LOWER   = re.compile(r"rigorous lower bound\s+L\(y\) = (\S+)", re.M)
VIOL    = re.compile(r"primal feasibility\s+violated by (\S+)", re.M)
OBJX    = re.compile(r"objective of x\s+(\S+)", re.M)
OBJS    = re.compile(r"objective solver stated\s+(\S+)", re.M)


def run_verify(path, opts):
    """Solve, write a certificate, then check it with the independent tool."""
    if BIN is None:
        return {"error": "solver binary not found"}
    checker = os.path.join(ROOT, "tools", "verify_certificate.py")
    if not os.path.isfile(checker):
        return {"error": "tools/verify_certificate.py is missing"}
    tl = max(0.1, min(HARD_TIME_LIMIT, float(opts.get("timeLimit", 60))))
    fd, cert = tempfile.mkstemp(suffix=".cert")
    os.close(fd)
    try:
        try:
            solve = subprocess.run([BIN, path, "--certificate", cert,
                                    "--time-limit", "%g" % tl],
                                   capture_output=True, text=True, timeout=tl + 60)
        except subprocess.TimeoutExpired:
            return {"error": "the solve timed out before a certificate was written"}
        res = kv(solve.stdout or "")
        if not os.path.getsize(cert):
            return {"error": "no certificate was produced for this model",
                    "note": ("a branch-and-bound tree emits no dual multipliers and "
                             "infeasibility found inside presolve produces no Farkas ray; "
                             "both are reported as unprovable rather than passed silently"),
                    "status": res.get("status")}
        try:
            chk = subprocess.run([sys.executable, checker, path, cert],
                                 capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return {"error": "the checker timed out"}
        text = (chk.stdout or "") + (chk.stderr or "")
        v = VERDICT.search(text)
        def grab(rx):
            m = rx.search(text)
            return m.group(1) if m else None
        return {
            "status": res.get("status"),
            "objective": res.get("obj"),
            "certBytes": os.path.getsize(cert),
            "verdict": v.group(1).strip() if v else None,
            "lowerBound": grab(LOWER),
            "violation": grab(VIOL),
            "objectiveExact": grab(OBJX),
            "objectiveStated": grab(OBJS),
            "checker": os.path.relpath(checker, ROOT),
            "raw": text[-6000:],
        }
    finally:
        try:
            os.unlink(cert)
        except OSError:
            pass



# ------------------------------------------------------------------ solution
# A console that never shows the answer is a strange console. For a refinery
# model the variable values ARE the blend, and until now the page showed
# everything about the solve except what it decided.

SOL_META = re.compile(r"^#\s+(\w+)\s+(.+?)\s*$")


def run_solution(path, opts):
    if BIN is None:
        return {"error": "solver binary not found"}
    tl = max(0.1, min(HARD_TIME_LIMIT, float(opts.get("timeLimit", 60))))
    fd, sol = tempfile.mkstemp(suffix=".sol")
    os.close(fd)
    try:
        argv, _ = build_argv(path, opts)
        argv = [a for a in argv if a not in ("-vv", "--progress")]
        argv += ["--solution", sol]
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=tl + 60)
        except subprocess.TimeoutExpired:
            return {"error": "the solve timed out"}
        res = kv(p.stdout or "")
        mc = None
        for line in (p.stdout or "").splitlines():
            m = COND_RE.match(line)
            if m:
                mc = m.group(1)
        meta, cols = {}, []
        try:
            with open(sol) as fh:
                for line in fh:
                    if line.startswith("#"):
                        m = SOL_META.match(line)
                        if m and m.group(1) not in ("columns",):
                            meta[m.group(1)] = m.group(2)
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    try:
                        v = float(parts[1])
                        rc = float(parts[2]) if len(parts) > 2 else 0.0
                    except ValueError:
                        continue
                    cols.append({"name": parts[0], "value": v, "rc": rc})
        except OSError as e:
            return {"error": "could not read the solution file: %s" % e}
        nz = [c for c in cols if abs(c["value"]) > 1e-9]
        nz.sort(key=lambda c: -abs(c["value"]))
        return {
            "status": res.get("status"), "objective": res.get("obj"),
            "cond": mc, "meta": meta,
            "total": len(cols), "nonzero": len(nz),
            "columns": nz[:60],
        }
    finally:
        try:
            os.unlink(sol)
        except OSError:
            pass


def stream_solve(path, opts, emit):
    """Run one solve, pushing events as the solver produces them."""
    if BIN is None:
        # Named "fail", not "error": EventSource dispatches an SSE event called
        # "error" to the same handler as a dropped connection, so a legible
        # message would arrive looking like a network fault.
        emit("fail", {"message": "solver binary not found — build it first "
                                 "(cmake --build build -j), or set IGAOS_BIN"})
        return
    argv, tl = build_argv(path, opts)
    emit("command", {"argv": [os.path.basename(argv[0])] + argv[1:]})

    t0 = time.time()
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    result = None
    cond = None
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            m = STAGE_RE.match(line)
            if m:
                name, state, rest = m.group(1), m.group(2), m.group(3)
                emit("stage", {"name": name, "state": state, "data": kv(rest),
                               "at": round(time.time() - t0, 4)})
                continue
            m = RESULT_RE.match(line)
            if m:
                result = kv(m.group(1))
                continue
            mc = COND_RE.match(line)
            if mc:
                cond = mc.group(1)
            if line.strip():
                emit("log", {"line": line})
    except SystemExit:
        # The browser went away (tab closed, Stop pressed). Nobody is listening,
        # so the solve is dead weight -- kill it rather than letting a 600-second
        # branch and bound run on in the background.
        proc.kill()
        raise
    finally:
        proc.stdout.close()
        try:
            proc.wait(timeout=tl + 30)
        except subprocess.TimeoutExpired:
            proc.kill()
    if result is not None and cond is not None:
        result["cond"] = cond
    emit("result", result or {"status": "error"})
    emit("done", {"wall": round(time.time() - t0, 4)})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    # ------------------------------------------------------------------ util

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _version(self):
        if BIN is None:
            return None
        try:
            cp = subprocess.run([BIN, "--version"], capture_output=True, text=True, timeout=10)
            return (cp.stdout or cp.stderr or "").strip().splitlines()[0]
        except Exception:
            return None

    # ------------------------------------------------------------------- GET

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            try:
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._send(500, "demo/index.html is missing", "text/plain")

        if u.path == "/api/env":
            return self._send(200, {"binary": BIN, "version": self._version(), "root": ROOT})

        if u.path == "/api/models":
            try:
                with open(os.path.join(MODELS, "manifest.json")) as f:
                    return self._send(200, json.load(f))
            except OSError:
                return self._send(200, [])

        if u.path == "/api/stream":
            return self.do_stream(parse_qs(u.query))

        return self._send(404, {"error": "not found"})

    # ---------------------------------------------------------------- stream

    def do_stream(self, q):
        one = lambda k, d=None: (q.get(k) or [d])[0]
        opts = {
            "algorithm": one("algorithm", "auto"),
            "cuts": one("cuts", "1") == "1",
            "presolve": one("presolve", "1") == "1",
            "scaling": one("scaling", "1") == "1",
            "crossover": one("crossover", "1") == "1",
            "threads": int(one("threads", "0") or 0),
            "timeLimit": float(one("timeLimit", "60") or 60),
        }
        mps = one("mps")
        tmp = None
        if mps:
            if len(mps) > MAX_UPLOAD:
                return self._send(400, {"error": "model too large"})
            fd, tmp = tempfile.mkstemp(suffix=".mps")
            with os.fdopen(fd, "w") as f:
                f.write(mps)
            target = tmp
        else:
            target = model_path(one("model"))
            if target is None:
                return self._send(400, {"error": "unknown model"})

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        lock = threading.Lock()

        def emit(kind, payload):
            with lock:
                try:
                    self.wfile.write(
                        ("event: %s\ndata: %s\n\n" % (kind, json.dumps(payload))).encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    raise SystemExit
        try:
            stream_solve(target, opts, emit)
        except SystemExit:
            pass
        except Exception as e:                       # never leave the page hanging
            try:
                emit("fail", {"message": str(e)})
            except Exception:
                pass
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # ------------------------------------------------------------------ POST

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("payload too large")
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": str(e)})

        tmp = None
        try:
            if req.get("mps"):
                fd, tmp = tempfile.mkstemp(suffix=".mps")
                with os.fdopen(fd, "w") as f:
                    f.write(req["mps"])
                target = tmp
            else:
                target = model_path(req.get("model"))
                if target is None:
                    return self._send(400, {"error": "unknown model"})

            if u.path == "/api/solution":
                return self._send(200, run_solution(target, req))

            if u.path == "/api/verify":
                return self._send(200, run_verify(target, req))

            if u.path == "/api/diagnose":
                return self._send(200, run_diagnose(target, req))

            if u.path == "/api/compare":
                out = {}
                for tag, cuts in (("withCuts", True), ("withoutCuts", False)):
                    o = dict(req, cuts=cuts)
                    out[tag] = collect(target, o)
                return self._send(200, out)

            if u.path == "/api/algorithms":
                out = {}
                for a in ("primal", "dual", "interior", "pdhg"):
                    out[a] = collect(target, dict(req, algorithm=a))
                return self._send(200, out)

            return self._send(404, {"error": "not found"})
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


def collect(path, opts):
    """Run a solve to completion and return everything it produced."""
    bag = {"log": [], "stages": []}

    def emit(kind, payload):
        if kind == "log":
            bag["log"].append(payload["line"])
        elif kind == "stage":
            bag["stages"].append(payload)
        elif kind == "result":
            bag.update(payload or {})
        elif kind == "fail":
            bag["error"] = payload.get("message")

    stream_solve(path, opts, emit)
    bag["log"] = "\n".join(bag["log"])
    return bag


def main():
    port = int(os.environ.get("IGAOS_PORT", "8420"))
    if BIN is None:
        print("!! No solver binary found.")
        print("   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j")
        print("   or set IGAOS_BIN=/path/to/igaos\n")
    else:
        print("solver:", BIN)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = "http://127.0.0.1:%d" % port
    print("IGAOS console on", url)
    print("Ctrl-C to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
