#!/usr/bin/env python3
"""
Public deployment wrapper for the IGAOS Solver Studio.

demo/server.py is a console to stand in front of: it binds 127.0.0.1, allows a
600-second solve and as many concurrent solves as there are visitors. On the
open internet each of those is a way to burn the box down, because every
request spawns a native solver process.

This wrapper leaves demo/server.py untouched and puts limits around it:

    binds 0.0.0.0:$PORT          so a container/PaaS can route to it
    IGAOS_TIME_LIMIT   20 s      hard ceiling per solve, whatever the query asks
    IGAOS_MAX_CONCURRENT 2       further solves get 429 instead of a queue
    IGAOS_THREADS      1         a visitor cannot ask for 64 threads
    IGAOS_MAX_UPLOAD   2 MiB     model upload cap
    /healthz                     for the platform's health check

Run it exactly like the console:  python3 deploy/serve.py
"""
import importlib.util
import json
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "igaos_console", os.path.join(ROOT, "demo", "server.py"))
console = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(console)

TIME_LIMIT = float(os.environ.get("IGAOS_TIME_LIMIT", "20"))
MAX_CONC = int(os.environ.get("IGAOS_MAX_CONCURRENT", "2"))
THREADS = int(os.environ.get("IGAOS_THREADS", "1"))
MAX_UPLOAD = int(os.environ.get("IGAOS_MAX_UPLOAD", str(2 << 20)))
PORT = int(os.environ.get("PORT") or os.environ.get("IGAOS_PORT") or 8420)
HOST = os.environ.get("IGAOS_HOST", "0.0.0.0")

console.HARD_TIME_LIMIT = TIME_LIMIT
console.MAX_UPLOAD = MAX_UPLOAD

_slots = threading.BoundedSemaphore(MAX_CONC)
_started = time.time()


class PublicHandler(console.Handler):
    server_version = "igaos-studio"
    sys_version = ""

    def _busy(self):
        self._send(429, {"error": "busy — %d solves already running, try again in "
                                  "a moment" % MAX_CONC})

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/healthz", "/health"):
            return self._send(200, {
                "ok": console.BIN is not None,
                "binary": os.path.basename(console.BIN or ""),
                "uptime_s": round(time.time() - _started, 1),
                "limits": {"time_limit_s": TIME_LIMIT, "max_concurrent": MAX_CONC,
                           "threads": THREADS, "max_upload_bytes": MAX_UPLOAD},
            })
        return super().do_GET()

    def do_stream(self, q):
        # a visitor does not get to choose the machine's workload
        q = dict(q)
        q["threads"] = [str(THREADS)]
        try:
            asked = float((q.get("timeLimit") or [TIME_LIMIT])[0] or TIME_LIMIT)
        except ValueError:
            asked = TIME_LIMIT
        q["timeLimit"] = [str(min(max(asked, 0.1), TIME_LIMIT))]
        if not _slots.acquire(blocking=False):
            return self._busy()
        try:
            return super().do_stream(q)
        finally:
            _slots.release()

    def do_POST(self):
        # /api/compare runs 2 solves, /api/algorithms runs 4 — one slot covers the set
        if not _slots.acquire(blocking=False):
            return self._busy()
        try:
            return super().do_POST()
        finally:
            _slots.release()


def main():
    if console.BIN is None:
        print("!! no solver binary — set IGAOS_BIN or build it first", file=sys.stderr)
        sys.exit(1)
    print("igaos     :", console.BIN)
    print("listening :", "http://%s:%d" % (HOST, PORT))
    print("limits    : %.0fs per solve · %d concurrent · %d thread(s) · %d KiB upload"
          % (TIME_LIMIT, MAX_CONC, THREADS, MAX_UPLOAD // 1024), flush=True)
    ThreadingHTTPServer((HOST, PORT), PublicHandler).serve_forever()


if __name__ == "__main__":
    main()
