# Putting IGAOS Solver Studio on the internet

`demo/server.py` binds `127.0.0.1`, allows a 600-second solve and runs as many
solves at once as there are visitors. That is right for a console you stand in
front of and wrong for a public URL, because **every request spawns a native
solver process**. `deploy/serve.py` wraps it without touching it and puts a
ceiling on all three.

Everything here was built and smoke-tested on Linux (GCC 13): the solver
compiles in ~26 s, `blend_m` returns the same objective as on macOS
(`-2135858.97349`), the stream works, and the concurrency gate returns 429.

---

## What the wrapper changes

| | console (`demo/server.py`) | public (`deploy/serve.py`) |
|---|---|---|
| bind | `127.0.0.1` | `0.0.0.0:$PORT` |
| time limit | up to 600 s | `IGAOS_TIME_LIMIT`, default **20 s**, enforced whatever the query asks |
| concurrent solves | unbounded | `IGAOS_MAX_CONCURRENT`, default **2**, then HTTP 429 |
| threads | caller's choice | `IGAOS_THREADS`, default **1** |
| upload | 8 MiB | `IGAOS_MAX_UPLOAD`, default **2 MiB** |
| health | — | `GET /healthz` |

Nothing else is different: the same binary, the same `/api/stream` SSE, the
same page.

Run it locally exactly as you would the console:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
python3 deploy/serve.py            # http://0.0.0.0:8420
```

---

## Option A — Fly.io  (recommended: real CPU, scales to zero)

```bash
brew install flyctl          # or: curl -L https://fly.io/install.sh | sh
flyctl auth login
cd /Users/trijalpg/Downloads/SIH/IGAOS/igaos
flyctl launch --no-deploy --copy-config --name igaos-solver-studio
flyctl deploy
flyctl open
```

`deploy/fly.toml` is already written: Singapore region, port 8420, forced
HTTPS, `/healthz` check, and `auto_stop_machines = "suspend"` so it costs
nothing while nobody is looking at it and wakes on the next request. Change
`app = ` if the name is taken.

## Option B — Render  (simplest, but the free tier sleeps)

Push the repo to GitHub, then in Render: **New → Blueprint → pick the repo**.
`deploy/render.yaml` supplies everything. The `starter` plan is set because a
free instance sleeps after 15 minutes and takes ~40 s to wake — bad in front of
a jury. Switch `plan: starter` to `plan: free` if that is acceptable.

## Option C — Any VPS you already have

```bash
scp -r igaos/ you@your-server:/opt/igaos
ssh you@your-server
cd /opt/igaos
# point a DNS A record at this box, then put the name in deploy/Caddyfile
docker compose -f deploy/docker-compose.yml up -d --build
```

Caddy fetches the TLS certificate itself. The `Caddyfile` disables buffering on
`/api/stream` (SSE breaks behind a buffering proxy) and rate-limits solve
endpoints to 10/minute per IP.

## Option D — Docker anywhere

```bash
docker build -f deploy/Dockerfile -t igaos-studio .
docker run --rm -p 8420:8420 \
  -e IGAOS_TIME_LIMIT=20 -e IGAOS_MAX_CONCURRENT=2 \
  --cpus 1.5 --memory 1g igaos-studio
```

---

## Before it is public

- **The repo is private.** Render and Fly both need access; Fly builds from
  your local checkout, Render needs the GitHub connection.
- **Set a lower `IGAOS_TIME_LIMIT` for a demo** (10 s) if the machine is small.
  A 20-second MILP on one shared vCPU is a slow page.
- **The 19 shipped models are read-only and path-checked** (`model_path()`
  rejects anything outside `demo/models`), so the exposed surface is: upload an
  MPS, run it under a time limit, read the stream. No file writes outside a
  temp file that is deleted after the solve.
- **Put a real rate limit in front** if the URL is shared publicly — the Caddy
  config does this; Fly and Render do not by default.
- Consider a soft gate (a shared password via a Caddy `basicauth` block) if the
  URL goes in the deck before judging.
