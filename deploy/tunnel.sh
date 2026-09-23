#!/usr/bin/env bash
# Put the Solver Studio on the public internet from THIS machine, free, no account.
#
#   ./deploy/tunnel.sh
#
# Cloudflare's quick tunnel hands you an https://<random>.trycloudflare.com URL
# that forwards to the console running here. The solve runs on this Mac — which
# is far faster than any free cloud CPU — and costs nothing. The URL lives as
# long as this command runs, and changes each time you start it.
#
# Ctrl-C stops both the tunnel and the server.

set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${PORT:-8420}"
LOG="$(mktemp -t igaos-tunnel)"

command -v cloudflared >/dev/null || {
  echo "cloudflared is not installed:"
  echo "    brew install cloudflared"
  exit 1
}

if [ ! -x build/igaos ]; then
  echo "==> building the solver first"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build build -j >/dev/null
fi

# public limits, but generous — this machine has real cores
export IGAOS_HOST=127.0.0.1          # only the tunnel can reach it, not the LAN
export PORT
export IGAOS_TIME_LIMIT="${IGAOS_TIME_LIMIT:-30}"
export IGAOS_MAX_CONCURRENT="${IGAOS_MAX_CONCURRENT:-3}"
export IGAOS_THREADS="${IGAOS_THREADS:-2}"

python3 deploy/serve.py &
SERVER=$!
trap 'kill $SERVER $TUNNEL 2>/dev/null || true' EXIT INT TERM
sleep 2
curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null || { echo "server did not start"; exit 1; }

echo "==> opening the tunnel"
cloudflared tunnel --url "http://127.0.0.1:$PORT" --no-autoupdate >"$LOG" 2>&1 &
TUNNEL=$!

for _ in $(seq 1 30); do
  URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [ -n "$URL" ] && break
  sleep 1
done

if [ -z "${URL:-}" ]; then
  echo "no tunnel URL after 30 s — the log:"; tail -20 "$LOG"; exit 1
fi

printf '\n\033[1m  %s\033[0m\n\n' "$URL"
echo "  solver on this machine · ${IGAOS_TIME_LIMIT}s per solve · ${IGAOS_MAX_CONCURRENT} concurrent · ${IGAOS_THREADS} threads"
echo "  Ctrl-C to take it offline."
command -v open >/dev/null && open "$URL"
wait $TUNNEL
