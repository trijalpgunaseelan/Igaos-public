#!/usr/bin/env bash
# Deploy IGAOS Solver Studio to Fly.io. Idempotent: run it again to ship an update.
#
#   ./deploy/fly-deploy.sh                 # uses the app name in deploy/fly.toml
#   ./deploy/fly-deploy.sh my-app-name     # or pick your own
#
# Fly builds the image on their builder, so Docker is not needed on this machine.

set -euo pipefail
cd "$(dirname "$0")/.."                 # repository root
CONFIG="deploy/fly.toml"
APP="${1:-$(sed -n 's/^app *= *"\(.*\)".*/\1/p' "$CONFIG" | head -1)}"
REGION="$(sed -n 's/^primary_region *= *"\(.*\)".*/\1/p' "$CONFIG" | head -1)"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

command -v flyctl >/dev/null || {
  echo "flyctl is not installed. Install it, then run this again:"
  echo "    brew install flyctl        # or: curl -L https://fly.io/install.sh | sh"
  exit 1
}

flyctl auth whoami >/dev/null 2>&1 || {
  say "signing in to Fly"
  flyctl auth login
}
echo "account : $(flyctl auth whoami)"
echo "app     : $APP"
echo "region  : $REGION"

if flyctl apps list 2>/dev/null | awk '{print $1}' | grep -qx "$APP"; then
  say "app $APP already exists — deploying an update"
else
  say "creating $APP"
  flyctl apps create "$APP" || {
    echo
    echo "That name is taken. Pick another:  ./deploy/fly-deploy.sh igaos-<something>"
    exit 1
  }
fi

say "deploying (Fly builds deploy/Dockerfile remotely — first build ~3 min)"
flyctl deploy --config "$CONFIG" --app "$APP" --remote-only --yes

URL="https://$APP.fly.dev"
say "checking $URL/healthz"
for i in $(seq 1 30); do
  if curl -fsS --max-time 10 "$URL/healthz" >/tmp/igaos_health.json 2>/dev/null; then
    cat /tmp/igaos_health.json; echo
    say "live: $URL"
    command -v open >/dev/null && open "$URL"
    exit 0
  fi
  sleep 4
done

echo "health check did not answer in ~2 minutes. Look at the logs:"
echo "    flyctl logs --app $APP"
exit 1
