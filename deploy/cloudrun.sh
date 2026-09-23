#!/usr/bin/env bash
# Deploy to Google Cloud Run — always-on, scales to zero, inside the free tier.
#
#   ./deploy/cloudrun.sh                 # uses project from `gcloud config`
#   ./deploy/cloudrun.sh my-project-id
#
# Free tier: 180,000 vCPU-seconds and 2M requests a month, forever. A billing
# account (card) must be attached to the project, but nothing is charged while
# you stay inside the allowance — and the service scales to zero between demos.

set -euo pipefail
cd "$(dirname "$0")/.."
SERVICE="${SERVICE:-igaos-solver-studio}"
REGION="${REGION:-asia-south1}"          # Mumbai
PROJECT="${1:-$(gcloud config get-value project 2>/dev/null)}"

command -v gcloud >/dev/null || {
  echo "gcloud is not installed:  brew install --cask google-cloud-sdk"; exit 1; }
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || {
  echo "no project. Run:  gcloud auth login && gcloud config set project <id>"; exit 1; }

echo "project : $PROJECT"
echo "service : $SERVICE  ($REGION)"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

IMAGE="gcr.io/$PROJECT/$SERVICE"
echo "==> building the image with Cloud Build"
gcloud builds submit --project "$PROJECT" --config deploy/cloudbuild.yaml \
  --substitutions=_IMAGE="$IMAGE" .

echo "==> deploying"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" \
  --platform managed --allow-unauthenticated \
  --port 8420 --cpu 1 --memory 512Mi \
  --min-instances 0 --max-instances 3 --concurrency 4 --timeout 300 \
  --set-env-vars IGAOS_TIME_LIMIT=20,IGAOS_MAX_CONCURRENT=2,IGAOS_THREADS=1

URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format='value(status.url)')"
echo
curl -fsS "$URL/healthz" && echo
printf '\n\033[1m  %s\033[0m\n\n' "$URL"
