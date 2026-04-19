#!/bin/bash
# deploy/deploy_cloud_run.sh
#
# Deploy all Spectre Gold AI services to Google Cloud Run.
#
# Services deployed:
#   1. spectre-gold-agent   — ADK Agent Service  (port 8080)
#   2. spectre-gold-genkit  — Genkit Flow Server (port 4001)
#   3. spectre-gold-backend — FastAPI backend    (port 8080)
#
# Prerequisites:
#   - gcloud CLI authenticated:  gcloud auth login && gcloud auth configure-docker
#   - Docker daemon running
#   - GOOGLE_API_KEY exported in the current shell
#
# Usage (from project root):
#   chmod +x deploy/deploy_cloud_run.sh
#   export GOOGLE_API_KEY=your_key
#   export GCP_PROJECT_ID=your-project-id
#   ./deploy/deploy_cloud_run.sh
#
# Optional: store secrets in Secret Manager instead of plain env vars
#   gcloud secrets create GOOGLE_API_KEY --data-file=<(echo -n "$GOOGLE_API_KEY")
#   Then replace --set-env-vars with --set-secrets=GOOGLE_API_KEY=GOOGLE_API_KEY:latest

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
REGION="${GCP_REGION:-us-central1}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
# Optional Vertex AI Search datastore (leave blank to use Google Search grounding)
VERTEX_SEARCH_DATASTORE_ID="${VERTEX_SEARCH_DATASTORE_ID:-}"

if [[ -z "$GOOGLE_API_KEY" ]]; then
  echo "ERROR: GOOGLE_API_KEY must be set (export GOOGLE_API_KEY=...)."
  exit 1
fi

IMAGE_BASE="gcr.io/${PROJECT_ID}"

echo "=== Spectre Gold AI — Cloud Run Deployment ==="
echo "Project : ${PROJECT_ID}"
echo "Region  : ${REGION}"
echo ""

# ── 1. Enable required APIs ───────────────────────────────────────────────────
gcloud config set project "${PROJECT_ID}" --quiet
gcloud services enable \
  run.googleapis.com \
  containerregistry.googleapis.com \
  aiplatform.googleapis.com \
  --quiet

# ── Helper: build image from a Dockerfile relative to project root ────────────
build_and_push() {
  local name="$1"
  local dockerfile="$2"
  echo "--- Building ${name} ---"
  docker build -f "${dockerfile}" -t "${IMAGE_BASE}/${name}:latest" .
  docker push "${IMAGE_BASE}/${name}:latest"
}

# ── 2. ADK Agent Service ──────────────────────────────────────────────────────
build_and_push "spectre-gold-agent" "agent_service/Dockerfile"

AGENT_URL=$(gcloud run deploy spectre-gold-agent \
  --image   "${IMAGE_BASE}/spectre-gold-agent:latest" \
  --region  "${REGION}" \
  --port    8080 \
  --set-env-vars "GOOGLE_API_KEY=${GOOGLE_API_KEY}" \
  --allow-unauthenticated \
  --memory  2Gi \
  --cpu     2 \
  --min-instances 0 \
  --max-instances 5 \
  --format  "value(status.url)")
echo "ADK Agent URL : ${AGENT_URL}"

# ── 3. Genkit Flow Service ────────────────────────────────────────────────────
build_and_push "spectre-gold-genkit" "genkit_flows/Dockerfile"

GENKIT_ENV_VARS="GOOGLE_API_KEY=${GOOGLE_API_KEY},FLOW_SERVER_PORT=4001,NODE_ENV=production"

GENKIT_URL=$(gcloud run deploy spectre-gold-genkit \
  --image   "${IMAGE_BASE}/spectre-gold-genkit:latest" \
  --region  "${REGION}" \
  --port    4001 \
  --set-env-vars "${GENKIT_ENV_VARS}" \
  --allow-unauthenticated \
  --memory  512Mi \
  --cpu     1 \
  --min-instances 0 \
  --max-instances 3 \
  --format  "value(status.url)")
echo "Genkit URL    : ${GENKIT_URL}"

# ── 4. FastAPI Backend ────────────────────────────────────────────────────────
# Build the backend image (uses a Dockerfile in the project root if present,
# or a minimal inline one). If you have a Backend/Dockerfile, point to it.
if [[ ! -f "Backend/Dockerfile" ]]; then
  echo "--- Creating temporary Backend/Dockerfile ---"
  cat > Backend/Dockerfile <<'BACKEOF'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN touch .env
EXPOSE 8080
RUN useradd -m appuser && chown -R appuser /app
USER appuser
ENV BACKEND_PORT=8080
CMD ["sh", "-c", "uvicorn Backend.main:app --host 0.0.0.0 --port ${BACKEND_PORT:-8080}"]
BACKEOF
fi

build_and_push "spectre-gold-backend" "Backend/Dockerfile"

BACKEND_ENV_VARS="GOOGLE_API_KEY=${GOOGLE_API_KEY}"
BACKEND_ENV_VARS+=",GENKIT_FLOW_URL=${GENKIT_URL}"
BACKEND_ENV_VARS+=",ADK_AGENT_SERVICE_URL=${AGENT_URL}"
BACKEND_ENV_VARS+=",BACKEND_PORT=8080"
if [[ -n "${VERTEX_SEARCH_DATASTORE_ID}" ]]; then
  BACKEND_ENV_VARS+=",VERTEX_SEARCH_DATASTORE_ID=${VERTEX_SEARCH_DATASTORE_ID}"
fi

BACKEND_URL=$(gcloud run deploy spectre-gold-backend \
  --image   "${IMAGE_BASE}/spectre-gold-backend:latest" \
  --region  "${REGION}" \
  --port    8080 \
  --set-env-vars "${BACKEND_ENV_VARS}" \
  --allow-unauthenticated \
  --memory  2Gi \
  --cpu     2 \
  --min-instances 0 \
  --max-instances 5 \
  --format  "value(status.url)")
echo "FastAPI URL   : ${BACKEND_URL}"

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Set these in your Streamlit .env (or Cloud Run env vars for app.py):"
echo "  API_BASE_URL=${BACKEND_URL}"
echo "  GENKIT_FLOW_URL=${GENKIT_URL}"
echo "  ADK_AGENT_SERVICE_URL=${AGENT_URL}"
echo ""
echo "Quick smoke tests:"
echo "  curl ${BACKEND_URL}/docs"
echo "  curl ${GENKIT_URL}/health"
echo "  curl -X POST ${AGENT_URL}/agent/health"
