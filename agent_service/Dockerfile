# ADK Agent Service — Dockerfile
#
# Builds the standalone agent service (agent_service/) as a Cloud Run image.
#
# Build:
#   docker build -f agent_service/Dockerfile -t spectre-gold-agent-service .
#
# Run locally:
#   docker run -p 8080:8080 -e GOOGLE_API_KEY=your_key spectre-gold-agent-service
#
# Deploy to Cloud Run:
#   gcloud run deploy spectre-gold-agent-service \
#     --image gcr.io/PROJECT_ID/spectre-gold-agent-service \
#     --port 8080 \
#     --set-env-vars GOOGLE_API_KEY=your_key

FROM python:3.11-slim

# System dependencies for TensorFlow and yfinance
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY agent_service/requirements.txt ./agent_service/requirements.txt
RUN pip install --no-cache-dir -r agent_service/requirements.txt

# Copy project files needed by the agent service
COPY agent_service/ ./agent_service/
COPY News/ ./News/
COPY Models/ ./Models/
COPY Backend/ ./Backend/

# Create a minimal .env placeholder (real values injected via Cloud Run secrets)
RUN touch .env

EXPOSE 8080

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["uvicorn", "agent_service.server:app", "--host", "0.0.0.0", "--port", "8080"]
