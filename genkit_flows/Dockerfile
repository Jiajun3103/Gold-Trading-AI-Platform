# Genkit Flows Service — Dockerfile
#
# Builds the TypeScript Genkit flow server as a Cloud Run image.
# The Node HTTP server listens on FLOW_SERVER_PORT (default 4001).
# Genkit Dev UI (port 4000) is NOT started in production.
#
# Build (from project root):
#   docker build -f genkit_flows/Dockerfile -t spectre-gold-genkit .
#
# Run locally:
#   docker run -p 4001:4001 -e GOOGLE_API_KEY=your_key spectre-gold-genkit
#
# Cloud Run (PORT is injected automatically):
#   gcloud run deploy ... --port 4001

FROM node:20-slim

WORKDIR /app

# Copy package files first for layer caching
COPY genkit_flows/package*.json ./
RUN npm ci --omit=dev

# Copy source and compile TypeScript
COPY genkit_flows/ .
RUN npm run build

# Cloud Run injects PORT; our index.ts reads FLOW_SERVER_PORT with fallback to PORT then 4001
ENV NODE_ENV=production
ENV FLOW_SERVER_PORT=4001

EXPOSE 4001

# Health check — Cloud Run also probes / automatically
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD node -e "require('http').get('http://localhost:' + (process.env.FLOW_SERVER_PORT||4001) + '/health', r => process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Run the compiled flow server directly (no genkit CLI in production)
CMD ["node", "dist/index.js"]
