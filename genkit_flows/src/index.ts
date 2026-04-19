/**
 * index.ts — Genkit app entry point for Spectre Gold AI.
 *
 * Structure:
 *   src/genkit.ts              ← ai instance + dotenv (no flow imports)
 *   src/index.ts               ← this file: imports flows to register them + starts HTTP server
 *   src/flows/marketIntelligence.ts  ← imports ai from ../genkit
 *
 * Run:
 *   npm run dev          (tsx src/index.ts)          → HTTP server only on port 4001
 *   npm run genkit:ui    (genkit start -- tsx src/index.ts) → HTTP + Genkit Dev UI
 *
 * FastAPI calls: POST http://localhost:4001/marketIntelligenceFlow
 *                POST http://localhost:4001/goldChatFlow
 */

import http from "http";

// genkit.ts handles dotenv + ai init
import "./genkit";

// Importing flows registers them with the Genkit registry automatically
import { marketIntelligenceFlow, goldChatFlow } from "./flows/marketIntelligence";

// Re-export for external use (tests, server.ts, etc.)
export { marketIntelligenceFlow, goldChatFlow };

console.log("[Spectre Genkit] Flows registered: marketIntelligenceFlow, goldChatFlow");

// ---------------------------------------------------------------------------
// Minimal HTTP REST server — exposes flows so FastAPI can call them
// ---------------------------------------------------------------------------
const FLOW_PORT = parseInt(process.env.FLOW_SERVER_PORT ?? "4001", 10);

const FLOW_MAP: Record<string, (input: unknown) => Promise<unknown>> = {
  "/marketIntelligenceFlow": (input) => marketIntelligenceFlow(input as Parameters<typeof marketIntelligenceFlow>[0]),
  "/goldChatFlow":           (input) => goldChatFlow(input as Parameters<typeof goldChatFlow>[0]),
};

const server = http.createServer(async (req, res) => {
  const url = req.url ?? "/";

  // Health check
  if (req.method === "GET" && url === "/health") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", flows: Object.keys(FLOW_MAP) }));
    return;
  }

  const flowFn = req.method === "POST" ? FLOW_MAP[url] : undefined;
  if (!flowFn) {
    res.writeHead(404, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: `No flow at ${req.method} ${url}` }));
    return;
  }

  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", async () => {
    try {
      const parsed   = JSON.parse(body || "{}");
      // Accept both { data: {...} } (FastAPI convention) and raw input
      const flowInput = parsed.data !== undefined ? parsed.data : parsed;

      console.log(`[Spectre Genkit] → ${url} called — input:`, JSON.stringify(flowInput).slice(0, 200));
      const result = await flowFn(flowInput);
      console.log(`[Spectre Genkit] ✓ ${url} success`);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ result }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.error(`[Spectre Genkit] ✗ ${url} error:`, msg);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: msg }));
    }
  });
});

server.listen(FLOW_PORT, () => {
  console.log(`[Spectre Genkit] Flow HTTP server listening on http://localhost:${FLOW_PORT}`);
  console.log(`[Spectre Genkit] Endpoints:`);
  Object.keys(FLOW_MAP).forEach((path) =>
    console.log(`  POST http://localhost:${FLOW_PORT}${path}`)
  );
});

