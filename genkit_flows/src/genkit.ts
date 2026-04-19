/**
 * genkit.ts — Isolated Genkit + Gemini initialisation.
 *
 * This module ONLY creates the `ai` instance.
 * It never imports from flows or index.ts, breaking the circular dependency.
 *
 * All other modules import `ai` from here.
 */

// ⚠️  Must be first — loads .env before any process.env access
import "dotenv/config";

import { genkit } from "genkit";
import { googleAI } from "@genkit-ai/googleai";

// ---------------------------------------------------------------------------
// Debug: confirm which key was found (shown in genkit:ui console)
// ---------------------------------------------------------------------------
const _gak = process.env.GOOGLE_API_KEY;
const _mak = process.env.GEMINI_API_KEY;

console.log("[Spectre Genkit] DEBUG env check:");
console.log(
  `  GOOGLE_API_KEY : ${_gak ? "FOUND (" + _gak.slice(0, 8) + "...)" : "NOT SET"}`
);
console.log(
  `  GEMINI_API_KEY : ${_mak ? "FOUND (" + _mak.slice(0, 8) + "...)" : "NOT SET"}`
);

const RESOLVED_KEY = _gak || _mak || "";
console.log(
  `  Resolved key   : ${RESOLVED_KEY ? "present ✓" : "MISSING — check .env file ✗"}`
);

if (!RESOLVED_KEY) {
  console.warn(
    "[Spectre Genkit] WARNING: No API key found. " +
      "Ensure GOOGLE_API_KEY or GEMINI_API_KEY is set in genkit_flows/.env"
  );
}

// ---------------------------------------------------------------------------
// Genkit instance — exported for all flow files
// ---------------------------------------------------------------------------
export const ai = genkit({
  plugins: [
    googleAI({
      apiKey: RESOLVED_KEY || undefined, // pass undefined so plugin uses its own env fallback
    }),
  ],
  name: "spectre-gold-ai",
});
