/**
 * marketIntelligence.ts — Genkit flow for AI gold market intelligence report.
 *
 * This flow replicates the logic in News/news.py but runs inside the
 * Genkit framework so it can:
 *   - Be tested in the Genkit Developer UI (genkit start)
 *   - Be deployed to Cloud Run with tracing, evals, and streaming
 *   - Be called from the FastAPI layer via HTTP
 *
 * Flow input:  { question?: string, includeAlerts?: boolean }
 * Flow output: MarketIntelligenceReport (typed via Zod)
 */

import { z } from "genkit";
import { ai } from "../genkit";

// ---------------------------------------------------------------------------
// Output schema (mirrors News/news.py REPORT_SCHEMA)
// ---------------------------------------------------------------------------
const MarketReportSchema = z.object({
  sentiment: z
    .enum([
      "Strongly Bullish",
      "Bullish",
      "Neutral",
      "Bearish",
      "Strongly Bearish",
    ])
    .describe("Overall market sentiment toward gold"),

  sentiment_score: z
    .number()
    .min(-1)
    .max(1)
    .describe("Numerical sentiment: +1 strongly bullish, -1 strongly bearish"),

  reason: z.string().describe("One-paragraph explanation of the sentiment call"),

  key_drivers: z
    .array(z.string())
    .max(5)
    .describe("Up to 5 bullet-point market drivers"),

  risk_factors: z
    .array(z.string())
    .max(4)
    .describe("Up to 4 risk factors that could reverse the trend"),

  price_outlook: z.object({
    short_term: z.string().describe("24-72 hour price outlook"),
    medium_term: z.string().describe("1-2 week price outlook"),
  }),

  confidence: z
    .enum(["High", "Medium", "Low"])
    .describe("Confidence in the analysis given available information"),
});

export type MarketReport = z.infer<typeof MarketReportSchema>;

// ---------------------------------------------------------------------------
// Flow input schema
// ---------------------------------------------------------------------------
const MarketIntelligenceInput = z.object({
  question: z
    .string()
    .optional()
    .default("What is the current gold market outlook?"),
  includeAlerts: z.boolean().optional().default(false),
});

// ---------------------------------------------------------------------------
// Market Intelligence Flow
// ---------------------------------------------------------------------------
export const marketIntelligenceFlow = ai.defineFlow(
  {
    name: "marketIntelligenceFlow",
    inputSchema: MarketIntelligenceInput,
    outputSchema: MarketReportSchema,
  },
  async (input) => {
    const systemPrompt = `
You are an expert gold market analyst with access to real-time financial news.
Analyse the current gold (XAU/USD) market using up-to-date information.

Provide a comprehensive but concise analysis covering:
1. Current price sentiment and direction
2. Key macro drivers such as USD strength, Fed policy, geopolitical events, and inflation
3. Technical signals if available, such as support/resistance levels and trend direction
4. Risk factors that could reverse the outlook
5. Short-term (24-72h) and medium-term (1-2 week) price outlook

Be objective and evidence-based.
Do not guarantee future prices.
Respond ONLY with valid JSON matching the output schema exactly.
    `.trim();

    const userPrompt = `
Question: ${input.question}

Include alerts in reasoning: ${input.includeAlerts ? "Yes" : "No"}

Provide the gold market intelligence report now.
    `.trim();

    const response = await ai.generate({
      model: "googleai/gemini-2.5-flash",
      system: systemPrompt,
      prompt: userPrompt,
      output: {
        schema: MarketReportSchema,
      },
      config: {
        temperature: 0.3,
      },
    });

    if (!response.output) {
      throw new Error("No structured output returned from marketIntelligenceFlow.");
    }

    return response.output as MarketReport;
  }
);

// ---------------------------------------------------------------------------
// Simple chat flow for direct Q&A
// ---------------------------------------------------------------------------
export const goldChatFlow = ai.defineFlow(
  {
    name: "goldChatFlow",
    inputSchema: z.object({
      question: z.string().min(1).max(2000),
      context: z
        .object({
          current_price: z.number().optional(),
          sentiment: z.string().optional(),
          risk_level: z.string().optional(),
          suggested_action: z.string().optional(),
        })
        .optional(),
    }),
    outputSchema: z.object({
      answer: z.string(),
      disclaimer: z.string(),
    }),
  },
  async (input) => {
    const contextStr = input.context
      ? JSON.stringify(input.context, null, 2)
      : "No live context provided.";

    const outputSchema = z.object({
      answer: z.string(),
      disclaimer: z.string(),
    });

    const systemPrompt = `
You are Spectre Gold AI, an expert gold trading assistant.
Answer the user's question in 3 to 5 clear sentences.
Use the platform context if provided.
Keep the answer practical, concise, and easy to understand.
End with a one-sentence investment disclaimer.
Respond ONLY with valid JSON matching the required schema.
    `.trim();

    const userPrompt = `
Platform context:
${contextStr}

User question:
${input.question}

Respond with JSON in this format:
{
  "answer": "...",
  "disclaimer": "..."
}
    `.trim();

    const response = await ai.generate({
      model: "googleai/gemini-2.5-flash",
      system: systemPrompt,
      prompt: userPrompt,
      output: {
        schema: outputSchema,
      },
      config: {
        temperature: 0.4,
      },
    });

    if (!response.output) {
      throw new Error("No structured output returned from goldChatFlow.");
    }

    return response.output as { answer: string; disclaimer: string };
  }
);