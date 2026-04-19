"""
Agentic Decision Workflow — Spectre Gold AI

Four agents run sequentially to produce an explainable answer:

  Step 1  MarketDataAgent        — snapshots price, forecast, direction, confidence
  Step 2  NewsAnalysisAgent      — snapshots news sentiment and source credibility
  Step 3  RiskGuardAgent         — evaluates signal consistency and volatility
  Step 4  DecisionExplanationAgent — synthesises all steps into a final answer via Gemini

Public entry point:
    from News.agents import WorkflowRunner
    result = WorkflowRunner().run(user_question, platform_context)

Returns:
    {
        "answer":       str,
        "steps":        list[{"agent": str, "status": str, "summary": str, "output": dict}],
        "was_agentic":  bool,
    }
"""
from __future__ import annotations

import importlib
import json
import os
from datetime import datetime
from typing import Any

from dotenv import load_dotenv

from News.risk_guard import evaluate_risk

# ---------------------------------------------------------------------------
# Optional Gemini SDK imports (mirrors chatbot.py pattern)
# ---------------------------------------------------------------------------
try:
    genai = importlib.import_module("google.genai")
except ImportError:
    genai = None

try:
    legacy_genai = importlib.import_module("google.generativeai")
except ImportError:
    legacy_genai = None

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(_ROOT_DIR, ".env")


def _get_google_api_key() -> str | None:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return key.strip() if isinstance(key, str) else None


# ---------------------------------------------------------------------------
# Keyword detection — decides whether to run the full agentic workflow
# ---------------------------------------------------------------------------
_AGENTIC_KEYWORDS = [
    "should i buy", "should i sell", "should i trade", "should i invest",
    "buy gold", "sell gold", "worth buying", "good time to buy", "good time to sell",
    "why is gold", "why gold", "is gold", "gold dropping", "gold falling",
    "gold rising", "gold spike",
    "trustworthy", "reliable", "can i trust", "is the signal", "is it safe",
    "what should i", "recommend", "do you suggest", "advice",
    "is it risky", "high risk", "risk level",
    "dropping", "falling", "rising", "spike", "crash", "surge",
    "should i hold", "should i wait",
]


def is_agentic_question(question: str) -> bool:
    """Return True if the question warrants a full multi-step agent workflow."""
    q = question.lower().strip()
    return any(kw in q for kw in _AGENTIC_KEYWORDS)


# ---------------------------------------------------------------------------
# Agent 1 — MarketDataAgent
# ---------------------------------------------------------------------------
class MarketDataAgent:
    name = "MarketDataAgent"
    description = "Snapshots current price, LSTM forecast, direction and confidence"

    def run(self, platform_context: dict) -> dict[str, Any]:
        try:
            last_price = float(platform_context.get("last_price") or 0)
            tech_pred = float(platform_context.get("technical_predicted_price") or 0)
            final_pred = float(platform_context.get("final_predicted_price") or 0)
            hybrid_change = float(platform_context.get("hybrid_change") or 0)
            sentiment_label = platform_context.get("sentiment_label", "Unknown")

            direction = "UP" if hybrid_change > 0 else "DOWN"
            move_pct = abs(hybrid_change) / max(last_price, 1e-9) * 100
            if move_pct >= 0.45:
                forecast_confidence = "High"
            elif move_pct >= 0.20:
                forecast_confidence = "Medium"
            else:
                forecast_confidence = "Low"

            output = {
                "last_price": round(last_price, 2),
                "technical_predicted_price": round(tech_pred, 2),
                "final_predicted_price": round(final_pred, 2),
                "hybrid_change": round(hybrid_change, 2),
                "direction": direction,
                "forecast_confidence": forecast_confidence,
                "sentiment_label": sentiment_label,
            }
            summary = (
                f"Gold @ ${last_price:,.2f} → forecast ${final_pred:,.2f} "
                f"({direction}, {forecast_confidence} confidence)"
            )
            return {"status": "ok", "output": output, "summary": summary}

        except Exception as exc:
            return {
                "status": "error",
                "output": {"error": str(exc)},
                "summary": f"Market data unavailable: {exc}",
            }


# ---------------------------------------------------------------------------
# Agent 2 — NewsAnalysisAgent
# ---------------------------------------------------------------------------
class NewsAnalysisAgent:
    name = "NewsAnalysisAgent"
    description = "Snapshots news sentiment, credibility, and source quality"

    def run(self, news_analysis: dict | None) -> dict[str, Any]:
        if news_analysis is None:
            return {
                "status": "unavailable",
                "output": {
                    "sentiment": "Neutral",
                    "reason": "No news analysis in current session.",
                    "source_count": 0,
                    "source_status": "unavailable",
                    "fallback_used": False,
                },
                "summary": "No news analysis available — technical data only.",
            }
        try:
            sentiment = news_analysis.get("sentiment", "Neutral")
            reason = (news_analysis.get("reason") or "")[:300]
            sources = news_analysis.get("sources") or []
            source_status = news_analysis.get("source_status", "unavailable")
            fallback_used = bool(news_analysis.get("fallback_used", False))

            output = {
                "sentiment": sentiment,
                "reason": reason,
                "source_count": len(sources),
                "source_status": source_status,
                "fallback_used": fallback_used,
            }
            credibility = "LOW" if (fallback_used or source_status in ("unavailable", "no_grounding")) else "OK"
            summary = (
                f"Sentiment: {sentiment} | Sources: {len(sources)} | "
                f"Credibility: {credibility}"
            )
            return {"status": "ok", "output": output, "summary": summary}

        except Exception as exc:
            return {
                "status": "error",
                "output": {"error": str(exc), "sentiment": "Neutral"},
                "summary": f"News analysis failed: {exc}",
            }


# ---------------------------------------------------------------------------
# Agent 3 — RiskGuardAgent
# ---------------------------------------------------------------------------
class RiskGuardAgent:
    name = "RiskGuardAgent"
    description = "Evaluates signal conflict, news credibility, and market volatility"

    def run(
        self,
        market_output: dict,
        news_output: dict,
        platform_context: dict,
    ) -> dict[str, Any]:
        try:
            # Use pre-computed risk_result if already in platform context
            pre = platform_context.get("risk_result")
            if pre and isinstance(pre, dict) and pre.get("risk_level"):
                summary = (
                    f"Risk: {pre['risk_level']} | "
                    f"Confidence: {int(pre.get('confidence_score', 0) * 100)}% | "
                    f"Warnings: {', '.join(pre.get('warnings', [])) or 'none'}"
                )
                return {"status": "ok", "output": pre, "summary": summary}

            # Compute fresh
            sentiment_adj_pct = float(platform_context.get("sentiment_adjustment_pct") or 0)
            result = evaluate_risk(
                sentiment=news_output.get("sentiment", "Neutral"),
                sentiment_score=sentiment_adj_pct / 100,
                hybrid_change=market_output.get("hybrid_change", 0),
                current_price=market_output.get("last_price", 0),
                prev_price=float(platform_context.get("prev_price") or market_output.get("last_price") or 0),
                news_analysis=platform_context.get("news_analysis"),
            )
            summary = (
                f"Risk: {result['risk_level']} | "
                f"Confidence: {int(result.get('confidence_score', 0) * 100)}% | "
                f"Warnings: {', '.join(result.get('warnings', [])) or 'none'}"
            )
            return {"status": "ok", "output": result, "summary": summary}

        except Exception as exc:
            return {
                "status": "error",
                "output": {
                    "risk_level": "Unknown",
                    "confidence_score": 0.5,
                    "factors": [],
                    "warnings": [],
                    "summary": "Risk evaluation failed.",
                    "error": str(exc),
                },
                "summary": f"Risk evaluation failed: {exc}",
            }


# ---------------------------------------------------------------------------
# Agent 4 — DecisionExplanationAgent
# ---------------------------------------------------------------------------
_DECISION_SYSTEM_PROMPT = """
You are Spectre Gold AI Decision Engine, an agentic AI assistant for a professional gold trading platform.

You have completed a 3-step pipeline:
1. MarketDataAgent    — captured current price, forecast direction, and confidence
2. NewsAnalysisAgent  — captured news sentiment and source credibility
3. RiskGuardAgent     — evaluated signal consistency, volatility, and overall risk level

Your job:
- Directly answer the user's question using the analysis data provided.
- Be specific: reference actual prices, sentiment labels, and risk levels from the data.
- Explain the reasoning in 4-8 sentences, conversational and clear.
- If risk is Medium or High, mention it and give one brief caution.
- Do not invent any data. Use only what is in the analysis JSON.
- Do not use excessive bullet points. Paragraph format preferred.
- Do not give guaranteed financial advice. One brief disclaimer at the end is enough.
""".strip()


class DecisionExplanationAgent:
    name = "DecisionExplanationAgent"
    description = "Synthesises market, news, and risk data into an explainable answer via Gemini"

    def run(
        self,
        user_question: str,
        market_output: dict,
        news_output: dict,
        risk_output: dict,
    ) -> dict[str, Any]:
        api_key = _get_google_api_key()
        if not api_key:
            return {
                "status": "fallback",
                "output": {"answer": self._offline_answer(user_question, market_output, news_output, risk_output)},
                "summary": "Gemini unavailable — offline synthesis used",
            }

        analysis_payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "user_question": user_question,
            "step1_market": market_output,
            "step2_news": news_output,
            "step3_risk": {
                "risk_level": risk_output.get("risk_level"),
                "confidence_score": risk_output.get("confidence_score"),
                "warnings": risk_output.get("warnings", []),
                "factors": risk_output.get("factors", []),
                "summary": risk_output.get("summary", ""),
            },
        }

        prompt = (
            f"{_DECISION_SYSTEM_PROMPT}\n\n"
            "Multi-agent analysis JSON:\n"
            f"{json.dumps(analysis_payload, ensure_ascii=True)}\n\n"
            "Answer the user's question based on this data."
        )

        try:
            answer = ""
            if genai is not None:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                answer = (getattr(response, "text", "") or "").strip()

            if not answer and legacy_genai is not None:
                legacy_genai.configure(api_key=api_key)
                model = legacy_genai.GenerativeModel(model_name="gemini-2.5-flash")
                response = model.generate_content(prompt)
                answer = (getattr(response, "text", "") or "").strip()

            if not answer:
                answer = self._offline_answer(user_question, market_output, news_output, risk_output)

            return {
                "status": "ok",
                "output": {"answer": answer},
                "summary": "Decision synthesised via Gemini",
            }

        except Exception as exc:
            return {
                "status": "fallback",
                "output": {"answer": self._offline_answer(user_question, market_output, news_output, risk_output)},
                "summary": f"Gemini failed ({exc}) — offline synthesis used",
            }

    @staticmethod
    def _offline_answer(
        user_question: str,
        market_output: dict,
        news_output: dict,
        risk_output: dict,
    ) -> str:
        direction = market_output.get("direction", "unknown")
        last_price = market_output.get("last_price", "N/A")
        final_pred = market_output.get("final_predicted_price", "N/A")
        confidence = market_output.get("forecast_confidence", "unknown")
        sentiment = news_output.get("sentiment", "Neutral")
        risk_level = risk_output.get("risk_level", "Unknown")
        risk_summary = risk_output.get("summary", "")

        return (
            f"Based on the current platform data: gold is at ${last_price:,.2f} "
            f"with the AI forecasting a move {direction} to ${final_pred:,.2f} "
            f"({confidence} confidence). "
            f"News sentiment is {sentiment}. "
            f"Risk assessment: {risk_level}. "
            f"{risk_summary} "
            "Live AI synthesis is temporarily unavailable — please retry shortly."
        )


# ---------------------------------------------------------------------------
# WorkflowRunner — orchestrates all 4 agents
# ---------------------------------------------------------------------------
class WorkflowRunner:
    """
    Orchestrates the 4-agent decision workflow.

    Usage:
        result = WorkflowRunner().run(user_question, platform_context)

    platform_context must contain at minimum:
        last_price, technical_predicted_price, final_predicted_price,
        hybrid_change, sentiment_label, sentiment_adjustment_pct,
        news_analysis (dict or None), risk_result (dict or None), prev_price
    """

    def __init__(self) -> None:
        self._market_agent = MarketDataAgent()
        self._news_agent = NewsAnalysisAgent()
        self._risk_agent = RiskGuardAgent()
        self._decision_agent = DecisionExplanationAgent()

    def run(self, user_question: str, platform_context: dict) -> dict[str, Any]:
        """
        Run the full agentic workflow if the question warrants it.
        Falls back to a simple summary if Gemini is unavailable at all steps.

        Returns:
            {
                "answer":       str,
                "steps":        list[{"agent", "status", "summary", "output"}],
                "was_agentic":  bool,
            }
        """
        if not is_agentic_question(user_question):
            return {"answer": None, "steps": [], "was_agentic": False}

        steps: list[dict] = []

        # Step 1 — market data
        market_step = self._market_agent.run(platform_context)
        steps.append({
            "agent": MarketDataAgent.name,
            "status": market_step["status"],
            "summary": market_step["summary"],
            "output": market_step["output"],
        })
        market_out = market_step["output"]

        # Step 2 — news analysis
        news_analysis = platform_context.get("news_analysis") if isinstance(platform_context, dict) else None
        news_step = self._news_agent.run(news_analysis)
        steps.append({
            "agent": NewsAnalysisAgent.name,
            "status": news_step["status"],
            "summary": news_step["summary"],
            "output": news_step["output"],
        })
        news_out = news_step["output"]

        # Step 3 — risk guard
        risk_step = self._risk_agent.run(market_out, news_out, platform_context)
        steps.append({
            "agent": RiskGuardAgent.name,
            "status": risk_step["status"],
            "summary": risk_step["summary"],
            "output": risk_step["output"],
        })
        risk_out = risk_step["output"]

        # Step 4 — decision explanation via Gemini
        decision_step = self._decision_agent.run(user_question, market_out, news_out, risk_out)
        steps.append({
            "agent": DecisionExplanationAgent.name,
            "status": decision_step["status"],
            "summary": decision_step["summary"],
            "output": decision_step["output"],
        })
        answer = decision_step["output"].get("answer", "")

        return {"answer": answer, "steps": steps, "was_agentic": True}
