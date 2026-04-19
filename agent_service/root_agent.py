"""
root_agent.py — ADK Root Agent for Spectre Gold AI.

Defines the Agent object that orchestrates the 5 tool wrappers in the same
logical order as the current WorkflowRunner (Market → News → Risk → Decision).

When google-adk is installed this is a real ADK agent.
When it is not installed, the module exports a lightweight compatible shim
so the rest of the codebase does not break during local development.

Usage (with ADK installed):
    from agent_service.root_agent import root_agent
    # register with ADK Runner or deploy via `adk deploy`

Usage (standalone / fallback):
    from agent_service.root_agent import root_agent
    result = root_agent.run_sync(question, platform_context)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_service.tools import (
    get_adk_tools,
    market_snapshot_tool,
    risk_guard_tool,
    suggested_action_tool,
    smart_alerts_tool,
    market_intelligence_tool,
)
from agent_service.llm_client import generate_text

# ---------------------------------------------------------------------------
# System instruction for the ADK agent
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTION = """
You are Spectre Gold AI, an intelligent gold market assistant built on Google ADK.

When a user asks a buy/sell/risk/market question, follow this exact reasoning sequence:

Step 1 — Call market_snapshot_tool to get live price and LSTM technical forecast.
Step 2 — If news data is not provided in context, call market_intelligence_tool.
Step 3 — Call risk_guard_tool with price data and sentiment from Step 1-2.
Step 4 — Call suggested_action_tool to get a structured Buy/Sell/Hold/Wait recommendation.
Step 5 — Call smart_alerts_tool to check for any active alert conditions.
Step 6 — Synthesise a clear, honest answer using all tool outputs.

Style rules:
- Be specific: reference actual prices, risk levels, and confidence scores from tool outputs.
- Write in clear paragraphs (3-8 sentences), not bullet lists.
- If risk is Medium or High, mention it and give one brief caution.
- Never invent price data. Use only what tools return.
- This is informational only — do not guarantee profits.
- One brief disclaimer at the end is enough.
""".strip()


# ---------------------------------------------------------------------------
# Try to build a real ADK Agent; fall back to a compatible shim
# ---------------------------------------------------------------------------
try:
    from google.adk.agents import Agent  # type: ignore

    root_agent = Agent(
        name="spectre_gold_root_agent",
        model="gemini-2.5-flash",
        description=(
            "Spectre Gold AI root agent — orchestrates market snapshot, "
            "risk guard, suggested action, smart alerts, and market intelligence."
        ),
        instruction=_SYSTEM_INSTRUCTION,
        tools=get_adk_tools(),
    )

    _ADK_AVAILABLE = True

except ImportError:
    _ADK_AVAILABLE = False

    # ── Shim: mirrors the ADK Agent interface used by server.py ─────────────
    class _SpectreAgentShim:
        """
        Lightweight fallback that runs the same 5-step orchestration without
        the ADK framework.  Compatible with server.py's run_sync() call.
        """

        name = "spectre_gold_root_agent"

        def run_sync(self, question: str, platform_context: dict) -> dict[str, Any]:
            """
            Execute the 5-step workflow and return the same dict shape as
            the ADK Runner would produce.
            """
            steps: list[dict] = []
            errors: list[str] = []

            # Step 1 — market snapshot
            try:
                snap = market_snapshot_tool(
                    sentiment_label=platform_context.get("sentiment_label", "Neutral (Technical Only)"),
                    sentiment_score=float(platform_context.get("sentiment_adjustment_pct", 0) / 100),
                )
                steps.append({"tool": "market_snapshot_tool", "status": "ok", "output": snap})
            except Exception as exc:
                snap = {}
                steps.append({"tool": "market_snapshot_tool", "status": "error", "output": str(exc)})
                errors.append(str(exc))

            # Step 2 — market intelligence (use cached if available)
            news_analysis = platform_context.get("news_analysis") if isinstance(platform_context, dict) else None
            if not news_analysis:
                try:
                    news_analysis = market_intelligence_tool()
                    steps.append({"tool": "market_intelligence_tool", "status": "ok", "output": news_analysis})
                except Exception as exc:
                    steps.append({"tool": "market_intelligence_tool", "status": "error", "output": str(exc)})
                    errors.append(str(exc))

            sentiment = (news_analysis or {}).get("sentiment", "Neutral")

            # Step 3 — risk guard
            try:
                risk = risk_guard_tool(
                    current_price=snap.get("last_price", 0),
                    prev_price=snap.get("prev_price", 0),
                    hybrid_change=snap.get("hybrid_change", 0),
                    sentiment=sentiment,
                    sentiment_score=float(platform_context.get("sentiment_adjustment_pct", 0) / 100),
                    news_analysis=news_analysis,
                )
                steps.append({"tool": "risk_guard_tool", "status": "ok", "output": risk})
            except Exception as exc:
                risk = {"risk_level": "Unknown", "confidence_score": 0.5, "warnings": [], "summary": ""}
                steps.append({"tool": "risk_guard_tool", "status": "error", "output": str(exc)})
                errors.append(str(exc))

            # Step 4 — suggested action
            try:
                action = suggested_action_tool(
                    current_price=snap.get("last_price", 0),
                    prev_price=snap.get("prev_price", 0),
                    hybrid_change=snap.get("hybrid_change", 0),
                    sentiment=sentiment,
                    sentiment_score=float(platform_context.get("sentiment_adjustment_pct", 0) / 100),
                    risk_result=risk,
                    news_analysis=news_analysis,
                )
                steps.append({"tool": "suggested_action_tool", "status": "ok", "output": action})
            except Exception as exc:
                action = {}
                steps.append({"tool": "suggested_action_tool", "status": "error", "output": str(exc)})
                errors.append(str(exc))

            # Step 5 — smart alerts
            try:
                alerts = smart_alerts_tool(
                    current_price=snap.get("last_price", 0),
                    prev_price=snap.get("prev_price", 0),
                    sentiment=sentiment,
                    prev_sentiment=platform_context.get("prev_sentiment", "Unknown"),
                    risk_result=risk,
                    news_analysis=news_analysis,
                )
                steps.append({"tool": "smart_alerts_tool", "status": "ok", "output": alerts})
            except Exception as exc:
                alerts = []
                steps.append({"tool": "smart_alerts_tool", "status": "error", "output": str(exc)})

            # Step 6 — LLM synthesis
            analysis_payload = {
                "user_question": question,
                "market_snapshot": snap,
                "sentiment": sentiment,
                "risk": {
                    "risk_level": risk.get("risk_level"),
                    "confidence_score": risk.get("confidence_score"),
                    "warnings": risk.get("warnings", []),
                    "summary": risk.get("summary", ""),
                },
                "suggested_action": {
                    "action": action.get("suggested_action"),
                    "confidence": action.get("confidence"),
                    "rationale": action.get("rationale"),
                },
                "active_alerts": len(alerts),
            }

            prompt = (
                f"{_SYSTEM_INSTRUCTION}\n\n"
                "Multi-step analysis output:\n"
                f"{json.dumps(analysis_payload, ensure_ascii=True)}\n\n"
                "Answer the user's question based on this data."
            )

            answer = generate_text(prompt)
            if not answer:
                # Offline fallback
                answer = (
                    f"Gold is currently at ${snap.get('last_price', 'N/A'):,.2f}. "
                    f"The AI hybrid forecast is "
                    f"${snap.get('hybrid_predicted_price', 'N/A'):,.2f} "
                    f"({snap.get('direction', 'unknown')} direction, "
                    f"{snap.get('forecast_confidence', 'unknown')} confidence). "
                    f"News sentiment: {sentiment}. "
                    f"Risk level: {risk.get('risk_level', 'Unknown')}. "
                    f"Suggested action: {action.get('suggested_action', 'Wait')}. "
                    f"{risk.get('summary', '')} "
                    "Live AI synthesis is temporarily unavailable."
                )

            return {
                "answer": answer,
                "steps": steps,
                "was_agentic": True,
                "adk_native": False,
            }

    root_agent = _SpectreAgentShim()
