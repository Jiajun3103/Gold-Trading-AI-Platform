"""
tools.py — ADK-compatible tool wrappers for Spectre Gold AI.

Each function is a pure Python callable that:
  - wraps existing deterministic business logic (no duplication)
  - can be decorated with google.adk.tools.FunctionTool when ADK is installed
  - works standalone (no ADK required) — the server calls them directly too

Existing modules imported (not copied):
  News/risk_guard.py       → evaluate_risk()
  News/decision_engine.py  → build_suggested_action()
  News/smart_alerts.py     → detect_alerts()
  News/news.py             → get_ai_market_report()
"""
from __future__ import annotations

import os
import sys
from typing import Any

# Make sure the project root is on the path so News.* imports work
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from News.risk_guard import evaluate_risk
from News.decision_engine import build_suggested_action
from News.smart_alerts import detect_alerts
from News.news import get_ai_market_report


# ---------------------------------------------------------------------------
# Tool 1 — market_snapshot_tool
# ---------------------------------------------------------------------------
def market_snapshot_tool(
    sentiment_label: str = "Neutral (Technical Only)",
    sentiment_score: float = 0.0,
) -> dict[str, Any]:
    """
    Fetch live gold price via yfinance, run the LSTM model, and compute the
    hybrid forecast.

    Args:
        sentiment_label: Current news sentiment label (from a prior news call).
        sentiment_score: Fractional sentiment adjustment, e.g. +0.005 / -0.005.

    Returns a dict with:
        last_price, prev_price, lstm_technical_prediction,
        hybrid_predicted_price, hybrid_change, direction,
        forecast_confidence, sentiment_label
    """
    import numpy as np
    import tensorflow as tf
    import joblib
    import yfinance as yf

    model_path  = os.path.join(_PROJECT_ROOT, "Models", "gold_price_model.keras")
    scaler_path = os.path.join(_PROJECT_ROOT, "Models", "gold_scaler.bin")

    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    df = yf.Ticker("GC=F").history(period="100d").reset_index()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    closes = df["Close"].values

    last_price = float(closes[-1])
    prev_price = float(closes[-2]) if len(closes) > 1 else last_price

    scaled  = scaler.transform(closes[-60:].reshape(-1, 1))
    pred_sc = model.predict(scaled.reshape(1, 60, 1), verbose=0)
    tech_price = float(scaler.inverse_transform(pred_sc)[0][0])

    hybrid_price  = tech_price * (1 + sentiment_score)
    hybrid_change = hybrid_price - last_price

    move_pct = abs(hybrid_change) / max(last_price, 1e-9) * 100
    if move_pct >= 0.45:
        confidence = "High"
    elif move_pct >= 0.20:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "last_price":                round(last_price, 2),
        "prev_price":                round(prev_price, 2),
        "lstm_technical_prediction": round(tech_price, 2),
        "hybrid_predicted_price":    round(hybrid_price, 2),
        "hybrid_change":             round(hybrid_change, 2),
        "direction":                 "UP" if hybrid_change > 0 else "DOWN",
        "forecast_confidence":       confidence,
        "sentiment_label":           sentiment_label,
    }


# ---------------------------------------------------------------------------
# Tool 2 — risk_guard_tool
# ---------------------------------------------------------------------------
def risk_guard_tool(
    current_price: float,
    prev_price: float,
    hybrid_change: float,
    sentiment: str = "Neutral",
    sentiment_score: float = 0.0,
    news_analysis: dict | None = None,
) -> dict[str, Any]:
    """
    Evaluate market risk using the deterministic Risk Guard engine.

    Returns a risk dict with keys:
        risk_level, confidence_score, factors, warnings, summary
    """
    return evaluate_risk(
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        hybrid_change=hybrid_change,
        current_price=current_price,
        prev_price=prev_price,
        news_analysis=news_analysis,
    )


# ---------------------------------------------------------------------------
# Tool 3 — suggested_action_tool
# ---------------------------------------------------------------------------
def suggested_action_tool(
    current_price: float,
    prev_price: float,
    hybrid_change: float,
    sentiment: str = "Neutral",
    sentiment_score: float = 0.0,
    risk_result: dict | None = None,
    news_analysis: dict | None = None,
) -> dict[str, Any]:
    """
    Generate a Buy / Sell / Hold / Wait action card from the decision engine.

    Returns a dict with keys:
        suggested_action, confidence, confidence_score,
        risk_level, supporting_factors, contradicting_factors, rationale
    """
    return build_suggested_action(
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        hybrid_change=hybrid_change,
        current_price=current_price,
        prev_price=prev_price,
        risk_result=risk_result or {},
        news_analysis=news_analysis,
    )


# ---------------------------------------------------------------------------
# Tool 4 — smart_alerts_tool
# ---------------------------------------------------------------------------
def smart_alerts_tool(
    current_price: float,
    prev_price: float,
    sentiment: str = "Neutral",
    prev_sentiment: str = "Unknown",
    risk_result: dict | None = None,
    news_analysis: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Detect active smart alerts from live platform signals.

    Returns a list of alert dicts, each with:
        id, severity, icon, title, message
    """
    return detect_alerts(
        current_price=current_price,
        prev_price=prev_price,
        sentiment=sentiment,
        prev_sentiment=prev_sentiment,
        risk_result=risk_result or {},
        news_analysis=news_analysis,
    )


# ---------------------------------------------------------------------------
# Tool 5 — market_intelligence_tool
# ---------------------------------------------------------------------------
def market_intelligence_tool() -> dict[str, Any]:
    """
    Generate a real-time AI market intelligence report using Gemini with
    Google Search grounding.

    Returns a dict with keys:
        sentiment, reason, full_report, sources, source_status,
        fallback_used, sdk
    """
    return get_ai_market_report()


# ---------------------------------------------------------------------------
# ADK FunctionTool registration (only when google-adk is installed)
# ---------------------------------------------------------------------------
def get_adk_tools() -> list:
    """
    Return a list of ADK FunctionTool objects wrapping the 5 tools above.
    Returns an empty list if google-adk is not installed (graceful degradation).
    """
    try:
        from google.adk.tools import FunctionTool  # type: ignore
        return [
            FunctionTool(market_snapshot_tool),
            FunctionTool(risk_guard_tool),
            FunctionTool(suggested_action_tool),
            FunctionTool(smart_alerts_tool),
            FunctionTool(market_intelligence_tool),
        ]
    except ImportError:
        return []
