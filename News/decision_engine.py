"""
Explainable Suggested Action Engine — Spectre Gold AI

Combines forecast data, news sentiment, and Risk Guard output into a single
structured decision card — no external API calls, always instant.

Public entry point:
    from News.decision_engine import build_suggested_action
    card = build_suggested_action(
        sentiment, sentiment_score, hybrid_change,
        current_price, prev_price, risk_result, news_analysis
    )

Returns:
    {
        "suggested_action":    "Buy" | "Hold" | "Sell" | "Wait",
        "confidence":          "High" | "Medium" | "Low",
        "confidence_score":    float,         # 0.0 – 1.0
        "risk_level":          "Low" | "Medium" | "High" | "Unknown",
        "supporting_factors":  list[str],
        "contradicting_factors": list[str],
        "rationale":           str,           # one-sentence plain-English summary
    }

Decision matrix (rows = forecast direction, cols = risk level):

              Low risk   Medium risk   High risk
  UP forecast   Buy        Hold          Wait
  DOWN forecast Sell       Hold          Wait
  Flat          Hold       Hold          Wait

Sentiment further tightens confidence:
  - Matching sentiment boosts confidence one tier
  - Opposing sentiment drops confidence one tier
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forecast_direction(hybrid_change: float) -> str:
    if hybrid_change > 0.5:
        return "UP"
    if hybrid_change < -0.5:
        return "DOWN"
    return "FLAT"


def _compute_base_action(direction: str, risk_level: str) -> str:
    if risk_level == "High":
        return "Wait"
    if direction == "UP" and risk_level == "Low":
        return "Buy"
    if direction == "DOWN" and risk_level == "Low":
        return "Sell"
    # Medium risk or Flat → Hold in all cases
    return "Hold"


def _compute_confidence_score(
    action: str,
    direction: str,
    sentiment: str,
    risk_level: str,
    risk_confidence: float,
) -> float:
    """
    Confidence in the suggested action (not just signal confidence).
    Starts from risk_confidence, then applies modifiers.
    """
    score = float(risk_confidence)

    # Sentiment alignment
    if action == "Buy" and sentiment == "Bullish":
        score = min(1.0, score + 0.10)
    elif action == "Sell" and sentiment == "Bearish":
        score = min(1.0, score + 0.10)
    elif action in ("Buy", "Sell") and sentiment not in ("Neutral",):
        # opposing sentiment
        sent_matches = (action == "Buy" and sentiment == "Bullish") or \
                       (action == "Sell" and sentiment == "Bearish")
        if not sent_matches:
            score = max(0.10, score - 0.12)

    # High risk pulls confidence toward 0.5
    if risk_level == "High":
        score = min(score, 0.55)
    elif risk_level == "Medium":
        score = min(score, 0.72)

    # Flat direction
    if direction == "FLAT":
        score = max(0.10, score - 0.08)

    return round(score, 2)


def _confidence_tier(score: float) -> str:
    if score >= 0.70:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Factor builders
# ---------------------------------------------------------------------------

def _build_supporting_factors(
    action: str,
    direction: str,
    sentiment: str,
    hybrid_change: float,
    current_price: float,
    risk_level: str,
    risk_warnings: list[str],
    source_count: int,
) -> list[str]:
    factors: list[str] = []

    # Forecast direction support
    if direction == "UP" and action in ("Buy", "Hold"):
        factors.append(
            f"AI LSTM model forecasts an upward move "
            f"(+${abs(hybrid_change):,.2f} from current ${current_price:,.2f})"
        )
    elif direction == "DOWN" and action in ("Sell", "Hold"):
        factors.append(
            f"AI LSTM model forecasts a downward move "
            f"(-${abs(hybrid_change):,.2f} from current ${current_price:,.2f})"
        )

    # Sentiment alignment
    if action == "Buy" and sentiment == "Bullish":
        factors.append("News sentiment is Bullish — aligns with the upward forecast")
    elif action == "Sell" and sentiment == "Bearish":
        factors.append("News sentiment is Bearish — aligns with the downward forecast")
    elif action in ("Hold", "Wait") and sentiment == "Neutral":
        factors.append("News sentiment is Neutral — no strong directional bias")

    # Low risk
    if risk_level == "Low":
        factors.append("Risk Guard reports Low risk — signal consistency is high")

    # Good source coverage
    if source_count >= 3:
        factors.append(f"News analysis backed by {source_count} grounded sources")

    # No conflicts
    if "SIGNAL_CONFLICT" not in risk_warnings:
        factors.append("No conflict detected between technical forecast and news direction")

    return factors


def _build_contradicting_factors(
    action: str,
    direction: str,
    sentiment: str,
    risk_level: str,
    risk_warnings: list[str],
    risk_factors: list[str],
    source_count: int,
    prev_price: float,
    current_price: float,
) -> list[str]:
    factors: list[str] = []

    # Signal conflict
    if "SIGNAL_CONFLICT" in risk_warnings:
        factors.append(
            "Signal conflict: forecast direction and news sentiment are not aligned"
        )

    # Opposing sentiment
    if action == "Buy" and sentiment == "Bearish":
        factors.append("News sentiment is Bearish — contradicts a Buy decision")
    elif action == "Sell" and sentiment == "Bullish":
        factors.append("News sentiment is Bullish — contradicts a Sell decision")

    # High / Medium risk
    if risk_level == "High":
        factors.append("Risk Guard reports High risk — multiple conflicting signals present")
    elif risk_level == "Medium":
        factors.append("Risk Guard reports Medium risk — some uncertainty remains")

    # Low credibility news
    if "LOW_CREDIBILITY_NEWS" in risk_warnings:
        factors.append("News report has low credibility (fallback mode or no grounded sources)")

    # Volatility
    if "HIGH_VOLATILITY" in risk_warnings and prev_price > 0 and current_price > 0:
        pct = abs(current_price - prev_price) / prev_price * 100
        factors.append(f"High market volatility detected ({pct:.2f}% daily move)")
    elif "MODERATE_VOLATILITY" in risk_warnings:
        factors.append("Moderate price volatility adds uncertainty to the signal")

    # Insufficient sources
    if source_count < 2:
        factors.append("Limited news source coverage — analysis may be incomplete")

    return factors


def _build_rationale(
    action: str,
    confidence: str,
    direction: str,
    sentiment: str,
    risk_level: str,
    hybrid_change: float,
    current_price: float,
) -> str:
    direction_str = f"up by ${abs(hybrid_change):,.2f}" if direction == "UP" else \
                    f"down by ${abs(hybrid_change):,.2f}" if direction == "DOWN" else "flat"
    risk_clause = (
        "with a clean signal" if risk_level == "Low"
        else f"though risk is {risk_level.lower()} — apply caution"
    )
    return (
        f"Forecast is {direction_str} from ${current_price:,.2f}, "
        f"news is {sentiment}, risk is {risk_level}: "
        f"suggested action is {action} ({confidence} confidence, {risk_clause})."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_suggested_action(
    sentiment: str,
    sentiment_score: float,
    hybrid_change: float,
    current_price: float,
    prev_price: float,
    risk_result: dict | None,
    news_analysis: dict | None = None,
) -> dict[str, Any]:
    """
    Build an explainable suggested action card from platform signals.

    Args:
        sentiment:       "Bullish" | "Bearish" | "Neutral"
        sentiment_score: numeric adjustment (e.g. 0.005 or -0.005)
        hybrid_change:   final_predicted_price - last_price
        current_price:   latest observed gold price
        prev_price:      previous session close
        risk_result:     output from evaluate_risk(), or None
        news_analysis:   full news dict, used for source count

    Returns:
        Full action card dict (see module docstring).
    """
    risk_result = risk_result or {}
    risk_level = risk_result.get("risk_level", "Unknown")
    risk_confidence = float(risk_result.get("confidence_score", 0.5))
    risk_warnings: list[str] = risk_result.get("warnings", [])
    risk_factors: list[str] = risk_result.get("factors", [])

    source_count = 0
    if isinstance(news_analysis, dict):
        source_count = len(news_analysis.get("sources") or [])

    direction = _forecast_direction(hybrid_change)

    # Map "Unknown" risk to "Medium" for decision purposes
    effective_risk = risk_level if risk_level in ("Low", "Medium", "High") else "Medium"

    action = _compute_base_action(direction, effective_risk)

    confidence_score = _compute_confidence_score(
        action, direction, sentiment, risk_level, risk_confidence
    )
    confidence = _confidence_tier(confidence_score)

    supporting = _build_supporting_factors(
        action, direction, sentiment, hybrid_change,
        current_price, risk_level, risk_warnings, source_count
    )
    contradicting = _build_contradicting_factors(
        action, direction, sentiment, risk_level,
        risk_warnings, risk_factors, source_count, prev_price, current_price
    )
    rationale = _build_rationale(
        action, confidence, direction, sentiment, risk_level, hybrid_change, current_price
    )

    return {
        "suggested_action": action,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "risk_level": risk_level,
        "supporting_factors": supporting,
        "contradicting_factors": contradicting,
        "rationale": rationale,
    }
