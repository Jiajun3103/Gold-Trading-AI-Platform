"""
AI Risk Guard — deterministic risk analysis engine for Spectre Gold AI.

Evaluates market signal quality across four dimensions:
  1. Signal conflict   — prediction direction vs news sentiment disagree
  2. News credibility  — fallback mode, missing grounding, too few sources
  3. Sentiment quality — label and adjustment factor are inconsistent
  4. Price volatility  — unusual intraday / session price movement

No external API calls are made here. This module is always available
and returns a result in under 1 ms.

Example return value:
{
    "risk_level": "High",
    "confidence_score": 0.49,
    "factors": [
        "News is Bearish but technical model predicts a price increase",
        "News report was generated without live search grounding — credibility may be low",
        "Large daily price movement of 1.82% detected — elevated market volatility"
    ],
    "warnings": ["SIGNAL_CONFLICT", "LOW_CREDIBILITY_NEWS", "HIGH_VOLATILITY"],
    "summary": "⚠️ High Risk: Multiple conflicting or unreliable signals detected. ..."
}
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Thresholds (tweak without touching logic)
# ---------------------------------------------------------------------------
_HIGH_VOLATILITY_PCT = 1.5   # daily move > 1.5 % → high volatility flag
_MED_VOLATILITY_PCT = 0.8    # daily move > 0.8 % → moderate volatility flag
_MIN_CREDIBLE_SOURCES = 2    # fewer sources than this → low-credibility flag


# ---------------------------------------------------------------------------
# Individual check functions — each returns (triggered: bool, reason: str)
# ---------------------------------------------------------------------------

def _check_signal_conflict(sentiment: str, hybrid_change: float) -> tuple[bool, str]:
    """Prediction direction vs news sentiment disagree."""
    pred_up = hybrid_change > 0
    if sentiment == "Bullish" and not pred_up:
        return True, "News is Bullish but technical model predicts a price decline"
    if sentiment == "Bearish" and pred_up:
        return True, "News is Bearish but technical model predicts a price increase"
    return False, ""


def _check_sentiment_quality(sentiment: str, sentiment_score: float) -> tuple[bool, str]:
    """
    Sentiment label and numeric adjustment are misaligned.
    (e.g. label says Bullish but adjustment factor rounds to zero)
    """
    if sentiment in ("Bullish", "Bearish") and abs(sentiment_score) < 0.001:
        return (
            True,
            f"Sentiment label is '{sentiment}' but the numeric adjustment factor is near zero "
            "— the signal may be unreliable",
        )
    return False, ""


def _check_news_credibility(news_analysis: dict | None) -> tuple[bool, str]:
    """Low source count, fallback mode, or missing grounding = suspicious."""
    if news_analysis is None:
        return (
            True,
            "No news analysis available — risk assessment is based on technical data only",
        )

    source_status = news_analysis.get("source_status", "unavailable")
    sources = news_analysis.get("sources", [])
    fallback_used = news_analysis.get("fallback_used", False)

    if source_status in ("unavailable", "no_grounding"):
        return (
            True,
            "News report was generated without live search grounding — credibility may be low",
        )
    if fallback_used:
        return (
            True,
            "AI news agent used fallback mode — real-time search was unavailable this session",
        )
    if len(sources) < _MIN_CREDIBLE_SOURCES:
        return (
            True,
            f"Only {len(sources)} news source(s) returned — insufficient for high-confidence analysis",
        )
    return False, ""


def _check_volatility(current_price: float, prev_price: float) -> tuple[str | None, str]:
    """
    Returns (level, message) where level is 'high', 'medium', or None.
    """
    if current_price <= 0 or prev_price <= 0:
        return None, ""
    pct = abs(current_price - prev_price) / prev_price * 100
    if pct >= _HIGH_VOLATILITY_PCT:
        return "high", f"Large daily price movement of {pct:.2f}% detected — elevated market volatility"
    if pct >= _MED_VOLATILITY_PCT:
        return "medium", f"Moderate daily price movement of {pct:.2f}% detected"
    return None, ""


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _compute_risk_level(warning_codes: list[str], vol_level: str | None) -> str:
    """
    High   → 3+ risk factors, OR 2+ non-volatility factors AND high volatility
    Medium → 1-2 factors, OR moderate/high volatility alone
    Low    → nothing flagged
    """
    serious = [w for w in warning_codes if w not in ("MODERATE_VOLATILITY",)]
    if len(serious) >= 3:
        return "High"
    if len(serious) >= 2 and vol_level == "high":
        return "High"
    if len(serious) >= 2:
        return "Medium"
    if len(serious) >= 1 or vol_level in ("high", "medium"):
        return "Medium"
    return "Low"


def _compute_confidence_score(factor_count: int) -> float:
    """
    Confidence in the trading *signal* (not in the risk level itself).
    Starts at 0.88, -0.13 per risk factor, floor 0.12.
    """
    return round(max(0.12, 0.88 - factor_count * 0.13), 2)


def _build_summary(risk_level: str, warnings: list[str]) -> str:
    if risk_level == "High":
        parts = ["⚠️ High Risk: Multiple conflicting or unreliable signals detected."]
        if "SIGNAL_CONFLICT" in warnings:
            parts.append("Technical prediction contradicts news sentiment.")
        if "LOW_CREDIBILITY_NEWS" in warnings:
            parts.append("News data quality is insufficient for confident analysis.")
        if "HIGH_VOLATILITY" in warnings:
            parts.append("Unusually large price movement observed.")
        parts.append("Consider waiting for clearer signals before trading.")
        return " ".join(parts)

    if risk_level == "Medium":
        parts = ["⚡ Medium Risk: Some uncertainty is present."]
        if "SIGNAL_CONFLICT" in warnings:
            parts.append("Prediction direction and sentiment are not fully aligned.")
        if "MODERATE_VOLATILITY" in warnings or "HIGH_VOLATILITY" in warnings:
            parts.append("Price is moving with above-average momentum.")
        if "LOW_CREDIBILITY_NEWS" in warnings:
            parts.append("News analysis confidence is limited.")
        parts.append("Proceed with caution and ensure stop-loss levels are set.")
        return " ".join(parts)

    return (
        "✅ Low Risk: Signals are broadly consistent. "
        "Standard position sizing and risk management still apply."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_risk(
    sentiment: str,
    sentiment_score: float,
    hybrid_change: float,
    current_price: float,
    prev_price: float,
    news_analysis: dict | None = None,
) -> dict[str, Any]:
    """
    Evaluate market signal risk from available platform data.

    Args:
        sentiment:       "Bullish" | "Bearish" | "Neutral"
        sentiment_score: numeric adjustment applied to prediction (e.g. 0.005)
        hybrid_change:   final_predicted_price - last_price
        current_price:   latest observed gold price (USD)
        prev_price:      previous session close price (USD)
        news_analysis:   full dict from get_ai_market_report(), or None

    Returns:
        {
            "risk_level":       "Low" | "Medium" | "High",
            "confidence_score": float  (0.12 – 0.88),
            "factors":          list[str],   # human-readable reasons
            "warnings":         list[str],   # machine-readable codes
            "summary":          str
        }
    """
    factors: list[str] = []
    warnings: list[str] = []

    # 1. Signal conflict
    hit, msg = _check_signal_conflict(sentiment, hybrid_change)
    if hit:
        factors.append(msg)
        warnings.append("SIGNAL_CONFLICT")

    # 2. Sentiment quality
    hit, msg = _check_sentiment_quality(sentiment, sentiment_score)
    if hit:
        factors.append(msg)
        warnings.append("UNRELIABLE_SENTIMENT")

    # 3. News credibility
    hit, msg = _check_news_credibility(news_analysis)
    if hit:
        factors.append(msg)
        warnings.append("LOW_CREDIBILITY_NEWS")

    # 4. Volatility
    vol_level, vol_msg = _check_volatility(current_price, prev_price)
    if vol_msg:
        factors.append(vol_msg)
        warnings.append("HIGH_VOLATILITY" if vol_level == "high" else "MODERATE_VOLATILITY")

    risk_level = _compute_risk_level(warnings, vol_level)
    confidence_score = _compute_confidence_score(len(factors))
    summary = _build_summary(risk_level, warnings)

    return {
        "risk_level": risk_level,
        "confidence_score": confidence_score,
        "factors": factors,
        "warnings": warnings,
        "summary": summary,
    }
