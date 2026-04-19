"""
Smart Alerts Engine — Spectre Gold AI

Detects five alert conditions from live platform signals and returns a
list of structured alert objects sorted by severity. No external calls.

Public entry point:
    from News.smart_alerts import detect_alerts

    alerts = detect_alerts(
        current_price   = last_price,
        prev_price      = prev_price,
        sentiment       = _raw_sentiment,
        prev_sentiment  = st.session_state.prev_sentiment,
        risk_result     = st.session_state.risk_result,
        news_analysis   = st.session_state.news_analysis,
    )

Each alert dict:
    {
        "id":       str,   # stable identifier, e.g. "LARGE_PRICE_MOVEMENT"
        "severity": str,   # "critical" | "warning" | "info"
        "icon":     str,   # emoji prefix
        "title":    str,   # short headline
        "message":  str,   # one-sentence explanation
    }

Severity order (highest first):  critical > warning > info
"""

from __future__ import annotations
from typing import Any

# ---------------------------------------------------------------------------
# Alert IDs (stable, used as dismiss keys)
# ---------------------------------------------------------------------------

LARGE_PRICE_MOVEMENT = "LARGE_PRICE_MOVEMENT"
SENTIMENT_REVERSAL   = "SENTIMENT_REVERSAL"
HIGH_RISK_CONDITION  = "HIGH_RISK_CONDITION"
CONFLICTING_SIGNALS  = "CONFLICTING_SIGNALS"
SUSPICIOUS_NEWS      = "SUSPICIOUS_NEWS"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

LARGE_MOVE_PCT      = 1.0   # % daily move classified as "large"
CRITICAL_MOVE_PCT   = 2.5   # % daily move classified as "critical"

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def _detect_large_price_movement(
    current_price: float, prev_price: float
) -> dict | None:
    if prev_price <= 0 or current_price <= 0:
        return None
    pct = abs(current_price - prev_price) / prev_price * 100
    direction = "up" if current_price > prev_price else "down"

    if pct >= CRITICAL_MOVE_PCT:
        return {
            "id": LARGE_PRICE_MOVEMENT,
            "severity": "critical",
            "icon": "🚨",
            "title": f"Extreme Price Move: {direction} {pct:.2f}%",
            "message": (
                f"Gold moved {direction} {pct:.2f}% "
                f"(${prev_price:,.2f} → ${current_price:,.2f}). "
                "Extreme volatility — consider reducing exposure."
            ),
        }
    if pct >= LARGE_MOVE_PCT:
        return {
            "id": LARGE_PRICE_MOVEMENT,
            "severity": "warning",
            "icon": "⚡",
            "title": f"Notable Price Movement: {direction} {pct:.2f}%",
            "message": (
                f"Gold moved {direction} {pct:.2f}% "
                f"(${prev_price:,.2f} → ${current_price:,.2f}). "
                "Above-average daily range detected."
            ),
        }
    return None


def _detect_sentiment_reversal(
    sentiment: str, prev_sentiment: str
) -> dict | None:
    if not prev_sentiment or prev_sentiment == "Unknown":
        return None
    if sentiment == prev_sentiment:
        return None
    # Both must be meaningful (not both Neutral)
    if sentiment == "Neutral" and prev_sentiment == "Neutral":
        return None
    return {
        "id": SENTIMENT_REVERSAL,
        "severity": "warning",
        "icon": "🔄",
        "title": f"Sentiment Reversal: {prev_sentiment} → {sentiment}",
        "message": (
            f"News sentiment shifted from {prev_sentiment} to {sentiment} "
            "since the last report. Review the latest market intelligence."
        ),
    }


def _detect_high_risk_condition(risk_result: dict) -> dict | None:
    risk_level = risk_result.get("risk_level", "Unknown")
    if risk_level != "High":
        return None
    confidence = int(risk_result.get("confidence_score", 0) * 100)
    warnings = risk_result.get("warnings", [])
    first_factor = risk_result.get("factors", ["Multiple conflicting signals."])[0]
    return {
        "id": HIGH_RISK_CONDITION,
        "severity": "critical",
        "icon": "🛑",
        "title": f"High Risk Condition Detected ({confidence}% confidence)",
        "message": (
            f"Risk Guard flagged a HIGH risk state: {first_factor} "
            f"[{', '.join(warnings) or 'see risk panel'}]. "
            "Avoid new positions until risk subsides."
        ),
    }


def _detect_conflicting_signals(risk_result: dict) -> dict | None:
    warnings: list[str] = risk_result.get("warnings", [])
    if "SIGNAL_CONFLICT" not in warnings:
        return None
    factors = risk_result.get("factors", [])
    detail = next((f for f in factors if "conflict" in f.lower()), "Forecast and news sentiment are not aligned.")
    return {
        "id": CONFLICTING_SIGNALS,
        "severity": "warning",
        "icon": "⚠️",
        "title": "Conflicting Signals Detected",
        "message": (
            f"{detail} "
            "Do not act on the forecast alone — wait for signal alignment."
        ),
    }


def _detect_suspicious_news(
    risk_result: dict, news_analysis: dict | None
) -> dict | None:
    warnings: list[str] = risk_result.get("warnings", [])
    sources = (news_analysis or {}).get("sources") or []
    source_count = len(sources)
    source_status = (news_analysis or {}).get("source_status", "unavailable")

    has_credibility_flag = "LOW_CREDIBILITY_NEWS" in warnings
    is_fallback = source_status in ("unavailable", "fallback") or source_count < 2

    if not has_credibility_flag and not is_fallback:
        return None

    if has_credibility_flag:
        return {
            "id": SUSPICIOUS_NEWS,
            "severity": "warning",
            "icon": "🔍",
            "title": "Low News Credibility",
            "message": (
                f"Only {source_count} grounded source(s) found. "
                "The market report may rely on unverified information — "
                "treat sentiment signals with caution."
            ),
        }
    # Fallback mode (source_status == unavailable / < 2 sources)
    return {
        "id": SUSPICIOUS_NEWS,
        "severity": "info",
        "icon": "ℹ️",
        "title": "Limited News Coverage",
        "message": (
            f"Market report used {source_count} grounded source(s). "
            "Sentiment accuracy may be lower than normal."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_alerts(
    current_price: float,
    prev_price: float,
    sentiment: str,
    prev_sentiment: str,
    risk_result: dict | None,
    news_analysis: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Run all detectors and return active alerts sorted critical-first.

    Args:
        current_price:  Latest observed gold price.
        prev_price:     Previous session close price.
        sentiment:      Current news sentiment ("Bullish" | "Bearish" | "Neutral").
        prev_sentiment: Sentiment from the previous news fetch (stored in session).
        risk_result:    Output of evaluate_risk(), or None.
        news_analysis:  Full news dict from get_ai_market_report(), or None.

    Returns:
        List of alert dicts, sorted by severity (critical first).
    """
    risk_result = risk_result or {}
    news_analysis = news_analysis or {}

    raw_alerts: list[dict] = []

    # 1. Large price movement
    alert = _detect_large_price_movement(current_price, prev_price)
    if alert:
        raw_alerts.append(alert)

    # 2. Sentiment reversal (only when news has been fetched at least once)
    alert = _detect_sentiment_reversal(sentiment, prev_sentiment)
    if alert:
        raw_alerts.append(alert)

    # 3. High risk condition
    alert = _detect_high_risk_condition(risk_result)
    if alert:
        raw_alerts.append(alert)

    # 4. Conflicting signals
    alert = _detect_conflicting_signals(risk_result)
    if alert:
        raw_alerts.append(alert)

    # 5. Suspicious / low-credibility news (only when news is loaded)
    if news_analysis:
        alert = _detect_suspicious_news(risk_result, news_analysis)
        if alert:
            raw_alerts.append(alert)

    # Sort: critical → warning → info, then deduplicate by id (first wins)
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for a in sorted(raw_alerts, key=lambda x: _SEVERITY_RANK.get(x["severity"], 9)):
        if a["id"] not in seen_ids:
            seen_ids.add(a["id"])
            deduped.append(a)

    return deduped
