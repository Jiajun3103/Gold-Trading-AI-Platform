import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
import json
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from html import escape
import unicodedata
import re

_PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio.json")

_PORTFOLIO_DEFAULTS = {
    "sim_wallet_balance": 10000.0,
    "sim_position_oz": 0.0,
    "auto_trade_history": [],
}

def _load_portfolio() -> dict:
    try:
        with open(_PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "sim_wallet_balance": float(data.get("sim_wallet_balance", 10000.0)),
            "sim_position_oz": float(data.get("sim_position_oz", 0.0)),
            "auto_trade_history": list(data.get("auto_trade_history", [])),
        }
    except (FileNotFoundError, ValueError, KeyError):
        return dict(_PORTFOLIO_DEFAULTS)

def _save_portfolio():
    data = {
        "sim_wallet_balance": st.session_state.sim_wallet_balance,
        "sim_position_oz": st.session_state.sim_position_oz,
        "auto_trade_history": st.session_state.auto_trade_history,
    }
    try:
        with open(_PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
from News.news import get_ai_market_report
from News.chatbot import generate_platform_chat_response, agentic_dispatch
from News.risk_guard import evaluate_risk
from News.decision_engine import build_suggested_action
from News.smart_alerts import detect_alerts
import requests as _requests

# ---------------------------------------------------------------------------
# Genkit market intelligence — calls FastAPI, falls back to direct Python
# ---------------------------------------------------------------------------
_BACKEND_URL = os.environ.get("API_BASE_URL", "http://localhost:8010").rstrip("/")

def _call_genkit_market_intelligence(question: str = "What is the current gold market outlook?") -> dict:
    """Call FastAPI /genkit/market-intelligence; fall back to direct Python on failure."""
    fastapi_url = f"{_BACKEND_URL}/genkit/market-intelligence"
    print(f"[Streamlit] _call_genkit_market_intelligence → calling FastAPI at {fastapi_url}")
    try:
        resp = _requests.post(
            fastapi_url,
            json={"question": question, "includeAlerts": True},
            timeout=35,
        )
        print(f"[Streamlit] FastAPI response status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"[Streamlit] FastAPI response source: {data.get('source')} status: {data.get('status')}")
            if data.get("status") == "success":
                # Normalise to existing session_state shape so renders stay compatible
                if not data.get("full_report"):
                    drivers = "\n".join(f"• {d}" for d in data.get("key_drivers", []))
                    risks   = "\n".join(f"• {r}" for r in data.get("risk_factors", []))
                    outlook = data.get("price_outlook", {})
                    data["full_report"] = (
                        f"**Key Drivers**\n{drivers}\n\n"
                        f"**Risk Factors**\n{risks}\n\n"
                        f"**Short-term Outlook:** {outlook.get('short_term', '')}\n"
                        f"**Medium-term Outlook:** {outlook.get('medium_term', '')}"
                    )
                # Map grounded_sources (merged hybrid field) → sources for display
                if not data.get("sources"):
                    data["sources"] = data.get("grounded_sources", [])
                # source_status comes directly from the merged response
                data.setdefault("source_status", data.get("source_status") or (
                    "genkit_grounded" if data.get("source") == "genkit" else "python_fallback"
                ))
                print(f"[Streamlit] using source: {data.get('source')} | "
                      f"grounded sources: {len(data.get('sources', []))} | "
                      f"source_status: {data.get('source_status')} (via FastAPI)")
                return data
        else:
            print(f"[Streamlit] FastAPI non-200: {resp.status_code} — {resp.text[:200]}")
    except Exception as exc:
        print(f"[Streamlit] FastAPI call failed: {exc}")
    # Fallback: call Python module directly
    print("[Streamlit] falling back to direct Python get_ai_market_report()")
    result = get_ai_market_report()
    result.setdefault("source", "python_fallback")
    result.setdefault("context_source", "Python Fallback")
    return result

# --- PAGE CONFIG ---
st.set_page_config(page_title="Spectre Gold AI", page_icon="💰", layout="wide")

if 'news_analysis' not in st.session_state:
    st.session_state.news_analysis = None
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None
if 'last_workflow_steps' not in st.session_state:
    st.session_state.last_workflow_steps = []
if 'last_workflow_was_agentic' not in st.session_state:
    st.session_state.last_workflow_was_agentic = False
if 'action_card' not in st.session_state:
    st.session_state.action_card = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'dismissed_alert_ids' not in st.session_state:
    st.session_state.dismissed_alert_ids = set()
if 'prev_sentiment' not in st.session_state:
    st.session_state.prev_sentiment = 'Unknown'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chat_widget_open' not in st.session_state:
    st.session_state.chat_widget_open = False
if 'chat_pending_message' not in st.session_state:
    st.session_state.chat_pending_message = None
if 'chat_is_typing' not in st.session_state:
    st.session_state.chat_is_typing = False
# Portfolio session state — initialised here so sidebar can read live values
if 'sim_wallet_balance' not in st.session_state:
    _saved = _load_portfolio()
    st.session_state.sim_wallet_balance  = _saved["sim_wallet_balance"]
    st.session_state.sim_position_oz     = _saved["sim_position_oz"]
    st.session_state.auto_trade_history  = _saved["auto_trade_history"]


def _build_platform_context(last_price, tech_predicted_price, final_predicted_price, hybrid_change, sentiment_label, sentiment_score, prev_price=0.0):
    return {
        "timestamp": datetime.now().isoformat(),
        "last_price": round(last_price, 2),
        "prev_price": round(prev_price, 2),
        "technical_predicted_price": round(tech_predicted_price, 2),
        "final_predicted_price": round(final_predicted_price, 2),
        "hybrid_change": round(hybrid_change, 2),
        "sentiment_label": sentiment_label,
        "sentiment_adjustment_pct": round(sentiment_score * 100, 2),
        "wallet_balance": 10000.0,
        "risk_result": st.session_state.risk_result,
        "risk_monitor": {
            "news_sentiment": st.session_state.news_analysis.get('sentiment') if st.session_state.news_analysis else "Neutral",
            "inflation_sentiment": "BULLISH FOR GOLD",
        },
        "news_analysis": st.session_state.news_analysis if st.session_state.news_analysis else {
            "sentiment": "Neutral",
            "reason": "No generated report in current session",
            "full_report": "",
            "sources": [],
            "source_status": "unavailable",
        },
    }


def _sanitize_chat_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    return text.strip()


def _chat_now():
    return datetime.now().strftime("%H:%M")


_SOURCE_LABELS = {
    "genkit_goldChatFlow": "⚡ Handled by: Genkit goldChatFlow",
    "adk_agent":          "🤖 Handled by: ADK Agent",
    "workflow_runner":    "🔧 Handled by: Local WorkflowRunner",
    "gemini_direct":      "✨ Handled by: Direct Gemini Fallback",
    "python_fallback":    "🐍 Handled by: Python Fallback",
    "fallback":           "💬 Handled by: Static Fallback",
}


def _append_chat_message(role, content, read=False, source=""):
    st.session_state.chat_history.append(
        {
            "role": role,
            "content": _sanitize_chat_text(content),
            "time": _chat_now(),
            "read": bool(read),
            "source": source,
        }
    )


def _mark_last_user_read():
    for idx in range(len(st.session_state.chat_history) - 1, -1, -1):
        msg = st.session_state.chat_history[idx]
        if msg.get("role") == "user":
            msg["read"] = True
            return


def _render_history_html(messages):
    blocks = [
        "<div class='sg-row sg-assistant-row'>"
        "<div class='sg-avatar sg-assistant-avatar'>SG</div>"
        "<div class='sg-bubble-wrap'>"
        "<div class='sg-message sg-assistant'>"
        "Hello, I am Spectre Gold AI. Ask me to explain your current signal, forecast, or risks."
        "</div>"
        "<div class='sg-meta'>" + _chat_now() + "</div>"
        "</div></div>"
    ]

    for msg in messages[-20:]:
        role = msg.get("role", "assistant")
        content = escape(_sanitize_chat_text(msg.get("content", "")))
        time_text = escape(msg.get("time", ""))
        role_class = "sg-user" if role == "user" else "sg-assistant"
        row_class = "sg-user-row" if role == "user" else "sg-assistant-row"
        avatar_class = "sg-user-avatar" if role == "user" else "sg-assistant-avatar"
        avatar_text = "U" if role == "user" else "SG"
        read_mark = "<span class='sg-read'>✓✓</span>" if role == "user" and msg.get("read") else "<span class='sg-read sg-unread'>✓</span>"

        if role == "user":
            blocks.append(
                f"<div class='sg-row {row_class}'>"
                f"<div class='sg-bubble-wrap'>"
                f"<div class='sg-message {role_class}'>{content}</div>"
                f"<div class='sg-meta'>{time_text}{read_mark}</div>"
                f"</div>"
                f"<div class='sg-avatar {avatar_class}'>{avatar_text}</div>"
                f"</div>"
            )
        else:
            src_key = msg.get("source", "")
            src_label = escape(_SOURCE_LABELS.get(src_key, ""))
            src_html = f"<div class='sg-source'>{src_label}</div>" if src_label else ""
            blocks.append(
                f"<div class='sg-row {row_class}'>"
                f"<div class='sg-avatar {avatar_class}'>{avatar_text}</div>"
                f"<div class='sg-bubble-wrap'>"
                f"<div class='sg-message {role_class}'>{content}</div>"
                f"<div class='sg-meta'>{time_text}</div>"
                f"{src_html}"
                f"</div>"
                f"</div>"
            )

    if st.session_state.chat_is_typing:
        blocks.append(
            "<div class='sg-row sg-assistant-row'>"
            "<div class='sg-avatar sg-assistant-avatar'>SG</div>"
            "<div class='sg-bubble-wrap'>"
            "<div class='sg-message sg-assistant sg-typing'>"
            "<span></span><span></span><span></span>"
            "</div>"
            "<div class='sg-meta'>typing...</div>"
            "</div></div>"
        )

    return "".join(blocks)

# --- LOAD AI MODELS ---
@st.cache_resource 
def load_assets():
    model = tf.keras.models.load_model('Models/gold_price_model.keras')
    scaler = joblib.load('Models/gold_scaler.bin')
    return model, scaler

model, scaler = load_assets()

# --- LOAD DATA ---
@st.cache_data
def load_live_data():
    gold_data = yf.Ticker("GC=F")
    df = gold_data.history(period="100d")
    df = df.reset_index()
    df = df[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
    df['Date'] = df['Date'].dt.tz_localize(None)
    return df

df = load_live_data()

# --- SIDEBAR ---
st.sidebar.title("🏆 Spectre Gold AI")

# ── Navigation ────────────────────────────────────────────────
st.sidebar.markdown("### Navigate")
_active_page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "💹 Trading Simulator", "🌐 AI Intelligence"],
    label_visibility="collapsed",
    key="nav_page",
)

st.sidebar.divider()

# ── Portfolio summary ─────────────────────────────────────────
st.sidebar.subheader("👤 Portfolio")
_sb_pos_val = st.session_state.sim_position_oz * 0.0  # will be filled after price calc; show on next render
st.sidebar.metric("Cash Balance", f"${st.session_state.sim_wallet_balance:,.2f}")
st.sidebar.metric("Gold Held", f"{st.session_state.sim_position_oz:,.4f} oz")

if st.sidebar.button("Clear Cache & Reset"):
    st.session_state.news_analysis = None
    st.session_state.chat_history = []
    st.session_state.sim_wallet_balance = 10000.0
    st.session_state.sim_position_oz = 0.0
    st.session_state.auto_trade_history = []
    _save_portfolio()
    st.cache_data.clear()
    st.rerun()

last_60_days = df['Close'].values[-60:].reshape(-1, 1)
scaled_input = scaler.transform(last_60_days)
current_batch = scaled_input.reshape((1, 60, 1))
prediction_scaled = model.predict(current_batch)
tech_predicted_price = float(scaler.inverse_transform(prediction_scaled)[0][0])

sentiment_score = 0.0
sentiment_label = "Neutral (Technical Only)"

if st.session_state.news_analysis:
    sent = st.session_state.news_analysis.get('sentiment', 'Neutral')
    if sent == "Bullish":
        sentiment_score = 0.005 
        sentiment_label = "BULLISH (Hybrid)"
    elif sent == "Bearish":
        sentiment_score = -0.005 
        sentiment_label = "BEARISH (Hybrid)"
    else:
        sentiment_label = "NEUTRAL (Hybrid)"

final_predicted_price = tech_predicted_price * (1 + sentiment_score)
last_price = float(df['Close'].iloc[-1])
hybrid_change = final_predicted_price - last_price
prev_price = float(df['Close'].iloc[-2]) if len(df) > 1 else last_price

# 4. AI Risk Guard — evaluated every render cycle, always up-to-date
_raw_sentiment = "Neutral"
if st.session_state.news_analysis:
    _raw_sentiment = st.session_state.news_analysis.get('sentiment', 'Neutral')

st.session_state.risk_result = evaluate_risk(
    sentiment=_raw_sentiment,
    sentiment_score=sentiment_score,
    hybrid_change=hybrid_change,
    current_price=last_price,
    prev_price=prev_price,
    news_analysis=st.session_state.news_analysis,
)

# 5. Suggested Action Card — computed every render cycle
st.session_state.action_card = build_suggested_action(
    sentiment=_raw_sentiment,
    sentiment_score=sentiment_score,
    hybrid_change=hybrid_change,
    current_price=last_price,
    prev_price=prev_price,
    risk_result=st.session_state.risk_result,
    news_analysis=st.session_state.news_analysis,
)

# 6. Smart Alerts — computed every render cycle
st.session_state.alerts = detect_alerts(
    current_price=last_price,
    prev_price=prev_price,
    sentiment=_raw_sentiment,
    prev_sentiment=st.session_state.prev_sentiment,
    risk_result=st.session_state.risk_result,
    news_analysis=st.session_state.news_analysis,
)

# ============================================================
# SIDEBAR — Live Risk Monitor (reads previous render's risk_result)
# ============================================================
st.sidebar.divider()
st.sidebar.subheader("🛡️ Live Risk Monitor")
_sb_rr = st.session_state.risk_result or {}
_sb_rl = _sb_rr.get("risk_level", "Unknown")
_sb_cs = _sb_rr.get("confidence_score", 0.0)
if _sb_rl == "High":
    st.sidebar.error(f"Risk Level: **{_sb_rl}** | Confidence: {int(_sb_cs * 100)}%")
elif _sb_rl == "Medium":
    st.sidebar.warning(f"Risk Level: **{_sb_rl}** | Confidence: {int(_sb_cs * 100)}%")
else:
    st.sidebar.success(f"Risk Level: **{_sb_rl}** | Confidence: {int(_sb_cs * 100)}%")
if st.session_state.news_analysis:
    st.sidebar.caption(f"News Sentiment: {st.session_state.news_analysis.get('sentiment', 'N/A')}")
_sb_warns = _sb_rr.get("warnings", [])
if _sb_warns:
    st.sidebar.caption("Warnings: " + " · ".join(_sb_warns))

# ============================================================
# PAGE HEADER — Title + three clearly labelled timestamps
# ============================================================
_market_date_str = pd.Timestamp(df['Date'].iloc[-1]).strftime('%Y-%m-%d') if not df.empty else "N/A"
_app_refresh_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

st.title("🏆 Spectre Gold AI Trading Platform")
_hc1, _hc2, _hc3 = st.columns(3)
_hc1.caption(f"🔄 App Refreshed: **{_app_refresh_str}**")
_hc2.caption(f"📅 Latest Market Data: **{_market_date_str}**")
_hc3.caption(f"🤖 Forecast Generated: **{_app_refresh_str}**")
st.divider()


# ============================================================
# PAGE: 📊 DASHBOARD
# ============================================================
if _active_page == "📊 Dashboard":

    # ── Smart Alerts banner ───────────────────────────────────
    _active_alerts = [
        a for a in st.session_state.alerts
        if a["id"] not in st.session_state.dismissed_alert_ids
    ]
    if _active_alerts:
        _ALERT_STYLES = {
            "critical": {"bg": "#fef2f2", "border": "#fca5a5", "text": "#991b1b"},
            "warning":  {"bg": "#fffbeb", "border": "#fde68a", "text": "#92400e"},
            "info":     {"bg": "#eff6ff", "border": "#bfdbfe", "text": "#1e40af"},
        }
        _ahc, _adc = st.columns([5, 1])
        with _ahc:
            st.markdown(f"**🔔 {len(_active_alerts)} Active Alert{'s' if len(_active_alerts) != 1 else ''}**")
        with _adc:
            if st.button("Clear All", key="clear_all_alerts"):
                st.session_state.dismissed_alert_ids = {a["id"] for a in st.session_state.alerts}
                st.rerun()
        for _alert in _active_alerts:
            _ast = _ALERT_STYLES.get(_alert["severity"], _ALERT_STYLES["info"])
            _ac_col, _ad_col = st.columns([11, 1])
            with _ac_col:
                st.markdown(
                    f"""<div style='background:{_ast["bg"]};border-left:5px solid {_ast["border"]};
                                border-radius:8px;padding:0.65rem 1rem;margin-bottom:6px;'>
                        <span style='font-weight:700;color:{_ast["text"]};font-size:0.95rem;'>
                            {_alert["icon"]} {_alert["title"]}
                        </span><br/>
                        <span style='font-size:0.85rem;color:{_ast["text"]};'>{_alert["message"]}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with _ad_col:
                if st.button("✕", key=f"dismiss_{_alert['id']}", help="Dismiss"):
                    st.session_state.dismissed_alert_ids.add(_alert["id"])
                    st.rerun()
        st.divider()

    # ── AI Hybrid Forecast ────────────────────────────────────
    st.subheader("🤖 AI Hybrid Forecast")
    _fc1, _fc2 = st.columns(2)
    with _fc1:
        _forecast_date = (pd.Timestamp(df['Date'].iloc[-1]) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        st.metric(
            label=f"Predicted Close — {_forecast_date}",
            value=f"${final_predicted_price:,.2f}",
            delta=f"{hybrid_change:+,.2f} USD",
            delta_color="normal" if hybrid_change > 0 else "inverse",
        )
        if hybrid_change > 0:
            st.success(f"Combined Signal: **{sentiment_label}**")
        else:
            st.error(f"Combined Signal: **{sentiment_label}**")
        if st.session_state.news_analysis:
            st.caption(f"ℹ️ Price adjusted {sentiment_score*100:+.1f}% based on latest news sentiment.")
        else:
            st.caption("ℹ️ No news loaded — showing pure LSTM technical forecast.")
    with _fc2:
        st.metric("LSTM Technical Prediction", f"${tech_predicted_price:,.2f}")
        st.metric(
            "Last Observed Close",
            f"${last_price:,.2f}",
            delta=f"{last_price - prev_price:+,.2f} vs prior session",
            delta_color="normal" if last_price >= prev_price else "inverse",
        )
    st.divider()

    # ── AI Risk Guard ─────────────────────────────────────────
    st.subheader("🛡️ AI Risk Guard")
    _rr = st.session_state.risk_result or {}
    _rl = _rr.get("risk_level", "Unknown")
    _cs = _rr.get("confidence_score", 0.0)
    _factors = _rr.get("factors", [])
    _warnings = _rr.get("warnings", [])
    _summary = _rr.get("summary", "Risk data unavailable.")
    _RISK_COLORS = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626", "Unknown": "#64748b"}
    _RISK_BG     = {"Low": "#f0fdf4", "Medium": "#fffbeb", "High": "#fef2f2", "Unknown": "#f8fafc"}
    _RISK_BORDER = {"Low": "#bbf7d0", "Medium": "#fde68a", "High": "#fecaca", "Unknown": "#e2e8f0"}
    _rc   = _RISK_COLORS.get(_rl, "#64748b")
    _rb   = _RISK_BG.get(_rl, "#f8fafc")
    _rbrd = _RISK_BORDER.get(_rl, "#e2e8f0")
    _WARN_LABELS = {
        "SIGNAL_CONFLICT":      "⚡ Signal Conflict",
        "UNRELIABLE_SENTIMENT": "⚠️ Unreliable Sentiment",
        "LOW_CREDIBILITY_NEWS": "📰 Low Credibility News",
        "HIGH_VOLATILITY":      "🔥 High Volatility",
        "MODERATE_VOLATILITY":  "📊 Moderate Volatility",
    }
    _badge_html = " ".join(
        f"<span style='background:{_rc};color:#fff;padding:3px 10px;border-radius:20px;"
        f"font-size:0.78rem;font-weight:600;margin-right:4px;'>{_WARN_LABELS.get(w, w)}</span>"
        for w in _warnings
    ) or "<span style='color:#64748b;font-size:0.82rem;'>No warnings</span>"
    st.markdown(
        f"""
        <div style='background:{_rb};border:1.5px solid {_rbrd};border-radius:14px;
                    padding:1rem 1.25rem;margin-bottom:0.5rem;'>
            <div style='display:flex;align-items:center;gap:1rem;flex-wrap:wrap;'>
                <div>
                    <div style='font-size:0.78rem;color:#64748b;font-weight:600;
                                text-transform:uppercase;letter-spacing:.05em;'>Risk Level</div>
                    <div style='font-size:2rem;font-weight:800;color:{_rc};line-height:1.1;'>{_rl}</div>
                </div>
                <div style='border-left:2px solid {_rbrd};padding-left:1rem;'>
                    <div style='font-size:0.78rem;color:#64748b;font-weight:600;
                                text-transform:uppercase;letter-spacing:.05em;'>Signal Confidence</div>
                    <div style='font-size:2rem;font-weight:800;color:#0f172a;line-height:1.1;'>{int(_cs * 100)}%</div>
                </div>
                <div style='flex:1;min-width:200px;'>
                    <div style='font-size:0.78rem;color:#64748b;font-weight:600;
                                text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;'>
                        Active Warnings
                    </div>
                    {_badge_html}
                </div>
            </div>
            <div style='margin-top:0.75rem;font-size:0.92rem;color:#1e293b;line-height:1.5;'>{_summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if _factors:
        with st.expander("View Risk Factor Details", expanded=False):
            for i, f in enumerate(_factors, 1):
                st.markdown(f"**{i}.** {f}")
    else:
        st.caption("No risk factors detected in current signals.")
    st.divider()

    # ── Suggested Action Card ─────────────────────────────────
    st.subheader("🎯 AI Suggested Action")
    _ac = st.session_state.action_card or {}
    _sa      = _ac.get("suggested_action", "Wait")
    _sa_conf = _ac.get("confidence", "Low")
    _sa_cs   = _ac.get("confidence_score", 0.0)
    _sa_rl   = _ac.get("risk_level", "Unknown")
    _sa_sup  = _ac.get("supporting_factors", [])
    _sa_con  = _ac.get("contradicting_factors", [])
    _sa_rat  = _ac.get("rationale", "")
    _ACTION_CFG = {
        "Buy":  {"bg": "#dcfce7", "border": "#86efac", "badge_bg": "#16a34a", "icon": "🟢"},
        "Sell": {"bg": "#fef2f2", "border": "#fca5a5", "badge_bg": "#dc2626", "icon": "🔴"},
        "Hold": {"bg": "#fffbeb", "border": "#fde68a", "badge_bg": "#d97706", "icon": "🟡"},
        "Wait": {"bg": "#f8fafc", "border": "#cbd5e1", "badge_bg": "#64748b", "icon": "⚪"},
    }
    _cfg = _ACTION_CFG.get(_sa, _ACTION_CFG["Wait"])
    _CONF_COLORS = {"High": "#16a34a", "Medium": "#d97706", "Low": "#dc2626"}
    _conf_color = _CONF_COLORS.get(_sa_conf, "#64748b")
    st.markdown(
        f"""
        <div style='background:{_cfg["bg"]};border:2px solid {_cfg["border"]};border-radius:16px;
                    padding:1.1rem 1.4rem;'>
            <div style='display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;'>
                <div style='text-align:center;min-width:90px;'>
                    <div style='font-size:2.6rem;line-height:1;'>{_cfg["icon"]}</div>
                    <div style='background:{_cfg["badge_bg"]};color:#fff;font-size:1.1rem;
                                font-weight:800;border-radius:10px;padding:4px 16px;
                                margin-top:6px;display:inline-block;'>{_sa}</div>
                </div>
                <div style='border-left:2px solid {_cfg["border"]};padding-left:1.1rem;'>
                    <div style='font-size:0.78rem;color:#64748b;font-weight:600;
                                text-transform:uppercase;letter-spacing:.05em;'>Confidence</div>
                    <div style='font-size:1.6rem;font-weight:800;color:{_conf_color};'>
                        {_sa_conf}&nbsp;<span style='font-size:1rem;color:#94a3b8;'>({int(_sa_cs*100)}%)</span>
                    </div>
                    <div style='font-size:0.78rem;color:#64748b;font-weight:600;margin-top:6px;
                                text-transform:uppercase;letter-spacing:.05em;'>Risk Level</div>
                    <div style='font-size:1rem;font-weight:700;color:#0f172a;'>{_sa_rl}</div>
                </div>
                <div style='flex:1;min-width:220px;font-size:0.9rem;color:#334155;
                            line-height:1.5;border-left:2px solid {_cfg["border"]};padding-left:1.1rem;'>
                    <em>{_sa_rat}</em>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _fac_col1, _fac_col2 = st.columns(2)
    with _fac_col1:
        st.markdown("**✅ Supporting Factors**")
        if _sa_sup:
            for f in _sa_sup:
                st.markdown(
                    f"<div style='font-size:0.88rem;padding:4px 0;border-bottom:1px solid #f1f5f9;'>"
                    f"+ {f}</div>", unsafe_allow_html=True)
        else:
            st.caption("No strong supporting factors identified.")
    with _fac_col2:
        st.markdown("**⚠️ Contradicting Factors**")
        if _sa_con:
            for f in _sa_con:
                st.markdown(
                    f"<div style='font-size:0.88rem;padding:4px 0;border-bottom:1px solid #f1f5f9;'>"
                    f"− {f}</div>", unsafe_allow_html=True)
        else:
            st.caption("No contradicting factors detected.")
    st.caption("⚠️ AI-generated analysis for informational purposes only. Not financial advice.")
    st.divider()

    # ── Historical Price Chart ────────────────────────────────
    st.subheader("📈 Historical Gold Price")
    chart_df = df[['Date', 'Close']].copy().sort_values('Date')
    chart_df['Daily_Change'] = chart_df['Close'].diff()
    chart_df['Daily_Change_Pct'] = chart_df['Close'].pct_change() * 100
    chart_df['MA7'] = chart_df['Close'].rolling(window=7).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df['Date'],
            y=chart_df['Close'],
            mode='lines',
            name='Gold Close',
            line=dict(color='#d4af37', width=2.6),
            customdata=np.stack(
                [chart_df['Daily_Change'].fillna(0).values,
                 chart_df['Daily_Change_Pct'].fillna(0).values],
                axis=-1,
            ),
            hovertemplate=(
                '<b>Date:</b> %{x|%Y-%m-%d}<br>'
                '<b>Close:</b> $%{y:,.2f}<br>'
                '<b>Daily Change:</b> %{customdata[0]:+.2f}<br>'
                '<b>Daily %:</b> %{customdata[1]:+.2f}%<extra></extra>'
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df['Date'],
            y=chart_df['MA7'],
            mode='lines',
            name='MA7',
            line=dict(color='#2563eb', width=1.8, dash='dot'),
            hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br><b>MA7:</b> $%{y:,.2f}<extra></extra>',
        )
    )
    # Latest marker — positioned at actual last market data date, never at datetime.now()
    _latest_row = chart_df.iloc[-1]
    _latest_mkt_ts = pd.Timestamp(_latest_row['Date'])
    fig.add_trace(
        go.Scatter(
            x=[_latest_mkt_ts],
            y=[_latest_row['Close']],
            mode='markers',
            name='Latest Close',
            showlegend=False,
            marker=dict(size=11, color='#ef4444', line=dict(color='white', width=1.4)),
            hovertemplate=(
                f'<b>Latest Market Close ({_latest_mkt_ts.strftime("%Y-%m-%d")})</b><br>'
                '<b>Close:</b> $%{y:,.2f}<extra></extra>'
            ),
        )
    )
    close_min = float(chart_df['Close'].min())
    close_max = float(chart_df['Close'].max())
    padding = max((close_max - close_min) * 0.06, 1.0)
    fig.update_layout(
        template='plotly_white',
        height=460,
        margin=dict(l=16, r=16, t=74, b=54),
        legend=dict(orientation='h', yanchor='top', y=-0.22, xanchor='left', x=0.0),
        hovermode='x unified',
        xaxis=dict(
            title='Date',
            showgrid=False,
            rangeselector=dict(
                x=0, y=1.14,
                bgcolor='rgba(248, 250, 252, 0.95)',
                activecolor='rgba(37, 99, 235, 0.22)',
                buttons=[
                    dict(count=7,  label='1W', step='day',   stepmode='backward'),
                    dict(count=1,  label='1M', step='month', stepmode='backward'),
                    dict(count=3,  label='3M', step='month', stepmode='backward'),
                    dict(step='all', label='All'),
                ]
            ),
        ),
        yaxis=dict(
            title='Gold Price (USD)',
            showgrid=True,
            gridcolor='rgba(15, 23, 42, 0.08)',
            zeroline=False,
            range=[close_min - padding, close_max + padding],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Chart covers {len(chart_df)} trading sessions. "
        f"Latest market close: **{_latest_mkt_ts.strftime('%Y-%m-%d')}** "
        f"at **${float(_latest_row['Close']):,.2f}**."
    )


# ============================================================
# PAGE: 💹 TRADING SIMULATOR
# ============================================================
elif _active_page == "💹 Trading Simulator":

    current_price = float(last_price)
    forecast_move_pct = abs(hybrid_change) / max(current_price, 1e-9) * 100
    if forecast_move_pct >= 0.45:
        confidence = "High"
    elif forecast_move_pct >= 0.20:
        confidence = "Medium"
    else:
        confidence = "Low"
    bullish_hybrid = "BULLISH" in str(sentiment_label).upper()

    # ── Portfolio Summary ─────────────────────────────────────
    st.subheader("💼 Portfolio Summary")
    pos_value      = st.session_state.sim_position_oz * current_price
    total_equity   = st.session_state.sim_wallet_balance + pos_value
    unrealized_pnl = (current_price - prev_price) * st.session_state.sim_position_oz
    _p1, _p2, _p3, _p4 = st.columns(4)
    _p1.metric("Cash Balance", f"${st.session_state.sim_wallet_balance:,.2f}")
    _p2.metric("Gold Held", f"{st.session_state.sim_position_oz:,.4f} oz")
    _p3.metric("Total Equity", f"${total_equity:,.2f}")
    _p4.metric(
        "Unrealised P&L", f"${unrealized_pnl:+,.2f}",
        delta_color="normal" if unrealized_pnl >= 0 else "inverse",
    )
    st.caption(
        f"Live Gold Price: **${current_price:,.2f}**  ·  "
        f"Signal: **{sentiment_label}**  ·  "
        f"Forecast Confidence: **{confidence}**  ·  "
        f"Market Data: **{_market_date_str}**"
    )
    st.divider()

    # ── Execute Trade ─────────────────────────────────────────
    st.subheader("⚡ Execute Trade")
    amount = st.number_input("Trade Amount (oz)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
    _trade_cost = amount * current_price
    st.caption(f"Estimated cost / proceeds at current price: **${_trade_cost:,.2f}**")
    _tc1, _tc2 = st.columns(2)
    if _tc1.button("🟢  BUY GOLD", use_container_width=True, type="primary"):
        if st.session_state.sim_wallet_balance >= _trade_cost:
            st.session_state.sim_wallet_balance -= _trade_cost
            st.session_state.sim_position_oz    += amount
            st.session_state.auto_trade_history.append(
                {"Time": datetime.now().strftime("%H:%M:%S"), "Action": f"MANUAL BUY {amount:.4f} oz @ ${current_price:,.2f}"}
            )
            _save_portfolio()
            st.toast(f"✅ Simulated BUY: {amount:.2f} oz @ ${current_price:,.2f}")
        else:
            st.warning("Insufficient simulated cash balance.")
    if _tc2.button("🔴  SELL GOLD", use_container_width=True):
        sell_oz = min(amount, st.session_state.sim_position_oz)
        if sell_oz > 0:
            st.session_state.sim_wallet_balance += sell_oz * current_price
            st.session_state.sim_position_oz    -= sell_oz
            st.session_state.auto_trade_history.append(
                {"Time": datetime.now().strftime("%H:%M:%S"), "Action": f"MANUAL SELL {sell_oz:.4f} oz @ ${current_price:,.2f}"}
            )
            _save_portfolio()
            st.toast(f"✅ Simulated SELL: {sell_oz:.2f} oz @ ${current_price:,.2f}")
        else:
            st.warning("No simulated gold position to sell.")
    st.divider()

    # ── Smart Auto-Trade Engine ───────────────────────────────
    st.subheader("🤖 Smart Auto-Trade Engine")
    auto_enabled = st.toggle("Enable Auto-Trade Engine", value=False)
    _ar1, _ar2 = st.columns(2)
    take_profit    = _ar1.number_input("Take Profit ($)",   min_value=0.0, value=round(current_price * 1.010, 2), step=1.0)
    stop_loss      = _ar2.number_input("Stop Loss ($)",     min_value=0.0, value=round(current_price * 0.990, 2), step=1.0)
    _ar3, _ar4 = st.columns(2)
    add_on_trigger = _ar3.number_input("Add-on Trigger ($)", min_value=0.0, value=round(current_price * 1.005, 2), step=1.0)
    auto_add_size  = _ar4.number_input("Auto Add (oz)",      min_value=0.1, max_value=100.0, value=0.5, step=0.1)
    if auto_enabled:
        triggered_actions = []
        if prev_price < take_profit <= current_price and st.session_state.sim_position_oz > 0:
            sell_oz = st.session_state.sim_position_oz
            st.session_state.sim_wallet_balance += sell_oz * current_price
            st.session_state.sim_position_oz = 0.0
            triggered_actions.append(f"TP SELL {sell_oz:.2f} oz @ ${current_price:,.2f}")
        if prev_price > stop_loss >= current_price and st.session_state.sim_position_oz > 0:
            sell_oz = st.session_state.sim_position_oz
            st.session_state.sim_wallet_balance += sell_oz * current_price
            st.session_state.sim_position_oz = 0.0
            triggered_actions.append(f"SL SELL {sell_oz:.2f} oz @ ${current_price:,.2f}")
        if (prev_price < add_on_trigger <= current_price
                and bullish_hybrid and confidence in ("Medium", "High")):
            add_cost = auto_add_size * current_price
            if st.session_state.sim_wallet_balance >= add_cost:
                st.session_state.sim_wallet_balance -= add_cost
                st.session_state.sim_position_oz    += auto_add_size
                triggered_actions.append(f"AUTO ADD {auto_add_size:.2f} oz @ ${current_price:,.2f} ({confidence})")
            else:
                triggered_actions.append("AUTO ADD skipped — insufficient cash")
        for act in triggered_actions:
            st.session_state.auto_trade_history.append(
                {"Time": datetime.now().strftime("%H:%M:%S"), "Action": act}
            )
        if triggered_actions:
            _save_portfolio()
            st.success("Auto triggered: " + " | ".join(triggered_actions))
    st.divider()

    # ── Trade Action History ──────────────────────────────────
    st.subheader("📋 Trade Action History")
    if st.session_state.auto_trade_history:
        st.dataframe(
            pd.DataFrame(st.session_state.auto_trade_history[-20:]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No auto-trading actions recorded yet.")


# ============================================================
# PAGE: 🌐 AI INTELLIGENCE
# ============================================================
elif _active_page == "🌐 AI Intelligence":

    # ── Market Intelligence Report ────────────────────────────
    st.subheader("🌐 AI Global Intelligence Report")
    st.caption("Powered by Google Gemini 2.5 Flash with real-time Google Search grounding.")
    if st.button("Generate Real-Time Market Intelligence Report", type="primary"):
        with st.spinner("Analyzing global news and market signals..."):
            print("[Streamlit] 'Generate Market Intelligence Report' button pressed")
            analysis = _call_genkit_market_intelligence()
            print(f"[Streamlit] analysis complete — source: {analysis.get('source', 'unknown')}")
            # Track previous sentiment for reversal detection before overwriting
            if st.session_state.news_analysis:
                st.session_state.prev_sentiment = st.session_state.news_analysis.get('sentiment', 'Unknown')
            st.session_state.news_analysis = analysis
            st.session_state.dismissed_alert_ids = set()  # fresh report = fresh alerts
            st.rerun()

    if st.session_state.news_analysis:
        _na = st.session_state.news_analysis
        _na_sources = _na.get('sources', [])
        _na_source_status = _na.get('source_status', 'unavailable')
        _na_context_source = _na.get('context_source', 'Google Search Grounding')

        # ── Context source indicator ──────────────────────────────────────────
        _ctx_icon_map = {
            "Vertex AI Search":        ("🔍", "#1a56db"),
            "Google Search Grounding": ("🌐", "#059669"),
            "Python Fallback":         ("🐍", "#d97706"),
        }
        _ctx_icon, _ctx_color = _ctx_icon_map.get(_na_context_source, ("ℹ️", "#64748b"))
        st.markdown(
            f"<div style='font-size:0.78rem;color:{_ctx_color};margin-bottom:6px;'>"
            f"{_ctx_icon} <b>Context Source:</b> {_na_context_source}</div>",
            unsafe_allow_html=True,
        )

        if _na['sentiment'] == "Bullish":
            st.success(f"### 🚀 Live Market Sentiment: **{_na['sentiment']}**")
        elif _na['sentiment'] == "Bearish":
            st.error(f"### 🔻 Live Market Sentiment: **{_na['sentiment']}**")
        else:
            st.warning(f"### ⚖️ Live Market Sentiment: **{_na['sentiment']}**")

        reason_text = str(_na.get('reason', 'No summary reason available.'))
        if len(reason_text) > 220:
            reason_text = reason_text[:220].rstrip() + "..."
        st.info(f"**AI Summary Reason:** {reason_text}")

        with st.container(border=True):
            st.markdown(_na['full_report'])

        # ── Genkit structured fields (only shown when Genkit powered) ─────────
        _key_drivers = _na.get('key_drivers', [])
        _risk_factors = _na.get('risk_factors', [])
        _price_outlook = _na.get('price_outlook', {})
        _confidence = _na.get('confidence', '')
        _source = _na.get('source', '')

        if _key_drivers or _risk_factors or _price_outlook:
            with st.expander("📊 Structured Analysis" + (" · Powered by Genkit" if _source == 'genkit' else ""), expanded=True):
                cols = st.columns(2)
                with cols[0]:
                    if _key_drivers:
                        st.markdown("**🚀 Key Drivers**")
                        for d in _key_drivers:
                            st.markdown(f"- {d}")
                    if _price_outlook:
                        st.markdown("**📅 Price Outlook**")
                        if _price_outlook.get('short_term'):
                            st.markdown(f"- **Short-term:** {_price_outlook['short_term']}")
                        if _price_outlook.get('medium_term'):
                            st.markdown(f"- **Medium-term:** {_price_outlook['medium_term']}")
                with cols[1]:
                    if _risk_factors:
                        st.markdown("**⚠️ Risk Factors**")
                        for r in _risk_factors:
                            st.markdown(f"- {r}")
                    if _confidence:
                        badge_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(_confidence, "⚪")
                        st.markdown(f"**Confidence:** {badge_color} {_confidence}")

        # ── Grounded Sources ──────────────────────────────────
        st.markdown("#### 📎 Grounded Sources")
        if _na_sources:
            for _src in _na_sources:
                _src_title = _src.get('title', 'Untitled source')
                _src_url   = _src.get('url')
                if _src_url:
                    st.markdown(f"- [{_src_title}]({_src_url})")
        else:
            if _na_source_status == "fallback_no_search":
                st.caption("No source links — fallback mode without Google Search grounding.")
            elif _na_source_status == "no_grounding":
                st.caption("No source links — model response did not include grounding metadata.")
            elif _na_source_status == "grounding_without_sources":
                st.caption("Grounding metadata present, but no valid URL sources were returned.")
            else:
                st.caption("No grounded source links were returned for this summary.")
    else:
        st.info("Click **Generate Real-Time Market Intelligence Report** above to load live news analysis.")
        st.caption("The AI will analyse current gold market conditions using Google Search-grounded sources.")

    st.divider()

    # ── Agent Reasoning Trace ─────────────────────────────────
    if st.session_state.last_workflow_was_agentic and st.session_state.last_workflow_steps:
        st.subheader("🔍 Last Agent Reasoning Trace")
        _step_icons   = {
            "MarketDataAgent": "📊", "NewsAnalysisAgent": "📰",
            "RiskGuardAgent": "🛡️", "DecisionExplanationAgent": "🤖",
        }
        _status_color = {"ok": "#16a34a", "fallback": "#d97706", "unavailable": "#94a3b8", "error": "#dc2626"}
        with st.expander("View Agent Reasoning Steps", expanded=True):
            for i, step in enumerate(st.session_state.last_workflow_steps, 1):
                agent   = step.get("agent", "Agent")
                status  = step.get("status", "ok")
                summary = step.get("summary", "")
                icon    = _step_icons.get(agent, "▶")
                color   = _status_color.get(status, "#64748b")
                st.markdown(
                    f"<div style='padding:6px 0;border-bottom:1px solid #f1f5f9;'>"
                    f"<span style='font-weight:700;color:#0f172a;'>{icon} Step {i}: {agent}</span>"
                    f"<span style='float:right;font-size:0.75rem;color:{color};font-weight:600;'>"
                    f"{'✓' if status == 'ok' else '⚠'} {status.upper()}</span><br>"
                    f"<span style='font-size:0.82rem;color:#475569;'>{summary}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.divider()

    # ── Inline Chat Assistant ─────────────────────────────────
    st.subheader("💬 Chat with Spectre Gold AI")
    st.caption("Ask about current signals, forecast logic, risk factors, or market intelligence.")

    # Render existing history using native chat bubbles
    if st.session_state.chat_history:
        for _msg in st.session_state.chat_history[-20:]:
            with st.chat_message(_msg.get("role", "assistant")):
                st.write(_msg.get("content", ""))
                st.caption(_msg.get("time", ""))
    else:
        with st.chat_message("assistant"):
            st.write("Hello, I am Spectre Gold AI. Ask me to explain your current signal, forecast, or risks.")

    _inline_input = st.chat_input("Ask about current gold signals, risks, or market conditions...")
    if _inline_input and _inline_input.strip():
        _append_chat_message("user", _inline_input.strip(), read=False)
        with st.chat_message("user"):
            st.write(_inline_input.strip())
        with st.chat_message("assistant"):
            with st.spinner("Typing..."):
                _pctx = _build_platform_context(
                    last_price=last_price,
                    tech_predicted_price=tech_predicted_price,
                    final_predicted_price=final_predicted_price,
                    hybrid_change=hybrid_change,
                    sentiment_label=sentiment_label,
                    sentiment_score=sentiment_score,
                    prev_price=prev_price,
                )
                _reply, _steps, _was_agentic, _src = agentic_dispatch(_inline_input.strip(), _pctx)
                print(f"[Streamlit] inline chat answered by source: {_src}")
            st.write(_reply)
            st.caption(f"{_chat_now()}  ·  {_SOURCE_LABELS.get(_src, _src)}")
        _append_chat_message("assistant", _reply, read=True, source=_src)
        _mark_last_user_read()
        st.session_state.last_workflow_steps = _steps
        st.session_state.last_workflow_was_agentic = _was_agentic
        st.rerun()


# ============================================================
# FLOATING CHAT WIDGET — available on all pages (fixed position)
# ============================================================
st.markdown(
    """
    <style>
    .st-key-chat_launcher_root {
        position: fixed;
        right: 20px;
        bottom: 20px;
        z-index: 1002;
        width: auto !important;
    }
    .st-key-chat_launcher_root button {
        width: 72px;
        height: 72px;
        border-radius: 999px;
        border: 2px solid rgba(255, 255, 255, 0.92);
        background: radial-gradient(circle at 30% 28%, #38bdf8 0%, #0ea5e9 45%, #0369a1 100%);
        color: #fff;
        font-size: 28px;
        box-shadow: 0 10px 28px rgba(2, 132, 199, 0.45), 0 0 0 8px rgba(14, 165, 233, 0.18);
        transition: transform 0.16s ease, box-shadow 0.16s ease;
    }
    .st-key-chat_launcher_root button:hover {
        transform: scale(1.06);
        box-shadow: 0 14px 34px rgba(2, 132, 199, 0.5), 0 0 0 10px rgba(14, 165, 233, 0.22);
    }
    .st-key-chat_popup_root {
        position: fixed;
        right: 20px;
        bottom: 90px;
        z-index: 1001;
        width: min(390px, calc(100vw - 24px));
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.12);
        border-radius: 16px;
        box-shadow: 0 14px 36px rgba(15, 23, 42, 0.2);
        padding: 0.7rem 0.8rem 0.85rem 0.8rem;
    }
    .sg-chat-title { font-weight: 700; font-size: 1.02rem; margin: 0; color: #0f172a; }
    .sg-chat-subtitle { margin: 0.15rem 0 0.6rem 0; color: #475569; font-size: 0.86rem; }
    .sg-chat-history {
        height: 300px; overflow-y: auto;
        border: 1px solid rgba(15, 23, 42, 0.1); border-radius: 12px;
        padding: 0.65rem;
        background: linear-gradient(180deg, #f8fbff 0%, #f1f5f9 100%);
        margin-bottom: 0.55rem;
    }
    .sg-row { display: flex; align-items: flex-end; gap: 0.45rem; margin-bottom: 0.5rem; }
    .sg-assistant-row { justify-content: flex-start; }
    .sg-user-row { justify-content: flex-end; }
    .sg-avatar {
        width: 28px; height: 28px; border-radius: 999px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.67rem; font-weight: 700; flex-shrink: 0;
    }
    .sg-assistant-avatar { background: #0ea5e9; color: #fff; }
    .sg-user-avatar { background: #cbd5e1; color: #1e293b; }
    .sg-bubble-wrap { max-width: 82%; }
    .sg-message {
        font-size: 0.9rem; line-height: 1.45;
        padding: 0.56rem 0.72rem; border-radius: 14px;
        white-space: pre-wrap; word-break: break-word;
    }
    .sg-user { background: #dbeafe; color: #1e3a8a; border-bottom-right-radius: 5px; }
    .sg-assistant { background: #e2e8f0; color: #0f172a; border-bottom-left-radius: 5px; }
    .sg-meta { color: #64748b; font-size: 0.7rem; margin-top: 0.18rem; padding: 0 0.1rem; }
    .sg-read { margin-left: 0.35rem; color: #0ea5e9; font-weight: 700; }
    .sg-unread { color: #94a3b8; }
    .sg-typing { display: inline-flex; gap: 0.25rem; align-items: center; min-height: 32px; }
    .sg-typing span {
        width: 6px; height: 6px; border-radius: 999px; background: #64748b;
        display: inline-block; animation: sg-bounce 1.2s infinite ease-in-out;
    }
    .sg-typing span:nth-child(2) { animation-delay: 0.12s; }
    .sg-typing span:nth-child(3) { animation-delay: 0.24s; }
    .sg-source { color: #0ea5e9; font-size: 0.68rem; margin-top: 0.1rem; padding: 0 0.1rem; font-style: italic; }
    .st-key-chat_popup_root button[kind='primary'] {
        min-height: 38px; border-radius: 10px; font-size: 1rem; font-weight: 700; padding: 0;
    }
    @keyframes sg-bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.6; }
        40% { transform: scale(1); opacity: 1; }
    }
    .st-key-chat_popup_root [data-testid='InputInstructions'] { display: none; }
    .st-key-chat_popup_minimize button {
        border-radius: 999px; border: 1px solid rgba(15, 23, 42, 0.2);
        width: 34px; height: 34px; padding: 0;
    }
    @media (max-width: 640px) {
        .st-key-chat_launcher_root { right: 12px; bottom: 14px; }
        .st-key-chat_popup_root { right: 10px; bottom: 78px; width: calc(100vw - 20px); }
        .sg-chat-history { height: 240px; }
        .sg-bubble-wrap { max-width: 86%; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

chat_launcher = st.container(key="chat_launcher_root")
with chat_launcher:
    launcher_clicked = st.button("💬", key="chat_launcher")

if launcher_clicked:
    st.session_state.chat_widget_open = not st.session_state.chat_widget_open
    st.rerun()

if st.session_state.chat_widget_open:
    chat_popup = st.container(key="chat_popup_root")
    with chat_popup:
        title_col, btn_col = st.columns([0.82, 0.18])
        with title_col:
            st.markdown("<p class='sg-chat-title'>Spectre Gold Chat</p>", unsafe_allow_html=True)
            st.markdown("<p class='sg-chat-subtitle'>Ask about current signals, forecast logic, and risk context.</p>", unsafe_allow_html=True)
        with btn_col:
            if st.button("✕", key="chat_popup_minimize"):
                st.session_state.chat_widget_open = False
                st.rerun()

        history_placeholder = st.empty()
        history_html = _render_history_html(st.session_state.chat_history)
        history_placeholder.markdown(f"<div class='sg-chat-history'>{history_html}</div>", unsafe_allow_html=True)

        with st.form("chat_widget_form", clear_on_submit=True):
            input_col, send_col = st.columns([0.84, 0.16])
            with input_col:
                user_message = st.text_input(
                    "Message",
                    key="chat_widget_input",
                    label_visibility="collapsed",
                    placeholder="Type your question...",
                )
            with send_col:
                send_clicked = st.form_submit_button("➤", use_container_width=True, type="primary")

        if send_clicked and user_message and user_message.strip():
            _append_chat_message("user", user_message.strip(), read=False)
            st.session_state.chat_pending_message = user_message.strip()
            st.session_state.chat_is_typing = True

            history_html = _render_history_html(st.session_state.chat_history)
            history_placeholder.markdown(f"<div class='sg-chat-history'>{history_html}</div>", unsafe_allow_html=True)

            platform_context = _build_platform_context(
                last_price=last_price,
                tech_predicted_price=tech_predicted_price,
                final_predicted_price=final_predicted_price,
                hybrid_change=hybrid_change,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                prev_price=prev_price,
            )

            assistant_reply, workflow_steps, was_agentic, chat_source = agentic_dispatch(
                st.session_state.chat_pending_message, platform_context
            )
            print(f"[Streamlit] floating widget answered by source: {chat_source}")
            _append_chat_message("assistant", assistant_reply, read=True, source=chat_source)
            _mark_last_user_read()

            st.session_state.last_workflow_steps     = workflow_steps
            st.session_state.last_workflow_was_agentic = was_agentic
            st.session_state.chat_pending_message    = None
            st.session_state.chat_is_typing          = False

            history_html = _render_history_html(st.session_state.chat_history)
            history_placeholder.markdown(f"<div class='sg-chat-history'>{history_html}</div>", unsafe_allow_html=True)
