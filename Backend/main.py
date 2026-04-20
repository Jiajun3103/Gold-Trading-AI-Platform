from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
import sys 
from dotenv import load_dotenv
from News.chatbot import generate_platform_chat_response, agentic_dispatch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from News.news import get_ai_market_report, get_market_sentiment
from News.risk_guard import evaluate_risk
from News.agents import WorkflowRunner
from News.decision_engine import build_suggested_action
from News.smart_alerts import detect_alerts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 1. Load Environment Variables (API Keys)
load_dotenv()

app = FastAPI(title="Aurum AI Trading Backend")

# 2. Allow your Frontend (React/Next.js) to talk to this Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your website URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the AI Model and Scaler when the server starts
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "gold_price_model.keras")
SCALER_PATH = os.path.join(PROJECT_ROOT, "Models", "gold_scaler.bin")

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.get("/")
def home():
    key = os.environ.get("GOOGLE_API_KEY")
    masked_key = f"{key[:4]}***" if key else "NOT FOUND"
    return {"message": "Aurum AI Backend is running!", "debug_key": masked_key}
    #return {"message": "Aurum AI Backend is running!"}

@app.get("/predict-price")
async def predict_price():
    """
    This endpoint will be called by your 'Prediction Charts Page'
    """
    try:
        # 1. Load the latest data from your CSV
        # Path is ../Data_preparation/GC_F_historical.csv
        csv_path = os.path.join(PROJECT_ROOT, "Data_preparation", "GC_F_historical.csv")
        df = pd.read_csv(csv_path, skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
        df.dropna(inplace=True)
        
        # 2. Get the last 60 days of prices and scale them
        last_60_days = df['Close'].values[-60:].reshape(-1, 1)
        scaled_input = scaler.transform(last_60_days)
        
        # 3. Reshape for the AI (1 sample, 60 time steps, 1 feature)
        current_batch = scaled_input.reshape((1, 60, 1))
        
        # 4. Predict tomorrow (Technical Analysis)
        prediction_scaled = model.predict(current_batch)
        prediction_tech = float(scaler.inverse_transform(prediction_scaled)[0][0])
        
        # --- NEW: Integrate News Sentiment (Fundamental Analysis) ---
        report_data = get_ai_market_report()
        sentiment_label, sentiment_score = get_market_sentiment(report_data)
        
        adjustment = prediction_tech * (sentiment_score * 0.005)
        final_prediction = prediction_tech + adjustment
        
        return {
            "status": "success",
            "last_closing_price": float(df['Close'].iloc[-1]),
            "technical_prediction": round(prediction_tech, 2),
            "news_sentiment": sentiment_label,
            "sentiment_adjustment": round(adjustment, 2),
            "predicted_tomorrow_price": round(final_prediction, 2), # 这是最终的混合价格
            "trend": "UP" if final_prediction > df['Close'].iloc[-1] else "DOWN",
            "reason": report_data.get("reason", ""),
            "full_report": report_data.get("full_report", ""),
            "sources": report_data.get("sources", []),
            "source_status": report_data.get("source_status", "unavailable"),
            "sdk": report_data.get("sdk", "none"),
            "search_mode": report_data.get("search_mode", False),
            "raw_has_grounding": report_data.get("raw_has_grounding", False),
            "fallback_used": report_data.get("fallback_used", False)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/market-sentiment")
async def get_sentiment():
    """
    This will call your News Agent to get real-world analysis
    """
    report_data = get_ai_market_report()
    sentiment, score = get_market_sentiment(report_data)
    
    # return {"sentiment": "Bullish", "score": 0.8, "reason": "Geopolitical tensions rising."}
    return {
        "sentiment": sentiment, 
        "score": score, 
        "reason": report_data.get("reason", ""),
        "full_report": report_data.get("full_report", ""),
        "sources": report_data.get("sources", []),
        "source_status": report_data.get("source_status", "unavailable"),
        "sdk": report_data.get("sdk", "none"),
        "search_mode": report_data.get("search_mode", False),
        "raw_has_grounding": report_data.get("raw_has_grounding", False),
        "fallback_used": report_data.get("fallback_used", False),
        "status": "Live from Gemini AI"
    }

class RiskRequest(BaseModel):
    sentiment: str = "Neutral"
    sentiment_score: float = 0.0
    hybrid_change: float = 0.0
    current_price: float = 0.0
    prev_price: float = 0.0
    news_analysis: dict | None = None


@app.post("/risk-analysis")
async def risk_analysis(req: RiskRequest):
    """
    AI Risk Guard endpoint.
    Evaluates signal consistency, news credibility, and market volatility.
    Returns risk_level, confidence_score, factors, warnings, and a summary.

    Example request body:
    {
        "sentiment": "Bearish",
        "sentiment_score": -0.005,
        "hybrid_change": 12.5,
        "current_price": 3350.0,
        "prev_price": 3290.0,
        "news_analysis": { ... }
    }
    """
    try:
        result = evaluate_risk(
            sentiment=req.sentiment,
            sentiment_score=req.sentiment_score,
            hybrid_change=req.hybrid_change,
            current_price=req.current_price,
            prev_price=req.prev_price,
            news_analysis=req.news_analysis,
        )
        result["status"] = "success"
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "risk_level": "Unknown",
            "confidence_score": 0.0,
            "factors": [],
            "warnings": [],
            "summary": "Risk analysis failed — please retry.",
        }


class ActionRequest(BaseModel):
    sentiment: str = "Neutral"
    sentiment_score: float = 0.0
    hybrid_change: float = 0.0
    current_price: float = 0.0
    prev_price: float = 0.0
    risk_result: dict | None = None
    news_analysis: dict | None = None


@app.post("/suggested-action")
async def suggested_action(req: ActionRequest):
    """
    Explainable Suggested Action endpoint.
    Combines forecast direction, news sentiment, and risk evaluation into
    a structured Buy / Hold / Sell / Wait decision card.

    Example request:
    {
        "sentiment": "Bullish",
        "sentiment_score": 0.005,
        "hybrid_change": 15.0,
        "current_price": 3350.0,
        "prev_price": 3335.0,
        "risk_result": { "risk_level": "Low", "confidence_score": 0.75, ... },
        "news_analysis": { "sources": [...], ... }
    }
    """
    try:
        card = build_suggested_action(
            sentiment=req.sentiment,
            sentiment_score=req.sentiment_score,
            hybrid_change=req.hybrid_change,
            current_price=req.current_price,
            prev_price=req.prev_price,
            risk_result=req.risk_result,
            news_analysis=req.news_analysis,
        )
        card["status"] = "success"
        return card
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "suggested_action": "Wait",
            "confidence": "Low",
            "confidence_score": 0.0,
            "risk_level": "Unknown",
            "supporting_factors": [],
            "contradicting_factors": [],
            "rationale": "Action evaluation failed — please retry.",
        }


class AlertsRequest(BaseModel):
    current_price: float = 0.0
    prev_price: float = 0.0
    sentiment: str = "Neutral"
    prev_sentiment: str = "Unknown"
    risk_result: dict | None = None
    news_analysis: dict | None = None


@app.post("/smart-alerts")
async def smart_alerts(req: AlertsRequest):
    """
    Smart Alerts endpoint — detects up to 5 alert conditions:
      1. LARGE_PRICE_MOVEMENT  — daily move > 1% (warning) or > 2.5% (critical)
      2. SENTIMENT_REVERSAL    — sentiment changed since last news fetch
      3. HIGH_RISK_CONDITION   — Risk Guard returned "High" risk level
      4. CONFLICTING_SIGNALS   — SIGNAL_CONFLICT warning present
      5. SUSPICIOUS_NEWS       — LOW_CREDIBILITY_NEWS or < 2 grounded sources

    Returns sorted list (critical first) with id / severity / title / message.

    Example request:
    {
        "current_price": 3350.0,
        "prev_price": 3335.0,
        "sentiment": "Bearish",
        "prev_sentiment": "Bullish",
        "risk_result": { "risk_level": "High", "warnings": ["SIGNAL_CONFLICT"], ... },
        "news_analysis": { "sources": [], "source_status": "fallback", ... }
    }
    """
    try:
        alerts = detect_alerts(
            current_price=req.current_price,
            prev_price=req.prev_price,
            sentiment=req.sentiment,
            prev_sentiment=req.prev_sentiment,
            risk_result=req.risk_result,
            news_analysis=req.news_analysis,
        )
        return {"status": "success", "alerts": alerts, "count": len(alerts)}
    except Exception as e:
        return {"status": "error", "message": str(e), "alerts": [], "count": 0}


# ---------------------------------------------------------------------------
# Genkit — marketIntelligenceFlow integration
# ---------------------------------------------------------------------------

class GenkitMarketIntelligenceRequest(BaseModel):
    question: str = "What is the current gold market outlook?"
    includeAlerts: bool = False


class PriceOutlook(BaseModel):
    short_term: str
    medium_term: str


class GenkitMarketIntelligenceResponse(BaseModel):
    sentiment: str
    sentiment_score: float
    reason: str
    key_drivers: list[str]
    risk_factors: list[str]
    price_outlook: PriceOutlook
    confidence: str
    source: str = "genkit"   # "genkit" | "python_fallback" | "error"
    status: str = "success"
    # Grounded sources from the Python pipeline (merged post-Genkit)
    grounded_sources: list[dict] = []
    source_count: int = 0
    source_status: str = "unavailable"
    # Human-readable context/retrieval backend label for UI display
    context_source: str = "Google Search Grounding"


@app.post("/genkit/market-intelligence", response_model=GenkitMarketIntelligenceResponse)
async def genkit_market_intelligence(req: GenkitMarketIntelligenceRequest):
    """
    Call the Genkit marketIntelligenceFlow and return structured gold market analysis.

    Routing:
    1. Genkit flow server (GENKIT_FLOW_URL env var, default http://localhost:4001)
    2. Python fallback via get_ai_market_report() if Genkit is unavailable

    Request:
        { "question": "What is the current gold market outlook?", "includeAlerts": false }

    Response:
        Structured MarketIntelligenceReport (sentiment, drivers, outlook, etc.)
    """
    print("[FastAPI] /genkit/market-intelligence called — question:", req.question[:80])
    # GENKIT_FLOW_URL must point to the Genkit Flow HTTP server (default port 4001)
    # The Genkit Developer UI runs on port 4000 and is NOT a REST API.
    genkit_base = os.environ.get("GENKIT_FLOW_URL", "http://localhost:4001").rstrip("/")
    genkit_url = f"{genkit_base}/marketIntelligenceFlow"
    print(f"[FastAPI] attempting Genkit request → {genkit_url}")

    # ── Tier 1: Genkit flow server ────────────────────────────────────────────
    try:
        import httpx
        payload = {
            "data": {
                "question": req.question,
                "includeAlerts": req.includeAlerts,
            }
        }
        resp = httpx.post(
            genkit_url,
            json=payload,
            timeout=30.0,
        )
        print(f"[FastAPI] Genkit HTTP status: {resp.status_code}")
        if resp.status_code == 200:
            body = resp.json()
            # Genkit wraps the flow output in {"result": {...}}
            result = body.get("result", body)
            result["source"] = "genkit"
            result["status"] = "success"
            print("[FastAPI] Genkit success — merging Python grounded sources")

            # ── Merge: fetch grounded sources from Python pipeline ────────────
            try:
                py_report = get_ai_market_report()
                grounded_sources = py_report.get("sources", [])
                source_status    = py_report.get("source_status", "unavailable")
                result["grounded_sources"] = grounded_sources
                result["source_count"]     = len(grounded_sources)
                result["source_status"]    = source_status
                result["context_source"]   = py_report.get("context_source", "Google Search Grounding")
                print(f"[FastAPI] Python grounded sources fetched: {len(grounded_sources)} sources (status={source_status})")
            except Exception as src_exc:
                print(f"[FastAPI] Python grounded source fetch failed: {src_exc} — continuing without sources")
                result.setdefault("grounded_sources", [])
                result.setdefault("source_count", 0)
                result.setdefault("source_status", "unavailable")
                result.setdefault("context_source", "Google Search Grounding")

            print("[FastAPI] merge success — returning hybrid Genkit+grounded response")
            return result
        else:
            print(f"[FastAPI] Genkit non-200 ({resp.status_code}), body: {resp.text[:300]}")
    except Exception as exc:
        print(f"[FastAPI] Genkit request failed: {exc}")
        # fall through to Python fallback

    # ── Tier 2: Python fallback (existing News/news.py) ───────────────────────
    print("[FastAPI] fallback to Python report (get_ai_market_report)")
    try:
        raw = get_ai_market_report()
        grounded_sources = raw.get("sources", [])
        source_status    = raw.get("source_status", "unavailable")
        # Map the existing Python report shape to the Genkit response schema
        return {
            "sentiment": raw.get("sentiment", "Neutral"),
            "sentiment_score": _sentiment_to_score(raw.get("sentiment", "Neutral")),
            "reason": raw.get("reason", raw.get("full_report", "No analysis available.")),
            "key_drivers": _extract_list(raw.get("full_report", ""), "drivers", 3),
            "risk_factors": _extract_list(raw.get("full_report", ""), "risks", 2),
            "price_outlook": {
                "short_term": "See full report for details.",
                "medium_term": "See full report for details.",
            },
            "confidence": "Medium",
            "source": "python_fallback",
            "status": "success",
            "grounded_sources": grounded_sources,
            "source_count": len(grounded_sources),
            "source_status": source_status,
            "context_source": raw.get("context_source", "Google Search Grounding"),
        }
    except Exception as e:
        return {
            "sentiment": "Neutral",
            "sentiment_score": 0.0,
            "reason": "Market intelligence temporarily unavailable.",
            "key_drivers": [],
            "risk_factors": [],
            "price_outlook": {"short_term": "Unavailable", "medium_term": "Unavailable"},
            "confidence": "Low",
            "source": "error",
            "status": "error",
        }


def _sentiment_to_score(label: str) -> float:
    """Map sentiment label to a numeric score matching Genkit's schema."""
    mapping = {
        "Strongly Bullish": 0.9,
        "Bullish": 0.6,
        "Neutral": 0.0,
        "Bearish": -0.6,
        "Strongly Bearish": -0.9,
    }
    return mapping.get(label, 0.0)


def _extract_list(text: str, _hint: str, limit: int) -> list[str]:
    """Return up to `limit` non-empty lines from a block of text as a list."""
    lines = [ln.strip("•- \t") for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if len(ln) > 10][:limit]


# ---------------------------------------------------------------------------
# Genkit Chat — goldChatFlow integration
# ---------------------------------------------------------------------------

class GenkitChatContextSchema(BaseModel):
    current_price: float | None = None
    sentiment: str | None = None
    risk_level: str | None = None
    suggested_action: str | None = None


class GenkitChatRequest(BaseModel):
    question: str
    context: GenkitChatContextSchema | None = None


class GenkitChatResponse(BaseModel):
    answer: str
    disclaimer: str
    source: str = "genkit_goldChatFlow"  # "genkit_goldChatFlow" | "python_fallback"


@app.post("/genkit/chat", response_model=GenkitChatResponse)
async def genkit_chat(req: GenkitChatRequest):
    """
    Call the Genkit goldChatFlow for explanation/advice-style gold questions.

    Routing:
    1. Genkit flow server (GENKIT_FLOW_URL env var, default http://localhost:4001)
    2. Python fallback via generate_platform_chat_response() if Genkit unavailable

    Request:
        { "question": "Should I buy gold today?",
          "context": { "current_price": 4879.6, "sentiment": "Bullish", ... } }
    """
    print(f"[FastAPI] /genkit/chat called — question: {req.question[:80]}")
    genkit_base = os.environ.get("GENKIT_FLOW_URL", "http://localhost:4001").rstrip("/")
    genkit_url = f"{genkit_base}/goldChatFlow"
    print(f"[FastAPI] attempting Genkit goldChatFlow → {genkit_url}")

    # ── Tier 1: Genkit flow server ────────────────────────────────────────────
    try:
        import httpx
        ctx = req.context.model_dump(exclude_none=True) if req.context else {}
        payload = {"data": {"question": req.question, "context": ctx}}
        resp = httpx.post(genkit_url, json=payload, timeout=30.0)
        print(f"[FastAPI] Genkit goldChatFlow HTTP status: {resp.status_code}")
        if resp.status_code == 200:
            body = resp.json()
            result = body.get("result", body)
            answer = result.get("answer", "")
            disclaimer = result.get("disclaimer", "")
            if answer:
                print("[FastAPI] Genkit goldChatFlow success")
                return GenkitChatResponse(
                    answer=answer,
                    disclaimer=disclaimer,
                    source="genkit_goldChatFlow",
                )
        else:
            print(f"[FastAPI] Genkit goldChatFlow non-200 ({resp.status_code}): {resp.text[:300]}")
    except Exception as exc:
        print(f"[FastAPI] Genkit goldChatFlow request failed: {exc}")

    # ── Tier 2: Python fallback ───────────────────────────────────────────────
    print("[FastAPI] fallback to Python chat (generate_platform_chat_response)")
    try:
        platform_ctx = {}
        if req.context:
            platform_ctx = {
                "last_price": req.context.current_price,
                "sentiment_label": req.context.sentiment,
                "risk_level": req.context.risk_level,
                "suggested_action": req.context.suggested_action,
            }
        fallback_answer = generate_platform_chat_response(req.question, platform_ctx)
        return GenkitChatResponse(
            answer=fallback_answer,
            disclaimer="This is an AI-generated response. It is not financial advice.",
            source="python_fallback",
        )
    except Exception as exc2:
        print(f"[FastAPI] Python fallback also failed: {exc2}")
        return GenkitChatResponse(
            answer="I'm temporarily unavailable. Please try again shortly.",
            disclaimer="AI chat service is temporarily unavailable.",
            source="error",
        )


# ---------------------------------------------------------------------------
# Agentic Chat
# ---------------------------------------------------------------------------

class AgenticChatRequest(BaseModel):
    user_message: str
    platform_context: dict = {}


@app.post("/agentic-chat")
async def agentic_chat(req: AgenticChatRequest):
    """
    Agentic Decision Workflow endpoint.

    Routing priority:
    1. ADK Agent Service (HTTP) — if ADK_AGENT_SERVICE_URL env var is set.
    2. Local WorkflowRunner (4-agent pipeline).

    Example request:
    {
        "user_message": "Should I buy gold today?",
        "platform_context": { "last_price": 3350.0, ... }
    }
    """
    if not req.user_message or not req.user_message.strip():
        return {"status": "error", "message": "user_message is required"}

    # ── Tier 1: ADK Agent Service (Cloud Run) ────────────────────────────────
    adk_url = os.environ.get("ADK_AGENT_SERVICE_URL", "").rstrip("/")
    if adk_url:
        try:
            import httpx
            payload = {
                "question": req.user_message,
                "platform_context": req.platform_context,
            }
            resp = httpx.post(f"{adk_url}/agent/run", json=payload, timeout=30.0)
            if resp.status_code == 200:
                data = resp.json()
                data["status"] = "success"
                data.setdefault("steps", [])
                data.setdefault("was_agentic", True)
                return data
        except Exception:
            pass  # fall through to local runner

    # ── Tier 2: Local WorkflowRunner ─────────────────────────────────────────
    try:
        result = WorkflowRunner().run(req.user_message, req.platform_context)
        result["status"] = "success"
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "answer": "Agentic workflow failed — please retry.",
            "steps": [],
            "was_agentic": False,
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8010)