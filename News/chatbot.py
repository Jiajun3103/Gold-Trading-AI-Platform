import os
import json
import importlib
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(ROOT_DIR, ".env")

try:
    genai = importlib.import_module("google.genai")
except ImportError:
    genai = None

try:
    legacy_genai = importlib.import_module("google.generativeai")
except ImportError:
    legacy_genai = None


SYSTEM_PROMPT = """
You are Spectre Gold AI Chatbot, a conversational in-platform assistant for a Gold Trading AI Platform.

Primary behavior:
- Act like a helpful chat assistant, not a report generator.
- Sound clear, natural, and supportive.
- Keep answers concise by default.

What to explain:
- current sentiment
- predicted gold price
- hybrid forecast meaning
- risk monitor signals
- AI intelligence summary
- likely reasons behind bullish, bearish, or neutral signals
- key risks users should watch

Style rules:
- Answer the user's question first.
- Use short paragraphs and natural transitions.
- Avoid formal report tone, academic style, and long essay-like outputs.
- Avoid excessive headings, bullet lists, and repeated disclaimers.
- Default to about 3 to 6 sentences unless the user asks for more detail.

Reasoning pattern when relevant:
1) Briefly state what the platform currently shows.
2) Explain the main reason in simple terms.
3) Mention one or two risks or uncertainties.
4) Stop unless the user asks for deeper detail.

Important constraints:
- Always use platform context first when available.
- Never invent prices, sources, indicators, news, or portfolio values.
- If data is missing, say so simply.
- Do not give guaranteed profit advice.
- If caution is needed, keep it brief and natural.
""".strip()


def _get_google_api_key():
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return api_key.strip() if isinstance(api_key, str) else None


def _compact_context(platform_context):
    if not isinstance(platform_context, dict):
        return {}

    compact = dict(platform_context)

    news = compact.get("news_analysis")
    if isinstance(news, dict):
        sources = news.get("sources", [])
        compact["news_analysis"] = {
            "sentiment": news.get("sentiment"),
            "reason": news.get("reason"),
            "full_report": news.get("full_report"),
            "source_status": news.get("source_status"),
            "sources": [
                {
                    "title": src.get("title"),
                    "url": src.get("url"),
                    "domain": src.get("domain"),
                }
                for src in sources[:8]
                if isinstance(src, dict)
            ],
        }

    return compact


def _fallback_response(user_message, platform_context):
    sentiment = platform_context.get("sentiment_label", "unknown")
    predicted = platform_context.get("final_predicted_price")
    last_price = platform_context.get("last_price")
    return (
        "Based on the current platform output, live AI chat is temporarily unavailable, "
        "but I can still summarize what we have. "
        f"The current hybrid sentiment is {sentiment}, with a predicted price of "
        f"{predicted if predicted is not None else 'not available'} and latest observed price of "
        f"{last_price if last_price is not None else 'not available'}. "
        "Please retry shortly after checking API availability."
    )


def _generate_with_new_sdk(user_message, platform_context):
    api_key = _get_google_api_key()
    client = genai.Client(api_key=api_key)

    context_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "platform_context": _compact_context(platform_context),
        "user_question": user_message,
    }

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Platform context JSON:\n"
        f"{json.dumps(context_payload, ensure_ascii=True)}\n\n"
        "Answer the user question using only the provided platform context. "
        "If a requested metric is missing, explicitly say it is unavailable in the current platform output."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return (getattr(response, "text", "") or "").strip()


def _generate_with_legacy_sdk(user_message, platform_context):
    api_key = _get_google_api_key()
    legacy_genai.configure(api_key=api_key)
    model = legacy_genai.GenerativeModel(model_name="gemini-2.5-flash")

    context_payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "platform_context": _compact_context(platform_context),
        "user_question": user_message,
    }

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        "Platform context JSON:\n"
        f"{json.dumps(context_payload, ensure_ascii=True)}\n\n"
        "Answer the user question using only the provided platform context. "
        "If a requested metric is missing, explicitly say it is unavailable in the current platform output."
    )

    response = model.generate_content(prompt)
    return (getattr(response, "text", "") or "").strip()


def generate_platform_chat_response(user_message, platform_context):
    if not isinstance(user_message, str) or not user_message.strip():
        return "Please enter a question about the current platform outputs."

    api_key = _get_google_api_key()
    if not api_key:
        return _fallback_response(user_message, platform_context)

    try:
        if genai is not None:
            answer = _generate_with_new_sdk(user_message, platform_context)
            if answer:
                return answer
        if legacy_genai is not None:
            answer = _generate_with_legacy_sdk(user_message, platform_context)
            if answer:
                return answer
    except Exception:
        return _fallback_response(user_message, platform_context)

    return _fallback_response(user_message, platform_context)


# ---------------------------------------------------------------------------
# Agentic dispatch — routes to multi-step workflow or single-shot response
# ---------------------------------------------------------------------------

# Keywords that indicate a gold explanation / advice / outlook question
_EXPLANATION_KEYWORDS = (
    "should i", "buy gold", "sell gold", "hold gold",
    "why is gold", "why gold", "explain", "what does", "what is",
    "bullish", "bearish", "neutral", "outlook", "forecast",
    "good time", "right time", "market look", "price going",
    "risk mean", "medium risk", "high risk", "low risk",
    "signal mean", "recommend", "advice", "suggest",
)


def _is_explanation_question(text: str) -> bool:
    """Return True if the question is explanation/advice-style (route to Genkit first)."""
    lower = text.lower()
    return any(kw in lower for kw in _EXPLANATION_KEYWORDS)


def agentic_dispatch(
    user_message: str,
    platform_context: dict,
) -> tuple[str, list, bool, str]:
    """
    Entry point for the chat widget.

    Returns:
        (answer: str, steps: list, was_agentic: bool, source: str)

    source values:
        "genkit_goldChatFlow"  — answered by Genkit goldChatFlow via FastAPI
        "adk_agent"            — answered by ADK Agent Service
        "workflow_runner"      — answered by local WorkflowRunner
        "gemini_direct"        — answered by single-shot Gemini call
        "fallback"             — static fallback response

    Routing priority:
    0. Genkit goldChatFlow (HTTP via FastAPI) — for explanation/advice questions.
    1. ADK Agent Service (HTTP) — if ADK_AGENT_SERVICE_URL is set in env.
    2. Local WorkflowRunner (4-agent pipeline in agents.py).
    3. Single-shot generate_platform_chat_response fallback.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        return "Please enter a question about the current platform outputs.", [], False, "fallback"

    # ── Tier 0: Genkit goldChatFlow via FastAPI ──────────────────────────────
    # Route explanation/advice-style questions through Genkit for richer answers
    if _is_explanation_question(user_message):
        backend_url = os.environ.get("API_BASE_URL", "http://localhost:8010").rstrip("/")
        genkit_chat_url = f"{backend_url}/genkit/chat"
        print(f"[Chatbot] explanation question detected → trying Genkit via {genkit_chat_url}")
        try:
            import httpx
            ctx = {
                "current_price": platform_context.get("last_price"),
                "sentiment":     platform_context.get("sentiment_label"),
                "risk_level":    platform_context.get("risk_level"),
                "suggested_action": platform_context.get("suggested_action"),
            }
            # Strip None values so Genkit schema optional fields work cleanly
            ctx = {k: v for k, v in ctx.items() if v is not None}
            payload = {"question": user_message, "context": ctx}
            resp = httpx.post(genkit_chat_url, json=payload, timeout=30.0)
            print(f"[Chatbot] Genkit /genkit/chat HTTP status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer", "")
                disclaimer = data.get("disclaimer", "")
                source = data.get("source", "genkit_goldChatFlow")
                if answer:
                    full_reply = answer
                    if disclaimer:
                        full_reply = f"{answer}\n\n*{disclaimer}*"
                    print(f"[Chatbot] Genkit goldChatFlow success (source={source})")
                    return full_reply, [], False, source
            else:
                print(f"[Chatbot] Genkit /genkit/chat non-200 ({resp.status_code}) — falling through")
        except Exception as exc:
            print(f"[Chatbot] Genkit /genkit/chat failed: {exc} — falling through")

    # ── Tier 1: Try the ADK Agent Service over HTTP ──────────────────────────
    adk_url = os.environ.get("ADK_AGENT_SERVICE_URL", "").rstrip("/")
    if adk_url:
        try:
            import httpx  # soft dependency — only needed when ADK service is used
            payload = {
                "question": user_message,
                "platform_context": platform_context,
                "session_id": platform_context.get("session_id"),
            }
            resp = httpx.post(
                f"{adk_url}/agent/run",
                json=payload,
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["answer"], data.get("steps", []), True, "adk_agent"
        except Exception:
            pass  # fall through to local runner

    # ── Tier 2: Local WorkflowRunner (original path) ─────────────────────────
    try:
        from News.agents import WorkflowRunner  # local import avoids circular dep
        result = WorkflowRunner().run(user_message, platform_context)
        if result["was_agentic"] and result.get("answer"):
            return result["answer"], result["steps"], True, "workflow_runner"
    except Exception:
        pass  # fall through to single-shot

    # ── Tier 3: Single-shot LLM call ─────────────────────────────────────────
    answer = generate_platform_chat_response(user_message, platform_context)
    return answer, [], False, "gemini_direct"
