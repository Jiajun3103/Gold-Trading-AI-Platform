"""
server.py — HTTP server exposing the ADK root agent.

This is a standalone FastAPI service (separate from Backend/main.py).
It exposes the root agent over HTTP so:
  - Streamlit / FastAPI can call it via httpx
  - It can be deployed as an independent Cloud Run service
  - It works with OR without the real google-adk library installed

Endpoints:
  POST /agent/run        — main agentic chat
  GET  /agent/health     — liveness probe

Run locally:
  uvicorn agent_service.server:app --port 8020 --reload

Cloud Run:
  CMD ["uvicorn", "agent_service.server:app", "--host", "0.0.0.0", "--port", "8080"]
"""
from __future__ import annotations

import os
import sys
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_service.root_agent import root_agent, _ADK_AVAILABLE  # noqa: E402

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Spectre Gold AI — Agent Service",
    description=(
        "ADK-compatible agent service that orchestrates market snapshot, "
        "risk guard, suggested action, smart alerts, and market intelligence."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AgentRunRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    platform_context: dict = Field(default_factory=dict)
    session_id: str | None = Field(default=None)


class AgentRunResponse(BaseModel):
    answer: str
    steps: list[dict]
    was_agentic: bool
    adk_native: bool
    session_id: str


# ---------------------------------------------------------------------------
# In-memory session store (sufficient for hackathon; swap for Redis in prod)
# ---------------------------------------------------------------------------
_sessions: dict[str, list[dict]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/agent/health")
def health():
    return {
        "status": "ok",
        "adk_native": _ADK_AVAILABLE,
        "agent": root_agent.name,
    }


@app.post("/agent/run", response_model=AgentRunResponse)
def run_agent(req: AgentRunRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # If using the real ADK Agent (google-adk installed), route through Runner
    if _ADK_AVAILABLE:
        try:
            import asyncio
            from google.adk.runners import Runner               # type: ignore
            from google.adk.sessions import InMemorySessionService  # type: ignore
            from google.genai import types as genai_types       # type: ignore

            _svc    = InMemorySessionService()
            _runner = Runner(
                agent=root_agent,
                app_name="spectre_gold_ai",
                session_service=_svc,
            )
            session = _svc.get_session("spectre_gold_ai", "user", session_id)
            if session is None:
                session = _svc.create_session(
                    app_name="spectre_gold_ai",
                    user_id="user",
                    session_id=session_id,
                    state={"platform_context": req.platform_context},
                )

            message = genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=req.question)],
            )

            async def _adk_run():
                final_text = ""
                async for event in _runner.run_async(
                    user_id="user",
                    session_id=session_id,
                    new_message=message,
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        final_text = event.content.parts[0].text
                return final_text

            answer = asyncio.get_event_loop().run_until_complete(_adk_run())
            return AgentRunResponse(
                answer=answer,
                steps=[],
                was_agentic=True,
                adk_native=True,
                session_id=session_id,
            )
        except Exception as exc:
            # Fallback to shim if ADK Runner fails
            pass

    # Shim path (no ADK or ADK Runner failed)
    result = root_agent.run_sync(req.question, req.platform_context)

    # Store session history
    history = _sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": result["answer"]})

    return AgentRunResponse(
        answer=result["answer"],
        steps=result.get("steps", []),
        was_agentic=result.get("was_agentic", True),
        adk_native=result.get("adk_native", False),
        session_id=session_id,
    )
