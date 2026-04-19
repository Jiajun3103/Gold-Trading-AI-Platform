"""
llm_client.py — Centralized Gemini / Vertex AI LLM client for Spectre Gold AI.

All modules that need to call Gemini should import from here instead of
doing their own SDK initialization. This makes it easy to:
  - switch from google.genai to Vertex AI SDK in one place
  - handle both SDK versions with one fallback chain
  - unit-test by monkey-patching generate_text()

Environment variables (one is enough):
  GOOGLE_API_KEY   or   GEMINI_API_KEY
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any

from dotenv import load_dotenv

_ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOTENV_PATH = os.path.join(_ROOT_DIR, ".env")

# ---------------------------------------------------------------------------
# Optional SDK imports — gracefully absent if not installed
# ---------------------------------------------------------------------------
try:
    _genai = importlib.import_module("google.genai")
except ImportError:
    _genai = None

try:
    _legacy = importlib.import_module("google.generativeai")
except ImportError:
    _legacy = None


def get_api_key() -> str | None:
    """Return the Google API key from environment, or None if unset."""
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return key.strip() if isinstance(key, str) and key.strip() else None


def generate_text(
    prompt: str,
    model: str = "gemini-2.5-flash",
    *,
    response_schema: dict[str, Any] | None = None,
) -> str | None:
    """
    Generate text from Gemini.  Tries the new SDK first, falls back to the
    legacy SDK.  Returns the text string, or None if both fail.

    Args:
        prompt:          Full prompt string.
        model:           Gemini model name.
        response_schema: Optional JSON schema dict for structured output
                         (only honoured by the new SDK).
    """
    api_key = get_api_key()
    if not api_key:
        return None

    # ── new SDK (google.genai) ─────────────────────────────────────────────
    if _genai is not None:
        try:
            client = _genai.Client(api_key=api_key)
            kwargs: dict[str, Any] = {"model": model, "contents": prompt}

            if response_schema is not None:
                try:
                    types = importlib.import_module("google.genai.types")
                    gc = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                    )
                    kwargs["config"] = gc
                except Exception:
                    pass  # schema support optional; proceed without it

            response = client.models.generate_content(**kwargs)
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text
        except Exception:
            pass  # fall through to legacy

    # ── legacy SDK (google.generativeai) ──────────────────────────────────
    if _legacy is not None:
        try:
            _legacy.configure(api_key=api_key)
            m = _legacy.GenerativeModel(model_name=model)
            response = m.generate_content(prompt)
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text
        except Exception:
            pass

    return None


def generate_json(
    prompt: str,
    schema: dict[str, Any],
    model: str = "gemini-2.5-flash",
) -> dict | None:
    """
    Convenience wrapper: generate structured JSON output and parse it.
    Falls back to regex extraction if structured mode is unavailable.

    Returns a dict on success, or None if the call failed.
    """
    import re

    text = generate_text(prompt, model=model, response_schema=schema)
    if not text:
        return None

    # If the model returned clean JSON, parse directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract first {...} block from the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None
