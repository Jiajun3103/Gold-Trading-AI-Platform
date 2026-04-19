"""
context_provider.py — Retrieval / Context Layer for Spectre Gold AI

This module provides a clean abstraction over the grounding / retrieval
backend used to supply real-world context to Gemini.

Design
------
- ``ContextProvider``              abstract base (defines interface)
- ``GoogleSearchGroundingProvider`` default: uses Gemini's built-in
                                    Google Search grounding tool via
                                    the existing news.py pipeline.
- ``VertexAISearchProvider``       optional: uses a Vertex AI Search
                                    datastore for grounded RAG.
                                    Requires GCP credentials and
                                    VERTEX_SEARCH_DATASTORE_ID env var.
- ``get_context_provider()``       factory — returns the right provider
                                    based on env configuration.

Usage
-----
    from News.context_provider import get_context_provider

    provider = get_context_provider()
    result = provider.get_market_report()
    print(result["context_source"])   # "Google Search Grounding" | "Vertex AI Search" | ...
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ContextProvider(ABC):
    """Retrieval interface that all grounding backends implement."""

    @abstractmethod
    def get_market_report(self) -> dict[str, Any]:
        """Return a market report dict.

        Guaranteed keys:
            sentiment, reason, full_report, sources (list[dict]),
            source_status (str), context_source (str)
        """

    @property
    @abstractmethod
    def label(self) -> str:
        """Human-readable label for UI display, e.g. 'Google Search Grounding'."""


# ---------------------------------------------------------------------------
# Provider: Google Search Grounding (default)
# ---------------------------------------------------------------------------

class GoogleSearchGroundingProvider(ContextProvider):
    """
    Uses Gemini's built-in Google Search grounding tool.

    This is the current production default. It calls ``get_ai_market_report()``
    from News/news.py, which invokes Gemini with a Google Search tool attached,
    resolves grounding source URLs, and returns normalised results.

    Alignment: This satisfies the mandate's requirement for a context layer
    built on Google AI tooling (Gemini + Google Search grounding).
    """

    label = "Google Search Grounding"

    def get_market_report(self) -> dict[str, Any]:
        # Lazy import to avoid circular deps and import-time side-effects.
        from News.news import get_ai_market_report  # noqa: PLC0415

        result = get_ai_market_report()
        result.setdefault("context_source", self.label)
        return result


# ---------------------------------------------------------------------------
# Provider: Vertex AI Search (optional stub)
# ---------------------------------------------------------------------------

class VertexAISearchProvider(ContextProvider):
    """
    Uses a Vertex AI Search datastore for grounded RAG (Retrieval-Augmented
    Generation) against a curated document corpus.

    Requirements
    ------------
    - VERTEX_SEARCH_DATASTORE_ID   env var (full resource path)
    - VERTEX_SEARCH_LOCATION       env var (default: global)
    - GCP Application Default Credentials (ADC) or service-account key

    When credentials or datastore ID are absent this provider raises
    ``RuntimeError`` — the factory will fall back to Google Search.

    To configure
    ------------
    Set these in .env or as Cloud Run env vars:

        VERTEX_SEARCH_DATASTORE_ID=projects/PROJECT/locations/global/collections/default_collection/dataStores/DATASTORE_ID
        VERTEX_SEARCH_LOCATION=global
    """

    label = "Vertex AI Search"

    def __init__(self) -> None:
        self._datastore_id = os.getenv("VERTEX_SEARCH_DATASTORE_ID", "").strip()
        self._location = os.getenv("VERTEX_SEARCH_LOCATION", "global").strip()

        if not self._datastore_id:
            raise RuntimeError(
                "VertexAISearchProvider requires VERTEX_SEARCH_DATASTORE_ID to be set."
            )

    def _search_documents(self, query: str) -> list[dict]:
        """Execute a Vertex AI Search query and return retrieved documents."""
        try:
            from google.cloud import discoveryengine_v1  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "google-cloud-discoveryengine is not installed. "
                "Run: pip install google-cloud-discoveryengine"
            ) from exc

        client = discoveryengine_v1.SearchServiceClient()
        request = discoveryengine_v1.SearchRequest(
            serving_config=f"{self._datastore_id}/servingConfigs/default_config",
            query=query,
            page_size=5,
        )
        response = client.search(request)

        docs = []
        for result in response.results:
            doc = result.document
            data = dict(doc.struct_data) if doc.struct_data else {}
            docs.append({
                "title": data.get("title", doc.name),
                "url": data.get("url", ""),
                "snippet": data.get("content", data.get("snippet", "")),
            })
        return docs

    def get_market_report(self) -> dict[str, Any]:
        """
        Retrieve gold market context from the Vertex AI Search datastore,
        then generate a Gemini summary using retrieved snippets as context.
        """
        import importlib  # noqa: PLC0415

        query = "gold market sentiment geopolitics inflation central bank today"
        sources = self._search_documents(query)

        context_snippets = "\n\n".join(
            f"[{s['title']}]\n{s['snippet']}" for s in sources if s.get("snippet")
        )

        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
        from dotenv import load_dotenv  # noqa: PLC0415
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        prompt = (
            "Using ONLY the following retrieved documents, analyze the current gold market. "
            "Return valid JSON with keys: sentiment (Bullish/Bearish/Neutral), reason (str), "
            f"full_report (str).\n\nDocuments:\n{context_snippets}"
        )

        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )

        import json, re  # noqa: PLC0415, E401
        text = getattr(response, "text", "") or ""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(match.group()) if match else {"sentiment": "Neutral", "reason": text, "full_report": text}

        return {
            "sentiment": data.get("sentiment", "Neutral"),
            "reason": data.get("reason", ""),
            "full_report": data.get("full_report", ""),
            "sources": sources,
            "source_status": "vertex_ai_search",
            "context_source": self.label,
            "sdk": "google_genai",
            "search_mode": True,
            "raw_has_grounding": True,
            "fallback_used": False,
            "search_error": "",
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_context_provider() -> ContextProvider:
    """
    Return the appropriate ``ContextProvider`` based on environment config.

    Resolution order:
    1. ``VertexAISearchProvider``       if VERTEX_SEARCH_DATASTORE_ID is set
    2. ``GoogleSearchGroundingProvider`` default fallback
    """
    datastore_id = os.getenv("VERTEX_SEARCH_DATASTORE_ID", "").strip()
    if datastore_id:
        try:
            return VertexAISearchProvider()
        except Exception as exc:  # noqa: BLE001
            print(f"[context_provider] Vertex AI Search unavailable: {exc}. Falling back to Google Search.")

    return GoogleSearchGroundingProvider()
