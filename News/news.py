import os
import json
import re
import importlib
from urllib.parse import urlparse
from functools import lru_cache
from dotenv import load_dotenv
from datetime import datetime

try:
    genai = importlib.import_module("google.genai")
    types = importlib.import_module("google.genai.types")
except ImportError:
    genai = None
    types = None

try:
    legacy_genai = importlib.import_module("google.generativeai")
except ImportError:
    legacy_genai = None

try:
    requests = importlib.import_module("requests")
except ImportError:
    requests = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(ROOT_DIR, ".env")

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["Bullish", "Bearish", "Neutral"]
        },
        "reason": {"type": "string"},
        "full_report": {"type": "string"}
    },
    "required": ["sentiment", "reason", "full_report"]
}


def _get_google_api_key():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        return api_key.strip()
    # Force-load project root .env so key resolution does not depend on cwd.
    load_dotenv(dotenv_path=DOTENV_PATH, override=False)
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return api_key.strip() if isinstance(api_key, str) else None


def _read_field(value, *field_names, default=None):
    for field_name in field_names:
        if isinstance(value, dict) and field_name in value:
            return value.get(field_name)
        attr = getattr(value, field_name, None)
        if attr is not None:
            return attr
    return default


def _to_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return {}


def _extract_json_payload(text):
    json_match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return json.loads(text)


def _parse_response_payload(response):
    parsed = _read_field(response, "parsed")
    if parsed:
        parsed_dict = _to_dict(parsed)
        if parsed_dict:
            return parsed_dict
        if isinstance(parsed, dict):
            return parsed
    response_text = _read_field(response, "text", default="") or ""
    return _extract_json_payload(response_text)


def _normalize_report_content(data):
    report_data = data if isinstance(data, dict) else {}
    sentiment = report_data.get("sentiment", "Neutral")
    reason = report_data.get("reason", "")
    full_report = report_data.get("full_report", "")

    if sentiment not in {"Bullish", "Bearish", "Neutral"}:
        sentiment = "Neutral"

    if isinstance(full_report, dict):
        full_report = "\n\n".join(
            str(section_text) for section_text in full_report.values() if section_text
        )
    if isinstance(full_report, list):
        full_report = "\n\n".join(str(part) for part in full_report)
    if not isinstance(full_report, str):
        full_report = str(full_report)
    full_report = re.sub(r"\[.*?\]", "", full_report)

    if not isinstance(reason, str):
        reason = str(reason)

    return {
        "sentiment": sentiment,
        "reason": reason,
        "full_report": full_report,
    }


def _normalize_url(url):
    if not isinstance(url, str):
        return ""
    cleaned = url.strip().rstrip(".,)")
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return cleaned
    return ""


def _normalize_domain(domain):
    if not isinstance(domain, str):
        return ""
    cleaned = domain.strip().lower()
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    return cleaned


def _is_provider_domain(domain):
    normalized = _normalize_domain(domain)
    return normalized in {
        "vertexaisearch.cloud.google.com",
        "grounding-api-redirect.googleapis.com",
    }


@lru_cache(maxsize=256)
def _resolve_provider_redirect(url):
    if not requests:
        return ""
    normalized_url = _normalize_url(url)
    if not normalized_url:
        return ""

    try:
        response = requests.get(normalized_url, timeout=8, allow_redirects=True)
        resolved = _normalize_url(getattr(response, "url", ""))
        if resolved and not _is_provider_domain(urlparse(resolved).netloc):
            return resolved
    except Exception:
        return ""
    return ""


def _looks_like_domain(value):
    if not isinstance(value, str):
        return False
    return bool(re.match(r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$", value.strip().lower()))


def _humanize_domain(domain):
    normalized = _normalize_domain(domain)
    if not normalized:
        return "Untitled source"
    base = normalized.split(".")[0]
    base = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
    base = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", base)
    base = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", base)
    return " ".join(part.capitalize() for part in re.split(r"[-_\s]+", base) if part)


def _build_source_label(raw_title, resolved_domain):
    if isinstance(raw_title, str):
        title = raw_title.strip()
        if title:
            if _looks_like_domain(title):
                return _humanize_domain(title)
            parsed = _normalize_url(title)
            if parsed:
                return _humanize_domain(urlparse(parsed).netloc)
            return title
    return _humanize_domain(resolved_domain)


def _extract_candidate_urls(source_obj):
    urls = []
    for key in [
        "uri",
        "url",
        "link",
        "source_uri",
        "sourceUrl",
        "resolved_url",
        "resolvedUrl",
    ]:
        parsed = _normalize_url(_read_field(source_obj, key))
        if parsed:
            urls.append(parsed)
    return urls


def _source_from_candidate(source_like):
    source_obj = _to_dict(source_like)
    candidate_urls = _extract_candidate_urls(source_obj)
    if not candidate_urls:
        return None

    provider_urls = [u for u in candidate_urls if _is_provider_domain(urlparse(u).netloc)]
    real_urls = [u for u in candidate_urls if not _is_provider_domain(urlparse(u).netloc)]

    resolved_url = ""
    if not real_urls and provider_urls:
        resolved_url = _resolve_provider_redirect(provider_urls[0])

    preferred_url = real_urls[0] if real_urls else (resolved_url or candidate_urls[0])
    provider_url = provider_urls[0] if provider_urls else ""

    title = _read_field(source_obj, "title", "name", "display_name", default="")

    title_domain = ""
    if isinstance(title, str):
        title_clean = title.strip().lower()
        if _looks_like_domain(title_clean):
            title_domain = title_clean
        else:
            parsed_title_url = _normalize_url(title_clean)
            if parsed_title_url:
                title_domain = _normalize_domain(urlparse(parsed_title_url).netloc)

    preferred_domain = _normalize_domain(urlparse(preferred_url).netloc)
    display_domain = preferred_domain
    if _is_provider_domain(preferred_domain) and title_domain:
        display_domain = title_domain

    label = _build_source_label(title, display_domain)

    return {
        "title": label,
        "url": preferred_url,
        "domain": display_domain,
        "provider_url": provider_url,
        "provider_domain": _normalize_domain(urlparse(provider_url).netloc) if provider_url else "",
    }


def _extract_sources_from_response(response):
    response_dict = _to_dict(response)
    candidates = _read_field(response_dict, "candidates", default=[]) or []
    sources = []
    seen_urls = set()
    seen_identity = set()
    raw_has_grounding = False

    for candidate in candidates:
        candidate_dict = _to_dict(candidate)
        grounding_metadata = _read_field(
            candidate_dict,
            "grounding_metadata",
            "groundingMetadata",
            default={},
        )
        grounding_dict = _to_dict(grounding_metadata)
        if grounding_dict:
            raw_has_grounding = True

        grounding_chunks = _read_field(
            grounding_dict,
            "grounding_chunks",
            "groundingChunks",
            default=[],
        ) or []

        if grounding_chunks:
            raw_has_grounding = True

        for chunk in grounding_chunks:
            chunk_dict = _to_dict(chunk)
            web_data = _read_field(chunk_dict, "web")
            retrieved_context = _read_field(chunk_dict, "retrieved_context", "retrievedContext")
            source_data = _read_field(chunk_dict, "source")

            for candidate_source in [web_data, retrieved_context, source_data, chunk_dict]:
                source = _source_from_candidate(candidate_source)
                if not source:
                    continue
                url = source["url"]
                identity_key = f"{_normalize_domain(source.get('domain', ''))}|{source.get('title', '').strip().lower()}"

                if url in seen_urls or identity_key in seen_identity:
                    continue
                seen_urls.add(url)
                seen_identity.add(identity_key)
                sources.append(source)

    return sources, raw_has_grounding


def _resolve_source_status(search_mode, raw_has_grounding, sources):
    if search_mode and raw_has_grounding and sources:
        return "ok"
    if search_mode and raw_has_grounding and not sources:
        return "grounding_without_sources"
    if search_mode and not raw_has_grounding:
        return "no_grounding"
    if not search_mode:
        return "fallback_no_search"
    return "unavailable"


def _build_result(data, sources, sdk, search_mode, raw_has_grounding, fallback_used, search_error=""):
    normalized_data = _normalize_report_content(data)
    source_status = _resolve_source_status(search_mode, raw_has_grounding, sources)
    normalized_data["sources"] = sources
    normalized_data["source_status"] = source_status
    normalized_data["context_source"] = (
        "Google Search Grounding" if search_mode and raw_has_grounding
        else "Python Fallback"
    )
    normalized_data["sdk"] = sdk
    normalized_data["search_mode"] = bool(search_mode)
    normalized_data["raw_has_grounding"] = bool(raw_has_grounding)
    normalized_data["fallback_used"] = bool(fallback_used)
    normalized_data["search_error"] = search_error if isinstance(search_error, str) else str(search_error)
    return normalized_data


def _generate_report_with_new_sdk(today_date):
    api_key = _get_google_api_key()
    prompt = (
        f"Date: {today_date}. Analyze recent gold market news with focus on geopolitics, "
        "inflation, central banks, and risk sentiment. Keep the report concise, factual, "
        "and suitable for a trading dashboard. Return valid JSON only with keys: "
        "sentiment, reason, full_report."
    )

    with genai.Client(api_key=api_key) as client:
        search_tool = types.Tool(google_search=types.GoogleSearch())

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    tools=[search_tool],
                ),
            )
            data = _parse_response_payload(response)
            sources, raw_has_grounding = _extract_sources_from_response(response)
            return _build_result(
                data=data,
                sources=sources,
                sdk="google_genai",
                search_mode=True,
                raw_has_grounding=raw_has_grounding,
                fallback_used=False,
            )
        except Exception as e:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=REPORT_SCHEMA,
                ),
            )
            data = _parse_response_payload(response)
            sources, raw_has_grounding = _extract_sources_from_response(response)
            return _build_result(
                data=data,
                sources=sources,
                sdk="google_genai",
                search_mode=False,
                raw_has_grounding=raw_has_grounding,
                fallback_used=True,
                search_error=str(e),
            )


def _generate_report_with_legacy_sdk(today_date):
    api_key = _get_google_api_key()
    legacy_genai.configure(api_key=api_key)
    prompt = (
        f"Date: {today_date}. Analyze recent gold market news with focus on geopolitics, "
        "inflation, central banks, and risk sentiment. Return valid JSON only with keys: "
        "sentiment, reason, full_report."
    )

    try:
        model = legacy_genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            tools=[{"google_search": {}}],
        )
        response = model.generate_content(prompt)
        data = _extract_json_payload(response.text)
        sources, raw_has_grounding = _extract_sources_from_response(response)
        return _build_result(
            data=data,
            sources=sources,
            sdk="google_generativeai",
            search_mode=True,
            raw_has_grounding=raw_has_grounding,
            fallback_used=False,
        )
    except Exception as e:
        model = legacy_genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt)
        data = _extract_json_payload(response.text)
        sources, raw_has_grounding = _extract_sources_from_response(response)
        return _build_result(
            data=data,
            sources=sources,
            sdk="google_generativeai",
            search_mode=False,
            raw_has_grounding=raw_has_grounding,
            fallback_used=True,
            search_error=str(e),
        )


def _empty_result(reason):
    return {
        "sentiment": "Neutral",
        "reason": reason,
        "full_report": "",
        "sources": [],
        "source_status": "unavailable",
        "context_source": "Python Fallback",
        "sdk": "none",
        "search_mode": False,
        "raw_has_grounding": False,
        "fallback_used": False,
        "search_error": "",
    }


def _friendly_error_result(error):
    raw = str(error) if error else "Unknown error"
    normalized = raw.lower()

    reason = "AI service temporarily unavailable"
    full_report = "The system switched to a safe fallback mode. Please retry shortly."

    if "resource_exhausted" in normalized or "quota" in normalized or "429" in normalized:
        reason = "Gemini API quota reached"
        full_report = (
            "Live AI analysis is temporarily paused because API usage limit was reached. "
            "Please retry after the quota window resets or use a key with available quota."
        )
    elif "api key" in normalized or "permission" in normalized or "unauth" in normalized or "401" in normalized or "403" in normalized:
        reason = "Gemini API authentication issue"
        full_report = "Please verify GOOGLE_API_KEY and API permissions, then retry."
    elif "timeout" in normalized or "connection" in normalized or "dns" in normalized or "unreachable" in normalized:
        reason = "Network issue while contacting Gemini API"
        full_report = "Please check network connectivity and retry."
    elif "invalid_argument" in normalized or "400" in normalized:
        reason = "Gemini request format issue"
        full_report = "The request was rejected by the API format validator. The system used fallback mode."

    result = _empty_result(reason)
    result["full_report"] = full_report
    result["search_error"] = raw
    result["fallback_used"] = True
    return result


def get_ai_market_report():
    today_date = datetime.now().strftime("%B %d, %Y")
    api_key = _get_google_api_key()

    try:
        if not api_key:
            result = _empty_result("API Key missing")
            result["full_report"] = "Set GOOGLE_API_KEY in .env to enable live AI market intelligence."
            return result

        if genai is not None and types is not None:
            return _generate_report_with_new_sdk(today_date)
        if legacy_genai is not None:
            return _generate_report_with_legacy_sdk(today_date)

        result = _empty_result("No Gemini SDK installed")
        result["full_report"] = "Install google-genai to enable live AI market intelligence."
        return result
    except Exception as e:
        return _friendly_error_result(e)


def get_market_sentiment(report_data=None):
    if report_data is None:
        report_data = get_ai_market_report()
    sentiment = report_data.get("sentiment", "Neutral")
    score = 0.8 if sentiment == "Bullish" else -0.8 if sentiment == "Bearish" else 0.0
    return sentiment, score