"""
AI document analysis via local LLM (Ollama).

Runs on:
  - Mac M4 Pro 64GB: llama3.1:8b or mistral:7b
  - Data centre:     llama3.1:70b or larger

The LLM analyses each document for:
  1. Relevance to criminal investigation (0–1 score)
  2. Category assignment (from known + discovered categories)
  3. New category discovery
  4. Entity extraction with roles/relationships
  5. Brief plain-language summary of why the document matters
"""

import json
import logging
import re

import requests
from config import Config

logger = logging.getLogger(__name__)

_cached_model: str | None = None


def _ollama_url() -> str:
    return Config.OLLAMA_URL


def _model_score(name: str) -> tuple[int, int]:
    """Score a model by family recency then size. Higher = preferred."""
    name_l = name.lower()
    # Newer instruction-tuned families first — they produce better JSON
    family_order = [
        "llama3.2", "llama3.1", "llama3",
        "qwen2.5", "qwen2",
        "gemma2",
        "mistral",
        "phi3",
        "gemma", "phi",
        "llama2", "llama",
    ]
    family_score = 0
    for i, fam in enumerate(family_order):
        if fam in name_l:
            family_score = len(family_order) - i
            break

    size_scores = {
        "70b": 70, "32b": 32, "22b": 22, "20b": 20,
        "13b": 13, "8b": 8, "7b": 7, "3b": 3, "2b": 2,
    }
    size_score = 0
    for size_str, score in size_scores.items():
        if size_str in name_l:
            size_score = score
            break

    return (family_score, size_score)


def _get_model() -> str:
    """Get the configured model, or auto-detect the best available one."""
    global _cached_model
    if Config.OLLAMA_MODEL:
        return Config.OLLAMA_MODEL
    if _cached_model:
        return _cached_model
    try:
        resp = requests.get(f"{_ollama_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if models:
                _cached_model = max(models, key=_model_score)
                logger.info(f"Auto-selected model: {_cached_model}")
                return _cached_model
    except Exception:
        pass
    return "llama3.1:8b"


def _doc_context_chars(model: str) -> int:
    """Return how many document chars to send, scaled to model capacity."""
    name_l = model.lower()
    if any(s in name_l for s in ["70b", "32b", "22b"]):
        return 8000
    if any(s in name_l for s in ["20b", "13b"]):
        return 5000
    return 3000


# The system prompt encodes what "relevant" means for the Epstein files.
SYSTEM_PROMPT = """\
You are an investigative analyst examining documents from the Jeffrey Epstein \
criminal files, released by the U.S. Department of Justice, FBI, and \
congressional committees.

Your purpose is to help the public understand these documents. You have a \
clear understanding of what IS and IS NOT relevant:

RELEVANT — matters of public interest:
- Sex trafficking and sexual abuse, especially of minors
- Child exploitation. There is no such thing as "child prostitution" — \
a child cannot consent. Any document referring to minors in sexual contexts \
describes child rape, not prostitution.
- Blackmail, coercion, and kompromat
- Connections to intelligence services (CIA, Mossad, MI6, FBI, etc.)
- Financial crimes: money laundering, fraud, shell companies, suspicious \
transfers, tax evasion
- Undermining nation states or democratic institutions
- Corruption, cover-ups, obstruction of justice
- The network: how powerful people enabled, participated in, or covered up \
these crimes
- Attitudes toward victims and the less fortunate
- Any document that provides CONTEXT for the above, even if it seems \
innocuous on its own (e.g. travel arrangements that place someone at a \
location where abuse occurred)

NOT RELEVANT — mundane content:
- Music newsletters, album recommendations, audiophile content
- Generic marketing emails, spam, promotions
- Restaurant reservations or shopping (unless they reveal patterns)
- Normal business correspondence with no connection to criminal activity

You must be unflinching. Do not euphemise. Call trafficking "trafficking", \
call child rape "child rape". The legal system's failures are part of the \
story.\
"""

ANALYSIS_PROMPT = """\
Analyse the following document from the Epstein files.

Return ONLY valid JSON with these fields:
{{
  "relevance_score": <float 0.0 to 1.0>,
  "summary": "<1-3 sentences: why this document matters or doesn't>",
  "categories": ["<list of categories this document belongs to>"],
  "new_categories": ["<categories NOT in the known list that you've discovered>"],
  "entities": [
    {{"name": "<person/org/place>", "type": "<PERSON|ORG|LOCATION>", "role": "<brief role>"}}
  ],
  "connections": "<any connections to other known events, people, or patterns>",
  "is_relevant": <true or false>
}}

Known categories (use these when they fit):
{known_categories}

Document ID: {file_id}
Source: {source}
Date: {date}

--- DOCUMENT TEXT ---
{text}
--- END ---

Return ONLY the JSON object, no markdown fences, no explanation.\
"""

CATEGORY_DISCOVERY_PROMPT = """\
You have analysed many documents from the Epstein files. Based on the \
following batch of document summaries, identify any NEW categories or \
themes that emerge — topics that don't fit neatly into the existing \
categories but appear in multiple documents.

Existing categories:
{known_categories}

Recent document summaries:
{summaries}

Return ONLY valid JSON:
{{
  "new_categories": [
    {{
      "name": "<category name>",
      "slug": "<url-safe-slug>",
      "description": "<what this category covers>",
      "evidence": "<which documents suggest this category>"
    }}
  ]
}}

Only suggest categories that represent genuine patterns across multiple \
documents. Do not suggest one-off topics. Return ONLY the JSON.\
"""


def check_ollama() -> dict:
    """Check if Ollama is running and which models are available."""
    model = _get_model()
    try:
        resp = requests.get(f"{_ollama_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {
                "status": "running",
                "url": _ollama_url(),
                "models": models,
                "configured_model": model,
                "model_available": any(
                    model.split(":")[0] in m for m in models
                ),
            }
    except requests.ConnectionError:
        pass
    except Exception as e:
        logger.warning(f"Ollama check error: {e}")

    return {
        "status": "not_running",
        "url": _ollama_url(),
        "models": [],
        "configured_model": model,
        "model_available": False,
    }


def _query_ollama(prompt: str, system: str = SYSTEM_PROMPT, temperature: float = 0.1) -> str | None:
    """Send a prompt to Ollama via the chat API and return the response text."""
    model = _get_model()
    try:
        resp = requests.post(
            f"{_ollama_url()}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
                    "num_predict": 4096,
                },
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "")
        else:
            logger.warning(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
    except requests.ConnectionError:
        logger.error("Cannot connect to Ollama. Is it running? Install: https://ollama.ai")
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
    return None


def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from an LLM response, handling markdown fences and junk."""
    if not text:
        return None

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object in the response
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON from LLM response: {text[:200]}")
    return None


def analyse_document(text: str, file_id: str = "", source: str = "",
                     date: str = "", known_categories: list[str] | None = None) -> dict | None:
    """
    Run full AI analysis on a document.

    Returns dict with: relevance_score, summary, categories, new_categories,
    entities, connections, is_relevant.
    """
    if not text or len(text.strip()) < 20:
        return None

    cats = known_categories or _default_categories()
    ctx_chars = _doc_context_chars(_get_model())
    prompt = ANALYSIS_PROMPT.format(
        file_id=file_id,
        source=source,
        date=date or "unknown",
        text=text[:ctx_chars],
        known_categories=", ".join(cats),
    )

    raw = _query_ollama(prompt)
    result = _parse_json_response(raw)

    if result is None:
        logger.debug(f"JSON parse failed for {file_id}, retrying...")
        raw = _query_ollama(prompt, temperature=0.05)
        result = _parse_json_response(raw)

    if result:
        # Normalise and validate
        result["relevance_score"] = max(0.0, min(1.0, float(result.get("relevance_score", 0))))
        result["is_relevant"] = result.get("is_relevant", result["relevance_score"] > 0.2)
        result["categories"] = result.get("categories", [])
        result["new_categories"] = result.get("new_categories", [])
        result["entities"] = result.get("entities", [])
        result["summary"] = result.get("summary", "")
        result["connections"] = result.get("connections", "")

    return result


def discover_categories(summaries: list[dict],
                        known_categories: list[str] | None = None) -> list[dict]:
    """
    Given a batch of recent document summaries, ask the LLM to identify
    new categories/themes that emerge across multiple documents.
    """
    if not summaries:
        return []

    cats = known_categories or _default_categories()

    summary_text = "\n".join(
        f"- [{s.get('file_id', '?')}] {s.get('summary', 'No summary')}"
        for s in summaries[:30]
    )

    prompt = CATEGORY_DISCOVERY_PROMPT.format(
        known_categories=", ".join(cats),
        summaries=summary_text,
    )

    raw = _query_ollama(prompt, temperature=0.3)
    result = _parse_json_response(raw)

    if result is None:
        logger.debug("Category discovery JSON parse failed, retrying...")
        raw = _query_ollama(prompt, temperature=0.2)
        result = _parse_json_response(raw)

    if result and "new_categories" in result:
        return result["new_categories"]
    return []


def _default_categories() -> list[str]:
    return [
        "Trafficking",
        "Sexual Abuse",
        "Child Exploitation",
        "Blackmail & Coercion",
        "Intelligence Services",
        "Financial Crime",
        "Corruption & Cover-up",
        "Legal Proceedings",
        "Flight Logs",
        "Communications",
        "Photographs",
        "Associates & Network",
        "Properties",
        "Victims",
    ]
