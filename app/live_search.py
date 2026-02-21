"""
Live search against external Epstein file archives.

Queries the public API at epsteininvestigation.org (207K+ documents, 23K+ entities,
3K+ flight records) as a fallback when the local database has no results, and
as a way to discover and ingest documents on the fly.

Also constructs direct links to jmail.world and justice.gov/epstein for
documents we can't fetch via API.
"""

import logging
from urllib.parse import quote_plus

import requests

from app.nlp import resolve_name, clean_query, generate_query_variants, score_relevance
from config import Config

logger = logging.getLogger(__name__)

API_BASE = Config.ARCHIVE_API_URL
JMAIL_BASE = Config.JMAIL_BASE_URL
DOJ_BASE = Config.DOJ_BASE_URL

_session = requests.Session()
_session.headers.update({
    "User-Agent": "EpsteinFilesSearch/1.0 (research tool)",
    "Accept": "application/json",
})

REQUEST_TIMEOUT = 15


def search_external(query: str, page: int = 1, limit: int = 25) -> dict:
    """
    Search the epsteininvestigation.org full-text search API.

    Cascade:
    1. Clean the query (strip quotes, whitespace, punctuation).
    2. Try the cleaned query verbatim.
    3. Try a known-name resolution ("Lesly Goff" → "Lesley Groff").
    4. If still no results, try generic fuzzy variants of each word
       ("Nick Leee" → "Nick Lees", "Nick Lee", etc.).
    5. Also try the entity search API — if a fuzzy entity match is
       found, search documents for that entity's real name.
    """
    query = clean_query(query)
    resolved = resolve_name(query)

    # Build ordered list of search terms to try
    search_terms = _unique_ordered([query, resolved])

    all_results, total = _run_search_terms(search_terms, page, limit)

    # If first round found nothing, try generic fuzzy variants
    if not all_results:
        fuzzy_terms = generate_query_variants(query, max_total=6)
        # Also try variants of the resolved name
        if resolved != query:
            fuzzy_terms += generate_query_variants(resolved, max_total=4)
        # Deduplicate, skip already-tried terms
        fuzzy_terms = _unique_ordered(fuzzy_terms, exclude=set(search_terms))
        if fuzzy_terms:
            logger.info(f"Trying fuzzy variants for '{query}': {fuzzy_terms[:6]}")
            all_results, total = _run_search_terms(fuzzy_terms, page, limit)

    # Last resort: entity-based lookup
    if not all_results:
        entity_name = _entity_fuzzy_lookup(query)
        if entity_name and entity_name.lower() != query.lower():
            logger.info(f"Entity lookup resolved '{query}' → '{entity_name}'")
            resolved = entity_name
            all_results, total = _run_search_terms([entity_name], page, limit)

    seen_ids = set()
    deduped = []
    for doc in all_results:
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            deduped.append(_normalise_external_doc(doc))

    actual_resolved = resolved if resolved.lower() != query.lower() else None
    return {
        "items": deduped[:limit],
        "total": total,
        "page": page,
        "pages": (total + limit - 1) // limit if total else 0,
        "query": query,
        "resolved_name": actual_resolved,
        "source": "epsteininvestigation.org",
        "external": True,
    }


def _run_search_terms(terms: list[str], page: int, limit: int) -> tuple[list, int]:
    """Run the search API for each term; return combined (results, max_total)."""
    all_results = []
    total = 0
    for term in terms:
        try:
            resp = _session.get(
                f"{API_BASE}/search",
                params={"q": term, "page": page, "limit": limit},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                for doc in data.get("data", []):
                    doc["_search_term"] = term
                all_results.extend(data.get("data", []))
                total = max(total, data.get("total", 0))
                if total > 0:
                    break  # found results, stop trying more variants
        except Exception as e:
            logger.warning(f"External search failed for '{term}': {e}")
    return all_results, total


def _entity_fuzzy_lookup(query: str) -> str | None:
    """
    Search the external entities API for a close match.
    Lets the 23K+ entity database act as a fuzzy dictionary.
    """
    words = query.split()
    if not words:
        return None

    # Try each word against the entity API
    for word in words:
        if len(word) < 3:
            continue
        try:
            resp = _session.get(
                f"{API_BASE}/entities",
                params={"q": word, "type": "person", "limit": 10},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                entities = resp.json().get("data", [])
                names = [e["name"] for e in entities if e.get("name")]
                if names:
                    from rapidfuzz import process as rfprocess, fuzz as rffuzz
                    match = rfprocess.extractOne(
                        query, names, scorer=rffuzz.WRatio
                    )
                    if match and match[1] >= 65:
                        return match[0]
        except Exception:
            pass
    return None


def _unique_ordered(items: list[str], exclude: set | None = None) -> list[str]:
    """Deduplicate a list while preserving order."""
    exclude = {x.lower() for x in (exclude or set())}
    seen = set()
    result = []
    for item in items:
        low = item.lower()
        if low not in seen and low not in exclude:
            seen.add(low)
            result.append(item)
    return result


def search_external_documents(query: str, doc_type: str = None,
                               source: str = None, page: int = 1,
                               limit: int = 25) -> dict:
    """Search the documents endpoint with type/source filters."""
    query = clean_query(query)
    params = {"q": query, "page": page, "limit": limit}
    if doc_type:
        params["type"] = doc_type
    if source:
        params["source"] = source

    try:
        resp = _session.get(
            f"{API_BASE}/documents",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            items = [_normalise_external_doc(d) for d in data.get("data", [])]
            return {
                "items": items,
                "total": data.get("total", 0),
                "page": data.get("page", page),
                "pages": (data.get("total", 0) + limit - 1) // limit,
                "query": query,
                "source": "epsteininvestigation.org",
                "external": True,
            }
    except Exception as e:
        logger.warning(f"External document search failed: {e}")

    return _empty_result(query)


def search_external_entities(query: str, entity_type: str = None,
                              page: int = 1, limit: int = 25) -> dict:
    """Search entities (people, orgs, locations) from the external API."""
    query = clean_query(query)
    resolved = resolve_name(query)

    params = {"page": page, "limit": limit}
    if entity_type:
        params["type"] = entity_type

    all_entities = []
    for term in {query, resolved}:
        params["q"] = term
        try:
            resp = _session.get(
                f"{API_BASE}/entities",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                all_entities.extend(data.get("data", []))
        except Exception as e:
            logger.warning(f"External entity search failed for '{term}': {e}")

    seen = set()
    deduped = []
    for ent in all_entities:
        if ent["id"] not in seen:
            seen.add(ent["id"])
            deduped.append(ent)

    return {
        "items": deduped[:limit],
        "total": len(deduped),
        "page": page,
        "query": query,
        "resolved_name": resolved if resolved != query else None,
        "source": "epsteininvestigation.org",
        "external": True,
    }


def search_external_flights(passenger: str = None, airport: str = None,
                             date_from: str = None, date_to: str = None,
                             page: int = 1, limit: int = 25) -> dict:
    """Search flight logs from the external API."""
    params = {"page": page, "limit": limit}
    if passenger:
        params["passenger"] = passenger
    if airport:
        params["airport"] = airport
    if date_from:
        params["date_from"] = date_from
    if date_to:
        params["date_to"] = date_to

    try:
        resp = _session.get(
            f"{API_BASE}/flights",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            return {
                "items": data.get("data", []),
                "total": data.get("total", 0),
                "page": data.get("page", page),
                "pages": (data.get("total", 0) + limit - 1) // limit,
                "source": "epsteininvestigation.org",
                "external": True,
            }
    except Exception as e:
        logger.warning(f"External flight search failed: {e}")

    return _empty_result("")


def get_external_document(slug: str) -> dict | None:
    """Fetch a single document by slug from the external API."""
    try:
        resp = _session.get(
            f"{API_BASE}/documents",
            params={"q": slug, "limit": 1},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                return _normalise_external_doc(data["data"][0])
    except Exception as e:
        logger.warning(f"External document fetch failed for '{slug}': {e}")
    return None


def build_jmail_url(file_id: str) -> str:
    """Build a jmail.world URL for a given file ID."""
    slug = file_id.lower().replace("_", "-")
    return f"{JMAIL_BASE}/thread/{slug}"


def build_doj_url(file_id: str, dataset: str = None) -> str:
    """Build a justice.gov URL for a given file ID."""
    if dataset:
        ds_num = dataset.replace("doj_dataset_", "")
        return f"{DOJ_BASE}/files/DataSet%20{ds_num}/{file_id}.pdf"
    return f"{DOJ_BASE}"


def _normalise_external_doc(doc: dict) -> dict:
    """Normalise an external API document to match our internal format."""
    file_id = doc.get("slug", doc.get("id", "unknown")).upper()
    excerpt = doc.get("excerpt", "")

    relevance_score, relevance_cats = score_relevance(excerpt) if excerpt else (0, [])

    source_str = doc.get("source", "")
    doj_url = None
    if "doj_dataset" in source_str:
        doj_url = build_doj_url(
            doc.get("slug", "").upper(),
            source_str,
        )

    return {
        "id": doc.get("id"),
        "file_id": file_id,
        "title": doc.get("title", file_id),
        "body": excerpt,
        "doc_type": doc.get("document_type", "document"),
        "source": source_str,
        "source_url": doc.get("file_url"),
        "original_url": doc.get("source_url") or doj_url,
        "date": doc.get("document_date"),
        "page_count": doc.get("page_count"),
        "relevance_score": relevance_score,
        "relevance_categories": relevance_cats,
        "jmail_url": build_jmail_url(doc.get("slug", "")),
        "entities": [],
        "categories": [],
        "external": True,
    }


def _empty_result(query: str) -> dict:
    return {
        "items": [],
        "total": 0,
        "page": 1,
        "pages": 0,
        "query": query,
        "source": "epsteininvestigation.org",
        "external": True,
    }
