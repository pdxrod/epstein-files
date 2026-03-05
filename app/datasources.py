"""
Data source ingestion for external structured datasets:
  - Hugging Face: tensonaut/EPSTEIN_FILES_20K (25,800 OCR'd documents)
  - epsteininvestigation.org CSVs: entities, relationships, flights
  - DocETL-style processing of bulk text through local LLM

These complement the live API search by providing bulk data for AI analysis.
"""

import csv
import hashlib
import io
import json
import logging
import os
import re
from datetime import datetime

import requests
from config import Config

logger = logging.getLogger(__name__)

_ARCHIVE_BASE = Config.ARCHIVE_API_URL.replace("/api/v1", "")

ARCHIVE_CSV_URLS = {
    "entities": f"{_ARCHIVE_BASE}/api/download/entities",
    "flights": f"{_ARCHIVE_BASE}/api/download/flights",
    "relationships": f"{_ARCHIVE_BASE}/api/download/relationships",
    "emails": f"{_ARCHIVE_BASE}/api/download/emails",
}


def import_archive_entities(db):
    """Import entity data from epsteininvestigation.org CSV."""
    from app.models import Entity

    logger.info("Downloading entities CSV...")
    resp = requests.get(ARCHIVE_CSV_URLS["entities"], timeout=30)
    resp.raise_for_status()

    # Pre-load all existing entities to avoid N+1 queries
    existing = {(e.name, e.entity_type): e for e in Entity.query.all()}

    reader = csv.DictReader(io.StringIO(resp.text))
    count = 0
    for row in reader:
        name = row.get("name", "").strip()
        if not name:
            continue

        etype = _normalise_entity_type(row.get("entity_type", "PERSON"))
        entity = existing.get((name, etype))
        if entity:
            entity.mention_count = max(
                entity.mention_count or 0,
                int(row.get("document_count", 0) or 0),
            )
            if row.get("role_description"):
                entity.description = row["role_description"]
        else:
            entity = Entity(
                name=name,
                canonical_name=name,
                entity_type=etype,
                description=row.get("role_description", ""),
                mention_count=int(row.get("document_count", 0) or 0),
            )
            db.session.add(entity)
            existing[(name, etype)] = entity
        count += 1

    db.session.commit()
    logger.info(f"Imported/updated {count} entities from archive CSV")
    return count


def import_archive_relationships(db):
    """Import entity relationship graph from epsteininvestigation.org CSV."""
    from app.models import Entity, EntityRelationship

    logger.info("Downloading relationships CSV...")
    resp = requests.get(ARCHIVE_CSV_URLS["relationships"], timeout=30)
    resp.raise_for_status()

    # Pre-load caches to avoid N+1 queries
    entity_cache = {e.name: e for e in Entity.query.all()}
    existing_rels = {
        (r.entity_a_id, r.entity_b_id, r.relationship_type)
        for r in db.session.query(
            EntityRelationship.entity_a_id,
            EntityRelationship.entity_b_id,
            EntityRelationship.relationship_type,
        ).all()
    }

    reader = csv.DictReader(io.StringIO(resp.text))
    count = 0
    for row in reader:
        name_a = row.get("entity_a", "").strip()
        name_b = row.get("entity_b", "").strip()
        if not name_a or not name_b:
            continue

        entity_a = _get_or_create_entity(db, name_a, cache=entity_cache)
        entity_b = _get_or_create_entity(db, name_b, cache=entity_cache)

        # Flush only when new entities need IDs assigned
        if entity_a.id is None or entity_b.id is None:
            db.session.flush()

        rel_type = row.get("relationship_type", "associated").strip()
        strength = float(row.get("strength", 0) or 0)

        key = (entity_a.id, entity_b.id, rel_type)
        if key not in existing_rels:
            rel = EntityRelationship(
                entity_a_id=entity_a.id,
                entity_b_id=entity_b.id,
                relationship_type=rel_type,
                strength=strength,
                source="epsteininvestigation.org",
            )
            db.session.add(rel)
            existing_rels.add(key)
            count += 1

    db.session.commit()
    logger.info(f"Imported {count} entity relationships")
    return count


def import_archive_flights(db):
    """Import flight log data from epsteininvestigation.org CSV."""
    from app.models import FlightRecord

    logger.info("Downloading flights CSV...")
    resp = requests.get(ARCHIVE_CSV_URLS["flights"], timeout=30)
    resp.raise_for_status()

    # Pre-load existing flight keys to avoid duplicates
    existing_keys = {
        (date_str, tail, dep, arr)
        for date_str, tail, dep, arr in db.session.query(
            FlightRecord.flight_date_str,
            FlightRecord.aircraft_tail,
            FlightRecord.departure_code,
            FlightRecord.arrival_code,
        ).all()
    }

    reader = csv.DictReader(io.StringIO(resp.text))
    count = 0
    for row in reader:
        date_str = row.get("flight_date", "").strip()
        tail = row.get("aircraft_tail_number", "")
        dep = row.get("departure_airport_code", "")
        arr = row.get("arrival_airport_code", "")

        key = (date_str, tail, dep, arr)
        if key in existing_keys:
            continue

        flight_date = None
        if date_str:
            try:
                flight_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        record = FlightRecord(
            flight_date=flight_date,
            flight_date_str=date_str,
            aircraft_tail=tail,
            pilot=row.get("pilot_name", ""),
            departure_code=dep,
            departure_airport=row.get("departure_airport", ""),
            arrival_code=arr,
            arrival_airport=row.get("arrival_airport", ""),
            passengers=row.get("passenger_names", ""),
            source="epsteininvestigation.org",
        )
        db.session.add(record)
        existing_keys.add(key)
        count += 1

    db.session.commit()
    logger.info(f"Imported {count} flight records")
    return count


def import_huggingface_dataset(db, batch_size=100, max_docs=None):
    """
    Download and import the tensonaut/EPSTEIN_FILES_20K dataset from Hugging Face.
    25,800 OCR'd documents from the House Oversight Committee release.

    Tries multiple download methods:
      1. Streaming CSV download (avoids loading full file into memory)
      2. Parquet via HF datasets library
      3. Bulk fetch from epsteininvestigation.org API as last resort
    """
    HF_URLS = [
        "https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K/resolve/main/EPS_FILES_20K_NOV2025.csv?download=true",
        "https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K/resolve/main/EPS_FILES_20K_NOV2025.csv",
    ]

    # Method 1: Streaming CSV download
    for url in HF_URLS:
        logger.info(f"Trying streaming download: {url[:80]}...")
        try:
            resp = requests.get(
                url, timeout=300, stream=True,
                headers={"User-Agent": "epstein-files-search/1.0"},
            )
            if resp.status_code == 200:
                lines = (
                    line.decode("utf-8", errors="replace")
                    for line in resp.iter_lines()
                )
                reader = csv.DictReader(lines)
                fieldnames = reader.fieldnames or []
                if "filename" in fieldnames and "text" in fieldnames:
                    count = _ingest_csv_rows(db, reader, batch_size, max_docs)
                    if count > 0:
                        return count
                    logger.warning("CSV parsed but 0 docs imported")
                else:
                    logger.warning(f"Unexpected CSV format, fields: {fieldnames}")
            else:
                logger.warning(f"HF download returned {resp.status_code}")
        except Exception as e:
            logger.warning(f"Streaming download failed: {e}")

    # Method 2: HF datasets library
    count = _import_hf_via_library(db, batch_size, max_docs)
    if count > 0:
        return count

    # Method 3: Bulk fetch from epsteininvestigation.org API
    logger.info("Falling back to bulk API fetch from epsteininvestigation.org...")
    return _import_via_api_bulk(db, batch_size, max_docs or Config.BULK_IMPORT_MAX_DOCS)


def _ingest_csv_rows(db, rows, batch_size=100, max_docs=None):
    """Ingest documents from an iterable of CSV row dicts."""
    from app.models import Document

    # Pre-load existing file_ids to avoid N+1 duplicate checks
    existing_ids = {r[0] for r in db.session.query(Document.file_id).all()}

    count = 0
    skipped = 0

    for row in rows:
        if max_docs and count >= max_docs:
            break

        filename = row.get("filename", "").strip()
        text = row.get("text", "").strip()

        if not text or len(text) < 30:
            skipped += 1
            continue

        file_id = _filename_to_file_id(filename)
        if file_id in existing_ids:
            skipped += 1
            continue

        content_hash = hashlib.sha256(text[:2000].encode()).hexdigest()[:32]
        doc = Document(
            file_id=file_id,
            title=file_id,
            body=text,
            doc_type=_classify_doc_type(filename, text),
            source="huggingface-house-oversight",
            source_url="https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K",
            content_hash=content_hash,
            processed=False,
        )
        db.session.add(doc)
        existing_ids.add(file_id)
        count += 1

        if count % batch_size == 0:
            db.session.commit()
            _update_fts_batch(db, batch_size)
            logger.info(f"  ...imported {count} documents ({skipped} skipped)")

    db.session.commit()
    _update_fts_batch(db, count % batch_size or batch_size)
    logger.info(f"Imported {count} documents from CSV ({skipped} skipped)")
    return count


def _import_hf_via_library(db, batch_size=100, max_docs=None):
    """Try loading via the `datasets` library (handles auth, parquet, etc.)."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.info("datasets library not installed, skipping")
        return 0

    from app.models import Document

    logger.info("Loading HF dataset via datasets library...")
    try:
        ds = load_dataset("tensonaut/EPSTEIN_FILES_20K", split="train")
    except Exception as e:
        logger.warning(f"datasets library failed: {e}")
        return 0

    # Pre-load existing file_ids to avoid N+1 duplicate checks
    existing_ids = {r[0] for r in db.session.query(Document.file_id).all()}

    count = 0
    skipped = 0
    for row in ds:
        if max_docs and count >= max_docs:
            break

        filename = row.get("filename", "")
        text = row.get("text", "")
        if not text or len(text.strip()) < 30:
            skipped += 1
            continue

        file_id = _filename_to_file_id(filename)
        if file_id in existing_ids:
            skipped += 1
            continue

        doc = Document(
            file_id=file_id,
            title=file_id,
            body=text.strip(),
            doc_type=_classify_doc_type(filename, text),
            source="huggingface-house-oversight",
            processed=False,
        )
        db.session.add(doc)
        existing_ids.add(file_id)
        count += 1

        if count % batch_size == 0:
            db.session.commit()
            logger.info(f"  ...imported {count} documents")

    db.session.commit()
    logger.info(f"Imported {count} documents via datasets library ({skipped} skipped)")
    return count


def _import_via_api_bulk(db, batch_size=100, max_docs=None):
    """
    Bulk-fetch documents from the epsteininvestigation.org API.

    Strategy:
      1. Page through /documents for metadata (21,880 items — titles, types,
         dates, PDF URLs)
      2. Fetch richer excerpts via /search for common terms
      3. Documents without text are still stored (the AI worker can fetch
         text later via search when it analyses them)
    """
    from app.models import Document

    API_BASE = Config.ARCHIVE_API_URL
    count = 0
    max_docs = max_docs or Config.BULK_IMPORT_MAX_DOCS

    # Phase A: Bulk import metadata from /documents
    logger.info("Bulk import phase A: document metadata...")
    page = 1
    per_page = 100

    while count < max_docs:
        try:
            resp = requests.get(
                f"{API_BASE}/documents",
                params={"page": page, "limit": per_page},
                timeout=30,
            )
            if resp.status_code != 200:
                logger.warning(f"API /documents returned {resp.status_code} on page {page}")
                break

            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                if count >= max_docs:
                    break

                file_id = str(item.get("slug") or item.get("id", ""))
                if not file_id:
                    continue

                existing = Document.query.filter_by(file_id=file_id).first()
                if existing:
                    continue

                excerpt = item.get("excerpt") or ""
                doc = Document(
                    file_id=file_id,
                    title=item.get("title") or file_id,
                    body=excerpt if len(excerpt) > 30 else "",
                    doc_type=_api_type_to_doc_type(item.get("document_type", "")),
                    source="epsteininvestigation.org",
                    source_url=item.get("source_url") or item.get("file_url"),
                    original_url=item.get("file_url"),
                    date_str=item.get("document_date"),
                    page_count=item.get("page_count"),
                    processed=False,
                )
                db.session.add(doc)
                count += 1

            db.session.commit()
            logger.info(f"  ...imported {count} documents (page {page})")

            page += 1
            if len(items) < per_page:
                break

        except Exception as e:
            logger.error(f"Bulk import error on page {page}: {e}")
            db.session.rollback()
            break

    # Phase B: Enrich with search excerpts for key terms
    logger.info("Bulk import phase B: enriching with search text...")
    search_terms = [
        "trafficking", "victim", "deposition", "maxwell", "FBI",
        "email", "financial", "abuse", "travel", "phone", "police",
        "testimony", "massage", "flight", "photograph",
    ]
    enriched = 0
    for term in search_terms:
        if count >= max_docs:
            break
        enriched += _enrich_from_search(db, API_BASE, term, max_pages=5)

    db.session.commit()
    _update_fts_batch(db, count)
    logger.info(f"Bulk API import complete: {count} metadata + {enriched} enriched with text")
    return count


def _enrich_from_search(db, api_base, term, max_pages=5):
    """
    Fetch search results and either update existing documents or create new ones.
    Search results have 300-char excerpts — better than nothing for AI analysis.
    """
    from app.models import Document

    enriched = 0
    for page in range(1, max_pages + 1):
        try:
            resp = requests.get(
                f"{api_base}/search",
                params={"q": term, "page": page, "limit": 100},
                timeout=30,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                slug = str(item.get("slug") or "")
                excerpt = item.get("excerpt") or ""
                if not slug or len(excerpt) < 50:
                    continue

                doc = Document.query.filter_by(file_id=slug).first()
                if doc:
                    if not doc.body or len(doc.body) < len(excerpt):
                        doc.body = excerpt
                        enriched += 1
                else:
                    doc = Document(
                        file_id=slug,
                        title=item.get("title") or slug,
                        body=excerpt,
                        doc_type=_api_type_to_doc_type(item.get("document_type", "")),
                        source="epsteininvestigation.org",
                        source_url=item.get("file_url"),
                        date_str=item.get("document_date"),
                        processed=False,
                    )
                    db.session.add(doc)
                    enriched += 1

            db.session.commit()
        except Exception as e:
            logger.warning(f"Enrichment failed for term '{term}' page {page}: {e}")
            db.session.rollback()
            break

    return enriched


def _api_type_to_doc_type(raw: str) -> str:
    mapping = {
        "foia_release": "foia",
        "court_filing": "legal",
        "email": "email",
        "deposition": "deposition",
        "fbi_file": "fbi",
        "flight_log": "flight_log",
        "photograph": "photograph",
    }
    return mapping.get(raw, "document")


def import_all_archive_csvs(db):
    """Import all structured data from epsteininvestigation.org."""
    results = {}
    results["entities"] = import_archive_entities(db)
    results["relationships"] = import_archive_relationships(db)
    results["flights"] = import_archive_flights(db)
    return results


def _get_or_create_entity(db, name, etype="PERSON", cache=None):
    from app.models import Entity
    if cache is not None:
        if name in cache:
            return cache[name]
        entity = Entity(name=name, canonical_name=name, entity_type=etype, mention_count=0)
        db.session.add(entity)
        cache[name] = entity
        return entity
    entity = Entity.query.filter_by(name=name).first()
    if not entity:
        entity = Entity(name=name, canonical_name=name, entity_type=etype, mention_count=0)
        db.session.add(entity)
    return entity


def _normalise_entity_type(raw: str) -> str:
    raw = raw.strip().upper()
    mapping = {
        "PERSON": "PERSON",
        "ORGANIZATION": "ORG",
        "ORG": "ORG",
        "LOCATION": "LOCATION",
        "LOC": "LOCATION",
        "AIRCRAFT": "AIRCRAFT",
        "PROPERTY": "LOCATION",
    }
    return mapping.get(raw, raw)


def _filename_to_file_id(filename: str) -> str:
    """Extract a meaningful file ID from the HF dataset filename."""
    # e.g. "Epstein_Files/NOV2025/EFTA00123456.txt" -> "EFTA00123456"
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    return name or filename


def _classify_doc_type(filename: str, text: str) -> str:
    """Quick heuristic classification of document type."""
    lower = text[:500].lower()
    if "deposition" in lower or "q:" in lower or "a:" in lower:
        return "deposition"
    if "from:" in lower and "to:" in lower and ("subject:" in lower or "sent:" in lower):
        return "email"
    if "court" in lower or "case no" in lower or "plaintiff" in lower:
        return "legal"
    if "fbi" in lower or "federal bureau" in lower:
        return "fbi"
    if ".jpg" in filename.lower() or ".png" in filename.lower():
        return "image_ocr"
    return "document"


def _update_fts_batch(db, limit=100):
    """Sync recently added documents into the FTS index."""
    try:
        db.session.execute(
            db.text(
                "INSERT OR REPLACE INTO documents_fts(rowid, title, body, sender, recipients, subject) "
                "SELECT id, COALESCE(title,''), COALESCE(body,''), COALESCE(sender,''), "
                "COALESCE(recipients,''), COALESCE(subject,'') "
                "FROM documents WHERE id NOT IN (SELECT rowid FROM documents_fts) "
                "LIMIT :limit"
            ),
            {"limit": limit},
        )
        db.session.commit()
    except Exception as e:
        logger.warning(f"FTS update failed: {e}")
        db.session.rollback()
