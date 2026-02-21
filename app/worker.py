"""
Background worker that continuously:
  1. Imports structured data (entities, relationships, flights) from CSV sources
  2. Processes un-analysed local documents (e.g. from HuggingFace bulk import)
  3. Fetches new documents from external API (epsteininvestigation.org)
  4. Sends each document through the AI analysis pipeline (Ollama LLM)
  5. Stores AI-discovered categories, entities, relevance scores in the DB
  6. Periodically runs category discovery across recent analyses

Runs in a background thread so the web UI stays responsive.
"""

import json
import logging
import re
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_worker_thread: threading.Thread | None = None
_worker_stop = threading.Event()
_worker_status = {
    "running": False,
    "phase": "idle",
    "documents_analysed": 0,
    "documents_imported": 0,
    "categories_discovered": 0,
    "current_task": "idle",
    "last_error": None,
    "started_at": None,
    "recent_analyses": [],
}
_status_lock = threading.Lock()

CATEGORY_DISCOVERY_BATCH = 15
ANALYSIS_DELAY = 2
FETCH_DELAY = 1
FETCH_BATCH_SIZE = 20


def get_worker_status() -> dict:
    with _status_lock:
        return dict(_worker_status)


def start_worker(app):
    """Start the background analysis worker."""
    global _worker_thread

    if _worker_thread and _worker_thread.is_alive():
        logger.info("Worker already running")
        return False

    _worker_stop.clear()
    _worker_thread = threading.Thread(
        target=_worker_loop,
        args=(app,),
        daemon=True,
        name="ai-worker",
    )
    _worker_thread.start()

    with _status_lock:
        _worker_status["running"] = True
        _worker_status["started_at"] = datetime.utcnow().isoformat()
        _worker_status["current_task"] = "starting"

    logger.info("Background AI worker started")
    return True


def stop_worker():
    """Signal the worker to stop."""
    _worker_stop.set()
    with _status_lock:
        _worker_status["running"] = False
        _worker_status["current_task"] = "stopping"
    logger.info("Worker stop requested")
    return True


def _update_status(**kwargs):
    with _status_lock:
        _worker_status.update(kwargs)


def _worker_loop(app):
    """Main worker loop — three phases."""
    with app.app_context():
        from app import db
        from app.models import Document, Entity, Category
        from app.ai import analyse_document, discover_categories, check_ollama
        from app.live_search import search_external
        from app.scraper import ensure_system_categories

        ensure_system_categories()

        # Check Ollama
        ollama_status = check_ollama()
        if ollama_status["status"] != "running":
            _update_status(
                current_task="error: Ollama not running",
                last_error="Ollama is not running. Install from https://ollama.ai and run: ollama pull llama3.1:8b",
                running=False,
            )
            logger.error("Ollama not running. Worker cannot start.")
            return

        if not ollama_status["model_available"]:
            model = ollama_status["configured_model"]
            _update_status(
                current_task=f"error: model {model} not found",
                last_error=f"Model {model} not available. Run: ollama pull {model}",
                running=False,
            )
            logger.error(f"Model {model} not available")
            return

        logger.info("Worker loop started, Ollama connected")

        # ── Phase 1: Import structured CSV data ──
        _phase_import_csvs(db)

        # ── Phase 2: Analyse un-processed local documents ──
        _phase_analyse_local(db, Document, Category, analyse_document, discover_categories)

        # ── Phase 3: Fetch + analyse from external API ──
        _phase_fetch_and_analyse(db, Document, Category, analyse_document,
                                 discover_categories, search_external)

        _update_status(running=False, current_task="stopped")
        logger.info("Worker stopped")


def _phase_import_csvs(db):
    """Phase 1: Import entities, relationships, flights from CSV if not done."""
    from app.models import Entity, EntityRelationship, FlightRecord

    entity_count = Entity.query.count()
    rel_count = EntityRelationship.query.count()
    flight_count = FlightRecord.query.count()

    if entity_count < 50 or rel_count < 100 or flight_count < 10:
        _update_status(phase="importing CSVs", current_task="importing archive data...")
        logger.info("Phase 1: Importing structured data from epsteininvestigation.org")
        try:
            from app.datasources import import_all_archive_csvs
            results = import_all_archive_csvs(db)
            imported = sum(results.values())
            with _status_lock:
                _worker_status["documents_imported"] += imported
            logger.info(f"CSV import complete: {results}")
        except Exception as e:
            logger.warning(f"CSV import failed (non-fatal): {e}")
            _update_status(last_error=f"CSV import: {e}")
    else:
        logger.info("Phase 1: Structured data already imported, skipping")


def _phase_analyse_local(db, Document, Category, analyse_document, discover_categories):
    """Phase 2: Analyse any local documents that haven't been processed by AI yet."""
    unprocessed = Document.query.filter_by(processed=False).count()
    if unprocessed == 0:
        logger.info("Phase 2: No unprocessed local documents, skipping")
        return

    _update_status(phase="analysing local docs", current_task=f"{unprocessed} docs queued")
    logger.info(f"Phase 2: {unprocessed} local documents awaiting AI analysis")

    analyses_since_discovery = 0
    recent_summaries = []
    processed_count = 0

    while not _worker_stop.is_set():
        batch = (
            Document.query
            .filter_by(processed=False)
            .order_by(Document.id)
            .limit(10)
            .all()
        )
        if not batch:
            break

        known_cats = [c.name for c in Category.query.all()]

        for doc in batch:
            if _worker_stop.is_set():
                return

            if not doc.body or len(doc.body) < 30:
                doc.processed = True
                db.session.commit()
                continue

            _update_status(current_task=f"AI: {doc.file_id} ({processed_count}/{unprocessed})")
            logger.info(f"Analysing local doc {doc.file_id}...")

            analysis = analyse_document(
                text=doc.body,
                file_id=doc.file_id,
                source=doc.source or "",
                date=doc.date_str or "",
                known_categories=known_cats,
            )

            if analysis:
                _apply_analysis_to_doc(db, doc, analysis)
                processed_count += 1
                analyses_since_discovery += 1
                _record_analysis(doc.file_id, analysis, recent_summaries)

                for new_cat in analysis.get("new_categories", []):
                    _ensure_ai_category(db, new_cat)

            else:
                doc.processed = True
                db.session.commit()

            time.sleep(ANALYSIS_DELAY)

        if analyses_since_discovery >= CATEGORY_DISCOVERY_BATCH and recent_summaries:
            _run_category_discovery(db, known_cats, recent_summaries, discover_categories)
            analyses_since_discovery = 0
            recent_summaries = []

    logger.info(f"Phase 2 complete: {processed_count} documents analysed")


def _phase_fetch_and_analyse(db, Document, Category, analyse_document,
                              discover_categories, search_external):
    """Phase 3: Continuously fetch from external API and analyse."""
    _update_status(phase="fetching & analysing")

    analyses_since_discovery = 0
    recent_summaries = []
    search_page = 1
    search_terms = [
        "trafficking victim", "Ghislaine Maxwell", "abuse minor",
        "flight log passenger", "deposition testimony",
        "financial transfer wire", "blackmail recording",
        "Prince Andrew", "Les Wexner", "Palm Beach police",
        "Virginia Roberts", "massage", "young girls",
        "Jean-Luc Brunel", "Little St James", "plea deal",
        "Sarah Kellen", "Nadia Marcinkova",
        "cover up obstruction", "intelligence agency",
        "shell company trust", "pilot manifest",
        "house oversight committee", "FBI interview",
    ]
    search_idx = 0

    while not _worker_stop.is_set():
        try:
            term = search_terms[search_idx % len(search_terms)]
            _update_status(current_task=f"fetching: {term} (page {search_page})")

            ext_results = search_external(term, page=search_page, limit=FETCH_BATCH_SIZE)
            docs = ext_results.get("items", [])

            if not docs:
                search_idx += 1
                search_page = 1
                time.sleep(FETCH_DELAY)
                continue

            known_cats = [c.name for c in Category.query.all()]

            for doc_data in docs:
                if _worker_stop.is_set():
                    break

                file_id = doc_data.get("file_id", "")
                body = doc_data.get("body", "")

                if not body or len(body) < 30:
                    continue

                existing = Document.query.filter_by(file_id=file_id).first()
                if existing and existing.processed:
                    continue

                _update_status(current_task=f"AI: {file_id}")
                logger.info(f"Analysing {file_id}...")

                analysis = analyse_document(
                    text=body,
                    file_id=file_id,
                    source=doc_data.get("source", ""),
                    date=doc_data.get("date", ""),
                    known_categories=known_cats,
                )

                if analysis:
                    _store_analysis_from_external(db, doc_data, analysis)
                    analyses_since_discovery += 1
                    _record_analysis(file_id, analysis, [])

                    for new_cat in analysis.get("new_categories", []):
                        _ensure_ai_category(db, new_cat)

                time.sleep(ANALYSIS_DELAY)

            if analyses_since_discovery >= CATEGORY_DISCOVERY_BATCH:
                _run_category_discovery(db, known_cats,
                                        _worker_status.get("recent_analyses", []),
                                        discover_categories)
                analyses_since_discovery = 0

            if len(docs) < FETCH_BATCH_SIZE:
                search_idx += 1
                search_page = 1
            else:
                search_page += 1

            time.sleep(FETCH_DELAY)

        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
            _update_status(last_error=str(e))
            time.sleep(10)


def _apply_analysis_to_doc(db, doc, analysis):
    """Apply AI analysis results to an existing Document object."""
    from app.models import Entity, Category

    doc.relevance_score = analysis.get("relevance_score", 0)
    doc.relevance_categories = json.dumps(
        analysis.get("categories", []) + analysis.get("new_categories", [])
    )
    doc.ai_summary = analysis.get("summary", "")
    doc.ai_connections = analysis.get("connections", "")
    doc.processed = True

    for ent_data in analysis.get("entities", []):
        ent_name = ent_data.get("name", "").strip()
        ent_type = ent_data.get("type", "PERSON").upper()
        if not ent_name or len(ent_name) < 2:
            continue

        entity = Entity.query.filter_by(name=ent_name, entity_type=ent_type).first()
        if not entity:
            entity = Entity(
                name=ent_name, canonical_name=ent_name,
                entity_type=ent_type,
                description=ent_data.get("role", ""),
                mention_count=1,
            )
            db.session.add(entity)
        else:
            entity.mention_count = (entity.mention_count or 0) + 1
        if entity not in doc.entities:
            doc.entities.append(entity)

    all_cats = analysis.get("categories", []) + analysis.get("new_categories", [])
    for cat_name in all_cats:
        if not cat_name or not isinstance(cat_name, str):
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", cat_name.lower()).strip("-")
        if not slug:
            continue
        cat = Category.query.filter_by(slug=slug).first()
        if cat and cat not in doc.categories:
            doc.categories.append(cat)
            cat.document_count = (cat.document_count or 0) + 1

    try:
        db.session.commit()
        _update_fts(db, doc)
    except Exception as e:
        logger.warning(f"DB commit failed for {doc.file_id}: {e}")
        db.session.rollback()


def _store_analysis_from_external(db, doc_data: dict, analysis: dict):
    """Store an AI analysis result from an external API document."""
    from app.models import Document

    file_id = doc_data.get("file_id", "unknown")
    existing = Document.query.filter_by(file_id=file_id).first()

    if existing:
        doc = existing
    else:
        doc = Document(file_id=file_id)
        db.session.add(doc)

    doc.title = doc_data.get("title") or file_id
    doc.body = doc_data.get("body", "")
    doc.doc_type = doc_data.get("doc_type", "document")
    doc.source = doc_data.get("source", "external")
    doc.source_url = doc_data.get("source_url")
    doc.original_url = doc_data.get("original_url")
    doc.date_str = doc_data.get("date")

    db.session.flush()
    _apply_analysis_to_doc(db, doc, analysis)


def _record_analysis(file_id, analysis, recent_summaries):
    """Record an analysis in the worker status for UI display."""
    summary_entry = {
        "file_id": file_id,
        "summary": analysis.get("summary", ""),
        "relevance": analysis.get("relevance_score", 0),
        "categories": analysis.get("categories", []),
        "new_categories": analysis.get("new_categories", []),
    }
    with _status_lock:
        _worker_status["documents_analysed"] += 1
        _worker_status["recent_analyses"].append(summary_entry)
        _worker_status["recent_analyses"] = _worker_status["recent_analyses"][-20:]
    recent_summaries.append(summary_entry)


def _run_category_discovery(db, known_cats, recent_summaries, discover_categories):
    """Run category discovery on recent summaries."""
    _update_status(current_task="discovering categories...")
    logger.info(f"Running category discovery on {len(recent_summaries)} summaries")

    new_cats = discover_categories(recent_summaries, known_cats)
    for cat_data in new_cats:
        name = cat_data if isinstance(cat_data, str) else cat_data.get("name", "")
        _ensure_ai_category(db, cat_data)
        with _status_lock:
            _worker_status["categories_discovered"] += 1


def _update_fts(db, doc):
    """Update FTS index for a single document."""
    try:
        db.session.execute(
            db.text(
                "INSERT OR REPLACE INTO documents_fts(rowid, title, body, sender, recipients, subject) "
                "VALUES (:id, :title, :body, :sender, :recipients, :subject)"
            ),
            {
                "id": doc.id,
                "title": doc.title or "",
                "body": doc.body or "",
                "sender": doc.sender or "",
                "recipients": doc.recipients or "",
                "subject": doc.subject or "",
            },
        )
        db.session.commit()
    except Exception:
        db.session.rollback()


def _ensure_ai_category(db, name_or_data):
    """Create an AI-discovered category if it doesn't exist."""
    from app.models import Category

    if isinstance(name_or_data, dict):
        name = name_or_data.get("name", "")
        slug = name_or_data.get("slug", "")
        desc = name_or_data.get("description", "")
    else:
        name = str(name_or_data).strip()
        slug = ""
        desc = ""

    if not name or len(name) < 3:
        return

    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    existing = Category.query.filter_by(slug=slug).first()
    if not existing:
        cat = Category(
            name=name,
            slug=slug,
            description=desc or f"AI-discovered category: {name}",
            is_system=False,
            document_count=0,
        )
        db.session.add(cat)
        try:
            db.session.commit()
            logger.info(f"New AI category discovered: {name}")
        except Exception:
            db.session.rollback()
