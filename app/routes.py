"""
Flask routes: page views and JSON API.

When the local database has no results for a query, automatically falls back
to the public Epstein Files API at epsteininvestigation.org (207K+ documents).
"""

import logging
import threading

from flask import (
    Blueprint, render_template, request, jsonify, session,
    redirect, url_for, abort, current_app,
)

from app import db
from app.search import (
    search_fulltext, search_by_name, search_by_date, search_by_category,
    get_interesting_documents, get_document_context, get_all_categories,
    get_all_entities, get_date_histogram, get_stats,
)
from app.live_search import (
    search_external, search_external_documents, search_external_entities,
    search_external_flights, get_external_document,
)
from app.models import Document, Category, IngestJob
from app.ai import check_ollama
from app.worker import start_worker, stop_worker, get_worker_status

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)
api_bp = Blueprint("api", __name__)

_MAX_LIMIT = 200


def _age_verified():
    return session.get("age_verified", False)


def _parse_search_filters() -> dict:
    """Extract and clean search filter parameters from the request."""
    filters = {
        "doc_type": request.args.get("doc_type"),
        "source": request.args.get("source"),
        "min_relevance": request.args.get("min_relevance"),
        "date_from": request.args.get("date_from"),
        "date_to": request.args.get("date_to"),
    }
    return {k: v for k, v in filters.items() if v}


def _search_with_fallback(query, page, search_type, filters):
    """Search local DB first; if empty, fall back to external API."""
    results = {"items": [], "total": 0, "page": page, "pages": 0}

    if search_type == "name":
        results = search_by_name(query, page=page)
    elif search_type == "date":
        results = search_by_date(
            start_date=filters.get("date_from"),
            end_date=filters.get("date_to"),
            page=page,
        )
    elif search_type == "category":
        results = search_by_category(query, page=page)
    else:
        results = search_fulltext(query, page=page, filters=filters)

    if not results.get("items"):
        logger.info(f"No local results for '{query}', querying external API…")
        ext = search_external(query, page=page)
        if ext.get("items"):
            return ext

    return results


# ─── Age gate ────────────────────────────────────────────────────────────────

@main_bp.before_request
def check_age_gate():
    allowed = {"main.age_gate", "main.verify_age", "static"}
    if request.endpoint not in allowed and not _age_verified():
        return redirect(url_for("main.age_gate"))


@main_bp.route("/age-check")
def age_gate():
    return render_template("age_gate.html")


@main_bp.route("/verify-age", methods=["POST"])
def verify_age():
    if request.form.get("age_confirmed") == "yes":
        session["age_verified"] = True
        return redirect(url_for("main.index"))
    return redirect(url_for("main.age_gate"))


# ─── Pages ───────────────────────────────────────────────────────────────────

@main_bp.route("/")
def index():
    stats = get_stats()
    categories = get_all_categories()
    has_data = stats.get("total_documents", 0) > 0
    worker = get_worker_status()
    ai_categories = (
        Category.query.filter_by(is_system=False)
        .order_by(Category.document_count.desc())
        .all()
    )
    return render_template(
        "index.html", stats=stats, categories=categories,
        has_data=has_data, worker=worker,
        ai_categories=[c.to_dict() for c in ai_categories],
    )


@main_bp.route("/search")
def search_page():
    query = request.args.get("q", "").strip()
    search_type = request.args.get("type", "fulltext")
    page = request.args.get("page", 1, type=int)

    results = {"items": [], "total": 0, "page": 1, "pages": 0}

    if query:
        results = _search_with_fallback(query, page, search_type, _parse_search_filters())

    categories = get_all_categories()
    return render_template(
        "search_results.html",
        results=results,
        query=query,
        search_type=search_type,
        categories=categories,
    )


@main_bp.route("/document/<int:doc_id>")
def document_view(doc_id):
    context = get_document_context(doc_id)
    if not context:
        abort(404)
    return render_template("document.html", context=context)


@main_bp.route("/external/<slug>")
def external_document_view(slug):
    """View a document fetched live from the external API."""
    doc = get_external_document(slug)
    if not doc:
        abort(404)
    context = {
        "document": doc,
        "full_body": doc.get("body", ""),
        "thread": None,
        "related_documents": [],
        "timeline": [],
        "entities": [],
        "categories": [],
        "external": True,
    }
    return render_template("document.html", context=context)


@main_bp.route("/category/<slug>")
def category_view(slug):
    page = request.args.get("page", 1, type=int)
    results = search_by_category(slug, page=page)

    if not results.get("items"):
        ext = search_external_documents(slug, page=page)
        if ext.get("items"):
            results = ext

    return render_template("category.html", results=results, slug=slug)


@main_bp.route("/random")
def random_page():
    docs = get_interesting_documents(count=20)
    if not docs:
        ext = search_external("epstein victim trafficking", page=1, limit=20)
        docs = ext.get("items", [])
    return render_template("random.html", documents=docs)


@main_bp.route("/timeline")
def timeline_page():
    histogram = get_date_histogram()
    return render_template("timeline.html", histogram=histogram)


@main_bp.route("/people")
def people_page():
    page = request.args.get("page", 1, type=int)
    entities = get_all_entities(entity_type="PERSON", page=page, per_page=100)

    if not entities.get("items"):
        ext = search_external_entities("", entity_type="person", page=page)
        if ext.get("items"):
            entities = {
                "items": [
                    {
                        "id": e.get("id"),
                        "name": e.get("name"),
                        "canonical_name": e.get("name"),
                        "entity_type": "PERSON",
                        "mention_count": e.get("document_count", 0),
                    }
                    for e in ext["items"]
                ],
                "total": ext.get("total", 0),
                "page": page,
                "pages": ext.get("pages", 0),
                "external": True,
            }

    return render_template("people.html", entities=entities)


@main_bp.route("/flights")
def flights_page():
    passenger = request.args.get("passenger", "")
    airport = request.args.get("airport", "")
    page = request.args.get("page", 1, type=int)

    results = search_external_flights(
        passenger=passenger or None,
        airport=airport or None,
        page=page,
    )
    return render_template("flights.html", results=results, passenger=passenger, airport=airport)


@main_bp.route("/admin")
def admin_page():
    stats = get_stats()
    jobs = IngestJob.query.order_by(IngestJob.id.desc()).limit(20).all()
    ollama = check_ollama()
    worker = get_worker_status()
    ai_categories = (
        Category.query.filter_by(is_system=False)
        .order_by(Category.document_count.desc())
        .all()
    )
    return render_template(
        "admin.html", stats=stats, jobs=jobs,
        ollama=ollama, worker=worker, ai_categories=ai_categories,
    )


# ─── JSON API ────────────────────────────────────────────────────────────────

@api_bp.before_request
def api_age_check():
    if not _age_verified():
        return jsonify({"error": "Age verification required"}), 403


@api_bp.route("/search")
def api_search():
    query = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    results = _search_with_fallback(query, page, "fulltext", _parse_search_filters())
    return jsonify(results)


@api_bp.route("/search/name")
def api_search_name():
    name = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    if not name:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    results = search_by_name(name, page=page)
    if not results.get("items"):
        ext = search_external(name, page=page)
        if ext.get("items"):
            return jsonify(ext)
    return jsonify(results)


@api_bp.route("/search/date")
def api_search_date():
    page = request.args.get("page", 1, type=int)
    return jsonify(search_by_date(
        start_date=request.args.get("start"),
        end_date=request.args.get("end"),
        year=request.args.get("year", type=int),
        month=request.args.get("month", type=int),
        page=page,
    ))


@api_bp.route("/search/category/<slug>")
def api_search_category(slug):
    page = request.args.get("page", 1, type=int)
    return jsonify(search_by_category(slug, page=page))


@api_bp.route("/document/<int:doc_id>")
def api_document(doc_id):
    context = get_document_context(doc_id)
    if not context:
        return jsonify({"error": "Document not found"}), 404
    return jsonify(context)


@api_bp.route("/random")
def api_random():
    count = request.args.get("count", 20, type=int)
    docs = get_interesting_documents(count=min(count, 50))
    if not docs:
        ext = search_external("epstein victim trafficking", limit=count)
        return jsonify(ext.get("items", []))
    return jsonify(docs)


@api_bp.route("/categories")
def api_categories():
    return jsonify(get_all_categories())


@api_bp.route("/entities")
def api_entities():
    entity_type = request.args.get("type")
    page = request.args.get("page", 1, type=int)
    return jsonify(get_all_entities(entity_type=entity_type, page=page))


@api_bp.route("/timeline")
def api_timeline():
    return jsonify(get_date_histogram())


@api_bp.route("/stats")
def api_stats():
    return jsonify(get_stats())


@api_bp.route("/external/search")
def api_external_search():
    """Search external API directly."""
    query = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    return jsonify(search_external(query, page=page))


@api_bp.route("/external/entities")
def api_external_entities():
    query = request.args.get("q", "")
    entity_type = request.args.get("type")
    page = request.args.get("page", 1, type=int)
    return jsonify(search_external_entities(query, entity_type=entity_type, page=page))


@api_bp.route("/external/flights")
def api_external_flights():
    return jsonify(search_external_flights(
        passenger=request.args.get("passenger"),
        airport=request.args.get("airport"),
        date_from=request.args.get("date_from"),
        date_to=request.args.get("date_to"),
        page=request.args.get("page", 1, type=int),
    ))


@api_bp.route("/worker/start", methods=["POST"])
def api_worker_start():
    """Start the background AI analysis worker."""
    started = start_worker(current_app._get_current_object())
    return jsonify({"started": started, "status": get_worker_status()})


@api_bp.route("/worker/stop", methods=["POST"])
def api_worker_stop():
    """Stop the background AI analysis worker."""
    stop_worker()
    return jsonify({"stopped": True, "status": get_worker_status()})


@api_bp.route("/worker/status")
def api_worker_status():
    return jsonify(get_worker_status())


@api_bp.route("/ai/status")
def api_ai_status():
    return jsonify(check_ollama())


def _run_in_background(fn, *args):
    """Run a function in a daemon thread with its own app context."""
    app = current_app._get_current_object()
    def _target():
        with app.app_context():
            try:
                fn(*args)
            except Exception as e:
                logger.error(f"Background task {fn.__name__} failed: {e}", exc_info=True)
    threading.Thread(target=_target, daemon=True).start()


@api_bp.route("/ingest", methods=["POST"])
def api_ingest():
    """Trigger ingestion from various sources."""
    data = request.get_json() or {}
    source = data.get("source", "doj")

    if source == "doj":
        from app.scraper import ingest_from_doj
        _run_in_background(ingest_from_doj)
        return jsonify({"status": "started", "message": "DOJ ingestion running in background"})

    elif source == "local":
        directory = data.get("directory")
        if not directory:
            return jsonify({"error": "directory is required for local source"}), 400
        from app.scraper import ingest_local_pdfs
        count = ingest_local_pdfs(directory)
        return jsonify({"status": "completed", "documents_processed": count})

    elif source == "text":
        file_id = data.get("file_id")
        text = data.get("text")
        metadata = data.get("metadata", {})
        if not file_id or not text:
            return jsonify({"error": "file_id and text are required"}), 400
        from app.scraper import ingest_text_content
        doc = ingest_text_content(file_id, text, metadata)
        return jsonify({"status": "ok", "document": doc.to_dict() if doc else None})

    elif source == "huggingface":
        max_docs = data.get("max_docs")
        from app.datasources import import_huggingface_dataset
        _run_in_background(import_huggingface_dataset, db, max_docs)
        return jsonify({"status": "started", "message": "HuggingFace import running in background"})

    elif source == "archive_csvs":
        try:
            from app.datasources import import_all_archive_csvs
            results = import_all_archive_csvs(db)
            return jsonify({"status": "completed", "results": results})
        except Exception as e:
            logger.error(f"CSV import failed: {e}", exc_info=True)
            return jsonify({"status": "error", "error": str(e)}), 500

    return jsonify({"error": f"Unknown source: {source}"}), 400


@api_bp.route("/data/relationships")
def api_relationships():
    """Get entity relationship graph."""
    from app.models import EntityRelationship, Entity
    page = request.args.get("page", 1, type=int)
    limit = min(request.args.get("limit", 50, type=int), _MAX_LIMIT)
    entity = request.args.get("entity", "")

    query = EntityRelationship.query
    if entity:
        matches = Entity.query.filter(Entity.name.ilike(f"%{entity}%")).all()
        ids = [e.id for e in matches]
        if ids:
            query = query.filter(
                db.or_(
                    EntityRelationship.entity_a_id.in_(ids),
                    EntityRelationship.entity_b_id.in_(ids),
                )
            )
        else:
            return jsonify({"items": [], "total": 0})

    total = query.count()
    rels = query.order_by(EntityRelationship.strength.desc()).offset(
        (page - 1) * limit
    ).limit(limit).all()

    return jsonify({
        "items": [r.to_dict() for r in rels],
        "total": total,
        "page": page,
    })


@api_bp.route("/data/flights")
def api_local_flights():
    """Get locally imported flight records."""
    from app.models import FlightRecord
    page = request.args.get("page", 1, type=int)
    limit = min(request.args.get("limit", 50, type=int), _MAX_LIMIT)
    passenger = request.args.get("passenger", "")

    query = FlightRecord.query
    if passenger:
        query = query.filter(FlightRecord.passengers.ilike(f"%{passenger}%"))

    total = query.count()
    flights = query.order_by(FlightRecord.flight_date.desc()).offset(
        (page - 1) * limit
    ).limit(limit).all()

    return jsonify({
        "items": [f.to_dict() for f in flights],
        "total": total,
        "page": page,
    })
