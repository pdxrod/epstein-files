"""
Flask routes: page views and JSON API.
"""

from flask import (
    Blueprint, render_template, request, jsonify, session,
    redirect, url_for, abort,
)

from app import db
from app.search import (
    search_fulltext, search_by_name, search_by_date, search_by_category,
    get_interesting_documents, get_document_context, get_all_categories,
    get_all_entities, get_date_histogram, get_stats,
)
from app.models import Document, Category

main_bp = Blueprint("main", __name__)
api_bp = Blueprint("api", __name__)


def _age_verified():
    return session.get("age_verified", False)


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


@main_bp.route("/")
def index():
    stats = get_stats()
    categories = get_all_categories()
    return render_template("index.html", stats=stats, categories=categories)


@main_bp.route("/search")
def search_page():
    query = request.args.get("q", "").strip()
    search_type = request.args.get("type", "fulltext")
    page = request.args.get("page", 1, type=int)

    results = {"items": [], "total": 0, "page": 1, "pages": 0}

    if query:
        filters = {
            "doc_type": request.args.get("doc_type"),
            "source": request.args.get("source"),
            "min_relevance": request.args.get("min_relevance"),
            "date_from": request.args.get("date_from"),
            "date_to": request.args.get("date_to"),
        }
        filters = {k: v for k, v in filters.items() if v}

        if search_type == "name":
            results = search_by_name(query, page=page)
        elif search_type == "date":
            results = search_by_date(
                start_date=request.args.get("date_from"),
                end_date=request.args.get("date_to"),
                page=page,
            )
        elif search_type == "category":
            results = search_by_category(query, page=page)
        else:
            results = search_fulltext(query, page=page, filters=filters)

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


@main_bp.route("/category/<slug>")
def category_view(slug):
    page = request.args.get("page", 1, type=int)
    results = search_by_category(slug, page=page)
    return render_template("category.html", results=results, slug=slug)


@main_bp.route("/random")
def random_page():
    docs = get_interesting_documents(count=20)
    return render_template("random.html", documents=docs)


@main_bp.route("/timeline")
def timeline_page():
    histogram = get_date_histogram()
    return render_template("timeline.html", histogram=histogram)


@main_bp.route("/people")
def people_page():
    page = request.args.get("page", 1, type=int)
    entities = get_all_entities(entity_type="PERSON", page=page, per_page=100)
    return render_template("people.html", entities=entities)


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

    filters = {
        "doc_type": request.args.get("doc_type"),
        "source": request.args.get("source"),
        "min_relevance": request.args.get("min_relevance"),
        "date_from": request.args.get("date_from"),
        "date_to": request.args.get("date_to"),
    }
    filters = {k: v for k, v in filters.items() if v}

    return jsonify(search_fulltext(query, page=page, filters=filters))


@api_bp.route("/search/name")
def api_search_name():
    name = request.args.get("q", "").strip()
    page = request.args.get("page", 1, type=int)
    if not name:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    return jsonify(search_by_name(name, page=page))


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
    return jsonify(get_interesting_documents(count=min(count, 50)))


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


@api_bp.route("/ingest", methods=["POST"])
def api_ingest():
    """Trigger ingestion from various sources."""
    data = request.get_json() or {}
    source = data.get("source", "doj")

    if source == "doj":
        from app.scraper import ingest_from_doj
        job = ingest_from_doj()
        return jsonify({"status": "started", "job_id": job.id})
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

    return jsonify({"error": f"Unknown source: {source}"}), 400
