"""
Search engine combining FTS5, fuzzy matching, and ML-based ranking.

Provides:
- Full-text search with SQLite FTS5 (porter stemming)
- Fuzzy name search (misspelling-tolerant via RapidFuzz)
- Date-range search
- Category/topic filtering
- Context-aware result grouping
- "Interesting documents" random sampling weighted by relevance
"""

import re
from datetime import datetime

from rapidfuzz import fuzz, process
from sqlalchemy import func, or_, and_, desc, text as sql_text

from app import db
from app.models import (
    Document, Entity, Category, Thread, NameVariant,
    document_entities, document_categories,
)
from app.nlp import resolve_name, clean_query, FuzzySearcher, KNOWN_NAMES


def search_fulltext(query: str, page: int = 1, per_page: int = 25, filters: dict | None = None):
    """
    Full-text search with FTS5 + fuzzy fallback.
    Returns paginated results ordered by relevance.
    """
    query = clean_query(query)
    filters = filters or {}
    results = {"items": [], "total": 0, "page": page, "pages": 0, "query": query}

    fts_query = _build_fts_query(query)

    try:
        fts_results = db.session.execute(
            sql_text(
                "SELECT rowid, rank FROM documents_fts "
                "WHERE documents_fts MATCH :query "
                "ORDER BY rank "
                "LIMIT :limit OFFSET :offset"
            ),
            {
                "query": fts_query,
                "limit": per_page,
                "offset": (page - 1) * per_page,
            },
        ).fetchall()

        count_result = db.session.execute(
            sql_text(
                "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH :query"
            ),
            {"query": fts_query},
        ).scalar()

    except Exception:
        fts_results = []
        count_result = 0

    if fts_results:
        doc_ids = [r[0] for r in fts_results]
        docs_query = Document.query.filter(
            Document.id.in_(doc_ids),
            Document.is_duplicate == False,
        )
        docs_query = _apply_filters(docs_query, filters)
        docs = {d.id: d for d in docs_query.all()}

        for row_id, rank in fts_results:
            doc = docs.get(row_id)
            if doc:
                d = doc.to_dict()
                d["search_rank"] = abs(rank)
                results["items"].append(d)

        results["total"] = count_result
        results["pages"] = (count_result + per_page - 1) // per_page

    if not results["items"]:
        fuzzy_results = _fuzzy_search_fallback(query, page, per_page, filters)
        if fuzzy_results["items"]:
            return fuzzy_results

    return results


def search_by_name(name: str, page: int = 1, per_page: int = 25):
    """Search for documents mentioning a person (fuzzy name matching)."""
    name = clean_query(name)
    canonical = resolve_name(name)

    variants = NameVariant.query.filter_by(canonical=canonical).all()
    all_names = {canonical.lower()}
    for v in variants:
        all_names.add(v.variant.lower())

    all_names_list = list(KNOWN_NAMES) + list(all_names)
    fuzzy_matches = process.extract(
        name.lower(), [n.lower() for n in all_names_list], scorer=fuzz.WRatio, limit=10
    )
    for match_name, score, _ in fuzzy_matches:
        if score >= 65:
            resolved = resolve_name(match_name)
            all_names.add(resolved.lower())
            all_names.add(match_name.lower())

    entity_ids = (
        db.session.query(Entity.id)
        .filter(
            or_(
                func.lower(Entity.canonical_name).in_(all_names),
                func.lower(Entity.name).in_(all_names),
            )
        )
        .all()
    )
    entity_ids = [e[0] for e in entity_ids]

    if entity_ids:
        query = (
            Document.query.join(document_entities)
            .filter(
                document_entities.c.entity_id.in_(entity_ids),
                Document.is_duplicate == False,
            )
            .order_by(desc(Document.relevance_score), Document.date.desc())
        )
    else:
        fts_names = " OR ".join(f'"{n}"' for n in all_names if len(n) > 2)
        if not fts_names:
            fts_names = f'"{name}"'
        try:
            doc_ids = db.session.execute(
                sql_text(
                    "SELECT rowid FROM documents_fts "
                    "WHERE documents_fts MATCH :query LIMIT 1000"
                ),
                {"query": fts_names},
            ).fetchall()
            doc_ids = [r[0] for r in doc_ids]
        except Exception:
            doc_ids = []

        if doc_ids:
            query = Document.query.filter(
                Document.id.in_(doc_ids), Document.is_duplicate == False
            ).order_by(desc(Document.relevance_score))
        else:
            return {
                "items": [], "total": 0, "page": page, "pages": 0,
                "name": name, "canonical_name": canonical,
                "searched_variants": list(all_names),
            }

    total = query.count()
    docs = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "items": [d.to_dict() for d in docs],
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page,
        "name": name,
        "canonical_name": canonical,
        "searched_variants": list(all_names),
    }


def search_by_date(start_date: str = None, end_date: str = None,
                   year: int = None, month: int = None,
                   page: int = 1, per_page: int = 25):
    """Search documents by date range."""
    query = Document.query.filter(
        Document.date.isnot(None),
        Document.is_duplicate == False,
    )

    if year:
        start = datetime(year, month or 1, 1)
        if month:
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
        else:
            end = datetime(year + 1, 1, 1)
        query = query.filter(Document.date >= start, Document.date < end)
    else:
        if start_date:
            from app.nlp import parse_date
            start = parse_date(start_date)
            if start:
                query = query.filter(Document.date >= start)
        if end_date:
            from app.nlp import parse_date
            end = parse_date(end_date)
            if end:
                query = query.filter(Document.date <= end)

    query = query.order_by(Document.date.asc())
    total = query.count()
    docs = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "items": [d.to_dict() for d in docs],
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page,
    }


def search_by_category(category_slug: str, page: int = 1, per_page: int = 25):
    """Search documents by category."""
    category = Category.query.filter_by(slug=category_slug).first()
    if not category:
        return {"items": [], "total": 0, "page": page, "pages": 0, "category": None}

    query = (
        Document.query.join(document_categories)
        .filter(
            document_categories.c.category_id == category.id,
            Document.is_duplicate == False,
        )
        .order_by(desc(Document.relevance_score), Document.date.desc())
    )

    total = query.count()
    docs = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "items": [d.to_dict() for d in docs],
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page,
        "category": category.to_dict(),
    }


def get_interesting_documents(count: int = 20):
    """
    Return a weighted-random selection of interesting documents.
    Weights by relevance score so more important documents surface more often.
    """
    high_relevance = (
        Document.query.filter(
            Document.relevance_score > 0.1,
            Document.is_duplicate == False,
            Document.body.isnot(None),
        )
        .order_by(func.random())
        .limit(count)
        .all()
    )

    if len(high_relevance) < count:
        remaining = count - len(high_relevance)
        existing_ids = [d.id for d in high_relevance]
        more = (
            Document.query.filter(
                Document.is_duplicate == False,
                Document.body.isnot(None),
                ~Document.id.in_(existing_ids) if existing_ids else True,
            )
            .order_by(func.random())
            .limit(remaining)
            .all()
        )
        high_relevance.extend(more)

    results = sorted(high_relevance, key=lambda d: d.relevance_score or 0, reverse=True)
    return [d.to_dict() for d in results]


def get_document_context(doc_id: int) -> dict:
    """Get full context for a document: thread, related docs, entities, timeline."""
    doc = Document.query.get(doc_id)
    if not doc:
        return {}

    context = {
        "document": doc.to_dict(),
        "full_body": doc.body,
        "thread": None,
        "related_documents": [],
        "timeline": [],
        "entities": [e.to_dict() for e in doc.entities],
        "categories": [c.to_dict() for c in doc.categories],
    }

    if doc.thread_id:
        thread = Thread.query.get(doc.thread_id)
        if thread:
            context["thread"] = thread.to_dict()
            context["thread"]["messages"] = [
                d.to_dict()
                for d in Document.query.filter_by(thread_id=thread.id)
                .order_by(Document.date.asc())
                .all()
            ]

    entity_ids = [e.id for e in doc.entities]
    if entity_ids:
        related = (
            Document.query.join(document_entities)
            .filter(
                document_entities.c.entity_id.in_(entity_ids),
                Document.id != doc.id,
                Document.is_duplicate == False,
            )
            .order_by(desc(Document.relevance_score))
            .limit(20)
            .all()
        )
        context["related_documents"] = [d.to_dict() for d in related]

    if doc.date:
        from datetime import timedelta
        window_start = doc.date - timedelta(days=30)
        window_end = doc.date + timedelta(days=30)
        timeline = (
            Document.query.filter(
                Document.date.between(window_start, window_end),
                Document.id != doc.id,
                Document.is_duplicate == False,
            )
            .order_by(Document.date.asc())
            .limit(50)
            .all()
        )
        context["timeline"] = [d.to_dict() for d in timeline]

    return context


def get_all_categories():
    return [c.to_dict() for c in Category.query.order_by(Category.name).all()]


def get_all_entities(entity_type: str = None, page: int = 1, per_page: int = 50):
    query = Entity.query
    if entity_type:
        query = query.filter_by(entity_type=entity_type)
    query = query.order_by(desc(Entity.mention_count))
    total = query.count()
    entities = query.offset((page - 1) * per_page).limit(per_page).all()
    return {
        "items": [e.to_dict() for e in entities],
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page,
    }


def get_date_histogram():
    """Get document counts by year-month for timeline display."""
    results = (
        db.session.query(
            func.strftime("%Y-%m", Document.date).label("month"),
            func.count(Document.id).label("count"),
        )
        .filter(Document.date.isnot(None), Document.is_duplicate == False)
        .group_by("month")
        .order_by("month")
        .all()
    )
    return [{"month": r.month, "count": r.count} for r in results]


def get_stats():
    """Get overall database statistics."""
    return {
        "total_documents": Document.query.filter_by(is_duplicate=False).count(),
        "total_emails": Document.query.filter_by(doc_type="email", is_duplicate=False).count(),
        "total_entities": Entity.query.count(),
        "total_categories": Category.query.count(),
        "total_threads": Thread.query.count(),
        "total_duplicates": Document.query.filter_by(is_duplicate=True).count(),
        "high_relevance": Document.query.filter(
            Document.relevance_score > 0.3, Document.is_duplicate == False
        ).count(),
    }


def _build_fts_query(query: str) -> str:
    """
    Build an FTS5 query.

    Multi-word queries become NEAR phrases so that "nick lees" matches
    the two words adjacent to each other (with any punctuation/whitespace)
    but NOT "Nick ... Cathy Lees".

    Single-word queries use prefix matching ("groff*").
    """
    query = clean_query(query)
    words = query.split()

    if not words:
        return '""'

    if len(words) == 1:
        w = words[0]
        return f"{w}*" if len(w) > 2 else w

    # Multi-word: use NEAR with distance 0 (adjacent tokens only).
    # FTS5 tokeniser already strips punctuation, so "Nick;  lees"
    # becomes tokens [nick, lees] which are adjacent.
    safe_words = [w for w in words if w]
    return "NEAR(" + " ".join(safe_words) + ", 0)"


def _fuzzy_search_fallback(query: str, page: int, per_page: int, filters: dict):
    """Fall back to fuzzy name search when FTS finds nothing."""
    canonical = resolve_name(query)
    if canonical != query:
        return search_by_name(canonical, page, per_page)

    docs = (
        Document.query.filter(
            Document.is_duplicate == False,
            or_(
                Document.body.ilike(f"%{query}%"),
                Document.title.ilike(f"%{query}%"),
                Document.sender.ilike(f"%{query}%"),
                Document.recipients.ilike(f"%{query}%"),
                Document.subject.ilike(f"%{query}%"),
            ),
        )
        .order_by(desc(Document.relevance_score))
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    total = Document.query.filter(
        Document.is_duplicate == False,
        or_(
            Document.body.ilike(f"%{query}%"),
            Document.title.ilike(f"%{query}%"),
        ),
    ).count()

    return {
        "items": [d.to_dict() for d in docs],
        "total": total,
        "page": page,
        "pages": (total + per_page - 1) // per_page,
        "query": query,
        "fuzzy_fallback": True,
    }


def _apply_filters(query, filters: dict):
    if filters.get("doc_type"):
        query = query.filter(Document.doc_type == filters["doc_type"])
    if filters.get("source"):
        query = query.filter(Document.source == filters["source"])
    if filters.get("min_relevance"):
        query = query.filter(Document.relevance_score >= float(filters["min_relevance"]))
    if filters.get("date_from"):
        from app.nlp import parse_date
        dt = parse_date(filters["date_from"])
        if dt:
            query = query.filter(Document.date >= dt)
    if filters.get("date_to"):
        from app.nlp import parse_date
        dt = parse_date(filters["date_to"])
        if dt:
            query = query.filter(Document.date <= dt)
    return query
