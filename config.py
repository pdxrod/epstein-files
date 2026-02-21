import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def _int(key, default):
    return int(os.environ.get(key, default))


def _float(key, default):
    return float(os.environ.get(key, default))


class Config:
    # ── Server ────────────────────────────────────────────────
    PORT = _int("PORT", 5555)
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(32).hex())

    # ── Database ──────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'epstein.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")

    # ── Ollama / AI ───────────────────────────────────────────
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")  # empty = auto-detect

    # ── External APIs ─────────────────────────────────────────
    ARCHIVE_API_URL = os.environ.get(
        "ARCHIVE_API_URL", "https://www.epsteininvestigation.org/api/v1"
    )
    DOJ_BASE_URL = os.environ.get(
        "DOJ_BASE_URL", "https://www.justice.gov/epstein"
    )
    JMAIL_BASE_URL = os.environ.get("JMAIL_BASE_URL", "https://jmail.world")

    # ── Search & UI ───────────────────────────────────────────
    RESULTS_PER_PAGE = _int("RESULTS_PER_PAGE", 25)

    # ── Background worker ─────────────────────────────────────
    WORKER_ANALYSIS_DELAY = _float("WORKER_ANALYSIS_DELAY", 2)
    WORKER_FETCH_DELAY = _float("WORKER_FETCH_DELAY", 1)
    WORKER_FETCH_BATCH = _int("WORKER_FETCH_BATCH", 20)
    CATEGORY_DISCOVERY_BATCH = _int("CATEGORY_DISCOVERY_BATCH", 15)
    BULK_IMPORT_MAX_DOCS = _int("BULK_IMPORT_MAX_DOCS", 5000)

    # ── Deployment (read by Dockerfile / docker-compose) ──────
    GUNICORN_WORKERS = _int("GUNICORN_WORKERS", 4)
    GUNICORN_TIMEOUT = _int("GUNICORN_TIMEOUT", 120)

    # ── Celery (optional, for future use) ─────────────────────
    CELERY_BROKER_URL = os.environ.get(
        "CELERY_BROKER_URL", "redis://localhost:6379/0"
    )
    CELERY_RESULT_BACKEND = os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )

    # ── NLP reference data ────────────────────────────────────
    RELEVANCE_TOPICS = [
        "trafficking", "sexual abuse", "child exploitation", "prostitution",
        "blackmail", "intelligence services", "financial crime",
        "money laundering", "corruption", "coercion", "underage",
        "minor", "victim",
    ]

    KNOWN_ASSOCIATES = [
        "Ghislaine Maxwell", "Jean-Luc Brunel", "Sarah Kellen",
        "Nadia Marcinkova", "Lesley Groff", "Adriana Ross",
        "Les Wexner", "Alan Dershowitz", "Prince Andrew",
        "Bill Clinton", "Donald Trump", "Bill Gates",
        "Leon Black", "Glenn Dubin", "Virginia Giuffre",
        "Lex Wexner", "Eva Dubin", "Harvey Weinstein",
    ]

    LOCATIONS = [
        "Little St. James", "Great St. James", "Zorro Ranch",
        "71st Street", "New York", "Palm Beach", "Paris",
        "Virgin Islands", "New Mexico",
    ]
