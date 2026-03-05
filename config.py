import logging
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger(__name__)


def _int(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"Environment variable {key}={val!r} is not a valid integer")


def _float(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"Environment variable {key}={val!r} is not a valid float")


class Config:
    # ── Server ────────────────────────────────────────────────
    PORT = _int("PORT", 5555)

    _secret = os.environ.get("SECRET_KEY")
    if not _secret:
        logger.warning(
            "SECRET_KEY is not set — using a random key. "
            "All sessions will be lost on restart. "
            "Set SECRET_KEY in your environment for persistent sessions."
        )
        _secret = os.urandom(32).hex()
    SECRET_KEY = _secret

    # ── Database ──────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'epstein.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Allow SQLite to be used from multiple threads (background worker + web requests)
    SQLALCHEMY_ENGINE_OPTIONS = {"connect_args": {"check_same_thread": False}}
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")

    # ── Ollama / AI ───────────────────────────────────────────
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "")  # empty = auto-detect
    OLLAMA_TIMEOUT = _int("OLLAMA_TIMEOUT", 120)

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
