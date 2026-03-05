import logging
import os
import sqlite3

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import event
from sqlalchemy.engine import Engine
from config import Config

db = SQLAlchemy()
migrate = Migrate()


logger = logging.getLogger(__name__)


@event.listens_for(Engine, "connect")
def _set_sqlite_wal(dbapi_connection, connection_record):
    """Enable WAL + synchronous=NORMAL for SQLite: concurrent reads/writes, safe and fast."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    _configure_logging()

    os.makedirs(app.config["DATA_DIR"], exist_ok=True)
    os.makedirs(app.config["PDF_DIR"], exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)

    from app.routes import main_bp, api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    with app.app_context():
        db.create_all()
        _ensure_fts_tables(db)

    return app


def _configure_logging():
    """Suppress noisy third-party loggers; set up a consistent format for app logs."""
    for noisy in ("werkzeug", "alembic", "datasets", "requests", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    app_log = logging.getLogger("app")
    if not app_log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        app_log.addHandler(handler)
    app_log.setLevel(logging.INFO)


def _ensure_fts_tables(db):
    """Create FTS5 virtual tables for full-text search if they don't exist."""
    try:
        db.session.execute(
            db.text(
                """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                title, body, sender, recipients, subject,
                content='documents', content_rowid='id',
                tokenize='porter unicode61'
            )
        """
            )
        )
        db.session.execute(
            db.text(
                """
            CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                name, entity_type,
                content='entities', content_rowid='id',
                tokenize='porter unicode61'
            )
        """
            )
        )
        db.session.commit()
    except Exception as e:
        logger.error(f"Failed to create FTS tables: {e}", exc_info=True)
        db.session.rollback()
