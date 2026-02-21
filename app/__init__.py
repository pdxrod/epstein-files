from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config

db = SQLAlchemy()
migrate = Migrate()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    db.init_app(app)
    migrate.init_app(app, db)

    from app.routes import main_bp, api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    with app.app_context():
        db.create_all()
        _ensure_fts_tables(db)

    return app


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
    except Exception:
        db.session.rollback()
