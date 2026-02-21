import datetime
from app import db

document_entities = db.Table(
    "document_entities",
    db.Column("document_id", db.Integer, db.ForeignKey("documents.id"), primary_key=True),
    db.Column("entity_id", db.Integer, db.ForeignKey("entities.id"), primary_key=True),
)

document_categories = db.Table(
    "document_categories",
    db.Column("document_id", db.Integer, db.ForeignKey("documents.id"), primary_key=True),
    db.Column("category_id", db.Integer, db.ForeignKey("categories.id"), primary_key=True),
)

document_links = db.Table(
    "document_links",
    db.Column("source_id", db.Integer, db.ForeignKey("documents.id"), primary_key=True),
    db.Column("target_id", db.Integer, db.ForeignKey("documents.id"), primary_key=True),
    db.Column("link_type", db.String(50)),
)


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.String(100), unique=True, nullable=False, index=True)
    title = db.Column(db.String(500))
    body = db.Column(db.Text)
    doc_type = db.Column(db.String(50), index=True)
    source = db.Column(db.String(50), index=True)
    source_url = db.Column(db.String(1000))
    original_url = db.Column(db.String(1000))

    sender = db.Column(db.String(500))
    sender_email = db.Column(db.String(500))
    recipients = db.Column(db.Text)
    subject = db.Column(db.String(1000))

    date = db.Column(db.DateTime, index=True)
    date_str = db.Column(db.String(200))

    page_count = db.Column(db.Integer)
    content_hash = db.Column(db.String(64), index=True)
    is_duplicate = db.Column(db.Boolean, default=False, index=True)
    duplicate_of_id = db.Column(db.Integer, db.ForeignKey("documents.id"), nullable=True)

    thread_id = db.Column(db.Integer, db.ForeignKey("threads.id"), nullable=True)
    thread_position = db.Column(db.Integer)

    relevance_score = db.Column(db.Float, default=0.0, index=True)
    relevance_categories = db.Column(db.Text)

    volume = db.Column(db.String(50))
    folder = db.Column(db.String(200))

    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    processed = db.Column(db.Boolean, default=False, index=True)

    entities = db.relationship("Entity", secondary=document_entities, back_populates="documents")
    categories = db.relationship("Category", secondary=document_categories, back_populates="documents")
    thread = db.relationship("Thread", back_populates="documents")
    duplicate_of = db.relationship("Document", remote_side=[id], foreign_keys=[duplicate_of_id])

    def to_dict(self):
        return {
            "id": self.id,
            "file_id": self.file_id,
            "title": self.title,
            "body": self.body[:500] if self.body else None,
            "doc_type": self.doc_type,
            "source": self.source,
            "source_url": self.source_url,
            "original_url": self.original_url,
            "sender": self.sender,
            "sender_email": self.sender_email,
            "recipients": self.recipients,
            "subject": self.subject,
            "date": self.date.isoformat() if self.date else None,
            "date_str": self.date_str,
            "page_count": self.page_count,
            "relevance_score": self.relevance_score,
            "relevance_categories": self.relevance_categories,
            "volume": self.volume,
            "folder": self.folder,
            "entities": [e.to_dict() for e in self.entities],
            "categories": [c.to_dict() for c in self.categories],
            "thread_id": self.thread_id,
            "is_duplicate": self.is_duplicate,
        }


class Entity(db.Model):
    __tablename__ = "entities"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(500), nullable=False, index=True)
    canonical_name = db.Column(db.String(500), index=True)
    entity_type = db.Column(db.String(50), index=True)
    description = db.Column(db.Text)
    aliases = db.Column(db.Text)
    mention_count = db.Column(db.Integer, default=0)

    documents = db.relationship("Document", secondary=document_entities, back_populates="entities")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "mention_count": self.mention_count,
        }


class Category(db.Model):
    __tablename__ = "categories"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), unique=True, nullable=False)
    slug = db.Column(db.String(200), unique=True, nullable=False, index=True)
    description = db.Column(db.Text)
    is_system = db.Column(db.Boolean, default=False)
    document_count = db.Column(db.Integer, default=0)

    documents = db.relationship("Document", secondary=document_categories, back_populates="categories")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "document_count": self.document_count,
        }


class Thread(db.Model):
    __tablename__ = "threads"

    id = db.Column(db.Integer, primary_key=True)
    subject = db.Column(db.String(1000))
    first_date = db.Column(db.DateTime)
    last_date = db.Column(db.DateTime)
    message_count = db.Column(db.Integer, default=0)
    participants = db.Column(db.Text)

    documents = db.relationship("Document", back_populates="thread", order_by="Document.date")

    def to_dict(self):
        return {
            "id": self.id,
            "subject": self.subject,
            "first_date": self.first_date.isoformat() if self.first_date else None,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "message_count": self.message_count,
            "participants": self.participants,
        }


class NameVariant(db.Model):
    """Maps misspellings and variations to canonical names."""
    __tablename__ = "name_variants"

    id = db.Column(db.Integer, primary_key=True)
    variant = db.Column(db.String(500), nullable=False, index=True)
    canonical = db.Column(db.String(500), nullable=False, index=True)
    similarity_score = db.Column(db.Float)

    __table_args__ = (db.UniqueConstraint("variant", "canonical"),)


class IngestJob(db.Model):
    __tablename__ = "ingest_jobs"

    id = db.Column(db.Integer, primary_key=True)
    source = db.Column(db.String(100))
    url = db.Column(db.String(1000))
    status = db.Column(db.String(50), default="pending")
    documents_found = db.Column(db.Integer, default=0)
    documents_processed = db.Column(db.Integer, default=0)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    error = db.Column(db.Text)
