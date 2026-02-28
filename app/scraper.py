"""
Data ingestion pipeline for Epstein files.

Sources:
1. DOJ (justice.gov/epstein) - Official government releases
2. Jmail (jmail.world) - Processed/structured archive
3. Local PDF files - For offline ingestion
"""

import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from sqlalchemy import and_, or_

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

from app import db
from app.models import Document, Entity, Category, Thread, NameVariant, IngestJob, document_links
from app.nlp import (
    compute_content_hash,
    extract_entities,
    score_relevance,
    parse_email,
    parse_date,
    extract_dates,
    detect_embedded_emails,
    resolve_name,
)

logger = logging.getLogger(__name__)

DOJ_VOLUMES = {
    f"vol{str(i).zfill(5)}": f"https://www.justice.gov/epstein/files/DataSet%20{i}/"
    for i in range(1, 13)
}

SYSTEM_CATEGORIES = [
    ("trafficking", "trafficking", "Documents related to human/sex trafficking"),
    ("sexual-abuse", "Sexual Abuse", "Documents related to sexual abuse and assault"),
    ("child-exploitation", "Child Exploitation", "Documents involving minors"),
    ("blackmail", "Blackmail & Coercion", "Documents suggesting blackmail or coercion"),
    ("intelligence", "Intelligence Services", "Documents linking to intelligence agencies"),
    ("financial-crime", "Financial Crime", "Documents related to financial crimes"),
    ("corruption", "Corruption & Cover-up", "Documents related to obstruction or cover-ups"),
    ("legal", "Legal Proceedings", "Court documents, depositions, testimony"),
    ("flight-logs", "Flight Logs", "Flight records and passenger manifests"),
    ("communications", "Communications", "Emails, letters, and other correspondence"),
    ("photographs", "Photographs", "Photos and images"),
    ("associates", "Associates & Network", "Documents about Epstein's social network"),
    ("properties", "Properties", "Documents about properties and real estate"),
    ("victims", "Victims", "Documents related to victims and survivors"),
]


def ensure_system_categories():
    for slug, name, desc in SYSTEM_CATEGORIES:
        existing = Category.query.filter_by(slug=slug).first()
        if not existing:
            cat = Category(slug=slug, name=name, description=desc, is_system=True)
            db.session.add(cat)
    db.session.commit()


def extract_pdf_text(pdf_path: str) -> str | None:
    """Extract text from a PDF using pdfplumber with PyPDF2 as fallback."""
    text = None
    if pdfplumber:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                text = "\n\n".join(pages)
        except Exception as e:
            logger.warning(f"pdfplumber failed on {pdf_path}: {e}")

    if not text and PdfReader:
        try:
            reader = PdfReader(pdf_path)
            pages = [p.extract_text() for p in reader.pages if p.extract_text()]
            text = "\n\n".join(pages)
        except Exception as e:
            logger.warning(f"PyPDF2 failed on {pdf_path}: {e}")

    return text or None


class DOJScraper:
    """Scrapes documents from justice.gov/epstein."""

    BASE_URL = "https://www.justice.gov/epstein"
    DISCLOSURES_URL = "https://www.justice.gov/epstein/doj-disclosures"

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.pdf_dir = os.path.join(data_dir, "pdfs", "doj")
        os.makedirs(self.pdf_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EpsteinFilesResearch/1.0 (academic research tool)"
        })

    def discover_documents(self) -> list[dict]:
        """Discover document URLs from the DOJ disclosure pages."""
        docs = []
        try:
            resp = self.session.get(self.DISCLOSURES_URL, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if ".pdf" in href.lower():
                    full_url = urljoin(self.DISCLOSURES_URL, href)
                    file_id = self._extract_file_id(href)
                    docs.append({
                        "url": full_url,
                        "file_id": file_id,
                        "source": "doj",
                        "title": link.get_text(strip=True) or file_id,
                    })
        except Exception as e:
            logger.error(f"Error discovering DOJ documents: {e}")

        # Fetch all 12 volumes in parallel (4 workers — polite to the server)
        headers = dict(self.session.headers)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._fetch_volume, vol_name, vol_url, headers): vol_name
                for vol_name, vol_url in DOJ_VOLUMES.items()
            }
            for future in as_completed(futures):
                docs.extend(future.result())

        return docs

    @staticmethod
    def _fetch_volume(vol_name: str, vol_url: str, headers: dict) -> list[dict]:
        """Fetch a single DOJ volume page and return its document list."""
        docs = []
        try:
            resp = requests.get(vol_url, timeout=30, headers=headers)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    if ".pdf" in href.lower():
                        full_url = urljoin(vol_url, href)
                        file_id = DOJScraper._extract_file_id(href)
                        docs.append({
                            "url": full_url,
                            "file_id": file_id,
                            "source": "doj",
                            "volume": vol_name,
                            "title": link.get_text(strip=True) or file_id,
                        })
        except Exception as e:
            logger.warning(f"Error accessing volume {vol_name}: {e}")
        return docs

    def download_pdf(self, url: str, file_id: str) -> str | None:
        """Download a PDF and return local path."""
        safe_name = re.sub(r"[^\w.-]", "_", file_id) + ".pdf"
        local_path = os.path.join(self.pdf_dir, safe_name)
        if os.path.exists(local_path):
            return local_path
        try:
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return None

    def extract_text(self, pdf_path: str) -> str | None:
        """Extract text from a PDF file."""
        return extract_pdf_text(pdf_path)

    @staticmethod
    def _extract_file_id(url: str) -> str:
        match = re.search(r"(EFTA\d+)", url, re.IGNORECASE)
        if match:
            return match.group(1)
        basename = os.path.basename(url).replace(".pdf", "").replace("%20", "_")
        return basename


class JmailScraper:
    """Fetches processed data from jmail.world where available."""

    BASE_URL = "https://jmail.world"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EpsteinFilesResearch/1.0 (academic research tool)"
        })

    def fetch_thread(self, thread_id: str) -> dict | None:
        """Attempt to fetch a thread/document from jmail.world."""
        url = f"{self.BASE_URL}/thread/{thread_id}"
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "lxml")
                return {
                    "url": url,
                    "title": soup.title.string if soup.title else thread_id,
                    "text": soup.get_text(separator="\n", strip=True),
                }
        except Exception as e:
            logger.warning(f"Error fetching jmail thread {thread_id}: {e}")
        return None

    def search(self, query: str) -> list[dict]:
        """Search jmail.world (if API available)."""
        url = f"{self.BASE_URL}/api/search"
        try:
            resp = self.session.get(url, params={"q": query}, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("results", [])
        except Exception:
            pass
        return []


class DocumentProcessor:
    """Processes raw documents through the NLP pipeline and stores results."""

    def __init__(self):
        self._hash_cache = {}

    def load_hash_cache(self):
        docs = Document.query.filter(Document.content_hash.isnot(None)).all()
        self._hash_cache = {d.content_hash: d.id for d in docs}

    def process_document(self, file_id: str, text: str, metadata: dict) -> Document | None:
        """Full processing pipeline for a single document."""
        if not text or len(text.strip()) < 10:
            return None

        existing = Document.query.filter_by(file_id=file_id).first()
        if existing and existing.processed:
            return existing

        content_hash = compute_content_hash(text)
        duplicate_id = self._hash_cache.get(content_hash)
        if duplicate_id and not existing:
            doc = Document(
                file_id=file_id,
                content_hash=content_hash,
                is_duplicate=True,
                duplicate_of_id=duplicate_id,
                processed=True,
            )
            db.session.add(doc)
            db.session.commit()
            return doc

        doc_type = self._classify_type(text, metadata)
        email_data = parse_email(text) if doc_type == "email" else {}
        relevance_score, relevance_cats = score_relevance(text)
        entities = extract_entities(text)
        dates = extract_dates(text)

        if existing:
            doc = existing
        else:
            doc = Document(file_id=file_id)

        doc.title = metadata.get("title") or email_data.get("subject") or file_id
        doc.body = text
        doc.doc_type = doc_type
        doc.source = metadata.get("source", "unknown")
        doc.source_url = metadata.get("source_url")
        doc.original_url = metadata.get("original_url")
        doc.sender = email_data.get("sender") or metadata.get("sender")
        doc.sender_email = email_data.get("sender_email")
        doc.recipients = email_data.get("recipients") or metadata.get("recipients")
        doc.subject = email_data.get("subject") or metadata.get("subject")
        doc.date = email_data.get("date") or (dates[0] if dates else None)
        doc.date_str = email_data.get("date_str") or metadata.get("date_str")
        doc.page_count = metadata.get("page_count")
        doc.content_hash = content_hash
        doc.relevance_score = relevance_score
        doc.relevance_categories = json.dumps(relevance_cats) if relevance_cats else None
        doc.volume = metadata.get("volume")
        doc.folder = metadata.get("folder")
        doc.processed = True

        if not existing:
            db.session.add(doc)
        db.session.flush()

        # Batch-load all entities for this document in a single query
        if entities:
            filters = [
                and_(Entity.name == e["name"], Entity.entity_type == e["entity_type"])
                for e in entities
            ]
            known = Entity.query.filter(or_(*filters)).all()
            entity_map = {(e.name, e.entity_type): e for e in known}
        else:
            entity_map = {}

        for ent_data in entities:
            key = (ent_data["name"], ent_data["entity_type"])
            entity = entity_map.get(key)
            if not entity:
                entity = Entity(
                    name=ent_data["name"],
                    canonical_name=ent_data.get("canonical_name", ent_data["name"]),
                    entity_type=ent_data["entity_type"],
                    mention_count=1,
                )
                db.session.add(entity)
                entity_map[key] = entity
            else:
                entity.mention_count = (entity.mention_count or 0) + 1
            if entity not in doc.entities:
                doc.entities.append(entity)

            if ent_data.get("canonical_name") and ent_data["canonical_name"] != ent_data["name"]:
                _ensure_name_variant(ent_data["name"], ent_data["canonical_name"])

        for cat_slug in relevance_cats:
            cat = Category.query.filter_by(slug=cat_slug.replace("_", "-")).first()
            if cat and cat not in doc.categories:
                doc.categories.append(cat)
                cat.document_count = (cat.document_count or 0) + 1

        self._hash_cache[content_hash] = doc.id

        if doc_type == "email" and email_data.get("quoted_emails"):
            self._handle_embedded_emails(doc, email_data["quoted_emails"])

        db.session.commit()
        self._update_fts(doc)

        return doc

    def _classify_type(self, text: str, metadata: dict) -> str:
        # Explicit metadata takes precedence over text heuristics
        if metadata.get("doc_type"):
            return metadata["doc_type"]

        text_lower = text[:2000].lower()
        if any(
            marker in text_lower
            for marker in ["from:", "to:", "subject:", "sent:", "date:"]
        ):
            if text_lower.count("from:") >= 1 and text_lower.count("to:") >= 1:
                return "email"
        if any(
            marker in text_lower
            for marker in ["deposition", "testimony", "q.", "a.", "the witness"]
        ):
            return "legal"
        if any(
            marker in text_lower
            for marker in ["flight", "passenger", "tail number", "departure", "arrival"]
        ):
            return "flight_log"
        return "document"

    def _handle_embedded_emails(self, parent_doc: Document, quoted_emails: list[str]):
        """Store embedded/quoted emails as linked child documents."""
        for i, quoted_text in enumerate(quoted_emails):
            if not quoted_text or len(quoted_text) < 30:
                continue

            quoted_hash = compute_content_hash(quoted_text)

            if quoted_hash in self._hash_cache:
                # Already stored — just add the link to this parent
                existing_id = self._hash_cache[quoted_hash]
                try:
                    db.session.execute(
                        document_links.insert().prefix_with("OR IGNORE").values(
                            source_id=parent_doc.id,
                            target_id=existing_id,
                            link_type="embedded",
                        )
                    )
                    db.session.commit()
                except Exception:
                    db.session.rollback()
                continue

            child_file_id = f"{parent_doc.file_id}_emb_{i}"
            if Document.query.filter_by(file_id=child_file_id).first():
                continue

            email_data = parse_email(quoted_text)
            relevance_score, relevance_cats = score_relevance(quoted_text)
            dates = extract_dates(quoted_text)

            child = Document(
                file_id=child_file_id,
                body=quoted_text,
                doc_type="email",
                source=parent_doc.source,
                title=email_data.get("subject") or f"Embedded email from {parent_doc.file_id}",
                sender=email_data.get("sender"),
                sender_email=email_data.get("sender_email"),
                recipients=email_data.get("recipients"),
                subject=email_data.get("subject") or parent_doc.subject,
                date=email_data.get("date") or (dates[0] if dates else None),
                date_str=email_data.get("date_str"),
                content_hash=quoted_hash,
                relevance_score=relevance_score,
                relevance_categories=json.dumps(relevance_cats) if relevance_cats else None,
                processed=False,
            )
            db.session.add(child)
            db.session.flush()
            self._hash_cache[quoted_hash] = child.id

            try:
                db.session.execute(
                    document_links.insert().prefix_with("OR IGNORE").values(
                        source_id=parent_doc.id,
                        target_id=child.id,
                        link_type="embedded",
                    )
                )
                db.session.commit()
            except Exception as e:
                logger.warning(f"Failed to store embedded email {child_file_id}: {e}")
                db.session.rollback()

            self._update_fts(child)

    def _update_fts(self, doc: Document):
        """Update the FTS index for a document."""
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
        except Exception as e:
            logger.warning(f"FTS update failed for doc {doc.id}: {e}")
            db.session.rollback()


class ThreadBuilder:
    """Builds email threads from processed documents."""

    def build_threads(self):
        """Group emails by subject and participants into threads."""
        emails = (
            Document.query.filter_by(doc_type="email", is_duplicate=False)
            .filter(Document.subject.isnot(None))
            .order_by(Document.date.asc())
            .all()
        )

        subject_groups = {}
        for email in emails:
            normalised = self._normalise_subject(email.subject)
            if normalised not in subject_groups:
                subject_groups[normalised] = []
            subject_groups[normalised].append(email)

        for subject, group in subject_groups.items():
            if len(group) < 2:
                continue

            participants = set()
            for email in group:
                if email.sender:
                    participants.add(email.sender)
                if email.recipients:
                    for r in email.recipients.split(","):
                        participants.add(r.strip())

            thread = Thread(
                subject=group[0].subject,
                first_date=group[0].date,
                last_date=group[-1].date,
                message_count=len(group),
                participants="; ".join(sorted(participants)),
            )
            db.session.add(thread)
            db.session.flush()

            for pos, email in enumerate(group):
                email.thread_id = thread.id
                email.thread_position = pos

        db.session.commit()

    @staticmethod
    def _normalise_subject(subject: str) -> str:
        cleaned = re.sub(r"^(re|fw|fwd)\s*:\s*", "", subject.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned.lower().strip())
        return cleaned


def _ensure_name_variant(variant: str, canonical: str):
    existing = NameVariant.query.filter_by(variant=variant, canonical=canonical).first()
    if not existing:
        nv = NameVariant(
            variant=variant,
            canonical=canonical,
            similarity_score=1.0,
        )
        db.session.add(nv)


def ingest_local_pdfs(directory: str, source: str = "local"):
    """Ingest PDF files from a local directory."""
    processor = DocumentProcessor()
    processor.load_hash_cache()
    ensure_system_categories()

    pdf_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

    count = 0
    for pdf_path in pdf_files:
        file_id = DOJScraper._extract_file_id(pdf_path)
        text = extract_pdf_text(pdf_path)
        if text:
            metadata = {
                "source": source,
                "source_url": pdf_path,
                "title": os.path.basename(pdf_path),
                "folder": os.path.basename(os.path.dirname(pdf_path)),
            }
            doc = processor.process_document(file_id, text, metadata)
            if doc:
                count += 1
                if count % 50 == 0:
                    logger.info(f"Processed {count} documents...")

    logger.info(f"Ingestion complete. Processed {count} documents.")
    return count


def ingest_from_doj():
    """Run full DOJ ingestion pipeline."""
    from flask import current_app

    data_dir = current_app.config["DATA_DIR"]
    scraper = DOJScraper(data_dir)
    processor = DocumentProcessor()
    processor.load_hash_cache()
    ensure_system_categories()

    job = IngestJob(source="doj", url=DOJScraper.DISCLOSURES_URL, started_at=datetime.utcnow())
    db.session.add(job)
    db.session.commit()

    try:
        docs = scraper.discover_documents()
        job.documents_found = len(docs)
        job.status = "downloading"
        db.session.commit()

        logger.info(f"Discovered {len(docs)} documents from DOJ")

        for i, doc_meta in enumerate(docs):
            pdf_path = scraper.download_pdf(doc_meta["url"], doc_meta["file_id"])
            if pdf_path:
                text = extract_pdf_text(pdf_path)
                if text:
                    metadata = {
                        "source": "doj",
                        "source_url": doc_meta["url"],
                        "original_url": doc_meta["url"],
                        "title": doc_meta.get("title"),
                        "volume": doc_meta.get("volume"),
                    }
                    processor.process_document(doc_meta["file_id"], text, metadata)
                    job.documents_processed = (job.documents_processed or 0) + 1

            if i % 10 == 0:
                db.session.commit()
                logger.info(f"Progress: {i+1}/{len(docs)}")
            time.sleep(0.5)

        thread_builder = ThreadBuilder()
        thread_builder.build_threads()

        job.status = "completed"
        job.completed_at = datetime.utcnow()

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        logger.error(f"DOJ ingestion failed: {e}")

    db.session.commit()
    return job


def ingest_text_content(file_id: str, text: str, metadata: dict) -> Document | None:
    """Ingest a single piece of text content (useful for API-driven ingestion)."""
    processor = DocumentProcessor()
    processor.load_hash_cache()
    ensure_system_categories()
    return processor.process_document(file_id, text, metadata)
