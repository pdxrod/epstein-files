#!/usr/bin/env python3
"""
CLI for ingesting Epstein files into the search database.

Usage:
    # Ingest PDFs from a local directory
    python ingest.py local /path/to/pdfs

    # Ingest from DOJ website
    python ingest.py doj

    # Build email threads after ingestion
    python ingest.py threads

    # Show stats
    python ingest.py stats

    # Ingest a single text file or paste
    python ingest.py text --file-id EFTA00001234 --file /path/to/file.txt
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Epstein Files ingestion tool")
    subparsers = parser.add_subparsers(dest="command")

    local_parser = subparsers.add_parser("local", help="Ingest PDFs from a local directory")
    local_parser.add_argument("directory", help="Path to directory containing PDF files")
    local_parser.add_argument("--source", default="local", help="Source label")

    subparsers.add_parser("doj", help="Ingest from DOJ website")

    subparsers.add_parser("threads", help="Build email threads from processed documents")

    subparsers.add_parser("stats", help="Show database statistics")

    text_parser = subparsers.add_parser("text", help="Ingest a single text file")
    text_parser.add_argument("--file-id", required=True, help="Unique file identifier")
    text_parser.add_argument("--file", required=True, help="Path to text file")
    text_parser.add_argument("--source", default="manual", help="Source label")

    subparsers.add_parser("categories", help="Ensure system categories exist")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    from app import create_app

    app = create_app()

    with app.app_context():
        if args.command == "local":
            from app.scraper import ingest_local_pdfs

            count = ingest_local_pdfs(args.directory, source=args.source)
            logger.info(f"Done. Processed {count} documents.")

        elif args.command == "doj":
            from app.scraper import ingest_from_doj

            job = ingest_from_doj()
            logger.info(
                f"DOJ ingestion {job.status}. "
                f"Found: {job.documents_found}, Processed: {job.documents_processed}"
            )

        elif args.command == "threads":
            from app.scraper import ThreadBuilder

            builder = ThreadBuilder()
            builder.build_threads()
            logger.info("Thread building complete.")

        elif args.command == "stats":
            from app.search import get_stats

            stats = get_stats()
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

        elif args.command == "text":
            with open(args.file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            from app.scraper import ingest_text_content

            doc = ingest_text_content(
                args.file_id, text, {"source": args.source}
            )
            if doc:
                logger.info(f"Ingested: {doc.file_id} (relevance: {doc.relevance_score})")
            else:
                logger.warning("Document could not be processed.")

        elif args.command == "categories":
            from app.scraper import ensure_system_categories

            ensure_system_categories()
            logger.info("System categories created.")


if __name__ == "__main__":
    main()
