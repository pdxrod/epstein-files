# Epstein Files Search

AI-powered search tool for the Epstein files, making it easier to find relevant information across thousands of documents, emails, photos, and records released by the DOJ, House Oversight Committee, and other sources.

## What It Does

- **Fuzzy, misspelling-tolerant search** — searching for "Lesly Goff" finds "Lesley Groff"
- **Relevance scoring** — AI classifies documents by topic (trafficking, blackmail, financial crime, intelligence, etc.) and deprioritises irrelevant content (e.g. music newsletters)
- **Named Entity Recognition** — automatically extracts people, organisations, locations, and dates
- **Email threading & deduplication** — groups emails into conversations in chronological order; removes duplicates
- **Context linking** — shows related documents, timeline neighbours, and shared entities for every document
- **Multiple search modes** — by name, by date, by category, full-text, or random interesting documents
- **Age verification gate** — as required for this content

## Data Sources

- [justice.gov/epstein](https://www.justice.gov/epstein) — Official DOJ releases
- [jmail.world](https://jmail.world) — Jmail suite (processed archive)
- Local PDF files you supply

## Quick Start (Local Development)

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the app
python run.py

# Open http://localhost:5555
```

## Ingesting Documents

```bash
# Ingest PDFs from a local directory
python ingest.py local /path/to/pdf/folder

# Ingest from DOJ website (downloads PDFs automatically)
python ingest.py doj

# Build email threads after ingestion
python ingest.py threads

# Check stats
python ingest.py stats
```

## Deploying to Ubuntu (Docker)

The app runs in Docker, which works on Ubuntu 16.04 and above.

```bash
# On your server:
git clone <this-repo> /opt/epstein-files
cd /opt/epstein-files
sudo bash scripts/setup.sh
```

This will:
1. Install Docker and Docker Compose if not present
2. Build the Docker image (Python 3.11 with all ML dependencies)
3. Start the application on port 5555

### Ingestion on the server

```bash
# Copy PDFs to the data directory
scp -r ./pdfs/ user@server:/opt/epstein-files/data/pdfs/

# Run ingestion inside the container
docker-compose exec web python ingest.py local /app/data/pdfs
docker-compose exec web python ingest.py threads
```

### Manual commands

```bash
docker-compose logs -f web          # View logs
docker-compose exec web python ingest.py stats  # Statistics
docker-compose down                 # Stop
docker-compose up -d                # Start
```

## Search Features

### Fuzzy Name Matching
The system maintains a map of known name variants and misspellings. RapidFuzz handles similarity matching, so "Lesly Goff" → "Lesley Groff", "Ghislane Maxwell" → "Ghislaine Maxwell", etc.

### Relevance Scoring
Every document gets a 0–1 relevance score based on keyword density across categories of public interest:
- Child exploitation
- Sexual abuse / trafficking
- Blackmail and coercion
- Intelligence services
- Financial crime
- Corruption and cover-ups

Irrelevant content (marketing emails, music newsletters, spam) is automatically deprioritised.

### Email Threading
Emails with matching subjects are grouped into chronological threads. Embedded/quoted emails are detected so you see Email A, then Email B (without Email A repeated inside it).

### Context for Every Document
Each document page shows:
- The email thread it belongs to (in chronological order)
- Other documents mentioning the same people or entities
- A ±30 day timeline of neighbouring documents
- All extracted entities and categories

## API

All search functionality is also available via JSON API:

```
GET /api/search?q=trafficking
GET /api/search/name?q=Lesley+Groff
GET /api/search/date?year=2019&month=6
GET /api/search/category/trafficking
GET /api/document/123
GET /api/random
GET /api/categories
GET /api/entities?type=PERSON
GET /api/timeline
GET /api/stats
```

## Architecture

- **Backend**: Flask + SQLAlchemy + SQLite (with FTS5 for full-text search)
- **NLP**: spaCy (NER), RapidFuzz (fuzzy matching), scikit-learn (TF-IDF topic discovery)
- **Scraping**: requests + BeautifulSoup + pdfplumber/PyPDF2
- **Deployment**: Docker + Gunicorn, runs on Ubuntu 16.04+

## License

This tool is for research and public accountability purposes. All indexed documents are public records.
