# Epstein Files Search

AI-powered search tool for the Epstein files, making it easier to find relevant information across thousands of documents, emails, photos, and records released by the DOJ, House Oversight Committee, and other sources.

## What It Does

- **Fuzzy, misspelling-tolerant search** — searching for "Lesly Goff" finds "Lesley Groff"
- **AI document analysis** — a local LLM (via [Ollama](https://ollama.ai)) reads every document, scores it for relevance, extracts entities, and discovers new categories automatically
- **Relevance scoring** — AI classifies documents by topic (trafficking, blackmail, financial crime, intelligence, etc.) and deprioritises irrelevant content (e.g. music newsletters)
- **Named Entity Recognition** — extracts people, organisations, locations, and their roles (e.g. "Virginia Giuffre — Victim", "Prince Andrew — Participant")
- **Entity relationship graph** — 1,000+ connections from the archive (Epstein-Maxwell strength 421, Prince Andrew 50, etc.)
- **Email threading & deduplication** — groups emails into conversations in chronological order; removes duplicates
- **Context linking** — shows related documents, timeline neighbours, and shared entities for every document
- **Multiple search modes** — by name, by date, by category, full-text, or random interesting documents
- **Live search fallback** — queries the [Epstein Document Archive](https://www.epsteininvestigation.org) API (207K+ documents) when local database is empty
- **Age verification gate** — as required for this content

## Data Sources

- [epsteininvestigation.org](https://www.epsteininvestigation.org) — 207K+ documents via public API (live search, entity graph, flight logs)
- [justice.gov/epstein](https://www.justice.gov/epstein) — Official DOJ releases
- [jmail.world](https://jmail.world) — Jmail suite (emails, flights, photos, drive)
- [Hugging Face: tensonaut/EPSTEIN_FILES_20K](https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K) — 25,800 OCR'd documents from the Nov 2025 House Oversight Committee release (requires HF auth)
- [DocETL Epstein Email Explorer](https://www.docetl.org/showcase/epstein-email-explorer) — related project analysing 2,322 emails
- Local PDF files you supply

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Install Ollama (for AI analysis)
# Download from https://ollama.ai, then:
ollama pull llama3.1:8b    # or any model — the app auto-detects the best available

# Start the app
./epstein.sh start
```

### Managing the App

The `epstein.sh` script runs the app as a background daemon:

```bash
./epstein.sh start     # Start in the background, logs to data/epstein.log
./epstein.sh stop      # Stop the daemon (and any orphaned processes)
./epstein.sh restart   # Stop then start
./epstein.sh status    # Show PID and uptime, or "Not running"
./epstein.sh log       # Tail the log file (Ctrl-C to stop watching)
```

The app runs on **http://localhost:5555** by default. Set the `PORT` environment variable to change it.

For interactive/debug mode (logs to terminal, auto-reloads on code changes):

```bash
FLASK_DEBUG=1 python run.py
```

### First Run

1. Open http://localhost:5555 and click "Yes" on the age verification
2. Go to **Admin** and click **Import Archive CSVs** — imports 96 entities, 1,000 relationships, and 55 flight records (instant)
3. Click **Import HF Dataset** — bulk-imports document metadata from the archive API
4. Click **Start AI Worker** — begins analysing documents through your local LLM; categories appear on the home page as they're discovered

## AI Analysis

The background AI worker runs while the web UI is live. It:

1. **Imports structured data** — entities, relationships, flight logs from epsteininvestigation.org
2. **Analyses documents** — sends each through a local LLM (Ollama) which returns a relevance score, categories, entities with roles, and a plain-language summary
3. **Discovers new categories** — every 15 documents, asks the LLM to identify themes emerging across the batch
4. **Updates the home page** — new categories appear in real time under "AI-Discovered"

The system prompt encodes what "relevant" means: trafficking, child exploitation, blackmail, intelligence services, financial crime, corruption. It understands that there is no such thing as "child prostitution" and that a music newsletter is not what the public is interested in.

**Hardware requirements:**
- Mac M4 Pro (64 GB): runs `qwen2.5:32b` comfortably (~35s per document)
- Any machine with Ollama: auto-detects the largest available model
- Data centre: set `OLLAMA_MODEL=llama3.1:70b` for higher quality analysis

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

Or use the Admin page buttons to import from the archive API and Hugging Face.

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

For AI analysis on the server, install Ollama separately and pull a model:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:70b    # larger model for data centre hardware
```

### Ingestion on the server

```bash
# Copy PDFs to the data directory
scp -r ./pdfs/ user@server:/opt/epstein-files/data/pdfs/

# Run ingestion inside the container
docker-compose exec web python ingest.py local /app/data/pdfs
docker-compose exec web python ingest.py threads
```

## Search Features

### Fuzzy Name Matching
The system handles misspellings generically — RapidFuzz generates plausible variants for any word (repeated character collapse, phonetic substitutions, transpositions, deletions). "Lesly Goff" finds "Lesley Groff", "geoffrey epstien" finds "Jeffrey Epstein".

### Proximity Search
Multi-word queries match adjacent words only: "nick lees" finds "Nick; Lees" but not "Nick is going to the party for Cathy Lees". Quotes are stripped automatically.

### Relevance Scoring
Every AI-analysed document gets a 0–1 relevance score. A deposition about trafficking scores 0.95; a routine legal letter scores 0.1.

### Entity Relationship Graph
1,000+ connections imported from the archive, queryable via API:
- Epstein ↔ Maxwell: associate (strength 421)
- Epstein ↔ Prince Andrew: social-associate (strength 50)
- Epstein ↔ Donald Trump: social-associate (strength 46)
- Maxwell ↔ Virginia Giuffre: accused-by (strength 36)

### Email Threading
Emails with matching subjects are grouped into chronological threads. Embedded/quoted emails are detected so you see Email A, then Email B (without Email A repeated inside it).

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
GET /api/data/relationships?entity=Epstein
GET /api/data/flights?passenger=Gates
GET /api/worker/status
GET /api/ai/status
POST /api/worker/start
POST /api/worker/stop
POST /api/ingest  {"source": "huggingface"}
POST /api/ingest  {"source": "archive_csvs"}
```

## Architecture

- **Backend**: Flask + SQLAlchemy + SQLite (with FTS5 for full-text search)
- **AI**: Ollama (local LLM) for document analysis, category discovery, entity extraction
- **NLP**: spaCy (NER), RapidFuzz (fuzzy matching), scikit-learn (TF-IDF topic discovery)
- **Data**: epsteininvestigation.org API, Hugging Face datasets, DOJ PDFs
- **Deployment**: Docker + Gunicorn, runs on Ubuntu 16.04+

## License

This tool is for research and public accountability purposes. All indexed documents are public records.
