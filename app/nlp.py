"""
NLP and ML pipeline for document processing.

Handles:
- Named Entity Recognition (NER)
- Topic classification and relevance scoring
- Fuzzy name matching (misspelling tolerance)
- Email parsing, threading, and deduplication
- Date extraction and normalisation
"""

import hashlib
import re
from datetime import datetime

import dateparser
import numpy as np
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy

    _nlp = None

    def _get_nlp():
        global _nlp
        if _nlp is None:
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess, sys

                subprocess.check_call(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
                )
                _nlp = spacy.load("en_core_web_sm")
        return _nlp

except ImportError:
    _get_nlp = None

RELEVANCE_KEYWORDS = {
    "trafficking": [
        "traffick", "trafficking", "trafficked", "trafick", "traficking",
        "traficked", "sex trade", "human trafficking", "smuggling",
    ],
    "sexual_abuse": [
        "abuse", "abused", "assault", "assaulted", "rape", "raped",
        "molest", "molestation", "sexual", "victim", "survivor",
        "inappropriate", "forced", "coerced", "unwanted",
    ],
    "child_exploitation": [
        "minor", "minors", "underage", "under age", "child", "children",
        "juvenile", "teenager", "teen", "young girl", "young boy",
        "high school", "school girl", "16", "15", "14", "13", "12",
    ],
    "blackmail_coercion": [
        "blackmail", "blackmailed", "extort", "extortion", "leverage",
        "compromise", "compromising", "tape", "tapes", "recording",
        "recordings", "video", "hidden camera", "surveillance",
        "threaten", "threatened", "pressure", "pressured",
    ],
    "intelligence": [
        "intelligence", "cia", "fbi", "mossad", "mi5", "mi6",
        "government", "agent", "operative", "classified", "secret",
        "national security", "spy", "espionage", "informant",
        "wexner", "mega group", "les wexner",
    ],
    "financial_crime": [
        "money laundering", "fraud", "embezzle", "embezzlement",
        "offshore", "shell company", "tax evasion", "wire transfer",
        "suspicious", "account", "trust", "foundation", "donation",
        "payment", "millions", "billions", "financial",
    ],
    "corruption": [
        "cover up", "coverup", "cover-up", "obstruct", "obstruction",
        "destroy", "destroyed", "shred", "shredded", "delete",
        "deleted", "bribe", "bribery", "corrupt", "corruption",
        "plea deal", "sweetheart", "non-prosecution",
    ],
    "associates_network": [
        "maxwell", "ghislaine", "brunel", "jean-luc", "kellen",
        "sarah kellen", "marcinkova", "nadia", "groff", "lesley",
        "adriana ross", "haley robson",
    ],
    "locations": [
        "little st james", "little saint james", "great st james",
        "zorro ranch", "71st street", "palm beach", "pedophile island",
        "orgy island", "island", "new mexico ranch",
    ],
    "legal_proceedings": [
        "deposition", "testimony", "sworn", "affidavit", "subpoena",
        "court", "trial", "prosecution", "indictment", "plea",
        "sentence", "prison", "jail", "arrest", "investigation",
    ],
}

IRRELEVANT_PATTERNS = [
    r"hdtracks", r"audiophile", r"hi.?res", r"vinyl", r"album",
    r"newsletter.{0,20}unsubscribe", r"marketing.{0,20}email",
    r"spotify", r"itunes", r"concert tickets",
    r"amazon\.com.{0,50}order", r"shipping confirmation",
    r"your order has shipped",
]

_irrelevant_re = re.compile("|".join(IRRELEVANT_PATTERNS), re.IGNORECASE)

EMAIL_HEADER_RE = re.compile(
    r"(?:From|De)\s*:\s*(.+?)(?:\n|$)"
    r"|(?:To|À)\s*:\s*(.+?)(?:\n|$)"
    r"|(?:Date|Sent)\s*:\s*(.+?)(?:\n|$)"
    r"|Subject\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.MULTILINE,
)

QUOTED_EMAIL_RE = re.compile(
    r"(?:"
    r"-{3,}\s*(?:Original Message|Forwarded|Begin forwarded)\s*-{3,}"
    r"|On\s+.{10,60}\s+wrote:"
    r"|>{2,}"
    r"|_{5,}"
    r")",
    re.IGNORECASE,
)

DATE_PATTERNS = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{1,2},?\s+\d{4})\b",
    r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\.?\s+\d{4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
]

_date_re = re.compile("|".join(DATE_PATTERNS), re.IGNORECASE)

KNOWN_NAMES = [
    "Jeffrey Epstein", "Ghislaine Maxwell", "Jean-Luc Brunel",
    "Sarah Kellen", "Nadia Marcinkova", "Lesley Groff",
    "Adriana Ross", "Les Wexner", "Alan Dershowitz",
    "Prince Andrew", "Bill Clinton", "Donald Trump",
    "Bill Gates", "Leon Black", "Glenn Dubin",
    "Virginia Giuffre", "Virginia Roberts", "Eva Dubin",
    "Harvey Weinstein", "Haley Robson", "Lex Wexner",
    "Leslie Wexner", "Lesly Groff", "Leslie Groff",
    "Nada Marcinkova", "Nadia Marcinko",
    "Johnathan Epstein", "Jonathan Farkas",
]


def compute_content_hash(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def extract_entities(text: str) -> list[dict]:
    """Extract named entities using spaCy and known-name fuzzy matching."""
    entities = []
    seen = set()

    if _get_nlp is not None:
        nlp = _get_nlp()
        doc = nlp(text[:100000])  # spaCy can be slow on very long texts
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE", "NORP"):
                key = (ent.text.lower(), ent.label_)
                if key not in seen:
                    seen.add(key)
                    canonical = resolve_name(ent.text) if ent.label_ == "PERSON" else ent.text
                    entities.append({
                        "name": ent.text,
                        "canonical_name": canonical,
                        "entity_type": ent.label_,
                    })

    for name in KNOWN_NAMES:
        if name.lower() in text.lower():
            canonical = resolve_name(name)
            key = (canonical.lower(), "PERSON")
            if key not in seen:
                seen.add(key)
                entities.append({
                    "name": name,
                    "canonical_name": canonical,
                    "entity_type": "PERSON",
                })

    return entities


def resolve_name(name: str, threshold: int = 80) -> str:
    """Resolve a possibly misspelled name to its canonical form."""
    canonical_map = {
        "Jeffrey Epstein": ["Jeff Epstein", "Jeffery Epstein", "J Epstein", "JE",
                            "Johnathan Epstein", "Jeffry Epstein"],
        "Ghislaine Maxwell": ["Gislaine Maxwell", "Ghislane Maxwell", "Ghislain Maxwell",
                              "G Maxwell", "GM"],
        "Lesley Groff": ["Lesly Groff", "Leslie Groff", "Lesly Goff", "Lesley Goff",
                         "Lesely Groff"],
        "Jean-Luc Brunel": ["Jean Luc Brunel", "JL Brunel", "Jean-Luc Brunell",
                            "Jean Luck Brunel"],
        "Sarah Kellen": ["Sara Kellen", "Sarah Kelen", "Sara Kelen", "Sarah Kellan"],
        "Nadia Marcinkova": ["Nada Marcinkova", "Nadia Marcinko", "Nadia Marcinková",
                             "Nadia Marcincova"],
        "Virginia Giuffre": ["Virginia Roberts", "Virginia Roberts Giuffre",
                             "Virginia Guiffre", "Virginia Giufre"],
        "Les Wexner": ["Lex Wexner", "Leslie Wexner", "Les Wexnor", "L Wexner"],
        "Alan Dershowitz": ["Allen Dershowitz", "Alan Dershawitz", "Alan Dershowiz",
                            "A Dershowitz"],
        "Prince Andrew": ["Andrew Windsor", "Duke of York", "Prince Andrew Duke of York"],
        "Leon Black": ["Leon Blank", "L Black"],
        "Glenn Dubin": ["Glen Dubin", "Glenn Duben", "G Dubin"],
    }
    name_clean = name.strip()

    for canonical, variants in canonical_map.items():
        if name_clean.lower() == canonical.lower():
            return canonical
        for variant in variants:
            if name_clean.lower() == variant.lower():
                return canonical
            score = fuzz.ratio(name_clean.lower(), variant.lower())
            if score >= threshold:
                return canonical

    all_canonical = list(canonical_map.keys())
    match = process.extractOne(name_clean, all_canonical, scorer=fuzz.WRatio)
    if match and match[1] >= threshold:
        return match[0]

    return name_clean


def score_relevance(text: str) -> tuple[float, list[str]]:
    """
    Score document relevance to topics of public interest.
    Returns (score 0-1, list of matched categories).
    """
    if not text:
        return 0.0, []

    text_lower = text.lower()

    if _irrelevant_re.search(text_lower):
        irrelevant_density = len(_irrelevant_re.findall(text_lower)) / max(len(text_lower.split()), 1)
        if irrelevant_density > 0.05:
            return 0.0, []

    matched_categories = []
    total_score = 0.0

    weights = {
        "child_exploitation": 1.0,
        "sexual_abuse": 0.95,
        "trafficking": 0.95,
        "blackmail_coercion": 0.9,
        "intelligence": 0.85,
        "corruption": 0.85,
        "financial_crime": 0.8,
        "associates_network": 0.7,
        "locations": 0.6,
        "legal_proceedings": 0.5,
    }

    for category, keywords in RELEVANCE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits > 0:
            density = hits / len(keywords)
            weight = weights.get(category, 0.5)
            cat_score = min(density * weight * 2, weight)
            total_score += cat_score
            matched_categories.append(category)

    final_score = min(total_score / len(RELEVANCE_KEYWORDS), 1.0)
    return round(final_score, 4), matched_categories


def parse_email(text: str) -> dict:
    """Parse email text to extract headers, body, and quoted content."""
    result = {
        "sender": None,
        "sender_email": None,
        "recipients": None,
        "subject": None,
        "date": None,
        "date_str": None,
        "body": text,
        "quoted_emails": [],
    }

    from_match = re.search(r"(?:From|De)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if from_match:
        sender_raw = from_match.group(1).strip()
        email_match = re.search(r"[\w.+-]+@[\w.-]+", sender_raw)
        if email_match:
            result["sender_email"] = email_match.group(0)
            result["sender"] = re.sub(r"<.*?>", "", sender_raw).strip() or email_match.group(0)
        else:
            result["sender"] = sender_raw

    to_match = re.search(r"(?:To|À)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if to_match:
        result["recipients"] = to_match.group(1).strip()

    subj_match = re.search(r"Subject\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if subj_match:
        result["subject"] = subj_match.group(1).strip()

    date_match = re.search(r"(?:Date|Sent)\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if date_match:
        result["date_str"] = date_match.group(1).strip()
        result["date"] = parse_date(result["date_str"])

    parts = QUOTED_EMAIL_RE.split(text)
    if len(parts) > 1:
        result["body"] = parts[0].strip()
        for i in range(1, len(parts)):
            quoted = parts[i].strip()
            if len(quoted) > 20:
                result["quoted_emails"].append(quoted)

    return result


def parse_date(text: str) -> datetime | None:
    if not text:
        return None
    try:
        return dateparser.parse(
            text,
            settings={
                "PREFER_DAY_OF_MONTH": "first",
                "PREFER_DATES_FROM": "past",
            },
        )
    except Exception:
        return None


def extract_dates(text: str) -> list[datetime]:
    dates = []
    for match in _date_re.finditer(text):
        date_str = next(g for g in match.groups() if g)
        dt = parse_date(date_str)
        if dt and 1990 <= dt.year <= 2025:
            dates.append(dt)
    return sorted(set(dates))


def find_duplicate(text: str, existing_hashes: dict) -> int | None:
    """Check if document is a duplicate by content hash."""
    content_hash = compute_content_hash(text)
    return existing_hashes.get(content_hash)


def detect_embedded_emails(text: str) -> list[str]:
    """Detect if an email contains earlier emails embedded within it."""
    markers = list(QUOTED_EMAIL_RE.finditer(text))
    segments = []
    if markers:
        last_end = 0
        for m in markers:
            segment = text[last_end : m.start()].strip()
            if len(segment) > 50:
                segments.append(segment)
            last_end = m.end()
        trailing = text[last_end:].strip()
        if len(trailing) > 50:
            segments.append(trailing)
    return segments


class FuzzySearcher:
    """Misspelling-tolerant, context-aware searcher."""

    def __init__(self):
        self._name_index = {}
        self._tfidf = TfidfVectorizer(
            max_features=50000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._tfidf_matrix = None
        self._doc_ids = []

    def build_name_index(self, entities: list[dict]):
        for ent in entities:
            canonical = ent.get("canonical_name", ent["name"])
            name_lower = ent["name"].lower()
            self._name_index[name_lower] = canonical
            if canonical.lower() != name_lower:
                self._name_index[canonical.lower()] = canonical

    def search_name(self, query: str, threshold: int = 65) -> list[tuple[str, float]]:
        """Fuzzy search for a name, returning matches with scores."""
        results = process.extract(
            query.lower(),
            list(self._name_index.keys()),
            scorer=fuzz.WRatio,
            limit=20,
        )
        matches = []
        seen = set()
        for name, score, _ in results:
            if score >= threshold:
                canonical = self._name_index[name]
                if canonical not in seen:
                    seen.add(canonical)
                    matches.append((canonical, score))
        return sorted(matches, key=lambda x: x[1], reverse=True)

    def build_tfidf_index(self, documents: list[tuple[int, str]]):
        """Build TF-IDF index from (doc_id, text) pairs."""
        self._doc_ids = [d[0] for d in documents]
        texts = [d[1] for d in documents]
        if texts:
            self._tfidf_matrix = self._tfidf.fit_transform(texts)

    def search_semantic(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """Search by TF-IDF similarity."""
        if self._tfidf_matrix is None:
            return []
        query_vec = self._tfidf.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            (self._doc_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.01
        ]


class TopicDiscoverer:
    """Discovers document topics beyond the predefined categories."""

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
        )

    def discover_topics(self, documents: list[str], n_topics: int = 30) -> list[dict]:
        """Use TF-IDF to find prominent topics/phrases across documents."""
        if len(documents) < 2:
            return []

        tfidf_matrix = self._vectorizer.fit_transform(documents)
        feature_names = self._vectorizer.get_feature_names_out()

        mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = np.argsort(mean_tfidf)[-n_topics * 3:][::-1]

        skip_words = {
            "said", "would", "also", "one", "two", "new", "like",
            "time", "just", "know", "year", "page", "file", "document",
            "email", "sent", "received", "com", "gmail", "yahoo",
            "http", "https", "www",
        }

        topics = []
        for idx in top_indices:
            phrase = feature_names[idx]
            if phrase.lower() not in skip_words and len(phrase) > 2:
                topics.append({
                    "phrase": phrase,
                    "score": float(mean_tfidf[idx]),
                })
            if len(topics) >= n_topics:
                break

        return topics
