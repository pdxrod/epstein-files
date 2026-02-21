import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(32).hex())
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'epstein.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PDF_DIR = os.path.join(DATA_DIR, "pdfs")
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/0"
    )
    RESULTS_PER_PAGE = 25
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    DOJ_BASE_URL = "https://www.justice.gov/epstein"
    JMAIL_BASE_URL = "https://jmail.world"

    RELEVANCE_TOPICS = [
        "trafficking",
        "sexual abuse",
        "child exploitation",
        "prostitution",
        "blackmail",
        "intelligence services",
        "financial crime",
        "money laundering",
        "corruption",
        "coercion",
        "underage",
        "minor",
        "victim",
    ]

    KNOWN_ASSOCIATES = [
        "Ghislaine Maxwell",
        "Jean-Luc Brunel",
        "Sarah Kellen",
        "Nadia Marcinkova",
        "Lesley Groff",
        "Adriana Ross",
        "Les Wexner",
        "Alan Dershowitz",
        "Prince Andrew",
        "Bill Clinton",
        "Donald Trump",
        "Bill Gates",
        "Leon Black",
        "Glenn Dubin",
        "Virginia Giuffre",
        "Lex Wexner",
        "Eva Dubin",
        "Harvey Weinstein",
    ]

    LOCATIONS = [
        "Little St. James",
        "Great St. James",
        "Zorro Ranch",
        "71st Street",
        "New York",
        "Palm Beach",
        "Paris",
        "Virgin Islands",
        "New Mexico",
    ]
