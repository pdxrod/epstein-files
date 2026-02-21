FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libsqlite3-dev \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY . .

RUN mkdir -p data/pdfs

ENV PORT=5555 \
    GUNICORN_WORKERS=4 \
    GUNICORN_TIMEOUT=120

EXPOSE ${PORT}

CMD gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers "${GUNICORN_WORKERS}" \
    --timeout "${GUNICORN_TIMEOUT}" \
    run:app
