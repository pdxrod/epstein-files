#!/bin/bash
#
# Setup script for deploying Epstein Files Search on Ubuntu 16.04+
# Uses Docker for isolation (avoids ancient system Python).
#
set -euo pipefail

echo "=== Epstein Files Search — Setup ==="

# Detect if Docker is installed
if ! command -v docker &>/dev/null; then
    echo "[*] Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh
    systemctl enable docker
    systemctl start docker
    echo "[+] Docker installed."
fi

if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null 2>&1; then
    echo "[*] Installing Docker Compose..."
    COMPOSE_VERSION="2.24.5"
    curl -fsSL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "[+] Docker Compose installed."
fi

# Generate secret key if not set
if [ ! -f .env ]; then
    echo "SECRET_KEY=$(openssl rand -hex 32)" > .env
    echo "[+] Generated .env with secret key"
fi

# Create data directory
mkdir -p data/pdfs

echo "[*] Building Docker image (this may take a few minutes)..."
docker-compose build

echo "[*] Starting services..."
docker-compose up -d

echo ""
echo "=== Setup Complete ==="
echo ""
echo "The application is running at: http://$(hostname -I | awk '{print $1}'):5555"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f web     # View application logs"
echo "  docker-compose exec web python ingest.py doj       # Ingest DOJ documents"
echo "  docker-compose exec web python ingest.py local /app/data/pdfs  # Ingest local PDFs"
echo "  docker-compose exec web python ingest.py threads   # Build email threads"
echo "  docker-compose exec web python ingest.py stats     # Show statistics"
echo "  docker-compose down            # Stop services"
echo "  docker-compose up -d           # Start services"
echo ""
echo "To ingest documents, copy PDFs to ./data/pdfs/ then run:"
echo "  docker-compose exec web python ingest.py local /app/data/pdfs"
echo ""
