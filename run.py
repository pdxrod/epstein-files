#!/usr/bin/env python3
"""Run the Epstein Files Search application."""

import os
from config import Config
from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=Config.PORT, debug=debug, use_reloader=debug)
