#!/usr/bin/env bash
# Serve the web demo on http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")/../web"
echo "Open http://localhost:8000 in your browser"
exec python3 -m http.server 8000
