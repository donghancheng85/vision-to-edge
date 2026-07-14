#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$REPO_ROOT"

# Ensure uv is on PATH (handles fresh installs)
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] Installing dependencies with uv..."
uv sync

echo ""
echo "[setup] Done. Activate with:  source .venv/bin/activate"
echo "[setup] Or run directly with: uv run python <script>"
