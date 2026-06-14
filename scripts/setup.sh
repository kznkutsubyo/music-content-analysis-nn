#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

PYTHON_BIN="${PYTHON_BIN:-python3}"

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

echo "==> Checking required tools"
command_exists "$PYTHON_BIN" || {
  echo "Python was not found. Install Python 3.10+ or set PYTHON_BIN=/path/to/python." >&2
  exit 1
}
command_exists npm || {
  echo "npm was not found. Install Node.js 18+ and npm." >&2
  exit 1
}

echo "==> Creating backend virtual environment"
cd "$BACKEND_DIR"
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source "$BACKEND_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$BACKEND_DIR/requirements-runtime.txt"

echo "==> Installing frontend dependencies"
cd "$FRONTEND_DIR"
npm install

if [ ! -f ".env.local" ]; then
  cp .env.example .env.local
fi

echo
echo "Setup complete."
echo "Run the application with:"
echo "  ./scripts/run_app.sh"
