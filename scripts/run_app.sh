#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
MODEL_DEVICE="${MODEL_DEVICE:-cpu}"
API_BASE="http://${BACKEND_HOST}:${BACKEND_PORT}"

if [ ! -x "$BACKEND_DIR/.venv/bin/python" ]; then
  echo "Backend virtual environment was not found." >&2
  echo "Run ./scripts/setup.sh first." >&2
  exit 1
fi

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  echo "Frontend dependencies were not found." >&2
  echo "Run ./scripts/setup.sh first." >&2
  exit 1
fi

if [ ! -f "$BACKEND_DIR/gtzan_ast_best.pt" ]; then
  echo "Missing AST model: backend/gtzan_ast_best.pt" >&2
  echo "If you cloned the repository, run: git lfs pull" >&2
  exit 1
fi

cleanup() {
  if [ -n "${BACKEND_PID:-}" ] && kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [ -n "${FRONTEND_PID:-}" ] && kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on ${API_BASE}"
cd "$BACKEND_DIR"
MODEL_DEVICE="$MODEL_DEVICE" "$BACKEND_DIR/.venv/bin/python" -m uvicorn app:app \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" &
BACKEND_PID=$!

echo "==> Starting frontend on http://${FRONTEND_HOST}:${FRONTEND_PORT}"
cd "$FRONTEND_DIR"
VITE_API_BASE="$API_BASE" npm run dev -- \
  --host "$FRONTEND_HOST" \
  --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

echo
echo "Application is starting:"
echo "  Backend:  ${API_BASE}"
echo "  Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
echo
echo "Press Ctrl+C to stop both servers."

wait -n "$BACKEND_PID" "$FRONTEND_PID"
