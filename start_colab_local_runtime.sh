#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${1:-8888}"
ALLOW_ORIGIN="${ALLOW_ORIGIN:-https://colab.research.google.com}"
VENV_PATH="${VENV_PATH:-.venv311}"
LOG_FILE=".colab_runtime.log"
PID_FILE=".colab_runtime.pid"

if [[ ! -d "$VENV_PATH" ]]; then
  echo "Missing virtual environment: $VENV_PATH"
  echo "Create it first:"
  echo "python3 -m venv .venv311 && source .venv311/bin/activate && pip install -r requirements.txt"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use. Reusing existing runtime if it is Jupyter."
else
  nohup python -m notebook \
    --no-browser \
    --port="$PORT" \
    --ServerApp.port_retries=0 \
    --ServerApp.allow_origin="$ALLOW_ORIGIN" \
    --ServerApp.allow_credentials=True \
    > "$LOG_FILE" 2>&1 < /dev/null &
  sleep 4
fi

if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Failed to start Jupyter local runtime on port $PORT."
  echo "Check logs: $ROOT_DIR/$LOG_FILE"
  exit 1
fi

PID="$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN | head -n1 || true)"
if [[ -n "$PID" ]]; then
  echo "$PID" > "$PID_FILE"
fi

SERVER_URL="$(jupyter server list | awk 'NR>1 && $1 ~ /^http/ {print $1; exit}')"
if [[ -z "$SERVER_URL" ]]; then
  SERVER_URL="http://localhost:${PORT}"
fi

echo
echo "Local Colab runtime is ready."
echo "URL: $SERVER_URL"
echo "PID: ${PID:-unknown}"
echo "Log: $ROOT_DIR/$LOG_FILE"
echo
echo "Next step in browser:"
echo "Colab -> Connect (top-right) -> Connect to local runtime -> paste URL above."
