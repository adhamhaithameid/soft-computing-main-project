#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${1:-8888}"
PID_FILE=".colab_runtime.pid"

stop_pid() {
  local pid="$1"
  if kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    return 0
  fi
  return 1
}

STOPPED=0

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${PID:-}" ]] && stop_pid "$PID"; then
    echo "Stopped runtime PID $PID (from $PID_FILE)."
    STOPPED=1
  fi
  rm -f "$PID_FILE"
fi

PORT_PID="$(lsof -t -nP -iTCP:"$PORT" -sTCP:LISTEN | head -n1 || true)"
if [[ -n "$PORT_PID" ]] && stop_pid "$PORT_PID"; then
  echo "Stopped runtime PID $PORT_PID on port $PORT."
  STOPPED=1
fi

if [[ "$STOPPED" -eq 0 ]]; then
  echo "No local Colab runtime was found on port $PORT."
fi
