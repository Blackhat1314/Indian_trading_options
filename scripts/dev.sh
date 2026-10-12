#!/usr/bin/env bash
# Developer loop: backend stack (sim, 20x) + Vite dev server with hot reload on :5173.
#   ./scripts/dev.sh            # needs redis on localhost:6379 (docker run -d -p 6379:6379 redis:7-alpine)
set -euo pipefail
cd "$(dirname "$0")/.."
SIM_SPEED="${SIM_SPEED:-20}" python3 run.py --sim --speed "${SIM_SPEED:-20}" &
STACK=$!
trap 'kill $STACK 2>/dev/null || true' EXIT INT TERM
cd frontend
[ -d node_modules ] || npm ci
VITE_GATEWAY="${VITE_GATEWAY:-http://127.0.0.1:8000}" npx vite --host
