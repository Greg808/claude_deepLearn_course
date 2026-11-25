#!/usr/bin/env bash
# Frontend quality checks. Usage: scripts/frontend-quality.sh [--fix]
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d node_modules ] || npm ci

if [ "${1:-}" = "--fix" ]; then
    npx prettier --write frontend
else
    npx prettier --check frontend
fi
