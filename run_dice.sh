#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "Starting Dice Cube solver on http://localhost:8001"
python -m dicecube.server
