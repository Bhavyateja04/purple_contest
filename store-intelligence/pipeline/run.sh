#!/usr/bin/env bash
# One-command script to process all CCTV clips and emit events.
# Output: events.jsonl in the project root.
#
# Usage: bash pipeline/run.sh [--footage-dir /path/to/CCTV Footage]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
FOOTAGE_DIR="${1:-$(dirname "$PROJECT_DIR")/CCTV Footage}"
POS_FILE="$(dirname "$PROJECT_DIR")/Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
OUTPUT="${PROJECT_DIR}/events.jsonl"

echo "==> Store Intelligence Detection Pipeline"
echo "    Footage  : $FOOTAGE_DIR"
echo "    POS data : $POS_FILE"
echo "    Output   : $OUTPUT"
echo ""

# Activate venv if present
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

python3 "${SCRIPT_DIR}/detect.py" \
    --all \
    --footage-dir "$FOOTAGE_DIR" \
    --output "$OUTPUT" \
    --pos "$POS_FILE"

echo ""
echo "==> Done. Event count: $(wc -l < "$OUTPUT") events"
echo "==> Feed into API: curl -X POST http://localhost:8000/events/ingest ..."
