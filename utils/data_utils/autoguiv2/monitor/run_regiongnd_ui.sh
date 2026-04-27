#!/bin/bash
# Quick launcher for FuncRegionGnd Task Revising UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "🚀 Starting FuncRegionGnd Task Revising UI..."
echo "📁 Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  FastAPI not found. Installing dependencies..."
    pip3 install -r requirements-webui.txt
fi

# Default datasets root
DATASETS_ROOT="${DATASETS_ROOT:-/mnt/vdb1/hongxin_li/AutoGUIv2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-17806}"

echo "📊 Datasets root: $DATASETS_ROOT"
echo "🌐 Server: http://$HOST:$PORT"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -m utils.data_utils.autoguiv2.monitor.revise_regiongnd_tasks \
    --datasets-root "$DATASETS_ROOT" \
    --host "$HOST" \
    --port "$PORT"


