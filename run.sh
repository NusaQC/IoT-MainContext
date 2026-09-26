#!/usr/bin/env bash
# ==============================================================================
# NusaQC Edge Node Launcher (Raspberry Pi 4 / Closed-Loop Sortasi)
# ==============================================================================

set -e

# 1. Optimal Thread Tuning for Quad-Core ARM Cortex-A72
export NUSAQC_ORT_THREADS=${NUSAQC_ORT_THREADS:-4}
export PYTHONUNBUFFERED=1

# 2. Virtual Environment Detection & Activation
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "🐍 Activating virtualenv: $SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
elif [ -f "$HOME/nusaqc/venv/bin/activate" ]; then
    echo "🐍 Activating virtualenv: $HOME/nusaqc/venv"
    source "$HOME/nusaqc/venv/bin/activate"
fi

# 3. Launch Edge Orchestrator with User Arguments or Default Settings
DEFAULT_ARGS="--cam 0 --mjpeg-port 8080 --central-url http://192.168.137.1:8000"

if [ "$#" -eq 0 ]; then
    echo "🚀 Starting NusaQC Edge Orchestrator with default parameters..."
    exec python3 -m edge.main $DEFAULT_ARGS
else
    echo "🚀 Starting NusaQC Edge Orchestrator with custom arguments: $@"
    exec python3 -m edge.main "$@"
fi
