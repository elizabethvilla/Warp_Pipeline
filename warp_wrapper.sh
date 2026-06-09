#!/bin/bash

# This script serves as a wrapper to automatically load the required environment
# module before executing the main Python pipeline.

# Get the directory where this script is located to build a robust path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
GLOBAL_CONFIG="${SCRIPT_DIR}/config.py"

DATASET_NAME=$(grep "^dataset_name" "$GLOBAL_CONFIG" | awk -F'"' '{print $2}')
if [ -z "$DATASET_NAME" ]; then
    echo "Error: Could not extract dataset_name from config.py"
    exit 1
fi

LOCAL_CONFIG="$DATASET_NAME/config.py"

if [ -f "$LOCAL_CONFIG" ]; then
    echo "✅ Found local config: $LOCAL_CONFIG"
    echo "   Using parameters from this local snapshot."

elif [ ! -d "$DATASET_NAME" ]; then
    echo "🆕 New dataset detected: $DATASET_NAME"
    echo "   Creating folder and copying configuration snapshot..."
    mkdir -p "$DATASET_NAME"
    cp "$GLOBAL_CONFIG" "$LOCAL_CONFIG"

else
    echo "========================================================"
    echo "⛔️ BLOCKING ERROR"
    echo "Detected existing folder '$DATASET_NAME' but NO local 'config.py' found."
    echo ""
    echo "For safety, the pipeline strictly requires a local config snapshot."
    echo "ACTION REQUIRED:"
    echo "1. Verify which parameters were used for this old dataset."
    echo "2. Manually copy the correct config.py into '$DATASET_NAME/'."
    echo "========================================================"
    exit 1
fi

cd "$DATASET_NAME" || exit

echo "Loading required module: aretomo and warp/2"
module load aretomo
module load warp/2.0.0dev39

echo "Executing pipeline: run_pipeline.py"
python3 "${SCRIPT_DIR}/run_pipeline.py" "$@"

echo "Pipeline execution finished."
