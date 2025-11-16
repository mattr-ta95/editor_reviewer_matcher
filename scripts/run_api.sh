#!/bin/bash
# Script to run the FastAPI application

# Set environment variables
export FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-data/indices/specter2.index}"
export DATABASE_PATH="${DATABASE_PATH:-data/reviewers.db}"
export MODEL_NAME="${MODEL_NAME:-allenai/specter2_base}"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API
echo "Starting Semantic Reviewer Matching API..."
echo "Host: $HOST"
echo "Port: $PORT"
echo "Database: $DATABASE_PATH"
echo "Index: $FAISS_INDEX_PATH"
echo ""

uvicorn src.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL"
