#!/usr/bin/env bash
set -e

# Read configuration from HA addon options
CONFIG_PATH=/data/options.json

# Model settings
RERANKER_MODEL=$(jq -r '.reranker_model' $CONFIG_PATH)
EMBEDDING_MODEL=$(jq -r '.embedding_model' $CONFIG_PATH)
DEVICE=$(jq -r '.device' $CONFIG_PATH)
PORT=$(jq -r '.port' $CONFIG_PATH)
HF_HOME_PATH=$(jq -r '.HF_HOME' $CONFIG_PATH)

# Cache file settings
ANCHORS_FILE=$(jq -r '.anchors_file' $CONFIG_PATH)
USER_CACHE_FILE=$(jq -r '.user_cache_file' $CONFIG_PATH)

# Hybrid search settings
HYBRID_ALPHA=$(jq -r '.hybrid_alpha' $CONFIG_PATH)
HYBRID_NGRAM_SIZE=$(jq -r '.hybrid_ngram_size' $CONFIG_PATH)
VECTOR_THRESHOLD=$(jq -r '.vector_threshold' $CONFIG_PATH)
VECTOR_TOP_K=$(jq -r '.vector_top_k' $CONFIG_PATH)
RERANKER_THRESHOLD=$(jq -r '.reranker_threshold' $CONFIG_PATH)

echo "[INFO] Starting Semantic Cache & Reranker addon..."
echo "[INFO] Reranker model: $RERANKER_MODEL"
echo "[INFO] Embedding model: $EMBEDDING_MODEL"
echo "[INFO] Device: $DEVICE"
echo "[INFO] Port: $PORT"
echo "[INFO] Model cache: $HF_HOME_PATH"
echo "[INFO] Anchors file: $ANCHORS_FILE"
echo "[INFO] User cache file: $USER_CACHE_FILE"

# Create model cache directory
mkdir -p "$HF_HOME_PATH"

# Export as environment variables
export RERANKER_MODEL="$RERANKER_MODEL"
export EMBEDDING_MODEL="$EMBEDDING_MODEL"
export RERANKER_DEVICE="$DEVICE"
export ANCHORS_FILE="$ANCHORS_FILE"
export USER_CACHE_FILE="$USER_CACHE_FILE"
export HYBRID_ALPHA="$HYBRID_ALPHA"
export HYBRID_NGRAM_SIZE="$HYBRID_NGRAM_SIZE"
export VECTOR_THRESHOLD="$VECTOR_THRESHOLD"
export VECTOR_TOP_K="$VECTOR_TOP_K"
export RERANKER_THRESHOLD="$RERANKER_THRESHOLD"

# Set HuggingFace cache paths
export HF_HOME="$HF_HOME_PATH"
export TRANSFORMERS_CACHE="$HF_HOME_PATH"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME_PATH"

# Disable problematic HuggingFace features
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

# Start the FastAPI server
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
