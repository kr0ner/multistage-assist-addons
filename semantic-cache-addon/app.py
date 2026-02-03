"""
Semantic Cache & Reranker API.

Home Assistant addon that provides:
1. CrossEncoder reranking API (/rerank)
2. Full semantic cache lookup (/lookup)

Combines BM25 keyword search with vector similarity and reranking
for fast, accurate command resolution.
"""

import os
import re
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from cache_types import CacheEntry, DOMAIN_THRESHOLDS
from cache_loader import CacheLoader
from bm25_index import BM25Index
from file_watcher import CacheFileWatcher
import embedding as emb

# Configuration from environment (set by run.sh from addon options)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DEVICE = os.getenv("RERANKER_DEVICE", "cpu")
ANCHORS_FILE = os.getenv("ANCHORS_FILE", "/homeassistant/.storage/multistage_assist_anchors.json")
USER_CACHE_FILE = os.getenv("USER_CACHE_FILE", "/homeassistant/.storage/multistage_assist_semantic_cache.json")

# Hybrid search config
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))  # Weight for semantic vs BM25
HYBRID_NGRAM_SIZE = int(os.getenv("HYBRID_NGRAM_SIZE", "2"))
VECTOR_THRESHOLD = float(os.getenv("VECTOR_THRESHOLD", "0.5"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
RERANKER_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", "0.73"))

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("reranker")
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("watchdog").setLevel(logging.WARNING)

# Initialize FastAPI
app = FastAPI(
    title="Semantic Cache & Reranker API",
    description="Cache lookup + CrossEncoder reranking for Multi-Stage Assist",
    version="2.0.0",
)

# Global state
reranker_model: CrossEncoder = None
cache_loader: CacheLoader = None
bm25_index: BM25Index = None
file_watcher: CacheFileWatcher = None
loading = True


# ============================================================================
# Request/Response Models
# ============================================================================

class RerankRequest(BaseModel):
    """Request body for reranking."""
    query: str
    candidates: List[str]


class RerankResponse(BaseModel):
    """Response with reranking scores."""
    scores: List[float]
    best_index: int
    best_score: float


class LookupRequest(BaseModel):
    """Request body for cache lookup."""
    query: str


class LookupResponse(BaseModel):
    """Response from cache lookup."""
    found: bool
    intent: Optional[str] = None
    entity_ids: Optional[List[str]] = None
    slots: Optional[Dict[str, Any]] = None
    score: float = 0.0
    original_text: Optional[str] = None
    reranked: bool = False


class EmbedEntryRequest(BaseModel):
    """Single cache entry to embed."""
    text: str
    intent: str
    entity_ids: List[str] = []
    slots: Dict[str, Any] = {}


class EmbedRequest(BaseModel):
    """Request body for embedding cache entries."""
    entries: List[EmbedEntryRequest]


class EmbedEntryResponse(BaseModel):
    """Single cache entry with embedding."""
    text: str
    intent: str
    entity_ids: List[str]
    slots: Dict[str, Any]
    embedding: List[float]
    generated: bool = True


class EmbedResponse(BaseModel):
    """Response with embedded cache entries."""
    entries: List[EmbedEntryResponse]
    embedding_model: str
    embedding_dim: int


class EmbedTextRequest(BaseModel):
    """Request body for embedding a single text."""
    text: str


class EmbedTextResponse(BaseModel):
    """Response with text embedding."""
    text: str
    embedding: List[float]
    embedding_model: str
    embedding_dim: int


# ============================================================================
# Device Detection & Model Loading
# ============================================================================

def detect_best_device() -> str:
    """Auto-detect the best available device."""
    import torch

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "unknown"
        logger.info(f"CUDA available: {device_name}")
        return "cuda"

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("Apple MPS available")
        return "mps"

    logger.info("No GPU detected, using CPU")
    return "cpu"


def load_reranker_on_device(model_name: str, device: str) -> CrossEncoder:
    # CPU path with INT8 quantization (NUC optimization)
    if device == "cpu":
        # Cache path for serialized quantized model
        safe_name = model_name.replace("/", "_").replace("-", "_")
        cache_path = os.path.join(os.getenv("HF_HOME", "/share/semantic-cache"), f"{safe_name}_quantized.pt")
        
        # 1. Try Loading Cached Model (Fast & Low RAM)
        if os.path.exists(cache_path):
            logger.info(f"Loading cached quantized model from {cache_path}...")
            try:
                model = torch.load(cache_path)
                logger.info("✅ Cached Int8 Model Loaded.")
                return model
            except Exception as e:
                logger.warning(f"Failed to load cached model (corruption?): {e}")
                os.remove(cache_path) # Delete corrupt file

        # 2. Create & Cache Model (High RAM - One Time)
        logger.info(f"⚡ NUC MODE: Loading {model_name} with INT8 Quantization...")
        try:
            # Load the full FP32 model
            model = CrossEncoder(model_name, device="cpu", trust_remote_code=True)
            
            # Apply Dynamic Quantization
            model.model = torch.quantization.quantize_dynamic(
                model.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            # Warmup
            logger.info("Warming up...")
            model.predict([["warmup", "warmup"]])
            
            # Save to disk for next time
            logger.info(f"Saving quantized model to {cache_path}...")
            torch.save(model, cache_path)
            
            logger.info(f"✅ Model Quantized & Ready.")
            return model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            logger.warning("Falling back to standard FP32 CPU model")
            return CrossEncoder(model_name, device="cpu", trust_remote_code=True)

    # GPU paths (CUDA, MPS)
    kwargs = {"trust_remote_code": True}
    return CrossEncoder(model_name, device="cpu", **kwargs)


# ============================================================================
# Numeric Value Normalization
# ============================================================================

def normalize_numeric_value(text: str) -> Tuple[str, List[Any]]:
    """
    Normalize numeric values in text for generalized cache lookup.

    Returns: (normalized_text, extracted_values)
    Example: "Setze Rollo auf 75%" -> ("Setze Rollo auf 50 Prozent", [75])
    """
    extracted = []

    def replace_percent(match):
        val = match.group(1)
        extracted.append(int(val))
        return "50 Prozent"

    def replace_temp(match):
        val = match.group(1)
        try:
            if "." in val or "," in val:
                extracted.append(float(val.replace(",", ".")))
            else:
                extracted.append(int(val))
        except ValueError:
            pass
        return "21 Grad"

    # Percentages: "75%", "75 %", "75 Prozent"
    text_norm = re.sub(r"(\d+)\s*(?:%|Prozent|prozent)", replace_percent, text, flags=re.IGNORECASE)

    # Temperatures: "23.5 Grad", "23°"
    if text_norm == text:
        text_norm = re.sub(r"(\d+(?:[.,]\d+)?)\s*(?:Grad|°|grad)", replace_temp, text_norm)

    return text_norm, extracted


# ============================================================================
# Cache Reload Callback
# ============================================================================

def reload_cache() -> None:
    """Reload cache and rebuild BM25 index (called by file watcher)."""
    global bm25_index

    if cache_loader is None:
        logger.warning("Cache reload called but cache_loader not initialized")
        return

    try:
        cache_loader.reload()

        # Rebuild BM25 index
        if bm25_index is not None:
            logger.info("Rebuilding BM25 index...")
            bm25_index.build(cache_loader.get_texts())

        logger.info("Cache reload complete")
    except Exception as e:
        logger.error(f"Cache reload failed: {e}")


# ============================================================================
# Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    """Load all models and cache on startup."""
    global reranker_model, cache_loader, bm25_index, file_watcher, loading

    logger.info("=" * 60)
    logger.info("STARTING SEMANTIC CACHE & RERANKER ADDON")
    logger.info("=" * 60)
    logger.info(f"Reranker model: {RERANKER_MODEL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Anchors file: {ANCHORS_FILE}")
    logger.info(f"User cache file: {USER_CACHE_FILE}")
    logger.info("This may take several minutes on first run...")

    # Detect device
    actual_device = DEVICE
    if DEVICE == "auto":
        actual_device = detect_best_device()
        logger.info(f"Auto-detected device: {actual_device}")

    # Load reranker
    logger.info("Loading reranker model...")
    reranker_model = load_reranker_on_device(RERANKER_MODEL, actual_device)
    logger.info("Reranker model loaded")

    # Load embedding model
    logger.info("Loading embedding model...")
    emb.load_embedding_model(EMBEDDING_MODEL, actual_device)
    logger.info("Embedding model loaded")

    # Load cache
    logger.info("Loading cache files...")
    cache_loader = CacheLoader(ANCHORS_FILE, USER_CACHE_FILE)
    anchor_count, user_count = cache_loader.load()
    logger.info(f"Cache loaded: {anchor_count} anchors + {user_count} user entries")

    # Build BM25 index
    logger.info("Building BM25 index...")
    bm25_index = BM25Index(ngram_size=HYBRID_NGRAM_SIZE)
    bm25_index.build(cache_loader.get_texts())

    # Start file watcher for cache auto-reload
    logger.info("Starting cache file watcher...")
    file_watcher = CacheFileWatcher(
        file_paths=[ANCHORS_FILE, USER_CACHE_FILE],
        on_reload=reload_cache,
        poll_interval=30.0,
        debounce_seconds=2.0,
    )
    await file_watcher.start()

    loading = False
    logger.info("=" * 60)
    logger.info("READY! Endpoints: /health, /rerank, /lookup")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown():
    """Graceful shutdown."""
    global file_watcher

    logger.info("Shutting down...")
    if file_watcher:
        await file_watcher.stop()
    logger.info("Shutdown complete")


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    cache_entries = len(cache_loader.entries) if cache_loader else 0
    last_reload = cache_loader.last_reload_time if cache_loader else None
    return {
        "status": "loading" if loading else "ok",
        "reranker_model": RERANKER_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "device": DEVICE,
        "cache_entries": cache_entries,
        "last_reload": last_reload,
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """Rerank candidates against a query."""
    if loading or reranker_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")

    logger.info(f"Rerank: query='{request.query[:50]}' candidates={len(request.candidates)}")

    pairs = [[request.query, c] for c in request.candidates]
    raw_scores = reranker_model.predict(pairs)
    probs = 1 / (1 + np.exp(-raw_scores))

    scores = probs.tolist()
    best_idx = int(np.argmax(probs))

    logger.info(f"Rerank result: best_idx={best_idx}, best_score={probs[best_idx]:.4f}")

    return RerankResponse(
        scores=scores,
        best_index=best_idx,
        best_score=float(probs[best_idx]),
    )


@app.post("/lookup", response_model=LookupResponse)
async def lookup(request: LookupRequest):
    """
    Two-stage semantic cache lookup.

    Stage 1: Fast vector search + BM25 hybrid scoring
    Stage 2: Precise reranking with CrossEncoder

    Returns matched cache entry or found=false.
    """
    if loading:
        raise HTTPException(status_code=503, detail="Still loading")

    if not cache_loader or not cache_loader.is_loaded:
        return LookupResponse(found=False, score=0.0)

    if cache_loader.embeddings_matrix is None or len(cache_loader.entries) == 0:
        logger.debug("Cache empty")
        return LookupResponse(found=False, score=0.0)

    query = request.query
    logger.info(f"Lookup: '{query[:60]}'")

    # Normalize query (handle percentages, temperatures)
    query_norm, extracted_values = normalize_numeric_value(query)
    if query_norm != query:
        logger.debug(f"Normalized: '{query}' -> '{query_norm}' [{extracted_values}]")

    # Get query embedding
    query_emb = emb.get_embedding(query_norm)
    if query_emb is None:
        logger.warning("Failed to get embedding")
        return LookupResponse(found=False, score=0.0)

    logger.debug(f"Query embedding dim: {query_emb.shape}, cache matrix: {cache_loader.embeddings_matrix.shape}")

    # Compute cosine similarity (embeddings are already normalized)
    similarities = np.dot(cache_loader.embeddings_matrix, query_emb)

    # Log top semantic match for debugging
    top_sem_idx = int(np.argmax(similarities))
    top_sem_score = float(similarities[top_sem_idx])
    top_sem_entry = cache_loader.entries[top_sem_idx] if cache_loader.entries else None
    if top_sem_entry:
        logger.debug(f"Top semantic: score={top_sem_score:.4f}, text='{top_sem_entry.text[:60]}'")

    # Hybrid search: combine with BM25
    if bm25_index and bm25_index.is_built:
        bm25_scores = bm25_index.get_scores(query_norm)

        # Log top BM25 match
        top_bm25_idx = int(np.argmax(bm25_scores))
        top_bm25_score = float(bm25_scores[top_bm25_idx])
        top_bm25_entry = cache_loader.entries[top_bm25_idx] if cache_loader.entries else None
        if top_bm25_entry:
            logger.debug(f"Top BM25: score={top_bm25_score:.4f}, text='{top_bm25_entry.text[:60]}'")

        if len(bm25_scores) == len(similarities):
            hybrid_scores = HYBRID_ALPHA * similarities + (1 - HYBRID_ALPHA) * bm25_scores
            logger.debug(
                f"Hybrid: semantic_max={similarities.max():.3f}, "
                f"bm25_max={bm25_scores.max():.3f}, hybrid_max={hybrid_scores.max():.3f}"
            )
            similarities = hybrid_scores

    # Get candidates above threshold
    candidates: List[Tuple[float, int, CacheEntry]] = []
    for idx, score in enumerate(similarities):
        if score >= VECTOR_THRESHOLD:
            candidates.append((float(score), idx, cache_loader.entries[idx]))

    if not candidates:
        logger.debug(f"No candidates above threshold {VECTOR_THRESHOLD}")
        return LookupResponse(found=False, score=0.0)

    # Sort and take top-k
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:VECTOR_TOP_K]

    logger.debug(f"Found {len(candidates)} candidates (top: {candidates[0][0]:.3f})")

    # Stage 2: Reranking
    pairs = [[query_norm, c[2].text] for c in candidates]
    raw_scores = reranker_model.predict(pairs)
    probs = 1 / (1 + np.exp(-raw_scores))

    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    logger.debug(f"Reranker: best_idx={best_idx}, best_score={best_prob:.4f}")

    # Get domain-specific threshold
    _, cache_idx, entry = candidates[best_idx]
    domain = None
    if entry.entity_ids:
        parts = entry.entity_ids[0].split(".")
        if len(parts) > 1:
            domain = parts[0]
    threshold = DOMAIN_THRESHOLDS.get(domain, RERANKER_THRESHOLD)

    if best_prob < threshold:
        logger.info(f"Reranker blocked: {best_prob:.4f} < {threshold:.4f} (domain={domain})")
        return LookupResponse(found=False, score=best_prob)

    # Success! Build response
    slots = dict(entry.slots) if entry.slots else {}

    # Inject extracted numeric values
    if extracted_values:
        val = extracted_values[0]
        for key in ["position", "brightness", "temperature", "volume_level"]:
            if key in slots:
                logger.debug(f"Injecting {val} into slot '{key}'")
                slots[key] = val

    logger.info(
        f"HIT (score={best_prob:.3f}): '{query[:40]}' -> {entry.intent} [{entry.entity_ids}]"
    )

    return LookupResponse(
        found=True,
        intent=entry.intent,
        entity_ids=entry.entity_ids,
        slots=slots,
        score=best_prob,
        original_text=entry.text,
        reranked=True,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate embeddings for cache entries.

    Use this endpoint to create embeddings with the same model
    used for cache lookup, ensuring consistency.
    """
    if loading:
        raise HTTPException(status_code=503, detail="Still loading")

    if not request.entries:
        raise HTTPException(status_code=400, detail="No entries provided")

    logger.info(f"Embed: processing {len(request.entries)} entries")

    # Get embedding dimension
    embedding_dim = emb.get_embedding_dim()
    if embedding_dim is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    embedded_entries = []
    for entry in request.entries:
        # Generate embedding
        embedding = emb.get_embedding(entry.text)
        if embedding is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate embedding for: {entry.text[:50]}"
            )

        embedded_entries.append(EmbedEntryResponse(
            text=entry.text,
            intent=entry.intent,
            entity_ids=entry.entity_ids,
            slots=entry.slots,
            embedding=embedding.tolist(),
            generated=True,
        ))

    logger.info(f"Embed: generated {len(embedded_entries)} embeddings (dim={embedding_dim})")

    return EmbedResponse(
        entries=embedded_entries,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=embedding_dim,
    )


@app.post("/embed/text", response_model=EmbedTextResponse)
async def embed_text(request: EmbedTextRequest):
    """
    Generate embedding for a single text string.

    Simple endpoint for embedding individual texts.
    """
    if loading:
        raise HTTPException(status_code=503, detail="Still loading")

    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    logger.debug(f"Embed text: '{request.text[:60]}'")

    # Get embedding dimension
    embedding_dim = emb.get_embedding_dim()
    if embedding_dim is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")

    # Generate embedding
    embedding = emb.get_embedding(request.text)
    if embedding is None:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate embedding"
        )

    return EmbedTextResponse(
        text=request.text,
        embedding=embedding.tolist(),
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=embedding_dim,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9876, log_level="debug")
