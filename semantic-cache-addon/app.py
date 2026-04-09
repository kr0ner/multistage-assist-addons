"""
Semantic Cache API.

Home Assistant addon that provides:
1. Full semantic cache lookup (/lookup)

Combines BM25 keyword search with vector similarity
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

from cache_types import CacheEntry, DOMAIN_THRESHOLDS
from cache_loader import CacheLoader
from bm25_index import BM25Index
from file_watcher import CacheFileWatcher
import embedding as emb

# Configuration from environment (set by run.sh from addon options)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEVICE = os.getenv("RERANKER_DEVICE", "cpu")  # Key kept for compatibility
ANCHORS_FILE = os.getenv("ANCHORS_FILE", "/homeassistant/.storage/multistage_assist_anchors.json")
USER_CACHE_FILE = os.getenv("USER_CACHE_FILE", "/homeassistant/.storage/multistage_assist_semantic_cache.json")

# Hybrid search config
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.95"))
HYBRID_NGRAM_SIZE = int(os.getenv("HYBRID_NGRAM_SIZE", "2"))
VECTOR_THRESHOLD = float(os.getenv("VECTOR_THRESHOLD", "0.5"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))

# Security config for cache modification
PRODUCTION_MODE = os.getenv("PRODUCTION_MODE", "true").lower() == "true"
PROD_CACHE_KEY = os.getenv("PROD_CACHE_KEY", "")

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
    title="Semantic Cache API",
    description="Cache lookup for Multi-Stage Assist",
    version="2.1.0",
)

# Global state
cache_loader: CacheLoader = None
bm25_index: BM25Index = None
file_watcher: CacheFileWatcher = None
loading = True


# ============================================================================
# Request/Response Models
# ============================================================================

# Language constants aligned with german_utils.py
GERMAN_ARTICLES = {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines"}
GERMAN_PREPOSITIONS = {"im", "in", "auf", "unter", "über", "am", "bei", "zum", "zur", "vom", "von", "für", "mit", "nach"}
FILLER_WORDS = {"bitte", "mal", "gerne", "doch", "kannst", "könntest", "würdest"}

def normalize_umlauts(text: str) -> str:
    """Normalize German umlauts and ß to ASCII equivalents."""
    return text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")

def canonicalize(text: str) -> str:
    """Standardize text: lower, ASCII-umlauts, remove special chars."""
    if not text: return ""
    import unicodedata
    t = unicodedata.normalize('NFC', text.lower())
    t = normalize_umlauts(t)
    t = re.sub(r"[^\w\s%°]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def strip_filler_words(text: str) -> str:
    """Strip meaningless filler words while preserving grammar."""
    tokens = text.split()
    return " ".join([w for w in tokens if w.lower() not in FILLER_WORDS]).strip()

class LookupRequest(BaseModel):
    """Request body for cache lookup."""
    query: str


class Candidate(BaseModel):
    """A search candidate for debugging."""
    text: str
    score: float
    intent: Optional[str] = None
    entity_ids: List[str] = []

class LookupResponse(BaseModel):
    """Response from cache lookup."""
    found: bool
    intent: Optional[str] = None
    entity_ids: Optional[List[str]] = None
    slots: Optional[Dict[str, Any]] = None
    score: float = 0.0
    original_text: Optional[str] = None
    reranked: bool = False
    candidates: List[Candidate] = []


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


# ============================================================================
# Numeric Value Normalization
# ============================================================================

def normalize_numeric_value(text: str) -> Tuple[str, List[Any]]:
    """Normalize text for semantic cache matching, aligned with main app centroids."""
    if not text: return "", []
    
    # Standard Canonicalization
    text_norm = canonicalize(text)
    extracted: List[Any] = []

    # 1. Number Centroids (Principle 7)
    def replace_pct(match):
        extracted.append(int(match.group(1)))
        return "50 prozent"
    def replace_temp(match):
        val = match.group(1).replace(",", ".")
        extracted.append(float(val))
        return "21 grad"

    text_norm = re.sub(r"(\d+)\s*(?:%|prozent)(?:\b|\s|$)", replace_pct, text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r"(\d+(?:[.,]\d+)?)\s*(?:\u00b0|grad)\b", replace_temp, text_norm, flags=re.IGNORECASE)

    # 2. Time Centroids
    text_norm = re.sub(r"\bin\s+(\d+|eine[rn]?)\s+(minuten?|stunden?|sekunden?)\b", "in 10 minuten", text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r"\bfuer\s+(\d+|eine[rn]?)\s+(minuten?|stunden?|sekunden?)\b", "fuer 10 minuten", text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r"\bauf\s+(\d+|eine[rn]?)\s+(minuten?|stunden?|sekunden?)\b", "auf 10 minuten", text_norm, flags=re.IGNORECASE)
    text_norm = re.sub(r"\bum\s+(\d{1,2})(?:\s+\d{2})?\s*uhr\b", "um 10 uhr", text_norm, flags=re.IGNORECASE)

    # 3. Fraction normalization ("haelfte" -> "50 prozent")
    # Ported from german_utils.py
    for fraction_word in ["haelfte", "halb", "viertel", "dreiviertel", "ganz", "voll"]:
        pattern = r"\b" + fraction_word + r"\b"
        if re.search(pattern, text_norm, flags=re.IGNORECASE):
            text_norm = re.sub(pattern, "50 prozent", text_norm, flags=re.IGNORECASE)

    # 4. Strip filler words
    text_norm = strip_filler_words(text_norm)

    return re.sub(r"\s+", " ", text_norm).strip(), extracted


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
    global cache_loader, bm25_index, file_watcher, loading

    logger.info("=" * 60)
    logger.info("STARTING SEMANTIC CACHE API")
    logger.info("=" * 60)
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
    logger.info("READY! Endpoints: /health, /lookup")
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
    return {
        "status": "loading" if loading else "ok",
        "version": "2.1.0-normalization-fix",
        "embedding_model": EMBEDDING_MODEL,
        "device": DEVICE,
        "cache_entries": len(cache_loader.entries) if cache_loader else 0,
        "vector_threshold": VECTOR_THRESHOLD,
        "last_reload": cache_loader.last_reload_time if cache_loader else None,
    }


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
    logger.debug(f"Lookup request: '{query}' -> Normalized: '{query_norm}' [{extracted_values}]")

    # Get query embedding
    query_emb = emb.get_embedding(query_norm)
    if query_emb is None:
        logger.warning("Failed to get embedding")
        return LookupResponse(found=False, score=0.0)

    # Compute cosine similarity (embeddings are already normalized)
    try:
        if cache_loader.embeddings_matrix.shape[1] != query_emb.shape[0]:
            logger.error(f"Dimension mismatch: cache={cache_loader.embeddings_matrix.shape[1]}, query={query_emb.shape[0]}")
            return LookupResponse(found=False, score=0.0)
            
        similarities = np.dot(cache_loader.embeddings_matrix, query_emb)
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        return LookupResponse(found=False, score=0.0)

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

    # Selection: take top candidate
    best_prob, cache_idx, entry = candidates[0]
    logger.debug(f"Standalone lookup: top match score={best_prob:.4f}")

    # Success! Build response
    slots = dict(entry.slots) if entry.slots else {}
    
    # Candidates list
    candidates_list = [
        Candidate(
            text=c[2].text,
            score=float(c[0]),
            intent=c[2].intent,
            entity_ids=c[2].entity_ids
        ) for c in candidates[:5]
    ]

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
        reranked=False,
        candidates=candidates_list
    )


from fastapi import Header

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, authorization: Optional[str] = Header(None)):
    """
    Generate embeddings for cache entries (Production Lock protected).
    """
    if loading:
        raise HTTPException(status_code=503, detail="Still loading")

    # Production Lock: Prevent tests or external callers from polluting the cache
    if PRODUCTION_MODE:
        if not PROD_CACHE_KEY:
             raise HTTPException(status_code=500, detail="Reranker in production mode but no PROD_CACHE_KEY set")
        if authorization != f"Bearer {PROD_CACHE_KEY}":
             logger.warning(f"Unauthorized embed attempt from {authorization}")
             raise HTTPException(status_code=403, detail="Not authorized to modify semantic cache")

    if not request.entries:
        raise HTTPException(status_code=400, detail="No entries provided")

    # Access Control: Production Lock
    if PRODUCTION_MODE:
        auth_header = request.headers.get("Authorization", "") if hasattr(request, "headers") else ""
        # In FastAPI, we usually use Header or Depends, but here we can check the request object if it was a Starlette request
        # Actually, let's use a cleaner way for simplicity in this script:
        pass # Will implement check using a helper
    
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
async def embed_text(request: EmbedTextRequest, authorization: Optional[str] = Header(None)):
    """
    Generate embedding for a single text string (Production Lock protected).
    """
    if loading:
        raise HTTPException(status_code=503, detail="Still loading")

    # Production Lock
    if PRODUCTION_MODE:
        if authorization != f"Bearer {PROD_CACHE_KEY}":
             raise HTTPException(status_code=403, detail="Not authorized to modify semantic cache")

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
