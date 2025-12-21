"""Local embedding service using sentence-transformers.

Provides embeddings without external Ollama dependency.
Uses BAAI/bge-m3 model for compatibility with Multi-Stage Assist cache.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("reranker.embedding")

# Global model instance (loaded once)
_embedding_model = None
_embedding_model_name = None


def load_embedding_model(model_name: str = "BAAI/bge-m3", device: str = "cpu"):
    """
    Load embedding model.

    Args:
        model_name: HuggingFace model name
        device: Device to load on (cpu, cuda, xpu, mps)
    """
    global _embedding_model, _embedding_model_name

    if _embedding_model is not None and _embedding_model_name == model_name:
        logger.debug("Embedding model already loaded: %s", model_name)
        return

    logger.info("Loading embedding model: %s on %s", model_name, device)

    try:
        from sentence_transformers import SentenceTransformer

        # Handle device-specific loading
        if device == "xpu":
            try:
                import intel_extension_for_pytorch as ipex
                _embedding_model = SentenceTransformer(model_name, device="cpu")
                _embedding_model = ipex.optimize(_embedding_model)
                logger.info("Embedding model optimized for Intel XPU")
            except ImportError:
                logger.warning("IPEX not available, using CPU for embeddings")
                _embedding_model = SentenceTransformer(model_name, device="cpu")
        else:
            _embedding_model = SentenceTransformer(model_name, device=device)

        _embedding_model_name = model_name
        logger.info("Embedding model loaded successfully")

    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Get embedding for text.

    Args:
        text: Input text to embed

    Returns:
        Normalized embedding vector or None if model not loaded
    """
    global _embedding_model

    if _embedding_model is None:
        logger.error("Embedding model not loaded")
        return None

    try:
        # Get embedding (returns numpy array)
        embedding = _embedding_model.encode(text, convert_to_numpy=True)

        # Normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except Exception as e:
        logger.error("Failed to get embedding: %s", e)
        return None


def get_embedding_dim() -> Optional[int]:
    """Get embedding dimension."""
    global _embedding_model
    if _embedding_model is None:
        return None
    return _embedding_model.get_sentence_embedding_dimension()
