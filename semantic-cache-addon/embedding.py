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
        device: Device to load on (cpu, cuda, mps)
    """
    global _embedding_model, _embedding_model_name

    if _embedding_model is not None and _embedding_model_name == model_name:
        logger.debug("Embedding model already loaded: %s", model_name)
        return

    logger.info("Loading embedding model: %s on %s", model_name, device)
    logger.info("This may take several minutes for large models...")

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Downloading/loading model weights...")
        
        # 1. Attempt to disable token_type_ids natively during initialization
        try:
            _embedding_model = SentenceTransformer(
                model_name, 
                device=device,
                tokenizer_kwargs={"return_token_type_ids": False}
            )
        except TypeError:
            # Fallback if the sentence-transformers version doesn't support tokenizer_kwargs
            _embedding_model = SentenceTransformer(model_name, device=device)

        # 2. Hard-patch the tokenizer to ensure DistilBERT doesn't crash
        try:
            transformer_module = _embedding_model[0]
            # Verify if the underlying model architecture is DistilBERT
            if hasattr(transformer_module, 'auto_model') and transformer_module.auto_model.config.model_type == "distilbert":
                if hasattr(transformer_module, 'tokenizer') and "token_type_ids" in transformer_module.tokenizer.model_input_names:
                    # Remove token_type_ids so it isn't passed to the forward pass
                    transformer_module.tokenizer.model_input_names.remove("token_type_ids")
                    logger.info("Patched DistilBERT tokenizer: Removed 'token_type_ids' from model inputs.")
        except Exception as patch_error:
            logger.debug(f"Could not apply tokenizer patch: {patch_error}")

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
