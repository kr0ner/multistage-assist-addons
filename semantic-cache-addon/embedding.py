"""Local embedding service supporting both sentence-transformers and ONNX models.

Provides embeddings without external Ollama dependency.
Supports:
  - HuggingFace sentence-transformers models (loaded via SentenceTransformer)
  - ONNX Runtime models (for quantized/exported models)

Automatically detects ONNX model if model.onnx exists in the model directory.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("reranker.embedding")

# Global model instance (loaded once)
_embedding_model = None
_embedding_model_name = None
_embedding_backend = None  # "sentence-transformers" or "onnx"
_onnx_session = None
_onnx_tokenizer = None
_onnx_config = None


def load_embedding_model(model_name: str = "BAAI/bge-m3", device: str = "cpu"):
    """
    Load embedding model (auto-detects ONNX or sentence-transformers).

    Args:
        model_name: HuggingFace model name or path to local model directory
        device: Device to load on (cpu, cuda, mps) — only for sentence-transformers
    """
    global _embedding_model, _embedding_model_name, _embedding_backend
    global _onnx_session, _onnx_tokenizer, _onnx_config

    if _embedding_model_name == model_name and (_embedding_model is not None or _onnx_session is not None):
        logger.debug("Embedding model already loaded: %s", model_name)
        return

    # Detect if this is an ONNX model directory
    model_path = Path(model_name)
    is_onnx = model_path.is_dir() and (model_path / "model.onnx").exists()

    if is_onnx:
        _load_onnx_embedding(model_path)
    else:
        _load_sentence_transformer(model_name, device)


def _load_onnx_embedding(model_path: Path):
    """Load ONNX embedding model."""
    global _onnx_session, _onnx_tokenizer, _onnx_config, _embedding_model_name, _embedding_backend

    logger.info("Loading ONNX embedding model from %s ...", model_path)

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_path = model_path / "model.onnx"
        config_path = model_path / "model_config.json"

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        _onnx_session = ort.InferenceSession(str(onnx_path), sess_options)

        # Load tokenizer
        try:
            _onnx_tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        except OSError:
            logger.info("Tokenizer loading from model dir failed, falling back to xlm-roberta-base")
            _onnx_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

        # Load config
        _onnx_config = {"max_seq_length": 64, "pooling": "mean", "normalize": True}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                _onnx_config = json.load(f)

        _embedding_model_name = str(model_path)
        _embedding_backend = "onnx"
        logger.info("ONNX embedding model loaded (pooling=%s)", _onnx_config.get("pooling", "mean"))

    except Exception as e:
        logger.error("Failed to load ONNX embedding model: %s", e)
        raise


def _load_sentence_transformer(model_name: str, device: str):
    """Load sentence-transformers model."""
    global _embedding_model, _embedding_model_name, _embedding_backend

    logger.info("Loading embedding model: %s on %s", model_name, device)
    logger.info("This may take several minutes for large models...")

    try:
        from sentence_transformers import SentenceTransformer

        logger.info("Downloading/loading model weights...")
        
        try:
            _embedding_model = SentenceTransformer(
                model_name, 
                device=device,
                tokenizer_kwargs={"return_token_type_ids": False}
            )
        except TypeError:
            _embedding_model = SentenceTransformer(model_name, device=device)

        # Hard-patch for DistilBERT token_type_ids issue
        try:
            transformer_module = _embedding_model[0]
            if hasattr(transformer_module, 'auto_model') and transformer_module.auto_model.config.model_type == "distilbert":
                if hasattr(transformer_module, 'tokenizer') and "token_type_ids" in transformer_module.tokenizer.model_input_names:
                    transformer_module.tokenizer.model_input_names.remove("token_type_ids")
                    logger.info("Patched DistilBERT tokenizer: Removed 'token_type_ids' from model inputs.")
        except Exception as patch_error:
            logger.debug(f"Could not apply tokenizer patch: {patch_error}")

        _embedding_model_name = model_name
        _embedding_backend = "sentence-transformers"
        logger.info("Embedding model loaded successfully")

    except Exception as e:
        logger.error("Failed to load embedding model: %s", e)
        raise


def get_embedding(text: str) -> Optional[np.ndarray]:
    """
    Get embedding for text (works with both backends).

    Args:
        text: Input text to embed

    Returns:
        Normalized embedding vector or None if model not loaded
    """
    if _embedding_backend == "onnx":
        return _get_embedding_onnx(text)
    elif _embedding_backend == "sentence-transformers":
        return _get_embedding_st(text)
    else:
        logger.error("No embedding model loaded")
        return None


def _get_embedding_onnx(text: str) -> Optional[np.ndarray]:
    """Get embedding via ONNX Runtime."""
    try:
        max_len = _onnx_config.get("max_seq_length", 64)
        inputs = _onnx_tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        input_names = [inp.name for inp in _onnx_session.get_inputs()]
        feeds = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = inputs["input_ids"]
        if "attention_mask" in input_names:
            feeds["attention_mask"] = inputs["attention_mask"]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        outputs = _onnx_session.run(None, feeds)
        hidden_states = outputs[0]  # (1, seq_len, hidden_size)

        pooling = _onnx_config.get("pooling", "mean")
        if pooling == "mean":
            # Mean pooling with attention mask
            mask = inputs["attention_mask"].astype(np.float32)
            mask_expanded = np.expand_dims(mask, -1)
            embedding = (hidden_states * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1)
            embedding = embedding[0]
        else:
            # CLS pooling
            embedding = hidden_states[0, 0, :]

        # Normalize
        if _onnx_config.get("normalize", True):
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        return embedding.astype(np.float32)

    except Exception as e:
        logger.error("ONNX embedding failed: %s", e)
        return None


def _get_embedding_st(text: str) -> Optional[np.ndarray]:
    """Get embedding via sentence-transformers."""
    if _embedding_model is None:
        logger.error("Embedding model not loaded")
        return None

    try:
        embedding = _embedding_model.encode(text, convert_to_numpy=True)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    except Exception as e:
        logger.error("Failed to get embedding: %s", e)
        return None


def get_embedding_dim() -> Optional[int]:
    """Get embedding dimension."""
    if _embedding_backend == "onnx" and _onnx_session is not None:
        # Get hidden size from ONNX model output
        output_shape = _onnx_session.get_outputs()[0].shape
        if len(output_shape) == 3:
            return output_shape[2]  # (batch, seq, hidden)
        return output_shape[-1]
    elif _embedding_backend == "sentence-transformers" and _embedding_model is not None:
        return _embedding_model.get_sentence_embedding_dimension()
    return None
