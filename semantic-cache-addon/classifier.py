"""Intent + domain classifier using ONNX Runtime.

Loads a quantized multi-task classifier (shared encoder + two heads)
and provides fast inference for German smart home commands.

The classifier pre-filters cache search space by predicted intent/domain,
eliminating the main confusion categories (wrong intent, wrong area).

Model directory layout expected:
    model.onnx          - ONNX encoder model (INT8 quantized)
    head_weights.npz    - numpy classification head weights
    model_config.json   - label maps and model metadata
    tokenizer.json      - HuggingFace tokenizer files
    vocab.txt
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("semantic-cache.classifier")

# Global state
_classifier_session = None
_classifier_tokenizer = None
_head_weights = None
_model_config = None
_loaded_path = None


def load_classifier(model_dir: str) -> bool:
    """Load classifier model from directory.

    Args:
        model_dir: Path to classifier model directory

    Returns:
        True if loaded successfully, False otherwise
    """
    global _classifier_session, _classifier_tokenizer, _head_weights, _model_config, _loaded_path

    model_path = Path(model_dir)
    onnx_path = model_path / "model.onnx"
    heads_path = model_path / "head_weights.npz"
    config_path = model_path / "model_config.json"

    if _loaded_path == str(model_dir):
        logger.debug("Classifier already loaded from %s", model_dir)
        return True

    if not onnx_path.exists():
        logger.warning("Classifier ONNX not found: %s", onnx_path)
        return False

    if not heads_path.exists():
        logger.warning("Classifier head weights not found: %s", heads_path)
        return False

    if not config_path.exists():
        logger.warning("Classifier config not found: %s", config_path)
        return False

    logger.info("Loading classifier from %s ...", model_dir)

    try:
        import onnxruntime as ort

        # Load ONNX model
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        _classifier_session = ort.InferenceSession(str(onnx_path), sess_options)

        # Load head weights
        _head_weights = dict(np.load(str(heads_path)))

        # Load config with label maps
        with open(config_path, encoding="utf-8") as f:
            _model_config = json.load(f)

        # Load tokenizer
        from transformers import AutoTokenizer
        _classifier_tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        _loaded_path = str(model_dir)
        logger.info(
            "Classifier loaded: %d intents, %d domains",
            _model_config.get("num_intents", 0),
            _model_config.get("num_domains", 0),
        )
        return True

    except Exception as e:
        logger.error("Failed to load classifier: %s", e)
        _classifier_session = None
        _classifier_tokenizer = None
        _head_weights = None
        _model_config = None
        _loaded_path = None
        return False


def is_loaded() -> bool:
    """Check if classifier is loaded and ready."""
    return _classifier_session is not None and _head_weights is not None


def classify(text: str) -> Optional[Dict]:
    """Classify text into intent + domain.

    Args:
        text: Input text (German smart home command)

    Returns:
        Dict with intent, domain, confidences, or None if not loaded.
        {
            "intent": "HassTurnOn",
            "intent_confidence": 0.98,
            "domain": "light",
            "domain_confidence": 0.99,
            "intent_logits": [...],
            "domain_logits": [...],
        }
    """
    if not is_loaded():
        return None

    try:
        max_len = _model_config.get("max_seq_length", 64)
        inputs = _classifier_tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Build feeds based on model inputs
        input_names = [inp.name for inp in _classifier_session.get_inputs()]
        feeds = {}
        if "input_ids" in input_names:
            feeds["input_ids"] = inputs["input_ids"]
        if "attention_mask" in input_names:
            feeds["attention_mask"] = inputs["attention_mask"]
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(inputs["input_ids"])

        # Run encoder
        outputs = _classifier_session.run(None, feeds)

        # CLS token pooling → head classification
        cls_embedding = outputs[0][:, 0, :]

        intent_logits = (
            cls_embedding @ _head_weights["intent_weight"].T
            + _head_weights["intent_bias"]
        )[0]
        domain_logits = (
            cls_embedding @ _head_weights["domain_weight"].T
            + _head_weights["domain_bias"]
        )[0]

        # Softmax for confidences
        def _softmax(x):
            e = np.exp(x - np.max(x))
            return e / e.sum()

        intent_probs = _softmax(intent_logits)
        domain_probs = _softmax(domain_logits)

        intent_id = int(np.argmax(intent_probs))
        domain_id = int(np.argmax(domain_probs))

        id2intent = _model_config.get("id2intent", {})
        id2domain = _model_config.get("id2domain", {})

        return {
            "intent": id2intent.get(str(intent_id), f"unknown_{intent_id}"),
            "intent_confidence": float(intent_probs[intent_id]),
            "domain": id2domain.get(str(domain_id), f"unknown_{domain_id}"),
            "domain_confidence": float(domain_probs[domain_id]),
            "intent_top3": _top_k(intent_probs, id2intent, 3),
            "domain_top3": _top_k(domain_probs, id2domain, 3),
        }

    except Exception as e:
        logger.error("Classification failed: %s", e)
        return None


def _top_k(probs: np.ndarray, id2label: Dict, k: int = 3) -> List[Dict]:
    """Return top-k predictions with labels and scores."""
    top_ids = np.argsort(probs)[::-1][:k]
    return [
        {"label": id2label.get(str(int(i)), f"unknown_{i}"), "score": float(probs[i])}
        for i in top_ids
    ]


def get_label_maps() -> Optional[Dict]:
    """Return the label maps from the loaded model config."""
    if _model_config is None:
        return None
    return {
        "id2intent": _model_config.get("id2intent", {}),
        "id2domain": _model_config.get("id2domain", {}),
        "intent2id": _model_config.get("intent2id", {}),
        "domain2id": _model_config.get("domain2id", {}),
    }
