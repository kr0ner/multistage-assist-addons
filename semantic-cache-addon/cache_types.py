"""Shared types and constants for semantic cache.

Ported from Multi-Stage Assist reference implementation.
"""

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


# Default models (NUC-optimized)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Fast (118M params)
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # Smart (560M params, INT8 quantized on CPU)

# Configuration defaults
DEFAULT_RERANKER_THRESHOLD = 0.70  # Fallback for unknown domains
DEFAULT_VECTOR_THRESHOLD = 0.5  # Loose filter for candidate selection
DEFAULT_VECTOR_TOP_K = 10  # Number of candidates to rerank
DEFAULT_MAX_ENTRIES = 10000
MIN_CACHE_WORDS = 3

# Per-domain thresholds - optimized through systematic testing
DOMAIN_THRESHOLDS = {
    "light": 0.73,
    "switch": 0.73,
    "fan": 0.73,
    "cover": 0.73,
    "climate": 0.69,
}


@dataclass
class CacheEntry:
    """A cached command resolution."""

    text: str  # Original command text
    embedding: List[float]  # Embedding vector
    intent: str  # Resolved intent
    entity_ids: List[str]  # Resolved entity IDs
    slots: Dict[str, Any]  # Resolved slots
    required_disambiguation: bool = False  # True if user had to choose
    disambiguation_options: Optional[Dict[str, str]] = None  # {entity_id: name}
    hits: int = 0  # Number of times reused
    last_hit: str = ""  # ISO timestamp of last use
    verified: bool = True  # True if execution verified successful
    generated: bool = False  # True = pre-generated entry (from anchors.json)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        # Remove unknown fields that might be in old cache files
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)
