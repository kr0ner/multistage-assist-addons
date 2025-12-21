"""Cache loader for semantic cache files.

Loads pre-generated anchors and user-learned entries from JSON files.
Builds numpy matrix for fast cosine similarity search.
"""

import json
import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from cache_types import CacheEntry

logger = logging.getLogger("reranker.cache_loader")


class CacheLoader:
    """Load and manage semantic cache entries."""

    def __init__(
        self,
        anchors_file: str = "/homeassistant/.storage/multistage_assist_anchors.json",
        user_cache_file: str = "/homeassistant/.storage/multistage_assist_semantic_cache.json",
    ):
        """
        Initialize cache loader.

        Args:
            anchors_file: Path to pre-generated anchors JSON
            user_cache_file: Path to user-learned cache JSON
        """
        self.anchors_file = Path(anchors_file)
        self.user_cache_file = Path(user_cache_file)
        self._cache: List[CacheEntry] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._loaded = False
        self._lock = threading.RLock()
        self._last_reload_time: Optional[float] = None

    @property
    def entries(self) -> List[CacheEntry]:
        """Get all cache entries."""
        return self._cache

    @property
    def embeddings_matrix(self) -> Optional[np.ndarray]:
        """Get embeddings matrix for vector search."""
        return self._embeddings_matrix

    @property
    def is_loaded(self) -> bool:
        """Check if cache is loaded."""
        return self._loaded

    def load(self) -> Tuple[int, int]:
        """
        Load cache from disk.

        Returns:
            Tuple of (anchor_count, user_count)
        """
        anchor_count = 0
        user_count = 0

        # Load anchors (pre-generated entries)
        if self.anchors_file.exists():
            anchor_count = self._load_file(self.anchors_file, generated=True)
            logger.info("Loaded %d anchor entries from %s", anchor_count, self.anchors_file)

        # Load user-learned entries
        if self.user_cache_file.exists():
            user_count = self._load_file(self.user_cache_file, generated=False)
            logger.info("Loaded %d user entries from %s", user_count, self.user_cache_file)

        # Build embeddings matrix
        if self._cache:
            embeddings = [e.embedding for e in self._cache]
            self._embeddings_matrix = np.array(embeddings, dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self._embeddings_matrix, axis=1, keepdims=True)
            self._embeddings_matrix = self._embeddings_matrix / (norms + 1e-10)
            logger.info(
                "Built embeddings matrix: %s (%.1f MB)",
                self._embeddings_matrix.shape,
                self._embeddings_matrix.nbytes / 1024 / 1024,
            )

        self._loaded = True
        return anchor_count, user_count

    def _load_file(self, path: Path, generated: bool) -> int:
        """Load entries from a JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries = data.get("entries", [])
            count = 0

            for entry_data in entries:
                try:
                    # Remove legacy fields that might cause errors
                    entry_data.pop("is_anchor", None)
                    # Set generated flag
                    entry_data["generated"] = generated
                    entry = CacheEntry.from_dict(entry_data)
                    self._cache.append(entry)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to load entry: %s", e)

            return count

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in %s: %s", path, e)
            return 0
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            return 0

    def get_texts(self) -> List[str]:
        """Get all anchor texts for BM25 indexing."""
        return [e.text for e in self._cache]

    @property
    def last_reload_time(self) -> Optional[float]:
        """Get timestamp of last reload (or initial load)."""
        return self._last_reload_time

    def reload(self) -> Tuple[int, int]:
        """
        Thread-safe reload of cache from disk.

        Clears existing entries and reloads from both files.
        Returns:
            Tuple of (anchor_count, user_count)
        """
        import time

        with self._lock:
            logger.info("Reloading cache...")

            # Clear existing cache
            self._cache.clear()
            self._embeddings_matrix = None

            # Reload from disk
            result = self.load()

            # Update reload timestamp
            self._last_reload_time = time.time()

            logger.info(
                "Cache reloaded: %d anchors + %d user entries",
                result[0], result[1]
            )
            return result
