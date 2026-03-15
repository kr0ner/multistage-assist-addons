"""BM25 index for hybrid search.

Provides keyword-based scoring to complement semantic vector search.
Includes German-specific tokenization with n-gram support.
"""

import logging
import re
from typing import List, Optional

import numpy as np

logger = logging.getLogger("reranker.bm25")

# Try to import BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank_bm25 not installed - hybrid search disabled")


def tokenize_german(text: str, ngram_size: int = 2) -> List[str]:
    """
    Tokenize German text for BM25 keyword matching.

    Args:
        text: Input text to tokenize
        ngram_size: Maximum n-gram size (1=words only, 2=words+bigrams, etc.)

    Handles German specifics:
    - Lowercasing with umlauts preserved
    - Removes punctuation but keeps compound words
    - Keeps action keywords like 'an', 'aus' as separate tokens
    - Optionally generates n-grams for phrase-level matching
    """
    text = text.lower()
    # Remove punctuation except hyphens (for compound words)
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Split into words
    words = text.split()
    # Filter very short tokens except important ones
    important_shorts = {'an', 'aus', 'auf', 'zu', 'ab', 'um', 'im', 'in'}
    words = [t for t in words if len(t) > 1 or t in important_shorts]

    if ngram_size <= 1:
        return words

    # Generate n-grams (bigrams, trigrams, etc.)
    tokens = words.copy()  # Start with unigrams
    for n in range(2, ngram_size + 1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i + n])
            tokens.append(ngram)

    return tokens


class BM25Index:
    """BM25 index for keyword-based retrieval."""

    def __init__(self, ngram_size: int = 2):
        """
        Initialize BM25 index.

        Args:
            ngram_size: N-gram size for tokenization (1=words, 2=bigrams)
        """
        self.ngram_size = ngram_size
        self._index: Optional[BM25Okapi] = None
        self._corpus: List[List[str]] = []

    @property
    def is_available(self) -> bool:
        """Check if BM25 is available."""
        return BM25_AVAILABLE

    @property
    def is_built(self) -> bool:
        """Check if index is built."""
        return self._index is not None

    def build(self, texts: List[str]) -> int:
        """
        Build BM25 index from texts.

        Args:
            texts: List of anchor texts to index

        Returns:
            Number of indexed documents
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping index build")
            return 0

        if not texts:
            logger.warning("No texts to index")
            return 0

        # Tokenize all texts
        self._corpus = [tokenize_german(text, self.ngram_size) for text in texts]

        # Build index
        self._index = BM25Okapi(self._corpus)

        ngram_desc = "words" if self.ngram_size == 1 else f"up to {self.ngram_size}-grams"
        logger.info("Built BM25 index for %d entries (%s)", len(self._corpus), ngram_desc)

        return len(self._corpus)

    def get_scores(self, query: str) -> np.ndarray:
        """
        Get BM25 scores for query against all indexed documents.

        Args:
            query: Query string

        Returns:
            Normalized scores in range [0, 1]
        """
        if self._index is None:
            return np.array([])

        # Tokenize query with same n-gram size
        tokens = tokenize_german(query, self.ngram_size)
        scores = np.array(self._index.get_scores(tokens))

        # Normalize. 
        # INSTEAD of min-max scaling to 1.0 (which inflates noise),
        # we scale relative to the maximum POSSIBLE score for this query
        # to ensure that if only 1 word matches, the score is appropriately low.
        
        # Approximate the 'perfect match' score for this index by tokenizing a typical 
        # doc that contains all tokens.
        if scores.max() > 0:
            # We use the max score in the current query as a baseline, 
            # but we damp it if it's too low compared to a "good" match.
            # A score of ~5-10 per token is typical for BM25.
            # If we have 5 tokens, a perfect match might be 30.
            # If the max score is only 5, it's a weak match.
            
            # Simple improvement: Scale by match quality.
            # If we just divide by max, noise like "ist das" becomes 1.0.
            # Let's cap the normalization factor so that poor max scores result in poor normalized scores.
            norm_factor = max(scores.max(), 5.0 * len(tokens)) 
            scores = scores / norm_factor
            
        return scores

    def __len__(self) -> int:
        """Get number of indexed documents."""
        return len(self._corpus)
