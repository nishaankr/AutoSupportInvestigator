"""The local embedding model (default BAAI/bge-small-en-v1.5) — free, offline, deterministic.

Used at ingest (to cluster every record and index the canonicals) and at query time (to embed
the ticket). Loaded once, on first use.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from autosupport.config import settings


@cache
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


def embed(texts: list[str], batch_size: int = 128) -> np.ndarray:
    """L2-normalised embeddings, one row per input text, as float32. Normalised so a plain
    dot product is cosine similarity everywhere else in the codebase."""
    return _model().encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False
    ).astype(np.float32)
