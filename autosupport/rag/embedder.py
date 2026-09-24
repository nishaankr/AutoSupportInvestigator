"""Local sentence-transformers wrapper (architecture.md §2, default BAAI/bge-small-en-v1.5).

Built here at CP1 rather than CP2: `ingest/cluster.py` needs embeddings for every English
record (not just canonicals) to cluster, and `ingest/index.py` needs them to upsert
canonicals into Chroma, so ingest cannot run without this module existing first. CP2 adds
`rag/dense.py` (query-time similarity search) on top of it; this module is unchanged by that.
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
