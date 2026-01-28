from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float  # cosine similarity (higher = more similar)


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a query string into a normalized vector.
    Returns shape (D,)
    """
    vec = model.encode([query], normalize_embeddings=True)
    return np.asarray(vec[0], dtype=np.float32)

def retrieve_top_k(
    query: str,
    chunks: List[Chunk],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 5,
    min_score: float = 0.0,
    exclude_phrases: List[str] | None = None,
) -> List[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError("chunks and embeddings must have the same length")

    q = embed_query(query, model)
    scores = embeddings @ q  # cosine similarity

    # Rank all indices from best to worst
    ranked_idx = np.argsort(scores)[::-1]

    results: List[RetrievalResult] = []
    for idx in ranked_idx:
        score = float(scores[int(idx)])
        if score < min_score:
            break

        text_lower = chunks[int(idx)].text.lower()
        if exclude_phrases:
            if any(p.lower() in text_lower for p in exclude_phrases):
                continue

        results.append(RetrievalResult(chunk=chunks[int(idx)], score=score))
        if len(results) >= top_k:
            break

    return results

