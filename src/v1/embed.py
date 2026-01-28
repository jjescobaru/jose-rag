from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest import Chunk


DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _cache_paths(cache_dir: str | Path) -> Dict[str, Path]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return {
        "meta": cache_path / "embeddings_meta.json",
        "npz": cache_path / "embeddings.npz",
    }


def embed_chunks(
    chunks: List[Chunk],
    model_name: str = DEFAULT_EMBED_MODEL,
    cache_dir: str | Path = "index",
    force_recompute: bool = False,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Returns:
      embeddings: np.ndarray shape (N, D)
      chunk_ids:  list[str] length N, aligned with embeddings
      meta:       dict with model_name + chunk metadata (sources, etc.)
    Caches results to disk for reuse.
    """
    paths = _cache_paths(cache_dir)

    # 1) If cache exists and not forcing recompute, load it
    if not force_recompute and paths["meta"].exists() and paths["npz"].exists():
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        npz = np.load(paths["npz"])
        embeddings = npz["embeddings"]
        chunk_ids = meta["chunk_ids"]
        return embeddings, chunk_ids, meta

    # 2) Otherwise compute embeddings
    model = SentenceTransformer(model_name)

    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    # normalize_embeddings=True makes cosine similarity easy later
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # 3) Save cache (embeddings + metadata)
    meta = {
        "model_name": model_name,
        "chunk_ids": chunk_ids,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
        "embedding_dim": int(embeddings.shape[1]),
    }

    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")
    np.savez_compressed(paths["npz"], embeddings=embeddings)

    return embeddings, chunk_ids, meta

