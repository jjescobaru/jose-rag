from dataclasses import dataclass
from typing import List

from src.ingest import Chunk
from src.vectordb import query_collection


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    distance: float  # smaller is better


def retrieve_top_k_chroma(collection, top_k: int, query_embedding: List[float]) -> List[RetrievalResult]:
    res = query_collection(collection, query_embedding=query_embedding, top_k=top_k)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    # Chroma returns distances; smaller is better.
    # For a demo-friendly "higher is better", convert:
    # score = -distance  (simple)
    results: List[RetrievalResult] = []
    for i in range(len(ids)):
        md = metas[i]
        distance = float(dists[i])
        results.append(
            RetrievalResult(
                chunk_id=ids[i],
                source=md.get("source", ""),
                chunk_index=int(md.get("chunk_index", -1)),
                text=docs[i],
                distance=distance,
            )
        )
    return results

