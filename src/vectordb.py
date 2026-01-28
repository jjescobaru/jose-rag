import chromadb
from chromadb.config import Settings
from typing import List, Optional, Dict, Any

from src.ingest import Chunk


def get_chroma_client(persist_dir: str = "chroma_db"):
    return chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))


def get_or_create_collection(client, name: str = "condo_rules"):
    return client.get_or_create_collection(name=name)


def upsert_chunks(
    collection,
    chunks: List[Chunk],
    embeddings: List[List[float]],
):
    ids = [c.chunk_id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)


def query_collection(
    collection,
    query_embedding: List[float],
    top_k: int = 5,
):
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

