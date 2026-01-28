from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

BOILERPLATE = '### **INTERNAL REGULATIONS FOR COEXISTENCE AND ADMINISTRATION** **FOR THE VERTICAL CONDOMINIUM "TRIVENTO III"'

@dataclass(frozen=True)
class Document:
    source: str   # filename (chapter name)
    text: str     # full file content


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str

def is_bad_chunk(text: str) -> bool:
    t = text.strip()
    if len(t) < 80:          # too short to be meaningful
        return True
    alpha = sum(ch.isalpha() for ch in t)
    if alpha < 30:           # not enough real words
        return True
    return False

def load_documents(data_dir: str | Path) -> List[Document]:
    """
    Walk a directory, read .txt files, and return a list of Document objects.
    Files are loaded in a deterministic order.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    documents: List[Document] = []

    # Sort files alphabetically to ensure deterministic ordering
    for file_path in sorted(data_path.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        text = text.replace(BOILERPLATE, "")

        documents.append(
            Document(
                source=file_path.name,  # e.g. "Condominium regulations - chapter 3.txt"
                text=text,
            )
        )

    return documents


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []

    step = chunk_size - overlap
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Convert Documents into Chunk objects with deterministic IDs.
    Chunk ID format:
      "{source}::chunk_{i}"
    """
    all_chunks: List[Chunk] = []

    for doc in documents:
        pieces = chunk_text(
            doc.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # ✅ Filter BEFORE assigning chunk indices and IDs
        pieces = [p for p in pieces if not is_bad_chunk(p)]

        for i, piece in enumerate(pieces):
            chunk_id = f"{doc.source}::chunk_{i}"

            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=doc.source,
                    chunk_index=i,
                    text=piece,
                )
            )

    return all_chunks

