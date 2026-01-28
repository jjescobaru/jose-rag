# Minimal Local RAG System (Ollama + ChromaDB)

This project demonstrates a minimal Retrieval-Augmented Generation (RAG) system
running fully locally on a laptop CPU.

The system answers questions about condominium regulations by retrieving relevant
policy text and generating grounded responses with citations.

---

## Architecture Overview

Ingest → Chunk → Embed (Ollama) → Store (ChromaDB) → Retrieve → Augment → Generate (Ollama)

- **Embedding model:** nomic-embed-text (via Ollama)
- **LLM:** mistral:7b-instruct (via Ollama)
- **Vector store:** ChromaDB (local, persistent)

---

## Prerequisites

- Python 3.10+
- Ollama installed and running
- Git (optional, for cloning)

### Required Ollama models

```bash
ollama pull mistral:7b-instruct
ollama pull nomic-embed-text
```
## Setup

Create a virtual environment and install dependencies:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Running the Demo

Open the notebook:

```bash
jupyter notebook jose_rag.ipynb
```
Run cells top to bottom:

- Build the Chroma index
- Query the knowledge base
- Generate a grounded answer

## Dataset
The dataset consists of condominium regulations split into 10 text files (one per chapter), stored under the data/ directory.

Repeated boilerplate headers are removed during ingestion to improve retrieval quality.

## Notes / Design Choices

- Chunking uses fixed-size character windows with overlap.
- Low-quality chunks (very short or non-informative) are filtered.
- Citations are always required to ensure grounded responses.
- This is not production-ready but intentionally minimal and explainable.

## Repository Structure

```bash
src/
  ingest.py                       # load + chunk documents
  ollama_embed.py                 # embeddings via Ollama
  vectordb.py                     # ChromaDB wrapper
  retrieve_chroma.py              # vector retrieval
  rag.py                          # prompt augmentation
  generate.py                     # LLM generation
data/
  *.txt                           # condominium regulations
jose_rag.ipynb                    # interactive demo
Minimal RAG - Escobar Jose.pptx   # presentation in Power Point format
Minimal RAG - Escobar Jose.pdf    # presentation in PDF format
```



