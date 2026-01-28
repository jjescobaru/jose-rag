import requests
from typing import List


def ollama_embed(
    texts: List[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    timeout: int = 180,
) -> List[List[float]]:
    """
    Returns a list of embedding vectors (list[float]) aligned with input texts.
    Uses Ollama's embeddings API.
    """
    url = f"{base_url}/api/embed"
    payload = {"model": model, "input": texts}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    # Ollama returns embeddings for each input text
    # Key is typically "embeddings"
    return data["embeddings"]

