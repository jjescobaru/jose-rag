from __future__ import annotations

import requests


def generate_with_ollama(
    prompt: str,
    model: str = "mistral:7b-instruct",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.2,
    timeout: int = 180,
) -> str:
    """
    Send a prompt to a local Ollama model and return the generated text.
    """
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

