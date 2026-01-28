from __future__ import annotations

from typing import List

from src.retrieve import RetrievalResult
# Import the Chroma retrieval result type
from src.retrieve_chroma import RetrievalResult
from src.generate import generate_with_ollama


def build_augmented_prompt(
    query: str,
    retrieved: List[RetrievalResult],
) -> str:
    """
    Build a prompt that forces grounded answers + citations.
    Works with Chroma RetrievalResult (chunk_id/text/source).
    """
    # 1) System-style instructions (even though Ollama uses a single prompt string)
    rules = (
        "You are a helpful assistant answering questions about condominium regulations.\n"
        "Use ONLY the provided CONTEXT to answer.\n"
        "If the answer is not in the CONTEXT, say: \"I don't know based on the provided regulations.\"\n"
        "ALWAYS include citations.\n"
        "Every sentence that states a rule, time window, restriction, fine, or requirement MUST end with a citation\n"
        "in square brackets using the chunk_id, e.g. [Condominium regulations - chapter 5.txt::chunk_2].\n"
        "Be concise and factual.\n"
    )

    # 2) Format context with citations
    context_blocks = []
    for r in retrieved:
        chunk_id = r.chunk_id
        text = r.text.strip().replace("\n", " ")
        context_blocks.append(f"[{chunk_id}] {text}")

    context = "\n\n".join(context_blocks)

    # 3) Assemble final prompt
    prompt = (
        f"{rules}\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:\n"
    )

    return prompt

def answer_query_with_rag(
    query: str,
    retrieved: list[RetrievalResult],
    llm_model: str = "mistral:7b-instruct",
) -> str:
    prompt = build_augmented_prompt(query, retrieved)
    return generate_with_ollama(prompt, model=llm_model)
