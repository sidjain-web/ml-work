"""A minimal Retrieval-Augmented Generation (RAG) pipeline in pure NumPy.

Demonstrates the retrieval half of RAG without external services:
  1. embed a corpus of documents (hashing bag-of-words -> fixed-dim vectors),
  2. store the vectors,
  3. embed a query and retrieve the top-k most similar documents (cosine),
  4. assemble a grounded prompt.

The final "generation" step is intentionally a stub: swap `answer()` for a call
to your LLM of choice (e.g. the `nano_gpt` model in this repo, or an API model)
using the retrieved context.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    """Deterministic hash (Python's built-in str hash is salted per process)."""
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:8], "big")


def embed(text: str, dim: int = 256) -> np.ndarray:
    """Hashing bag-of-words embedding.

    Each token is hashed into one of `dim` buckets; a signed hash reduces
    collisions. Cheap, deterministic, and dependency-free — good enough to
    illustrate retrieval. Replace with real sentence embeddings for production.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        h = _stable_hash(tok)
        bucket = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@dataclass
class VectorStore:
    dim: int = 256
    docs: List[str] = field(default_factory=list)
    _matrix: np.ndarray | None = None

    def add(self, documents: List[str]) -> None:
        self.docs.extend(documents)
        embeddings = np.stack([embed(d, self.dim) for d in self.docs])
        self._matrix = embeddings

    def search(self, query: str, k: int = 3) -> List[Tuple[float, str]]:
        if self._matrix is None or len(self.docs) == 0:
            return []
        q = embed(query, self.dim)
        scores = self._matrix @ q            # cosine sim (vectors are unit-norm)
        k = min(k, len(self.docs))
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.docs[i]) for i in top]


def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\nAnswer:"
    )


def answer(query: str, store: VectorStore, k: int = 3) -> str:
    """Retrieve, then generate. The generation step is a stub — plug in an LLM."""
    hits = store.search(query, k=k)
    contexts = [doc for _, doc in hits]
    prompt = build_prompt(query, contexts)
    # --- replace this block with a real LLM call over `prompt` -----------
    top = contexts[0] if contexts else "(no relevant context found)"
    generated = f"[stub answer — pass this prompt to an LLM] Most relevant: {top}"
    # ---------------------------------------------------------------------
    return generated, prompt, hits


if __name__ == "__main__":
    corpus = [
        "The mitochondrion is the powerhouse of the cell, producing ATP.",
        "Python is a high-level programming language known for readability.",
        "Photosynthesis converts sunlight, water and CO2 into glucose and oxygen.",
        "Transformers use self-attention to model relationships between tokens.",
        "The Great Barrier Reef is the world's largest coral reef system.",
        "Gradient descent iteratively updates parameters to minimize a loss.",
    ]
    store = VectorStore(dim=256)
    store.add(corpus)

    query = "How do neural networks with attention represent language?"
    generated, prompt, hits = answer(query, store, k=2)

    print("Query:", query)
    print("\nTop matches:")
    for score, doc in hits:
        print(f"  {score:+.3f}  {doc}")
    print("\nAssembled prompt:\n" + prompt)
    print("\nGeneration:\n" + generated)
