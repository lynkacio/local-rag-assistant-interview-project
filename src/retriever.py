from collections.abc import Sequence
from typing import Protocol

import numpy as np

from .embedding import TextEmbedder
from .vector_store import VectorStore


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Retriever:
    """Embed the corpus once and retrieve chunks by cosine similarity."""

    def __init__(
        self,
        chunks: list[dict[str, str | int]],
        top_k: int = 3,
        min_score: float = 0.25,
        embedder: Embedder | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self._chunks = chunks
        self._top_k = top_k
        self._min_score = min_score
        self._embedder = embedder or TextEmbedder()
        self._store = None

        if chunks:
            vectors = self._embedder.encode([str(chunk["text"]) for chunk in chunks])
            if vectors.shape[0] != len(chunks):
                raise ValueError("The embedder must return one vector per chunk.")
            self._store = VectorStore(vectors)

    def retrieve(self, query: str) -> list[dict[str, str | int | float]]:
        if not query.strip():
            raise ValueError("Question must not be empty.")
        if self._store is None:
            return []

        query_vector = self._embedder.encode([query])
        matches = self._store.search(query_vector[0], self._top_k)
        return [
            {**self._chunks[index], "score": score}
            for index, score in matches
            if score >= self._min_score
        ]
