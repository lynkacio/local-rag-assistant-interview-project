import faiss
import numpy as np


class VectorStore:
    """FAISS inner-product index over L2-normalized document embeddings."""

    def __init__(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("The vector store requires a non-empty 2D matrix.")

        self._vectors = np.ascontiguousarray(matrix.copy())
        faiss.normalize_L2(self._vectors)
        self._index = faiss.IndexFlatIP(self._vectors.shape[1])
        self._index.add(self._vectors)

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1).copy()
        if query.shape[1] != self._vectors.shape[1]:
            raise ValueError("Query and document embeddings must have matching dimensions.")
        faiss.normalize_L2(query)

        scores, indices = self._index.search(query, min(top_k, self._index.ntotal))
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]
