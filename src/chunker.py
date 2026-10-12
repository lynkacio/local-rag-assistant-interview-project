from collections.abc import Iterable


def chunk_documents(
    documents: Iterable[dict[str, str]], size: int = 500, overlap: int = 100
) -> list[dict[str, str | int]]:
    """Split documents into overlapping character chunks while retaining metadata."""
    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Chunk overlap must be non-negative and smaller than size.")

    chunks = []
    for document in documents:
        text = document["text"].strip()
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + size // 2, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "source": document["source"],
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            if end >= len(text):
                break
            start = max(start + 1, end - overlap)

    return chunks
