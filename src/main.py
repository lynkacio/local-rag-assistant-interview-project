import json
from pathlib import Path
from typing import Any

from .chunker import chunk_documents
from .embedding import TextEmbedder
from .generator import UNAVAILABLE_ANSWER, generate_answer
from .loader import load_documents
from .retriever import Embedder, Retriever
from .validator import validate


def run_pipeline(
    documents_dir: str | Path,
    questions_file: str | Path,
    embedder: Embedder | None = None,
) -> list[dict[str, str | list[str]]]:
    documents = load_documents(documents_dir)
    if not documents:
        raise ValueError(f"No .md or .txt documents found in {documents_dir}.")

    with Path(questions_file).open(encoding="utf-8") as file:
        questions: Any = json.load(file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a JSON array.")
    for index, item in enumerate(questions):
        if not isinstance(item, dict) or not isinstance(item.get("question"), str):
            raise ValueError(
                f"Question at index {index} must be an object with a string 'question'."
            )

    chunks = chunk_documents(documents)
    retriever = Retriever(chunks, embedder=embedder or TextEmbedder())
    results = []
    for item in questions:
        contexts = retriever.retrieve(item["question"])
        answer = generate_answer(contexts, item["question"])
        if not validate(answer, contexts):
            answer = {"answer": UNAVAILABLE_ANSWER, "citations": []}
        results.append(answer)
    return results


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    results = run_pipeline(project_root / "docs", project_root / "questions.json")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
