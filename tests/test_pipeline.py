import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.chunker import chunk_documents
from src.generator import UNAVAILABLE_ANSWER, generate_answer
from src.loader import load_documents
from src.main import run_pipeline
from src.retriever import Retriever
from src.validator import validate


class FixedEmbedder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class DocumentLoadingTests(unittest.TestCase):
    def test_loads_supported_files_recursively_and_uses_relative_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "guide").mkdir()
            (root / "guide" / "setup.MD").write_text("Setup details.", encoding="utf-8")
            (root / "notes.txt").write_text("Text details.", encoding="utf-8")
            (root / "ignored.pdf").write_text("Not loaded.", encoding="utf-8")

            documents = load_documents(root)

        self.assertEqual(
            [document["source"] for document in documents],
            ["docs/guide/setup.MD", "docs/notes.txt"],
        )
        self.assertEqual(documents[0]["text"], "Setup details.")


class ChunkingTests(unittest.TestCase):
    def test_chunks_overlap_and_retain_document_metadata(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta"
        chunks = chunk_documents(
            [{"source": "docs/guide.txt", "text": text}], size=20, overlap=5
        )

        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            [chunk["chunk_index"] for chunk in chunks], list(range(len(chunks)))
        )
        self.assertTrue(all(chunk["source"] == "docs/guide.txt" for chunk in chunks))
        self.assertIn(chunks[0]["text"].split()[-1], chunks[1]["text"])

    def test_rejects_invalid_overlap(self) -> None:
        with self.assertRaises(ValueError):
            chunk_documents([], size=10, overlap=10)


class RetrievalTests(unittest.TestCase):
    def test_retrieves_by_vector_similarity_and_filters_irrelevant_matches(self) -> None:
        chunks = [
            {"source": "docs/api.md", "text": "API quota and request limits"},
            {"source": "docs/billing.md", "text": "Pricing and subscription details"},
        ]
        embedder = FixedEmbedder(
            {
                "API quota and request limits": [1.0, 0.0],
                "Pricing and subscription details": [0.0, 1.0],
                "How many API requests are allowed?": [0.9, 0.1],
                "What is the weather today?": [0.0, 0.0],
            }
        )
        retriever = Retriever(chunks, embedder=embedder, min_score=0.25)

        matches = retriever.retrieve("How many API requests are allowed?")
        unsupported_matches = retriever.retrieve("What is the weather today?")

        self.assertEqual(matches[0]["source"], "docs/api.md")
        self.assertGreater(matches[0]["score"], 0.99)
        self.assertEqual(unsupported_matches, [])


class CitationValidationTests(unittest.TestCase):
    def test_accepts_answer_grounded_in_its_cited_context(self) -> None:
        contexts = [
            {
                "source": "docs/api.md",
                "text": "API requests are limited to 1,000 per minute.",
            }
        ]
        answer = generate_answer(contexts, "What is the API limit?")

        self.assertTrue(validate(answer, contexts))
        self.assertEqual(answer["citations"], ["docs/api.md"])

    def test_rejects_unsupported_answer_and_invalid_citation(self) -> None:
        contexts = [
            {
                "source": "docs/api.md",
                "text": "API requests are limited to 1,000 per minute.",
            }
        ]

        self.assertFalse(
            validate(
                {"answer": "Requests are unlimited.", "citations": ["docs/api.md"]},
                contexts,
            )
        )
        self.assertFalse(
            validate(
                {"answer": contexts[0]["text"], "citations": ["docs/other.md"]},
                contexts,
            )
        )

    def test_unavailable_answer_is_valid_without_retrieved_context(self) -> None:
        answer = generate_answer([], "What is not in the documents?")

        self.assertEqual(answer["answer"], UNAVAILABLE_ANSWER)
        self.assertTrue(validate(answer, []))
        self.assertFalse(
            validate(answer, [{"source": "docs/api.md", "text": "Some context"}])
        )


class PipelineTests(unittest.TestCase):
    def test_replacement_questions_get_grounded_or_unavailable_json_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            documents_dir = root / "docs"
            documents_dir.mkdir()
            passage = "The service retains audit logs for 30 days."
            (documents_dir / "retention.txt").write_text(passage, encoding="utf-8")
            questions_file = root / "questions.json"
            questions_file.write_text(
                json.dumps(
                    [
                        {"id": "supported", "question": "How long are logs retained?"},
                        {
                            "id": "unsupported",
                            "question": "What is the weather today?",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            embedder = FixedEmbedder(
                {
                    passage: [1.0, 0.0],
                    "How long are logs retained?": [1.0, 0.0],
                    "What is the weather today?": [0.0, 1.0],
                }
            )

            results = run_pipeline(documents_dir, questions_file, embedder=embedder)

        self.assertEqual(
            results,
            [
                {"answer": passage, "citations": ["docs/retention.txt"]},
                {"answer": UNAVAILABLE_ANSWER, "citations": []},
            ],
        )
        self.assertTrue(all(set(result) == {"answer", "citations"} for result in results))


if __name__ == "__main__":
    unittest.main()
