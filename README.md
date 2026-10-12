# Local Knowledge-Base RAG Assistant

This command-line assistant answers questions from local Markdown and text files. It does not use a hosted language model or external knowledge: it extracts a concise, question-relevant sentence directly from retrieved context.

## Architecture

1. **Load:** Recursively read `.md` and `.txt` files from `docs/`. Citations use portable paths such as `docs/guides/api.md`.
2. **Chunk:** Split documents into overlapping text chunks while retaining source and chunk-index metadata.
3. **Embed:** Encode the chunks and questions with `sentence-transformers/all-MiniLM-L6-v2`.
4. **Retrieve:** Search a FAISS inner-product index of normalized embeddings (cosine similarity). Results below the relevance threshold are discarded.
5. **Generate:** Select the most question-relevant sentence from the top retrieved chunk, omitting Markdown headings. If the retrieved context contains no sentence that matches a meaningful term in the question, return `Information is not available in the knowledge base.`
6. **Validate:** Check that answer text is present in the retrieved context from the cited source. A failed check becomes the unavailable-information response.

The embedding model is downloaded by Sentence Transformers the first time the program runs. After that, it is cached locally.

## Installation

Use Python 3.10 or newer. From the project directory, install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Input files

Put any number of `.md` and `.txt` files under `docs/` (subdirectories are supported). `questions.json` must remain a JSON array of objects with a string `question` field; other fields, such as `id`, may be retained in the input:

```json
[
  {
    "id": "q1",
    "question": "What are the API limits?"
  }
]
```

## Run

From the project directory:

```bash
python -m src.main
```

The program writes a JSON array to standard output. Each item has the required answer and citation fields:

```json
[
  {
    "answer": "API limits are described in the API section.",
    "citations": [
      "docs/example.md"
    ]
  }
]
```

Answers are concise extracts from the cited retrieved text. Questions without a relevant supported sentence receive the exact unavailable-information answer and an empty citation list.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests cover document discovery, chunking, vector retrieval, citation checks, and unsupported questions. The retrieval tests use deterministic embedding vectors so they do not need to download the model.

## Project structure

```text
docs/                  Local knowledge base
questions.json         Questions to answer
src/
  loader.py            Document discovery and loading
  chunker.py           Overlapping text chunks
  embedding.py         Sentence Transformers embeddings
  vector_store.py      FAISS similarity index
  retriever.py         Retrieval and relevance threshold
  generator.py         Question-relevant sentence extraction
  validator.py         Citation and support validation
  main.py              Pipeline and CLI entry point
tests/                 Unit tests
requirements.txt       Python dependencies
```
