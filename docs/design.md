# Design Notes

## 1. Overview

This project implements a lightweight Retrieval-Augmented Generation (RAG) assistant that answers questions using only information retrieved from a local knowledge base.

The main design goal is to provide grounded answers with traceable citations while avoiding unsupported responses.

---

## 2. Requirements

The system should:

- Load local Markdown and text documents from the `docs/` directory.
- Split documents into manageable chunks.
- Generate semantic embeddings for document chunks.
- Retrieve relevant information using vector similarity search.
- Generate answers only from retrieved context.
- Return citations for supported answers.
- Reject questions that cannot be answered from the knowledge base.

---

## 3. Architecture
User Questions
|
v
questions.json
|
v
Document Loader
|
v
Chunking
|
v
Sentence Transformer Embedding
|
v
FAISS Vector Search
|
v
Retriever
|
v
Grounded Answer Generator
|
v
Citation Validator
|
v
JSON Output


---

## 4. Components

### Document Loader

Responsibilities:

- Scan local knowledge base files.
- Support `.md` and `.txt` formats.
- Preserve document source metadata for citations.

---

### Chunker

Responsibilities:

- Split large documents into smaller overlapping chunks.
- Maintain chunk-level metadata.

Reason:

Smaller chunks improve retrieval accuracy and allow more precise citations.

---

### Embedding Model

Technology:

- Sentence Transformers

Purpose:

Convert text into dense vectors representing semantic meaning.

---

### Vector Store

Technology:

- FAISS

Purpose:

Perform efficient similarity search over document embeddings.

---

### Retriever

Responsibilities:

- Embed user queries.
- Search the FAISS index.
- Return the most relevant document chunks.

---

### Generator

Responsibilities:

- Produce concise answers from retrieved context.
- Avoid using external knowledge.
- Return unavailable response when context is insufficient.

---

### Validator

Responsibilities:

- Verify citation availability.
- Check that answers are supported by retrieved context.
- Prevent unsupported responses.

---

## 5. Design Decisions

### Why FAISS?

FAISS provides efficient local vector similarity search without requiring external infrastructure.

This makes the project easy to reproduce and suitable for offline evaluation.

---

### Why Sentence Transformers?

Sentence Transformers provide lightweight semantic embeddings that are suitable for local inference.

---

### Why Extractive Generation?

The assignment prioritizes grounded answers and citation reliability.

The current implementation uses context-based answer extraction to ensure responses are fully supported by the knowledge base.

A generative LLM layer can be added on top of retrieved context in future versions.

---

## 6. Testing Strategy

The project includes automated tests covering:

- Document loading
- Chunking
- Semantic retrieval
- Citation validation
- Unsupported question handling

Run:python -m pytest
Expected:8 passed


---

## 7. Future Improvements

Potential extensions:

- Add LLM-based generation.
- Support additional document formats.
- Add persistent vector indexes.
- Add API service layer.
