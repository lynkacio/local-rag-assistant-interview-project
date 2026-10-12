from pathlib import Path


def load_documents(folder: str | Path = "docs") -> list[dict[str, str]]:
    """Load supported knowledge-base files with paths relative to the folder."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge-base directory does not exist: {root}")

    documents = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            documents.append(
                {
                    "source": f"docs/{path.relative_to(root).as_posix()}",
                    "text": path.read_text(encoding="utf-8"),
                }
            )
    return documents
