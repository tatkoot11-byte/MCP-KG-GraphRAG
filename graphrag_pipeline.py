from pathlib import Path
from typing import List, Dict


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"


def load_documents() -> List[Dict[str, str]]:
    """Load all source documents with stable document IDs."""
    documents = []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        documents.append(
            {
                "doc_id": path.stem,
                "title": path.stem,
                "text": path.read_text(encoding="utf-8"),
            }
        )

    return documents


def split_into_passages(
    text: str,
    doc_id: str,
    chunk_size: int = 80,
    overlap: int = 15,
) -> List[Dict[str, str]]:
    """Split text into overlapping passages with stable passage IDs."""
    words = text.split()

    if not words:
        return []

    passages = []
    start = 0
    passage_number = 1

    while start < len(words):
        end = min(start + chunk_size, len(words))
        passage_text = " ".join(words[start:end])

        passages.append(
            {
                "passage_id": f"{doc_id}_p{passage_number}",
                "doc_id": doc_id,
                "text": passage_text,
            }
        )

        if end >= len(words):
            break

        start = max(end - overlap, start + 1)
        passage_number += 1

    return passages


def build_passage_index() -> List[Dict[str, str]]:
    """Build the complete passage index from the source documents."""
    passages = []

    for document in load_documents():
        passages.extend(
            split_into_passages(
                text=document["text"],
                doc_id=document["doc_id"],
            )
        )

    return passages


if __name__ == "__main__":
    passages = build_passage_index()

    print(f"Documents loaded: {len(load_documents())}")
    print(f"Passages created: {len(passages)}")

    for passage in passages[:5]:
        print(
            f"{passage['passage_id']} | "
            f"{passage['doc_id']} | "
            f"{passage['text'][:120]}..."
        )