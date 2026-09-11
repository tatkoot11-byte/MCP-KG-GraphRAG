from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"

mcp = FastMCP("TaharaCo MCP Server")


class SearchDocsInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class FetchDocInput(BaseModel):
    doc_id: str = Field(..., min_length=1)


class SummarizeTextInput(BaseModel):
    text: str = Field(..., min_length=1)
    max_words: int = Field(default=100, ge=10, le=500)


def load_documents() -> List[dict]:
    documents = []

    for path in sorted(DOCS_DIR.glob("*.txt")):
        documents.append(
            {
                "doc_id": path.stem,
                "text": path.read_text(encoding="utf-8"),
            }
        )

    return documents


def score_document(query: str, text: str) -> int:
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())

    return len(query_words.intersection(text_words))


@mcp.tool()
def search_docs(
    query: str = Field(..., min_length=1),
    top_k: int = Field(default=3, ge=1, le=10),
) -> dict:
    """
    Search TaharaCo documents and return the most relevant passages.
    """

    validated = SearchDocsInput(
        query=query,
        top_k=top_k,
    )

    documents = load_documents()
    scored = []

    for doc in documents:
        score = score_document(
            validated.query,
            doc["text"],
        )

        scored.append(
            {
                "doc_id": doc["doc_id"],
                "score": score,
                "text": doc["text"][:1000],
            }
        )

    scored.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "query": validated.query,
        "results": scored[:validated.top_k],
    }


@mcp.tool()
def fetch_doc(
    doc_id: str = Field(..., min_length=1),
) -> dict:
    """
    Fetch the complete text of a TaharaCo document.
    """

    validated = FetchDocInput(
        doc_id=doc_id,
    )

    path = DOCS_DIR / f"{validated.doc_id}.txt"

    if not path.exists():
        return {
            "success": False,
            "error": f"Document '{validated.doc_id}' was not found.",
        }

    return {
        "success": True,
        "doc_id": validated.doc_id,
        "text": path.read_text(encoding="utf-8"),
    }


@mcp.tool()
def summarize_text(
    text: str = Field(..., min_length=1),
    max_words: int = Field(default=100, ge=10, le=500),
) -> dict:
    """
    Create a bounded summary of the supplied text.
    """

    validated = SummarizeTextInput(
        text=text,
        max_words=max_words,
    )

    words = validated.text.split()

    if len(words) <= validated.max_words:
        summary = validated.text
    else:
        summary = " ".join(
            words[:validated.max_words]
        ) + "..."

    return {
        "summary": summary,
        "word_limit": validated.max_words,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")