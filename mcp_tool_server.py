from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from dotenv import load_dotenv
import os


BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "data" / "docs"

load_dotenv(BASE_DIR / ".env")

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


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


@mcp.tool()
def search_docs(
    query: str = Field(..., min_length=1),
    top_k: int = Field(default=3, ge=1, le=10),
) -> dict:
    """
    Perform semantic search over TaharaCo documents.
    """

    validated = SearchDocsInput(
        query=query,
        top_k=top_k,
    )

    documents = load_documents()

    if not documents:
        return {
            "query": validated.query,
            "results": [],
        }

    embeddings = get_embeddings()

    query_embedding = embeddings.embed_query(
        validated.query
    )

    document_embeddings = embeddings.embed_documents(
        [doc["text"] for doc in documents]
    )

    scored = []

    for doc, doc_embedding in zip(
        documents,
        document_embeddings,
    ):
        score = cosine_similarity(
            query_embedding,
            doc_embedding,
        )

        scored.append(
            {
                "doc_id": doc["doc_id"],
                "score": round(score, 6),
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
    Generate an LLM-backed summary of the supplied text.
    """

    validated = SummarizeTextInput(
        text=text,
        max_words=max_words,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )

    prompt = f"""
Summarize the following text accurately.

Requirements:
- Preserve the important facts.
- Do not invent information.
- Keep the summary at or below {validated.max_words} words.
- Return only the summary.

Text:
{validated.text}
"""

    response = llm.invoke(prompt)

    summary = response.content.strip()

    words = summary.split()

    if len(words) > validated.max_words:
        summary = " ".join(
            words[:validated.max_words]
        ) + "..."

    return {
        "summary": summary,
        "word_limit": validated.max_words,
        "llm_backed": True,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")