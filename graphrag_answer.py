import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from scripts.hybrid_retriever import hybrid_search


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def build_context(result: dict) -> str:
    context_parts = []

    context_parts.append("VECTOR EVIDENCE:")

    for item in result["vector_results"]:
        context_parts.append(
            f"""
Passage ID: {item['passage_id']}
Document: {item['doc_id']}
Text: {item['text']}
"""
        )

    context_parts.append("\nGRAPH EVIDENCE:")

    for item in result["graph_results"]:
        context_parts.append(
            f"""
Person: {item['person']}
Role: {item['role']}
Project: {item['project']}
Graph path: {' -> '.join(item['graph_path'])}
"""
        )

    return "\n".join(context_parts)


def generate_answer(query: str) -> dict:
    retrieval = hybrid_search(
        query=query,
        vector_top_k=3,
        graph_limit=5,
    )

    context = build_context(retrieval)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    prompt = f"""
You are a GraphRAG assistant for TaharaCo.

Answer the user's question using ONLY the evidence provided below.

Important rules:
1. Do not invent facts.
2. Prefer graph evidence for relationships.
3. Use vector evidence for supporting document passages.
4. Include provenance in the final answer.
5. Mention the relevant passage IDs.
6. Mention the graph paths when graph evidence is used.
7. If the evidence is insufficient, clearly say so.

USER QUESTION:
{query}

EVIDENCE:
{context}

Return:
1. A concise answer.
2. Evidence / provenance.
"""

    response = llm.invoke(prompt)

    answer = response.content

    return {
        "query": query,
        "answer": answer,
        "vector_provenance": [
            {
                "passage_id": item["passage_id"],
                "doc_id": item["doc_id"],
            }
            for item in retrieval["vector_results"]
        ],
        "graph_provenance": [
            {
                "graph_path": item["graph_path"],
                "person": item["person"],
                "project": item["project"],
            }
            for item in retrieval["graph_results"]
        ],
    }


if __name__ == "__main__":

    query = "Who works on Project Aswan?"

    result = generate_answer(query)

    print("\n=== GRAPHRAG ANSWER ===")
    print(result["answer"])

    print("\n=== VECTOR PROVENANCE ===")

    for item in result["vector_provenance"]:
        print(
            f"{item['passage_id']} | "
            f"{item['doc_id']}"
        )

    print("\n=== GRAPH PROVENANCE ===")

    for item in result["graph_provenance"]:
        print(
            " -> ".join(item["graph_path"])
        )