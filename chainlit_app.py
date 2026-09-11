import os
from pathlib import Path

import chainlit as cl
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from scripts.hybrid_retriever import hybrid_search


PROJECT_ROOT = Path(__file__).resolve().parent

load_dotenv(PROJECT_ROOT / ".env")


def build_context(retrieval: dict) -> str:
    parts = []

    parts.append("VECTOR EVIDENCE:")

    for item in retrieval["vector_results"]:
        parts.append(
            f"""
Passage ID: {item['passage_id']}
Document: {item['doc_id']}
Text: {item['text']}
"""
        )

    parts.append("\nGRAPH EVIDENCE:")

    for item in retrieval["graph_results"]:
        parts.append(
            f"""
Person: {item['person']}
Role: {item['role']}
Project: {item['project']}
Graph path: {' -> '.join(item['graph_path'])}
"""
        )

    return "\n".join(parts)


def generate_graphrag_answer(query: str, retrieval: dict) -> str:
    context = build_context(retrieval)

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    prompt = f"""
You are a GraphRAG assistant for TaharaCo.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not invent facts.
2. Only state that a person works on a project if the graph evidence
   contains a WORKS_ON relationship.
3. Use vector evidence as supporting document evidence.
4. Include provenance.
5. Mention relevant passage IDs.
6. Mention graph paths when graph evidence is used.
7. If something is only a reporting relationship, do not describe it
   as a WORKS_ON relationship.
8. If the evidence is insufficient, say so clearly.

USER QUESTION:
{query}

EVIDENCE:
{context}

Return:
- A concise answer.
- Evidence / provenance.
"""

    response = llm.invoke(prompt)

    if isinstance(response.content, list):
        return "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in response.content
        )

    return str(response.content)


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "## TaharaCo GraphRAG Assistant\n\n"
            "Ask a question about Project Aswan, the teams, "
            "people, reporting chain, or vendors."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):

    query = message.content.strip()

    if not query:
        await cl.Message(
            content="Please enter a question."
        ).send()
        return

    # Step 1: Hybrid Retrieval
    async with cl.Step(name="1. Hybrid Retrieval") as step:

        retrieval = hybrid_search(
            query=query,
            vector_top_k=3,
            graph_limit=5,
        )

        step.output = (
            f"Vector results: "
            f"{len(retrieval['vector_results'])}\n"
            f"Graph results: "
            f"{len(retrieval['graph_results'])}"
        )

    # Step 2: Vector Evidence
    async with cl.Step(name="2. Vector Search") as step:

        lines = []

        for item in retrieval["vector_results"]:
            lines.append(
                f"- {item['passage_id']} "
                f"| preview: {str(item.get('text', ''))[:200]}"
                f"({item['doc_id']}) "
                f"distance={item['vector_distance']:.4f}"
            )

        step.output = "\n".join(lines) or "No vector results."

    # Step 3: Graph Evidence
    async with cl.Step(name="3. Neo4j Graph Search") as step:

        lines = []

        for item in retrieval["graph_results"]:
            path = " -> ".join(item["graph_path"])

            lines.append(
                f"- {path}"
            )

        step.output = "\n".join(lines) or "No graph results."

    # Step 4: Generate Answer
    async with cl.Step(name="4. GraphRAG Answer Generation") as step:

        answer = generate_graphrag_answer(
            query=query,
            retrieval=retrieval,
        )

        step.output = answer

    # Step 5: Provenance
    async with cl.Step(name="5. Provenance") as step:

        provenance_lines = []

        provenance_lines.append("Vector passages:")

        for item in retrieval["vector_results"]:
            provenance_lines.append(
                f"- {item['passage_id']} | {item['doc_id']}"
            )

        provenance_lines.append("\nGraph paths:")

        for item in retrieval["graph_results"]:
            provenance_lines.append(
                f"- {' -> '.join(item['graph_path'])}"
            )

        step.output = "\n".join(provenance_lines)

    await cl.Message(
        content=answer
    ).send()