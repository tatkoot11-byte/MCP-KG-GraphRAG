import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graphrag_pipeline import build_passage_index

load_dotenv(PROJECT_ROOT / ".env")

INDEX_FILE = PROJECT_ROOT / "data" / "faiss" / "passages.index"
PASSAGES_FILE = PROJECT_ROOT / "data" / "faiss" / "passages.npy"

NEO4J_URI = os.getenv(
    "NEO4J_URI",
    "neo4j://localhost:7687",
)
NEO4J_USER = os.getenv(
    "NEO4J_USER",
    "neo4j",
)
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def load_vector_store():
    index = faiss.read_index(str(INDEX_FILE))

    passages = np.load(
        PASSAGES_FILE,
        allow_pickle=True,
    ).tolist()

    return index, passages


def vector_search(query: str, top_k: int = 3):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    query_vector = np.array(
        [embeddings.embed_query(query)],
        dtype="float32",
    )

    index, passages = load_vector_store()

    distances, indices = index.search(
        query_vector,
        top_k,
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0],
    ):
        if index_id < 0:
            continue

        passage = passages[index_id]

        results.append(
            {
                "passage_id": passage["passage_id"],
                "doc_id": passage["doc_id"],
                "text": passage["text"],
                "vector_distance": float(distance),
            }
        )

    return results


def graph_search(query: str, limit: int = 5):
    """
    Search Neo4j for relevant 1–2 hop graph paths.

    The graph traversal supports the project's allowed relationship
    types while preserving the complete node and relationship path
    for GraphRAG provenance.
    """

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(
            NEO4J_USER,
            NEO4J_PASSWORD,
        ),
    )

    try:
        with driver.session() as session:

            result = session.run(
                """
                MATCH path=(
                    person:Person
                )-[
                    :WORKS_ON
                    |REPORTS_TO
                    |MEMBER_OF
                    |VENDOR_OF
                    |MENTIONS
                *1..2]->(target)

                WHERE
                    toLower($search_term)
                        CONTAINS
                    toLower(
                        coalesce(target.name, "")
                    )
                    OR
                    toLower($search_term)
                        CONTAINS
                    toLower(
                        coalesce(target.title, "")
                    )
                    OR
                    toLower(
                        coalesce(target.name, "")
                    )
                        CONTAINS
                    toLower($search_term)
                    OR
                    toLower(
                        coalesce(target.title, "")
                    )
                        CONTAINS
                    toLower($search_term)

                RETURN
                    person.name AS person,
                    person.role AS role,

                    target.name AS target,

                    CASE
                        WHEN target:Project
                        THEN target.name
                        ELSE null
                    END AS project,

                    [
                        relationship IN relationships(path)
                        | type(relationship)
                    ] AS relationship_types,

                    [
                        node IN nodes(path)
                        | coalesce(
                            node.name,
                            node.id,
                            "unknown"
                        )
                    ] AS graph_path

                ORDER BY person.name

                LIMIT $limit
                """,
                search_term=query,
                limit=limit,
            )

            return [
                record.data()
                for record in result
            ]

    finally:
        driver.close()


def hybrid_search(
    query: str,
    vector_top_k: int = 3,
    graph_limit: int = 5,
):
    vector_results = vector_search(
        query=query,
        top_k=vector_top_k,
    )

    graph_results = graph_search(
        query=query,
        limit=graph_limit,
    )

    return {
        "query": query,
        "vector_results": vector_results,
        "graph_results": graph_results,
    }


if __name__ == "__main__":

    result = hybrid_search(
        "Who works on Project Aswan?",
        vector_top_k=3,
        graph_limit=5,
    )

    print("\n=== VECTOR RESULTS ===")

    for item in result["vector_results"]:
        print(
            f"{item['passage_id']} | "
            f"{item['doc_id']} | "
            f"distance="
            f"{item['vector_distance']:.4f}"
        )

    print("\n=== GRAPH RESULTS ===")

    for item in result["graph_results"]:

        print(
            f"{item.get('person', 'unknown')} | "
            f"{item.get('role', 'unknown')} | "
            f"{item.get('target', 'unknown')}"
        )

        print(
            "Relationships:",
            " -> ".join(
                item.get(
                    "relationship_types",
                    [],
                )
            ),
        )

        print(
            "Graph path:",
            " -> ".join(
                item.get(
                    "graph_path",
                    [],
                )
            ),
        )