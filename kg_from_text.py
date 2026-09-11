import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from neo4j import GraphDatabase
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "data" / "docs"

load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "TaharaCoNeo4j2026",
)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class Entity(BaseModel):
    type: str = Field(
        description="One of Person, Team, Project, Vendor"
    )
    name: str = Field(min_length=1)
    role: str | None = None


class Relation(BaseModel):
    source_type: str
    source: str
    relation: str
    target_type: str
    target: str


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


ALLOWED_ENTITY_TYPES = {
    "Person",
    "Team",
    "Project",
    "Vendor",
}

ALLOWED_RELATIONS = {
    "WORKS_ON",
    "MEMBER_OF",
    "REPORTS_TO",
    "VENDOR_OF",
    "MENTIONS",
}


def normalize_name(name: str) -> str:
    """Normalize whitespace and Unicode punctuation."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def normalize_entity(entity: Entity) -> Entity:
    """Normalize extracted entity names before writing to Neo4j."""
    return Entity(
        type=entity.type.strip(),
        name=normalize_name(entity.name),
        role=normalize_name(entity.role) if entity.role else None,
    )


def normalize_relation(relation: Relation) -> Relation:
    """Normalize relation endpoints and relation type."""
    return Relation(
        source_type=relation.source_type.strip(),
        source=normalize_name(relation.source),
        relation=relation.relation.strip().upper(),
        target_type=relation.target_type.strip(),
        target=normalize_name(relation.target),
    )


def extract_from_text(text: str) -> Extraction:
    """Extract a strict entity/relation graph from one document."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )

    prompt = f"""
Extract a knowledge graph from the document below.

Return ONLY valid JSON matching this schema:

{{
  "entities": [
    {{
      "type": "Person|Team|Project|Vendor",
      "name": "string",
      "role": "string or null"
    }}
  ],
  "relations": [
    {{
      "source_type": "Person|Team|Project|Vendor",
      "source": "string",
      "relation": "WORKS_ON|MEMBER_OF|REPORTS_TO|VENDOR_OF|MENTIONS",
      "target_type": "Person|Team|Project|Vendor",
      "target": "string"
    }}
  ]
}}

Rules:
- Extract only facts explicitly supported by the document.
- Do not invent entities or relationships.
- Normalize duplicate mentions to the same canonical name.
- Use the exact entity names appearing in the document.
- Use REPORTS_TO for reporting relationships.
- Use MEMBER_OF for people belonging to teams.
- Use WORKS_ON for people working on projects.
- Use VENDOR_OF for a vendor providing services to a project.
- Do not create relationships that are not supported by the document.

DOCUMENT:
{text}
"""

    response = llm.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        content = "".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    content = str(content).strip()

    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(r"\s*```$", "", content)

    data = json.loads(content)

    extraction = Extraction.model_validate(data)

    entities = [
        normalize_entity(entity)
        for entity in extraction.entities
        if entity.type in ALLOWED_ENTITY_TYPES
    ]

    relations = [
        normalize_relation(relation)
        for relation in extraction.relations
        if relation.relation in ALLOWED_RELATIONS
    ]

    return Extraction(
        entities=entities,
        relations=relations,
    )


def ensure_constraints(session):
    """Ensure deterministic unique entity keys exist."""
    queries = [
        """
        CREATE CONSTRAINT person_name_unique IF NOT EXISTS
        FOR (p:Person)
        REQUIRE p.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT team_name_unique IF NOT EXISTS
        FOR (t:Team)
        REQUIRE t.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT project_name_unique IF NOT EXISTS
        FOR (p:Project)
        REQUIRE p.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT vendor_name_unique IF NOT EXISTS
        FOR (v:Vendor)
        REQUIRE v.name IS UNIQUE
        """,
        """
        CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
        FOR (d:Doc)
        REQUIRE d.doc_id IS UNIQUE
        """,
    ]

    for query in queries:
        session.run(query).consume()


def write_document(session, doc_id: str, title: str):
    """Create or update a document node."""
    session.run(
        """
        MERGE (d:Doc {doc_id: $doc_id})
        SET d.title = $title
        """,
        doc_id=doc_id,
        title=title,
    ).consume()


def write_entity(session, entity: Entity):
    """Merge an entity using its normalized canonical name."""
    query = f"""
    MERGE (n:{entity.type} {{name: $name}})
    """

    parameters = {
        "name": entity.name,
    }

    if entity.role:
        query += "\nSET n.role = $role"
        parameters["role"] = entity.role

    session.run(query, **parameters).consume()


def write_relation(session, relation: Relation):
    """Merge a supported relationship between canonical entities."""
    query = f"""
    MATCH (s:{relation.source_type} {{name: $source}})
    MATCH (t:{relation.target_type} {{name: $target}})
    MERGE (s)-[:{relation.relation}]->(t)
    """

    session.run(
        query,
        source=relation.source,
        target=relation.target,
    ).consume()


def link_document_mentions(
    session,
    doc_id: str,
    extraction: Extraction,
):
    """Attach the source document to every extracted entity."""
    for entity in extraction.entities:
        session.run(
            f"""
            MATCH (d:Doc {{doc_id: $doc_id}})
            MATCH (e:{entity.type} {{name: $name}})
            MERGE (d)-[:MENTIONS]->(e)
            """,
            doc_id=doc_id,
            name=entity.name,
        ).consume()


def process_document(session, path: Path):
    """Extract, normalize and write one document."""
    print(f"\n=== PROCESSING {path.name} ===")

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    title = text.splitlines()[0].strip()

    doc_id = path.stem

    extraction = extract_from_text(text)

    write_document(
        session,
        doc_id,
        title,
    )

    for entity in extraction.entities:
        write_entity(session, entity)

    for relation in extraction.relations:
        write_relation(session, relation)

    link_document_mentions(
        session,
        doc_id,
        extraction,
    )

    print(
        f"Entities: {len(extraction.entities)} | "
        f"Relations: {len(extraction.relations)}"
    )


def main():
    print("=== KG FROM TEXT ===")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Docs directory: {DOCS_DIR}")

    documents = sorted(DOCS_DIR.glob("*.txt"))

    if not documents:
        raise RuntimeError(
            f"No .txt documents found in {DOCS_DIR}"
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        with driver.session() as session:
            session.run("RETURN 1").consume()
            print("Neo4j connection: OK")

            ensure_constraints(session)

            for document in documents:
                process_document(
                    session,
                    document,
                )

            print("\n=== KG BUILD COMPLETE ===")

    finally:
        driver.close()


if __name__ == "__main__":
    main()