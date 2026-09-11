import json
from pathlib import Path

from neo4j import GraphDatabase
from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD is missing from .env")


OUTPUT_FILE = PROJECT_ROOT / "evaluation" / "graph_quality_results.json"


REQUIRED_NODE_LABELS = {
    "Person",
    "Team",
    "Project",
    "Doc",
    "Vendor",
}

REQUIRED_RELATIONSHIPS = {
    "WORKS_ON",
    "MEMBER_OF",
    "REPORTS_TO",
    "MENTIONS",
    "VENDOR_OF",
}


def run_query(session, query):
    result = session.run(query)
    return [record.data() for record in result]


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    checks = {}

    try:
        with driver.session() as session:

            # Check required node labels.
            labels_result = run_query(
                session,
                """
                MATCH (n)
                UNWIND labels(n) AS label
                RETURN DISTINCT label
                ORDER BY label
                """,
            )

            actual_labels = {
                row["label"]
                for row in labels_result
            }

            checks["required_node_labels"] = {
                "required": sorted(REQUIRED_NODE_LABELS),
                "found": sorted(actual_labels),
                "passed": REQUIRED_NODE_LABELS.issubset(
                    actual_labels
                ),
            }

            # Check required relationship types.
            relationships_result = run_query(
                session,
                """
                MATCH ()-[r]->()
                RETURN DISTINCT type(r) AS relationship_type
                ORDER BY relationship_type
                """,
            )

            actual_relationships = {
                row["relationship_type"]
                for row in relationships_result
            }

            checks["required_relationship_types"] = {
                "required": sorted(REQUIRED_RELATIONSHIPS),
                "found": sorted(actual_relationships),
                "passed": REQUIRED_RELATIONSHIPS.issubset(
                    actual_relationships
                ),
            }

            # Check duplicate Person names.
            duplicate_people = run_query(
                session,
                """
                MATCH (p:Person)
                WITH p.name AS name, count(p) AS count
                WHERE count > 1
                RETURN name, count
                ORDER BY name
                """,
            )

            checks["duplicate_person_names"] = {
                "duplicates": duplicate_people,
                "passed": len(duplicate_people) == 0,
            }

            # Check required Project Aswan graph paths.
            project_paths = run_query(
                session,
                """
                MATCH (p:Person)-[:WORKS_ON]->(pr:Project)
                WHERE pr.name = 'Project Aswan'
                RETURN p.name AS person
                ORDER BY person
                """,
            )

            checks["project_aswan_workers"] = {
                "results": project_paths,
                "expected_minimum": 4,
                "passed": len(project_paths) >= 4,
            }

            # Check vendor relationship.
            vendor_paths = run_query(
                session,
                """
                MATCH (v:Vendor)-[:VENDOR_OF]->(pr:Project)
                WHERE pr.name = 'Project Aswan'
                RETURN v.name AS vendor
                ORDER BY vendor
                """,
            )

            checks["project_aswan_vendor"] = {
                "results": vendor_paths,
                "passed": len(vendor_paths) >= 1,
            }

    finally:
        driver.close()

    overall_passed = all(
        check["passed"]
        for check in checks.values()
    )

    output = {
        "overall_passed": overall_passed,
        "checks": checks,
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=== GRAPH QUALITY CHECKS ===")
    print(f"Overall passed: {overall_passed}")

    for name, check in checks.items():
        print(f"- {name}: {check['passed']}")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()