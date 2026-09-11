
import json
from pathlib import Path
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


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


ALLOWED_RELATIONSHIP_PAIRS = {
    ("Person", "WORKS_ON", "Project"),
    ("Person", "MEMBER_OF", "Team"),
    ("Person", "REPORTS_TO", "Person"),
    ("Person", "SUPERVISES", "Team"),
    ("Doc", "MENTIONS", "Person"),
    ("Doc", "MENTIONS", "Team"),
    ("Doc", "MENTIONS", "Project"),
    ("Doc", "MENTIONS", "Vendor"),
    ("Vendor", "VENDOR_OF", "Project"),
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

            # 1. Required node labels
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

            missing_labels = REQUIRED_NODE_LABELS - actual_labels

            checks["required_node_labels"] = {
                "required": sorted(REQUIRED_NODE_LABELS),
                "found": sorted(actual_labels),
                "missing": sorted(missing_labels),
                "passed": len(missing_labels) == 0,
            }

            # 2. Required relationship types
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

            missing_relationship_types = (
                REQUIRED_RELATIONSHIPS - actual_relationships
            )

            checks["required_relationship_types"] = {
                "required": sorted(REQUIRED_RELATIONSHIPS),
                "found": sorted(actual_relationships),
                "missing": sorted(missing_relationship_types),
                "passed": len(missing_relationship_types) == 0,
            }

            # 3. Duplicate entity rate
            duplicate_result = run_query(
                session,
                """
                MATCH (n)
                WHERE any(label IN labels(n)
                    WHERE label IN [
                        'Person',
                        'Team',
                        'Project',
                        'Doc',
                        'Vendor'
                    ])
                WITH labels(n)[0] AS label,
                     n.name AS name,
                     count(n) AS count
                WHERE name IS NOT NULL AND count > 1
                RETURN label, name, count
                ORDER BY label, name
                """,
            )

            total_named_entities = run_query(
                session,
                """
                MATCH (n)
                WHERE any(label IN labels(n)
                    WHERE label IN [
                        'Person',
                        'Team',
                        'Project',
                        'Doc',
                        'Vendor'
                    ])
                  AND n.name IS NOT NULL
                RETURN count(n) AS total
                """,
            )[0]["total"]

            duplicate_entity_count = sum(
                row["count"] - 1
                for row in duplicate_result
            )

            duplicate_entity_rate = (
                duplicate_entity_count / total_named_entities
                if total_named_entities
                else 0.0
            )

            checks["duplicate_entity_rate"] = {
                "duplicate_entities": duplicate_result,
                "duplicate_count": duplicate_entity_count,
                "total_named_entities": total_named_entities,
                "rate": duplicate_entity_rate,
                "passed": duplicate_entity_rate == 0.0,
            }

            # 4. Missing relation rate
            relation_type_count = run_query(
                session,
                """
                MATCH ()-[r]->()
                RETURN count(r) AS total
                """,
            )[0]["total"]

            existing_required_types = len(
                actual_relationships & REQUIRED_RELATIONSHIPS
            )

            missing_relation_rate = (
                (
                    len(REQUIRED_RELATIONSHIPS)
                    - existing_required_types
                )
                / len(REQUIRED_RELATIONSHIPS)
                if REQUIRED_RELATIONSHIPS
                else 0.0
            )

            checks["missing_relation_rate"] = {
                "required_relation_types": len(
                    REQUIRED_RELATIONSHIPS
                ),
                "found_required_relation_types": (
                    existing_required_types
                ),
                "total_graph_relationships": relation_type_count,
                "rate": missing_relation_rate,
                "passed": missing_relation_rate == 0.0,
            }

            # 5. Schema violations
            schema_violations = run_query(
                session,
                """
                MATCH (a)-[r]->(b)
                WITH labels(a) AS source_labels,
                     type(r) AS rel_type,
                     labels(b) AS target_labels
                RETURN source_labels,
                       rel_type,
                       target_labels
                """,
            )

            violations = []

            for row in schema_violations:
                source_labels = row["source_labels"]
                target_labels = row["target_labels"]
                rel_type = row["rel_type"]

                valid = False

                for source in source_labels:
                    for target in target_labels:
                        if (
                            source,
                            rel_type,
                            target,
                        ) in ALLOWED_RELATIONSHIP_PAIRS:
                            valid = True

                if not valid:
                    violations.append(row)

            checks["schema_violations"] = {
                "count": len(violations),
                "violations": violations,
                "passed": len(violations) == 0,
            }

            # 6. Project Aswan workers
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

            # 7. Project Aswan vendor
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
        "metrics": {
            "duplicate_entity_rate": checks[
                "duplicate_entity_rate"
            ]["rate"],
            "missing_relation_rate": checks[
                "missing_relation_rate"
            ]["rate"],
            "schema_violations": checks[
                "schema_violations"
            ]["count"],
        },
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
    print(
        "Duplicate entity rate:",
        checks["duplicate_entity_rate"]["rate"],
    )
    print(
        "Missing relation rate:",
        checks["missing_relation_rate"]["rate"],
    )
    print(
        "Schema violations:",
        checks["schema_violations"]["count"],
    )

    for name, check in checks.items():
        print(f"- {name}: {check['passed']}")

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

