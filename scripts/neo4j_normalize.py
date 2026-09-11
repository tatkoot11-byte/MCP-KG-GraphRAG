import re

from neo4j import GraphDatabase

from scripts.neo4j_reset import URI, USER, PASSWORD


def normalize_name(name: str) -> str:
    """Normalize whitespace and casing for consistent entity matching."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def normalize_existing_names(tx):
    """Normalize whitespace in existing named nodes without changing relationships."""
    result = tx.run("""
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN elementId(n) AS node_id, n.name AS name
    """)

    updates = [
        {
            "node_id": row["node_id"],
            "name": normalize_name(row["name"]),
        }
        for row in result
    ]

    for item in updates:
        tx.run("""
            MATCH (n)
            WHERE elementId(n) = $node_id
            SET n.name = $name
        """, **item)


def find_duplicate_people(tx):
    """Report duplicate Person names without modifying the graph."""
    result = tx.run("""
        MATCH (p:Person)
        WITH toLower(trim(p.name)) AS normalized_name,
             collect(p.name) AS names,
             count(*) AS total
        WHERE total > 1
        RETURN normalized_name, names, total
        ORDER BY normalized_name
    """)

    return [
        {
            "normalized_name": row["normalized_name"],
            "names": row["names"],
            "count": row["total"],
        }
        for row in result
    ]


def main():
    if not PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is missing from .env")

    driver = GraphDatabase.driver(
        URI,
        auth=(USER, PASSWORD),
    )

    try:
        with driver.session() as session:
            session.execute_write(normalize_existing_names)

            duplicates = session.execute_read(find_duplicate_people)

            print("Name normalization completed.")
            print(f"Duplicate Person groups found: {len(duplicates)}")

            for group in duplicates:
                print(group)

            print("No duplicate nodes were deleted.")
            print("Existing relationship types were preserved.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()