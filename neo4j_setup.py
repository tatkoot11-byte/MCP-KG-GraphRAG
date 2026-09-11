import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv(
    "NEO4J_PASSWORD",
    "TaharaCoNeo4j2026",
)


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)


CONSTRAINTS = [
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
    CREATE CONSTRAINT doc_id_unique IF NOT EXISTS
    FOR (d:Doc)
    REQUIRE d.doc_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT vendor_name_unique IF NOT EXISTS
    FOR (v:Vendor)
    REQUIRE v.name IS UNIQUE
    """,
]


QUERIES = [
    (
        "QUERY 1 - BASIC MATCH",
        """
        MATCH (p:Person)-[:WORKS_ON]->(pr:Project)
        RETURN p.name AS person,
               p.role AS role,
               pr.name AS project
        ORDER BY person
        """,
    ),
    (
        "QUERY 2 - FILTERED MATCH",
        """
        MATCH (p:Person)-[:REPORTS_TO]->(manager:Person)
        WHERE manager.name = "Omar Nabil"
        RETURN p.name AS employee,
               p.role AS employee_role,
               manager.name AS manager,
               manager.role AS manager_role
        ORDER BY employee
        """,
    ),
    (
        "QUERY 3 - MULTI-HOP TRAVERSAL",
        """
        MATCH (p:Person)-[:WORKS_ON]->(pr:Project)
              <-[:VENDOR_OF]-(v:Vendor)
        RETURN p.name AS person,
               pr.name AS project,
               v.name AS vendor
        ORDER BY person
        """,
    ),
]


def setup_constraints(session):
    print("\n=== CREATING CONSTRAINTS ===")

    for query in CONSTRAINTS:
        session.run(query).consume()

    print("All required uniqueness constraints are ready.")


def run_queries(session):
    for title, query in QUERIES:
        print(f"\n=== {title} ===")

        result = session.run(query)

        rows = [record.data() for record in result]

        if not rows:
            print("No rows returned.")
        else:
            for row in rows:
                print(row)


def main():
    print("=== NEO4J SETUP ===")
    print(f"URI: {NEO4J_URI}")

    try:
        with driver.session() as session:
            session.run("RETURN 1").consume()
            print("Neo4j connection: OK")

            setup_constraints(session)
            run_queries(session)

    except Exception as exc:
        print(f"Neo4j setup failed: {exc}")
        raise

    finally:
        driver.close()


if __name__ == "__main__":
    main()