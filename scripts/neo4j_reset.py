import os

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def reset_database():
    if not PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is missing from .env")

    driver = GraphDatabase.driver(
        URI,
        auth=(USER, PASSWORD),
    )

    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Neo4j database reset successfully.")
    finally:
        driver.close()


if __name__ == "__main__":
    reset_database()