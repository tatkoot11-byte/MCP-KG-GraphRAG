import os
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from graphrag_pipeline import build_passage_index


load_dotenv(PROJECT_ROOT / ".env")

INDEX_DIR = PROJECT_ROOT / "data" / "faiss"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "passages.index"
PASSAGES_FILE = INDEX_DIR / "passages.npy"


def create_vector_store():
    passages = build_passage_index()

    if not passages:
        raise RuntimeError("No passages were found.")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    texts = [p["text"] for p in passages]
    vectors = embeddings.embed_documents(texts)

    matrix = np.array(vectors, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(INDEX_FILE))
    np.save(PASSAGES_FILE, np.array(passages, dtype=object), allow_pickle=True)

    print(f"Vector index created successfully.")
    print(f"Passages indexed: {len(passages)}")
    print(f"Embedding dimension: {matrix.shape[1]}")
    print(f"Index: {INDEX_FILE}")


if __name__ == "__main__":
    create_vector_store()