"""
vectorstore.py
--------------
Handles Step 2 of the RAG pipeline: Embedding + Storage.

Supports switching between FAISS (lightweight, local, great for
prototyping) and ChromaDB (persistent, closer to a production setup)
via a single flag — this satisfies the "Switch between FAISS and
ChromaDB" advanced feature from the project spec.

Author: Priyanshi | Celebal Excellence Data Science Internship 2026
"""

import os
from pathlib import Path
from typing import List, Literal

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
CHROMA_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma_db"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Free, local, offline-capable embedding model — no API key required.
    Swap `EMBEDDING_MODEL` env var for a different Hugging Face model
    (e.g. 'BAAI/bge-small-en-v1.5') without touching code.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore(
    chunks: List[Document],
    backend: Literal["faiss", "chroma"] = "faiss",
):
    """Embed chunks and build a fresh vectorstore from them."""
    embeddings = get_embedding_model()

    if backend == "faiss":
        store = FAISS.from_documents(chunks, embeddings)
        store.save_local(str(FAISS_INDEX_PATH))
        return store

    elif backend == "chroma":
        store = Chroma.from_documents(
            chunks, embeddings, persist_directory=str(CHROMA_DB_PATH)
        )
        store.persist()
        return store

    raise ValueError(f"Unknown backend: {backend}")


def load_vectorstore(backend: Literal["faiss", "chroma"] = "faiss"):
    """Load a previously persisted vectorstore from disk."""
    embeddings = get_embedding_model()

    if backend == "faiss":
        if not FAISS_INDEX_PATH.exists():
            return None
        return FAISS.load_local(
            str(FAISS_INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )

    elif backend == "chroma":
        if not CHROMA_DB_PATH.exists():
            return None
        return Chroma(persist_directory=str(CHROMA_DB_PATH), embedding_function=embeddings)

    raise ValueError(f"Unknown backend: {backend}")


def get_retriever(store, k: int = 4):
    """Return a retriever that fetches the top-k most relevant chunks."""
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})
