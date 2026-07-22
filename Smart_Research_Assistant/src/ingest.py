"""
ingest.py
---------
Handles Step 1 of the RAG pipeline: Data Ingestion.

Responsibilities:
  1. Load documents (PDF / TXT / web URL)
  2. Split them into overlapping chunks suitable for embedding
  3. Return a list of LangChain `Document` objects ready for the
     vectorstore module.

Author: Priyanshi | Celebal Excellence Data Science Internship 2026
Project: Smart Research Assistant (RAG-based Knowledge System)
"""

from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document


def _load_pdf_with_pypdf_directly(path: str) -> List[Document]:
    """
    Fallback PDF reader that calls pypdf directly instead of going through
    LangChain's PyPDFLoader. Used when PyPDFLoader returns empty text for
    every page (this can happen inconsistently across environments/versions
    even for PDFs that clearly contain selectable text).
    """
    from pypdf import PdfReader

    reader = PdfReader(path)
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        docs.append(Document(page_content=text, metadata={"source": path, "page": i}))
    return docs


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document based on its file extension.
    Supports: .pdf, .txt, .md

    For PDFs, tries LangChain's PyPDFLoader first, and falls back to a
    direct pypdf read if that yields no text at all (seen occasionally
    across different pypdf/environment versions). Raises a clear error
    if the PDF still has no extractable text (e.g. it's a scanned image
    with no OCR layer) instead of silently returning empty chunks.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()

        total_text = sum(len(d.page_content.strip()) for d in docs)
        if total_text == 0:
            docs = _load_pdf_with_pypdf_directly(str(path))
            total_text = sum(len(d.page_content.strip()) for d in docs)

        if total_text == 0:
            raise ValueError(
                f"No extractable text found in '{path.name}'. This usually means "
                "the PDF is a scanned image without a text layer. Please upload "
                "a PDF with selectable/typed text instead."
            )
        return docs

    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_from_url(url: str) -> List[Document]:
    """Load and parse web page content as a document."""
    loader = WebBaseLoader(url)
    return loader.load()


def load_directory(dir_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
    """Bulk-load every matching file inside a directory (e.g. data/sample_docs)."""
    loader = DirectoryLoader(dir_path, glob=glob_pattern, loader_cls=PyPDFLoader)
    return loader.load()


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.

    chunk_size / chunk_overlap are tunable — smaller chunks give more
    precise retrieval but can lose surrounding context; larger chunks
    do the opposite. 1000/150 is a reasonable default for policy docs
    and research papers.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    # Quick manual test: place a sample PDF in data/sample_docs/ and run
    # `python src/ingest.py` to sanity-check chunking.
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample_docs"
    pdfs = list(sample_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {sample_dir}. Add one to test ingestion.")
    else:
        docs = load_document(str(pdfs[0]))
        chunks = chunk_documents(docs)
        print(f"Loaded {len(docs)} page(s) -> split into {len(chunks)} chunks.")
        print("\nSample chunk:\n", chunks[0].page_content[:300])
