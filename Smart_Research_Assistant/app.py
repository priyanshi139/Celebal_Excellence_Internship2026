"""
app.py
------
Smart Research Assistant — Streamlit UI

Ties together the full RAG pipeline (ingest -> embed/store -> retrieve
-> generate -> evaluate) into an interactive chat interface.

Run locally:
    streamlit run app.py

Author: Priyanshi | Celebal Excellence Data Science Internship 2026
Project: Smart Research Assistant (RAG-based Knowledge System)
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ingest import load_document, chunk_documents
from src.vectorstore import build_vectorstore, get_embedding_model
from src.rag_chain import build_rag_chain, ask
from src.evaluate import evaluate_answer

load_dotenv()

st.set_page_config(page_title="Smart Research Assistant", page_icon="📚", layout="wide")

# ---------------------------------------------------------------- Sidebar
st.sidebar.title("📚 Smart Research Assistant")
st.sidebar.caption("RAG-based document Q&A — Celebal Excellence Internship 2026")

backend_choice = st.sidebar.radio(
    "Vector store backend", ["faiss", "chroma"], index=0,
    help="FAISS = fast local prototyping. Chroma = persistent, production-like.",
)
llm_choice = st.sidebar.radio(
    "LLM backend", ["huggingface", "openai"], index=0,
    help="Hugging Face works with no API key. OpenAI needs OPENAI_API_KEY in .env.",
)
top_k = st.sidebar.slider("Chunks retrieved per query (k)", 2, 8, 4)

st.sidebar.divider()
uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF / TXT)", type=["pdf", "txt"], accept_multiple_files=True
)
process_btn = st.sidebar.button("🔄 Process documents", type="primary")

# ---------------------------------------------------------------- Session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------- Document processing
if process_btn:
    if not uploaded_files:
        st.sidebar.warning("Upload at least one document first.")
    else:
        with st.spinner("Chunking + embedding documents..."):
            all_docs = []
            for f in uploaded_files:
                suffix = Path(f.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                all_docs.extend(load_document(tmp_path))
                os.unlink(tmp_path)

            chunks = chunk_documents(all_docs)
            st.session_state.vectorstore = build_vectorstore(chunks, backend=backend_choice)
            st.session_state.chain = build_rag_chain(
                st.session_state.vectorstore, backend=llm_choice, k=top_k
            )
        st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

# ---------------------------------------------------------------- Main chat UI
st.title("Ask your documents")

if st.session_state.chain is None:
    st.info("👈 Upload PDFs/TXT files in the sidebar and click **Process documents** to get started.")
else:
    query = st.chat_input("Ask a question about your uploaded documents...")

    if query:
        with st.spinner("Retrieving context and generating answer..."):
            result = ask(st.session_state.chain, query)
            contexts = [doc.page_content for doc in result["sources"]]
            scores = evaluate_answer(query, result["answer"], contexts)

        st.session_state.chat_history.append(
            {"question": query, "answer": result["answer"], "sources": result["sources"], "scores": scores}
        )

    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])

            with st.expander("📄 Supporting document context"):
                for i, doc in enumerate(turn["sources"], 1):
                    st.markdown(f"**Chunk {i}** — `{doc.metadata.get('source', 'unknown')}`")
                    st.text(doc.page_content[:400] + "...")

            with st.expander("📊 Evaluation metrics (reliability)"):
                st.json(turn["scores"])
