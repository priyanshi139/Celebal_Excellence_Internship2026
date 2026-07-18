# 📚 Smart Research Assistant (RAG-based Knowledge System)

**Celebal Excellence Data Science Internship 2026**
**Intern:** Priyanshi | **Timeline:** 14 days

## Overview

A Retrieval-Augmented Generation (RAG) AI assistant that lets users upload
documents (PDFs, text files) and ask natural-language questions. The
assistant retrieves relevant chunks from a vector database and generates
accurate, context-aware, source-backed answers — with reliability scored
via evaluation metrics.

## Architecture

```
User Query
    │
    ▼
Query Embedding  ──────────────► Vector DB (FAISS / ChromaDB)
    │                                     │
    │                          top-k relevant chunks
    │                                     │
    ▼                                     ▼
              Prompt = Query + Retrieved Context
                          │
                          ▼
                    LLM (OpenAI / Hugging Face)
                          │
                          ▼
        Answer + Source Chunks + Evaluation Score
```

## Tech Stack

| Component        | Choice                                              |
|-------------------|------------------------------------------------------|
| Orchestration      | LangChain                                            |
| Vector Store       | FAISS (default, local) / ChromaDB (persistent)      |
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2` (free, HF)  |
| LLM (generation)   | Hugging Face `flan-t5-base` (free) or OpenAI GPT-4o-mini |
| Evaluation         | RAGAs (faithfulness, relevance, context precision)   |
| UI                 | Streamlit                                            |

## Project Structure

```
smart-research-assistant/
├── app.py                     # Streamlit UI — main entry point
├── requirements.txt
├── .env.example                # copy to .env and configure
├── src/
│   ├── ingest.py               # load + chunk documents
│   ├── vectorstore.py          # embeddings + FAISS/Chroma storage
│   ├── rag_chain.py            # retrieval + LLM answer generation
│   └── evaluate.py             # RAGAs-based reliability scoring
├── notebooks/
│   └── Smart_Research_Assistant_RAG_Demo.ipynb   # step-by-step walkthrough
└── data/
    └── sample_docs/            # put sample PDFs here for testing
```

## Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-username>/smart-research-assistant.git
cd smart-research-assistant

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Leave OPENAI_API_KEY blank to run fully free via Hugging Face,
# or add a key to enable higher-quality OpenAI generation + RAGAs scoring.
```

## Run

**Streamlit app:**
```bash
streamlit run app.py
```

**Notebook walkthrough:**
```bash
jupyter notebook notebooks/Smart_Research_Assistant_RAG_Demo.ipynb
```
(Add a sample PDF to `data/sample_docs/` first.)

## Example Scenario

1. Upload an HR policy PDF
2. Ask: *"What is the maternity leave policy?"*
3. Output:
   - A precise, grounded answer
   - The exact supporting document chunks used
   - An evaluation score indicating answer reliability

## Evaluation Metrics

- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevance** — does it actually answer the question asked?
- **Context Precision** — how much of the retrieved context is relevant?

*(Falls back to a lexical-overlap heuristic if no OpenAI key is set, so evaluation still works fully offline.)*

## 14-Day Build Plan

| Days  | Milestone                                                        |
|-------|--------------------------------------------------------------------|
| 1–2   | Repo setup, ingestion + chunking (`ingest.py`) tested on sample docs |
| 3–5   | Embeddings + FAISS vectorstore (`vectorstore.py`), verify retrieval quality |
| 6–8   | RAG chain + prompt tuning (`rag_chain.py`), test with real questions |
| 9–10  | Streamlit UI (`app.py`) — upload, chat, source display             |
| 11–12 | Evaluation module (`evaluate.py`) — RAGAs integration + fallback   |
| 13    | ChromaDB backend switch, polish UI, edge-case testing              |
| 14    | README, demo notebook, final GitHub push + submission              |

## Future Enhancements

- Voice-based interaction
- Multi-language support
- Cloud deployment (AWS / GCP)
- Domain-specific assistants (legal, healthcare, finance)
- User authentication for enterprise use

## Resources

- Dataset reference: https://www.kaggle.com/datasets/shivani284/smart-rag-system
