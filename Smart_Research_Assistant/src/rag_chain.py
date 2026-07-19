"""
rag_chain.py
------------
Handles Steps 3 & 4 of the RAG pipeline: Query Processing + Answer
Generation.

Two LLM backends are supported:
  - "openai"      -> requires OPENAI_API_KEY, higher quality answers
  - "huggingface" -> free, local/hosted, no key needed (google/flan-t5-base)

This lets the project run end-to-end even before an OpenAI key is
available, which matters given the tight 14-day timeline.

Author: Priyanshi | Celebal Excellence Data Science Internship 2026
"""

import os
from typing import Literal

from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

RAG_PROMPT_TEMPLATE = """You are a precise research assistant. Use ONLY the
context below to answer the question. If the answer is not contained in the
context, say "I don't have enough information in the provided documents to
answer that."

Context:
{context}

Question: {question}

Answer (be concise and cite specific details from the context):"""


def get_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def get_llm(backend: Literal["openai", "huggingface"] = "huggingface"):
    if backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Either add it to .env or use "
                "LLM_BACKEND=huggingface instead."
            )
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    elif backend == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline

        pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256,
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"Unknown LLM backend: {backend}")


def build_rag_chain(vectorstore, backend: Literal["openai", "huggingface"] = "huggingface", k: int = 4):
    """
    Wire together retriever + LLM + prompt into a single RetrievalQA chain.
    `return_source_documents=True` lets the UI show which chunks backed
    the answer (the "Supporting document context" requirement).
    """
    llm = get_llm(backend)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": get_prompt()},
    )
    return chain


def ask(chain, question: str) -> dict:
    """
    Run a query through the RAG chain.
    Returns: { "answer": str, "sources": List[Document] }
    """
    result = chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": result.get("source_documents", []),
    }
