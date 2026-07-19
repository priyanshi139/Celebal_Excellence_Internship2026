"""
evaluate.py
-----------
Handles Step 5 of the RAG pipeline: Evaluation.

Uses RAGAs to score each answer on:
  - Faithfulness       : is the answer grounded in the retrieved context?
  - Answer Relevance   : does the answer actually address the question?
  - Context Precision  : how much of the retrieved context is relevant?

Note: Faithfulness and Answer Relevance require an LLM judge (OpenAI by
default in RAGAs). If no OPENAI_API_KEY is set, this module degrades
gracefully to a simple lexical-overlap heuristic so the app still shows
*some* reliability signal without failing.

Author: Priyanshi | Celebal Excellence Data Science Internship 2026
"""

import os
from typing import List, Dict


def _lexical_overlap_score(answer: str, contexts: List[str]) -> float:
    """
    Lightweight fallback metric (no API key needed): fraction of the
    answer's significant words that also appear in the retrieved
    context. Not a substitute for RAGAs, but a useful offline sanity
    signal while an OpenAI key isn't available.
    """
    context_text = " ".join(contexts).lower()
    answer_words = [w.strip(".,!?") for w in answer.lower().split() if len(w) > 4]
    if not answer_words:
        return 0.0
    matched = sum(1 for w in answer_words if w in context_text)
    return round(matched / len(answer_words), 3)


def evaluate_answer(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """
    Evaluate a single Q/A pair. Tries RAGAs first (requires
    OPENAI_API_KEY); falls back to the lexical heuristic otherwise.
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision

            data = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                }
            )
            result = evaluate(
                data, metrics=[faithfulness, answer_relevancy, context_precision]
            )
            return {k: round(v, 3) for k, v in result.items()}
        except Exception as e:
            print(f"[evaluate.py] RAGAs evaluation failed, falling back. Reason: {e}")

    return {
        "lexical_overlap_score": _lexical_overlap_score(answer, contexts),
        "note": "RAGAs skipped (no OPENAI_API_KEY) — showing offline heuristic instead.",
    }
