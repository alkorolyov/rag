"""Prompt templates for RAG pipeline."""

from langchain_core.documents import Document

from rag.models import docs_to_context

SYSTEM_PROMPT = """You are a biomedical assistant that answers questions using only the provided context.

Rules:
1. Use ONLY information from the provided passages
2. Cite sources using [doc_id=X] format
3. If information is insufficient, say so
4. Be concise and direct

Output format:
<Answer in 1-3 sentences>

Key points:
- <point 1> [doc_id=X]
- <point 2> [doc_id=Y]

If insufficient: "I don't have enough information to answer this question."
"""


def get_user_prompt(question: str, docs: list[Document]) -> str:
    """Format user prompt with context and question."""
    context = docs_to_context(docs, include_score=True)
    return f"""CONTEXT:
{context}

QUESTION: {question}"""
