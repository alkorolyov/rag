"""Minimal data models for RAG pipeline.

For most use cases, prefer LangChain's Document class directly:
    from langchain_core.documents import Document

This module provides utilities for chunk ID management and
compatibility with existing code.
"""

from typing import Any


def make_chunk_id(doc_id: str | int, chunk_index: int) -> str:
    """Create chunk ID from document ID and index."""
    return f"{doc_id}#{chunk_index}"


def parse_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Parse chunk ID into (doc_id, chunk_index)."""
    doc_id, idx = chunk_id.rsplit("#", 1)
    return doc_id, int(idx)


def docs_to_context(docs: list, include_score: bool = False) -> list[dict[str, Any]]:
    """Convert LangChain Documents to context format for prompts.

    Args:
        docs: List of LangChain Document objects
        include_score: Include score in output if available in metadata

    Returns:
        List of dicts with 'text', 'doc_id', and optionally 'score'
    """
    result = []
    for doc in docs:
        entry = {
            "text": doc.page_content,
            "doc_id": doc.metadata.get("doc_id", doc.metadata.get("id")),
        }
        if include_score and "score" in doc.metadata:
            entry["score"] = doc.metadata["score"]
        result.append(entry)
    return result
