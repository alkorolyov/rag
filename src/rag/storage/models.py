from typing import Any, Tuple, Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Represents a document or document chunk with metadata.

    Attributes:
        id: Unique identifier
            - For parent docs: "pubmed_12345678" (no '#' symbol)
            - For chunks: "pubmed_12345678#0" (includes '#' and chunk index)
        text: The text content (full document or chunk)
        score: Relevance score (used after retrieval/reranking)
        metadata: Additional metadata (e.g., source, date, parent_id)
        doc_type: Type of document - "parent" (full) or "chunk" (fragment)
    """
    id: str | int
    text: str
    score: float = 0.0
    doc_type: Literal["parent", "chunk"]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
        }

class SearchResult(BaseModel):
    """
    Represents a vector search result.

    Attributes:
        chunk_id: The chunk identifier
        score: Similarity score from vector search
        metadata: Optional metadata (for filtered search)
    """
    chunk_id: str
    score: float
    metadata: dict[str, Any] | None = None


# Chunk ID utilities
def make_chunk_id(doc_id: str | int, chunk_index: int) -> str:
    """
    Create a chunk ID from document ID and chunk index.

    Args:
        doc_id: The parent document identifier
        chunk_index: The index of this chunk within the document

    Returns:
        A formatted chunk ID string: "{doc_id}#{chunk_index}"

    Examples:
        >>> make_chunk_id("pubmed_12345", 0)
        "pubmed_12345#0"
        >>> make_chunk_id(12345, 2)
        "12345#2"
    """
    return f"{doc_id}#{chunk_index}"


def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    """
    Parse a chunk ID into document ID and chunk index.

    Args:
        chunk_id: The chunk identifier to parse

    Returns:
        A tuple of (doc_id, chunk_index)

    Raises:
        ValueError: If chunk_id format is invalid

    Examples:
        >>> parse_chunk_id("pubmed_12345#0")
        ("pubmed_12345", 0)
        >>> parse_chunk_id("12345#2")
        ("12345", 2)
    """
    try:
        doc_id, chunk_idx_str = chunk_id.rsplit("#", 1)
        chunk_index = int(chunk_idx_str)
        return doc_id, chunk_index
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Invalid chunk_id format: '{chunk_id}'. "
            f"Expected format: '{{doc_id}}#{{chunk_index}}'"
        ) from e
