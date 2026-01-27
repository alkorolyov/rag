"""Biomedical Named Entity Recognition.

This module implements entity-aware indexing for the CV claim:
"incorporating entity-aware indexing"

Uses scispaCy for biomedical entity extraction.
"""

from dataclasses import dataclass, field
from typing import Literal

import spacy
from langchain_core.documents import Document


EntityType = Literal["GENE", "DISEASE", "CHEMICAL", "SPECIES", "CELL", "OTHER"]


@dataclass
class Entity:
    """Extracted named entity."""
    text: str
    label: str
    start: int
    end: int


@dataclass
class ExtractionResult:
    """Result of entity extraction."""
    entities: list[Entity] = field(default_factory=list)

    def by_type(self, label: str) -> list[str]:
        """Get entity texts by type."""
        return [e.text for e in self.entities if e.label == label]

    def to_dict(self) -> dict[str, list[str]]:
        """Convert to dict grouped by entity type."""
        result: dict[str, list[str]] = {}
        for e in self.entities:
            if e.label not in result:
                result[e.label] = []
            if e.text not in result[e.label]:  # dedupe
                result[e.label].append(e.text)
        return result

    def all_texts(self) -> list[str]:
        """Get all entity texts (deduplicated)."""
        return list(set(e.text for e in self.entities))


class BiomedicalNER:
    """Biomedical Named Entity Recognition using scispaCy.

    Extracts genes, diseases, chemicals, and other biomedical entities.

    Example:
        >>> ner = BiomedicalNER()
        >>> result = ner.extract("BRCA1 mutations increase breast cancer risk")
        >>> result.to_dict()
        {'GENE': ['BRCA1'], 'DISEASE': ['breast cancer']}
    """

    # Map scispaCy labels to our simplified types
    LABEL_MAP = {
        # en_core_sci_sm labels
        "ENTITY": "OTHER",
        # en_ner_bc5cdr_md labels (if using)
        "DISEASE": "DISEASE",
        "CHEMICAL": "CHEMICAL",
        # en_ner_bionlp13cg_md labels (if using)
        "GENE_OR_GENE_PRODUCT": "GENE",
        "CANCER": "DISEASE",
        "ORGAN": "ANATOMY",
        "CELL": "CELL",
        "ORGANISM": "SPECIES",
        "AMINO_ACID": "CHEMICAL",
        "SIMPLE_CHEMICAL": "CHEMICAL",
    }

    def __init__(self, model: str = "en_core_sci_sm"):
        """Initialize NER model.

        Args:
            model: spaCy model name. Options:
                - "en_core_sci_sm" (default, general biomedical)
                - "en_core_sci_lg" (larger, more accurate)
                - "en_ner_bc5cdr_md" (diseases + chemicals)
                - "en_ner_bionlp13cg_md" (genes, cells, organisms)
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            raise RuntimeError(
                f"Model '{model}' not found. Install with:\n"
                f"  pip install scispacy\n"
                f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/{model}-0.5.4.tar.gz"
            )
        self.model_name = model

    def extract(self, text: str) -> ExtractionResult:
        """Extract entities from text.

        Args:
            text: Input text

        Returns:
            ExtractionResult with extracted entities
        """
        doc = self.nlp(text)

        entities = [
            Entity(
                text=ent.text,
                label=self.LABEL_MAP.get(ent.label_, ent.label_),
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]

        return ExtractionResult(entities=entities)

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract entities from multiple texts efficiently.

        Args:
            texts: List of input texts

        Returns:
            List of ExtractionResult objects
        """
        results = []
        for doc in self.nlp.pipe(texts, batch_size=32):
            entities = [
                Entity(
                    text=ent.text,
                    label=self.LABEL_MAP.get(ent.label_, ent.label_),
                    start=ent.start_char,
                    end=ent.end_char,
                )
                for ent in doc.ents
            ]
            results.append(ExtractionResult(entities=entities))
        return results

    def enrich_documents(
        self,
        documents: list[Document],
        metadata_key: str = "entities",
    ) -> list[Document]:
        """Add extracted entities to document metadata.

        Args:
            documents: List of LangChain Documents
            metadata_key: Key to store entities in metadata

        Returns:
            Documents with entities added to metadata
        """
        texts = [doc.page_content for doc in documents]
        results = self.extract_batch(texts)

        for doc, result in zip(documents, results):
            doc.metadata[metadata_key] = result.to_dict()

        return documents

    def enrich_with_filter_fields(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """Add flattened entity fields for Qdrant filtering.

        Adds individual metadata fields like 'entity_GENE', 'entity_DISEASE'
        that can be used directly in Qdrant filters.

        Args:
            documents: List of LangChain Documents

        Returns:
            Documents with entity filter fields in metadata
        """
        texts = [doc.page_content for doc in documents]
        results = self.extract_batch(texts)

        for doc, result in zip(documents, results):
            entity_dict = result.to_dict()
            doc.metadata["entities"] = entity_dict

            # Add flattened fields for filtering
            for entity_type, entity_list in entity_dict.items():
                doc.metadata[f"entity_{entity_type}"] = entity_list

            # Add combined field for keyword matching
            doc.metadata["entity_all"] = result.all_texts()

        return documents
