"""Chemistry tools for molecular analysis."""

from rag.chemistry.parsers import (
    MoleculeInfo,
    validate_smiles,
    parse_smiles,
    canonicalize_smiles,
    smiles_to_inchi,
    smiles_to_inchikey,
    name_to_smiles,
    get_pubchem_info,
)
from rag.chemistry.similarity import (
    FingerprintType,
    SimilarityResult,
    MolecularSimilarity,
    compute_similarity_matrix,
)
from rag.chemistry.chembl import (
    CompoundInfo,
    ActivityData,
    ChEMBLDatabase,
)

__all__ = [
    # Parsers
    "MoleculeInfo",
    "validate_smiles",
    "parse_smiles",
    "canonicalize_smiles",
    "smiles_to_inchi",
    "smiles_to_inchikey",
    "name_to_smiles",
    "get_pubchem_info",
    # Similarity
    "FingerprintType",
    "SimilarityResult",
    "MolecularSimilarity",
    "compute_similarity_matrix",
    # ChEMBL
    "CompoundInfo",
    "ActivityData",
    "ChEMBLDatabase",
]
