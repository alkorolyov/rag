"""Molecular similarity search using RDKit fingerprints."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class FingerprintType(str, Enum):
    """Supported fingerprint types."""

    MORGAN = "morgan"
    MACCS = "maccs"
    RDKIT = "rdkit"
    TOPOLOGICAL = "topological"
    ATOM_PAIR = "atom_pair"


@dataclass
class SimilarityResult:
    """Result of a similarity search."""

    smiles: str
    similarity: float
    chembl_id: str | None = None
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.similarity = round(self.similarity, 4)


class MolecularSimilarity:
    """RDKit-based molecular similarity search.

    Supports multiple fingerprint types and similarity metrics.

    Example:
        >>> sim = MolecularSimilarity(fingerprint="morgan")
        >>> results = sim.find_similar(
        ...     query_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        ...     database=["CCO", "CC(=O)O", ...],
        ...     top_k=10
        ... )
    """

    def __init__(
        self,
        fingerprint: Literal["morgan", "maccs", "rdkit", "topological", "atom_pair"] = "morgan",
        radius: int = 2,
        nbits: int = 2048,
    ):
        """Initialize similarity search.

        Args:
            fingerprint: Type of molecular fingerprint
            radius: Radius for Morgan fingerprints (default: 2)
            nbits: Number of bits for bit vector fingerprints (default: 2048)
        """
        self.fingerprint = fingerprint
        self.radius = radius
        self.nbits = nbits

        # Pre-initialize Morgan generator for efficiency
        if fingerprint == "morgan":
            self._morgan_gen = GetMorganGenerator(radius=radius, fpSize=nbits)
        else:
            self._morgan_gen = None

    def get_fingerprint(self, smiles: str):
        """Compute fingerprint for a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            RDKit fingerprint object

        Raises:
            ValueError: If SMILES is invalid
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if self.fingerprint == "morgan":
            return self._morgan_gen.GetFingerprint(mol)
        elif self.fingerprint == "maccs":
            return MACCSkeys.GenMACCSKeys(mol)
        elif self.fingerprint == "rdkit":
            return Chem.RDKFingerprint(mol, fpSize=self.nbits)
        elif self.fingerprint == "topological":
            return rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                mol, nBits=self.nbits
            )
        elif self.fingerprint == "atom_pair":
            return rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
                mol, nBits=self.nbits
            )
        else:
            raise ValueError(f"Unknown fingerprint type: {self.fingerprint}")

    def compute_similarity(self, smiles1: str, smiles2: str) -> float:
        """Compute Tanimoto similarity between two molecules.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string

        Returns:
            Tanimoto similarity (0.0 to 1.0)
        """
        fp1 = self.get_fingerprint(smiles1)
        fp2 = self.get_fingerprint(smiles2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def find_similar(
        self,
        query_smiles: str,
        database: list[str] | list[dict],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[SimilarityResult]:
        """Find similar compounds in a database.

        Args:
            query_smiles: Query SMILES string
            database: List of SMILES strings or dicts with 'smiles' key
                     (can include 'chembl_id', 'name', 'metadata')
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of SimilarityResult sorted by decreasing similarity
        """
        query_fp = self.get_fingerprint(query_smiles)
        results = []

        for item in database:
            # Handle both string and dict inputs
            if isinstance(item, str):
                smiles = item
                chembl_id = None
                name = None
                metadata = {}
            else:
                smiles = item.get("smiles")
                chembl_id = item.get("chembl_id")
                name = item.get("name")
                metadata = item.get("metadata", {})

            if not smiles:
                continue

            try:
                fp = self.get_fingerprint(smiles)
                similarity = DataStructs.TanimotoSimilarity(query_fp, fp)

                if similarity >= threshold:
                    results.append(
                        SimilarityResult(
                            smiles=smiles,
                            similarity=similarity,
                            chembl_id=chembl_id,
                            name=name,
                            metadata=metadata,
                        )
                    )
            except Exception:
                # Skip invalid SMILES
                continue

        # Sort by similarity (descending) and return top_k
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:top_k]

    def bulk_similarity(
        self,
        query_smiles: str,
        database: list[str],
        threshold: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Compute similarity to all compounds in database (optimized).

        Args:
            query_smiles: Query SMILES string
            database: List of SMILES strings
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity) tuples above threshold
        """
        query_fp = self.get_fingerprint(query_smiles)

        # Pre-compute all fingerprints
        fps = []
        valid_indices = []
        for i, smiles in enumerate(database):
            try:
                fp = self.get_fingerprint(smiles)
                fps.append(fp)
                valid_indices.append(i)
            except Exception:
                continue

        # Bulk similarity calculation
        similarities = DataStructs.BulkTanimotoSimilarity(query_fp, fps)

        # Filter by threshold
        results = [
            (valid_indices[i], sim)
            for i, sim in enumerate(similarities)
            if sim >= threshold
        ]

        return sorted(results, key=lambda x: x[1], reverse=True)


def compute_similarity_matrix(smiles_list: list[str], fingerprint: str = "morgan") -> list[list[float]]:
    """Compute pairwise similarity matrix for a list of compounds.

    Args:
        smiles_list: List of SMILES strings
        fingerprint: Fingerprint type to use

    Returns:
        NxN similarity matrix
    """
    sim = MolecularSimilarity(fingerprint=fingerprint)
    n = len(smiles_list)
    matrix = [[0.0] * n for _ in range(n)]

    # Compute fingerprints
    fps = []
    for smiles in smiles_list:
        try:
            fps.append(sim.get_fingerprint(smiles))
        except Exception:
            fps.append(None)

    # Compute pairwise similarities
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            if fps[i] is not None and fps[j] is not None:
                similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                matrix[i][j] = round(similarity, 4)
                matrix[j][i] = round(similarity, 4)

    return matrix
