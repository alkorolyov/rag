"""Tests for chemistry module."""

import pytest

from rag.chemistry import (
    validate_smiles,
    parse_smiles,
    canonicalize_smiles,
    smiles_to_inchi,
    smiles_to_inchikey,
    MolecularSimilarity,
    SimilarityResult,
    compute_similarity_matrix,
)


class TestParsers:
    """Tests for SMILES parsing functions."""

    def test_validate_smiles_valid(self):
        """Valid SMILES should return True."""
        assert validate_smiles("CCO") is True  # Ethanol
        assert validate_smiles("CC(=O)OC1=CC=CC=C1C(=O)O") is True  # Aspirin
        assert validate_smiles("c1ccccc1") is True  # Benzene

    def test_validate_smiles_invalid(self):
        """Invalid SMILES should return False."""
        assert validate_smiles("invalid") is False
        assert validate_smiles("") is False
        assert validate_smiles(None) is False
        assert validate_smiles("XYZ123") is False

    def test_parse_smiles_ethanol(self):
        """Parse ethanol and check properties."""
        info = parse_smiles("CCO")
        assert info.formula == "C2H6O"
        assert info.molecular_weight == pytest.approx(46.07, rel=0.01)
        assert info.num_atoms == 3  # Heavy atoms only (C, C, O)
        assert info.num_heavy_atoms == 3

    def test_parse_smiles_aspirin(self):
        """Parse aspirin and check properties."""
        info = parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert info.formula == "C9H8O4"
        assert info.molecular_weight == pytest.approx(180.16, rel=0.01)
        assert info.num_rings == 1

    def test_parse_smiles_invalid_raises(self):
        """Invalid SMILES should raise ValueError."""
        with pytest.raises(ValueError):
            parse_smiles("invalid")

    def test_canonicalize_smiles(self):
        """Canonicalize should normalize SMILES."""
        # Different representations of benzene
        canonical = canonicalize_smiles("c1ccccc1")
        assert canonical == "c1ccccc1"

        # Benzoic acid
        canonical = canonicalize_smiles("c1ccccc1C(=O)O")
        assert canonical is not None

    def test_canonicalize_invalid(self):
        """Invalid SMILES should return None."""
        assert canonicalize_smiles("invalid") is None

    def test_smiles_to_inchi(self):
        """Convert SMILES to InChI."""
        inchi = smiles_to_inchi("CCO")
        assert inchi is not None
        assert inchi.startswith("InChI=")

    def test_smiles_to_inchikey(self):
        """Convert SMILES to InChIKey."""
        inchikey = smiles_to_inchikey("CCO")
        assert inchikey is not None
        assert len(inchikey) == 27  # Standard InChIKey length


class TestSimilarity:
    """Tests for molecular similarity search."""

    @pytest.fixture
    def similarity(self):
        """Create MolecularSimilarity instance."""
        return MolecularSimilarity(fingerprint="morgan", radius=2)

    @pytest.fixture
    def test_compounds(self):
        """Test compound database."""
        return [
            {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "name": "Aspirin"},
            {"smiles": "OC(=O)C1=CC=CC=C1O", "name": "Salicylic acid"},
            {"smiles": "OC(=O)C1=CC=CC=C1", "name": "Benzoic acid"},
            {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "name": "Caffeine"},
            {"smiles": "CCO", "name": "Ethanol"},
        ]

    def test_compute_similarity_identical(self, similarity):
        """Identical molecules should have similarity 1.0."""
        sim = similarity.compute_similarity("CCO", "CCO")
        assert sim == pytest.approx(1.0)

    def test_compute_similarity_different(self, similarity):
        """Different molecules should have lower similarity."""
        sim = similarity.compute_similarity("CCO", "CCCCCCCCCC")
        assert 0.0 <= sim < 1.0

    def test_compute_similarity_similar(self, similarity):
        """Similar molecules should have high similarity."""
        # Ethanol vs Propanol
        sim = similarity.compute_similarity("CCO", "CCCO")
        assert sim > 0.4

    def test_find_similar(self, similarity, test_compounds):
        """Find similar compounds in database."""
        results = similarity.find_similar(
            query_smiles="CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            database=test_compounds,
            top_k=3,
            threshold=0.0,
        )
        assert len(results) == 3
        assert all(isinstance(r, SimilarityResult) for r in results)
        # First result should be aspirin itself (similarity 1.0)
        assert results[0].similarity == pytest.approx(1.0)
        assert results[0].name == "Aspirin"

    def test_find_similar_with_threshold(self, similarity, test_compounds):
        """Threshold should filter results."""
        results = similarity.find_similar(
            query_smiles="CCO",  # Ethanol
            database=test_compounds,
            top_k=10,
            threshold=0.5,
        )
        # All results should be above threshold
        assert all(r.similarity >= 0.5 for r in results)

    def test_find_similar_string_database(self, similarity):
        """Should work with simple string list."""
        smiles_list = ["CCO", "CCCO", "CCCCO", "c1ccccc1"]
        results = similarity.find_similar("CCO", smiles_list, top_k=2)
        assert len(results) == 2

    def test_fingerprint_types(self):
        """Test different fingerprint types."""
        for fp_type in ["morgan", "maccs", "rdkit"]:
            sim = MolecularSimilarity(fingerprint=fp_type)
            score = sim.compute_similarity("CCO", "CCCO")
            assert 0.0 <= score <= 1.0

    def test_bulk_similarity(self, similarity):
        """Test bulk similarity calculation."""
        database = ["CCO", "CCCO", "CCCCO", "c1ccccc1"]
        results = similarity.bulk_similarity("CCO", database, threshold=0.3)
        assert len(results) > 0
        # Results should be sorted by similarity
        similarities = [s for _, s in results]
        assert similarities == sorted(similarities, reverse=True)


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_compute_matrix(self):
        """Compute pairwise similarity matrix."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        matrix = compute_similarity_matrix(smiles)

        assert len(matrix) == 3
        assert len(matrix[0]) == 3
        # Diagonal should be 1.0
        for i in range(3):
            assert matrix[i][i] == 1.0
        # Should be symmetric
        assert matrix[0][1] == matrix[1][0]

    def test_matrix_with_invalid_smiles(self):
        """Matrix should handle invalid SMILES."""
        smiles = ["CCO", "invalid", "CCCO"]
        matrix = compute_similarity_matrix(smiles)
        # Should still return matrix, invalid entries are 0
        assert len(matrix) == 3
