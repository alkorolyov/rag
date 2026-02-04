"""Tests for chemistry module."""

import pytest

from rag.chemistry import (
    validate_smiles,
    parse_smiles,
    MolecularSimilarity,
    SimilarityResult,
)
# Import non-exported functions directly for testing
from rag.chemistry.parsers import canonicalize_smiles, smiles_to_inchi, smiles_to_inchikey
from rag.chemistry.similarity import compute_similarity_matrix


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


class TestSearchByTarget:
    """Tests for ChEMBL target-based search.

    Uses small targets for fast tests:
    - PLK1 (~52 activities) - has T210D mutation
    - Adenosine receptor A2a (~67 activities) - small protein target
    - B-raf (~10k activities) - only for filtered tests (min_pchembl, min_phase)
    """

    @pytest.fixture(scope="class")
    def chembl_db(self):
        """Create ChEMBL database connection (shared across class)."""
        from pathlib import Path
        from rag.chemistry.chembl import ChEMBLDatabase

        db_path = Path.home() / "data/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
        if not db_path.exists():
            pytest.skip(f"ChEMBL database not found at {db_path}")
        db = ChEMBLDatabase(db_path)
        yield db
        db.close()

    def test_search_by_target_basic(self, chembl_db):
        """Test basic target search returns results (uses potency filter for speed)."""
        from rag.chemistry.chembl import TargetActivityResult

        # Use B-raf with potency filter - fast because filters early
        results = chembl_db.search_by_target("B-raf", min_pchembl=7.0, limit=10)
        assert len(results) > 0
        assert all(isinstance(r, TargetActivityResult) for r in results)
        assert all("B-RAF" in r.target_name.upper() for r in results if r.target_name)

    def test_search_by_target_with_mutation(self, chembl_db):
        """Test mutation filter (uses potency filter for speed)."""
        # B-raf V600E with potency filter
        results = chembl_db.search_by_target("B-raf", mutation="V600E", min_pchembl=6.0, limit=10)
        assert len(results) > 0
        assert all(r.mutation and "V600E" in r.mutation for r in results)

    def test_search_by_target_approved_drugs(self, chembl_db):
        """Test min_phase filter for approved drugs (fast - filters early)."""
        # B-raf with min_phase=4 is fast because it filters early
        results = chembl_db.search_by_target("B-raf", min_phase=4, limit=20)
        assert len(results) > 0
        assert all(r.max_phase is not None and r.max_phase >= 4 for r in results)
        # Known BRAF inhibitors should appear
        names = {r.compound_name.upper() for r in results if r.compound_name}
        known_braf_drugs = {"VEMURAFENIB", "DABRAFENIB", "ENCORAFENIB"}
        assert names & known_braf_drugs, f"Expected one of {known_braf_drugs} in {names}"

    def test_search_by_target_activity_type(self, chembl_db):
        """Test activity_type filter."""
        # Use Adenosine receptor A2a - small target
        results = chembl_db.search_by_target("Adenosine receptor A2a", activity_type="IC50", limit=10)
        assert len(results) > 0
        assert all(r.activity_type == "IC50" for r in results)

    def test_search_by_target_potency_filter(self, chembl_db):
        """Test min_pchembl filter (fast - filters early)."""
        # B-raf with min_pchembl=8.0 is fast because it filters early
        results = chembl_db.search_by_target("B-raf", min_pchembl=8.0, limit=10)
        assert len(results) > 0
        assert all(r.pchembl_value is not None and r.pchembl_value >= 8.0 for r in results)

    def test_search_by_target_no_results(self, chembl_db):
        """Test nonexistent target returns empty list."""
        # Use potency filter to make the scan faster even for non-existent target
        results = chembl_db.search_by_target("NONEXISTENT_TARGET_XYZ123", min_pchembl=9.0)
        assert results == []

    def test_search_by_target_deduplicate(self, chembl_db):
        """Test deduplication returns unique compounds."""
        # Use B-raf with filters for speed
        results = chembl_db.search_by_target("B-raf", min_pchembl=7.0, deduplicate=True, limit=20)
        ids = [r.compound_chembl_id for r in results]
        assert len(ids) == len(set(ids)), "Deduplicated results should have unique compound IDs"

    def test_search_by_target_herg_safety(self, chembl_db):
        """Test hERG/KCNH2 search with potency filter (fast)."""
        # KCNH2 with min_pchembl filter is fast
        results = chembl_db.search_by_target("KCNH2", min_pchembl=7.0, limit=10)
        assert len(results) > 0
        assert all(r.pchembl_value >= 7.0 for r in results)

    def test_search_by_target_result_fields(self, chembl_db):
        """Test that all expected fields are populated."""
        # Use B-raf with potency filter for speed
        results = chembl_db.search_by_target("B-raf", min_pchembl=8.0, limit=5)
        assert len(results) > 0

        r = results[0]
        # Required fields should be present
        assert r.compound_chembl_id is not None
        assert r.smiles is not None
        assert r.activity_type is not None
        assert r.target_name is not None
        assert r.target_chembl_id is not None
        assert r.pchembl_value is not None

    def test_search_by_target_max_value_filter(self, chembl_db):
        """Test max_value_nm filter."""
        # B-raf with max_value + potency filter for speed
        results = chembl_db.search_by_target("B-raf", max_value_nm=100.0, min_pchembl=7.0, limit=10)
        assert len(results) > 0
        assert all(r.standard_value is None or r.standard_value <= 100.0 for r in results)
