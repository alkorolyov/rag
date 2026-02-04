"""LangChain tool definitions wrapping chemistry modules.

Each tool wraps functionality from parsers, similarity, chembl, or ner modules.
Tools are created via factory function that captures the context in closures.
"""

from dataclasses import asdict
from typing import Literal

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field

from rag.agents.context import ChemistryAgentContext
from rag.chemistry.parsers import (
    get_pubchem_info,
    name_to_smiles,
    parse_smiles,
    validate_smiles,
)


# ============================================================================
# Pydantic schemas for tool inputs
# ============================================================================


class AnalyzeMoleculeInput(BaseModel):
    """Input for analyze_molecule tool."""

    smiles: str = Field(description="SMILES string of the molecule to analyze")


class ResolveCompoundNameInput(BaseModel):
    """Input for resolve_compound_name tool."""

    name: str = Field(description="Common name of the compound (e.g., 'aspirin', 'imatinib')")


class FindSimilarCompoundsInput(BaseModel):
    """Input for find_similar_compounds tool."""

    smiles: str = Field(description="SMILES string of the query molecule")
    top_k: int = Field(default=10, description="Number of similar compounds to return")
    threshold: float = Field(
        default=0.5, description="Minimum Tanimoto similarity threshold (0.0-1.0)"
    )


class GetChemblCompoundInput(BaseModel):
    """Input for get_chembl_compound tool."""

    chembl_id: str = Field(description="ChEMBL compound ID (e.g., 'CHEMBL941')")


class SearchChemblByNameInput(BaseModel):
    """Input for search_chembl_by_name tool."""

    name: str = Field(description="Compound name to search (partial match supported)")
    limit: int = Field(default=10, description="Maximum number of results")


class GetBioactivitiesInput(BaseModel):
    """Input for get_bioactivities tool."""

    chembl_id: str = Field(description="ChEMBL compound ID")
    activity_type: str | None = Field(
        default=None, description="Filter by activity type (e.g., 'IC50', 'Ki', 'EC50')"
    )
    target_type: str | None = Field(
        default=None, description="Filter by target type (e.g., 'SINGLE PROTEIN')"
    )
    limit: int = Field(default=20, description="Maximum number of results")


class ExtractEntitiesInput(BaseModel):
    """Input for extract_entities tool."""

    text: str = Field(description="Text to extract biomedical entities from")


class SearchByTargetInput(BaseModel):
    """Input for search_compounds_by_target tool."""

    target_name: str = Field(
        description="Target name to search (e.g., 'BRAF', 'hERG', 'topoisomerase I')"
    )
    activity_type: str | None = Field(
        default=None,
        description="Filter by activity type (e.g., 'IC50', 'Ki', 'EC50', 'DC50' for PROTACs)",
    )
    min_pchembl: float | None = Field(
        default=None,
        description="Minimum pChEMBL value (e.g., 6.0 for sub-micromolar, 7.0 for sub-100nM)",
    )
    min_phase: int | None = Field(
        default=None,
        description="Minimum clinical phase (0=preclinical, 1-3=trials, 4=approved)",
    )
    mutation: str | None = Field(
        default=None, description="Filter by variant mutation (e.g., 'V600E')"
    )
    cell_line: str | None = Field(
        default=None, description="Filter by cell line name (e.g., 'melanoma', 'HEK293')"
    )
    limit: int = Field(default=20, description="Maximum number of results to return")


# ============================================================================
# Tool factory function
# ============================================================================


def create_tools(ctx: ChemistryAgentContext) -> list:
    """Create LangChain tools bound to the given context.

    Args:
        ctx: ChemistryAgentContext with database connections and models

    Returns:
        List of LangChain tools ready for use with an agent
    """

    @tool(args_schema=AnalyzeMoleculeInput)
    def analyze_molecule(smiles: str) -> dict:
        """Analyze a molecule from its SMILES string.

        Returns molecular properties including:
        - Molecular weight, formula
        - LogP (lipophilicity)
        - TPSA (polar surface area)
        - H-bond donors/acceptors
        - Ring count, rotatable bonds
        """
        if not validate_smiles(smiles):
            raise ToolException(f"Invalid SMILES string: {smiles}")

        try:
            info = parse_smiles(smiles)
            return asdict(info)
        except ValueError as e:
            raise ToolException(str(e))

    @tool(args_schema=ResolveCompoundNameInput)
    def resolve_compound_name(name: str) -> dict:
        """Resolve a compound name to its SMILES structure via PubChem.

        Use this when you have a drug or compound name and need its structure.
        Returns SMILES and additional PubChem data if available.
        """
        smiles = name_to_smiles(name)
        if smiles is None:
            raise ToolException(f"Could not find compound: {name}")

        result = {"name": name, "smiles": smiles}

        # Try to get additional PubChem info
        pubchem_info = get_pubchem_info(name)
        if pubchem_info:
            result.update(pubchem_info)

        return result

    @tool(args_schema=FindSimilarCompoundsInput)
    def find_similar_compounds(
        smiles: str, top_k: int = 10, threshold: float = 0.5
    ) -> list[dict]:
        """Find structurally similar compounds using molecular fingerprints.

        Searches approved drugs in ChEMBL using Tanimoto similarity with
        Morgan fingerprints. Returns compounds above the similarity threshold.
        """
        if not validate_smiles(smiles):
            raise ToolException(f"Invalid SMILES string: {smiles}")

        try:
            results = ctx.similarity.find_similar(
                query_smiles=smiles,
                database=ctx.compound_database,
                top_k=top_k,
                threshold=threshold,
            )
            return [asdict(r) for r in results]
        except ValueError as e:
            raise ToolException(str(e))

    @tool(args_schema=GetChemblCompoundInput)
    def get_chembl_compound(chembl_id: str) -> dict:
        """Get detailed compound information from ChEMBL by ID.

        Returns compound metadata including:
        - Name, SMILES structure
        - Molecule type
        - Max clinical phase (4 = approved)
        - First approval year
        """
        compound = ctx.chembl.get_compound(chembl_id)
        if compound is None:
            raise ToolException(f"Compound not found: {chembl_id}")
        return asdict(compound)

    @tool(args_schema=SearchChemblByNameInput)
    def search_chembl_by_name(name: str, limit: int = 10) -> list[dict]:
        """Search ChEMBL compounds by name (case-insensitive, partial match).

        Use this to find compounds when you know part of the name but not
        the exact ChEMBL ID. Returns matching compound metadata.
        """
        results = ctx.chembl.search_by_name(name, limit=limit)
        if not results:
            return []
        return [asdict(r) for r in results]

    @tool(args_schema=GetBioactivitiesInput)
    def get_bioactivities(
        chembl_id: str,
        activity_type: str | None = None,
        target_type: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Get bioactivity data for a compound from ChEMBL.

        Returns assay results including:
        - Activity type (IC50, Ki, EC50, etc.)
        - Value and units
        - Target name and type
        - Assay ID for reference

        Filter by activity_type (e.g., 'IC50') or target_type (e.g., 'SINGLE PROTEIN').
        """
        results = ctx.chembl.get_activities(
            chembl_id, activity_type=activity_type, target_type=target_type, limit=limit
        )
        if not results:
            return []
        return [asdict(r) for r in results]

    @tool(args_schema=ExtractEntitiesInput)
    def extract_entities(text: str) -> dict:
        """Extract biomedical named entities from text.

        Identifies and categorizes entities including:
        - GENE: Gene and protein names
        - DISEASE: Disease and condition names
        - CHEMICAL: Drug and chemical names
        - CELL: Cell types
        - SPECIES: Organism names

        Returns entities grouped by type.
        """
        result = ctx.ner.extract(text)
        return result.to_dict()

    @tool(args_schema=SearchByTargetInput)
    def search_compounds_by_target(
        target_name: str,
        activity_type: str | None = None,
        min_pchembl: float | None = None,
        min_phase: int | None = None,
        mutation: str | None = None,
        cell_line: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Find compounds with activity against a specific target.

        This is the reverse of get_bioactivities - instead of "what targets does
        this compound hit?", this answers "what compounds hit this target?".

        Use cases:
        - Find approved BRAF inhibitors: target_name="BRAF", min_phase=4
        - Find potent hERG blockers (cardiac safety): target_name="hERG", min_pchembl=6.0
        - Find BRAF V600E-selective inhibitors: target_name="BRAF", mutation="V600E"
        - Find PROTAC degraders: target_name="BRD4", activity_type="DC50"
        - Find compounds tested in melanoma cells: target_name="BRAF", cell_line="melanoma"

        Returns compounds sorted by potency (highest pChEMBL first).
        """
        results = ctx.chembl.search_by_target(
            target_name=target_name,
            activity_type=activity_type,
            min_pchembl=min_pchembl,
            min_phase=min_phase,
            mutation=mutation,
            cell_line=cell_line,
            deduplicate=True,  # Default to deduplicated for cleaner agent output
            limit=limit,
        )
        if not results:
            return []
        return [asdict(r) for r in results]

    return [
        analyze_molecule,
        resolve_compound_name,
        find_similar_compounds,
        get_chembl_compound,
        search_chembl_by_name,
        get_bioactivities,
        extract_entities,
        search_compounds_by_target,
    ]
