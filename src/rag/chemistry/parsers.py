"""SMILES parsing, validation, and compound name resolution."""

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


@dataclass
class MoleculeInfo:
    """Basic molecule information."""

    smiles: str
    canonical_smiles: str
    formula: str
    molecular_weight: float
    num_atoms: int
    num_heavy_atoms: int
    num_rings: int
    num_rotatable_bonds: int
    num_hbd: int  # H-bond donors
    num_hba: int  # H-bond acceptors
    tpsa: float  # Topological polar surface area
    logp: float  # Calculated LogP (Wildman-Crippen)


def validate_smiles(smiles: str) -> bool:
    """Check if SMILES string is valid.

    Args:
        smiles: SMILES string to validate

    Returns:
        True if valid, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def parse_smiles(smiles: str) -> MoleculeInfo:
    """Parse SMILES and compute basic molecular properties.

    Args:
        smiles: SMILES string

    Returns:
        MoleculeInfo with computed properties

    Raises:
        ValueError: If SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return MoleculeInfo(
        smiles=smiles,
        canonical_smiles=Chem.MolToSmiles(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        molecular_weight=round(Descriptors.MolWt(mol), 2),
        num_atoms=mol.GetNumAtoms(),
        num_heavy_atoms=mol.GetNumHeavyAtoms(),
        num_rings=rdMolDescriptors.CalcNumRings(mol),
        num_rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(mol),
        num_hbd=rdMolDescriptors.CalcNumHBD(mol),
        num_hba=rdMolDescriptors.CalcNumHBA(mol),
        tpsa=round(rdMolDescriptors.CalcTPSA(mol), 2),
        logp=round(Descriptors.MolLogP(mol), 2),
    )


def canonicalize_smiles(smiles: str) -> str | None:
    """Convert SMILES to canonical form.

    Args:
        smiles: Input SMILES string

    Returns:
        Canonical SMILES or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def smiles_to_inchi(smiles: str) -> str | None:
    """Convert SMILES to InChI.

    Args:
        smiles: Input SMILES string

    Returns:
        InChI string or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchi(mol)


def smiles_to_inchikey(smiles: str) -> str | None:
    """Convert SMILES to InChIKey.

    Args:
        smiles: Input SMILES string

    Returns:
        InChIKey string or None if conversion fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    inchi = Chem.MolToInchi(mol)
    if inchi is None:
        return None
    return Chem.InchiToInchiKey(inchi)


@lru_cache(maxsize=1000)
def name_to_smiles(name: str) -> str | None:
    """Convert compound name to SMILES via PubChem.

    Uses caching to avoid repeated API calls.

    Args:
        name: Compound name (e.g., "aspirin", "imatinib")

    Returns:
        Canonical SMILES or None if not found
    """
    try:
        import pubchempy as pcp

        results = pcp.get_compounds(name, "name")
        if results:
            return results[0].canonical_smiles
        return None
    except Exception:
        return None


def get_pubchem_info(name: str) -> dict | None:
    """Get compound info from PubChem by name.

    Args:
        name: Compound name

    Returns:
        Dict with PubChem info or None if not found
    """
    try:
        import pubchempy as pcp

        results = pcp.get_compounds(name, "name")
        if not results:
            return None

        compound = results[0]
        return {
            "cid": compound.cid,
            "name": name,
            "iupac_name": compound.iupac_name,
            "smiles": compound.canonical_smiles,
            "molecular_formula": compound.molecular_formula,
            "molecular_weight": compound.molecular_weight,
            "xlogp": compound.xlogp,
            "tpsa": compound.tpsa,
            "complexity": compound.complexity,
        }
    except Exception:
        return None
