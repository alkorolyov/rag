"""ChEMBL database access via local SQLite."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompoundInfo:
    """Basic compound information from ChEMBL."""

    chembl_id: str
    name: str | None
    smiles: str
    molecule_type: str | None
    max_phase: int | None  # 4 = approved drug
    first_approval: int | None


@dataclass
class ActivityData:
    """Bioactivity measurement from ChEMBL."""

    assay_chembl_id: str
    activity_type: str  # IC50, Ki, EC50, etc.
    pchembl_value: float  # -log10(molar), comparable across activity types
    standard_value: float | None  # Original value for reference
    standard_units: str | None  # Original units (nM, uM, etc.)
    relation: str | None  # =, <, >, etc.
    target_name: str | None
    target_chembl_id: str | None
    target_type: str | None  # SINGLE PROTEIN, PROTEIN COMPLEX, etc.


@dataclass
class TargetActivityResult:
    """Bioactivity data for a compound against a target.

    Used for target-based searches (target → compounds).
    """

    # Compound info
    compound_chembl_id: str
    compound_name: str | None
    smiles: str
    max_phase: int | None  # 0-4, None if unknown

    # Activity measurement
    activity_type: str  # IC50, Ki, EC50, DC50, etc.
    standard_value: float | None  # Original value (nM)
    standard_units: str | None  # nM, uM, etc.
    pchembl_value: float | None  # -log10(molar), comparable
    relation: str | None  # =, <, >, ~, etc.

    # Target context
    target_name: str
    target_chembl_id: str
    target_type: str | None  # SINGLE PROTEIN, PROTEIN COMPLEX, etc.

    # Assay context (optional)
    mutation: str | None  # e.g., "V600E"
    cell_line: str | None  # e.g., "A-375", "HEK293"


@dataclass
class DrugWarning:
    """Regulatory warning for a drug (FDA black box, withdrawals, etc.)."""

    chembl_id: str
    compound_name: str | None
    warning_type: str | None  # "Black Box Warning", "Withdrawn", etc.
    warning_class: str | None  # "hepatotoxicity", "cardiotoxicity", etc.
    warning_description: str | None
    warning_country: str | None
    warning_year: int | None
    efo_term: str | None  # Experimental Factor Ontology term


@dataclass
class LiteratureRef:
    """Literature reference from ChEMBL docs table."""

    pubmed_id: int | None
    doi: str | None
    title: str | None
    journal: str | None
    year: int | None
    abstract: str | None


class ChEMBLDatabase:
    """Query local ChEMBL SQLite database.

    Download the database from:
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/

    Example:
        >>> db = ChEMBLDatabase("data/chembl_34/chembl_34_sqlite/chembl_34.db")
        >>> compound = db.get_compound("CHEMBL941")  # Imatinib
        >>> activities = db.get_activities("CHEMBL941", activity_type="IC50")
    """

    def __init__(self, db_path: str | Path):
        """Initialize database connection.

        Args:
            db_path: Path to ChEMBL SQLite database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"ChEMBL database not found: {self.db_path}")
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy database connection."""
        if self._conn is None:
            # check_same_thread=False allows use across threads (needed for LangGraph)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_compound(self, chembl_id: str) -> CompoundInfo | None:
        """Get compound info by ChEMBL ID.

        Args:
            chembl_id: ChEMBL compound ID (e.g., "CHEMBL941")

        Returns:
            CompoundInfo or None if not found
        """
        query = """
        SELECT
            md.chembl_id,
            md.pref_name,
            cs.canonical_smiles,
            md.molecule_type,
            md.max_phase,
            md.first_approval
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE md.chembl_id = ?
        """
        cursor = self.conn.execute(query, (chembl_id.upper(),))
        row = cursor.fetchone()
        if row is None:
            return None

        return CompoundInfo(
            chembl_id=row["chembl_id"],
            name=row["pref_name"],
            smiles=row["canonical_smiles"],
            molecule_type=row["molecule_type"],
            max_phase=row["max_phase"],
            first_approval=row["first_approval"],
        )

    def search_by_name(self, name: str, limit: int = 10) -> list[CompoundInfo]:
        """Search compounds by name (case-insensitive, partial match).

        Args:
            name: Compound name to search
            limit: Maximum results to return

        Returns:
            List of matching compounds
        """
        query = """
        SELECT
            md.chembl_id,
            md.pref_name,
            cs.canonical_smiles,
            md.molecule_type,
            md.max_phase,
            md.first_approval
        FROM molecule_dictionary md
        LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE LOWER(md.pref_name) LIKE LOWER(?)
        AND cs.canonical_smiles IS NOT NULL
        LIMIT ?
        """
        cursor = self.conn.execute(query, (f"%{name}%", limit))
        return [
            CompoundInfo(
                chembl_id=row["chembl_id"],
                name=row["pref_name"],
                smiles=row["canonical_smiles"],
                molecule_type=row["molecule_type"],
                max_phase=row["max_phase"],
                first_approval=row["first_approval"],
            )
            for row in cursor.fetchall()
        ]

    def get_activities(
        self,
        chembl_id: str,
        activity_type: str | None = None,
        target_type: str | None = None,
        limit: int = 100,
    ) -> list[ActivityData]:
        """Get bioactivity data for a compound.

        Args:
            chembl_id: ChEMBL compound ID
            activity_type: Filter by activity type (IC50, Ki, EC50, etc.)
            target_type: Filter by target type (SINGLE PROTEIN, etc.)
            limit: Maximum results to return

        Returns:
            List of activity measurements
        """
        query = """
        SELECT
            ass.chembl_id as assay_chembl_id,
            a.standard_type,
            a.pchembl_value,
            a.standard_value,
            a.standard_units,
            a.standard_relation,
            td.pref_name as target_name,
            td.chembl_id as target_chembl_id,
            td.target_type
        FROM activities a
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        JOIN assays ass ON a.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        WHERE md.chembl_id = ?
        AND a.pchembl_value IS NOT NULL
        """
        params = [chembl_id.upper()]

        if activity_type:
            query += " AND a.standard_type = ?"
            params.append(activity_type.upper())

        if target_type:
            query += " AND td.target_type = ?"
            params.append(target_type.upper())

        query += " ORDER BY a.pchembl_value DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [
            ActivityData(
                assay_chembl_id=row["assay_chembl_id"],
                activity_type=row["standard_type"],
                pchembl_value=row["pchembl_value"],
                standard_value=row["standard_value"],
                standard_units=row["standard_units"],
                relation=row["standard_relation"],
                target_name=row["target_name"],
                target_chembl_id=row["target_chembl_id"],
                target_type=row["target_type"],
            )
            for row in cursor.fetchall()
        ]

    def get_approved_drugs(self, limit: int = 1000) -> list[CompoundInfo]:
        """Get list of approved drugs (max_phase = 4).

        Args:
            limit: Maximum results to return

        Returns:
            List of approved drug compounds
        """
        query = """
        SELECT
            md.chembl_id,
            md.pref_name,
            cs.canonical_smiles,
            md.molecule_type,
            md.max_phase,
            md.first_approval
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE md.max_phase = 4
        AND cs.canonical_smiles IS NOT NULL
        ORDER BY md.first_approval DESC
        LIMIT ?
        """
        cursor = self.conn.execute(query, (limit,))
        return [
            CompoundInfo(
                chembl_id=row["chembl_id"],
                name=row["pref_name"],
                smiles=row["canonical_smiles"],
                molecule_type=row["molecule_type"],
                max_phase=row["max_phase"],
                first_approval=row["first_approval"],
            )
            for row in cursor.fetchall()
        ]

    def search_by_target(
        self,
        target_name: str,
        activity_type: str | None = None,
        max_value_nm: float | None = None,
        min_pchembl: float | None = None,
        min_phase: int | None = None,
        cell_line: str | None = None,
        mutation: str | None = None,
        deduplicate: bool = False,
        limit: int = 50,
    ) -> list[TargetActivityResult]:
        """Search for compounds with activity against a target.

        This is a "reverse query" - find compounds given a target, instead of
        finding targets given a compound (which get_activities does).

        Args:
            target_name: Target name to search (partial match, case-insensitive).
                        Examples: "BRAF", "hERG", "topoisomerase I"
            activity_type: Filter by activity type (IC50, Ki, EC50, DC50, etc.)
            max_value_nm: Filter by standard_value <= X (in nM)
            min_pchembl: Filter by pchembl_value >= X (e.g., 6.0 for sub-uM)
            min_phase: Filter by max_phase >= X (0=preclinical, 4=approved)
            cell_line: Filter by cell line name (partial match)
            mutation: Filter by variant mutation (partial match, e.g., "V600E")
            deduplicate: If True, return only best activity per compound
            limit: Maximum results to return (default 50)

        Returns:
            List of TargetActivityResult sorted by pchembl_value (descending)

        Example:
            >>> db.search_by_target("BRAF", activity_type="IC50", mutation="V600E")
            [TargetActivityResult(compound_name='DABRAFENIB', pchembl_value=9.3, ...)]
        """
        # Simple query - deduplication is done in Python for better performance
        query = """
        SELECT
            md.chembl_id as compound_chembl_id,
            md.pref_name as compound_name,
            cs.canonical_smiles as smiles,
            md.max_phase,
            a.standard_type as activity_type,
            a.standard_value,
            a.standard_units,
            a.pchembl_value,
            a.standard_relation as relation,
            td.pref_name as target_name,
            td.chembl_id as target_chembl_id,
            td.target_type,
            vs.mutation,
            cd.cell_name as cell_line
        FROM target_dictionary td
        JOIN assays ass ON td.tid = ass.tid
        JOIN activities a ON ass.assay_id = a.assay_id
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        LEFT JOIN variant_sequences vs ON ass.variant_id = vs.variant_id
        LEFT JOIN cell_dictionary cd ON ass.cell_id = cd.cell_id
        WHERE LOWER(td.pref_name) LIKE LOWER(?)
        AND a.pchembl_value IS NOT NULL
        """

        params: list = [f"%{target_name}%"]

        # Add optional filters
        if activity_type:
            query += " AND a.standard_type = ?"
            params.append(activity_type.upper())

        if max_value_nm is not None:
            query += " AND a.standard_value <= ?"
            params.append(max_value_nm)

        if min_pchembl is not None:
            query += " AND a.pchembl_value >= ?"
            params.append(min_pchembl)

        if min_phase is not None:
            query += " AND md.max_phase >= ?"
            params.append(min_phase)

        if cell_line:
            query += " AND LOWER(cd.cell_name) LIKE LOWER(?)"
            params.append(f"%{cell_line}%")

        if mutation:
            query += " AND LOWER(vs.mutation) LIKE LOWER(?)"
            params.append(f"%{mutation}%")

        # For deduplication, fetch more results then filter in Python (faster than SQL window)
        fetch_limit = limit * 10 if deduplicate else limit
        query += " ORDER BY a.pchembl_value DESC LIMIT ?"
        params.append(fetch_limit)

        # Execute query
        cursor = self.conn.execute(query, params)

        results = [
            TargetActivityResult(
                compound_chembl_id=row["compound_chembl_id"],
                compound_name=row["compound_name"],
                smiles=row["smiles"],
                max_phase=row["max_phase"],
                activity_type=row["activity_type"],
                standard_value=row["standard_value"],
                standard_units=row["standard_units"],
                pchembl_value=row["pchembl_value"],
                relation=row["relation"],
                target_name=row["target_name"],
                target_chembl_id=row["target_chembl_id"],
                target_type=row["target_type"],
                mutation=row["mutation"],
                cell_line=row["cell_line"],
            )
            for row in cursor.fetchall()
        ]

        # Python-based deduplication: keep best (first) activity per compound
        if deduplicate:
            seen: set[str] = set()
            unique_results = []
            for r in results:
                if r.compound_chembl_id not in seen:
                    seen.add(r.compound_chembl_id)
                    unique_results.append(r)
                    if len(unique_results) >= limit:
                        break
            return unique_results

        return results

    def get_drug_warnings(
        self,
        chembl_id: str | None = None,
        compound_name: str | None = None,
        warning_class: str | None = None,
        limit: int = 50,
    ) -> list[DrugWarning]:
        """Get regulatory warnings (FDA black box, withdrawals) for drugs.

        Args:
            chembl_id: Filter by specific compound ChEMBL ID
            compound_name: Filter by compound name (partial match)
            warning_class: Filter by warning class (e.g., "hepatotoxicity")
            limit: Maximum results to return

        Returns:
            List of DrugWarning objects

        Example:
            >>> db.get_drug_warnings(compound_name="rosiglitazone")
            [DrugWarning(warning_class='cardiotoxicity', ...)]
        """
        query = """
        SELECT
            md.chembl_id,
            md.pref_name as compound_name,
            dw.warning_type,
            dw.warning_class,
            dw.warning_description,
            dw.warning_country,
            dw.warning_year,
            dw.efo_term
        FROM drug_warning dw
        JOIN molecule_dictionary md ON dw.molregno = md.molregno
        WHERE 1=1
        """
        params: list = []

        if chembl_id:
            query += " AND md.chembl_id = ?"
            params.append(chembl_id.upper())

        if compound_name:
            query += " AND LOWER(md.pref_name) LIKE LOWER(?)"
            params.append(f"%{compound_name}%")

        if warning_class:
            query += " AND LOWER(dw.warning_class) LIKE LOWER(?)"
            params.append(f"%{warning_class}%")

        query += " ORDER BY dw.warning_year DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)

        return [
            DrugWarning(
                chembl_id=row["chembl_id"],
                compound_name=row["compound_name"],
                warning_type=row["warning_type"],
                warning_class=row["warning_class"],
                warning_description=row["warning_description"],
                warning_country=row["warning_country"],
                warning_year=row["warning_year"],
                efo_term=row["efo_term"],
            )
            for row in cursor.fetchall()
        ]

    def get_compound_literature(
        self,
        chembl_id: str,
        limit: int = 10,
    ) -> list[LiteratureRef]:
        """Get literature references (PubMed) for a compound.

        Returns publications linked to assays involving the compound.

        Args:
            chembl_id: Compound ChEMBL ID
            limit: Maximum results to return

        Returns:
            List of LiteratureRef objects with PubMed IDs, titles, abstracts

        Example:
            >>> db.get_compound_literature("CHEMBL941")  # Imatinib
            [LiteratureRef(pubmed_id=12345, title='...', ...)]
        """
        query = """
        SELECT DISTINCT
            d.pubmed_id,
            d.doi,
            d.title,
            d.journal,
            d.year,
            d.abstract
        FROM docs d
        JOIN assays ass ON d.doc_id = ass.doc_id
        JOIN activities a ON ass.assay_id = a.assay_id
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        WHERE md.chembl_id = ?
        AND d.pubmed_id IS NOT NULL
        ORDER BY d.year DESC
        LIMIT ?
        """
        cursor = self.conn.execute(query, (chembl_id.upper(), limit))

        return [
            LiteratureRef(
                pubmed_id=row["pubmed_id"],
                doi=row["doi"],
                title=row["title"],
                journal=row["journal"],
                year=row["year"],
                abstract=row["abstract"],
            )
            for row in cursor.fetchall()
        ]

