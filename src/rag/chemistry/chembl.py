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
    value: float | None
    units: str | None
    relation: str | None  # =, <, >, etc.
    target_name: str | None
    target_chembl_id: str | None
    target_type: str | None  # SINGLE PROTEIN, PROTEIN COMPLEX, etc.


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
            self._conn = sqlite3.connect(str(self.db_path))
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
            a.assay_chembl_id,
            a.standard_type,
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
        """
        params = [chembl_id.upper()]

        if activity_type:
            query += " AND a.standard_type = ?"
            params.append(activity_type.upper())

        if target_type:
            query += " AND td.target_type = ?"
            params.append(target_type.upper())

        query += " ORDER BY a.standard_value ASC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [
            ActivityData(
                assay_chembl_id=row["assay_chembl_id"],
                activity_type=row["standard_type"],
                value=row["standard_value"],
                units=row["standard_units"],
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

    def get_compounds_for_target(
        self,
        target_chembl_id: str,
        activity_type: str = "IC50",
        max_value: float | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get compounds with activity against a specific target.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., "CHEMBL203" for EGFR)
            activity_type: Activity type to filter (default: IC50)
            max_value: Maximum activity value (e.g., 1000 for nM)
            limit: Maximum results

        Returns:
            List of dicts with compound info and activity
        """
        query = """
        SELECT DISTINCT
            md.chembl_id,
            md.pref_name,
            cs.canonical_smiles,
            a.standard_type,
            a.standard_value,
            a.standard_units
        FROM activities a
        JOIN molecule_dictionary md ON a.molregno = md.molregno
        JOIN compound_structures cs ON md.molregno = cs.molregno
        JOIN assays ass ON a.assay_id = ass.assay_id
        JOIN target_dictionary td ON ass.tid = td.tid
        WHERE td.chembl_id = ?
        AND a.standard_type = ?
        AND a.standard_value IS NOT NULL
        """
        params = [target_chembl_id.upper(), activity_type.upper()]

        if max_value is not None:
            query += " AND a.standard_value <= ?"
            params.append(max_value)

        query += " ORDER BY a.standard_value ASC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        return [
            {
                "chembl_id": row["chembl_id"],
                "name": row["pref_name"],
                "smiles": row["canonical_smiles"],
                "activity_type": row["standard_type"],
                "activity_value": row["standard_value"],
                "activity_units": row["standard_units"],
            }
            for row in cursor.fetchall()
        ]

    def get_all_smiles(self, limit: int = 100000) -> list[dict]:
        """Get SMILES for all compounds (for similarity search).

        Args:
            limit: Maximum compounds to return

        Returns:
            List of dicts with chembl_id, name, smiles
        """
        query = """
        SELECT
            md.chembl_id,
            md.pref_name,
            cs.canonical_smiles
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
        LIMIT ?
        """
        cursor = self.conn.execute(query, (limit,))
        return [
            {
                "chembl_id": row["chembl_id"],
                "name": row["pref_name"],
                "smiles": row["canonical_smiles"],
            }
            for row in cursor.fetchall()
        ]

    def get_smiles_by_ids(self, chembl_ids: list[str]) -> dict[str, str]:
        """Get SMILES for a list of ChEMBL IDs.

        Args:
            chembl_ids: List of ChEMBL compound IDs

        Returns:
            Dict mapping chembl_id to SMILES
        """
        if not chembl_ids:
            return {}

        placeholders = ",".join(["?"] * len(chembl_ids))
        query = f"""
        SELECT md.chembl_id, cs.canonical_smiles
        FROM molecule_dictionary md
        JOIN compound_structures cs ON md.molregno = cs.molregno
        WHERE md.chembl_id IN ({placeholders})
        """
        cursor = self.conn.execute(query, [cid.upper() for cid in chembl_ids])
        return {row["chembl_id"]: row["canonical_smiles"] for row in cursor.fetchall()}
