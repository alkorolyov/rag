"""MLflow tracking utilities for reproducible experiments."""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import mlflow
import yaml

from rag.logger import setup_logger

logger = setup_logger(__name__)


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except subprocess.CalledProcessError:
        logger.warning("Not in a git repository")
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return branch
    except subprocess.CalledProcessError:
        return "unknown"


def get_dvc_hash(dvc_file: Path) -> Optional[str]:
    """Get data hash from .dvc file."""
    try:
        with open(dvc_file) as f:
            meta = yaml.safe_load(f)
            return meta["outs"][0]["md5"]
    except (FileNotFoundError, KeyError, TypeError):
        logger.warning(f"Could not read DVC hash from {dvc_file}")
        return None


class ExperimentTracker:
    """Track experiments with MLflow + Git + DVC integration."""

    def __init__(
        self,
        experiment_name: str = "rag-experiments",
        tracking_uri: str = "http://localhost:5000"
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (default: Docker server)
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"Tracking to: {tracking_uri}")
        logger.info(f"Experiment: {experiment_name}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_system_info: bool = True
    ):
        """
        Start an MLflow run with automatic git/dvc tracking.

        Args:
            run_name: Name for this run
            tags: Additional tags to log
            log_system_info: Whether to log git/system info

        Returns:
            MLflow run context manager
        """
        run = mlflow.start_run(run_name=run_name)

        if log_system_info:
            # Log git info
            mlflow.set_tag("git_commit", get_git_commit())
            mlflow.set_tag("git_branch", get_git_branch())
            mlflow.set_tag("timestamp", datetime.now().isoformat())

            # Log custom tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, str(value))

        return run

    def log_dvc_data(self, dvc_file: Path, dataset_name: Optional[str] = None):
        """
        Log DVC data hash for reproducibility.

        Args:
            dvc_file: Path to .dvc file (e.g., data/dataset.dvc)
            dataset_name: Optional human-readable dataset name
        """
        data_hash = get_dvc_hash(dvc_file)
        if data_hash:
            mlflow.set_tag(f"dvc_{dvc_file.stem}", data_hash)
            logger.info(f"Logged DVC hash for {dvc_file.name}: {data_hash[:8]}...")

        # Set dataset name as tag (searchable)
        if dataset_name:
            mlflow.set_tag("dataset", dataset_name)

    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics with automatic name sanitization.

        Replaces @ with _at_ for MLflow compatibility (e.g., P@10 → P_at_10).
        This is reversible and won't conflict with existing underscores.
        """
        for key, value in metrics.items():
            # Sanitize metric name (replace @ with _at_)
            sanitized_key = key.replace('@', '_at_')
            mlflow.log_metric(sanitized_key, value, step=step)

    @staticmethod
    def restore_at_notation(metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Convert MLflow metrics back to @ notation for reporting.

        Args:
            metrics: Dictionary with sanitized metric names (P_at_10)

        Returns:
            Dictionary with original notation (P@10)
        """
        return {k.replace('_at_', '@'): v for k, v in metrics.items()}

    def log_artifact(self, artifact_path: Path):
        """Log a single artifact file."""
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))
        else:
            logger.warning(f"Artifact not found: {artifact_path}")
