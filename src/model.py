"""Model training, structure learning, and persistence."""

import logging
import pickle
from pathlib import Path

import pandas as pd
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from .config import MODEL_FILE
from .security import verify_file_integrity

logger = logging.getLogger(__name__)

# Improved structure from the original notebook analysis
# This structure was derived from Hill-Climbing + expert improvements
IMPROVED_EDGES: list[tuple[str, str]] = [
    ("Evaluacion crediticia historica", "Riesgo"),
    ("Evaluacion crediticia historica", "Duracion"),
    ("Edad", "Residencia"),
    ("Edad", "Genero"),
    ("Genero", "Residencia"),
    ("Evaluacion Sueldo", "Monto del credito"),
    ("Residencia", "Monto del credito"),
    ("Nivel de ahorro", "Riesgo"),
    ("Monto del credito", "Duracion"),
    ("Monto del credito", "Riesgo"),
    ("Duracion", "Riesgo"),
    ("Proposito", "Monto del credito"),
]


def learn_structure(data: pd.DataFrame) -> BayesianNetwork:
    """Learn network structure using Hill-Climbing with BIC score.

    Args:
        data: Training data

    Returns:
        BayesianNetwork with learned structure (not yet fitted)
    """
    hc = HillClimbSearch(data=data)
    model = hc.estimate(scoring_method='bic-d', show_progress=True)

    return model


def get_improved_structure() -> BayesianNetwork:
    """Get the expert-improved network structure.

    This structure was derived from Hill-Climbing analysis followed by
    expert improvements to correct causal directions based on domain knowledge.

    Returns:
        BayesianNetwork with the improved structure
    """
    return BayesianNetwork(IMPROVED_EDGES)


def fit_parameters(model: BayesianNetwork, data: pd.DataFrame) -> BayesianNetwork:
    """Fit CPD parameters using Maximum Likelihood Estimation.

    Args:
        model: BayesianNetwork with structure defined
        data: Training data

    Returns:
        Fitted BayesianNetwork
    """
    # Create a new BayesianNetwork from the edges
    bayesian_model = BayesianNetwork(model.edges())

    # Fit CPDs
    bayesian_model.fit(data=data, estimator=MaximumLikelihoodEstimator)

    # Validate
    if not bayesian_model.check_model():
        raise ValueError("Model validation failed after fitting")

    return bayesian_model


def train_model(data: pd.DataFrame, use_improved_structure: bool = True) -> BayesianNetwork:
    """Complete training pipeline: structure learning + parameter fitting.

    Args:
        data: Training data
        use_improved_structure: If True, use the expert-improved structure.
                                If False, learn structure with Hill-Climbing.

    Returns:
        Trained BayesianNetwork
    """
    if use_improved_structure:
        logger.info("Using expert-improved structure from notebook analysis...")
        structure = get_improved_structure()
        logger.info("  Structure has %d edges", len(structure.edges()))
    else:
        logger.info("Learning structure with Hill-Climbing + BIC...")
        structure = learn_structure(data)
        logger.info("  Found %d edges", len(structure.edges()))

    logger.info("Fitting parameters with MLE...")
    model = fit_parameters(structure, data)
    logger.info("  Model fitted successfully")

    return model


def save_model(model: BayesianNetwork, path: Path = MODEL_FILE) -> None:
    """Save model to disk.

    Args:
        model: BayesianNetwork to save
        path: File path for saving
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: Path = MODEL_FILE) -> BayesianNetwork:
    """Load model from disk.

    Args:
        path: File path to load from

    Returns:
        Loaded BayesianNetwork
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Verify model integrity before loading
    checksums_file = path.parent / "checksums.json"
    if not verify_file_integrity(path, checksums_file=checksums_file):
        raise ValueError(f"Model file {path} failed integrity check")

    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def get_or_create_model(data: pd.DataFrame | None = None) -> BayesianNetwork:
    """Get existing model or create new one.

    Args:
        data: Training data (required if model doesn't exist)

    Returns:
        BayesianNetwork model
    """
    try:
        return load_model()
    except FileNotFoundError:
        if data is None:
            from .data import load_data
            data = load_data()
        model = train_model(data)
        save_model(model)
        return model
