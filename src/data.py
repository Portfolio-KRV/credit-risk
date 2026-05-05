"""Data loading and preprocessing for the Credit Risk model."""

import contextlib
from pathlib import Path
from typing import Any

import pandas as pd

from .config import COLUMNS, DATA_FILE


def load_data(filepath: Path | None = None) -> pd.DataFrame:
    """Load the credit risk dataset.

    Args:
        filepath: Path to CSV file. If None, uses default DATA_FILE.

    Returns:
        pandas DataFrame with credit risk data
    """
    filepath = filepath or DATA_FILE

    # Load data, excluding the index column
    data = pd.read_csv(filepath, usecols=range(1, 11))

    # Convert all columns to category type for pgmpy compatibility
    for col in data.columns:
        data[col] = data[col].astype(str).astype('category')

    return data


def get_feature_columns() -> list[str]:
    """Get list of feature column names (excluding target).

    Returns:
        List of feature column names
    """
    all_cols = list(COLUMNS.values())
    target = COLUMNS["risk"]
    return [col for col in all_cols if col != target]


def get_target_column() -> str:
    """Get the target column name.

    Returns:
        Target column name
    """
    return COLUMNS["risk"]


def validate_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate and convert evidence values to correct types.

    Args:
        evidence: Dictionary of variable -> value pairs

    Returns:
        Validated evidence dictionary
    """
    from .config import VARIABLES

    validated = {}
    for var, value in evidence.items():
        if var not in VARIABLES:
            raise ValueError(f"Unknown variable: {var}")

        valid_values = VARIABLES[var]["values"]

        # Convert to appropriate type
        if isinstance(valid_values[0], int) and not isinstance(value, int):
            with contextlib.suppress(ValueError, TypeError):
                value = int(value)

        if value not in valid_values:
            raise ValueError(
                f"Invalid value '{value}' for {var}. "
                f"Valid values: {valid_values}"
            )

        validated[var] = value

    return validated


def get_data_summary(data: pd.DataFrame) -> dict[str, Any]:
    """Get summary statistics of the dataset.

    Args:
        data: Credit risk DataFrame

    Returns:
        Dictionary with summary statistics
    """
    target = get_target_column()

    return {
        "n_samples": len(data),
        "n_features": len(data.columns) - 1,
        "risk_distribution": data[target].value_counts().to_dict(),
        "columns": list(data.columns),
    }
