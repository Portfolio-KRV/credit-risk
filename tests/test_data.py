"""Tests for data loading and preprocessing."""

import pandas as pd

from src.data import get_data_summary, get_feature_columns, get_target_column, load_data


class TestLoadData:
    """Tests for data loading."""

    def test_load_data_returns_dataframe(self):
        """Test that load_data returns a DataFrame."""
        data = load_data()
        assert isinstance(data, pd.DataFrame)

    def test_load_data_has_correct_columns(self):
        """Test that loaded data has expected columns."""
        data = load_data()
        assert "Riesgo" in data.columns
        assert "Edad" in data.columns
        assert len(data.columns) == 10

    def test_load_data_not_empty(self):
        """Test that loaded data is not empty."""
        data = load_data()
        assert len(data) > 0


class TestFeatureColumns:
    """Tests for column utilities."""

    def test_get_feature_columns_excludes_target(self):
        """Test that feature columns don't include target."""
        features = get_feature_columns()
        assert "Riesgo" not in features

    def test_get_target_column(self):
        """Test getting target column name."""
        target = get_target_column()
        assert target == "Riesgo"


class TestDataSummary:
    """Tests for data summary."""

    def test_get_data_summary(self):
        """Test data summary function."""
        data = load_data()
        summary = get_data_summary(data)

        assert "n_samples" in summary
        assert "n_features" in summary
        assert "risk_distribution" in summary
        assert summary["n_samples"] > 0
