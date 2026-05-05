"""Tests for model training and inference."""

import pytest

from src.data import load_data
from src.evaluate import predict_risk, query, run_validation_queries
from src.model import fit_parameters, get_improved_structure, learn_structure, train_model


@pytest.fixture(scope="module")
def data():
    """Load data once for all tests."""
    return load_data()


@pytest.fixture(scope="module")
def trained_model(data):
    """Train model once for all tests."""
    return train_model(data)


class TestStructureLearning:
    """Tests for structure learning."""

    def test_learn_structure_returns_model(self, data):
        """Test that structure learning returns a model."""
        structure = learn_structure(data)
        assert structure is not None
        assert len(structure.edges()) > 0

    def test_get_improved_structure(self, data):
        """Test that the expert-improved structure is returned."""
        improved = get_improved_structure()
        assert improved is not None
        assert len(improved.edges()) > 0


class TestParameterLearning:
    """Tests for parameter learning."""

    def test_fit_parameters(self, data):
        """Test parameter fitting."""
        improved = get_improved_structure()
        model = fit_parameters(improved, data)

        assert model.check_model()


class TestTrainModel:
    """Tests for complete training pipeline."""

    def test_train_model_returns_valid_model(self, data):
        """Test that train_model returns a valid model."""
        model = train_model(data)
        assert model.check_model()

    def test_model_has_risk_variable(self, trained_model):
        """Test that model includes Risk variable."""
        assert "Riesgo" in trained_model.nodes()


class TestQuery:
    """Tests for probabilistic queries."""

    def test_query_risk_by_age(self, trained_model):
        """Test query for risk by age."""
        result = query(trained_model, "Riesgo", {"Edad": "19 to 28"})

        assert "bad" in result["probabilities"]
        assert "good" in result["probabilities"]
        # Probabilities should sum to ~1
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    def test_query_risk_by_duration(self, trained_model):
        """Test query for risk by duration."""
        result = query(trained_model, "Riesgo", {"Duracion": "24 to 72"})

        # Long duration should have higher risk
        assert result["probabilities"]["bad"] > 0.4


class TestPredictRisk:
    """Tests for prediction function."""

    def test_predict_with_historical_eval(self, trained_model):
        """Test prediction with historical evaluation."""
        result = predict_risk(trained_model, historical_eval=0)

        assert result["probability_bad"] > 0.5  # Poor history = high risk
        assert result["prediction"] == "High Risk"

    def test_predict_with_good_profile(self, trained_model):
        """Test prediction with good profile."""
        result = predict_risk(trained_model, historical_eval=4, savings="rich")

        assert result["probability_bad"] < 0.3  # Good profile = low risk

    def test_predict_response_structure(self, trained_model):
        """Test prediction response has all fields."""
        result = predict_risk(trained_model, historical_eval=2)

        assert "prediction" in result
        assert "probability_bad" in result
        assert "probability_good" in result
        assert "risk_level" in result
        assert "evidence_provided" in result


class TestValidationQueries:
    """Tests for validation queries."""

    def test_validation_queries_reasonable(self, trained_model):
        """Test that validation queries return reasonable results."""
        results = run_validation_queries(trained_model)

        # At least some queries should pass (allowing for structure learning variability)
        passed = sum(1 for r in results if r["passed"])
        assert passed >= 2  # At least 2 of 4 should pass
