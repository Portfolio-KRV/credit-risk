"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from api.app import app
from src.data import load_data
from src.model import train_model


@pytest.fixture(scope="module")
def client():
    """Create a test client with model loaded."""
    import api.app as app_module
    data = load_data()
    app_module.model = train_model(data)

    with TestClient(app) as client:
        yield client


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_format(self, client):
        """Test health response has expected fields."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_with_historical_eval(self, client):
        """Test prediction with historical evaluation."""
        response = client.post("/predict", json={"historical_eval": 0})

        assert response.status_code == 200
        data = response.json()
        assert data["probability_bad"] > 0.5

    def test_predict_with_multiple_factors(self, client):
        """Test prediction with multiple factors."""
        response = client.post("/predict", json={
            "historical_eval": 2,
            "duration": "12 to 24",
            "savings": "little",
        })

        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data

    def test_predict_empty_evidence(self, client):
        """Test prediction with no evidence."""
        response = client.post("/predict", json={})

        assert response.status_code == 200

    def test_predict_invalid_value(self, client):
        """Test prediction with invalid value."""
        response = client.post("/predict", json={"historical_eval": 10})

        assert response.status_code == 422  # Validation error


class TestRiskByHistoryEndpoint:
    """Tests for risk by history endpoint."""

    def test_risk_by_history_returns_200(self, client):
        """Test endpoint returns 200."""
        response = client.get("/risk-by-history")

        assert response.status_code == 200

    def test_risk_by_history_has_all_levels(self, client):
        """Test all ECH levels are returned."""
        response = client.get("/risk-by-history")
        data = response.json()

        # Should have keys for 0, 1, 2, 3, 4
        assert len(data) == 5


class TestVariablesEndpoint:
    """Tests for variables info endpoint."""

    def test_variables_returns_200(self, client):
        """Test variables endpoint returns 200."""
        response = client.get("/variables")

        assert response.status_code == 200

    def test_variables_has_risk(self, client):
        """Test Risk variable is included."""
        response = client.get("/variables")
        data = response.json()

        assert "Riesgo" in data
