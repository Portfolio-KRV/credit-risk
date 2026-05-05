"""Configuration for the Credit Risk model."""

from pathlib import Path
from typing import Any

# Project paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data file
DATA_FILE: Path = DATA_DIR / "Riesgo_Credito.csv"

# Model file
MODEL_FILE: Path = MODELS_DIR / "bayesian_network.pkl"

# Column names (Spanish - as in original dataset)
COLUMNS: dict[str, str] = {
    "historical_eval": "Evaluacion crediticia historica",
    "age": "Edad",
    "gender": "Genero",
    "salary_eval": "Evaluacion Sueldo",
    "residence": "Residencia",
    "savings": "Nivel de ahorro",
    "credit_amount": "Monto del credito",
    "duration": "Duracion",
    "purpose": "Proposito",
    "risk": "Riesgo",
}

# Variable metadata for API
VARIABLES: dict[str, dict[str, Any]] = {
    "Evaluacion crediticia historica": {
        "name_en": "Historical Credit Evaluation",
        "name_es": "Evaluacion Crediticia Historica",
        "values": [0, 1, 2, 3, 4],
        "labels": {0: "Very Poor", 1: "Poor", 2: "Average", 3: "Good", 4: "Excellent"},
    },
    "Edad": {
        "name_en": "Age",
        "name_es": "Edad",
        "values": ["19 to 28", "28 to 38", "38 to 75"],
        "labels": {"19 to 28": "Young", "28 to 38": "Adult", "38 to 75": "Senior"},
    },
    "Genero": {
        "name_en": "Gender",
        "name_es": "Genero",
        "values": ["male", "female"],
        "labels": {"male": "Male", "female": "Female"},
    },
    "Evaluacion Sueldo": {
        "name_en": "Salary Evaluation",
        "name_es": "Evaluacion Sueldo",
        "values": [1, 2, 3],
        "labels": {1: "Low", 2: "Medium", 3: "High"},
    },
    "Residencia": {
        "name_en": "Residence",
        "name_es": "Residencia",
        "values": ["own", "free", "rent"],
        "labels": {"own": "Owner", "free": "Free", "rent": "Renter"},
    },
    "Nivel de ahorro": {
        "name_en": "Savings Level",
        "name_es": "Nivel de Ahorro",
        "values": ["no account", "little", "moderate", "quite rich", "rich"],
        "labels": {
            "no account": "No Account",
            "little": "Little",
            "moderate": "Moderate",
            "quite rich": "Quite Rich",
            "rich": "Rich",
        },
    },
    "Monto del credito": {
        "name_en": "Credit Amount",
        "name_es": "Monto del Credito",
        "values": ["250 to 1554", "1554 to 3368", "3368 to 18424"],
        "labels": {
            "250 to 1554": "Low (250-1554)",
            "1554 to 3368": "Medium (1554-3368)",
            "3368 to 18424": "High (3368-18424)",
        },
    },
    "Duracion": {
        "name_en": "Duration",
        "name_es": "Duracion",
        "values": ["4 to 12", "12 to 24", "24 to 72"],
        "labels": {
            "4 to 12": "Short (4-12 months)",
            "12 to 24": "Medium (12-24 months)",
            "24 to 72": "Long (24-72 months)",
        },
    },
    "Proposito": {
        "name_en": "Purpose",
        "name_es": "Proposito",
        "values": ["radio/TV", "education", "car", "furniture/equipment", "business"],
        "labels": {
            "radio/TV": "Radio/TV",
            "education": "Education",
            "car": "Car",
            "furniture/equipment": "Furniture",
            "business": "Business",
        },
    },
    "Riesgo": {
        "name_en": "Risk",
        "name_es": "Riesgo",
        "values": ["good", "bad"],
        "labels": {"good": "Low Risk", "bad": "High Risk"},
    },
}

# Structure improvements (edges to reverse)
# Format: (from_node, to_node) - these edges should be reversed from what Hill-Climbing finds
EDGE_IMPROVEMENTS: list[tuple[str, str, str, str]] = [
    ("Riesgo", "Nivel de ahorro", "Nivel de ahorro", "Riesgo"),
    ("Monto del credito", "Evaluacion Sueldo", "Evaluacion Sueldo", "Monto del credito"),
    ("Residencia", "Genero", "Genero", "Residencia"),
    ("Monto del credito", "Proposito", "Proposito", "Monto del credito"),
    ("Duracion", "Evaluacion crediticia historica", "Evaluacion crediticia historica", "Duracion"),
]
