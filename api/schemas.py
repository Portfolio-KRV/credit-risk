"""Pydantic schemas for API request/response validation."""

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for credit risk prediction."""

    historical_eval: int | None = Field(
        None,
        ge=0,
        le=4,
        description="Historical credit evaluation (0=worst to 4=best)",
        examples=[2],
    )
    age: Literal["19 to 28", "28 to 38", "38 to 75"] | None = Field(
        None,
        description="Age range",
        examples=["28 to 38"],
    )
    gender: Literal["male", "female"] | None = Field(
        None,
        description="Gender",
        examples=["male"],
    )
    salary_eval: int | None = Field(
        None,
        ge=1,
        le=3,
        description="Salary evaluation (1=low to 3=high)",
        examples=[2],
    )
    residence: Literal["own", "free", "rent"] | None = Field(
        None,
        description="Residence type",
        examples=["own"],
    )
    savings: Literal["no account", "little", "moderate", "quite rich", "rich"] | None = Field(
        None,
        description="Savings level",
        examples=["moderate"],
    )
    credit_amount: Literal["250 to 1554", "1554 to 3368", "3368 to 18424"] | None = Field(
        None,
        description="Credit amount range",
        examples=["1554 to 3368"],
    )
    duration: Literal["4 to 12", "12 to 24", "24 to 72"] | None = Field(
        None,
        description="Loan duration in months",
        examples=["12 to 24"],
    )
    purpose: Literal["radio/TV", "education", "car", "furniture/equipment", "business"] | None = Field(
        None,
        description="Loan purpose",
        examples=["car"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "historical_eval": 2,
                    "age": "28 to 38",
                    "duration": "12 to 24",
                    "savings": "little",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for credit risk prediction."""

    prediction: str = Field(
        ...,
        description="Prediction: 'High Risk' or 'Low Risk'",
        examples=["Low Risk"],
    )
    probability_bad: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of high risk (bad)",
        examples=[0.32],
    )
    probability_good: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of low risk (good)",
        examples=[0.68],
    )
    risk_level: str = Field(
        ...,
        description="Risk level: 'Low', 'Medium', or 'High'",
        examples=["Medium"],
    )
    evidence_provided: dict = Field(
        ...,
        description="The evidence used for prediction",
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    version: str = Field(..., examples=["1.0.0"])


class QueryRequest(BaseModel):
    """Request schema for custom probabilistic queries."""

    target: str = Field(
        ...,
        description="Target variable to query",
        examples=["Riesgo"],
    )
    evidence: dict | None = Field(
        None,
        description="Evidence variables",
        examples=[{"Edad": "19 to 28"}],
    )


class QueryResponse(BaseModel):
    """Response schema for custom queries."""

    variable: str
    variable_name: str
    evidence: dict
    probabilities: dict
