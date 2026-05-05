"""FastAPI application for Credit Risk assessment."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src import __version__
from src.api_common import create_app, limiter, register_error_handlers
from src.data import load_data
from src.evaluate import get_risk_by_historical_eval, predict_risk, query
from src.exceptions import (
    ConfigurationError,
    DataError,
    DataValidationError,
    ModelError,
    ModelLoadError,
    ModelNotLoadedError,
)
from src.model import get_or_create_model
from src.validators import validate_numeric_range

from .schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    data = load_data()
    model = get_or_create_model(data)
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down")


app = create_app(
    title="Credit Risk Assessment API",
    description="""
    API for credit risk assessment using Bayesian Networks.

    This model uses Hill-Climbing structure learning with BIC score,
    followed by expert-guided improvements and Maximum Likelihood parameter estimation.

    ## Variables

    - **Historical Credit Evaluation** (0-4): Past credit history rating
    - **Age**: Age range of the applicant
    - **Gender**: Male or female
    - **Salary Evaluation** (1-3): Income assessment
    - **Residence**: Ownership status (own, free, rent)
    - **Savings Level**: Amount of savings
    - **Credit Amount**: Requested loan amount range
    - **Duration**: Loan duration in months
    - **Purpose**: Reason for the loan

    ## Usage

    Provide any combination of known client attributes to get a credit risk estimate.
    """,
    version=__version__,
    lifespan=lifespan,
)
register_error_handlers(
    app,
    {
        ModelNotLoadedError: (503, "model_not_loaded"),
        ModelLoadError: (500, "model_load_error"),
        ModelError: (500, "model_error"),
        DataValidationError: (422, "validation_error"),
        DataError: (500, "data_error"),
        ConfigurationError: (500, "configuration_error"),
    },
    expose_message=(DataValidationError,),
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit("60/minute")
async def health_check(request: Request):
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version=__version__,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
@limiter.limit("20/minute")
async def predict(request: Request, body: PredictionRequest):
    """Predict credit risk given client attributes.

    Provide any combination of known attributes. Unspecified attributes
    will be marginalized out using the model's learned distributions.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if body.historical_eval is not None:
            validate_numeric_range(float(body.historical_eval), 0.0, 4.0, "historical_eval")
        if body.salary_eval is not None:
            validate_numeric_range(float(body.salary_eval), 1.0, 3.0, "salary_eval")

        result = predict_risk(
            model,
            historical_eval=body.historical_eval,
            age=body.age,
            gender=body.gender,
            salary_eval=body.salary_eval,
            residence=body.residence,
            savings=body.savings,
            credit_amount=body.credit_amount,
            duration=body.duration,
            purpose=body.purpose,
        )

        return PredictionResponse(**result)
    except ValueError as e:
        raise DataValidationError(str(e)) from e


@app.post("/query", response_model=QueryResponse, tags=["Advanced"])
@limiter.limit("20/minute")
async def custom_query(request: Request, body: QueryRequest):
    """Execute a custom probabilistic query on the Bayesian Network."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        result = query(model, body.target, body.evidence)
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing query: {e!s}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid query parameters. Please check your input.",
        ) from e


@app.get("/risk-by-history", tags=["Analysis"])
@limiter.limit("60/minute")
async def risk_by_historical_evaluation(request: Request):
    """Get risk probabilities for each historical credit evaluation level."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return get_risk_by_historical_eval(model)


@app.get("/variables", tags=["Info"])
@limiter.limit("60/minute")
async def get_variables(request: Request):
    """Get information about model variables and their valid values."""
    from src.config import VARIABLES
    return VARIABLES
