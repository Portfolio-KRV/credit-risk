"""Evaluation and inference for the Credit Risk model."""

from typing import Any

from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

from .config import COLUMNS, VARIABLES


def create_inference_engine(model: BayesianNetwork) -> VariableElimination:
    """Create an inference engine for the model.

    Args:
        model: BayesianNetwork model

    Returns:
        VariableElimination inference object
    """
    return VariableElimination(model)


def query(
    model: BayesianNetwork,
    target: str,
    evidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Perform a probabilistic query on the model.

    Args:
        model: BayesianNetwork model
        target: Variable to query
        evidence: Dictionary of observed variables

    Returns:
        Dictionary with state probabilities
    """
    infer = create_inference_engine(model)
    result = infer.query([target], evidence=evidence or {})

    states = result.state_names[target]
    probabilities = result.values.tolist()

    return {
        "variable": target,
        "variable_name": VARIABLES.get(target, {}).get("name_en", target),
        "evidence": evidence or {},
        "probabilities": {str(state): round(prob, 4) for state, prob in zip(states, probabilities, strict=False)},
    }


def predict_risk(
    model: BayesianNetwork,
    historical_eval: int | None = None,
    age: str | None = None,
    gender: str | None = None,
    salary_eval: int | None = None,
    residence: str | None = None,
    savings: str | None = None,
    credit_amount: str | None = None,
    duration: str | None = None,
    purpose: str | None = None,
) -> dict[str, Any]:
    """Predict credit risk given client attributes.

    Args:
        model: BayesianNetwork model
        historical_eval: Historical credit evaluation (0-4)
        age: Age range ("19 to 28", "28 to 38", "38 to 75")
        gender: Gender ("male", "female")
        salary_eval: Salary evaluation (1-3)
        residence: Residence type ("own", "free", "rent")
        savings: Savings level ("no account", "little", "moderate", "quite rich", "rich")
        credit_amount: Credit amount range
        duration: Loan duration range
        purpose: Loan purpose

    Returns:
        Prediction result with probabilities
    """
    evidence = {}

    if historical_eval is not None:
        evidence[COLUMNS["historical_eval"]] = str(historical_eval)
    if age is not None:
        evidence[COLUMNS["age"]] = age
    if gender is not None:
        evidence[COLUMNS["gender"]] = gender
    if salary_eval is not None:
        evidence[COLUMNS["salary_eval"]] = str(salary_eval)
    if residence is not None:
        evidence[COLUMNS["residence"]] = residence
    if savings is not None:
        evidence[COLUMNS["savings"]] = savings
    if credit_amount is not None:
        evidence[COLUMNS["credit_amount"]] = credit_amount
    if duration is not None:
        evidence[COLUMNS["duration"]] = duration
    if purpose is not None:
        evidence[COLUMNS["purpose"]] = purpose

    result = query(model, COLUMNS["risk"], evidence)

    prob_bad = result["probabilities"].get("bad", 0)
    prob_good = result["probabilities"].get("good", 0)

    return {
        "prediction": "High Risk" if prob_bad > 0.5 else "Low Risk",
        "probability_bad": prob_bad,
        "probability_good": prob_good,
        "risk_level": _get_risk_level(prob_bad),
        "evidence_provided": {
            "historical_eval": historical_eval,
            "age": age,
            "gender": gender,
            "salary_eval": salary_eval,
            "residence": residence,
            "savings": savings,
            "credit_amount": credit_amount,
            "duration": duration,
            "purpose": purpose,
        },
    }


def _get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability >= 0.6:
        return "High"
    elif probability >= 0.35:
        return "Medium"
    else:
        return "Low"


def run_validation_queries(model: BayesianNetwork) -> list[dict[str, Any]]:
    """Run validation queries from the notebook.

    Args:
        model: BayesianNetwork model

    Returns:
        List of query results
    """
    queries = [
        {
            "name": "P(Risk=bad | Age=19-28)",
            "target": COLUMNS["risk"],
            "evidence": {COLUMNS["age"]: "19 to 28"},
            "expected": 0.312,
            "state": "bad",
        },
        {
            "name": "P(Risk=bad | Duration=24-72)",
            "target": COLUMNS["risk"],
            "evidence": {COLUMNS["duration"]: "24 to 72"},
            "expected": 0.522,
            "state": "bad",
        },
        {
            "name": "P(Risk=bad | ECH=0)",
            "target": COLUMNS["risk"],
            "evidence": {COLUMNS["historical_eval"]: "0"},
            "expected": 0.645,
            "state": "bad",
        },
        {
            "name": "P(Risk=bad | ECH=4)",
            "target": COLUMNS["risk"],
            "evidence": {COLUMNS["historical_eval"]: "4"},
            "expected": 0.188,
            "state": "bad",
        },
    ]

    results = []
    for q in queries:
        result = query(model, q["target"], q["evidence"])
        actual = result["probabilities"].get(q["state"], 0)
        passed = abs(actual - q["expected"]) < 0.05  # 5% tolerance

        results.append({
            "query": q["name"],
            "expected": q["expected"],
            "actual": actual,
            "passed": passed,
        })

    return results


def get_risk_by_historical_eval(model: BayesianNetwork) -> dict[str, dict[str, float]]:
    """Get risk probabilities for each historical evaluation level.

    Args:
        model: BayesianNetwork model

    Returns:
        Dictionary mapping ECH level to risk probabilities
    """
    results = {}
    for ech in range(5):
        result = query(
            model,
            COLUMNS["risk"],
            {COLUMNS["historical_eval"]: str(ech)}
        )
        results[str(ech)] = {
            "bad": result["probabilities"].get("bad", 0),
            "good": result["probabilities"].get("good", 0),
        }
    return results
