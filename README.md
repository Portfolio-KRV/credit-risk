# Credit Risk Assessment — Bayesian Network

A Bayesian network that estimates loan default risk from partial client
attributes. Built with Hill-Climbing structure search (BIC score), then
hand-corrected so the edges match how a credit officer actually reasons —
not just what the data autocorrelates. Any subset of features can be
provided and the rest are marginalized out.

## What I built

- **Structure learning + expert correction** (`src/model.py`): Hill-Climbing
  with BIC finds the initial DAG; `IMPROVED_EDGES` overrides arc directions
  that the algorithm got wrong (e.g. salary causes credit amount, not
  vice versa).
- **Inference engine** wrapping pgmpy's Variable Elimination. Every endpoint
  returns calibrated probabilities, not just the argmax.
- **FastAPI service** with `/predict`, `/query`, `/risk-by-history` endpoints,
  rate-limited and CORS-enabled via `src/api_common.py`.
- **Validation queries** that check the model produces business-sensible
  answers (e.g. "very poor history" must yield >50% high-risk probability).

## Why it matters

Bayesian networks are one of the few model classes that handle missing
features natively — you don't need to impute. For a credit form where
applicants skip fields, that's a real product win over a black-box
classifier that demands a fixed feature vector.

## Tech stack

pgmpy (Hill-Climbing, MLE, Variable Elimination) · NetworkX · FastAPI · pandas · NumPy

## Quickstart

```bash
# install (one-time)
pip install -e ".[api,notebook,dev]"

# run the API
uvicorn api.app:app --reload --port 8005

# predict risk from a partial profile
curl -X POST http://localhost:8005/predict \
  -H 'Content-Type: application/json' \
  -d '{"historical_eval": 0, "savings": "poor"}'

# inspect risk vs. historical evaluation level
curl http://localhost:8005/risk-by-history
```

## Live demo

Hosted on Hugging Face Spaces:
[kevinreyesds/credit-risk](https://huggingface.co/spaces/kevinreyesds/credit-risk)
*(wired through the portfolio at `/projects/credit-risk/demo`)*

## Tests

```bash
pytest
ruff check .
```

Tests cover structure learning, parameter fitting, probabilistic queries,
the `/predict` contract, and the validation-query suite.

## Repository

[Portfolio-KRV/credit-risk](https://github.com/Portfolio-KRV/credit-risk)

## License

[MIT](LICENSE)
