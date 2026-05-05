# Methodology

## Overview

The project follows a three-stage approach:
1. **Structure Learning**: Discover network topology from data
2. **Structure Refinement**: Expert-guided improvements
3. **Parameter Learning**: Estimate conditional probabilities

## Stage 1: Structure Learning

### Hill-Climbing Algorithm

Hill-Climbing is a local search optimization algorithm that:
1. Starts with an empty or random network structure
2. Iteratively makes local modifications (add/remove/reverse edges)
3. Evaluates each modification using a scoring function
4. Accepts modifications that improve the score
5. Stops when no improvement is possible

### BIC Score (Bayesian Information Criterion)

The BIC score balances model fit with complexity:

```
BIC = log(L) - (k/2) * log(n)
```

Where:
- L = likelihood of the data given the model
- k = number of parameters
- n = number of data points

BIC penalizes complex models, preventing overfitting.

### Implementation

```python
from pgmpy.estimators import HillClimbSearch, BicScore

score_method = BicScore(data=data)
hc = HillClimbSearch(data=data, scoring_method=score_method)
model = hc.estimate()
```

## Stage 2: Structure Refinement

The automatically learned structure may have inverted causalities. Expert review identified 5 edges to reverse:

| Original Edge | Corrected Edge | Justification |
|---------------|----------------|---------------|
| Riesgo → Nivel de ahorro | Nivel de ahorro → Riesgo | Savings determine risk, not vice versa |
| Monto del credito → Evaluacion Sueldo | Evaluacion Sueldo → Monto del credito | Salary determines credit limit |
| Residencia → Genero | Genero → Residencia | Gender may influence residence ownership |
| Monto del credito → Proposito | Proposito → Monto del credito | Purpose determines required amount |
| Duracion → Evaluacion crediticia historica | Evaluacion crediticia historica → Duracion | History determines trust for duration |

## Stage 3: Parameter Learning

### Maximum Likelihood Estimation (MLE)

CPD parameters are learned from data using MLE:

```python
from pgmpy.estimators import MaximumLikelihoodEstimator

bayesian_model.fit(data=data, estimator=MaximumLikelihoodEstimator)
```

MLE estimates P(X|Parents(X)) by counting occurrences in the dataset.

## Inference

### Variable Elimination

For queries like P(Riesgo | Evidence), we use Variable Elimination:

```python
from pgmpy.inference import VariableElimination

infer = VariableElimination(bayesian_model)
result = infer.query(["Riesgo"], evidence={"Edad": "19 to 28"})
```

## Final Network Structure

```
Evaluacion crediticia historica → Riesgo
Evaluacion crediticia historica → Duracion
Edad → Residencia, Genero
Genero → Residencia
Evaluacion Sueldo → Monto del credito
Residencia → Monto del credito
Proposito → Monto del credito
Monto del credito → Duracion, Riesgo
Duracion → Riesgo
Nivel de ahorro → Riesgo
```

## Model Validation

The model is validated by:
1. `model.check_model()` - Ensures CPDs are valid
2. Independence assertions - Verifies d-separation properties
3. Query validation - Compares with expected domain knowledge
