# Results

## Query Results

### Query 1: Risk by Age Group

**P(Riesgo = bad | Edad = "19 to 28")**

| Risk | Probability |
|------|-------------|
| bad | **31.2%** |
| good | 68.8% |

**Interpretation**: Young applicants (19-28) have a 31.2% probability of being high-risk clients.

---

### Query 2: Risk by Historical Credit Evaluation

**P(Riesgo | Evaluacion crediticia historica)**

| ECH Score | P(bad) | P(good) |
|-----------|--------|---------|
| 0 (worst) | **64.5%** | 35.5% |
| 1 | **59.6%** | 40.4% |
| 2 | 32.6% | **67.4%** |
| 3 | 35.1% | **64.9%** |
| 4 (best) | 18.8% | **81.2%** |

**Interpretation**: Historical credit evaluation is a strong predictor. Clients with ECH=0 have 3.4x higher risk than those with ECH=4.

---

### Query 3: Risk by Loan Duration

**P(Riesgo = bad | Duracion = "24 to 72")**

| Risk | Probability |
|------|-------------|
| bad | **52.2%** |
| good | 47.8% |

**Interpretation**: Long-term loans (24-72 months) have over 50% probability of default, significantly higher than short-term loans.

---

## Key Findings

1. **Historical credit evaluation is the strongest predictor**
   - ECH=0: 64.5% risk vs ECH=4: 18.8% risk
   - Difference: 45.7 percentage points

2. **Loan duration significantly impacts risk**
   - Short (4-12 months): ~25% risk
   - Long (24-72 months): 52.2% risk
   - Longer loans = higher default probability

3. **Age has moderate influence**
   - Young applicants have slightly higher risk
   - But effect is smaller than ECH or duration

4. **Structure learning successfully identified relationships**
   - Hill-Climbing found meaningful connections
   - Expert refinement improved causal interpretability

## Model Validation

| Aspect | Status |
|--------|--------|
| Model consistency | Passed (`check_model() = True`) |
| Local independencies | Verified |
| Global Markov properties | 42 independence assertions for Risk variable |
| Domain logic | Consistent with business expectations |

## Practical Applications

1. **Loan Approval**: Estimate risk before granting credit
2. **Interest Rate Setting**: Higher risk = higher interest rate
3. **Credit Limit**: Adjust limits based on risk profile
4. **Portfolio Management**: Monitor high-risk segments
5. **Regulatory Compliance**: Explainable risk assessments
