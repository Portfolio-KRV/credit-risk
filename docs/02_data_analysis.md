# Data Analysis

## Dataset Overview

**File**: `Riesgo_Credito.csv`
**Records**: ~1,000 loan applications
**Features**: 10 attributes including target variable

## Variables Description

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| Evaluacion crediticia historica | Categorical | 0, 1, 2, 3, 4 | Historical credit rating (0=worst, 4=best) |
| Edad | Categorical | "19 to 28", "28 to 38", "38 to 75" | Age ranges |
| Genero | Categorical | "male", "female" | Gender |
| Evaluacion Sueldo | Categorical | 1, 2, 3 | Salary evaluation rating |
| Residencia | Categorical | "own", "free", "rent" | Residence type |
| Nivel de ahorro | Categorical | "no account", "little", "moderate", "quite rich", "rich" | Savings level |
| Monto del credito | Categorical | "250 to 1554", "1554 to 3368", "3368 to 18424" | Credit amount ranges |
| Duracion | Categorical | "4 to 12", "12 to 24", "24 to 72" | Loan duration in months |
| Proposito | Categorical | "radio/TV", "education", "car", "furniture/equipment", "business" | Loan purpose |
| **Riesgo** | Binary | "good", "bad" | **Target variable** - Credit risk |

## Key Observations

### Risk Distribution by Historical Credit Evaluation

| ECH Score | P(Risk=bad) | P(Risk=good) |
|-----------|-------------|--------------|
| 0 (worst) | 64.5% | 35.5% |
| 1 | 59.6% | 40.4% |
| 2 | 32.6% | 67.4% |
| 3 | 35.1% | 64.9% |
| 4 (best) | 18.8% | 81.2% |

**Insight**: Better historical credit evaluation strongly correlates with lower risk.

### Risk by Loan Duration

| Duration | P(Risk=bad) |
|----------|-------------|
| 4 to 12 months | ~25% |
| 12 to 24 months | ~35% |
| 24 to 72 months | 52.2% |

**Insight**: Longer loan durations have significantly higher default risk.

### Risk by Age

| Age Range | P(Risk=bad) |
|-----------|-------------|
| 19 to 28 | 31.2% |
| 28 to 38 | ~28% |
| 38 to 75 | ~25% |

**Insight**: Younger applicants have slightly higher risk.

## Data Quality

- All variables are categorical (discretized)
- No missing values in the dataset
- Variables are already binned into meaningful categories
- Dataset is ready for Bayesian Network learning
