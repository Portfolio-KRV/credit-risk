# Problem Definition

## Context

Credit risk assessment is a critical process in the banking and financial industry. When granting loans, financial institutions need to evaluate the probability that a borrower will default on their obligations. Poor risk assessment can lead to significant financial losses.

This project applies **Bayesian Networks** to model the probabilistic relationships between various client attributes and their credit risk level. Unlike black-box machine learning models, Bayesian Networks provide interpretable causal relationships that can be validated by domain experts.

## Problem Statement

Given a set of client attributes:
- **Historical credit evaluation** (rating 0-4)
- **Age** (age ranges)
- **Gender** (male/female)
- **Salary evaluation** (rating 1-3)
- **Residence type** (own/free/rent)
- **Savings level** (5 categories)
- **Credit amount** (monetary ranges)
- **Duration** (loan duration ranges)
- **Purpose** (loan purpose category)

**Objective**: Predict the credit risk level (good/bad) for loan applicants.

## Approach

We use a two-phase approach:

### Phase 1: Structure Learning
Learn the network structure automatically from data using:
- **Hill-Climbing** search algorithm
- **BIC Score** (Bayesian Information Criterion) as optimization metric

### Phase 2: Expert Refinement
Manually adjust the learned structure based on domain knowledge:
- Correct inverted causal relationships
- Ensure business logic consistency
- Validate against expert assumptions

### Network Structure

The final improved network captures relationships like:
- Savings level → Risk (higher savings = lower risk)
- Credit duration → Risk (longer duration = higher risk)
- Historical evaluation → Duration (better history = more trust)
- Purpose → Credit amount (different purposes need different amounts)

## Success Criteria

1. Learn a valid Bayesian Network structure from data
2. Improve structure based on business domain knowledge
3. Achieve interpretable probability estimates
4. Provide API for real-time risk assessment
5. Validate that longer loans have higher default risk
