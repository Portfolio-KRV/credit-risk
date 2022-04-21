# Credit-risk
Developed in the Introduction to Data Science course by: Kevin Reyes.
## Objectives
- Identify risk clients when granting loans using Bayesian Networks.
- Find optimal structure of the Bayesian network using Hill-Climbing algorithm and BICScore method.
- Manually improve network structure so that it is consistent with the business context.

## Description
Bayesian network model for credit risk prediction.

## Conclusions
- The connections found by the Hill-climbing method with BICScore make sense in the context of the data, however, some directions of causalities are reversed.
- The automated construction of the Bayesian network is a good starting point to model a business context, however, it must be analyzed under the business experience to make pertinent modifications.
- Loans with a longer duration have a higher probability of high risk.

## Technology stack
- Pgmpy.
- Networkx.
- Pylab.
- Pandas.
- Numpy.