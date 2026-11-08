# Airline Customer Satisfaction: Decision Tree Classifier

Predicting whether an airline passenger will be satisfied with their flight, and finding which features drive satisfaction most, using a decision tree built with scikit-learn.

## Overview

This project works through a consulting scenario for an airline. Using survey responses from 129,880 customers (class, flight distance, inflight entertainment, seat comfort and more), I build, tune and evaluate a decision tree classifier to predict satisfaction.

## Workflow

1. Load and explore the data (data types, class balance, missing values)
2. Clean the data (drop 393 rows with missing values, leaving 129,487)
3. Encode features (ordinal mapping for `Class`, binary target, one-hot encoding with `drop_first=True`)
4. Split into 75% train and 25% test sets
5. Fit a baseline `DecisionTreeClassifier`
6. Evaluate with accuracy, precision, recall, F1 and a confusion matrix
7. Tune `max_depth` and `min_samples_leaf` with `GridSearchCV` (5-fold CV, refit on F1)
8. Visualize the tree and review feature importances

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline tree (test set) | 0.935 | 0.943 | 0.939 | 0.941 |
| Tuned tree (cross-validated) | 0.941 | 0.955 | 0.936 | 0.945 |

- Best hyperparameters: `max_depth=18`, `min_samples_leaf=2`
- Most important features: inflight entertainment (0.42), seat comfort (0.19), ease of online booking (0.07)
- The first split is on inflight entertainment, and errors are balanced between false positives and false negatives

## Recommendations

- Prioritize investment in inflight entertainment and seat comfort, then the online booking experience
- Use the model to flag customers likely to be dissatisfied
- Test a random forest or gradient boosting model before deployment, since a single tree can overfit

## Repository contents

- `airline_satisfaction_decision_tree.ipynb`: the full analysis notebook with outputs

## Data

Invistico_Airline.csv; The dataset is from a Coursera course activity and is not included in this repo.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook airline_satisfaction_decision_tree.ipynb
```

Note: the grid search covers 276 parameter combinations with 5-fold CV and can take several minutes.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
