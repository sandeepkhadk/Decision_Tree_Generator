from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, f1_score,
                             mean_squared_error, r2_score)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd
import numpy as np


def determine_problem_type(y):
    """Determine if classification or regression"""
    if pd.api.types.is_numeric_dtype(y):
        return "regression" if y.nunique() > 10 else "classification"
    return "classification"


def create_model(problem_type, params):
    """Instantiate appropriate model"""
    if problem_type == "classification":
        return DecisionTreeClassifier(**params)
    return DecisionTreeRegressor(**params)


def evaluate_model(model, X, y_true, problem_type):
    """Evaluate model performance"""
    y_pred = model.predict(X)

    if problem_type == "classification":
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred, average='weighted')
        }
    return {
        "R² Score": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }
