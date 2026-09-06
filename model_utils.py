from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass(slots=True)
class ModelArtifacts:
    pipeline: Any
    preprocessor: ColumnTransformer
    feature_names: list[str]
    problem_type: str
    target_name: str
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    classes_: list[str] | None = None


def determine_problem_type(y: pd.Series) -> str:
    """Infer whether the target is best treated as classification or regression."""
    if pd.api.types.is_numeric_dtype(y):
        return "regression" if y.nunique(dropna=True) > 10 else "classification"
    return "classification"


def create_model(problem_type: str, params: dict[str, Any]):
    """Create a decision tree estimator for the detected problem type."""
    if problem_type == "classification":
        return DecisionTreeClassifier(**params)
    return DecisionTreeRegressor(**params)


def evaluate_predictions(y_true, y_pred, problem_type: str) -> dict[str, Any]:
    """Compute a metrics dictionary for either classification or regression."""
    if problem_type == "classification":
        labels = pd.Index(y_true.astype(str)).unique()
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_true, y_pred),
            "Classification Report": classification_report(
                y_true,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
            "Labels": [str(label) for label in labels],
        }

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }
