from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
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
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
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
    target_encoder: LabelEncoder | None = None


def determine_problem_type(y: pd.Series) -> str:
    """Infer whether the target is best treated as classification or regression."""
    if pd.api.types.is_numeric_dtype(y):
        return "regression" if y.nunique(dropna=True) > 10 else "classification"
    return "classification"


def encode_classification_target(y: pd.Series) -> tuple[pd.Series, LabelEncoder]:
    encoder = LabelEncoder()
    encoded = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
    return encoded, encoder


def decode_predictions(predictions, target_encoder: LabelEncoder | None):
    if target_encoder is None:
        return predictions
    array = np.asarray(predictions)
    decoded = target_encoder.inverse_transform(array.astype(int))
    return decoded


def supports_optional_booster(model_name: str) -> bool:
    return model_name in {"xgboost", "lightgbm", "catboost"}


def create_model(model_name: str, problem_type: str, params: dict[str, Any]):
    """Create a tree-based estimator for the detected problem type."""
    if model_name == "decision_tree":
        if problem_type == "classification":
            return DecisionTreeClassifier(**params)
        return DecisionTreeRegressor(**params)

    if model_name == "random_forest":
        if problem_type == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    if model_name == "extra_trees":
        if problem_type == "classification":
            return ExtraTreesClassifier(**params)
        return ExtraTreesRegressor(**params)

    if model_name == "gradient_boosting":
        if problem_type == "classification":
            return GradientBoostingClassifier(**params)
        return GradientBoostingRegressor(**params)

    if model_name == "hist_gradient_boosting":
        if problem_type == "classification":
            return HistGradientBoostingClassifier(**params)
        return HistGradientBoostingRegressor(**params)

    if model_name == "adaboost":
        if problem_type == "classification":
            return AdaBoostClassifier(**params)
        return AdaBoostRegressor(**params)

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor  # pyright: ignore[reportMissingImports]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "XGBoost is not installed. Add xgboost to requirements.txt and reinstall dependencies."
            ) from exc
        if problem_type == "classification":
            return XGBClassifier(**params)
        return XGBRegressor(**params)

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor  # pyright: ignore[reportMissingImports]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LightGBM is not installed. Add lightgbm to requirements.txt and reinstall dependencies."
            ) from exc
        if problem_type == "classification":
            return LGBMClassifier(**params)
        return LGBMRegressor(**params)

    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor  # pyright: ignore[reportMissingImports]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "CatBoost is not installed. Add catboost to requirements.txt and reinstall dependencies."
            ) from exc
        if problem_type == "classification":
            return CatBoostClassifier(**params)
        return CatBoostRegressor(**params)

    raise ValueError(f"Unsupported model type: {model_name}")


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
