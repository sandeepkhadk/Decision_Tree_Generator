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
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
except ImportError:  # pragma: no cover - only hit on very old scikit-learn
    HistGradientBoostingClassifier = None
    HistGradientBoostingRegressor = None


MODEL_CATALOG: list[dict[str, str]] = [
    {
        "name": "decision_tree",
        "label": "Decision Tree",
        "icon": "🌳",
        "description": "Configure and visualize a single, fully interpretable decision tree.",
    },
    {
        "name": "random_forest",
        "label": "Random Forest",
        "icon": "🌲",
        "description": "Build and explore an ensemble of bagged decision trees.",
    },
    {
        "name": "extra_trees",
        "label": "Extra Trees",
        "icon": "🌿",
        "description": "Train an extremely randomized tree ensemble for fast, low-variance predictions.",
    },
    {
        "name": "gradient_boosting",
        "label": "Gradient Boosting",
        "icon": "⚡",
        "description": "Sequentially boost shallow trees to minimize prediction error.",
    },
    {
        "name": "hist_gradient_boosting",
        "label": "HistGradientBoosting",
        "icon": "📈",
        "description": "Train a histogram-based boosting model built for large tabular datasets.",
    },
    {
        "name": "adaboost",
        "label": "AdaBoost",
        "icon": "🚀",
        "description": "Combine many weak learners into a single, reweighted ensemble.",
    },
    {
        "name": "xgboost",
        "label": "XGBoost",
        "icon": "🧩",
        "description": "Train a high-performance gradient boosting model with XGBoost.",
    },
    {
        "name": "lightgbm",
        "label": "LightGBM",
        "icon": "🔥",
        "description": "Train a fast, leaf-wise gradient boosting model with LightGBM.",
    },
    {
        "name": "catboost",
        "label": "CatBoost",
        "icon": "🐱",
        "description": "Train a gradient boosting model tuned for categorical-heavy data.",
    },
]

MODEL_LOOKUP = {entry["name"]: entry for entry in MODEL_CATALOG}


def get_model_meta(model_name: str) -> dict[str, str]:
    """Look up display metadata for a model, falling back to a generic entry."""
    return MODEL_LOOKUP.get(
        model_name,
        {"name": model_name, "label": model_name.replace("_", " ").title(), "icon": "🔷", "description": ""},
    )


def get_estimator_count(model) -> int | None:
    """Best-effort count of estimators/iterations a fitted model actually used."""
    for attr in ("n_estimators", "n_iter_", "tree_count_"):
        value = getattr(model, attr, None)
        if isinstance(value, (int, np.integer)):
            return int(value)
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        try:
            return len(estimators)
        except TypeError:
            return None
    return None


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
        if HistGradientBoostingClassifier is None:
            raise ImportError(
                "HistGradientBoosting requires a newer scikit-learn version. Please upgrade scikit-learn."
            )
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
