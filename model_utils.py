from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
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

CLASS_WEIGHT_SUPPORTED_MODELS = {"decision_tree", "random_forest", "extra_trees"}

try:
    from imblearn.over_sampling import SMOTE  # pyright: ignore[reportMissingImports]
    from imblearn.over_sampling import RandomOverSampler  # pyright: ignore[reportMissingImports]
    from imblearn.under_sampling import RandomUnderSampler  # pyright: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency
    SMOTE = None
    RandomOverSampler = None
    RandomUnderSampler = None

IMBLEARN_AVAILABLE = SMOTE is not None


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


def supports_class_weight(model_name: str) -> bool:
    return model_name in CLASS_WEIGHT_SUPPORTED_MODELS


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


def predict_classes_from_proba(y_proba, threshold: float = 0.5):
    """Convert class probabilities into predicted labels using a threshold for binary tasks."""
    probabilities = np.asarray(y_proba)
    if probabilities.ndim == 1:
        return (probabilities >= threshold).astype(int)
    if probabilities.shape[1] == 2:
        return (probabilities[:, 1] >= threshold).astype(int)
    return probabilities.argmax(axis=1)


def resample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, str | None]:
    """Apply simple random over/under sampling to the training split."""
    if strategy == "none":
        return X_train, y_train, None

    balanced_strategy = strategy.lower().replace(" ", "_")
    if balanced_strategy not in {"random_oversample", "random_undersample", "smote"}:
        raise ValueError("Unsupported resampling strategy.")

    if balanced_strategy == "smote":
        if SMOTE is None:
            raise ImportError("SMOTE requires imbalanced-learn. Install imbalanced-learn to use this option.")
        class_counts = y_train.value_counts()
        if len(class_counts) < 2:
            return X_train, y_train, None
        minority_count = int(class_counts.min())
        if minority_count < 2:
            raise ValueError("SMOTE requires at least 2 samples in the minority class.")
        k_neighbors = min(5, minority_count - 1)
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return (
            pd.DataFrame(X_resampled, columns=X_train.columns),
            pd.Series(y_resampled, name=y_train.name),
            f"SMOTE generated synthetic samples with k_neighbors={k_neighbors}.",
        )

    frame = X_train.copy()
    frame["__target__"] = y_train.values
    counts = frame["__target__"].value_counts()
    if len(counts) < 2:
        return X_train, y_train, None

    if balanced_strategy == "random_oversample":
        if RandomOverSampler is not None:
            sampler = RandomOverSampler(random_state=random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            return (
                pd.DataFrame(X_resampled, columns=X_train.columns),
                pd.Series(y_resampled, name=y_train.name),
                "Random oversampling balanced the training classes.",
            )
        target_size = int(counts.max())
        sampled_frames = []
        for label, group in frame.groupby("__target__", sort=False):
            sampled_frames.append(group.sample(n=target_size, replace=True, random_state=random_state))
        balanced = pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=random_state)
        return (
            balanced.drop(columns=["__target__"]).reset_index(drop=True),
            balanced["__target__"].reset_index(drop=True),
            f"Random oversampling expanded each class to {target_size} rows.",
        )

    if RandomUnderSampler is not None:
        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return (
            pd.DataFrame(X_resampled, columns=X_train.columns),
            pd.Series(y_resampled, name=y_train.name),
            "Random undersampling balanced the training classes.",
        )

    target_size = int(counts.min())
    sampled_frames = []
    for label, group in frame.groupby("__target__", sort=False):
        sampled_frames.append(group.sample(n=target_size, replace=False, random_state=random_state))
    balanced = pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=random_state)
    return (
        balanced.drop(columns=["__target__"]).reset_index(drop=True),
        balanced["__target__"].reset_index(drop=True),
        f"Random undersampling reduced each class to {target_size} rows.",
    )


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


def evaluate_predictions(
    y_true,
    y_pred,
    problem_type: str,
    y_proba=None,
    class_labels: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute a metrics dictionary for either classification or regression."""
    if problem_type == "classification":
        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        if class_labels is not None:
            report_labels = list(range(len(class_labels)))
            target_names = [str(label) for label in class_labels]
        else:
            report_labels = list(pd.Index(pd.Series(y_true_array).astype(str)).unique())
            target_names = [str(label) for label in report_labels]

        report = classification_report(
            y_true_array,
            y_pred_array,
            labels=report_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        warnings: list[str] = []
        zero_recall_classes: list[str] = []
        if len(np.unique(y_pred_array)) == 1:
            warnings.append("Model predicts only one class - check for class imbalance.")
        for label_name in target_names:
            class_row = report.get(label_name, {})
            if float(class_row.get("recall", 0.0)) == 0.0:
                warnings.append(f"Class {label_name} has 0 recall.")
                zero_recall_classes.append(str(label_name))

        roc_auc = None
        pr_auc = None
        if y_proba is not None:
            probabilities = np.asarray(y_proba)
            if len(report_labels) == 2 and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
                positive_scores = probabilities[:, 1]
                roc_auc = roc_auc_score(y_true_array, positive_scores)
                pr_auc = average_precision_score(y_true_array, positive_scores)
            elif probabilities.ndim == 2 and probabilities.shape[1] == len(report_labels):
                y_binarized = label_binarize(y_true_array, classes=report_labels)
                roc_auc = roc_auc_score(y_binarized, probabilities, average="macro", multi_class="ovr")
                pr_auc = average_precision_score(y_binarized, probabilities, average="macro")

        result = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true_array, y_pred_array, average="weighted", zero_division=0),
            "Recall": recall_score(y_true_array, y_pred_array, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0),
            "Macro Precision": precision_score(y_true_array, y_pred_array, average="macro", zero_division=0),
            "Macro Recall": recall_score(y_true_array, y_pred_array, average="macro", zero_division=0),
            "Macro F1": f1_score(y_true_array, y_pred_array, average="macro", zero_division=0),
            "Confusion Matrix": confusion_matrix(y_true_array, y_pred_array, labels=report_labels),
            "Classification Report": report,
            "Labels": target_names,
            "Per Class Metrics": {
                label: report[label]
                for label in target_names
                if label in report
            },
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc,
            "Warnings": warnings,
            "Degenerate Prediction": len(np.unique(y_pred_array)) == 1,
            "Zero Recall Classes": zero_recall_classes,
            "Decision Threshold": threshold,
        }
        return result

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }
