from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    APP_NAME,
    APP_TAGLINE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MIN_SAMPLES_LEAF,
    DEFAULT_MIN_SAMPLES_SPLIT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
)
from data_utils import (
    build_signature,
    detect_supported_columns,
    file_hash,
    load_dataset,
    summarize_dataset,
    validate_file_size,
)
from model_utils import (
    ModelArtifacts,
    decode_predictions,
    encode_classification_target,
    create_model,
    determine_problem_type,
    evaluate_predictions,
)
from preprocessing import create_preprocessor, detect_column_types, get_feature_names
from tree_utils import figure_to_png_bytes, get_tree_depth, plot_tree_figure


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


st.set_page_config(page_title=APP_NAME, page_icon="🌳", layout="wide")


CSS = """
<style>
    .main { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .hero-card {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06);
    }
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
    }
    .status-success {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    .status-error {
        background: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
</style>
"""


@st.cache_data(show_spinner=False)
def cached_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return load_dataset(file_bytes, filename)


def init_state() -> None:
    defaults = {
        "dataset": None,
        "dataset_name": None,
        "dataset_signature": None,
        "summary": None,
        "artifacts": None,
        "metrics": None,
        "train_metrics": None,
        "test_metrics": None,
        "prediction": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_header() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin:0;font-size:2.2rem;">{APP_NAME}</h1>
            <p style="margin:0.4rem 0 0;color:#475569;font-size:1.02rem;">{APP_TAGLINE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_overview(summary: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Duplicates", summary["duplicate_rows"])
    col4.metric("Missing cells", int(summary["missing_values"].sum()))


def render_summary_tables(summary: dict[str, Any]) -> None:
    tabs = st.tabs(["Preview", "Columns", "Missing Values", "Statistics"])
    with tabs[0]:
        st.dataframe(summary["preview"], use_container_width=True)
    with tabs[1]:
        st.dataframe(
            pd.DataFrame(
                {
                    "Column": summary["column_names"],
                    "Data type": [summary["dtypes"][name] for name in summary["column_names"]],
                }
            ),
            use_container_width=True,
        )
    with tabs[2]:
        st.dataframe(summary["missing_values"].reset_index(), use_container_width=True)
    with tabs[3]:
        if summary["numeric_summary"].empty:
            st.info("No numeric columns were detected for descriptive statistics.")
        else:
            st.dataframe(summary["numeric_summary"], use_container_width=True)


def build_training_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: list[str],
    target_name: str,
    problem_type: str,
    model_name: str,
    tree_params: dict[str, Any],
) -> tuple[Pipeline, list[str], list[str] | None, Any | None]:
    numeric_cols, categorical_cols = detect_column_types(X_train)
    preprocessor = create_preprocessor(numeric_cols, categorical_cols, X_train)
    estimator = create_model(model_name, problem_type, tree_params)
    target_encoder = None
    y_fit = y_train
    if problem_type == "classification":
        y_fit, target_encoder = encode_classification_target(y_train)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_fit)

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_preprocessor)
    class_names = [str(value) for value in target_encoder.classes_] if target_encoder is not None else None

    artifacts = ModelArtifacts(
        pipeline=pipeline,
        preprocessor=fitted_preprocessor,
        feature_names=feature_names,
        problem_type=problem_type,
        target_name=target_name,
        feature_columns=feature_columns,
        categorical_columns=categorical_cols,
        numeric_columns=numeric_cols,
        classes_=class_names,
        target_encoder=target_encoder,
    )
    st.session_state.artifacts = artifacts
    return pipeline, feature_names, class_names, target_encoder


def render_feature_importance(model, feature_names: list[str]) -> None:
    if not hasattr(model, "feature_importances_"):
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances[importances > 0].sort_values(ascending=False).head(20)
    if importances.empty:
        return

    st.markdown("### Feature Importance")
    figure, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(importances) + 2)), dpi=180)
    importances.sort_values().plot(kind="barh", ax=axis, color="#2563eb")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title("Top Features")
    figure.tight_layout()
    st.pyplot(figure, clear_figure=False)


def render_prediction_form(artifacts: ModelArtifacts, dataset: pd.DataFrame) -> None:
    st.subheader("Prediction")
    if artifacts is None:
        st.info("Train a model to enable predictions.")
        return

    with st.form("prediction_form"):
        inputs: dict[str, Any] = {}
        for column in artifacts.feature_columns:
            series = dataset[column]
            if pd.api.types.is_numeric_dtype(series):
                min_value = float(series.min()) if series.notna().any() else 0.0
                max_value = float(series.max()) if series.notna().any() else 1.0
                default_value = float(series.dropna().median()) if series.notna().any() else 0.0
                inputs[column] = st.number_input(column, value=default_value, min_value=min_value, max_value=max_value)
            else:
                options = [str(value) for value in series.dropna().astype(str).unique().tolist()[:200]]
                options = options or [""]
                inputs[column] = st.selectbox(column, options=options)

        submitted = st.form_submit_button("Generate Prediction")

    if not submitted:
        return

    input_frame = pd.DataFrame([inputs])
    prediction = artifacts.pipeline.predict(input_frame)[0]
    prediction = decode_predictions([prediction], artifacts.target_encoder)[0]
    st.session_state.prediction = prediction

    st.markdown(
        f"<div class='metric-card'><h3 style='margin:0;'>Prediction</h3><p style='font-size:1.35rem;margin:0.35rem 0 0;'><strong>{prediction}</strong></p></div>",
        unsafe_allow_html=True,
    )

    if artifacts.problem_type == "classification" and hasattr(artifacts.pipeline.named_steps["model"], "predict_proba"):
        probabilities = artifacts.pipeline.predict_proba(input_frame)[0]
        class_labels = artifacts.target_encoder.classes_ if artifacts.target_encoder is not None else artifacts.pipeline.named_steps["model"].classes_
        proba_frame = pd.DataFrame({"Class": class_labels, "Probability": probabilities})
        st.dataframe(proba_frame, use_container_width=True)


def render_evaluation(metrics: dict[str, Any], problem_type: str) -> None:
    st.subheader("Model Evaluation")
    if problem_type == "classification":
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
        col2.metric("Precision", f"{metrics['Precision']:.3f}")
        col3.metric("Recall", f"{metrics['Recall']:.3f}")
        col4.metric("F1-score", f"{metrics['F1 Score']:.3f}")
        st.dataframe(pd.DataFrame(metrics["Classification Report"]).transpose(), use_container_width=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics['MAE']:.3f}")
        col2.metric("MSE", f"{metrics['MSE']:.3f}")
        col3.metric("RMSE", f"{metrics['RMSE']:.3f}")
        col4.metric("R²", f"{metrics['R2']:.3f}")


def render_confusion_matrix(metrics: dict[str, Any]) -> None:
    if "Confusion Matrix" not in metrics:
        return

    labels = metrics.get("Labels", [])
    matrix = metrics["Confusion Matrix"]
    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Confusion Matrix")
    st.pyplot(figure, clear_figure=True)


def main() -> None:
    init_state()
    render_header()

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx", "xls"])
        st.caption("Files are validated locally and never executed.")

    if uploaded_file is not None:
        try:
            validate_file_size(getattr(uploaded_file, "size", None))
            file_bytes = uploaded_file.getvalue()
            dataset_signature = file_hash(file_bytes)
            if dataset_signature != st.session_state.dataset_signature:
                dataset = cached_dataset(file_bytes, uploaded_file.name)
                if dataset.empty:
                    raise ValueError("The uploaded dataset is empty. Please provide a file with at least one row.")
                st.session_state.dataset = dataset
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.dataset_signature = dataset_signature
                st.session_state.summary = summarize_dataset(dataset, uploaded_file.name)
                st.session_state.artifacts = None
                st.session_state.metrics = None
                st.session_state.prediction = None
        except Exception as exc:
            LOGGER.exception("Dataset upload failed")
            st.error(str(exc))
            return

    dataset = st.session_state.dataset
    if dataset is None:
        st.info("Upload a dataset to begin.")
        st.stop()

    summary = st.session_state.summary
    render_dataset_overview(summary)

    st.markdown("### Dataset Explorer")
    render_summary_tables(summary)

    numeric_cols, categorical_cols, unsupported_cols = detect_supported_columns(dataset)
    if unsupported_cols:
        st.warning(f"Unsupported columns were detected and will be ignored automatically: {', '.join(unsupported_cols)}")

    st.markdown("### Preprocessing and Model Setup")
    target_col = st.selectbox("Target column", options=list(dataset.columns), index=len(dataset.columns) - 1)
    feature_candidates = [column for column in dataset.columns if column != target_col and column not in unsupported_cols]
    selected_features = st.multiselect(
        "Feature columns",
        options=feature_candidates,
        default=feature_candidates,
    )
    if not selected_features:
        st.warning("Select at least one feature column.")
        st.stop()

    train_size = 1 - st.slider("Test split", min_value=0.1, max_value=0.5, value=float(DEFAULT_TEST_SIZE), step=0.05)
    random_state = st.number_input("Random state", value=int(DEFAULT_RANDOM_STATE), step=1)

    problem_type = determine_problem_type(dataset[target_col])
    model_options = [
        ("Decision Tree", "decision_tree"),
        ("Random Forest", "random_forest"),
        ("Extra Trees", "extra_trees"),
        ("Gradient Boosting", "gradient_boosting"),
        ("HistGradientBoosting", "hist_gradient_boosting"),
        ("AdaBoost", "adaboost"),
    ]
    model_label = st.selectbox("Model type", options=[label for label, _ in model_options])
    model_name = dict(model_options)[model_label]

    st.markdown("#### Tree Model Configuration")
    if model_name in {"decision_tree", "random_forest", "extra_trees"}:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            criterion_options = ["squared_error", "friedman_mse", "absolute_error", "poisson"] if problem_type == "regression" else ["gini", "entropy", "log_loss"]
            criterion = st.selectbox("Criterion", options=criterion_options)
        with col2:
            max_depth = st.number_input("Max depth", min_value=1, max_value=100, value=int(DEFAULT_MAX_DEPTH), step=1)
        with col3:
            min_samples_split = st.number_input("Min samples split", min_value=2, max_value=100, value=int(DEFAULT_MIN_SAMPLES_SPLIT), step=1)
        with col4:
            min_samples_leaf = st.number_input("Min samples leaf", min_value=1, max_value=100, value=int(DEFAULT_MIN_SAMPLES_LEAF), step=1)

        if model_name in {"random_forest", "extra_trees"}:
            n_estimators = st.number_input("Number of trees", min_value=10, max_value=500, value=100, step=10)
        else:
            n_estimators = None
        learning_rate = None
        subsample = None
        max_iter = None
        max_leaf_nodes = None
        l2_regularization = None
        colsample_bytree = None
        max_bin = None
        num_leaves = None
    elif model_name == "gradient_boosting":
        st.caption("Gradient Boosting uses shallow decision trees as weak learners.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            loss_options = ["squared_error", "absolute_error", "huber", "quantile"] if problem_type == "regression" else ["log_loss", "exponential"]
            loss = st.selectbox("Loss", options=loss_options)
        with col2:
            learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        with col3:
            n_estimators = st.number_input("Number of boosting stages", min_value=10, max_value=500, value=100, step=10)
        with col4:
            subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f")
        col5, col6 = st.columns(2)
        with col5:
            max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=3, step=1)
        with col6:
            min_samples_split = st.number_input("Min samples split", min_value=2, max_value=100, value=int(DEFAULT_MIN_SAMPLES_SPLIT), step=1)
        min_samples_leaf = st.number_input("Min samples leaf", min_value=1, max_value=100, value=int(DEFAULT_MIN_SAMPLES_LEAF), step=1)
        max_iter = None
        max_leaf_nodes = None
        l2_regularization = None
        criterion = loss
    elif model_name == "hist_gradient_boosting":
        st.caption("HistGradientBoosting is optimized for larger tabular datasets and does not expose a plottable tree structure.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        with col2:
            max_iter = st.number_input("Max iterations", min_value=10, max_value=1000, value=100, step=10)
        with col3:
            max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=3, step=1)
        with col4:
            min_samples_leaf = st.number_input("Min samples leaf", min_value=1, max_value=100, value=20, step=1)
        col5, col6 = st.columns(2)
        with col5:
            max_leaf_nodes = st.number_input("Max leaf nodes", min_value=2, max_value=255, value=31, step=1)
        with col6:
            l2_regularization = st.number_input("L2 regularization", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.3f")
        criterion = None
        min_samples_split = None
        subsample = None
        n_estimators = None
        colsample_bytree = None
        max_bin = None
        num_leaves = None
    else:
        st.caption("AdaBoost combines many weak learners into a stronger ensemble.")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.number_input("Number of estimators", min_value=10, max_value=500, value=50, step=10)
        with col2:
            learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=1.0, step=0.05, format="%.3f")
        criterion = None
        max_depth = None
        min_samples_split = None
        min_samples_leaf = None
        subsample = None
        max_iter = None
        max_leaf_nodes = None
        l2_regularization = None
        colsample_bytree = None
        max_bin = None
        num_leaves = None
        loss = None

    if model_name in {"xgboost", "lightgbm", "catboost"}:
        st.caption("These libraries can improve accuracy on tabular data but require extra dependencies.")
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("Number of trees", min_value=10, max_value=1000, value=200, step=10)
        with col2:
            learning_rate = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
        with col3:
            max_depth = st.number_input("Max depth", min_value=1, max_value=20, value=6, step=1)

        if model_name == "xgboost":
            col4, col5 = st.columns(2)
            with col4:
                subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f")
            with col5:
                colsample_bytree = st.number_input("Column sample by tree", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f")
            criterion = None
            min_samples_split = None
            min_samples_leaf = None
            max_iter = None
            max_leaf_nodes = None
            l2_regularization = None
            num_leaves = None
            max_bin = None
        elif model_name == "lightgbm":
            col4, col5 = st.columns(2)
            with col4:
                subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f")
            with col5:
                colsample_bytree = st.number_input("Feature fraction", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f")
            num_leaves = st.number_input("Num leaves", min_value=2, max_value=255, value=31, step=1)
            l2_regularization = st.number_input("L2 regularization", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.3f")
            criterion = None
            min_samples_split = None
            min_samples_leaf = None
            max_iter = None
            max_leaf_nodes = None
            max_bin = None
        else:
            col4, col5 = st.columns(2)
            with col4:
                subsample = st.number_input("RSM", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f")
            with col5:
                max_bin = st.number_input("Max bin", min_value=32, max_value=255, value=254, step=1)
            criterion = None
            min_samples_split = None
            min_samples_leaf = None
            max_iter = None
            max_leaf_nodes = None
            l2_regularization = None
            colsample_bytree = None
            num_leaves = None

    class_weight = None
    if model_name in {"decision_tree", "random_forest", "extra_trees"} and (pd.api.types.is_object_dtype(dataset[target_col]) or pd.api.types.is_categorical_dtype(dataset[target_col])):
        class_weight = st.selectbox("Class weight", options=[None, "balanced"], format_func=lambda value: "None" if value is None else value)

    train_trigger = st.button("Train Model", type="primary")

    if train_trigger:
        try:
            features = dataset[selected_features].copy()
            target = dataset[target_col].copy()
            if features.empty:
                raise ValueError("No usable feature columns were selected.")
            if target.isna().all():
                raise ValueError("The target column contains only missing values.")

            stratify = target if problem_type == "classification" and target.nunique(dropna=True) > 1 else None
            effective_test_size = 1 - train_size
            if stratify is not None:
                minimum_test_size = target.nunique(dropna=True) / len(target)
                if effective_test_size < minimum_test_size:
                    effective_test_size = minimum_test_size
                    st.info(
                        "The selected test split was too small for stratified classification, so it was increased automatically."
                    )
            X_train, X_test, y_train, y_test = train_test_split(
                features,
                target,
                test_size=effective_test_size,
                random_state=int(random_state),
                stratify=stratify,
            )

            tree_params: dict[str, Any] = {"random_state": int(random_state)}
            if model_name in {"decision_tree", "random_forest", "extra_trees"}:
                tree_params.update(
                    {
                        "criterion": criterion,
                        "max_depth": int(max_depth),
                        "min_samples_split": int(min_samples_split),
                        "min_samples_leaf": int(min_samples_leaf),
                    }
                )
                if n_estimators is not None:
                    tree_params["n_estimators"] = int(n_estimators)
                if class_weight is not None and problem_type == "classification":
                    tree_params["class_weight"] = class_weight
            elif model_name == "gradient_boosting":
                tree_params.update(
                    {
                        "loss": criterion,
                        "learning_rate": float(learning_rate),
                        "n_estimators": int(n_estimators),
                        "subsample": float(subsample),
                        "max_depth": int(max_depth),
                        "min_samples_split": int(min_samples_split),
                        "min_samples_leaf": int(min_samples_leaf),
                    }
                )
            elif model_name == "hist_gradient_boosting":
                tree_params.update(
                    {
                        "learning_rate": float(learning_rate),
                        "max_iter": int(max_iter),
                        "max_depth": int(max_depth),
                        "min_samples_leaf": int(min_samples_leaf),
                        "max_leaf_nodes": int(max_leaf_nodes),
                        "l2_regularization": float(l2_regularization),
                    }
                )
            elif model_name == "adaboost":
                tree_params.update(
                    {
                        "n_estimators": int(n_estimators),
                        "learning_rate": float(learning_rate),
                    }
                )
            elif model_name == "xgboost":
                tree_params.update(
                    {
                        "n_estimators": int(n_estimators),
                        "learning_rate": float(learning_rate),
                        "max_depth": int(max_depth),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample_bytree),
                        "tree_method": "hist",
                        "eval_metric": "logloss" if problem_type == "classification" else "rmse",
                        "verbosity": 0,
                    }
                )
            elif model_name == "lightgbm":
                tree_params.update(
                    {
                        "n_estimators": int(n_estimators),
                        "learning_rate": float(learning_rate),
                        "max_depth": int(max_depth),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample_bytree),
                        "num_leaves": int(num_leaves),
                        "reg_lambda": float(l2_regularization),
                        "verbosity": -1,
                    }
                )
            elif model_name == "catboost":
                tree_params.update(
                    {
                        "iterations": int(n_estimators),
                        "learning_rate": float(learning_rate),
                        "depth": int(max_depth),
                        "random_seed": int(random_state),
                        "verbose": False,
                    }
                )

            with st.spinner(f"Training the {model_label.lower()}..."):
                pipeline, feature_names, class_names, target_encoder = build_training_pipeline(
                    X_train,
                    y_train,
                    selected_features,
                    target_col,
                    problem_type,
                    model_name,
                    tree_params,
                )

            train_predictions = pipeline.predict(X_train)
            test_predictions = pipeline.predict(X_test)
            if target_encoder is not None:
                y_train_eval = target_encoder.transform(y_train.astype(str))
                y_test_eval = target_encoder.transform(y_test.astype(str))
            else:
                y_train_eval = y_train
                y_test_eval = y_test
            train_metrics = evaluate_predictions(y_train_eval, train_predictions, problem_type)
            test_metrics = evaluate_predictions(y_test_eval, test_predictions, problem_type)

            st.session_state.metrics = {"train": train_metrics, "test": test_metrics}
            st.session_state.train_metrics = train_metrics
            st.session_state.test_metrics = test_metrics
            st.session_state.artifacts = ModelArtifacts(
                pipeline=pipeline,
                preprocessor=pipeline.named_steps["preprocessor"],
                feature_names=feature_names,
                problem_type=problem_type,
                target_name=target_col,
                feature_columns=selected_features,
                categorical_columns=detect_column_types(features)[1],
                numeric_columns=detect_column_types(features)[0],
                classes_=class_names,
                target_encoder=target_encoder,
            )
            st.success("Model trained successfully.")
            st.session_state.training_signature = build_signature(
                {
                    "dataset": st.session_state.dataset_signature,
                    "features": selected_features,
                    "target": target_col,
                    "model": model_name,
                    "params": tree_params,
                    "train_size": train_size,
                }
            )
        except Exception as exc:
            LOGGER.exception("Model training failed")
            st.error(str(exc))
            return

    artifacts = st.session_state.artifacts
    if artifacts is not None and st.session_state.metrics is not None:
        render_evaluation(st.session_state.test_metrics, artifacts.problem_type)
        if artifacts.problem_type == "classification":
            render_confusion_matrix(st.session_state.test_metrics)

        render_feature_importance(artifacts.pipeline.named_steps["model"], artifacts.feature_names)

        st.markdown("### Tree Visualization")
        tree_depth = get_tree_depth(artifacts.pipeline.named_steps["model"])
        if tree_depth <= 0:
            st.info("Tree visualization is not available for this model.")
        else:
            if tree_depth <= 1:
                st.caption("The trained tree is shallow, so visualization depth is fixed at 1.")
                depth_limit = 1
            else:
                depth_limit = st.slider(
                    "Visualization depth limit",
                    min_value=1,
                    max_value=tree_depth,
                    value=min(4, tree_depth),
                )
            try:
                fig = plot_tree_figure(
                    artifacts.pipeline.named_steps["model"],
                    artifacts.feature_names,
                    artifacts.classes_,
                    max_depth=depth_limit,
                )
            except ValueError as exc:
                st.info(str(exc))
            else:
                if hasattr(artifacts.pipeline.named_steps["model"], "estimators_"):
                    st.caption("The visualization shows the first tree in the ensemble.")
                image_bytes = None
                try:
                    image_bytes = figure_to_png_bytes(fig)
                except Exception:
                    image_bytes = None

                if image_bytes is not None:
                    st.image(image_bytes, use_container_width=True)
                else:
                    st.pyplot(fig, clear_figure=False)

                if image_bytes is not None:
                    st.download_button(
                        "Download tree visualization",
                        data=image_bytes,
                        file_name="decision_tree.png",
                        mime="image/png",
                    )

        render_prediction_form(artifacts, dataset)

    st.markdown("### About")
    st.write(
        "This app uses a scikit-learn tree-based pipeline with built-in preprocessing, train/test splitting, evaluation, and per-session prediction support."
    )


if __name__ == "__main__":
    main()
