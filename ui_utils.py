from __future__ import annotations

import logging
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

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
    detect_supported_columns,
    build_signature,
    file_hash,
    load_dataset,
    summarize_class_distribution,
    summarize_dataset,
    validate_file_size,
)
from model_utils import (
    ModelArtifacts,
    decode_predictions,
    determine_problem_type,
    evaluate_predictions,
    get_estimator_count,
    get_model_meta,
    IMBLEARN_AVAILABLE,
    predict_classes_from_proba,
    supports_class_weight,
)
from preprocessing import detect_column_types
from report_utils import (
    build_confusion_matrix_figure,
    build_dataset_profile_report,
    build_feature_importance_figure,
    build_json_download,
    build_model_bundle,
    build_result_pdf,
    build_tree_visualization_figure,
    file_name_safe,
)
from tree_utils import (
    build_tree_viewer_html,
    figure_to_png_bytes,
    figure_to_svg_markup,
    get_tree_depth,
    get_tree_estimator_count,
    plot_tree_figure,
)


LOGGER = logging.getLogger(__name__)


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
        "nav_page": "dashboard",
        "trained_model_name": None,
        "last_train_meta": None,
        "target_distribution": None,
        "classification_eval": None,
        "decision_threshold": 0.5,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_to(page: str) -> None:
    st.session_state.nav_page = page


def render_empty_state(
    title: str,
    message: str,
    action_label: str | None = None,
    action_page: str | None = None,
) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-state-title">{title}</div>
            <div class="empty-state-message">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if action_label and action_page:
        st.button(action_label, on_click=go_to, args=(action_page,), type="primary")


def render_error(message: str, exc: Exception | None = None) -> None:
    st.markdown(f'<div class="status-error">{message}</div>', unsafe_allow_html=True)
    if exc is not None:
        with st.expander("Technical details"):
            st.code(str(exc))


def render_dataset_overview(summary: dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", summary["rows"])
    col2.metric("Columns", summary["columns"])
    col3.metric("Duplicates", summary["duplicate_rows"])
    col4.metric("Missing cells", int(summary["missing_values"].sum()))


def render_target_distribution(target_distribution: dict[str, Any]) -> None:
    if not target_distribution:
        return

    st.markdown("#### Target Class Distribution")
    distribution = pd.DataFrame(target_distribution.get("distribution", []))
    if distribution.empty:
        st.info("No target distribution is available for the selected target column.")
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Minority share", f"{target_distribution.get('minority_share', 0.0):.1f}%")
    metric_columns[1].metric("Majority share", f"{target_distribution.get('majority_share', 0.0):.1f}%")
    metric_columns[2].metric("Imbalance ratio", target_distribution.get("imbalance_ratio", "-"))
    metric_columns[3].metric("Flagged", "Yes" if target_distribution.get("is_imbalanced") else "No")

    if target_distribution.get("is_imbalanced"):
        st.warning("The target is imbalanced. Consider class weights, resampling, or threshold tuning before training.")

    st.dataframe(distribution, use_container_width=True)


def resampling_option_labels() -> dict[str, str]:
    return {
        "none": "None",
        "random_oversample": "Random oversampling",
        "random_undersample": "Random undersampling",
        "smote": "SMOTE (imbalanced-learn)",
    }


def render_dataset_downloads(summary: dict[str, Any], dataset_name: str, unsupported_cols: list[str] | None = None) -> None:
    report = build_dataset_profile_report(summary, dataset_name, unsupported_cols=unsupported_cols)
    st.download_button(
        "Download dataset profile report",
        data=build_json_download(report),
        file_name=f"{file_name_safe(dataset_name)}_profile_report.json",
        mime="application/json",
        use_container_width=True,
        key=f"download_profile_{file_name_safe(dataset_name)}",
    )


def render_model_downloads(
    artifacts: ModelArtifacts,
    model_name: str,
    dataset_name: str,
    train_meta: dict[str, Any],
) -> None:
    st.download_button(
        "Download trained model bundle",
        data=build_model_bundle(artifacts, model_name, dataset_name, train_meta),
        file_name=f"{file_name_safe(dataset_name)}_{model_name}_bundle.pkl",
        mime="application/octet-stream",
        use_container_width=True,
        key=f"download_bundle_{model_name}",
    )

    profile_payload = build_dataset_profile_report(
        st.session_state.summary,
        dataset_name,
        target_col=artifacts.target_name,
        selected_features=artifacts.feature_columns,
        target_distribution=st.session_state.get("target_distribution"),
    )
    st.download_button(
        "Download training profile report",
        data=build_json_download(profile_payload),
        file_name=f"{file_name_safe(dataset_name)}_{model_name}_training_report.json",
        mime="application/json",
        use_container_width=True,
        key=f"download_training_report_{model_name}",
    )


def render_result_downloads(
    artifacts: ModelArtifacts,
    model_name: str,
    dataset_name: str,
    train_meta: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    st.download_button(
        "Download PDF result report",
        data=build_result_pdf(dataset_name, model_name, artifacts, train_meta, metrics),
        file_name=f"{file_name_safe(dataset_name)}_{model_name}_result_report.pdf",
        mime="application/pdf",
        use_container_width=True,
        key=f"download_result_pdf_{model_name}",
    )


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


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-block">
                <div class="brand-title">{APP_NAME}</div>
                <div class="brand-tagline">{APP_TAGLINE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def nav_button(label: str, page_key: str) -> None:
            is_active = st.session_state.nav_page == page_key
            st.button(
                label,
                key=f"nav_{page_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=go_to,
                args=(page_key,),
            )

        nav_button("🏠  Dashboard", "dashboard")

        st.markdown('<div class="nav-section-label">Models</div>', unsafe_allow_html=True)
        for entry in st.session_state.get("model_catalog", []):
            nav_button(f"{entry['icon']}  {entry['label']}", entry["name"])

        st.markdown('<div class="nav-section-label">Data</div>', unsafe_allow_html=True)
        nav_button("📊  Dataset", "dataset")

        st.divider()
        if st.session_state.dataset is not None:
            st.caption(f"Active dataset: **{st.session_state.dataset_name}**")
            st.caption(f"{st.session_state.summary['rows']:,} rows · {st.session_state.summary['columns']} columns")
        else:
            st.caption("No dataset uploaded yet.")


def render_dashboard() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin:0;font-size:2.1rem;">Welcome to {APP_NAME}</h1>
            <p style="margin:0.5rem 0 0;color:#475569;font-size:1.05rem;">
                Build and visualize machine learning models. Upload a dataset, configure an algorithm,
                and explore trained trees and ensembles interactively.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("📊  Upload Dataset", use_container_width=True, on_click=go_to, args=("dataset",))
    with col2:
        default_model = st.session_state.trained_model_name or st.session_state.model_catalog[0]["name"]
        st.button(
            "⚙️  Create Model",
            use_container_width=True,
            type="primary",
            on_click=go_to,
            args=(default_model,))

    if st.session_state.dataset is not None:
        st.markdown("### Current Dataset")
        render_dataset_overview(st.session_state.summary)

    st.markdown("### Available Models")
    columns = st.columns(3)
    for index, entry in enumerate(st.session_state.model_catalog):
        with columns[index % 3]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-card-icon">{entry['icon']}</div>
                        <div class="model-card-title">{entry['label']}</div>
                        <div class="model-card-desc">{entry['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Configure",
                    key=f"card_{entry['name']}",
                    use_container_width=True,
                    on_click=go_to,
                    args=(entry["name"],),
                )


def render_feature_importance(model, feature_names: list[str]) -> None:
    figure = build_feature_importance_figure(model, feature_names)
    if figure is None:
        return

    st.markdown("### Feature Importance")
    st.pyplot(figure, clear_figure=False)
    plt.close(figure)


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
    if artifacts.problem_type == "classification" and hasattr(artifacts.pipeline.named_steps["model"], "predict_proba"):
        probabilities = artifacts.pipeline.predict_proba(input_frame)[0]
        threshold = float(st.session_state.get("decision_threshold", 0.5))
        prediction_index = predict_classes_from_proba([probabilities], threshold=threshold)[0]
        prediction = decode_predictions([prediction_index], artifacts.target_encoder)[0]
    else:
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
        col2.metric("Weighted F1", f"{metrics['F1 Score']:.3f}")
        col3.metric("Macro F1", f"{metrics.get('Macro F1', 0.0):.3f}")
        col4.metric("ROC-AUC", "-" if metrics.get("ROC AUC") is None else f"{metrics['ROC AUC']:.3f}")

        secondary = st.columns(3)
        secondary[0].metric("Weighted precision", f"{metrics['Precision']:.3f}")
        secondary[1].metric("Weighted recall", f"{metrics['Recall']:.3f}")
        secondary[2].metric("PR-AUC", "-" if metrics.get("PR AUC") is None else f"{metrics['PR AUC']:.3f}")

        for warning in metrics.get("Warnings", []):
            st.warning(warning)

        st.dataframe(pd.DataFrame(metrics["Classification Report"]).transpose(), use_container_width=True)
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics['MAE']:.3f}")
        col2.metric("MSE", f"{metrics['MSE']:.3f}")
        col3.metric("RMSE", f"{metrics['RMSE']:.3f}")
        col4.metric("R²", f"{metrics['R2']:.3f}")


def render_precision_recall_curve(y_true: pd.Series, y_proba) -> None:
    if y_proba is None:
        return

    probabilities = pd.DataFrame(y_proba)
    if probabilities.shape[1] != 2:
        return

    precision, recall, _ = precision_recall_curve(y_true, probabilities.iloc[:, 1])
    curve_frame = pd.DataFrame({"Recall": recall[:-1], "Precision": precision[:-1]}).set_index("Recall")
    st.markdown("### Precision-Recall Curve")
    st.line_chart(curve_frame)


def render_confusion_matrix(metrics: dict[str, Any]) -> None:
    figure = build_confusion_matrix_figure(metrics)
    if figure is None:
        return

    st.pyplot(figure, clear_figure=True)
    plt.close(figure)


def render_model_info_cards(
    artifacts: ModelArtifacts,
    model_name: str,
    tree_params: dict[str, Any],
    n_samples: int,
) -> None:
    model_obj = artifacts.pipeline.named_steps["model"]
    meta = get_model_meta(model_name)
    estimator_count = get_estimator_count(model_obj)
    max_depth_value = tree_params.get("max_depth")
    if max_depth_value is None:
        max_depth_value = get_tree_depth(model_obj) or None

    cards: list[tuple[str, str]] = [
        ("Model Type", meta["label"]),
        ("Problem Type", artifacts.problem_type.capitalize()),
    ]
    if estimator_count:
        cards.append(("Estimators", f"{estimator_count:,}"))
    if max_depth_value:
        cards.append(("Max Depth", str(max_depth_value)))
    cards.append(("Features", str(len(artifacts.feature_columns))))
    cards.append(("Samples", f"{n_samples:,}"))
    cards.append(("Status", "Trained"))

    st.markdown("### Model Information")
    columns = st.columns(len(cards))
    for column, (label, value) in zip(columns, cards):
        with column:
            st.markdown(
                f"""<div class="metric-card"><div class="metric-card-label">{label}</div>
                <div class="metric-card-value">{value}</div></div>""",
                unsafe_allow_html=True,
            )


def render_model_config(model_name: str, problem_type: str, dataset: pd.DataFrame, target_col: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "criterion": None,
        "max_depth": None,
        "min_samples_split": None,
        "min_samples_leaf": None,
        "n_estimators": None,
        "learning_rate": None,
        "subsample": None,
        "max_iter": None,
        "max_leaf_nodes": None,
        "l2_regularization": None,
        "colsample_bytree": None,
        "max_bin": None,
        "num_leaves": None,
        "class_weight": None,
        "use_class_weight": False,
        "resampling_strategy": "none",
    }
    key_prefix = model_name

    if problem_type == "classification":
        with st.expander("Imbalance handling tips", expanded=False):
            st.markdown(
                """
- Use **balanced class weights** first when the minority class is under-represented but still has enough examples.
- Try **resampling** when the class gap is large or the model keeps predicting the majority class.
- Adjust the **decision threshold** after training if recall and precision need to be traded off for the positive class.
                """.strip()
            )

    if model_name in {"decision_tree", "random_forest", "extra_trees"}:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            criterion_options = ["squared_error", "friedman_mse", "absolute_error", "poisson"] if problem_type == "regression" else ["gini", "entropy", "log_loss"]
            values["criterion"] = st.selectbox("Criterion", options=criterion_options, key=f"criterion_{key_prefix}")
        with col2:
            values["max_depth"] = st.number_input("Max depth", min_value=1, max_value=100, value=int(DEFAULT_MAX_DEPTH), step=1, key=f"maxdepth_{key_prefix}")
        with col3:
            values["min_samples_split"] = st.number_input("Min samples split", min_value=2, max_value=100, value=int(DEFAULT_MIN_SAMPLES_SPLIT), step=1, key=f"minsplit_{key_prefix}")
        with col4:
            values["min_samples_leaf"] = st.number_input("Min samples leaf", min_value=1, max_value=100, value=int(DEFAULT_MIN_SAMPLES_LEAF), step=1, key=f"minleaf_{key_prefix}")

        if model_name in {"random_forest", "extra_trees"}:
            values["n_estimators"] = st.number_input("Number of trees", min_value=10, max_value=500, value=100, step=10, key=f"ntrees_{key_prefix}")

        if problem_type == "classification" and supports_class_weight(model_name):
            values["use_class_weight"] = st.checkbox(
                "Use balanced class weights",
                value=True,
                key=f"balanced_class_weight_{key_prefix}",
            )

        if problem_type == "classification":
            values["resampling_strategy"] = st.selectbox(
                "Optional resampling",
                options=["none", "random_oversample", "random_undersample", "smote"],
                format_func=lambda value: resampling_option_labels()[value],
                key=f"resample_{key_prefix}",
            )
            if values["resampling_strategy"] == "smote" and not IMBLEARN_AVAILABLE:
                st.warning("SMOTE is unavailable in this environment because imbalanced-learn is not installed.")

    elif model_name == "gradient_boosting":
        st.caption("Gradient Boosting uses shallow decision trees as weak learners.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            loss_options = ["squared_error", "absolute_error", "huber", "quantile"] if problem_type == "regression" else ["log_loss", "exponential"]
            values["criterion"] = st.selectbox("Loss", options=loss_options, key=f"loss_{key_prefix}")
        with col2:
            values["learning_rate"] = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", key=f"lr_{key_prefix}")
        with col3:
            values["n_estimators"] = st.number_input("Number of boosting stages", min_value=10, max_value=500, value=100, step=10, key=f"nstages_{key_prefix}")
        with col4:
            values["subsample"] = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key=f"subsample_{key_prefix}")
        col5, col6 = st.columns(2)
        with col5:
            values["max_depth"] = st.number_input("Max depth", min_value=1, max_value=20, value=3, step=1, key=f"maxdepth_{key_prefix}")
        with col6:
            values["min_samples_split"] = st.number_input("Min samples split", min_value=2, max_value=100, value=int(DEFAULT_MIN_SAMPLES_SPLIT), step=1, key=f"minsplit_{key_prefix}")
        values["min_samples_leaf"] = st.number_input("Min samples leaf", min_value=1, max_value=100, value=int(DEFAULT_MIN_SAMPLES_LEAF), step=1, key=f"minleaf_{key_prefix}")

    elif model_name == "hist_gradient_boosting":
        st.caption("HistGradientBoosting is optimized for larger tabular datasets and does not expose a plottable tree structure.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            values["learning_rate"] = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", key=f"lr_{key_prefix}")
        with col2:
            values["max_iter"] = st.number_input("Max iterations", min_value=10, max_value=1000, value=100, step=10, key=f"maxiter_{key_prefix}")
        with col3:
            values["max_depth"] = st.number_input("Max depth", min_value=1, max_value=20, value=3, step=1, key=f"maxdepth_{key_prefix}")
        with col4:
            values["min_samples_leaf"] = st.number_input("Min samples leaf", min_value=1, max_value=100, value=20, step=1, key=f"minleaf_{key_prefix}")
        col5, col6 = st.columns(2)
        with col5:
            values["max_leaf_nodes"] = st.number_input("Max leaf nodes", min_value=2, max_value=255, value=31, step=1, key=f"maxleaf_{key_prefix}")
        with col6:
            values["l2_regularization"] = st.number_input("L2 regularization", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.3f", key=f"l2_{key_prefix}")

    elif model_name == "adaboost":
        st.caption("AdaBoost combines many weak learners into a stronger ensemble.")
        col1, col2 = st.columns(2)
        with col1:
            values["n_estimators"] = st.number_input("Number of estimators", min_value=10, max_value=500, value=50, step=10, key=f"nestimators_{key_prefix}")
        with col2:
            values["learning_rate"] = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=1.0, step=0.05, format="%.3f", key=f"lr_{key_prefix}")

    elif model_name in {"xgboost", "lightgbm", "catboost"}:
        st.caption("These libraries can improve accuracy on tabular data but require extra dependencies.")
        col1, col2, col3 = st.columns(3)
        with col1:
            values["n_estimators"] = st.number_input("Number of trees", min_value=10, max_value=1000, value=200, step=10, key=f"ntrees_{key_prefix}")
        with col2:
            values["learning_rate"] = st.number_input("Learning rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f", key=f"lr_{key_prefix}")
        with col3:
            values["max_depth"] = st.number_input("Max depth", min_value=1, max_value=20, value=6, step=1, key=f"maxdepth_{key_prefix}")

        if model_name == "xgboost":
            col4, col5 = st.columns(2)
            with col4:
                values["subsample"] = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key=f"subsample_{key_prefix}")
            with col5:
                values["colsample_bytree"] = st.number_input("Column sample by tree", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key=f"colsample_{key_prefix}")
        elif model_name == "lightgbm":
            col4, col5 = st.columns(2)
            with col4:
                values["subsample"] = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key=f"subsample_{key_prefix}")
            with col5:
                values["colsample_bytree"] = st.number_input("Feature fraction", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key=f"colsample_{key_prefix}")
            values["num_leaves"] = st.number_input("Num leaves", min_value=2, max_value=255, value=31, step=1, key=f"numleaves_{key_prefix}")
            values["l2_regularization"] = st.number_input("L2 regularization", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.3f", key=f"l2_{key_prefix}")
        else:
            col4, col5 = st.columns(2)
            with col4:
                values["subsample"] = st.number_input("RSM", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key=f"rsm_{key_prefix}")
            with col5:
                values["max_bin"] = st.number_input("Max bin", min_value=32, max_value=255, value=254, step=1, key=f"maxbin_{key_prefix}")

    return values


def build_tree_params(model_name: str, problem_type: str, random_state: int, values: dict[str, Any]) -> dict[str, Any]:
    tree_params: dict[str, Any] = {"random_state": int(random_state)}

    if model_name in {"decision_tree", "random_forest", "extra_trees"}:
        tree_params.update(
            {
                "criterion": values["criterion"],
                "max_depth": int(values["max_depth"]),
                "min_samples_split": int(values["min_samples_split"]),
                "min_samples_leaf": int(values["min_samples_leaf"]),
            }
        )
        if values["n_estimators"] is not None:
            tree_params["n_estimators"] = int(values["n_estimators"])
        if problem_type == "classification" and values.get("use_class_weight"):
            tree_params["class_weight"] = "balanced"
    elif model_name == "gradient_boosting":
        tree_params.update(
            {
                "loss": values["criterion"],
                "learning_rate": float(values["learning_rate"]),
                "n_estimators": int(values["n_estimators"]),
                "subsample": float(values["subsample"]),
                "max_depth": int(values["max_depth"]),
                "min_samples_split": int(values["min_samples_split"]),
                "min_samples_leaf": int(values["min_samples_leaf"]),
            }
        )
    elif model_name == "hist_gradient_boosting":
        tree_params.update(
            {
                "learning_rate": float(values["learning_rate"]),
                "max_iter": int(values["max_iter"]),
                "max_depth": int(values["max_depth"]),
                "min_samples_leaf": int(values["min_samples_leaf"]),
                "max_leaf_nodes": int(values["max_leaf_nodes"]),
                "l2_regularization": float(values["l2_regularization"]),
            }
        )
    elif model_name == "adaboost":
        tree_params.update(
            {
                "n_estimators": int(values["n_estimators"]),
                "learning_rate": float(values["learning_rate"]),
            }
        )
    elif model_name == "xgboost":
        tree_params.update(
            {
                "n_estimators": int(values["n_estimators"]),
                "learning_rate": float(values["learning_rate"]),
                "max_depth": int(values["max_depth"]),
                "subsample": float(values["subsample"]),
                "colsample_bytree": float(values["colsample_bytree"]),
                "tree_method": "hist",
                "eval_metric": "logloss" if problem_type == "classification" else "rmse",
                "verbosity": 0,
            }
        )
    elif model_name == "lightgbm":
        tree_params.update(
            {
                "n_estimators": int(values["n_estimators"]),
                "learning_rate": float(values["learning_rate"]),
                "max_depth": int(values["max_depth"]),
                "subsample": float(values["subsample"]),
                "colsample_bytree": float(values["colsample_bytree"]),
                "num_leaves": int(values["num_leaves"]),
                "reg_lambda": float(values["l2_regularization"]),
                "verbosity": -1,
            }
        )
    elif model_name == "catboost":
        tree_params.update(
            {
                "iterations": int(values["n_estimators"]),
                "learning_rate": float(values["learning_rate"]),
                "depth": int(values["max_depth"]),
                "random_seed": int(random_state),
                "verbose": False,
            }
        )

    return tree_params


def render_tree_visualization_section(artifacts: ModelArtifacts, model_name: str) -> None:
    st.markdown("### Tree Visualization")
    model_obj = artifacts.pipeline.named_steps["model"]
    estimator_count = get_tree_estimator_count(model_obj)
    if estimator_count <= 0:
        st.info("Tree visualization is not available for this model type.")
        return

    estimator_index = 0
    if estimator_count > 1:
        max_selectable = min(estimator_count, 50)
        tree_choice = st.selectbox(
            f"Select a tree to inspect (1-{max_selectable} of {estimator_count})",
            options=list(range(1, max_selectable + 1)),
            key=f"tree_index_{model_name}",
        )
        estimator_index = tree_choice - 1
        if estimator_count > 50:
            st.caption(f"Showing the first 50 of {estimator_count:,} trees to keep the app responsive.")

    tree_depth = get_tree_depth(model_obj, estimator_index)
    if tree_depth <= 0:
        st.info("Tree visualization is not available for this model.")
        return

    if tree_depth <= 1:
        st.caption("The trained tree is shallow, so visualization depth is fixed at 1.")
        depth_limit = 1
    else:
        depth_limit = st.slider(
            "Visualization depth limit",
            min_value=1,
            max_value=tree_depth,
            value=min(4, tree_depth),
            key=f"depth_{model_name}",
        )

    try:
        fig = plot_tree_figure(
            model_obj,
            artifacts.feature_names,
            artifacts.classes_,
            max_depth=depth_limit,
            estimator_index=estimator_index,
        )
    except ValueError as exc:
        st.info(str(exc))
        return

    if estimator_count > 1:
        st.caption(f"Showing tree {estimator_index + 1} of {estimator_count}.")

    image_bytes = None
    try:
        image_bytes = figure_to_png_bytes(fig)
        svg_markup = figure_to_svg_markup(fig)
        viewer_id = f"{model_name}-{estimator_index}-{depth_limit}"
        components.html(build_tree_viewer_html(svg_markup, viewer_id, height=620), height=650, scrolling=False)
    except Exception:
        LOGGER.exception("Falling back to static tree rendering")
        st.pyplot(fig, clear_figure=False)
    finally:
        plt.close(fig)

    if image_bytes is not None:
        st.download_button(
            "Download tree visualization",
            data=image_bytes,
            file_name="decision_tree.png",
            mime="image/png",
            key=f"download_{model_name}",
        )


def render_dataset_page() -> None:
    st.markdown("## Dataset")
    st.caption("Upload a CSV or Excel file to make it available to every model.")

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "Drag and drop your dataset here, or click to browse",
            type=["csv", "xlsx", "xls"],
            key="dataset_uploader",
        )
        st.caption("Files are validated locally and never executed. Supported formats: CSV, XLS, XLSX.")

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
                st.session_state.trained_model_name = None
                st.session_state.last_train_meta = None
        except Exception as exc:
            LOGGER.exception("Dataset upload failed")
            render_error("Unable to load this dataset.", exc)
            return

    dataset = st.session_state.dataset
    if dataset is None:
        render_empty_state(
            "No dataset uploaded yet",
            "Upload a CSV or Excel file above to explore it and start training models.",
        )
        return

    summary = st.session_state.summary
    st.markdown("### Dataset Overview")
    render_dataset_overview(summary)
    render_dataset_downloads(summary, st.session_state.dataset_name or "dataset")
    st.markdown("### Dataset Explorer")
    render_summary_tables(summary)


def render_dashboard() -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin:0;font-size:2.1rem;">Welcome to {APP_NAME}</h1>
            <p style="margin:0.5rem 0 0;color:#475569;font-size:1.05rem;">
                Build and visualize machine learning models. Upload a dataset, configure an algorithm,
                and explore trained trees and ensembles interactively.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.button("📊  Upload Dataset", use_container_width=True, on_click=go_to, args=("dataset",))
    with col2:
        default_model = st.session_state.trained_model_name or st.session_state.model_catalog[0]["name"]
        st.button(
            "⚙️  Create Model",
            use_container_width=True,
            type="primary",
            on_click=go_to,
            args=(default_model,),
        )

    if st.session_state.dataset is not None:
        st.markdown("### Current Dataset")
        render_dataset_overview(st.session_state.summary)

    st.markdown("### Available Models")
    columns = st.columns(3)
    for index, entry in enumerate(st.session_state.model_catalog):
        with columns[index % 3]:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="model-card-icon">{entry['icon']}</div>
                        <div class="model-card-title">{entry['label']}</div>
                        <div class="model-card-desc">{entry['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.button(
                    "Configure",
                    key=f"card_{entry['name']}",
                    use_container_width=True,
                    on_click=go_to,
                    args=(entry["name"],),
                )


def render_model_page(
    model_name: str,
    build_training_pipeline: Callable[..., Any],
    build_tree_params: Callable[..., dict[str, Any]],
) -> None:
    meta = get_model_meta(model_name)
    st.markdown(f"## {meta['icon']}  {meta['label']}")
    st.caption(meta["description"])

    dataset = st.session_state.dataset
    if dataset is None:
        render_empty_state(
            "No dataset uploaded yet",
            "Upload a dataset first, then come back to configure and train this model.",
            action_label="Upload Dataset",
            action_page="dataset",
        )
        return

    numeric_cols, categorical_cols, unsupported_cols = detect_supported_columns(dataset)
    if unsupported_cols:
        st.warning(f"Unsupported columns were detected and will be ignored automatically: {', '.join(unsupported_cols)}")

    with st.container(border=True):
        st.markdown("#### Data & Training Setup")
        target_col = st.selectbox(
            "Target column",
            options=list(dataset.columns),
            index=len(dataset.columns) - 1,
            key=f"target_{model_name}",
        )
        feature_candidates = [column for column in dataset.columns if column != target_col and column not in unsupported_cols]
        selected_features = st.multiselect(
            "Feature columns",
            options=feature_candidates,
            default=feature_candidates,
            key=f"features_{model_name}",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            train_size = 1 - st.slider(
                "Test split", min_value=0.1, max_value=0.5, value=float(DEFAULT_TEST_SIZE), step=0.05, key=f"split_{model_name}"
            )
        with col_b:
            random_state = st.number_input("Random state", value=int(DEFAULT_RANDOM_STATE), step=1, key=f"seed_{model_name}")

    if not selected_features:
        st.warning("Select at least one feature column.")
        return

    problem_type = determine_problem_type(dataset[target_col])

    target_distribution = summarize_class_distribution(dataset[target_col]) if problem_type == "classification" else None
    st.session_state.target_distribution = target_distribution
    if target_distribution is not None:
        with st.container(border=True):
            render_target_distribution(target_distribution)

    with st.container(border=True):
        st.markdown("#### Model Configuration")
        config_values = render_model_config(model_name, problem_type, dataset, target_col)

    train_trigger = st.button("Train Model", type="primary", key=f"train_{model_name}")

    if train_trigger:
        try:
            features = dataset[selected_features].copy()
            target = dataset[target_col].copy()
            if features.empty:
                raise ValueError("No usable feature columns were selected.")
            training_frame = features.join(target.rename(target_col)).dropna(subset=[target_col])
            dropped_target_rows = len(dataset) - len(training_frame)
            if dropped_target_rows > 0:
                st.warning(
                    f"Dropped {dropped_target_rows} row(s) with missing target values before training."
                )

            if training_frame.empty:
                raise ValueError("The target column contains only missing values.")

            features = training_frame[selected_features]
            target = training_frame[target_col]

            stratify = None
            if problem_type == "classification" and target.nunique(dropna=True) > 1:
                class_counts = target.value_counts(dropna=True)
                if class_counts.min() >= 2:
                    stratify = target
                else:
                    st.warning(
                        "Some classes have only one sample, so the train/test split will be unstratified."
                    )

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

            tree_params = build_tree_params(model_name, problem_type, random_state, config_values)

            status = st.status(f"Generating {meta['label']} model...", expanded=True)
            with status:
                status.write("Processing dataset...")
                pipeline, feature_names, class_names, target_encoder, resample_message = build_training_pipeline(
                    X_train,
                    y_train,
                    selected_features,
                    target_col,
                    problem_type,
                    model_name,
                    tree_params,
                    resampling_strategy=config_values.get("resampling_strategy", "none"),
                    random_state=int(random_state),
                )
                status.write("Training model...")
                if resample_message:
                    status.write(resample_message)

                train_predictions = pipeline.predict(X_train)
                test_predictions = pipeline.predict(X_test)
                train_probabilities = None
                test_probabilities = None
                if problem_type == "classification" and hasattr(pipeline.named_steps["model"], "predict_proba"):
                    train_probabilities = pipeline.predict_proba(X_train)
                    test_probabilities = pipeline.predict_proba(X_test)
                if target_encoder is not None:
                    y_train_eval = target_encoder.transform(y_train.astype(str))
                    y_test_eval = target_encoder.transform(y_test.astype(str))
                else:
                    y_train_eval = y_train
                    y_test_eval = y_test
                train_metrics = evaluate_predictions(
                    y_train_eval,
                    train_predictions,
                    problem_type,
                    y_proba=train_probabilities,
                    class_labels=list(class_names) if class_names is not None else None,
                )
                test_metrics = evaluate_predictions(
                    y_test_eval,
                    test_predictions,
                    problem_type,
                    y_proba=test_probabilities,
                    class_labels=list(class_names) if class_names is not None else None,
                )
                status.write("Building visualization...")
                status.update(label=f"{meta['label']} trained successfully.", state="complete")

            st.session_state.metrics = {"train": train_metrics, "test": test_metrics}
            st.session_state.train_metrics = train_metrics
            st.session_state.test_metrics = test_metrics
            if problem_type == "classification":
                st.session_state.classification_eval = {
                    "y_train_eval": y_train_eval,
                    "y_test_eval": y_test_eval,
                    "train_probabilities": train_probabilities,
                    "test_probabilities": test_probabilities,
                    "class_labels": list(class_names) if class_names is not None else None,
                    "y_test_raw": y_test,
                    "y_train_raw": y_train,
                }
            else:
                st.session_state.classification_eval = None
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
            st.session_state.trained_model_name = model_name
            st.session_state.last_train_meta = {
                "tree_params": tree_params,
                "n_samples": int(len(dataset)),
                "train_size": float(train_size),
                "test_size": float(effective_test_size),
            }
            st.session_state.training_signature = build_signature(
                {
                    "dataset": st.session_state.dataset_signature,
                    "features": selected_features,
                    "target": target_col,
                    "model": model_name,
                    "params": tree_params,
                    "train_size": train_size,
                    "resampling": config_values.get("resampling_strategy", "none"),
                    "class_weight": config_values.get("use_class_weight", False),
                }
            )
        except Exception as exc:
            LOGGER.exception("Model training failed")
            render_error("Unable to generate the model. Please check your dataset and model configuration.", exc)
            return

    artifacts = st.session_state.artifacts
    if artifacts is None or st.session_state.metrics is None or st.session_state.trained_model_name != model_name:
        render_empty_state(
            "No model generated yet",
            'Configure the parameters above and click "Train Model" to see results here.',
        )
        return

    train_meta = st.session_state.last_train_meta or {}
    display_metrics = st.session_state.test_metrics
    evaluation_state = st.session_state.get("classification_eval")
    if artifacts.problem_type == "classification" and evaluation_state and evaluation_state.get("test_probabilities") is not None:
        class_labels = evaluation_state.get("class_labels") or artifacts.classes_
        if class_labels and len(class_labels) == 2:
            threshold = st.slider(
                "Decision threshold",
                min_value=0.05,
                max_value=0.95,
                value=float(st.session_state.get("decision_threshold", 0.5)),
                step=0.01,
                key=f"decision_threshold_{model_name}",
            )
            st.session_state.decision_threshold = float(threshold)
            st.caption(f"Positive-class predictions are now made when probability >= {threshold:.2f}.")
            threshold_predictions = predict_classes_from_proba(evaluation_state["test_probabilities"], threshold=threshold)
            display_metrics = evaluate_predictions(
                evaluation_state["y_test_eval"],
                threshold_predictions,
                "classification",
                y_proba=evaluation_state["test_probabilities"],
                class_labels=class_labels,
                threshold=float(threshold),
            )
            st.session_state.test_metrics = display_metrics
        else:
            display_metrics = st.session_state.test_metrics

    render_model_info_cards(
        artifacts,
        model_name,
        train_meta.get("tree_params", {}),
        train_meta.get("n_samples", len(dataset)),
    )
    render_model_downloads(artifacts, model_name, st.session_state.dataset_name or "dataset", train_meta)
    render_evaluation(display_metrics, artifacts.problem_type)
    render_result_downloads(
        artifacts,
        model_name,
        st.session_state.dataset_name or "dataset",
        train_meta,
        display_metrics,
    )
    if artifacts.problem_type == "classification":
        render_confusion_matrix(display_metrics)
        if evaluation_state and evaluation_state.get("test_probabilities") is not None:
            render_precision_recall_curve(evaluation_state["y_test_eval"], evaluation_state["test_probabilities"])

    render_feature_importance(artifacts.pipeline.named_steps["model"], artifacts.feature_names)
    render_tree_visualization_section(artifacts, model_name)
    render_prediction_form(artifacts, dataset)
