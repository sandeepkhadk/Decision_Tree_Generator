from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from config import APP_NAME
import model_utils as model_utils_module
from model_utils import ModelArtifacts, create_model, encode_classification_target
from preprocessing import create_preprocessor, detect_column_types, get_feature_names
from ui_utils import (
    build_tree_params,
    init_state,
    render_dashboard,
    render_dataset_page,
    render_model_page,
    render_sidebar,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title=APP_NAME, page_icon="🌳", layout="wide")

MODEL_CATALOG = getattr(model_utils_module, "MODEL_CATALOG", [])

CSS = """
<style>
    .main { background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%); }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 100%; }

    .hero-card {
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 20px 50px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .brand-block { padding: 0.2rem 0 1rem; }
    .brand-title { font-size: 1.2rem; font-weight: 700; color: #0f172a; line-height: 1.25; }
    .brand-tagline { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }

    .nav-section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #94a3b8;
        text-transform: uppercase;
        margin: 0.9rem 0 0.35rem 0.1rem;
    }

    section[data-testid="stSidebar"] button {
        transition: background-color 0.15s ease, transform 0.1s ease;
        justify-content: flex-start;
    }
    section[data-testid="stSidebar"] button:hover { transform: translateX(1px); }

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        transition: box-shadow 0.15s ease;
    }
    .metric-card:hover { box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08); }
    .metric-card-label { font-size: 0.78rem; color: #64748b; font-weight: 600; }
    .metric-card-value { font-size: 1.3rem; color: #0f172a; font-weight: 700; margin-top: 0.2rem; word-break: break-word; }

    .model-card { padding: 0.2rem 0 0.6rem; }
    .model-card-icon { font-size: 1.6rem; }
    .model-card-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem; }
    .model-card-desc { font-size: 0.85rem; color: #64748b; margin-top: 0.3rem; min-height: 2.6rem; }

    .empty-state {
        background: white;
        border: 1px dashed rgba(100, 116, 139, 0.35);
        border-radius: 16px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        margin: 0.5rem 0 1rem;
    }
    .empty-state-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; }
    .empty-state-message { font-size: 0.9rem; color: #64748b; margin-top: 0.4rem; }

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

    .responsive-scroll {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }

    table { width: 100%; }

    @media (max-width: 768px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; padding-top: 0.9rem; padding-bottom: 1.2rem; }
        .hero-card { padding: 1rem; border-radius: 16px; }
        .brand-title { font-size: 1.05rem; }
        .brand-tagline { font-size: 0.75rem; }
        .metric-card { padding: 0.85rem 0.9rem; }
        .metric-card-value { font-size: 1.1rem; }
        .model-card-desc { min-height: unset; }
        section[data-testid="stSidebar"] button { white-space: normal; height: auto; }
    }

    @media (max-width: 480px) {
        .hero-card h1 { font-size: 1.55rem !important; }
        .hero-card p { font-size: 0.95rem !important; }
        .metric-card-label { font-size: 0.72rem; }
        .metric-card-value { font-size: 1rem; }
    }

    button:focus-visible, a:focus-visible {
        outline: 2px solid #2563eb !important;
        outline-offset: 2px !important;
    }
</style>
"""


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


def main() -> None:
    init_state()
    st.session_state.model_catalog = MODEL_CATALOG
    st.markdown(CSS, unsafe_allow_html=True)
    render_sidebar()

    page = st.session_state.nav_page
    if page == "dashboard":
        render_dashboard()
    elif page == "dataset":
        render_dataset_page()
    elif page in [entry["name"] for entry in MODEL_CATALOG]:
        render_model_page(page, build_training_pipeline, build_tree_params)
    else:
        st.session_state.nav_page = "dashboard"
        render_dashboard()


if __name__ == "__main__":
    main()
