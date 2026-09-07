from __future__ import annotations

import json
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import ConfusionMatrixDisplay

from config import APP_NAME
from model_utils import ModelArtifacts, get_model_meta
from tree_utils import get_tree_depth, get_tree_estimator_count, plot_tree_figure


def build_dataset_profile_report(
    summary: dict[str, Any],
    dataset_name: str,
    unsupported_cols: list[str] | None = None,
    target_col: str | None = None,
    selected_features: list[str] | None = None,
    target_distribution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "report_type": "dataset_profile",
        "dataset_name": dataset_name,
        "target_column": target_col,
        "selected_features": selected_features or [],
        "unsupported_columns": unsupported_cols or [],
        "rows": int(summary["rows"]),
        "columns": int(summary["columns"]),
        "duplicate_rows": int(summary["duplicate_rows"]),
        "missing_cells": int(summary["missing_values"].sum()),
        "column_names": list(summary["column_names"]),
        "dtypes": dict(summary["dtypes"]),
        "missing_values": summary["missing_values"].to_dict(),
        "numeric_summary": summary["numeric_summary"].reset_index().to_dict(orient="records"),
        "preview": summary["preview"].to_dict(orient="records"),
    }

    if target_distribution is not None:
        report["target_distribution"] = target_distribution

    return report


def build_model_bundle(
    artifacts: ModelArtifacts,
    model_name: str,
    dataset_name: str,
    train_meta: dict[str, Any] | None = None,
) -> bytes:
    payload = {
        "bundle_version": 1,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "train_meta": train_meta or {},
        "artifacts": artifacts,
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def build_json_download(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")


def file_name_safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(name).stem) or "dataset"


def build_feature_importance_figure(model, feature_names: list[str]):
    if not hasattr(model, "feature_importances_"):
        return None

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances[importances > 0].sort_values(ascending=False).head(20)
    if importances.empty:
        return None

    figure, axis = plt.subplots(figsize=(10, max(4, 0.35 * len(importances) + 2)), dpi=180)
    importances.sort_values().plot(kind="barh", ax=axis, color="#2563eb")
    axis.set_xlabel("Importance")
    axis.set_ylabel("Feature")
    axis.set_title("Top Features")
    figure.tight_layout()
    return figure


def build_confusion_matrix_figure(metrics: dict[str, Any]):
    if "Confusion Matrix" not in metrics:
        return None

    labels = metrics.get("Labels", [])
    matrix = metrics["Confusion Matrix"]
    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Confusion Matrix")
    figure.tight_layout()
    return figure


def build_tree_visualization_figure(artifacts: ModelArtifacts):
    model = artifacts.pipeline.named_steps["model"]
    estimator_count = get_tree_estimator_count(model)
    if estimator_count <= 0:
        return None

    estimator_index = 0
    tree_depth = get_tree_depth(model, estimator_index)
    if tree_depth <= 0:
        return None

    depth_limit = 1 if tree_depth <= 1 else min(4, tree_depth)
    try:
        return plot_tree_figure(
            model,
            artifacts.feature_names,
            artifacts.classes_,
            max_depth=depth_limit,
            estimator_index=estimator_index,
        )
    except ValueError:
        return None


def build_result_discussion(metrics: dict[str, Any], artifacts: ModelArtifacts) -> tuple[str, str]:
    """Create a short discussion and recommendation summary for the report."""
    if artifacts.problem_type != "classification":
        return (
            "The regression model produced the evaluation metrics shown above. Review the error values alongside the tree visualizations to judge whether the model is capturing the signal well.",
            "If the errors are still high, consider feature engineering, a different tree depth, or trying an alternative tree-based model.",
        )

    accuracy = float(metrics.get("Accuracy", 0.0))
    weighted_f1 = float(metrics.get("F1 Score", 0.0))
    macro_f1 = float(metrics.get("Macro F1", weighted_f1))
    roc_auc = metrics.get("ROC AUC")
    pr_auc = metrics.get("PR AUC")
    warnings = metrics.get("Warnings", [])
    minority_share = None

    if metrics.get("Labels") and metrics.get("Per Class Metrics"):
        per_class = metrics["Per Class Metrics"]
        support_values = [int(values.get("support", 0)) for values in per_class.values() if isinstance(values, dict)]
        total_support = sum(support_values)
        if total_support:
            minority_share = min(support_values) / total_support * 100.0

    discussion_parts = [
        f"The classifier achieved {accuracy:.1%} accuracy with a weighted F1-score of {weighted_f1:.3f} and a macro F1-score of {macro_f1:.3f}.",
    ]
    if roc_auc is not None:
        discussion_parts.append(f"ROC-AUC was {float(roc_auc):.3f}, which helps show ranking quality beyond accuracy.")
    if pr_auc is not None:
        discussion_parts.append(f"PR-AUC was {float(pr_auc):.3f}, which is especially important when the positive class is rare.")
    if minority_share is not None:
        discussion_parts.append(f"The smallest class represents about {minority_share:.1f}% of the labeled data.")
    if warnings:
        discussion_parts.append("The evaluation also flagged class imbalance or zero-recall behavior, so accuracy alone is not a reliable success signal here.")

    recommendation_parts = [
        "Keep balanced class weights enabled or try resampling if the model still predicts the majority class too often.",
        "Use the threshold slider to trade precision for recall when the positive class is the one you care about most.",
        "If recall remains near zero for any class, inspect the class distribution, feature quality, and train/test split before trusting the model.",
    ]
    if weighted_f1 < 0.6 or warnings:
        recommendation_parts.insert(0, "The model is not yet reliable enough for deployment on its current setting.")

    return " ".join(discussion_parts), " ".join(recommendation_parts)


def build_result_pdf(
    dataset_name: str,
    model_name: str,
    artifacts: ModelArtifacts,
    train_meta: dict[str, Any],
    metrics: dict[str, Any],
) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyTextSmall",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )

    model_meta = get_model_meta(model_name)
    elements: list[Any] = [
        Paragraph(f"{APP_NAME} - Result Report", title_style),
        Paragraph(f"Dataset: {dataset_name}", body_style),
        Paragraph(f"Model: {model_meta['label']}", body_style),
        Paragraph(f"Problem type: {artifacts.problem_type.capitalize()}", body_style),
        Spacer(1, 0.14 * inch),
    ]

    warnings = metrics.get("Warnings", [])
    if warnings:
        elements.append(Paragraph("Warnings", heading_style))
        for warning in warnings:
            elements.append(Paragraph(f"- {warning}", body_style))
        elements.append(Spacer(1, 0.08 * inch))

    if metrics.get("Degenerate Prediction"):
        elements.append(Paragraph("Diagnostic Note", heading_style))
        elements.append(
            Paragraph(
                "Model predicts only one class - check for class imbalance, class weights, resampling, and the decision threshold.",
                body_style,
            )
        )
        elements.append(Spacer(1, 0.08 * inch))

    zero_recall_classes = metrics.get("Zero Recall Classes", [])
    if zero_recall_classes:
        elements.append(Paragraph("Zero Recall Classes", heading_style))
        elements.append(Paragraph(
            ", ".join(str(label) for label in zero_recall_classes),
            body_style,
        ))
        elements.append(Spacer(1, 0.08 * inch))

    summary_rows = [
        ["Feature columns", str(len(artifacts.feature_columns))],
        ["Samples", str(train_meta.get("n_samples", "-"))],
        ["Train/test split", f"{train_meta.get('train_size', 'default')} / {train_meta.get('test_size', 'default')}"],
        ["Target column", artifacts.target_name],
    ]
    summary_table = Table(summary_rows, colWidths=[1.7 * inch, 4.7 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(Paragraph("Training Summary", heading_style))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.15 * inch))

    metric_rows = [["Metric", "Value"]]
    for key, value in metrics.items():
        if key in {
            "Confusion Matrix",
            "Classification Report",
            "Labels",
            "Per Class Metrics",
            "Warnings",
            "Degenerate Prediction",
            "Zero Recall Classes",
        }:
            continue
        metric_rows.append([str(key), f"{value:.4f}" if isinstance(value, (int, float)) else str(value)])

    metric_table = Table(metric_rows, colWidths=[2.0 * inch, 4.4 * inch])
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(Paragraph("Evaluation Metrics", heading_style))
    elements.append(metric_table)

    if artifacts.problem_type == "classification" and metrics.get("Per Class Metrics"):
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("Per-Class Breakdown", heading_style))
        per_class_rows = [["Class", "Precision", "Recall", "F1", "Support"]]
        for label, values in metrics["Per Class Metrics"].items():
            per_class_rows.append(
                [
                    str(label),
                    f"{values.get('precision', 0.0):.4f}",
                    f"{values.get('recall', 0.0):.4f}",
                    f"{values.get('f1-score', 0.0):.4f}",
                    str(int(values.get("support", 0))),
                ]
            )

        per_class_table = Table(per_class_rows, colWidths=[1.6 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch, 1.0 * inch])
        per_class_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(per_class_table)

    if metrics.get("ROC AUC") is not None or metrics.get("PR AUC") is not None:
        elements.append(Spacer(1, 0.12 * inch))
        auc_rows = [["Threshold", str(metrics.get("Decision Threshold", 0.5))]]
        if metrics.get("ROC AUC") is not None:
            auc_rows.append(["ROC-AUC", f"{metrics['ROC AUC']:.4f}"])
        if metrics.get("PR AUC") is not None:
            auc_rows.append(["PR-AUC", f"{metrics['PR AUC']:.4f}"])
        auc_table = Table(auc_rows, colWidths=[1.6 * inch, 1.6 * inch])
        auc_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        elements.append(auc_table)

    if artifacts.problem_type == "classification":
        discussion_text, recommendation_text = build_result_discussion(metrics, artifacts)
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("Discussion", heading_style))
        elements.append(Paragraph(discussion_text, body_style))
        elements.append(Spacer(1, 0.08 * inch))
        elements.append(Paragraph("Recommendation", heading_style))
        elements.append(Paragraph(recommendation_text, body_style))

    if artifacts.problem_type == "classification" and metrics.get("Degenerate Prediction"):
        elements.append(Spacer(1, 0.12 * inch))
        elements.append(Paragraph("Degenerate Prediction Flag", heading_style))
        elements.append(
            Paragraph(
                "Model predicts only one class - check for class imbalance.",
                body_style,
            )
        )

    tree_figure = build_tree_visualization_figure(artifacts)
    if tree_figure is not None:
        elements.append(Spacer(1, 0.18 * inch))
        elements.append(Paragraph("Tree Visualization", heading_style))
        image_buffer = BytesIO()
        tree_figure.savefig(image_buffer, format="png", bbox_inches="tight", dpi=220)
        image_buffer.seek(0)
        elements.append(Image(image_buffer, width=6.8 * inch, height=4.2 * inch))
        plt.close(tree_figure)

    feature_figure = build_feature_importance_figure(artifacts.pipeline.named_steps["model"], artifacts.feature_names)
    if feature_figure is not None:
        elements.append(Spacer(1, 0.18 * inch))
        elements.append(Paragraph("Top Feature Importance", heading_style))
        image_buffer = BytesIO()
        feature_figure.savefig(image_buffer, format="png", bbox_inches="tight", dpi=220)
        image_buffer.seek(0)
        elements.append(Image(image_buffer, width=6.8 * inch, height=3.2 * inch))
        plt.close(feature_figure)

    confusion_figure = build_confusion_matrix_figure(metrics)
    if confusion_figure is not None and artifacts.problem_type == "classification":
        elements.append(Spacer(1, 0.18 * inch))
        elements.append(Paragraph("Confusion Matrix Chart", heading_style))
        image_buffer = BytesIO()
        confusion_figure.savefig(image_buffer, format="png", bbox_inches="tight", dpi=220)
        image_buffer.seek(0)
        elements.append(Image(image_buffer, width=6.8 * inch, height=4.2 * inch))
        plt.close(confusion_figure)

    document.build(elements)
    return buffer.getvalue()