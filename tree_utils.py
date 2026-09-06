from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
from sklearn import tree


def _resolve_tree_model(model):
    if hasattr(model, "tree_"):
        return model
    estimators = getattr(model, "estimators_", None)
    if estimators is not None and len(estimators) > 0:
        first_estimator = estimators[0]
        if isinstance(first_estimator, (list, tuple)) and first_estimator:
            first_estimator = first_estimator[0]
        elif hasattr(first_estimator, "__getitem__") and not hasattr(first_estimator, "tree_"):
            try:
                first_estimator = first_estimator[0]
            except Exception:
                pass
        if hasattr(first_estimator, "tree_"):
            return first_estimator
    return model


def plot_tree_figure(model, feature_names: list[str], class_names: list[str] | None, max_depth: int | None = None):
    """Render a decision tree with a depth cap to keep large trees readable."""
    tree_model = _resolve_tree_model(model)
    if not hasattr(tree_model, "tree_"):
        raise ValueError("Tree visualization is not available for this model.")
    tree_depth = int(getattr(tree_model, "tree_", None).max_depth) if hasattr(tree_model, "tree_") else 1
    figure_width = max(18, min(34, 0.45 * max(1, len(feature_names))))
    depth_for_layout = tree_depth if max_depth is None else min(tree_depth, max_depth)
    figure_height = max(10, min(24, 2.5 + 0.8 * max(1, depth_for_layout)))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=260)

    fontsize = max(7, min(12, 12 - max(0, len(feature_names) // 10)))

    tree.plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        max_depth=max_depth,
        fontsize=fontsize,
        label="all",
        ax=axis,
    )
    axis.set_title("Tree Visualization")
    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure) -> bytes:
    """Convert a matplotlib figure to PNG bytes for download."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=300)
    buffer.seek(0)
    return buffer.getvalue()


def get_tree_depth(model) -> int:
    tree_model = _resolve_tree_model(model)
    return int(getattr(tree_model, "tree_", None).max_depth) if hasattr(tree_model, "tree_") else 0
