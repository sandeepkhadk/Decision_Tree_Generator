from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
from sklearn import tree


def plot_tree_figure(model, feature_names: list[str], class_names: list[str] | None, max_depth: int | None = None):
    """Render a decision tree with a depth cap to keep large trees readable."""
    tree_depth = int(getattr(model, "tree_", None).max_depth) if hasattr(model, "tree_") else 1
    figure_width = max(18, min(34, 0.45 * max(1, len(feature_names))))
    depth_for_layout = tree_depth if max_depth is None else min(tree_depth, max_depth)
    figure_height = max(10, min(24, 2.5 + 0.8 * max(1, depth_for_layout)))
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=180)

    fontsize = max(7, min(12, 12 - max(0, len(feature_names) // 10)))

    tree.plot_tree(
        model,
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
    axis.set_title("Decision Tree Visualization")
    figure.tight_layout()
    return figure


def figure_to_png_bytes(figure) -> bytes:
    """Convert a matplotlib figure to PNG bytes for download."""
    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", dpi=240)
    buffer.seek(0)
    return buffer.getvalue()


def get_tree_depth(model) -> int:
    return int(getattr(model, "tree_", None).max_depth) if hasattr(model, "tree_") else 0
