from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from sklearn import tree


def plot_tree_figure(model, feature_names: list[str], class_names: list[str] | None, max_depth: int | None = None):
    """Render a decision tree with a depth cap to keep large trees readable."""
    figure_width = max(16, min(28, 0.35 * max(1, len(feature_names))))
    figure, axis = plt.subplots(figsize=(figure_width, 10))

    tree.plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        max_depth=max_depth,
        fontsize=8,
        ax=axis,
    )
    axis.set_title("Decision Tree Visualization")
    figure.tight_layout()
    return figure


def get_tree_depth(model) -> int:
    return int(getattr(model, "tree_", None).max_depth) if hasattr(model, "tree_") else 0
