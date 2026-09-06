from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
from sklearn import tree


_PREFIX_PATTERN = re.compile(r"^(num|cat)__")


def humanize_feature_names(feature_names: list[str]) -> list[str]:
    """Strip ColumnTransformer prefixes so tree labels stay short and readable."""
    return [_PREFIX_PATTERN.sub("", name) for name in feature_names]


def _unwrap_estimator(candidate):
    """Recursively unwrap array/list-wrapped sub-estimators (e.g. Gradient Boosting)."""
    if isinstance(candidate, (list, tuple)) and candidate:
        return _unwrap_estimator(candidate[0])
    if hasattr(candidate, "tree_"):
        return candidate
    if hasattr(candidate, "__len__") and hasattr(candidate, "__getitem__") and not hasattr(candidate, "tree_"):
        try:
            if len(candidate) > 0:
                return _unwrap_estimator(candidate[0])
        except Exception:
            pass
    return candidate


def _resolve_tree_model(model, estimator_index: int = 0):
    """Return a plottable single tree, selecting one estimator from an ensemble if needed."""
    if hasattr(model, "tree_"):
        return model
    estimators = getattr(model, "estimators_", None)
    if estimators is not None and len(estimators) > 0:
        index = min(max(estimator_index, 0), len(estimators) - 1)
        return _unwrap_estimator(estimators[index])
    return model


def get_tree_estimator_count(model) -> int:
    """Return how many individually plottable trees a model exposes (0 if none)."""
    if hasattr(model, "tree_"):
        return 1
    estimators = getattr(model, "estimators_", None)
    return len(estimators) if estimators is not None else 0


def plot_tree_figure(
    model,
    feature_names: list[str],
    class_names: list[str] | None,
    max_depth: int | None = None,
    estimator_index: int = 0,
):
    """Render a decision tree, sizing the figure from its actual leaf count and depth.

    This avoids overlapping nodes: instead of guessing dimensions from the number of
    input features, the layout is driven by the fitted tree's real shape (number of
    leaves visible at the chosen depth, and the depth itself), so small, deep, and wide
    trees all get proportionate space.
    """
    tree_model = _resolve_tree_model(model, estimator_index)
    if not hasattr(tree_model, "tree_"):
        raise ValueError("Tree visualization is not available for this model.")

    display_names = humanize_feature_names(feature_names)
    sk_tree = tree_model.tree_
    full_depth = int(sk_tree.max_depth)
    depth_for_layout = full_depth if max_depth is None else min(full_depth, max_depth)
    depth_for_layout = max(depth_for_layout, 0)
    visible_nodes = min(int(sk_tree.n_leaves), 2 ** depth_for_layout) or 1

    figure_width = min(90, max(12, visible_nodes * 2.1))
    figure_height = min(48, max(8, (depth_for_layout + 1) * 2.6))
    max_pixel_span = 7000
    dpi = min(260, max(90, max_pixel_span / max(figure_width, figure_height)))

    figure, axis = plt.subplots(figsize=(figure_width, figure_height), dpi=dpi)
    fontsize = min(13, max(6, 15 - visible_nodes // 6))

    tree.plot_tree(
        tree_model,
        feature_names=display_names,
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


def figure_to_svg_markup(figure) -> str:
    """Return raw SVG markup so it can be embedded and manipulated client-side."""
    buffer = BytesIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    return buffer.getvalue().decode("utf-8")


def figure_to_svg_data_uri(figure) -> str:
    """Convert a matplotlib figure to an SVG data URI (kept for simple <img> embedding)."""
    svg_text = figure_to_svg_markup(figure)
    encoded_svg = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded_svg}"


def get_tree_depth(model, estimator_index: int = 0) -> int:
    tree_model = _resolve_tree_model(model, estimator_index)
    return int(tree_model.tree_.max_depth) if hasattr(tree_model, "tree_") else 0


def build_tree_viewer_html(svg_markup: str, viewer_id: str, height: int = 620) -> str:
    """Build a self-contained pan/zoom/fullscreen viewer for an inline SVG tree diagram."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "-", viewer_id) or "tree"
    return f"""
<div id="wrap-{safe_id}" class="tv-wrap">
  <style>
    #wrap-{safe_id} {{ position: relative; width: 100%; height: {height}px; border: 1px solid #e2e8f0;
      border-radius: 12px; background: #ffffff; overflow: hidden; font-family: -apple-system, "Segoe UI", sans-serif; }}
    #toolbar-{safe_id} {{ position: absolute; top: 10px; right: 10px; z-index: 10; display: flex; gap: 6px; }}
    #toolbar-{safe_id} button {{ border: 1px solid #cbd5e1; background: #ffffff; color: #0f172a;
      border-radius: 8px; width: 34px; height: 34px; cursor: pointer; font-size: 15px; line-height: 1;
      transition: background-color 0.15s ease; }}
    #toolbar-{safe_id} button:hover {{ background: #f1f5f9; }}
    #toolbar-{safe_id} button:focus-visible {{ outline: 2px solid #2563eb; outline-offset: 2px; }}
    #stage-{safe_id} {{ position: absolute; inset: 0; cursor: grab; touch-action: none; }}
    #stage-{safe_id}.dragging {{ cursor: grabbing; }}
    #canvas-{safe_id} {{ position: absolute; top: 0; left: 0; transform-origin: 0 0; }}
    #canvas-{safe_id} svg {{ display: block; }}
    #hint-{safe_id} {{ position: absolute; left: 10px; bottom: 8px; font-size: 11px; color: #94a3b8; z-index: 5; }}
  </style>
  <div id="toolbar-{safe_id}">
    <button id="zoomin-{safe_id}" title="Zoom in" aria-label="Zoom in">+</button>
    <button id="zoomout-{safe_id}" title="Zoom out" aria-label="Zoom out">&#8722;</button>
    <button id="fit-{safe_id}" title="Fit view" aria-label="Fit view">&#10530;</button>
    <button id="reset-{safe_id}" title="Reset view" aria-label="Reset view">&#8635;</button>
    <button id="fullscreen-{safe_id}" title="Fullscreen" aria-label="Toggle fullscreen">&#9974;</button>
  </div>
  <div id="stage-{safe_id}">
    <div id="canvas-{safe_id}">{svg_markup}</div>
  </div>
  <div id="hint-{safe_id}">Scroll to zoom &middot; drag to pan</div>
  <script>
  (function() {{
    const wrap = document.getElementById("wrap-{safe_id}");
    const stage = document.getElementById("stage-{safe_id}");
    const canvas = document.getElementById("canvas-{safe_id}");
    let scale = 1, originX = 0, originY = 0, dragging = false, startX = 0, startY = 0;

    function apply() {{
      canvas.style.transform = "translate(" + originX + "px, " + originY + "px) scale(" + scale + ")";
    }}

    function svgSize() {{
      const svg = canvas.querySelector("svg");
      if (!svg) return null;
      if (svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) {{
        return {{ width: svg.viewBox.baseVal.width, height: svg.viewBox.baseVal.height }};
      }}
      const rect = svg.getBoundingClientRect();
      return {{ width: rect.width, height: rect.height }};
    }}

    function fit() {{
      const size = svgSize();
      if (!size || !size.width || !size.height) return;
      const stageRect = stage.getBoundingClientRect();
      scale = Math.min(stageRect.width / size.width, stageRect.height / size.height) * 0.96;
      originX = (stageRect.width - size.width * scale) / 2;
      originY = (stageRect.height - size.height * scale) / 2;
      apply();
    }}

    document.getElementById("zoomin-{safe_id}").addEventListener("click", function() {{
      scale = Math.min(6, scale * 1.25); apply();
    }});
    document.getElementById("zoomout-{safe_id}").addEventListener("click", function() {{
      scale = Math.max(0.05, scale / 1.25); apply();
    }});
    document.getElementById("reset-{safe_id}").addEventListener("click", fit);
    document.getElementById("fit-{safe_id}").addEventListener("click", fit);

    stage.addEventListener("wheel", function(event) {{
      event.preventDefault();
      const delta = event.deltaY > 0 ? 0.9 : 1.1;
      scale = Math.min(6, Math.max(0.05, scale * delta));
      apply();
    }}, {{ passive: false }});

    stage.addEventListener("mousedown", function(event) {{
      dragging = true; startX = event.clientX - originX; startY = event.clientY - originY;
      stage.classList.add("dragging");
    }});
    window.addEventListener("mousemove", function(event) {{
      if (!dragging) return;
      originX = event.clientX - startX; originY = event.clientY - startY; apply();
    }});
    window.addEventListener("mouseup", function() {{
      dragging = false; stage.classList.remove("dragging");
    }});

    function fullscreenDoc() {{
      try {{
        if (window.frameElement && window.parent && window.parent.document) {{
          return window.parent.document;
        }}
      }} catch (e) {{ /* cross-origin parent, fall back to own document */ }}
      return document;
    }}

    document.getElementById("fullscreen-{safe_id}").addEventListener("click", function() {{
      try {{
        const fsDoc = fullscreenDoc();
        if (fsDoc.fullscreenElement) {{
          fsDoc.exitFullscreen();
        }} else {{
          const target = window.frameElement || wrap;
          if (target.requestFullscreen) {{
            target.requestFullscreen();
          }}
        }}
      }} catch (e) {{ /* fullscreen unsupported in this context */ }}
    }});

    document.addEventListener("fullscreenchange", function() {{
      setTimeout(fit, 80);
    }});
    try {{
      const fsDoc = fullscreenDoc();
      if (fsDoc !== document) {{
        fsDoc.addEventListener("fullscreenchange", function() {{
          setTimeout(fit, 80);
        }});
      }}
    }} catch (e) {{ /* ignore */ }}
    window.addEventListener("resize", function() {{ setTimeout(fit, 60); }});
    setTimeout(fit, 80);
  }})();
  </script>
</div>
"""

