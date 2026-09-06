from __future__ import annotations

import os


APP_NAME = os.getenv("APP_NAME", "Decision Tree Generator")
APP_TAGLINE = os.getenv(
    "APP_TAGLINE",
    "Upload a dataset, train a decision tree, inspect the model, and make predictions.",
)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
DEFAULT_TEST_SIZE = float(os.getenv("DEFAULT_TEST_SIZE", "0.2"))
DEFAULT_RANDOM_STATE = int(os.getenv("DEFAULT_RANDOM_STATE", "42"))
DEFAULT_MAX_DEPTH = int(os.getenv("DEFAULT_MAX_DEPTH", "5"))
DEFAULT_MIN_SAMPLES_SPLIT = int(os.getenv("DEFAULT_MIN_SAMPLES_SPLIT", "2"))
DEFAULT_MIN_SAMPLES_LEAF = int(os.getenv("DEFAULT_MIN_SAMPLES_LEAF", "1"))

PLOT_CONFIG = {
    "pairplot_sample_size": 100,
    "heatmap_figsize": (10, 8),
    "feature_importance_figsize": (10, 6),
}

COLUMN_TYPES = {
    "numeric": ["int64", "float64"],
    "categorical": ["object", "category"],
}
