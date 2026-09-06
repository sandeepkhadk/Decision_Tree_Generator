from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MAX_UPLOAD_MB


_ALLOWED_FILE_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe representation of an uploaded filename."""
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "dataset"


def file_hash(file_bytes: bytes) -> str:
    """Create a stable hash for cache and session keys."""
    return hashlib.sha256(file_bytes).hexdigest()


def build_signature(payload: dict[str, Any]) -> str:
    """Create a deterministic signature for the current training configuration."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def validate_file_size(file_size: int | None) -> None:
    """Raise a friendly error when uploads exceed the configured size limit."""
    if file_size is None:
        return

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if file_size > max_bytes:
        raise ValueError(
            f"The uploaded file is too large. Please upload a file smaller than {MAX_UPLOAD_MB} MB."
        )


def validate_extension(filename: str) -> None:
    """Reject unsupported file extensions before parsing."""
    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_FILE_EXTENSIONS:
        raise ValueError("Please upload a CSV or Excel file with a supported extension.")


def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded dataset into a DataFrame with friendly error handling."""
    validate_extension(filename)

    buffer = io.BytesIO(file_bytes)
    extension = Path(filename).suffix.lower()

    try:
        if extension == ".csv":
            return pd.read_csv(buffer, low_memory=False)
        return pd.read_excel(buffer)
    except Exception as exc:  # pragma: no cover - guarded by user-facing error handling
        raise ValueError(
            "Unable to process this dataset. Please verify that the file is a valid CSV or Excel file."
        ) from exc


def summarize_dataset(df: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Build a compact dataset profile for the dashboard and explorer tabs."""
    duplicate_rows = int(df.duplicated().sum())
    missing_values = df.isna().sum().sort_values(ascending=False)
    if df.empty:
        numeric_preview = pd.DataFrame()
    else:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_preview = df[numeric_columns].describe().transpose() if numeric_columns else pd.DataFrame()
    full_preview = df.head(10)

    return {
        "filename": sanitize_filename(filename),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_summary": numeric_preview,
        "preview": full_preview,
    }


def detect_supported_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split columns into numeric, categorical, and unsupported groups."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    supported = set(numeric_cols) | set(categorical_cols)
    unsupported_cols = [column for column in df.columns if column not in supported]
    return numeric_cols, categorical_cols, unsupported_cols
