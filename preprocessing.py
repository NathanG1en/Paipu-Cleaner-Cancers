"""
Text preprocessing functions for cancer sample classification.

Provides text cleaning, normalization, and column analysis utilities.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import polars as pl

from config import PRIORITY_COLS

if TYPE_CHECKING:
    pass


# =============================================================================
# Text Preprocessing Functions
# =============================================================================

def clean_texts(
    row: Dict[str, Any],
    priority_cols: Tuple[str, ...] = PRIORITY_COLS,
    use_normalized: bool = False,
) -> str:
    """
    Combine and clean text fields from a row into a single string.

    Args:
        row: Dictionary containing row data (column name -> value).
        priority_cols: Column names to extract text from.
        use_normalized: If True, look for "_norm" suffixed columns first.

    Returns:
        Combined, cleaned text string suitable for NLP processing.
    """
    texts: List[str] = []

    for col in priority_cols:
        col_name = "{}_norm".format(col) if use_normalized else col
        if col_name not in row and use_normalized:
            col_name = col

        val = row.get(col_name)
        if val is None or (isinstance(val, str) and val.lower() in ("", "nan", "none")):
            continue

        text = str(val).strip()
        if text:
            texts.append(text)

    combined = " ".join(texts)
    combined = re.sub(r"\s+", " ", combined).strip()

    return combined


def _has_alphabetic(text: str) -> bool:
    """Check if text contains at least one alphabetic character."""
    return bool(re.search(r"[a-zA-Z]", text))


def normalize_text_column(col_expr: pl.Expr) -> pl.Expr:
    """
    Normalize a Polars text column expression for consistent comparison.

    Applies: UTF-8 cast, null fill, lowercase, separator normalization,
    whitespace collapsing, and edge trimming.
    """
    return (
        col_expr.cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[_/|\\]", " ")
        .str.replace_all(r"[^\w\s\-]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


# =============================================================================
# Text Column Analysis Functions
# =============================================================================

def identify_candidate_text_columns(
    df: pl.DataFrame,
    config=None,
) -> List[str]:
    """
    Identify DataFrame columns that are good candidates for text analysis.
    """
    from config import DEFAULT_CONFIG
    if config is None:
        config = DEFAULT_CONFIG

    candidates: List[str] = []
    n_rows = len(df)

    for col in df.columns:
        if df[col].dtype != pl.Utf8:
            continue

        col_lower = col.lower()
        if any(p in col_lower for p in config.exclude_patterns):
            continue

        non_null_count = df[col].drop_nulls().len()
        non_null_pct = non_null_count / n_rows if n_rows > 0 else 0

        if non_null_pct < config.min_non_null_pct:
            continue

        avg_len = df.select(
            pl.col(col).drop_nulls().str.len_chars().mean()
        ).item()

        if avg_len is None or avg_len < config.min_avg_length:
            continue

        candidates.append(col)

    return candidates


def preprocess_text_columns(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    suffix: str = "_norm",
) -> pl.DataFrame:
    """Normalize specified text columns and add as new columns with suffix."""
    if columns is None:
        columns = [c for c in PRIORITY_COLS if c in df.columns]

    normalizations = [
        normalize_text_column(pl.col(col)).alias("{}{}".format(col, suffix))
        for col in columns
        if col in df.columns
    ]

    if normalizations:
        df = df.with_columns(normalizations)

    return df


def get_text_column_stats(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Get statistics for text columns to assess their quality."""
    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == pl.Utf8]

    stats: List[Dict[str, Any]] = []
    n_rows = len(df)

    for col in columns:
        if col not in df.columns:
            continue

        non_null_count = df[col].drop_nulls().len()
        unique_count = df[col].n_unique()
        avg_len = df.select(
            pl.col(col).drop_nulls().str.len_chars().mean()
        ).item()

        stats.append({
            "column_name": col,
            "non_null_count": non_null_count,
            "non_null_pct": non_null_count / n_rows if n_rows > 0 else 0,
            "avg_length": avg_len or 0.0,
            "unique_count": unique_count,
        })

    return pl.DataFrame(stats)
