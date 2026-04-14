"""
Metadata enrichment for classified samples.

Adds derived boolean columns (is_cell_line, is_benign) by scanning
multiple text columns for relevant patterns. Designed to run as a
post-classification step so the flags appear alongside final_label.
"""

from __future__ import annotations

from typing import List

import polars as pl

from config import (
    CELL_LINE_PATTERN,
    CELL_LINE_SEARCH_COLS,
    BENIGN_PATTERN,
    BENIGN_SEARCH_COLS,
)
from preprocessing import normalize_text_column


def _detect_flag(
    df: pl.DataFrame,
    pattern: str,
    search_cols: tuple[str, ...],
    flag_name: str,
    use_normalized: bool = True,
) -> pl.DataFrame:
    """
    Scan multiple columns for a regex pattern and produce a single
    boolean flag column.

    For each column in *search_cols* that exists in the DataFrame,
    normalize the text and check for a match. The final flag is True
    if **any** column matches.

    Args:
        df: Input DataFrame.
        pattern: Regex pattern to search for (case-insensitive via normalize).
        search_cols: Ordered tuple of column names to inspect.
        flag_name: Name of the resulting boolean column.
        use_normalized: Whether to prefer pre-normalized ``_norm`` columns.

    Returns:
        DataFrame with *flag_name* column added.
    """
    match_cols: List[str] = []

    for col in search_cols:
        # Prefer the pre-normalized version if available
        col_ref = col
        if use_normalized and f"{col}_norm" in df.columns:
            col_ref = f"{col}_norm"
        elif col not in df.columns:
            continue

        tmp_col = f"_tmp_{flag_name}_{col}"
        df = df.with_columns(
            normalize_text_column(pl.col(col_ref))
            .str.contains(pattern)
            .alias(tmp_col)
        )
        match_cols.append(tmp_col)

    if match_cols:
        df = df.with_columns(
            pl.any_horizontal([pl.col(c) for c in match_cols]).alias(flag_name)
        )
        # Clean up temporaries
        df = df.drop(match_cols)
    else:
        df = df.with_columns(pl.lit(False).alias(flag_name))

    return df


def _has_cell_line_column_value(df: pl.DataFrame) -> pl.Expr:
    """
    Check if the dedicated ``cell_line`` column contains a real
    (non-null, non-"nan") value, indicating the sample is from a cell line.

    The column often holds semicolon-delimited strings with interspersed
    "nan" tokens (e.g. ``"nan 4T1 nan nan"``).  This helper returns True
    when **any** token in the value is not "nan".
    """
    if "cell_line" not in df.columns:
        return pl.lit(False)

    return (
        pl.col("cell_line")
        .fill_null("")
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.replace_all(r"[_/|\\;,]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        # After cleanup, if only "nan" tokens remain -> no real value
        .str.replace_all(r"\bnan\b", "")
        .str.replace_all(r"\s+", "")
        .str.len_chars()
        .gt(0)
    )


def enrich_metadata(
    df: pl.DataFrame,
    use_normalized: bool = True,
) -> pl.DataFrame:
    """
    Add ``is_cell_line`` and ``is_benign`` boolean columns to the DataFrame.

    Scans configurable sets of text columns for cell-line and benign
    indicators, producing one boolean flag per concept.  For ``is_cell_line``,
    also checks whether the dedicated ``cell_line`` column carries a real
    (non-null, non-"nan") value so that arbitrary cell-line names are caught.

    Args:
        df: DataFrame (typically post-classification).
        use_normalized: Whether to prefer ``_norm`` columns for matching.

    Returns:
        DataFrame with ``is_cell_line`` and ``is_benign`` columns added.
    """
    # Regex-based cell-line detection across multiple columns
    df = _detect_flag(
        df,
        pattern=CELL_LINE_PATTERN,
        search_cols=CELL_LINE_SEARCH_COLS,
        flag_name="is_cell_line",
        use_normalized=use_normalized,
    )

    # Short-circuit: also flag rows where the dedicated cell_line column
    # has any real (non-nan) value, regardless of regex match
    has_cl_col = _has_cell_line_column_value(df)
    df = df.with_columns(
        (pl.col("is_cell_line") | has_cl_col).alias("is_cell_line")
    )

    df = _detect_flag(
        df,
        pattern=BENIGN_PATTERN,
        search_cols=BENIGN_SEARCH_COLS,
        flag_name="is_benign",
        use_normalized=use_normalized,
    )

    # Post-processing: un-flag rows where "adenoma" matched but "carcinoma"
    # is also present in the same column (mixed adenoma/carcinoma ≠ benign).
    # Polars regex doesn't support lookaheads, so we handle this here.
    carcinoma_in_any: List[pl.Expr] = []
    for col in BENIGN_SEARCH_COLS:
        col_ref = col
        if use_normalized and f"{col}_norm" in df.columns:
            col_ref = f"{col}_norm"
        elif col not in df.columns:
            continue
        carcinoma_in_any.append(
            normalize_text_column(pl.col(col_ref))
            .str.contains(r"carcinomas?\b")
        )

    if carcinoma_in_any:
        has_carcinoma = pl.any_horizontal(carcinoma_in_any)
        adenoma_only = pl.any_horizontal([
            normalize_text_column(
                pl.col(c if c in df.columns else f"{c}_norm")
            ).str.contains(r"\badenomas?\b")
            for c in BENIGN_SEARCH_COLS
            if c in df.columns or f"{c}_norm" in df.columns
        ])
        # Un-flag: if benign was set AND the match was from adenoma AND
        # carcinoma is also present → not truly benign
        df = df.with_columns(
            pl.when(pl.col("is_benign") & adenoma_only & has_carcinoma)
            .then(pl.lit(False))
            .otherwise(pl.col("is_benign"))
            .alias("is_benign")
        )

    return df
