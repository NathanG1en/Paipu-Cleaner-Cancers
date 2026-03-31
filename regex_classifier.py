"""
Regex-based cancer classification.

Provides pattern-matching classification using configurable regex patterns
for detecting cancer and non-cancer signals in text metadata.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import polars as pl

from config import (
    CANCER_POS,
    CANCER_NEG,
    ONCO_TRAPS,
    ClassificationLabel as CL,
)
from preprocessing import normalize_text_column


def apply_regex_classification(
    df: pl.DataFrame,
    priority_cols: List[str],
    use_normalized: bool = True,
) -> pl.DataFrame:
    """
    Apply regex-based cancer classification to DataFrame.

    Args:
        df: Input DataFrame.
        priority_cols: Columns to search.
        use_normalized: Whether to use normalized columns.

    Returns:
        DataFrame with regex_label and regex_reason columns added.
    """
    sample_name_col = "title" if "title" in df.columns else "biosample"

    def get_col(col: str) -> str:
        if use_normalized and "{}_norm".format(col) in df.columns:
            return "{}_norm".format(col)
        return col

    # Check sample name
    df = df.with_columns([
        normalize_text_column(pl.col(sample_name_col))
        .str.contains(CANCER_POS)
        .alias("cancer_in_sample_name"),

        normalize_text_column(pl.col(sample_name_col))
        .str.contains(CANCER_NEG)
        .alias("negative_in_sample_name"),

        normalize_text_column(pl.col(sample_name_col))
        .str.contains(ONCO_TRAPS)
        .alias("onco_trap_in_sample_name"),
    ])

    # Check priority columns
    for col in priority_cols:
        col_ref = get_col(col)
        if col_ref in df.columns:
            df = df.with_columns([
                normalize_text_column(pl.col(col_ref))
                .str.contains(CANCER_POS)
                .alias("cancer_in_{}".format(col)),

                normalize_text_column(pl.col(col_ref))
                .str.contains(CANCER_NEG)
                .alias("negative_in_{}".format(col)),
            ])

    # Count mentions
    cancer_cols = ["cancer_in_{}".format(c) for c in priority_cols if "cancer_in_{}".format(c) in df.columns]
    negative_cols = ["negative_in_{}".format(c) for c in priority_cols if "negative_in_{}".format(c) in df.columns]

    df = df.with_columns([
        pl.sum_horizontal([pl.col(c) for c in cancer_cols]).alias("n_cancer_mentions")
        if cancer_cols else pl.lit(0).alias("n_cancer_mentions"),

        pl.sum_horizontal([pl.col(c) for c in negative_cols]).alias("n_negative_mentions")
        if negative_cols else pl.lit(0).alias("n_negative_mentions"),
    ])

    # Track sample-level vs study-level signals
    sample_level_cols = ["source_name", "tissue"]

    sample_neg_cols = [
        "negative_in_{}".format(c) for c in sample_level_cols
        if "negative_in_{}".format(c) in df.columns
    ]
    sample_cancer_cols = [
        "cancer_in_{}".format(c) for c in sample_level_cols
        if "cancer_in_{}".format(c) in df.columns
    ]

    df = df.with_columns([
        (pl.sum_horizontal([pl.col(c) for c in sample_neg_cols]).alias("n_sample_negative")
         if sample_neg_cols else pl.lit(0).alias("n_sample_negative")),
        (pl.sum_horizontal([pl.col(c) for c in sample_cancer_cols]).alias("n_sample_cancer")
         if sample_cancer_cols else pl.lit(0).alias("n_sample_cancer")),
    ])

    # Check for ctrl/control/normal in title
    df = df.with_columns(
        normalize_text_column(pl.col(sample_name_col))
        .str.contains(r"(?:\bctrl\b|\bcontrol\b|\bnormal\b)")
        .alias("ctrl_in_title")
    )

    # Check for control/normal in sample-level columns (source_name, tissue)
    sample_ctrl_cols = []
    for col in ["source_name", "tissue"]:
        col_ref = get_col(col)
        if col_ref in df.columns:
            df = df.with_columns(
                normalize_text_column(pl.col(col_ref))
                .str.contains(r"(?:\bctrl\b|\bcontrol\b|\bnormal\b)")
                .alias(f"ctrl_in_{col}")
            )
            sample_ctrl_cols.append(f"ctrl_in_{col}")

    if sample_ctrl_cols:
        df = df.with_columns(
            pl.any_horizontal([pl.col(c) for c in sample_ctrl_cols])
            .alias("ctrl_in_sample_cols")
        )
    else:
        df = df.with_columns(pl.lit(False).alias("ctrl_in_sample_cols"))

    # Determine regex label
    df = df.with_columns([
        # Sample-level columns say normal/control with no sample-level cancer
        pl.when(
            pl.col("ctrl_in_sample_cols") &
            (pl.col("n_sample_cancer") == 0)
        )
        .then(pl.lit(CL.LIKELY_NON_CANCER.value))
        .when(pl.col("onco_trap_in_sample_name"))
        .then(pl.lit(CL.UNCERTAIN_ONCO_TRAP.value))
        .when(
            (pl.col("n_sample_negative") >= 1) &
            (pl.col("n_sample_cancer") == 0)
        )
        .then(pl.lit(CL.LIKELY_NON_CANCER.value))
        .when(
            pl.col("ctrl_in_title") &
            (pl.col("n_sample_cancer") == 0) &
            (pl.col("n_cancer_mentions") >= 1)
        )
        .then(pl.lit(CL.LIKELY_NON_CANCER.value))
        .when(
            pl.col("cancer_in_sample_name") &
            (pl.col("n_cancer_mentions") >= 1) &
            (pl.col("n_negative_mentions") == 0)
        )
        .then(pl.lit(CL.CONFIDENT_CANCER.value))
        .when(
            (pl.col("cancer_in_sample_name") & (pl.col("n_negative_mentions") == 0)) |
            (pl.col("n_cancer_mentions") >= 2)
        )
        .then(pl.lit(CL.LIKELY_CANCER.value))
        .when(
            pl.col("negative_in_sample_name") |
            (pl.col("n_negative_mentions") >= 1)
        )
        .then(pl.lit(CL.LIKELY_NON_CANCER.value))
        .when(pl.col("n_cancer_mentions") == 1)
        .then(pl.lit(CL.UNCERTAIN_WEAK_SIGNAL.value))
        .otherwise(pl.lit(CL.UNCERTAIN_NO_SIGNAL.value))
        .alias("regex_label")
    ])

    # Build explanation
    df = df.with_columns(
        pl.concat_str([
            pl.when(pl.col("onco_trap_in_sample_name"))
            .then(pl.lit("onco-trap"))
            .otherwise(pl.lit("")),

            pl.when(pl.col("n_negative_mentions") > 0)
            .then(pl.lit(",neg-context"))
            .otherwise(pl.lit("")),

            pl.when(pl.col("n_cancer_mentions") >= 2)
            .then(pl.lit(",strong-cancer-signal"))
            .otherwise(
                pl.when(pl.col("n_cancer_mentions") == 1)
                .then(pl.lit(",weak-cancer-signal"))
                .otherwise(pl.lit(""))
            ),
        ], separator="")
        .str.replace_all(r"^,", "")
        .str.replace_all(r",$", "")
        .alias("regex_reason")
    )

    # Clean up temporary boolean columns
    temp_cols = (
        ["cancer_in_sample_name", "negative_in_sample_name", "onco_trap_in_sample_name",
         "n_cancer_mentions", "n_negative_mentions", "n_sample_negative",
         "n_sample_cancer", "ctrl_in_title", "ctrl_in_sample_cols"]
        + [f"ctrl_in_{c}" for c in ["source_name", "tissue"]]
        + cancer_cols
        + negative_cols
    )
    temp_cols = [c for c in temp_cols if c in df.columns]
    df = df.drop(temp_cols)

    return df
