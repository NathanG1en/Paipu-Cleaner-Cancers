# text_column_processing.py - Text column preprocessing module

import polars as pl
from typing import Dict, List, Tuple

from config import ClassifierConfig, TextColumnConfig, DEFAULT_CONFIG


def normalize_text_column(col_expr: pl.Expr) -> pl.Expr:
    """
    Aggressive text normalization - applied once, reused everywhere.
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


def identify_viable_text_columns(
    df: pl.DataFrame,
    config: ClassifierConfig = DEFAULT_CONFIG,
) -> Dict[str, List[str]]:
    """
    Identify text columns organized by tier.
    
    Returns:
        {
            "priority": [...],    # Tier 1 - always use
            "secondary": [...],   # Tier 2 - use if present  
            "discovered": [...],  # Tier 3 - auto-discovered
        }
    """
    result: Dict[str, List[str]] = {"priority": [], "secondary": [], "discovered": []}
    n_rows = len(df)
    
    # Tier 1: Priority columns (if they exist)
    for col in config.priority_cols:
        if col in df.columns and df[col].dtype == pl.Utf8:
            result["priority"].append(col)
    
    # Tier 2: Secondary columns (if they exist)
    for col in config.secondary_cols:
        if col in df.columns and df[col].dtype == pl.Utf8:
            if col not in result["priority"]:
                result["secondary"].append(col)
    
    # Tier 3: Auto-discover additional viable columns
    known_cols = set(result["priority"] + result["secondary"])
    
    for col in df.columns:
        if col in known_cols:
            continue
        if df[col].dtype != pl.Utf8:
            continue
        
        # Skip excluded patterns
        col_lower = col.lower()
        if any(p in col_lower for p in config.exclude_patterns):
            continue
        
        # Check viability thresholds
        non_null_count = df[col].drop_nulls().len()
        non_null_pct = non_null_count / n_rows if n_rows > 0 else 0
        
        if non_null_pct < config.min_non_null_pct:
            continue
        
        avg_len = df.select(pl.col(col).drop_nulls().str.len_chars().mean()).item()
        if avg_len is None or avg_len < config.min_avg_length:
            continue
        
        result["discovered"].append(col)
    
    return result


def preprocess_dataframe(
    df: pl.DataFrame,
    config: ClassifierConfig = DEFAULT_CONFIG,
    include_discovered: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, List[str]]]:
    """
    Full preprocessing pipeline - run once at load time.
    
    Args:
        df: Raw DataFrame
        config: Column configuration
        include_discovered: Whether to include auto-discovered columns
        
    Returns:
        (preprocessed_df, column_info)
        
    Column naming: original columns unchanged, normalized versions get "_norm" suffix
    """
    # Step 1: Identify columns
    col_tiers = identify_viable_text_columns(df, config)
    
    # Step 2: Determine which columns to normalize
    cols_to_normalize = col_tiers["priority"] + col_tiers["secondary"]
    if include_discovered:
        cols_to_normalize += col_tiers["discovered"]
    
    # Step 3: Apply normalization (vectorized, single pass)
    if cols_to_normalize:
        df = df.with_columns([
            normalize_text_column(pl.col(col)).alias("{}_norm".format(col))
            for col in cols_to_normalize
        ])
    
    # Track which normalized columns were created
    col_tiers["normalized"] = ["{}_norm".format(c) for c in cols_to_normalize]
    
    return df, col_tiers
