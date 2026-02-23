"""
Cancer sample classification functions using regex patterns and MedSpaCy NLP.

This module provides utilities for:
- Text preprocessing and normalization
- Cancer/non-cancer classification via regex patterns
- Clinical NLP-based classification using MedSpaCy
- Automatic rule generation from disease metadata
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import polars as pl
from medspacy.ner import TargetRule

from config import (
    PRIORITY_COLS,
    CANCER_POS,
    CANCER_NEG,
    ONCO_TRAPS,
    CANCER_KEYWORDS,
    SPECIFIC_CANCER_TYPES,
    ClassifierConfig,
    DEFAULT_CONFIG,
)

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

# =============================================================================
# Module-level state
# =============================================================================

# Global NLP pipeline (lazy-loaded)
_nlp: Optional["Language"] = None


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
        # Try normalized column first if requested
        col_name = "{}_norm".format(col) if use_normalized else col
        if col_name not in row and use_normalized:
            col_name = col  # Fallback to original
            
        val = row.get(col_name)
        if val is None or (isinstance(val, str) and val.lower() in ("", "nan", "none")):
            continue
            
        text = str(val).strip()
        if text:
            texts.append(text)
    
    combined = " ".join(texts)
    
    # Normalize whitespace
    combined = re.sub(r"\s+", " ", combined).strip()
    
    return combined


def _has_alphabetic(text: str) -> bool:
    """
    Check if text contains at least one alphabetic character.
    
    Args:
        text: Input string to check.
        
    Returns:
        True if text contains alphabetic characters, False otherwise.
    """
    return bool(re.search(r"[a-zA-Z]", text))


def normalize_text_column(col_expr: pl.Expr) -> pl.Expr:
    """
    Normalize a Polars text column expression for consistent comparison.
    
    Applies:
    - UTF-8 casting
    - Null filling with empty string
    - Lowercase conversion
    - Separator normalization (_, /, |, \\ -> space)
    - Whitespace collapsing
    - Edge trimming
    
    Args:
        col_expr: Polars column expression to normalize.
        
    Returns:
        Normalized Polars expression.
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
# Classification Functions
# =============================================================================

def classify_cancer_samples(
    df: pl.DataFrame,
    nlp_pipeline: Optional["Language"] = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    use_normalized: bool = True,
    priority_cols: Tuple[str, ...] = PRIORITY_COLS,
) -> pl.DataFrame:
    """
    Classify samples as cancer/non-cancer/uncertain using regex and MedSpaCy.
    
    This function performs a two-stage classification:
    1. Regex-based pattern matching for quick filtering
    2. MedSpaCy NLP for context-aware classification
    
    Args:
        df: Input DataFrame with text metadata columns.
        nlp_pipeline: Initialized MedSpaCy Language pipeline. If None, uses global.
        batch_size: Number of texts to process per MedSpaCy batch.
        use_normalized: Whether to use pre-normalized columns (with "_norm" suffix).
        priority_cols: Columns to search for cancer indicators.
        
    Returns:
        DataFrame with added columns:
        - regex_label: Label from regex classification
        - regex_reason: Explanation for regex label
        - med_label: Label from MedSpaCy classification
        - med_reason: Explanation for MedSpaCy label
        - confidence_category: Final combined classification
    """
    # Ensure priority columns exist
    available_cols = [c for c in priority_cols if c in df.columns]
    
    # Stage 1: Regex-based classification
    df = _apply_regex_classification(df, available_cols, use_normalized)
    
    # Stage 2: MedSpaCy classification for uncertain samples
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()
    
    df = medspacy_classify_batch(
        df,
        nlp_pipeline=nlp_pipeline,
        batch_size=batch_size,
        priority_cols=available_cols,
        use_normalized=use_normalized,
    )
    
    # Combine results
    df = df.with_columns(
        pl.struct(["regex_label", "med_label"])
        .map_elements(
            lambda x: resolve_uncertain(x["regex_label"], x["med_label"]),
            return_dtype=pl.Utf8,
        )
        .alias("confidence_category")
    )
    
    return df


def _apply_regex_classification(
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
    
    # Build column references
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
    
    # Determine regex label
    df = df.with_columns([
        pl.when(pl.col("onco_trap_in_sample_name"))
        .then(pl.lit("uncertain_onco_trap"))
        .when(
            pl.col("cancer_in_sample_name") &
            (pl.col("n_cancer_mentions") >= 1) &
            (pl.col("n_negative_mentions") == 0)
        )
        .then(pl.lit("confident_cancer"))
        .when(
            (pl.col("cancer_in_sample_name") & (pl.col("n_negative_mentions") == 0)) |
            (pl.col("n_cancer_mentions") >= 2)
        )
        .then(pl.lit("likely_cancer"))
        .when(
            pl.col("negative_in_sample_name") |
            (pl.col("n_negative_mentions") >= 1)
        )
        .then(pl.lit("likely_non_cancer"))
        .when(pl.col("n_cancer_mentions") == 1)
        .then(pl.lit("uncertain_weak_signal"))
        .otherwise(pl.lit("uncertain_no_signal"))
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
    
    return df


def medspacy_classify(
    row_texts: str,
    nlp_pipeline: Optional["Language"] = None,
) -> Dict[str, Any]:
    """
    Classify a single text using MedSpaCy for negation and context detection.
    
    Args:
        row_texts: Combined text string to classify.
        nlp_pipeline: Initialized MedSpaCy pipeline. Uses global if None.
        
    Returns:
        Dictionary with keys:
        - label: "CANCER", "NON_CANCER", or "NO_SIGNAL"
        - reason: Explanation of the classification
        - entities: List of detected entity dicts
    """
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()
    
    if not row_texts or not _has_alphabetic(row_texts):
        return {
            "label": "NO_SIGNAL",
            "reason": "empty or no alphabetic content",
            "entities": [],
        }
    
    doc = nlp_pipeline(row_texts)
    
    entities: List[Dict[str, Any]] = []
    has_affirmed_cancer = False
    has_negated_cancer = False
    
    for ent in doc.ents:
        ent_info = {
            "text": ent.text,
            "label": ent.label_,
            "is_negated": getattr(ent._, "is_negated", False),
            "is_hypothetical": getattr(ent._, "is_hypothetical", False),
            "is_historical": getattr(ent._, "is_historical", False),
            "is_family": getattr(ent._, "is_family", False),
        }
        entities.append(ent_info)
        
        # Only count non-hypothetical, non-family mentions
        if ent_info["is_hypothetical"] or ent_info["is_family"]:
            continue
            
        if ent.label_ in ("CANCER", "CANCER_TYPE"):
            if ent_info["is_negated"]:
                has_negated_cancer = True
            else:
                has_affirmed_cancer = True
        elif ent.label_ == "NON_CANCER":
            has_negated_cancer = True
    
    # Determine label
    if has_affirmed_cancer and not has_negated_cancer:
        label = "CANCER"
        reason = "affirmed cancer entity detected"
    elif has_negated_cancer and not has_affirmed_cancer:
        label = "NON_CANCER"
        reason = "only negated/non-cancer entities"
    elif has_affirmed_cancer and has_negated_cancer:
        label = "CANCER"  # Affirmed takes precedence
        reason = "mixed signals, affirmed cancer present"
    else:
        label = "NO_SIGNAL"
        reason = "no cancer-related entities detected"
    
    return {
        "label": label,
        "reason": reason,
        "entities": entities,
    }


def medspacy_classify_batch(
    df: pl.DataFrame,
    nlp_pipeline: "Language",
    batch_size: int = DEFAULT_CONFIG.batch_size,
    priority_cols: Union[Tuple[str, ...], List[str]] = PRIORITY_COLS,
    use_normalized: bool = True,
) -> pl.DataFrame:
    """
    Apply MedSpaCy classification to a DataFrame in batches.
    
    Args:
        df: Input DataFrame.
        nlp_pipeline: Initialized MedSpaCy Language pipeline.
        batch_size: Number of documents per batch (for nlp.pipe()).
        priority_cols: Columns to extract text from.
        use_normalized: Whether to use "_norm" suffixed columns.
        
    Returns:
        DataFrame with med_label and med_reason columns added.
    """
    # Prepare text for each row
    rows_as_dicts = df.to_dicts()
    texts: List[str] = [
        clean_texts(row, tuple(priority_cols), use_normalized)
        for row in rows_as_dicts
    ]
    
    # Process in batches using nlp.pipe for efficiency
    labels: List[str] = []
    reasons: List[str] = []
    
    for doc in nlp_pipeline.pipe(texts, batch_size=batch_size):
        result = _classify_doc(doc)
        labels.append(result["label"])
        reasons.append(result["reason"])
    
    # Add results to DataFrame
    df = df.with_columns([
        pl.Series("med_label", labels),
        pl.Series("med_reason", reasons),
    ])
    
    return df


def _classify_doc(doc: "Doc") -> Dict[str, str]:
    """
    Classify a processed spaCy Doc object.
    
    Args:
        doc: Processed MedSpaCy document.
        
    Returns:
        Dictionary with "label" and "reason" keys.
    """
    has_affirmed_cancer = False
    has_negated_cancer = False
    
    for ent in doc.ents:
        is_negated = getattr(ent._, "is_negated", False)
        is_hypothetical = getattr(ent._, "is_hypothetical", False)
        is_family = getattr(ent._, "is_family", False)
        
        if is_hypothetical or is_family:
            continue
        
        if ent.label_ in ("CANCER", "CANCER_TYPE"):
            if is_negated:
                has_negated_cancer = True
            else:
                has_affirmed_cancer = True
        elif ent.label_ == "NON_CANCER":
            has_negated_cancer = True
    
    if has_affirmed_cancer and not has_negated_cancer:
        return {"label": "CANCER", "reason": "affirmed cancer entity"}
    elif has_negated_cancer and not has_affirmed_cancer:
        return {"label": "NON_CANCER", "reason": "negated/non-cancer only"}
    elif has_affirmed_cancer and has_negated_cancer:
        return {"label": "CANCER", "reason": "mixed, affirmed present"}
    else:
        return {"label": "NO_SIGNAL", "reason": "no cancer entities"}


def resolve_uncertain(
    regex_label: Optional[str],
    med_label: Optional[str],
) -> str:
    """
    Resolve final classification by combining regex and MedSpaCy results.
    
    Priority logic:
    1. Confident regex labels take precedence
    2. MedSpaCy can upgrade uncertain cases
    3. Fallback to regex label
    
    Args:
        regex_label: Classification from regex stage.
        med_label: Classification from MedSpaCy stage.
        
    Returns:
        Final confidence category string.
    """
    regex_label = regex_label or "uncertain_no_signal"
    med_label = med_label or "NO_SIGNAL"
    
    # High confidence regex results
    if regex_label == "confident_cancer":
        return "confident_cancer"
    
    if regex_label == "likely_non_cancer":
        # Check if MedSpaCy found cancer
        if med_label == "CANCER":
            return "confirmed_by_medspacy"
        return "confirmed_non_cancer"
    
    # Likely cancer - verify with MedSpaCy
    if regex_label == "likely_cancer":
        if med_label == "CANCER":
            return "likely_cancer"
        elif med_label == "NON_CANCER":
            return "likely_non_cancer"
        return "likely_cancer"  # Trust regex
    
    # Uncertain cases - rely on MedSpaCy
    if regex_label.startswith("uncertain"):
        if med_label == "CANCER":
            return "confirmed_by_medspacy"
        elif med_label == "NON_CANCER":
            return "likely_non_cancer"
        return regex_label  # Keep uncertain
    
    return regex_label


# =============================================================================
# MedSpaCy Pipeline Setup
# =============================================================================

def get_default_target_rules() -> Tuple[List[TargetRule], List[TargetRule]]:
    """
    Get default TargetRules for cancer and non-cancer entity detection.
    
    Returns:
        Tuple of (cancer_rules, non_cancer_rules) where each is a list
        of TargetRule objects for the MedSpaCy target matcher.
    """
    cancer_rules: List[TargetRule] = [
        # General cancer terms
        TargetRule("cancer", "CANCER", pattern=r"\bcancers?\b"),
        TargetRule("tumor", "CANCER", pattern=r"\btumou?rs?\b"),
        TargetRule("malignant", "CANCER", pattern=r"\bmalignan(?:t|cy)\b"),
        TargetRule("carcinoma", "CANCER", pattern=r"\bcarcinomas?\b"),
        TargetRule("neoplasm", "CANCER", pattern=r"\bneoplasms?\b"),
        TargetRule("metastasis", "CANCER", pattern=r"\bmetasta(?:s|t)(?:is|es)?\b"),
        TargetRule("adenocarcinoma", "CANCER_TYPE", pattern=r"\badenocarcinomas?\b"),
        TargetRule("sarcoma", "CANCER_TYPE", pattern=r"\bsarcomas?\b"),
        TargetRule("leukemia", "CANCER_TYPE", pattern=r"\bleuk[ae]mias?\b"),
        TargetRule("lymphoma", "CANCER_TYPE", pattern=r"\blymphomas?\b"),
        TargetRule("glioblastoma", "CANCER_TYPE", pattern=r"\bglioblastomas?\b"),
        TargetRule("melanoma", "CANCER_TYPE", pattern=r"\bmelanomas?\b"),
        TargetRule("myeloma", "CANCER_TYPE", pattern=r"\bmyelomas?\b"),
        TargetRule("neuroblastoma", "CANCER_TYPE", pattern=r"\bneuroblastomas?\b"),
        TargetRule("hepatocellular carcinoma", "CANCER_TYPE"),
        TargetRule("breast cancer", "CANCER_TYPE"),
        TargetRule("lung cancer", "CANCER_TYPE"),
        TargetRule("colon cancer", "CANCER_TYPE"),
        TargetRule("prostate cancer", "CANCER_TYPE"),
        TargetRule("pancreatic cancer", "CANCER_TYPE"),
        TargetRule("ovarian cancer", "CANCER_TYPE"),
        TargetRule("bladder cancer", "CANCER_TYPE"),
        TargetRule("skin cancer", "CANCER_TYPE"),
        TargetRule("brain cancer", "CANCER_TYPE"),
        TargetRule("liver cancer", "CANCER_TYPE"),
        TargetRule("kidney cancer", "CANCER_TYPE"),
        TargetRule("renal cell carcinoma", "CANCER_TYPE"),
        TargetRule("squamous cell carcinoma", "CANCER_TYPE"),
        TargetRule("basal cell carcinoma", "CANCER_TYPE"),
        TargetRule("non-small cell lung cancer", "CANCER_TYPE"),
        TargetRule("small cell lung cancer", "CANCER_TYPE"),
        TargetRule("triple negative breast cancer", "CANCER_TYPE"),
        TargetRule("HER2 positive", "CANCER_TYPE"),
        TargetRule("ER positive", "CANCER_TYPE"),
        TargetRule("oncogenic", "CANCER", pattern=r"\boncogen(?:ic|e|es)\b"),
    ]
    
    non_cancer_rules: List[TargetRule] = [
        TargetRule("normal", "NON_CANCER", pattern=r"\bnormal\b"),
        TargetRule("healthy", "NON_CANCER", pattern=r"\bhealthy\b"),
        TargetRule("control", "NON_CANCER", pattern=r"\b(?:ctrl|control)\b"),
        TargetRule("benign", "NON_CANCER", pattern=r"\bbenign\b"),
        TargetRule("non-tumor", "NON_CANCER", pattern=r"\bnon[-\s]?tumou?r(?:al)?\b"),
        TargetRule("non-cancer", "NON_CANCER", pattern=r"\bnon[-\s]?cancer(?:ous)?\b"),
        TargetRule("adjacent normal", "NON_CANCER"),
        TargetRule("tumor-adjacent normal", "NON_CANCER"),
        TargetRule("matched normal", "NON_CANCER"),
        TargetRule("sham", "NON_CANCER", pattern=r"\bsham\b"),
        TargetRule("unaffected", "NON_CANCER", pattern=r"\bunaffected\b"),
        TargetRule("wild type", "NON_CANCER", pattern=r"\bwild[-\s]?type\b"),
        TargetRule("WT", "NON_CANCER", pattern=r"\bWT\b"),
        # Onco-traps (false positives)
        TargetRule("oncorhynchus", "NON_CANCER", pattern=r"\boncorhynchus\b"),
        TargetRule("oncophora", "NON_CANCER", pattern=r"\boncophora\b"),
        TargetRule("oncotic", "NON_CANCER", pattern=r"\boncotic\b"),
        TargetRule("oncomodulin", "NON_CANCER", pattern=r"\boncomodulin\b"),
    ]
    
    return cancer_rules, non_cancer_rules


def initialize_medspacy_pipeline(
    *rule_lists: List[TargetRule],
) -> "Language":
    """
    Initialize and configure a MedSpaCy pipeline for cancer classification.
    
    Args:
        *rule_lists: Variable number of TargetRule lists to add to the pipeline.
        
    Returns:
        Configured spaCy Language pipeline with:
        - medspacy_target_matcher (entity detection)
        - medspacy_context (negation/context detection)
    """
    import medspacy
    
    nlp = medspacy.load(enable=["medspacy_target_matcher", "medspacy_context"])
    
    # Add target rules
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    for rule_list in rule_lists:
        if rule_list:
            target_matcher.add(rule_list)
    
    return nlp


def generate_disease_rules(
    unique_diseases: List[Optional[str]],
    nlp: "Language",
    existing_rules: List[TargetRule],
) -> Tuple[List[TargetRule], List[str]]:
    """
    Auto-generate TargetRules from unique disease values in metadata.
    
    Analyzes disease strings that aren't already covered by existing rules
    and creates new rules for cancer-related diseases.
    
    Args:
        unique_diseases: List of unique disease values from the dataset.
        nlp: Initialized MedSpaCy pipeline for checking existing coverage.
        existing_rules: Rules already in the pipeline (to avoid duplicates).
        
    Returns:
        Tuple of:
        - new_rules: List of auto-generated TargetRule objects
        - skipped: List of disease strings that were skipped (already covered or non-cancer)
    """
    new_rules: List[TargetRule] = []
    skipped: List[str] = []
    
    # Get existing rule literals for comparison
    existing_literals = {
        rule.literal.lower() 
        for rule in existing_rules 
        if hasattr(rule, "literal") and rule.literal
    }
    
    for disease in unique_diseases:
        if disease is None or not isinstance(disease, str):
            continue
            
        disease_clean = disease.strip().lower()
        
        if not disease_clean or disease_clean in ("nan", "none", "na", "n/a"):
            continue
        
        # Skip if already covered
        if disease_clean in existing_literals:
            skipped.append("{} (already exists)".format(disease))
            continue
        
        # Check if this looks like a cancer
        is_cancer_related = any(kw in disease_clean for kw in CANCER_KEYWORDS)
        
        if is_cancer_related:
            # Determine label based on specificity
            if any(kw in disease_clean for kw in SPECIFIC_CANCER_TYPES):
                label = "CANCER_TYPE"
            else:
                label = "CANCER"
            
            new_rules.append(TargetRule(disease.strip(), label))
        else:
            skipped.append("{} (not cancer-related)".format(disease))
    
    return new_rules, skipped


def get_nlp() -> "Language":
    """
    Get or initialize the global MedSpaCy pipeline.
    
    Returns:
        Initialized MedSpaCy Language pipeline.
    """
    global _nlp
    
    if _nlp is None:
        cancer_rules, non_cancer_rules = get_default_target_rules()
        _nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)
    
    return _nlp


# =============================================================================
# Text Column Analysis Functions
# =============================================================================

def identify_candidate_text_columns(
    df: pl.DataFrame,
    config: ClassifierConfig = DEFAULT_CONFIG,
) -> List[str]:
    """
    Identify DataFrame columns that are good candidates for text analysis.
    
    Args:
        df: Input DataFrame.
        config: Configuration with thresholds and exclusion patterns.
        
    Returns:
        List of column names suitable for text analysis.
    """
    candidates: List[str] = []
    n_rows = len(df)
    
    for col in df.columns:
        if df[col].dtype != pl.Utf8:
            continue
        
        # Check exclusion patterns
        col_lower = col.lower()
        if any(p in col_lower for p in config.exclude_patterns):
            continue
        
        # Check non-null percentage
        non_null_count = df[col].drop_nulls().len()
        non_null_pct = non_null_count / n_rows if n_rows > 0 else 0
        
        if non_null_pct < config.min_non_null_pct:
            continue
        
        # Check average length
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
    """
    Normalize specified text columns and add as new columns with suffix.
    
    Args:
        df: Input DataFrame.
        columns: Columns to normalize. If None, uses PRIORITY_COLS.
        suffix: Suffix for normalized column names.
        
    Returns:
        DataFrame with additional normalized columns.
    """
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
    """
    Get statistics for text columns to assess their quality.
    
    Args:
        df: Input DataFrame.
        columns: Columns to analyze. If None, analyzes all Utf8 columns.
        
    Returns:
        DataFrame with columns: column_name, non_null_count, non_null_pct,
        avg_length, unique_count
    """
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