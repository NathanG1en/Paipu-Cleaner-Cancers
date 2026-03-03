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
from functools import lru_cache
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
    CANCER_RULE_DEFINITIONS,
    NON_CANCER_RULE_DEFINITIONS,
    ClassifierConfig,
    DEFAULT_CONFIG,
    CONTEXT_RULE_DEFINITIONS,
)

import medspacy
from medspacy.context import ConTextRule

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


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
        - reason: Explanation of the classification WITH ACTUAL TERMS
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
    affirmed_cancer_terms: List[str] = []
    negated_cancer_terms: List[str] = []
    non_cancer_terms: List[str] = []

    for ent in doc.ents:
        ent_info = {
            "text": ent.text,
            "label": ent.label_,
            "is_negated": getattr(ent._, "is_negated", False),
            "is_hypothetical": getattr(ent._, "is_hypothetical", False),
            "is_historical": getattr(ent._, "is_family", False),
        }
        entities.append(ent_info)

        # Only count non-hypothetical, non-family mentions
        if ent_info["is_hypothetical"] or ent_info["is_family"]:
            continue

        if ent.label_ in ("CANCER", "CANCER_TYPE"):
            if ent_info["is_negated"]:
                negated_cancer_terms.append(ent.text)
            else:
                affirmed_cancer_terms.append(ent.text)
        elif ent.label_ == "NON_CANCER":
            non_cancer_terms.append(ent.text)

    # Determine label WITH ACTUAL TERMS
    has_affirmed = len(affirmed_cancer_terms) > 0
    has_negated = len(negated_cancer_terms) > 0
    has_non_cancer = len(non_cancer_terms) > 0

    if has_affirmed and not has_negated and not has_non_cancer:
        # Pure cancer signal
        terms_str = ", ".join(set(affirmed_cancer_terms))
        label = "CANCER"
        reason = f"cancer_terms: {terms_str}"

    elif has_negated and not has_affirmed:
        # Pure non-cancer signal
        terms_list = negated_cancer_terms + non_cancer_terms
        terms_str = ", ".join(set(terms_list))
        label = "NON_CANCER"
        reason = f"negated/non-cancer_terms: {terms_str}"

    elif has_affirmed and (has_negated or has_non_cancer):
        # Mixed signals - affirmed takes precedence
        affirmed_str = ", ".join(set(affirmed_cancer_terms))
        negated_str = ", ".join(set(negated_cancer_terms + non_cancer_terms))
        label = "CANCER"
        reason = f"mixed (affirmed: {affirmed_str}; negated: {negated_str})"

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
    Now tracks which columns contain the matched terms.
    """
    rows_as_dicts = df.to_dicts()

    labels: List[str] = []
    reasons: List[str] = []
    source_cols: List[str] = []  # NEW: Track source columns

    for row in rows_as_dicts:
        # Process each column individually to track sources
        all_results = []
        found_in_cols = []

        suffix = "_norm" if use_normalized else ""
        for col in priority_cols:
            col_key = f"{col}{suffix}"
            text = row.get(col_key) or row.get(col) or ""

            if isinstance(text, str) and text.strip() and _has_alphabetic(text):
                doc = nlp_pipeline(text.strip().lower())
                # Check if this doc has cancer entities
                for ent in doc.ents:
                    if ent.label_ in ("CANCER", "NON_CANCER") and not getattr(ent._, "is_negated", False):
                        found_in_cols.append(f"{col}:{ent.text}")

        # Now process combined text for overall classification
        combined_text = clean_texts(row, tuple(priority_cols), use_normalized)
        doc = nlp_pipeline(combined_text)
        result = _classify_doc(doc)

        labels.append(result["label"])

        # Enhance reason with column info
        if found_in_cols:
            col_info = " | found_in: " + ", ".join(set(found_in_cols[:5]))  # Limit to 5
            reasons.append(result["reason"] + col_info)
            source_cols.append(", ".join(set([c.split(":")[0] for c in found_in_cols])))
        else:
            reasons.append(result["reason"])
            source_cols.append("")

    # Add results to DataFrame
    df = df.with_columns([
        pl.Series("med_label", labels),
        pl.Series("med_reason", reasons),
        pl.Series("med_source_columns", source_cols),  # NEW COLUMN
    ])

    return df


def _classify_doc(doc: "Doc") -> Dict[str, str]:
    """
    Classify a single spaCy Doc based on medspacy entities.

    Returns:
        dict with keys: label, reason, entities (list of dicts)
    """
    cancer_count = 0
    non_cancer_count = 0
    negated_cancer_count = 0  # Track negated cancer terms
    entities = []

    # Track actual terms found
    cancer_terms = []
    non_cancer_terms = []
    negated_cancer_terms = []

    for ent in doc.ents:
        is_negated = getattr(ent._, "is_negated", False)

        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "is_negated": is_negated,
        })

        # Count based on label and negation status
        if ent.label_ == "CANCER":
            if is_negated:
                negated_cancer_count += 1
                negated_cancer_terms.append(ent.text)
            else:
                cancer_count += 1
                cancer_terms.append(ent.text)
        elif ent.label_ == "NON_CANCER":
            non_cancer_count += 1
            non_cancer_terms.append(ent.text)

    # Classification logic with negation awareness
    # Priority 1: If we have negated cancer terms, likely non-cancer
    if negated_cancer_count > 0 and cancer_count == 0:
        terms_str = ", ".join(set(negated_cancer_terms))  # Use set to deduplicate
        return {
            "label": "NON_CANCER",
            "reason": f"negated_cancer_terms: {terms_str}",
            "entities": entities,
        }

    # Priority 2: If we have more negations than affirmed cancer terms
    if negated_cancer_count > cancer_count:
        neg_terms = ", ".join(set(negated_cancer_terms))
        affirm_terms = ", ".join(set(cancer_terms)) if cancer_terms else "none"
        return {
            "label": "NON_CANCER",
            "reason": f"negation_dominant (negated: {neg_terms}; affirmed: {affirm_terms})",
            "entities": entities,
        }

    # Priority 3: Affirmed cancer terms (after negation check)
    if cancer_count > 0:
        terms_str = ", ".join(set(cancer_terms))
        return {
            "label": "CANCER",
            "reason": f"cancer_terms: {terms_str}",
            "entities": entities,
        }

    # Priority 4: Non-cancer indicators
    if non_cancer_count > 0:
        terms_str = ", ".join(set(non_cancer_terms))
        return {
            "label": "NON_CANCER",
            "reason": f"non_cancer_terms: {terms_str}",
            "entities": entities,
        }

    # No signal
    return {
        "label": "NO_SIGNAL",
        "reason": "no_relevant_terms",
        "entities": entities,
    }


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
# MedSpaCy Pipeline Setup - Singleton Pattern
# =============================================================================

@lru_cache(maxsize=1)
def get_default_target_rules() -> Tuple[Tuple[TargetRule, ...], Tuple[TargetRule, ...]]:
    """
    Get default TargetRules for cancer and non-cancer entity detection.

    This function is cached to avoid recreating rules on every call.

    Returns:
        Tuple of (cancer_rules, non_cancer_rules) where each is a tuple
        of TargetRule objects for the MedSpaCy target matcher.
    """
    cancer_rules: List[TargetRule] = []
    for literal, category, pattern in CANCER_RULE_DEFINITIONS:
        if pattern:
            cancer_rules.append(TargetRule(literal, category, pattern=pattern))
        else:
            cancer_rules.append(TargetRule(literal, category))

    non_cancer_rules: List[TargetRule] = []
    for literal, category, pattern in NON_CANCER_RULE_DEFINITIONS:
        if pattern:
            non_cancer_rules.append(TargetRule(literal, category, pattern=pattern))
        else:
            non_cancer_rules.append(TargetRule(literal, category))

    # Return as tuples for hashability (needed for lru_cache)
    return tuple(cancer_rules), tuple(non_cancer_rules)


def initialize_medspacy_pipeline(
    cancer_rules: List[TargetRule],
    non_cancer_rules: List[TargetRule],
) -> "Language":
    """
    Initialize a MedSpaCy pipeline with:
        - medspacy_target_matcher (entity detection)
        - medspacy_context (negation/context detection)

    Args:
        cancer_rules: List of TargetRule for cancer terms
        non_cancer_rules: List of TargetRule for non-cancer terms

    Returns:
        Configured MedSpaCy Language pipeline
    """
    import medspacy
    from medspacy.context import ConTextRule

    nlp = medspacy.load(enable=["medspacy_target_matcher", "medspacy_context"])

    # Add cancer and non-cancer target rules
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(cancer_rules + non_cancer_rules)

    # Get the context component and add custom negation rules from config
    context = nlp.get_pipe("medspacy_context")

    # Build ConTextRule objects from config definitions
    custom_negation_rules = [
        ConTextRule(literal, category, direction=direction)
        for literal, category, direction in CONTEXT_RULE_DEFINITIONS
    ]

    context.add(custom_negation_rules)

    return nlp


class NLPPipelineManager:
    """
    Singleton manager for the MedSpaCy NLP pipeline.

    This class provides thread-safe lazy initialization of the NLP pipeline
    and allows for customization with additional rules.

    Usage:
        # Get the default pipeline
        nlp = NLPPipelineManager.get_pipeline()

        # Get a pipeline with custom rules
        nlp = NLPPipelineManager.get_pipeline(additional_rules=[...])

        # Reset the pipeline (e.g., to add new rules)
        NLPPipelineManager.reset()
    """

    _instance: Optional["Language"] = None
    _additional_rules: List[TargetRule] = []

    @classmethod
    def get_pipeline(
        cls,
        additional_rules: Optional[List[TargetRule]] = None,
    ) -> "Language":
        """
        Get or create the singleton NLP pipeline.

        Args:
            additional_rules: Extra TargetRules to add to the pipeline.
                             Only applied on first initialization or after reset().

        Returns:
            Initialized MedSpaCy Language pipeline.
        """
        if cls._instance is None:
            cancer_rules, non_cancer_rules = get_default_target_rules()
            cls._instance = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)

            # Add any additional rules
            if additional_rules:
                cls._additional_rules = additional_rules
                target_matcher = cls._instance.get_pipe("medspacy_target_matcher")
                target_matcher.add(additional_rules)

        return cls._instance

    @classmethod
    def add_rules(cls, rules: List[TargetRule]) -> None:
        """
        Add rules to the existing pipeline.

        Args:
            rules: TargetRules to add.
        """
        pipeline = cls.get_pipeline()
        target_matcher = pipeline.get_pipe("medspacy_target_matcher")
        target_matcher.add(rules)
        cls._additional_rules.extend(rules)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton pipeline.

        Call this to force re-initialization on next get_pipeline() call.
        """
        cls._instance = None
        cls._additional_rules = []

    @classmethod
    def get_rule_count(cls) -> int:
        """Get the number of rules in the current pipeline."""
        if cls._instance is None:
            return 0
        target_matcher = cls._instance.get_pipe("medspacy_target_matcher")
        return len(target_matcher.rules)


def get_nlp(additional_rules: Optional[List[TargetRule]] = None) -> "Language":
    """
    Get the singleton MedSpaCy pipeline.

    This is the recommended way to get the NLP pipeline for classification.
    The pipeline is lazily initialized on first call and reused thereafter.

    Args:
        additional_rules: Extra TargetRules to add on first initialization.

    Returns:
        Initialized MedSpaCy Language pipeline.

    Example:
        # Simple usage
        nlp = get_nlp()
        doc = nlp("breast cancer tissue sample")

        # With additional rules
        from medspacy.ner import TargetRule
        custom_rules = [TargetRule("my_cancer_type", "CANCER_TYPE")]
        nlp = get_nlp(additional_rules=custom_rules)
    """
    return NLPPipelineManager.get_pipeline(additional_rules=additional_rules)


def reset_nlp() -> None:
    """
    Reset the singleton NLP pipeline.

    Use this if you need to reinitialize the pipeline with different rules.
    """
    NLPPipelineManager.reset()


def generate_disease_rules(
    unique_diseases: List[Optional[str]],
    nlp: "Language",
    existing_rules: Union[List[TargetRule], Tuple[TargetRule, ...]],
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