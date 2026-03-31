"""
Cancer sample classification functions using regex patterns and MedSpaCy NLP.

This module is a backward-compatible facade that re-exports all public functions
from the split modules. Existing code that does `from functions import ...` will
continue to work unchanged.

Modules:
    preprocessing   - Text cleaning and normalization
    regex_classifier - Regex-based cancer pattern matching
    nlp_classifier  - MedSpaCy NLP classification and result resolution
    pipeline        - MedSpaCy pipeline singleton management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import polars as pl
from medspacy.ner import TargetRule

from config import (
    PRIORITY_COLS,
    ClassificationLabel as CL,
    MedSpaCyLabel as ML,
    DEFAULT_CONFIG,
)

# Re-export from preprocessing
from preprocessing import (
    clean_texts,
    normalize_text_column,
    identify_candidate_text_columns,
    preprocess_text_columns,
    get_text_column_stats,
)

# Re-export from regex_classifier
from regex_classifier import apply_regex_classification

# Re-export from nlp_classifier
from nlp_classifier import (
    _classify_doc,
    medspacy_classify_batch,
    resolve_uncertain,
)

# Re-export from pipeline
from pipeline import (
    get_default_target_rules,
    initialize_medspacy_pipeline,
    NLPPipelineManager,
    get_nlp,
    reset_nlp,
    generate_disease_rules,
)

if TYPE_CHECKING:
    from spacy.language import Language


# =============================================================================
# Resolution Attribution
# =============================================================================

def _determine_resolved_by(
    regex_label: str,
    med_label: str,
    confidence_category: str,
) -> str:
    """
    Determine which stage actually produced the final classification.

    Uses the raw inputs (regex_label, med_label) to attribute correctly,
    rather than just looking at the final confidence_category.
    """
    regex_uncertain = regex_label.startswith("uncertain")
    med_no_signal = med_label in (ML.NO_SIGNAL.value, "NO_SIGNAL")

    # Neither stage found anything → defaulted to non-cancer
    if regex_uncertain and med_no_signal:
        return "default"

    # Regex was uncertain, but MedSpaCy found something → MedSpaCy decided
    if regex_uncertain and not med_no_signal:
        return "medspacy"

    # Regex was confident, MedSpaCy had no signal → regex decided
    if not regex_uncertain and med_no_signal:
        return "regex"

    # Both stages had a signal → both contributed
    return "regex+medspacy"


# =============================================================================
# Main Classification Orchestrator
# =============================================================================

def classify_cancer_samples(
    df: pl.DataFrame,
    nlp_pipeline: Optional["Language"] = None,
    batch_size: int = DEFAULT_CONFIG.batch_size,
    use_normalized: bool = True,
    priority_cols: Tuple[str, ...] = PRIORITY_COLS,
    fallback_providers: Optional[List] = None,
) -> pl.DataFrame:
    """
    Classify samples as cancer/non-cancer using regex, MedSpaCy, and
    optional fallback providers.

    Stages:
    1. Regex-based pattern matching for quick filtering
    2. MedSpaCy NLP for context-aware classification
    3. (Optional) Fallback pipeline for no-signal samples:
       expanded search → API enrichment → LLM classification

    Args:
        df: Input DataFrame with text metadata columns.
        nlp_pipeline: Initialized MedSpaCy pipeline. If None, uses singleton.
        batch_size: Number of texts per MedSpaCy batch.
        use_normalized: Whether to use pre-normalized columns.
        priority_cols: Columns to search for cancer indicators.
        fallback_providers: Optional list of FallbackProvider instances.
            Each is tried in order for unresolved samples.

    Returns:
        DataFrame with added columns: regex_label, regex_reason,
        med_label, med_reason, confidence_category, resolved_by.
    """
    available_cols = [c for c in priority_cols if c in df.columns]

    # Stage 1: Regex classification
    df = apply_regex_classification(df, available_cols, use_normalized)

    # Stage 2: MedSpaCy classification
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()

    df = medspacy_classify_batch(
        df,
        nlp_pipeline=nlp_pipeline,
        batch_size=batch_size,
        priority_cols=available_cols,
        use_normalized=use_normalized,
    )

    # Combine regex + MedSpaCy results
    df = df.with_columns(
        pl.struct(["regex_label", "med_label", "med_source_columns"])
        .map_elements(
            lambda x: resolve_uncertain(
                x["regex_label"], x["med_label"], x.get("med_source_columns", "")
            ),
            return_dtype=pl.Utf8,
        )
        .alias("confidence_category")
    )

    # Add resolved_by tracking — based on actual inputs, not just final output
    df = df.with_columns(
        pl.struct(["regex_label", "med_label", "confidence_category"])
        .map_elements(
            lambda x: _determine_resolved_by(
                x["regex_label"], x["med_label"], x["confidence_category"]
            ),
            return_dtype=pl.Utf8,
        )
        .alias("resolved_by")
    )

    # Stage 3: Fallback pipeline for no-signal samples
    if fallback_providers:
        from fallback import FallbackPipeline
        pipeline = FallbackPipeline(fallback_providers)
        df = pipeline.process(df)

    return df


# Backward compatibility: expose the private function name used by old code
_apply_regex_classification = apply_regex_classification


# Backward compatibility: medspacy_classify (now removed, but provide stub)
def medspacy_classify(row_texts, nlp_pipeline=None):
    """Deprecated: Use the pipeline directly via classify_cancer_samples()."""
    import warnings
    warnings.warn(
        "medspacy_classify() is deprecated. Use classify_cancer_samples() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()
    doc = nlp_pipeline(row_texts or "")
    return _classify_doc(doc)