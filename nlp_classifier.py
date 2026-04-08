"""
MedSpaCy-based NLP classification for cancer detection.

Provides context-aware classification using MedSpaCy for negation detection
and entity recognition, plus the resolution logic that combines regex and
NLP results into a final classification.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import polars as pl

from config import (
    PRIORITY_COLS,
    ClassificationLabel as CL,
    MedSpaCyLabel as ML,
)
from preprocessing import clean_texts, _has_alphabetic

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc


# =============================================================================
# MedSpaCy Classification
# =============================================================================

def _classify_doc(doc: "Doc") -> Dict[str, str]:
    """
    Classify a single spaCy Doc based on medspacy entities.

    Returns:
        dict with keys: label, reason, entities (list of dicts)
    """
    cancer_count = 0
    non_cancer_count = 0
    negated_cancer_count = 0
    entities = []

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
    if negated_cancer_count > 0 and cancer_count == 0:
        terms_str = ", ".join(set(negated_cancer_terms))
        return {
            "label": ML.NON_CANCER.value,
            "reason": f"negated_cancer_terms: {terms_str}",
            "entities": entities,
        }

    if negated_cancer_count > cancer_count:
        neg_terms = ", ".join(set(negated_cancer_terms))
        affirm_terms = ", ".join(set(cancer_terms)) if cancer_terms else "none"
        return {
            "label": ML.NON_CANCER.value,
            "reason": f"negation_dominant (negated: {neg_terms}; affirmed: {affirm_terms})",
            "entities": entities,
        }

    if cancer_count > 0:
        terms_str = ", ".join(set(cancer_terms))
        return {
            "label": ML.CANCER.value,
            "reason": f"cancer_terms: {terms_str}",
            "entities": entities,
        }

    if non_cancer_count > 0:
        terms_str = ", ".join(set(non_cancer_terms))
        return {
            "label": ML.NON_CANCER.value,
            "reason": f"non_cancer_terms: {terms_str}",
            "entities": entities,
        }

    return {
        "label": ML.NO_SIGNAL.value,
        "reason": "no_relevant_terms",
        "entities": entities,
    }


def medspacy_classify_batch(
        df: pl.DataFrame,
        nlp_pipeline: "Language",
        batch_size: int = 64,
        priority_cols: Union[Tuple[str, ...], List[str]] = PRIORITY_COLS,
        use_normalized: bool = True,
) -> pl.DataFrame:
    """
    Apply MedSpaCy classification to a DataFrame in batches.
    Tracks which columns contain the matched terms.
    """
    rows_as_dicts = df.to_dicts()

    labels: List[str] = []
    reasons: List[str] = []
    source_cols: List[str] = []

    for row in rows_as_dicts:
        found_in_cols = []      # All entity matches (for human-readable reason)
        cancer_in_cols = []     # Only CANCER entity matches (for resolve logic)

        suffix = "_norm" if use_normalized else ""
        for col in priority_cols:
            col_key = f"{col}{suffix}"
            text = row.get(col_key) or row.get(col) or ""

            if isinstance(text, str) and text.strip() and _has_alphabetic(text):
                doc = nlp_pipeline(text.strip().lower())
                for ent in doc.ents:
                    if not getattr(ent._, "is_negated", False):
                        if ent.label_ == ML.CANCER.value:
                            found_in_cols.append(f"{col}:{ent.text}")
                            cancer_in_cols.append(f"{col}:{ent.text}")
                        elif ent.label_ == ML.NON_CANCER.value:
                            found_in_cols.append(f"{col}:{ent.text}")

        # Process combined text for overall classification
        combined_text = clean_texts(row, tuple(priority_cols), use_normalized)
        doc = nlp_pipeline(combined_text)
        result = _classify_doc(doc)

        labels.append(result["label"])

        if found_in_cols:
            col_info = " | found_in: " + ", ".join(set(found_in_cols[:5]))
            reasons.append(result["reason"] + col_info)
            # med_source_columns only tracks where CANCER entities were found
            source_cols.append(", ".join(set([c.split(":")[0] for c in cancer_in_cols])))
        else:
            reasons.append(result["reason"])
            source_cols.append("")

    df = df.with_columns([
        pl.Series("med_label", labels),
        pl.Series("med_reason", reasons),
        pl.Series("med_source_columns", source_cols),
    ])

    return df


# =============================================================================
# Resolution Logic
# =============================================================================

def resolve_uncertain(
    regex_label: Optional[str],
    med_label: Optional[str],
    med_source_columns: Optional[str] = None,
) -> str:
    """
    Resolve final classification by combining regex and MedSpaCy results.

    Every sample gets a definitive cancer or non-cancer classification.
    Uses med_source_columns to determine if cancer was found in
    sample-level columns (source_name, tissue) vs study-level (title, cell_type).
    """
    regex_label = regex_label or CL.UNCERTAIN_NO_SIGNAL.value
    med_label = med_label or ML.NO_SIGNAL.value
    med_source_columns = med_source_columns or ""

    # Check if MedSpaCy found cancer in sample-level columns
    # diagnosis = patient condition, cell_type = cell line origin, title = study name
    # None of these describe the sample itself, so cancer found only there
    # should not override sample-level non-cancer evidence.
    study_level_only = {"title", "diagnosis", "cell_type"}
    source_cols_set = {c.strip() for c in med_source_columns.split(",") if c.strip()}
    cancer_in_sample_cols = bool(source_cols_set - study_level_only)

    # High confidence regex results
    if regex_label == CL.CONFIDENT_CANCER.value:
        return CL.CONFIDENT_CANCER.value

    if regex_label == CL.LIKELY_NON_CANCER.value:
        if med_label == ML.CANCER.value and cancer_in_sample_cols:
            return CL.CONFIRMED_BY_MEDSPACY.value
        # Cancer only in study-level cols: sample-level evidence prevails
        return CL.CONFIRMED_NON_CANCER.value

    # Likely cancer - verify with MedSpaCy
    if regex_label == CL.LIKELY_CANCER.value:
        if med_label == ML.CANCER.value:
            return CL.LIKELY_CANCER.value
        elif med_label == ML.NON_CANCER.value:
            return CL.LIKELY_NON_CANCER.value
        return CL.LIKELY_CANCER.value

    # Uncertain cases - rely on MedSpaCy
    if regex_label.startswith("uncertain"):
        if med_label == ML.CANCER.value:
            return CL.CONFIRMED_BY_MEDSPACY.value
        elif med_label == ML.NON_CANCER.value:
            return CL.CONFIRMED_NON_CANCER.value
        return CL.CONFIRMED_NON_CANCER.value

    return regex_label
