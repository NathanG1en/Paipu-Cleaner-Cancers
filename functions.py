"""
Pipeline functions extracted from new_pipeline.ipynb
Functions for cancer sample classification using regex and medspacy NLP.
"""

import polars as pl
import medspacy
from medspacy.ner import TargetRule
from medspacy.target_matcher import TargetMatcher
from typing import Tuple
from functools import lru_cache

# Update PRIORITY_COLS to match the config
PRIORITY_COLS = [
    "source_name", "tissue", "phenotype", "disease", 
    "cell_type", "tumor_type", "cancer_type"
]


def clean_texts(
    row: dict,
    priority_cols: list[str] | None = None,
    use_normalized: bool = False,
) -> list[tuple[str, str]]:
    """
    Extract non-null texts from row along with their source column names.
    
    Args:
        row: Dict-like row from DataFrame
        priority_cols: Columns to extract from. Defaults to module-level PRIORITY_COLS.
        use_normalized: If True, look for _norm suffix columns first.
    
    Returns:
        List of (text, column_name) tuples
    """
    if priority_cols is None:
        priority_cols = PRIORITY_COLS
    
    results = []
    for col in priority_cols:
        # Try normalized version first if requested
        lookup_col = f"{col}_norm" if use_normalized and f"{col}_norm" in row else col
        
        if lookup_col in row:
            val = row[lookup_col]
            if val not in (None, "None", "nan", "NaN", "", "null"):
                # Use original column name for provenance tracking
                results.append((str(val).strip(), col))
    
    return results


def classify_cancer_samples(
    df: pl.DataFrame,
    nlp_pipeline=None,
    batch_size: int = 64,
    use_normalized: bool = False,
) -> pl.DataFrame:
    """
    Classify samples as cancer / non-cancer / uncertain using MedSpaCy NLP.
    No regex - uses entity recognition and context detection.
    
    Args:
        df: Input DataFrame with text columns
        nlp_pipeline: MedSpaCy pipeline (uses global if None)
        batch_size: Batch size for processing
        use_normalized: If True, looks for pre-normalized columns (col_norm)
        
    Returns:
        DataFrame with added classification columns:
        - med_label: CANCER, NOT_CANCER, UNCERTAIN, NO_SIGNAL
        - med_reason: Explanation of classification
        - confidence_category: Mapped category for compatibility
    """
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()
    
    # Determine which columns to analyze
    priority_cols = [c for c in PRIORITY_COLS if c in df.columns]
    
    # Add sample name column
    sample_name_col = "title" if "title" in df.columns else "biosample"
    if sample_name_col in df.columns and sample_name_col not in priority_cols:
        priority_cols = [sample_name_col] + priority_cols
    
    # Collect text data for each row
    all_row_texts = []
    
    for row in df.iter_rows(named=True):
        row_texts = []
        for col in priority_cols:
            # Check for normalized version first if requested
            if use_normalized:
                norm_col = f"{col}_norm"
                text = row.get(norm_col) or row.get(col)
            else:
                text = row.get(col)
            
            if text and isinstance(text, str) and text.strip() and text.lower() != "nan":
                # Light normalization if not using pre-normalized
                if not use_normalized:
                    text = text.lower().replace("_", " ").replace("/", " ").replace("|", " ")
                    text = " ".join(text.split())  # collapse whitespace
                row_texts.append((text, col))
        
        all_row_texts.append(row_texts)
    
    # Batch classify with MedSpaCy
    med_labels, med_reasons = medspacy_classify_batch(
        all_row_texts, 
        nlp_pipeline, 
        batch_size=batch_size
    )
    
    # Map med_labels to confidence categories for backward compatibility
    label_to_confidence = {
        "CANCER": "confident_cancer",
        "NOT_CANCER": "likely_non_cancer", 
        "UNCERTAIN": "uncertain_weak_signal",
        "NO_SIGNAL": "uncertain_no_signal",
    }
    
    confidence_categories = [label_to_confidence.get(lbl, "uncertain_no_signal") for lbl in med_labels]
    
    # Add results to dataframe
    df = df.with_columns([
        pl.Series("med_label", med_labels),
        pl.Series("med_reason", med_reasons),
        pl.Series("confidence_category", confidence_categories),
    ])
    
    return df


# so I don't repeat it
PRIORITY_COLS = [
    "source_name", "tissue", "phenotype", "disease", "cell_type", "tumor_type",
    "sample_name", "condition", "tumor", "cell_type.2", "cell_type.3", 
    "celltype", "tissue_type", "health_state", "tissue_cell_type_source", 
    "source", "model", "tissue_cell_type", "cell_types"
]


def _has_alphabetic(text: str) -> bool:
    """Check if text contains at least one alphabetic character."""
    return any(c.isalpha() for c in text)


def medspacy_classify(row_texts, nlp_pipeline=None):
    """
    Classify a list of text fields using medspacy NER and context detection.
    Returns: "CANCER", "NOT_CANCER", "UNCERTAIN", or "NO_SIGNAL"
    
    Args:
        row_texts: List of text strings to classify
        nlp_pipeline: The medspacy pipeline to use. If None, uses global nlp.
    """
    # Use provided pipeline or fall back to global
    pipeline = nlp_pipeline if nlp_pipeline is not None else get_nlp()
    
    cancer_found = False
    non_cancer_found = False
    negation_found = False
    has_valid_text = False

    for text in row_texts:
        # Skip non-semantic input (numbers, IDs, hashes)
        if not text or not _has_alphabetic(text):
            continue
        
        has_valid_text = True
        doc = pipeline(text)
        for ent in doc.ents:
            if ent.label_ == "CANCER":
                if ent._.is_negated:
                    negation_found = True
                else:
                    cancer_found = True
            elif ent.label_ == "NON_CANCER":
                if not ent._.is_negated:
                    non_cancer_found = True

    # Return NO_SIGNAL if no valid alphabetic text was processed
    if not has_valid_text:
        return "NO_SIGNAL"

    # Decision hierarchy
    if cancer_found and not negation_found:
        return "CANCER"
    elif non_cancer_found and not cancer_found:
        return "NOT_CANCER"
    elif cancer_found and non_cancer_found:
        return "UNCERTAIN"
    elif negation_found:
        return "NOT_CANCER"
    else:
        return "NO_SIGNAL"


def medspacy_classify_batch(
    all_row_texts: list[list[tuple[str, str]]], 
    nlp_pipeline=None, 
    batch_size: int = 32
) -> tuple[list[str], list[str]]:
    """
    Batch process multiple rows of (text, column_name) pairs with MedSpacy.
    
    OPTIMIZED VERSION:
    - Uses nlp.pipe() for bulk processing
    - Deduplicates texts to avoid redundant NLP calls
    - Skips non-alphabetic text early
    
    Args:
        all_row_texts: List of [(text, col_name), ...] per row
        nlp_pipeline: MedSpacy pipeline (uses global if None)
        batch_size: Number of texts to process at once in nlp.pipe()
        
    Returns:
        Tuple of (labels, reasons) - two parallel lists
    """
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()
    
    labels = []
    reasons = []
    
    # Process in chunks of rows
    for i in range(0, len(all_row_texts), batch_size):
        batch = all_row_texts[i:i + batch_size]
        
        # OPTIMIZATION 1: Collect all unique texts from this batch
        unique_texts = set()
        for row_texts in batch:
            for text, _ in row_texts:
                if text and text.strip() and _has_alphabetic(text):
                    unique_texts.add(text)
        
        # OPTIMIZATION 2: Process all unique texts in bulk with nlp.pipe()
        unique_texts_list = list(unique_texts)
        if unique_texts_list:
            docs = list(nlp_pipeline.pipe(unique_texts_list, batch_size=min(64, len(unique_texts_list))))
            doc_map = dict(zip(unique_texts_list, docs))
        else:
            doc_map = {}
        
        # Now process each row using the cached doc results
        for row_texts in batch:
            if not row_texts:
                labels.append("NO_SIGNAL")
                reasons.append("No text to analyze")
                continue
            
            # OPTIMIZATION 3: Skip entire row if no alphabetic text
            valid_texts = [(text, col) for text, col in row_texts 
                          if text and text.strip() and _has_alphabetic(text)]
            
            if not valid_texts:
                labels.append("NO_SIGNAL")
                reasons.append("No alphabetic text in row")
                continue
            
            cancer_found = False
            non_cancer_found = False
            negation_found = False
            detected_terms = []
            
            # OPTIMIZATION 4: Deduplicate within row
            seen_texts = set()
            
            for text, col_name in valid_texts:
                # Skip duplicates within this row
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                
                # Use pre-computed doc from cache
                doc = doc_map.get(text)
                if doc is None:
                    continue
                    
                for ent in doc.ents:
                    # Skip entities that are purely numeric
                    if not _has_alphabetic(ent.text):
                        continue
                        
                    if ent.label_ == "CANCER":
                        if ent._.is_negated:
                            negation_found = True
                            detected_terms.append(f"negated:{ent.text} ({col_name})")
                        else:
                            cancer_found = True
                            detected_terms.append(f"{ent.text} ({col_name})")
                    elif ent.label_ == "NON_CANCER":
                        if not ent._.is_negated:
                            non_cancer_found = True
                            detected_terms.append(f"non-cancer:{ent.text} ({col_name})")
            
            # Decision logic
            if not detected_terms:
                labels.append("NO_SIGNAL")
                reasons.append("No valid alphabetic cancer-related text")
            elif cancer_found and not negation_found:
                labels.append("CANCER")
                reasons.append(f"Detected: {', '.join(detected_terms)}")
            elif non_cancer_found and not cancer_found:
                labels.append("NOT_CANCER")
                reasons.append(f"Non-cancer terms: {', '.join(detected_terms)}")
            elif cancer_found and non_cancer_found:
                labels.append("UNCERTAIN")
                reasons.append(f"Mixed signals: {', '.join(detected_terms)}")
            elif negation_found:
                labels.append("NOT_CANCER")
                reasons.append(f"Negated cancer terms: {', '.join(detected_terms)}")
            else:
                labels.append("NO_SIGNAL")
                reasons.append("No cancer-related terms detected")
    
    return labels, reasons


def resolve_uncertain(
    regex_label: str,
    med_label: str | None = None,
    regex_reason: str = "",
    med_reason: str = ""
) -> tuple[str, str, str, str, str]:
    """
    Merge regex classification with medspacy classification.

    Args:
        regex_label: The output of classify_cancer_samples() (e.g., "confident_cancer")
        med_label: MedSpacy label ("CANCER", "NOT_CANCER", "UNCERTAIN", "NO_SIGNAL", or "")
        regex_reason: Explanation from regex classifier
        med_reason: Explanation from medspacy

    Returns:
        Tuple of (final_label, regex_label, med_label, regex_reason, med_reason)
    """
    UNCERTAIN_REGEX = {"uncertain_no_signal", "uncertain_weak_signal"}

    # Handle missing/empty medspacy result
    if not med_label:
        return (
            regex_label,
            regex_label,
            "",
            regex_reason or f"Regex classifier produced '{regex_label}'.",
            "No medspacy signal."
        )

    # Build regex_reason if not provided
    if not regex_reason:
        regex_reason = f"Regex classifier produced '{regex_label}'."

    # Decision rules
    if regex_label in UNCERTAIN_REGEX:
        if med_label == "CANCER":
            final = "confirmed_by_medspacy"
        elif med_label == "NOT_CANCER":
            final = "confirmed_non_cancer"
        elif med_label == "UNCERTAIN":
            final = "uncertain_medspacy"
        else:
            final = regex_label

    # Regex says "likely non-cancer" but medspacy says "CANCER"
    elif regex_label == "likely_non_cancer" and med_label == "CANCER":
        final = "confirmed_by_medspacy"

    else:
        # Default to regex label
        final = regex_label

    return (final, regex_label, med_label, regex_reason, med_reason)


def get_default_target_rules():
    """
    Returns the default cancer and non-cancer target rules.
    
    Returns:
        tuple: A tuple containing (cancer_rules, non_cancer_rules)
    """
    from medspacy.ner import TargetRule
    
    # Cancer rules - all with IS_ALPHA guards where applicable
    cancer_rules = [
        # General cancer terms
        TargetRule(
            literal="cancer",
            category="CANCER",
            pattern=[{"LOWER": "cancer", "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="tumor",
            category="CANCER",
            pattern=[{"LOWER": {"IN": ["tumor", "tumour", "tumors", "tumours"]}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="carcinoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "carcinomas?"}, "IS_ALPHA": True}]
        ),

        # Specific cancer types
        TargetRule(
            literal="adenocarcinoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "adenocarcinomas?"}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="squamous cell carcinoma",
            category="CANCER",
            pattern=[
                {"LOWER": "squamous", "IS_ALPHA": True},
                {"LOWER": "cell", "IS_ALPHA": True},
                {"LOWER": {"REGEX": "carcinomas?"}, "IS_ALPHA": True}
            ]
        ),
        TargetRule(
            literal="small cell carcinoma",
            category="CANCER",
            pattern=[
                {"LOWER": "small", "IS_ALPHA": True},
                {"LOWER": "cell", "IS_ALPHA": True},
                {"LOWER": {"REGEX": "carcinomas?"}, "IS_ALPHA": True}
            ]
        ),
        TargetRule(
            literal="non-small cell carcinoma",
            category="CANCER",
            pattern=[
                {"LOWER": "non", "IS_ALPHA": True},
                {"IS_PUNCT": True, "OP": "?"},
                {"LOWER": "small", "IS_ALPHA": True},
                {"LOWER": "cell", "IS_ALPHA": True},
                {"LOWER": {"REGEX": "carcinomas?"}, "IS_ALPHA": True}
            ]
        ),

        # Leukemia/Lymphoma
        TargetRule(
            literal="leukemia",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "leuk[ae]mias?"}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="lymphoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "lymphomas?"}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="acute myeloid leukemia",
            category="CANCER",
            pattern=[
                {"LOWER": "acute", "IS_ALPHA": True},
                {"LOWER": "myeloid", "IS_ALPHA": True},
                {"LOWER": {"REGEX": "leuk[ae]mias?"}, "IS_ALPHA": True}
            ]
        ),

        # Sarcomas
        TargetRule(
            literal="sarcoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "sarcomas?"}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="osteosarcoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "osteosarcomas?"}, "IS_ALPHA": True}]
        ),

        # Brain tumors
        TargetRule(
            literal="glioblastoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "glioblastomas?"}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="glioma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "gliomas?"}, "IS_ALPHA": True}]
        ),

        # Melanoma
        TargetRule(
            literal="melanoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "melanomas?"}, "IS_ALPHA": True}]
        ),

        # Malignancy terms
        TargetRule(
            literal="malignant",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": r"(?<!pre[- ])malignan(t|cy)"}, "IS_ALPHA": True}]
        ),

        TargetRule(
            literal="neoplasm",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "neoplasms?"}, "IS_ALPHA": True}]
        ),
        TargetRule(
            literal="metastasis",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "metasta(sis|ses|tic)"}, "IS_ALPHA": True}]
        ),

        # Context-dependent patterns
        TargetRule(
            literal="malignant tissue",
            category="CANCER",
            pattern=[
                {"LOWER": "malignant", "IS_ALPHA": True},
                {"LOWER": {"IN": ["tissue", "cells", "lesion", "mass"]}, "IS_ALPHA": True}
            ]
        ),
        TargetRule(
            literal="cancerous tissue",
            category="CANCER",
            pattern=[
                {"LOWER": "cancerous", "IS_ALPHA": True},
                {"LOWER": {"IN": ["tissue", "cells", "lesion", "mass"]}, "IS_ALPHA": True}
            ]
        ),

        # Oncology context
        TargetRule(
            literal="oncology",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "oncolog(y|ic|ical)"}, "IS_ALPHA": True}],
        ),

        # Cell line patterns
        TargetRule(
            literal="cancer cell line",
            category="CANCER",
            pattern=[
                {"LOWER": {"IN": ["cancer", "tumor", "tumour"]}, "IS_ALPHA": True},
                {"LOWER": "cell", "IS_ALPHA": True},
                {"LOWER": {"IN": ["line", "lines"]}, "IS_ALPHA": True}
            ]
        ),

        TargetRule(
            literal="TIL",
            category="CANCER",
            pattern=[
                {"LOWER": {"IN": ["til", "tils"]}, "IS_ALPHA": True},
            ]
        )
    ]

    # Non-cancer rules - all with IS_ALPHA guards
    non_cancer_rules = [
        TargetRule(
            literal="normal tissue",
            category="NON_CANCER",
            pattern=[
                {"LOWER": {"IN": ["normal", "healthy", "control", "benign", "adjacent"]}, "IS_ALPHA": True},
                {"LOWER": {"IN": ["tissue", "sample", "cells", "fat", "pad", "organ"]}, "OP": "?", "IS_ALPHA": True}
            ]
        ),

        TargetRule(
            literal="benign lesion",
            category="NON_CANCER",
            pattern=[
                {"LOWER": "benign", "IS_ALPHA": True},
                {"LOWER": {"IN": ["lesion", "mass", "tumor", "tumour"]}, "IS_ALPHA": True}
            ]
        ),

        TargetRule(
            literal="premalignant",
            category="NON_CANCER",
            pattern=[{"LOWER": {"REGEX": "pre[- ]?malignan(t|cy)"}, "IS_ALPHA": True}]
        )
    ]

    return cancer_rules, non_cancer_rules


def initialize_medspacy_pipeline(*rule_lists):
    """
    Initialize medspacy pipeline with provided target rules.
    
    OPTIMIZED: Disables unused pipeline components for speed.
    
    Args:
        *rule_lists: Variable number of rule lists to add to the pipeline.
                     Each list should contain TargetRule objects.
                     If no lists provided, uses default rules from get_default_target_rules().
    
    Returns:
        nlp: The configured medspacy pipeline.
    """
    # OPTIMIZATION: Only enable what we need, disable everything else
    nlp = medspacy.load(
        enable=["ner", "context"],
        disable=["parser", "tagger", "lemmatizer", "attribute_ruler"]
    )
    tm = nlp.get_pipe("medspacy_target_matcher")

    # If no rule lists provided, use defaults
    if not rule_lists:
        cancer_rules, non_cancer_rules = get_default_target_rules()
        rule_lists = (cancer_rules, non_cancer_rules)
    
    # Add all rule lists to the target matcher
    for rules in rule_lists:
        tm.add(rules)

    return nlp


def generate_disease_rules(unique_diseases, nlp, existing_rules):
    """
    Auto-generate TargetRules for a list of unique diseases.
    Returns a list of new TargetRule objects.
    
    Filters out:
    - Numeric-only diseases
    - Very short diseases (< 2 alphabetic characters)
    - Duplicates of existing rules
    """

    # Expanded, high-recall cancer keywords
    KEYWORDS_CANCER = (
        "cancer", "malign*", "malignant", "neoplasm", "neoplastic",
        "oncolog*", "oncogen*", "tumor", "tumour", "carcinoma",
        "adenocarcinoma", "squamous cell carcinoma", "basal cell carcinoma",
        "sarcoma", "osteosarcoma", "chondrosarcoma", "liposarcoma",
        "glioma", "astrocytoma", "oligodendroglioma", "blastoma",
        "neuroblastoma", "retinoblastoma", "melanoma", "mesothelioma",
        "thelioma", "myeloma", "plasmacytoma", "leuk*", "leukemia",
        "lymphoma", "hodgkin", "non hodgkin", "myelodysplastic",
        "myeloproliferative", "breast cancer", "lung cancer", "colon cancer",
        "colorectal cancer", "prostate cancer", "pancreatic cancer",
        "hepatic cancer", "hepatocellular carcinoma", "renal cancer",
        "kidney cancer", "bladder cancer", "ovarian cancer", "cervical cancer",
        "endometrial cancer", "thyroid cancer", "brain tumor", "cns tumor",
        "metast*", "metastatic", "metastasis", "invasive", "advanced cancer",
        "recurrent", "relapsed", "progression",
    )

    _skip_literals = {rule.literal.lower() for rule in existing_rules}

    def _is_valid_disease_text(text: str) -> bool:
        """Check if text is valid for creating a rule."""
        if not text:
            return False
        alpha_chars = sum(1 for c in text if c.isalpha())
        return alpha_chars >= 2

    def _phrase_to_pattern(phrase: str):
        """Converts a phrase into a spaCy Matcher pattern with IS_ALPHA guards."""
        if phrase.endswith("*"):
            prefix = phrase[:-1]
            return [{"LOWER": {"REGEX": f"^{prefix}"}, "IS_ALPHA": True}]

        doc = nlp.make_doc(phrase.lower())
        pattern = []

        for token in doc:
            if token.is_space:
                continue
            if token.is_alpha:
                pattern.append({"LOWER": token.lower_, "IS_ALPHA": True})
            elif token.is_digit:
                pattern.append({"LIKE_NUM": True})
            else:
                pattern.append({"TEXT": token.text})

        return pattern

    auto_rules = []
    skipped_literals = []

    for disease in unique_diseases:
        if disease is None:
            continue
        disease_str = str(disease).strip()
        
        if not disease_str or disease_str.lower() in {"nan", "none", "null", ""}:
            skipped_literals.append(disease_str)
            continue
        
        if not _is_valid_disease_text(disease_str):
            skipped_literals.append(disease_str)
            continue
        
        norm_literal = disease_str.lower()
        if norm_literal in _skip_literals:
            skipped_literals.append(disease_str)
            continue
            
        pattern = _phrase_to_pattern(disease_str)
        if not pattern:
            skipped_literals.append(disease_str)
            continue
            
        category = (
            "CANCER"
            if any(kw.rstrip("*") in norm_literal for kw in KEYWORDS_CANCER)
            else "NON_CANCER"
        )
        
        auto_rules.append(
            TargetRule(
                literal=disease_str,
                category=category,
                pattern=pattern,
            )
        )
        _skip_literals.add(norm_literal)

    return auto_rules, skipped_literals


# Global nlp instance (initialize once when module is imported)
nlp = None


def get_nlp():
    """Get or initialize the global nlp pipeline."""
    global nlp
    if nlp is None:
        nlp = initialize_medspacy_pipeline()
    return nlp


# ============================================================================
# TEXT PREPROCESSING - Column Discovery & Normalization
# ============================================================================

def identify_candidate_text_columns(
    df: pl.DataFrame,
    min_avg_length: float = 10.0,
    min_non_null_pct: float = 0.01,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """
    Step 1: Identify candidate text columns using cheap heuristics.
    """
    if exclude_patterns is None:
        exclude_patterns = ["_id", "accession", "uuid", "hash", "checksum", "md5", "sha"]
    
    candidates = []
    n_rows = len(df)
    
    for col in df.columns:
        if df[col].dtype != pl.Utf8:
            continue
        
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue
        
        non_null_count = df[col].drop_nulls().len()
        non_null_pct = non_null_count / n_rows if n_rows > 0 else 0
        
        if non_null_pct < min_non_null_pct:
            continue
        
        avg_length = (
            df.select(
                pl.col(col)
                .drop_nulls()
                .str.len_chars()
                .mean()
            )
            .item()
        )
        
        if avg_length is None or avg_length < min_avg_length:
            continue
        
        candidates.append(col)
    
    return candidates


def normalize_text_column(col_expr: pl.Expr) -> pl.Expr:
    """
    Step 2: Normalize text aggressively for a single column expression.
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


def preprocess_text_columns(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    suffix: str = "_normalized",
    min_avg_length: float = 10.0,
    min_non_null_pct: float = 0.01,
) -> Tuple[pl.DataFrame, list[str]]:
    """
    Full text preprocessing pipeline: discover candidates + normalize.
    """
    if columns is None:
        columns = identify_candidate_text_columns(
            df, 
            min_avg_length=min_avg_length,
            min_non_null_pct=min_non_null_pct
        )
    else:
        columns = [c for c in columns if c in df.columns]
    
    if not columns:
        return df, []
    
    normalized_cols = []
    for col in columns:
        norm_col_name = f"{col}{suffix}"
        df = df.with_columns(
            normalize_text_column(pl.col(col)).alias(norm_col_name)
        )
        normalized_cols.append(norm_col_name)
    
    return df, normalized_cols


def get_text_column_stats(df: pl.DataFrame) -> pl.DataFrame:
    """
    Utility: Get statistics for all string columns to help tune thresholds.
    """
    stats = []
    n_rows = len(df)
    
    for col in df.columns:
        if df[col].dtype != pl.Utf8:
            continue
        
        col_data = df[col].drop_nulls()
        non_null_count = col_data.len()
        
        stats.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "non_null_count": non_null_count,
            "non_null_pct": round(non_null_count / n_rows * 100, 2) if n_rows > 0 else 0,
            "avg_length": round(col_data.str.len_chars().mean() or 0, 1),
            "max_length": col_data.str.len_chars().max() or 0,
            "unique_count": col_data.n_unique(),
        })

    return pl.DataFrame(stats).sort("avg_length", descending=True)
