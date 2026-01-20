"""
Pipeline functions extracted from new_pipeline.ipynb
Functions for cancer sample classification using regex and medspacy NLP.
"""

import polars as pl
import medspacy
from medspacy.ner import TargetRule
from medspacy.target_matcher import TargetMatcher
from typing import Tuple

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
    use_normalized: bool = False,
) -> pl.DataFrame:
    """
    Classify samples as cancer / non-cancer / uncertain based on metadata text patterns.
    Returns the same DataFrame with added classification columns.
    
    Args:
        df: Input DataFrame
        use_normalized: If True, use pre-normalized _norm columns (faster)
    """
    # --- Step 1: Setup ---
    priority_cols = [c for c in PRIORITY_COLS if c in df.columns]
    
    # Regex patterns
    CANCER_POS = r"(?:\bcancers?\b|\btumou?rs?\b|\bmalignan(?:t|cy)\b|\bcarcinomas?\b|\bneoplasms?\b|\bmetasta(?:s|t)es?\b|\badenocarcinomas?\b|\bsarcomas?\b|\bleukemi(?:a|as)\b|\blymphom(?:a|as)\b|\bglioblastomas?\b|\bmelanomas?\b|\boncolog(?:y|ic|ical)\b)"
    CANCER_NEG = r"(?:\bnormal\b|\bhealthy\b|\bctrl\b|\badjacent normal\b|\bnon[-\s]?tumou?r(?:al)?\b|\bbenign\b|\bnon[-\s]?cancer(?:ous)?\b|\bsham\b|\bunaffected\b)"
    ONCO_TRAPS = r"(?:\boncophora\b|\boncorhynchus\b|\boncotic\b|\boncomodulin\b)"

    def normalize_text(col_expr):
        """Normalize text - only needed if not using pre-normalized columns."""
        return (
            col_expr.cast(pl.Utf8)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[_/|]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def get_text_col(col_name: str) -> pl.Expr:
        """Get the appropriate column expression (normalized or raw)."""
        norm_col = f"{col_name}_norm"
        if use_normalized and norm_col in df.columns:
            # Already normalized, just use it directly
            return pl.col(norm_col)
        else:
            # Normalize on the fly
            return normalize_text(pl.col(col_name))

    # --- Step 2: Sample name detection ---
    sample_name_col = "title" if "title" in df.columns else "biosample"
    
    df = df.with_columns([
        get_text_col(sample_name_col).str.contains(CANCER_POS).alias("cancer_in_sample_name"),
        get_text_col(sample_name_col).str.contains(CANCER_NEG).alias("negative_in_sample_name"),
        get_text_col(sample_name_col).str.contains(ONCO_TRAPS).alias("onco_trap_in_sample_name"),
    ])

    # --- Step 3: Check priority columns ---
    for col in priority_cols:
        df = df.with_columns([
            get_text_col(col).str.contains(CANCER_POS).alias(f"cancer_in_{col}"),
            get_text_col(col).str.contains(CANCER_NEG).alias(f"negative_in_{col}"),
        ])

    # --- Step 4: Count mentions ---
    cancer_mention_cols = [f"cancer_in_{c}" for c in priority_cols]
    negative_mention_cols = [f"negative_in_{c}" for c in priority_cols]

    df = df.with_columns([
        pl.sum_horizontal([pl.col(c) for c in cancer_mention_cols if c in df.columns]).alias("n_cancer_mentions"),
        pl.sum_horizontal([pl.col(c) for c in negative_mention_cols if c in df.columns]).alias("n_negative_mentions"),
    ])

    # --- Step 5: Confidence category ---
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
        .alias("confidence_category")
    ])

    return df
import polars as pl
import medspacy
from medspacy.ner import TargetRule
from medspacy.target_matcher import TargetMatcher
from typing import Tuple

# so I don't repeat it
PRIORITY_COLS = [
    "source_name", "tissue", "phenotype", "disease", "cell_type", "tumor_type",
    "sample_name", "condition", "tumor", "cell_type.2", "cell_type.3", 
    "celltype", "tissue_type", "health_state", "tissue_cell_type_source", 
    "source", "model", "tissue_cell_type", "cell_types"
]



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

    for text in row_texts:
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
    
    Args:
        all_row_texts: List of [(text, col_name), ...] per row
        nlp_pipeline: MedSpacy pipeline (uses global if None)
        batch_size: Number of rows to process at once
        
    Returns:
        Tuple of (labels, reasons) - two parallel lists
    """
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp()
    
    labels = []
    reasons = []
    
    for i in range(0, len(all_row_texts), batch_size):
        batch = all_row_texts[i:i + batch_size]
        
        for row_texts in batch:
            if not row_texts:
                labels.append("NO_SIGNAL")
                reasons.append("No text to analyze")
                continue
            
            cancer_found = False
            non_cancer_found = False
            negation_found = False
            detected_terms = []  # Will store "term (column)" strings
            
            for text, col_name in row_texts:
                if not text or not text.strip():
                    continue
                doc = nlp_pipeline(text)
                for ent in doc.ents:
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
            if cancer_found and not negation_found:
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


# TODO: add "non" to negated
def get_default_target_rules():
    """
    Returns the default cancer and non-cancer target rules.
    
    Returns:
        tuple: A tuple containing (cancer_rules, non_cancer_rules)
    """
    from medspacy.ner import TargetRule
    
    # Cancer rules
    cancer_rules = [
        # General cancer terms
        TargetRule(
            literal="cancer",
            category="CANCER",
            pattern=[{"LOWER": "cancer"}]
        ),
        TargetRule(
            literal="tumor",
            category="CANCER",
            pattern=[{"LOWER": {"IN": ["tumor", "tumour", "tumors", "tumours"]}}]
        ),
        TargetRule(
            literal="carcinoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "carcinomas?"}}]
        ),

        # Specific cancer types
        TargetRule(
            literal="adenocarcinoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "adenocarcinomas?"}}]
        ),
        TargetRule(
            literal="squamous cell carcinoma",
            category="CANCER",
            pattern=[
                {"LOWER": "squamous"},
                {"LOWER": "cell"},
                {"LOWER": {"REGEX": "carcinomas?"}}
            ]
        ),
        TargetRule(
            literal="small cell carcinoma",
            category="CANCER",
            pattern=[
                {"LOWER": "small"},
                {"LOWER": "cell"},
                {"LOWER": {"REGEX": "carcinomas?"}}
            ]
        ),
        TargetRule(
            literal="non-small cell carcinoma",
            category="CANCER",
            pattern=[
                {"LOWER": "non"},
                {"IS_PUNCT": True, "OP": "?"},
                {"LOWER": "small"},
                {"LOWER": "cell"},
                {"LOWER": {"REGEX": "carcinomas?"}}
            ]
        ),

        # Leukemia/Lymphoma
        TargetRule(
            literal="leukemia",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "leuk[ae]mias?"}}]
        ),
        TargetRule(
            literal="lymphoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "lymphomas?"}}]
        ),
        TargetRule(
            literal="acute myeloid leukemia",
            category="CANCER",
            pattern=[
                {"LOWER": "acute"},
                {"LOWER": "myeloid"},
                {"LOWER": {"REGEX": "leuk[ae]mias?"}}
            ]
        ),

        # Sarcomas
        TargetRule(
            literal="sarcoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "sarcomas?"}}]
        ),
        TargetRule(
            literal="osteosarcoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "osteosarcomas?"}}]
        ),

        # Brain tumors
        TargetRule(
            literal="glioblastoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "glioblastomas?"}}]
        ),
        TargetRule(
            literal="glioma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "gliomas?"}}]
        ),

        # Melanoma
        TargetRule(
            literal="melanoma",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "melanomas?"}}]
        ),

        # Malignancy terms
        TargetRule(
            literal="malignant",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": r"(?<!pre[- ])malignan(t|cy)"}}]
        ),

        TargetRule(
            literal="neoplasm",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "neoplasms?"}}]
        ),
        TargetRule(
            literal="metastasis",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "metasta(sis|ses|tic)"}}]
        ),

        # Context-dependent patterns
        TargetRule(
            literal="malignant tissue",
            category="CANCER",
            pattern=[
                {"LOWER": "malignant"},
                {"LOWER": {"IN": ["tissue", "cells", "lesion", "mass"]}}
            ]
        ),
        TargetRule(
            literal="cancerous tissue",
            category="CANCER",
            pattern=[
                {"LOWER": "cancerous"},
                {"LOWER": {"IN": ["tissue", "cells", "lesion", "mass"]}}
            ]
        ),

        # Oncology context
        TargetRule(
            literal="oncology",
            category="CANCER",
            pattern=[{"LOWER": {"REGEX": "oncolog(y|ic|ical)"}}],
        ),

        # Cell line patterns
        TargetRule(
            literal="cancer cell line",
            category="CANCER",
            pattern=[
                {"LOWER": {"IN": ["cancer", "tumor", "tumour"]}},
                {"LOWER": "cell"},
                {"LOWER": {"IN": ["line", "lines"]}}
            ]
        ),

        TargetRule(
            literal="TIL",
            category="CANCER",
            pattern=[
                {"LOWER": {"IN": ["til", "tils", "t-i-l", "t.i.l.", "t.i.l.s."]}},
                {"LOWER": {"IN": ["tumor", "tumour"]}, "OP": "?"},
                {"LOWER": {"IN": ["infiltrating", "infiltrated"]}, "OP": "?"},
                {"LOWER": "lymphocytes", "OP": "?"},
            ]
        )
    ]

    # Non-cancer rules
    non_cancer_rules = [
        TargetRule(
            literal="normal tissue",
            category="NON_CANCER",
            pattern=[
                {"LOWER": {"IN": ["normal", "healthy", "control", "benign", "adjacent"]}},
                {"LOWER": {"IN": ["tissue", "sample", "cells", "fat", "pad", "organ"]}, "OP": "?"}
            ]
        ),

        TargetRule(
            literal="benign lesion",
            category="NON_CANCER",
            pattern=[
                {"LOWER": "benign"},
                {"LOWER": {"IN": ["lesion", "mass", "tumor", "tumour"]}}
            ]
        ),

        TargetRule(
            literal="premalignant",
            category="NON_CANCER",
            pattern=[{"LOWER": {"REGEX": "pre[- ]?malignan(t|cy)"}}]
        )
    ]

    return cancer_rules, non_cancer_rules


def initialize_medspacy_pipeline(*rule_lists):
    """
    Initialize medspacy pipeline with provided target rules.
    
    Args:
        *rule_lists: Variable number of rule lists to add to the pipeline.
                     Each list should contain TargetRule objects.
                     If no lists provided, uses default rules from get_default_target_rules().
    
    Returns:
        nlp: The configured medspacy pipeline.
    
    Example:
        # Use default rules
        nlp = initialize_medspacy_pipeline()
        
        # Use custom rules
        cancer_rules, non_cancer_rules = get_default_target_rules()
        nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)
        
        # Use custom rules with additional rules
        cancer_rules, non_cancer_rules = get_default_target_rules()
        custom_rules = [...]
        nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules, custom_rules)
    """

    
    nlp = medspacy.load(enable=["ner", "context"])
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

    Extension point:
        - ontology_id (ICD-10 / SNOMED / MeSH) can be added later
          without changing matching logic.
    """

    # Expanded, high-recall cancer keywords
    # Use * suffix to indicate prefix / regex match
    KEYWORDS_CANCER = (
        # Core malignancy
        "cancer",
        "malign*",
        "malignant",
        "neoplasm",
        "neoplastic",
        "oncolog*",
        "oncogen*",

        # Tumor morphology
        "tumor",
        "tumour",
        "carcinoma",
        "adenocarcinoma",
        "squamous cell carcinoma",
        "basal cell carcinoma",
        "sarcoma",
        "osteosarcoma",
        "chondrosarcoma",
        "liposarcoma",
        "glioma",
        "astrocytoma",
        "oligodendroglioma",
        "blastoma",
        "neuroblastoma",
        "retinoblastoma",
        "melanoma",
        "mesothelioma",
        "thelioma",
        "myeloma",
        "plasmacytoma",

        # Hematologic
        "leuk*",
        "leukemia",
        "lymphoma",
        "hodgkin",
        "non hodgkin",
        "myelodysplastic",
        "myeloproliferative",

        # Organ-specific (common)
        "breast cancer",
        "lung cancer",
        "colon cancer",
        "colorectal cancer",
        "prostate cancer",
        "pancreatic cancer",
        "hepatic cancer",
        "hepatocellular carcinoma",
        "renal cancer",
        "kidney cancer",
        "bladder cancer",
        "ovarian cancer",
        "cervical cancer",
        "endometrial cancer",
        "thyroid cancer",
        "brain tumor",
        "cns tumor",

        # Progression / severity
        "metast*",
        "metastatic",
        "metastasis",
        "invasive",
        "advanced cancer",
        "recurrent",
        "relapsed",
        "progression",
    )

    _skip_literals = {rule.literal.lower() for rule in existing_rules}

    def _phrase_to_pattern(phrase: str):
        """
        Converts a phrase into a spaCy Matcher pattern.
        Supports:
            - exact token matches
            - prefix matching via '*' suffix
        """
        # Prefix / regex rule
        if phrase.endswith("*"):
            prefix = phrase[:-1]
            return [{"LOWER": {"REGEX": f"^{prefix}"}}]

        doc = nlp.make_doc(phrase.lower())
        pattern = []

        for token in doc:
            if token.is_space:
                continue
            if token.is_alpha:
                pattern.append({"LOWER": token.lower_})
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
        if not disease_str or disease_str.lower() in {"nan", "none", "null"}:
            continue
        norm_literal = disease_str.lower()
        if norm_literal in _skip_literals:
            skipped_literals.append(disease_str)
            continue
        pattern = _phrase_to_pattern(disease_str)
        if not pattern:
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

    return auto_rules, skipped_literals  # <-- Make sure this exists!


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
    
    Filters 1000 columns → ~50-150 candidates based on:
    - dtype is string/Utf8
    - average string length > threshold
    - % non-null above threshold
    
    Args:
        df: Input DataFrame
        min_avg_length: Minimum average string length to consider (default 10 chars)
        min_non_null_pct: Minimum fraction of non-null values (default 1%)
        exclude_patterns: Column name patterns to exclude (e.g., ['_id', 'accession'])
    
    Returns:
        List of candidate column names suitable for text analysis
    """
    if exclude_patterns is None:
        exclude_patterns = ["_id", "accession", "uuid", "hash", "checksum", "md5", "sha"]
    
    candidates = []
    n_rows = len(df)
    
    for col in df.columns:
        # Skip non-string columns
        if df[col].dtype != pl.Utf8:
            continue
        
        # Skip columns matching exclude patterns
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in exclude_patterns):
            continue
        
        # Calculate non-null percentage
        non_null_count = df[col].drop_nulls().len()
        non_null_pct = non_null_count / n_rows if n_rows > 0 else 0
        
        if non_null_pct < min_non_null_pct:
            continue
        
        # Calculate average string length (on non-null values)
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
    
    Transformations:
    - Cast to string
    - Fill nulls with empty string
    - Lowercase
    - Strip punctuation (except hyphens within words)
    - Collapse whitespace
    - Strip leading/trailing whitespace
    
    Args:
        col_expr: Polars column expression
    
    Returns:
        Normalized column expression
    """
    return (
        col_expr.cast(pl.Utf8)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[_/|\\]", " ")  # Replace common separators with space
        .str.replace_all(r"[^\w\s\-]", "")  # Remove punctuation (keep hyphens)
        .str.replace_all(r"\s+", " ")  # Collapse whitespace
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
    
    Args:
        df: Input DataFrame
        columns: Specific columns to normalize. If None, auto-discovers candidates.
        suffix: Suffix for normalized column names (default "_normalized")
        min_avg_length: For auto-discovery, minimum average string length
        min_non_null_pct: For auto-discovery, minimum non-null percentage
    
    Returns:
        Tuple of:
        - DataFrame with normalized columns added
        - List of normalized column names
    """
    # Step 1: Identify candidates if not specified
    if columns is None:
        columns = identify_candidate_text_columns(
            df, 
            min_avg_length=min_avg_length,
            min_non_null_pct=min_non_null_pct
        )
    else:
        # Filter to only existing columns
        columns = [c for c in columns if c in df.columns]
    
    if not columns:
        return df, []
    
    # Step 2: Normalize all candidate columns
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
    
    Returns DataFrame with columns:
    - column_name
    - dtype
    - non_null_count
    - non_null_pct
    - avg_length
    - max_length
    - unique_count
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
