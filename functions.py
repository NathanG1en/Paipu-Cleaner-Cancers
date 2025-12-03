"""
Pipeline functions extracted from new_pipeline.ipynb
Functions for cancer sample classification using regex and medspacy NLP.
"""

import polars as pl
import medspacy
from medspacy.ner import TargetRule
from medspacy.target_matcher import TargetMatcher

# so I don't repeat it
PRIORITY_COLS = [
    "source_name", "tissue", "phenotype", "disease", "cell_type", "tumor_type",
    "sample_name", "condition", "tumor", "cell_type.2", "cell_type.3", 
    "celltype", "tissue_type", "health_state", "tissue_cell_type_source", 
    "source", "model", "tissue_cell_type", "cell_types"
]


def classify_cancer_samples(df: pl.DataFrame, PRIORITY_COLS=PRIORITY_COLS) -> pl.DataFrame:
    """
    Classify samples as cancer / non-cancer / uncertain based on metadata text patterns.
    Returns the same DataFrame with added classification columns.
    """

    # --- Step 1: Setup --- look
    PRIORITY_COLS = [c for c in PRIORITY_COLS if c in df.columns]

    # Regex patterns
    CANCER_POS = r"(?:\bcancers?\b|\btumou?rs?\b|\bmalignan(?:t|cy)\b|\bcarcinomas?\b|\bneoplasms?\b|\bmetasta(?:s|t)es?\b|\badenocarcinomas?\b|\bsarcomas?\b|\bleukemi(?:a|as)\b|\blymphom(?:a|as)\b|\bglioblastomas?\b|\bmelanomas?\b|\boncolog(?:y|ic|ical)\b)"
    CANCER_NEG = r"(?:\bnormal\b|\bhealthy\b|\bctrl\b|\badjacent normal\b|\bnon[-\s]?tumou?r(?:al)?\b|\bbenign\b|\bnon[-\s]?cancer(?:ous)?\b|\bsham\b|\bunaffected\b)"
    ONCO_TRAPS = r"(?:\boncophora\b|\boncorhynchus\b|\boncotic\b|\boncomodulin\b)"

    # Helper to normalize text columns
    def normalize_text(col_expr):
        return (
            col_expr.cast(pl.Utf8)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[_/|]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
    # TODO: change this variable name
    # --- Step 2: Sample name detection ---
    sample_name_col = "title" if "title" in df.columns else "biosample"

    df = df.with_columns([
        normalize_text(pl.col(sample_name_col)).str.contains(CANCER_POS).alias("cancer_in_sample_name"),
        normalize_text(pl.col(sample_name_col)).str.contains(CANCER_NEG).alias("negative_in_sample_name"),
        normalize_text(pl.col(sample_name_col)).str.contains(ONCO_TRAPS).alias("onco_trap_in_sample_name"),
    ])

    # --- Step 3: Check priority columns ---
    for col in PRIORITY_COLS:
        df = df.with_columns([
            normalize_text(pl.col(col)).str.contains(CANCER_POS).alias(f"cancer_in_{col}"),
            normalize_text(pl.col(col)).str.contains(CANCER_NEG).alias(f"negative_in_{col}"),
        ])

    # --- Step 4: Count mentions ---
    cancer_mention_cols = [f"cancer_in_{c}" for c in PRIORITY_COLS]
    negative_mention_cols = [f"negative_in_{c}" for c in PRIORITY_COLS]

    df = df.with_columns([
        pl.sum_horizontal([pl.col(c) for c in cancer_mention_cols if c in df.columns]).alias("n_cancer_mentions"),
        pl.sum_horizontal([pl.col(c) for c in negative_mention_cols if c in df.columns]).alias("n_negative_mentions"),
    ])

    # --- Step 4.5: Collect evidence ---
    def collect_evidence(row):
        cancer_hits = []
        non_hits = []

        for col in PRIORITY_COLS:
            if row.get(f"cancer_in_{col}", False):
                cancer_hits.append(col)
            if row.get(f"negative_in_{col}", False):
                non_hits.append(col)

        # sample_name fields
        if row.get("cancer_in_sample_name", False):
            cancer_hits.append(sample_name_col)
        if row.get("negative_in_sample_name", False):
            non_hits.append(sample_name_col)

        return cancer_hits, non_hits

    df = df.with_columns([
        pl.struct(df.columns).map_elements(lambda r: collect_evidence(r)[0]).alias("cancer_evidence"),
        pl.struct(df.columns).map_elements(lambda r: collect_evidence(r)[1]).alias("negative_evidence"),
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

    # --- Step 5.5: Normalize evidence list columns ---
    df = df.with_columns([
        # Replace NULL with []
        pl.when(pl.col("cancer_evidence").is_null())
        .then(pl.lit([]))
        .otherwise(pl.col("cancer_evidence"))
        .alias("cancer_evidence"),

        pl.when(pl.col("negative_evidence").is_null())
        .then(pl.lit([]))
        .otherwise(pl.col("negative_evidence"))
        .alias("negative_evidence"),
    ])

    # Ensure all elements inside the lists are strings
    df = df.with_columns([
        pl.col("cancer_evidence").list.eval(
            pl.when(pl.element().is_null())
            .then(pl.lit(""))
            .otherwise(pl.element().cast(pl.Utf8))
        ).alias("cancer_evidence"),

        pl.col("negative_evidence").list.eval(
            pl.when(pl.element().is_null())
            .then(pl.lit(""))
            .otherwise(pl.element().cast(pl.Utf8))
        ).alias("negative_evidence"),
    ])

    # --- Step 6: Decision explanation ---
    def build_decision_reason(row):
        cancer_hits = []
        non_hits = []

        for col in PRIORITY_COLS:
            if row.get(f"cancer_in_{col}", False):
                cancer_hits.append(col)
            if row.get(f"negative_in_{col}", False):
                non_hits.append(col)

        if row.get("cancer_in_sample_name", False):
            cancer_hits.append(sample_name_col)
        if row.get("negative_in_sample_name", False):
            non_hits.append(sample_name_col)

        if cancer_hits:
            return f"Cancer terms detected in: {', '.join(cancer_hits)}"
        elif non_hits:
            return f"Non-cancer terms detected in: {', '.join(non_hits)}"
        else:
            return "No cancer-related terms detected"

    df = df.with_columns([
        pl.struct(df.columns)
          .map_elements(build_decision_reason, return_dtype=pl.Utf8)
          .alias("decision_reason")
    ])

    return df


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


def medspacy_classify_batch(all_row_texts: list[list[str]], nlp_pipeline=None, batch_size: int = 32) -> tuple[list[str], list[str]]:
    """
    Batch process multiple rows of texts with MedSpacy.
    
    Args:
        all_row_texts: List of text lists, one per row
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
            detected_terms = []
            
            for text in row_texts:
                if not text or not text.strip():
                    continue
                doc = nlp_pipeline(text)
                for ent in doc.ents:
                    if ent.label_ == "CANCER":
                        if ent._.is_negated:
                            negation_found = True
                            detected_terms.append(f"negated:{ent.text}")
                        else:
                            cancer_found = True
                            detected_terms.append(ent.text)
                    elif ent.label_ == "NON_CANCER":
                        if not ent._.is_negated:
                            non_cancer_found = True
                            detected_terms.append(f"non-cancer:{ent.text}")
            
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


def clean_texts(row, priority_cols=PRIORITY_COLS):
    """
    Extract and clean text fields from a row dictionary.
    Returns a list of non-empty strings.
    """
    # TODO: look at this and possibly add PRIORITY_COLS, could add more complexity for no reason
    # * instead of ["source_name", ... "cell_type"]
    texts = [
        str(row[col]).strip()
        for col in PRIORITY_COLS
        if col in row and row[col] not in (None, "None", "nan", "NaN", "", "null")
    ]
    return texts


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
    """
    KEYWORDS_CANCER = (
        "cancer",
        "carcinoma",
        "sarcoma",
        "leuk",
        "lymphoma",
        "tumor",
        "tumour",
        "melanoma",
        "blastoma",
        "myeloma",
        "metast",
    )

    _skip_literals = {rule.literal.lower() for rule in existing_rules}

    def _phrase_to_pattern(phrase: str):
        doc = nlp.make_doc(phrase.lower())
        pattern = []
        for token in doc:
            if token.is_space:
                continue
            if token.is_alpha:
                pattern.append({"LOWER": token.lower_})
            elif token.is_digit:
                pattern.append({"LIKE_NUM": True})
            elif token.is_punct:
                pattern.append({"TEXT": token.text})
            else:
                pattern.append({"LOWER": token.lower_})
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
            if any(keyword in norm_literal for keyword in KEYWORDS_CANCER)
            else "NON_CANCER"
        )
        auto_rules.append(
            TargetRule(
                literal=disease_str,
                category=category,
                pattern=pattern,
            )
        )

    return auto_rules, skipped_literals


# Global nlp instance (initialize once when module is imported)
nlp = None


def get_nlp():
    """Get or initialize the global nlp pipeline."""
    global nlp
    if nlp is None:
        nlp = initialize_medspacy_pipeline()
    return nlp
