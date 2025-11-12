
import polars as pl

def classify_cancer_samples(df: pl.DataFrame) -> pl.DataFrame:
    """
    Classify samples as cancer / non-cancer / uncertain based on metadata text patterns.
    Returns the same DataFrame with added classification columns.
    """

    # --- Step 1: Setup ---
    PRIORITY_COLS = ["source_name", "tissue", "phenotype", "disease", "cell_type", "tumor_type"]
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

    # --- Step 3.5: Special rule for source_name and tissue with "tumor" ---
    # If tumor appears in source_name or tissue WITHOUT negation, it's definitely cancer
    has_tumor_source = False
    has_tumor_tissue = False
    
    if "source_name" in PRIORITY_COLS:
        df = df.with_columns([
            (normalize_text(pl.col("source_name")).str.contains(r"\btumou?rs?\b") & 
             ~normalize_text(pl.col("source_name")).str.contains(CANCER_NEG)).alias("tumor_in_source_name")
        ])
        has_tumor_source = True
    
    if "tissue" in PRIORITY_COLS:
        df = df.with_columns([
            (normalize_text(pl.col("tissue")).str.contains(r"\btumou?rs?\b") & 
             ~normalize_text(pl.col("tissue")).str.contains(CANCER_NEG)).alias("tumor_in_tissue")
        ])
        has_tumor_tissue = True

    # --- Step 4: Count mentions ---
    cancer_mention_cols = [f"cancer_in_{c}" for c in PRIORITY_COLS]
    negative_mention_cols = [f"negative_in_{c}" for c in PRIORITY_COLS]

    df = df.with_columns([
        pl.sum_horizontal([pl.col(c) for c in cancer_mention_cols if c in df.columns]).alias("n_cancer_mentions"),
        pl.sum_horizontal([pl.col(c) for c in negative_mention_cols if c in df.columns]).alias("n_negative_mentions"),
    ])

    # --- Step 5: Confidence category ---
    # Start with the base condition
    confidence_expr = pl.when(pl.col("onco_trap_in_sample_name")).then(pl.lit("uncertain_onco_trap"))
    
    # Add the DEFINITE CANCER rule: tumor in source_name or tissue (non-negated)
    tumor_condition = pl.lit(False)
    if has_tumor_source:
        tumor_condition = tumor_condition | pl.col("tumor_in_source_name")
    if has_tumor_tissue:
        tumor_condition = tumor_condition | pl.col("tumor_in_tissue")
    
    confidence_expr = confidence_expr.when(tumor_condition).then(pl.lit("confident_cancer"))
    
    # Continue with other conditions
    confidence_expr = (
        confidence_expr
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
    )
    
    df = df.with_columns([confidence_expr.alias("confidence_category")])

    return df
