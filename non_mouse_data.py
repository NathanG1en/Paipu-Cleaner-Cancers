# non_mouse_data.py - Classify non-mouse samples only

import polars as pl
from pathlib import Path

from functions import (
    classify_cancer_samples,
    medspacy_classify_batch,
    clean_texts,
    resolve_uncertain,
    initialize_medspacy_pipeline,
    generate_disease_rules,
    get_default_target_rules,
    PRIORITY_COLS,
)

from text_column_processing import (
    TextColumnConfig,
    preprocess_dataframe,
    identify_viable_text_columns,
)

FINAL_LABEL_MAP = {
    "confident_cancer": "CANCER",
    "likely_cancer": "CANCER",
    "confirmed_by_medspacy": "CANCER",
    "confirmed_non_cancer": "NON_CANCER",
    "likely_non_cancer": "NON_CANCER",
    "uncertain_no_signal": "UNCERTAIN",
    "uncertain_weak_signal": "UNCERTAIN",
    "uncertain_medspacy": "UNCERTAIN",
}

MOUSE_EXCLUDE = [
    "Mus musculus",
    "Mus musculus domesticus",
    "Mus musculus musculus",
]

if __name__ == "__main__":
    # Ensure outputs folder exists
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Step 1: Load raw data and filter out mouse samples
    # =========================================================================
    print("Loading data...")
    all_samples = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        schema_overrides={"group": pl.Utf8},
        infer_schema_length=0,
    )

    print(f"Total samples loaded: {len(all_samples)}")

    # Filter out mouse samples
    non_mouse = all_samples.filter(
        ~pl.col("organism_scientific_name").is_in(MOUSE_EXCLUDE)
    )
    print(
        f"Non-mouse samples: {len(non_mouse)} (excluded {len(all_samples) - len(non_mouse)} mouse samples)"
    )
    print(f"Total columns: {len(non_mouse.columns)}")

    # =========================================================================
    # Step 2: Text Preprocessing Pipeline
    # =========================================================================
    print("\n=== Text Preprocessing ===")

    config = TextColumnConfig()

    col_tiers = identify_viable_text_columns(non_mouse, config)
    print(
        f"Priority columns found: {len(col_tiers['priority'])} - {col_tiers['priority']}"
    )
    print(
        f"Secondary columns found: {len(col_tiers['secondary'])} - {col_tiers['secondary']}"
    )
    print(f"Auto-discovered columns: {len(col_tiers['discovered'])}")
    if col_tiers["discovered"]:
        print(f"  First 10: {col_tiers['discovered'][:10]}")

    non_mouse, col_tiers = preprocess_dataframe(
        non_mouse,
        config=config,
        include_discovered=False,
    )
    print(f"Created {len(col_tiers['normalized'])} normalized columns")

    # =========================================================================
    # Step 3: Initialize MedSpaCy Pipeline
    # =========================================================================
    from functions import (
        get_nlp,
        reset_nlp,
        NLPPipelineManager,
        generate_disease_rules,
        get_default_target_rules,
    )

    print("\nInitializing medspacy pipeline...")
    nlp = get_nlp()
    print(
        "Pipeline initialized. Total rules: {}".format(
            NLPPipelineManager.get_rule_count()
        )
    )

    # =========================================================================
    # Step 4: Generate disease-specific rules
    # =========================================================================
    cancer_rules, non_cancer_rules = get_default_target_rules()
    unique_diseases = non_mouse.select("disease").unique().to_series().to_list()
    auto_rules, skipped = generate_disease_rules(
        unique_diseases, nlp, cancer_rules + non_cancer_rules
    )

    if auto_rules:
        NLPPipelineManager.add_rules(auto_rules)
        print("Added {} auto-generated disease rules".format(len(auto_rules)))

    # =========================================================================
    # Step 5: Classification
    # =========================================================================
    print("\n=== Classification ===")
    predicted_df = classify_cancer_samples(
        non_mouse,
        nlp_pipeline=nlp,
        batch_size=64,
        use_normalized=True,
        use_fallback=False,
    )

    predicted_df = predicted_df.with_columns(
        pl.col("confidence_category").replace(FINAL_LABEL_MAP).alias("final_label")
    )

    print("\n=== Classification Summary ===")
    summary = (
        predicted_df.group_by("med_label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print(summary)

    # =========================================================================
    # Step 6: Display results
    # =========================================================================
    print("\n=== Final Classification Summary ===")
    final_summary = (
        predicted_df.group_by("final_label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print(final_summary)

    # =========================================================================
    # Step 7: Export results
    # =========================================================================
    cols_to_keep = [
        "run_accession",
        "experiment_alias",
        "bioproject",
        "organism_scientific_name",
        "source_name",
        "tissue",
        "phenotype",
        "disease",
        "cell_type",
        "tumor_type",
        "sample_name",
        "condition",
        "tumor",
        "cell_type.2",
        "cell_type.3",
        "celltype",
        "tissue_type",
        "health_state",
        "tissue_cell_type_source",
        "source",
        "model",
        "tissue_cell_type",
        "cell_types",
        "cancer_type",
        "is_cell_line",
        "is_benign",
        "final_label",
        "regex_label",
        "med_label",
        "regex_reason",
        "med_reason",
    ]

    cols_to_keep = [c for c in cols_to_keep if c in predicted_df.columns]

    output_file = output_dir / "non_mouse_classified.csv"
    predicted_df.select(cols_to_keep).write_csv(output_file)
    print(f"\n✓ Exported results to: {output_file}")

    # =========================================================================
    # Step 8: Export confirmed_by_medspacy subset
    # =========================================================================
    predicted_df_filtered = predicted_df.filter(
        pl.col("final_label") == "confirmed_by_medspacy"
    ).select(cols_to_keep)

    print(f"\n=== Confirmed by medspacy ({len(predicted_df_filtered)} samples) ===")
    print(predicted_df_filtered.head(20))
