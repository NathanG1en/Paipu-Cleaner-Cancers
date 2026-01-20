# full_data.py - Updated with preprocessing step

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

if __name__ == "__main__":
    # Ensure outputs folder exists
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Step 1: Load raw data
    # =========================================================================
    print("Loading data...")
    all_samples = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        schema_overrides={"group": pl.Utf8},
        infer_schema_length=0,
    )
    print(f"Total samples loaded: {len(all_samples)}")
    print(f"Total columns: {len(all_samples.columns)}")

    # =========================================================================
    # Step 2: Text Preprocessing Pipeline (NEW)
    # =========================================================================
    print("\n=== Text Preprocessing ===")

    # Configure preprocessing (use defaults or customize)
    config = TextColumnConfig(
        # Override defaults if needed:
        # min_avg_length=8.0,
        # min_non_null_pct=0.005,
    )

    # Identify column tiers (for reporting)
    col_tiers = identify_viable_text_columns(all_samples, config)
    print(f"Priority columns found: {len(col_tiers['priority'])} - {col_tiers['priority']}")
    print(f"Secondary columns found: {len(col_tiers['secondary'])} - {col_tiers['secondary']}")
    print(f"Auto-discovered columns: {len(col_tiers['discovered'])}")
    if col_tiers['discovered']:
        print(f"  First 10: {col_tiers['discovered'][:10]}")

    # Preprocess: normalize text columns (creates _norm suffix versions)
    all_samples, col_tiers = preprocess_dataframe(
        all_samples,
        config=config,
        include_discovered=False,  # Set True for comprehensive scanning
    )
    print(f"Created {len(col_tiers['normalized'])} normalized columns")

    # =========================================================================
    # Step 3: Initialize MedSpaCy Pipeline
    # =========================================================================
    cancer_rules, non_cancer_rules = get_default_target_rules()
    existing_rules = cancer_rules + non_cancer_rules

    print("\nInitializing medspacy pipeline...")
    nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)
    print(f"Pipeline initialized. Total rules: {len(nlp.get_pipe('medspacy_target_matcher').rules)}")

    # =========================================================================
    # Step 4: Generate disease-specific rules
    # =========================================================================
    unique_diseases = all_samples.select("disease").unique().to_series().to_list()
    auto_rules, skipped = generate_disease_rules(unique_diseases, nlp, existing_rules)

    if auto_rules:
        tm = nlp.get_pipe("medspacy_target_matcher")
        tm.add(auto_rules)
        print(f"Added {len(auto_rules)} auto-generated disease rules")

    # =========================================================================
    # Step 5: Regex-based classification
    # =========================================================================
    # Pass normalized column info so classifier can use pre-normalized text
    predicted_df = classify_cancer_samples(
        all_samples,
        use_normalized=True,  # Use _norm columns if available
    )

    print("\n=== Regex Classification Summary ===")
    regex_summary = (
        predicted_df
        .group_by("confidence_category")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(regex_summary)

    # =========================================================================
    # Step 6: Filter uncertain samples for MedSpaCy
    # =========================================================================
    uncertain_df = predicted_df.filter(
        pl.col("confidence_category").is_in([
            "uncertain_no_signal",
            "uncertain_weak_signal",
            "likely_non_cancer"
        ])
    )

    print(f"\n=== Medspacy Processing ===")
    print(f"Samples requiring medspacy analysis: {len(uncertain_df)}")

    # =========================================================================
    # Step 7: Process with MedSpaCy (batched)
    # =========================================================================
    all_texts = []
    for row in uncertain_df.iter_rows(named=True):
        texts = clean_texts(row, use_normalized=True)
        all_texts.append(texts if texts else [])

    print(f"Processing {len(all_texts)} samples in batches...")

    med_labels, med_reasons = medspacy_classify_batch(all_texts, nlp, batch_size=64)

    uncertain_df = uncertain_df.with_columns([
        pl.Series("med_label", med_labels),
        pl.Series("med_reason", med_reasons),
    ])

    # Stats
    print(f"\n=== Medspacy Results ===")
    print(f"CANCER detected: {med_labels.count('CANCER')}")
    print(f"NOT_CANCER detected: {med_labels.count('NOT_CANCER')}")
    print(f"NO_SIGNAL: {med_labels.count('NO_SIGNAL')}")
    print(f"UNCERTAIN: {med_labels.count('UNCERTAIN')}")

    # =========================================================================
    # Step 8: Join MedSpaCy results back
    # =========================================================================
    uncertain_df = uncertain_df.unique(subset=["run_accession"], keep="first")

    predicted_df = predicted_df.join(
        uncertain_df.select(["run_accession", "med_label", "med_reason"]),
        on="run_accession",
        how="left",
    )

    predicted_df = predicted_df.with_columns([
        pl.col("med_label").fill_null(""),
        pl.col("med_reason").fill_null(""),
    ])

    # =========================================================================
    # Step 9: Resolve final classification
    # =========================================================================
    final_labels = []
    regex_labels = []
    med_labels_out = []
    regex_reasons = []
    med_reasons_out = []

    for row in predicted_df.iter_rows(named=True):
        result = resolve_uncertain(
            regex_label=row["confidence_category"],
            med_label=row["med_label"],
            regex_reason=row.get("decision_reason", ""),
            med_reason=row["med_reason"],
        )
        final_labels.append(result[0])
        regex_labels.append(result[1])
        med_labels_out.append(result[2])
        regex_reasons.append(result[3])
        med_reasons_out.append(result[4])

    predicted_df = predicted_df.with_columns([
        pl.Series("final_label", final_labels),
        pl.Series("regex_label", regex_labels),
        pl.Series("regex_reason", regex_reasons),
    ])

    # =========================================================================
    # Step 10: Display results
    # =========================================================================
    print("\n=== Final Classification Summary ===")
    final_summary = (
        predicted_df
        .group_by("final_label")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(final_summary)

    # =========================================================================
    # Step 11: Export results
    # =========================================================================
    # Define columns to keep (exclude _norm columns from export)
    cols_to_keep = [
        "run_accession", "experiment_alias", "bioproject",
        "source_name", "tissue", "phenotype", "disease", "cell_type", "tumor_type",
        "sample_name", "condition", "tumor", "cell_type.2", "cell_type.3",
        "celltype", "tissue_type", "health_state", "tissue_cell_type_source",
        "source", "model", "tissue_cell_type", "cell_types", "cancer_type",
        "final_label", "regex_label", "med_label", "regex_reason", "med_reason",
    ]

    cols_to_keep = [c for c in cols_to_keep if c in predicted_df.columns]

    output_file = output_dir / "classified_samples.csv"
    predicted_df.select(cols_to_keep).write_csv(output_file)
    print(f"\n✓ Exported full results to: {output_file}")

    # =========================================================================
    # Step 12: Export confirmed_by_medspacy subset
    # =========================================================================
    predicted_df_filtered = (
        predicted_df
        .filter(pl.col("final_label") == "confirmed_by_medspacy")
        .select(cols_to_keep)
    )

    print(f"\n=== Confirmed by medspacy ({len(predicted_df_filtered)} samples) ===")
    print(predicted_df_filtered.head(20))

    confirmed_output = output_dir / "confirmed_by_medspacy.csv"
    predicted_df_filtered.write_csv(confirmed_output)
    print(f"✓ Exported medspacy-confirmed results to: {confirmed_output}")