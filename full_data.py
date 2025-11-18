import polars as pl
import medspacy
from medspacy.ner import TargetRule
from medspacy.target_matcher import TargetMatcher

from functions import (
    classify_cancer_samples,
    medspacy_classify,
    clean_texts,
    resolve_uncertain,
    initialize_medspacy_pipeline,
    generate_disease_rules,
    get_nlp,
    get_default_target_rules
)


if __name__ == "__main__":
    # Step 1: Get the default rules
    cancer_rules, non_cancer_rules = get_default_target_rules()
    
    # Combine them into a single list for existing_rules
    existing_rules = cancer_rules + non_cancer_rules
    
    # Step 2: Initialize the pipeline with these rules
    nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)

    # Step 3: Load data and generate disease-specific rules
    all_samples = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        schema_overrides={"group": pl.Utf8},
        infer_schema_length=0,
    )

    unique_diseases = all_samples.select("disease").unique().to_series().to_list()

    # Generate and add disease-specific rules
    auto_rules, skipped = generate_disease_rules(unique_diseases, nlp, existing_rules)

    if auto_rules:
        tm = nlp.get_pipe("medspacy_target_matcher")
        tm.add(auto_rules)
        print(f"Added {len(auto_rules)} auto-generated disease rules")
        if skipped:
            print(f"Skipped {len(skipped)} duplicate literals")

    # Step 4: Do initial prediction with regex-based classifier
    predicted_df = classify_cancer_samples(all_samples)

    # Step 5: Filter uncertain samples that need medspacy analysis
    uncertain_df = predicted_df.filter(
        pl.col("confidence_category").is_in([
            "uncertain_no_signal",
            "uncertain_weak_signal",
            "likely_non_cancer"
        ])
    )

    # Step 6: Process uncertain samples with medspacy
    results = []
    for row in uncertain_df.iter_rows(named=True):
        texts = clean_texts(row)
        if texts:  # skip empty rows
            result = medspacy_classify(texts, nlp)  # Pass nlp pipeline
        else:
            result = "NO_SIGNAL"
        results.append(result)

    # Add medspacy results to uncertain_df
    uncertain_df = uncertain_df.with_columns(
        pl.Series("medspacy_detected_cancer", results)
    )

    # Step 7: Join medspacy results back to main dataframe
    # Use a unique identifier for joining (adjust based on your data)
    uncertain_df = uncertain_df.unique(subset=["run_accession"], keep="first")

    predicted_df = predicted_df.join(
        uncertain_df.select(["run_accession", "medspacy_detected_cancer"]),
        on=["run_accession"],
        how="left",
    )

    # Step 8: Create final classification using resolve_uncertain
    predicted_df = predicted_df.with_columns(
        pl.struct(["confidence_category", "medspacy_detected_cancer"])
        .map_elements(
            lambda row: resolve_uncertain(
                row["confidence_category"],
                row["medspacy_detected_cancer"]
            ),
            return_dtype=pl.Utf8,
        )
        .alias("final_classification")
    )

    # Step 9: Display results
    idx = predicted_df.columns.index("cancer_type")
    cols_to_keep = ["experiment_alias", "source_name", "tissue", "phenotype", "disease"]
    cols_to_keep += ["final_classification", "medspacy_detected_cancer"]

    print(predicted_df.select(cols_to_keep))

    # Step 10: Filter for specific classifications if needed
    predicted_df_filtered = (
        predicted_df
        .filter(pl.col("final_classification") == "confirmed_by_medspacy")
        .select(cols_to_keep)
    )

    print("\n=== Confirmed by medspacy ===")
    print(predicted_df_filtered)


