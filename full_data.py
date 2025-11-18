import polars as pl
import medspacy
from medspacy.ner import TargetRule
from medspacy.target_matcher import TargetMatcher
from pathlib import Path

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
    # ensure outputs directory exists
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Get the default rules
    cancer_rules, non_cancer_rules = get_default_target_rules()
    
    # Combine them into a single list for existing_rules
    existing_rules = cancer_rules + non_cancer_rules
    
    # Step 2: Initialize the pipeline with these rules
    print("Initializing medspacy pipeline...")
    nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)
    print(f"Pipeline initialized. Total rules: {len(nlp.get_pipe('medspacy_target_matcher').rules)}")

    # Step 3: Load data and generate disease-specific rules
    all_samples = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        schema_overrides={"group": pl.Utf8},
        infer_schema_length=0,
    )

    print(f"\nTotal samples loaded: {len(all_samples)}")

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

    # Print regex classification summary
    print("\n=== Regex Classification Summary ===")
    regex_summary = (
        predicted_df
        .group_by("confidence_category")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(regex_summary)

    # Step 5: Filter uncertain samples that need medspacy analysis
    uncertain_df = predicted_df.filter(
        pl.col("confidence_category").is_in([
            "uncertain_no_signal", 
            "uncertain_weak_signal", 
            "likely_non_cancer"
        ])
    )

    print(f"\n=== Medspacy Processing ===")
    print(f"Samples requiring medspacy analysis: {len(uncertain_df)}")

    # Step 6: Process uncertain samples with medspacy
    results = []
    cancer_detected = 0
    not_cancer_detected = 0
    no_signal = 0
    
    for i, row in enumerate(uncertain_df.iter_rows(named=True)):
        texts = clean_texts(row)
        if texts:  # skip empty rows
            result = medspacy_classify(texts, nlp)  # Pass nlp pipeline
            if result == "CANCER":
                cancer_detected += 1
            elif result == "NOT_CANCER":
                not_cancer_detected += 1
            elif result == "NO_SIGNAL":
                no_signal += 1
        else:
            result = "NO_SIGNAL"
            no_signal += 1
        results.append(result)
        
        # Print progress every 100 samples
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(uncertain_df)} samples...")

    print(f"\n=== Medspacy Results ===")
    print(f"CANCER detected: {cancer_detected}")
    print(f"NOT_CANCER detected: {not_cancer_detected}")
    print(f"NO_SIGNAL: {no_signal}")
    print(f"UNCERTAIN: {len(results) - cancer_detected - not_cancer_detected - no_signal}")

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
    print("\n=== Final Classification Summary ===")
    final_summary = (
        predicted_df
        .group_by("final_classification")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )
    print(final_summary)

    # Define columns to keep (including run_accession)
    cols_to_keep = [
        "run_accession",
        "experiment_alias", 
        "source_name", 
        "tissue", 
        "phenotype", 
        "disease",
        "final_classification", 
        "medspacy_detected_cancer"
    ]

    print("\n=== Sample Results ===")
    print(predicted_df.select(cols_to_keep).head(20))

    # Step 10: Export minimal CSV to outputs folder
    output_file = output_dir / "classified_samples.csv"
    predicted_df.select(cols_to_keep).write_csv(output_file)
    print(f"\n✓ Exported full results to: {output_file}")

    # Step 11: Filter for specific classifications if needed
    predicted_df_filtered = (
        predicted_df
        .filter(pl.col("final_classification") == "confirmed_by_medspacy")
        .select(cols_to_keep)
    )

    print(f"\n=== Confirmed by medspacy ({len(predicted_df_filtered)} samples) ===")
    print(predicted_df_filtered.head(20))
    
    # Export confirmed_by_medspacy subset
    confirmed_output = output_dir / "confirmed_by_medspacy.csv"
    predicted_df_filtered.write_csv(confirmed_output)
    print(f"✓ Exported medspacy-confirmed results to: {confirmed_output}")
