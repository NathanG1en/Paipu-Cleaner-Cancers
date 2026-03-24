"""
Debug script for the cancer classification pipeline.
Extracted from test.ipynb "Run Classification Pipeline on Manual Label Data" section.

Usage:
    python debug_classification.py
    python debug_classification.py --input outputs/manual_label_not_mouse.xlsx
"""

import polars as pl
from pathlib import Path

from functions import (
    classify_cancer_samples,
    initialize_medspacy_pipeline,
    generate_disease_rules,
    get_default_target_rules,
)
from text_column_processing import TextColumnConfig, preprocess_dataframe


# ---------------------------------------------------------------------------
# Label mapping: confidence_category → final binary label
# ---------------------------------------------------------------------------
FINAL_LABEL_MAP = {
    "confident_cancer": "CANCER",
    "likely_cancer": "CANCER",
    "confirmed_by_medspacy": "CANCER",
    "confirmed_non_cancer": "NON_CANCER",
    "likely_non_cancer": "NON_CANCER",
    "uncertain_no_signal": "NON_CANCER",
    "uncertain_weak_signal": "NON_CANCER",
    "uncertain_medspacy": "NON_CANCER",
    "NO_SIGNAL": "NON_CANCER",
}


def load_data(input_path: str = "outputs/manual_label_not_mouse.xlsx") -> pl.DataFrame:
    """Load the manually labeled data from Excel."""
    df = pl.read_excel(input_path)

    # Drop trailing junk column if present (dtype fallback issue)
    if len(df.columns) > 14:
        df = df.select(df.columns[:-1])

    # Remove cancer_type column to prevent it from influencing classification
    if "cancer_type" in df.columns:
        df = df.drop("cancer_type")
        print("✓ Removed cancer_type column from analysis")

    print(f"Loaded {len(df)} samples from {input_path}")
    print(f"Columns: {df.columns}")
    return df


def setup_nlp_pipeline(df: pl.DataFrame):
    """
    Initialize the MedSpaCy pipeline with default + auto-generated disease rules.

    Returns:
        nlp: The configured MedSpaCy pipeline
    """
    cancer_rules, non_cancer_rules = get_default_target_rules()
    nlp = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)

    # Generate disease-specific rules from the data
    unique_diseases = df.select("disease").drop_nulls().unique().to_series().to_list()
    auto_rules, skipped = generate_disease_rules(
        unique_diseases, nlp, cancer_rules + non_cancer_rules
    )

    if auto_rules:
        tm = nlp.get_pipe("medspacy_target_matcher")
        tm.add(auto_rules)
        print(f"Added {len(auto_rules)} auto-generated disease rules")

    if skipped:
        print(f"Skipped {len(skipped)} diseases (already covered): {skipped[:5]}...")

    return nlp


def preprocess(df: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    """
    Preprocess text columns: normalize and create _norm columns.

    Returns:
        (preprocessed_df, col_tiers)
    """
    config = TextColumnConfig()
    df, col_tiers = preprocess_dataframe(df, config=config, include_discovered=False)
    print(
        f"Created {len(col_tiers['normalized'])} normalized columns: {col_tiers['normalized']}"
    )
    return df, col_tiers


def run_classification(df: pl.DataFrame, nlp) -> pl.DataFrame:
    """
    Run the full classification pipeline (regex + MedSpaCy).

    Returns:
        DataFrame with classification columns added.
    """
    predicted_df = classify_cancer_samples(
        df, nlp_pipeline=nlp, batch_size=64, use_normalized=True
    )

    # Map confidence_category → final binary label
    # NOTE: Using confidence_category (combined regex+MedSpaCy), NOT med_label alone
    predicted_df = predicted_df.with_columns(
        pl.col("confidence_category").replace(FINAL_LABEL_MAP).alias("final_label")
    )

    return predicted_df


def evaluate(predicted_df: pl.DataFrame) -> dict:
    """
    Compare predictions to ground truth (is_cancer column).

    Returns:
        Dictionary with TP, TN, FP, FN counts and metrics.
    """
    # Filter to only rows with ground truth labels
    labeled = predicted_df.filter(pl.col("is_cancer").is_not_null())
    unlabeled_count = len(predicted_df) - len(labeled)

    if unlabeled_count > 0:
        print(f"\n⚠️  {unlabeled_count} samples have no ground truth (is_cancer=null)")

    # Confusion matrix
    true_positives = labeled.filter(
        (pl.col("final_label") == "CANCER") & (pl.col("is_cancer") == 1)
    )
    true_negatives = labeled.filter(
        (pl.col("final_label") == "NON_CANCER") & (pl.col("is_cancer") == 0)
    )
    false_positives = labeled.filter(
        (pl.col("final_label") == "CANCER") & (pl.col("is_cancer") == 0)
    )
    false_negatives = labeled.filter(
        (pl.col("final_label") == "NON_CANCER") & (pl.col("is_cancer") == 1)
    )

    tp, tn, fp, fn = (
        len(true_positives),
        len(true_negatives),
        len(false_positives),
        len(false_negatives),
    )

    print(f"\n{'=' * 50}")
    print(f"=== Classification Summary ===")
    print(f"{'=' * 50}")
    summary = (
        predicted_df.group_by("final_label")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    print(summary)

    print(f"\n{'=' * 50}")
    print(f"=== Predictions vs Ground Truth ===")
    print(f"{'=' * 50}")
    comparison = (
        predicted_df.group_by("final_label", "is_cancer")
        .agg(pl.len().alias("count"))
        .sort("final_label", "is_cancer")
    )
    print(comparison)

    print(f"\n{'=' * 50}")
    print(f"=== Confusion Matrix ===")
    print(f"{'=' * 50}")
    print(f"True Positives:  {tp}")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")

    if tp + fn > 0:
        recall = tp / (tp + fn) * 100
        print(f"\nRecall (Sensitivity): {recall:.1f}% ({tp}/{tp + fn})")
    if tn + fp > 0:
        specificity = tn / (tn + fp) * 100
        print(f"Specificity:          {specificity:.1f}% ({tn}/{tn + fp})")
    if tp + fp > 0:
        precision = tp / (tp + fp) * 100
        print(f"Precision:            {precision:.1f}% ({tp}/{tp + fp})")

    # Show false negatives detail
    print(f"\n{'=' * 50}")
    print(f"=== False Negatives ({fn}) ===")
    print(f"{'=' * 50}")
    if fn > 0:
        display_cols = [
            "title",
            "tissue",
            "disease",
            "source_name",
            "regex_label",
            "med_label",
            "confidence_category",
            "final_label",
        ]
        display_cols = [c for c in display_cols if c in false_negatives.columns]
        print(false_negatives.select(display_cols))

    # Show false positives detail
    print(f"\n{'=' * 50}")
    print(f"=== False Positives ({fp}) ===")
    print(f"{'=' * 50}")
    if fp > 0:
        display_cols = [
            "title",
            "tissue",
            "disease",
            "source_name",
            "regex_label",
            "med_label",
            "confidence_category",
            "final_label",
        ]
        display_cols = [c for c in display_cols if c in false_positives.columns]
        print(false_positives.select(display_cols))

    return {
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def export_results(predicted_df: pl.DataFrame, results: dict):
    """Export false positives and false negatives to Excel for review."""
    export_cols = [
        "title",
        "tissue",
        "disease",
        "source_name",
        "final_label",
        "is_cancer",
        "regex_label",
        "regex_reason",
        "med_label",
        "med_reason",
        "med_source_columns",
        "confidence_category",
        "run_accession",
        "bioproject",
    ]
    export_cols = [c for c in export_cols if c in predicted_df.columns]

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    fp = results["false_positives"]
    fn = results["false_negatives"]

    if len(fp) > 0:
        fp.select(export_cols).write_excel(output_dir / "false_positives.xlsx")
        print(f"✓ Exported {len(fp)} false positives → outputs/false_positives.xlsx")

    if len(fn) > 0:
        fn.select(export_cols).write_excel(output_dir / "false_negatives.xlsx")
        print(f"✓ Exported {len(fn)} false negatives → outputs/false_negatives.xlsx")

    # Export full results
    predicted_df.select(export_cols).write_excel(output_dir / "full_predictions.xlsx")
    print(f"✓ Exported full predictions → outputs/full_predictions.xlsx")


def main(input_path: str = "outputs/manual_label_not_mouse.xlsx"):
    """Run the full classification pipeline end-to-end."""
    print("=" * 60)
    print("CANCER CLASSIFICATION PIPELINE - DEBUG MODE")
    print("=" * 60)

    # Step 1: Load data
    print("\n--- Step 1: Loading data ---")
    df = load_data(input_path)

    # Step 2: Setup NLP pipeline
    print("\n--- Step 2: Setting up NLP pipeline ---")
    nlp = setup_nlp_pipeline(df)

    # Step 3: Preprocess text columns
    print("\n--- Step 3: Preprocessing text columns ---")
    df, col_tiers = preprocess(df)

    # Step 4: Run classification
    print("\n--- Step 4: Running classification ---")
    predicted_df = run_classification(df, nlp)

    # Re-attach is_cancer from the original dataframe
    if "is_cancer" not in predicted_df.columns:
        predicted_df = predicted_df.with_columns(df["is_cancer"].alias("is_cancer"))

    print(f"\nAvailable columns after classification: {predicted_df.columns}")

    # Step 5: Evaluate
    print("\n--- Step 5: Evaluating predictions ---")
    results = evaluate(predicted_df)

    # Step 6: Export
    print("\n--- Step 6: Exporting results ---")
    export_results(predicted_df, results)

    return predicted_df, results


if __name__ == "__main__":
    import sys

    input_file = (
        sys.argv[1] if len(sys.argv) > 1 else "data/manual_label_not_mouse.xlsx"
    )
    predicted_df, results = main(input_file)
