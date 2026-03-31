# Import your functions from the new module
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
)
import os
from fallback import ExpandedSearchProvider
from providers import NCBIProvider, OllamaProvider, GeminiProvider

if __name__ == "__main__":
    # Initialize the NLP pipeline once
    nlp = get_nlp()

    # Read the manually-labelled xlsx dataset
    df = pl.read_excel("data/manual_label_not_mouse.xlsx")

    # Keep only the essential labeled columns from df (and drop any unnamed)
    label_cols = ["run_accession", "is_cancer"]
    if "cancer_type" in df.columns:
        label_cols.append("cancer_type")
    labeled_df = df.select([c for c in label_cols if c in df.columns])

    # Read the full metadata to get all columns
    combined_df = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        infer_schema_length=10000,
        ignore_errors=True,
    ).unique(subset=["run_accession"], keep="first")
    if "cancer_type" in combined_df.columns and "cancer_type" in labeled_df.columns:
        combined_df = combined_df.drop("cancer_type")

    # Join on run_accession using a left join to ensure we only keep rows
    # from the manual labels, preserving the exact rows you started with
    df = labeled_df.join(combined_df, on="run_accession", how="left")

    # Preserve ground-truth labels for validation, then remove them
    # so the classifier cannot use is_cancer or cancer_type as features
    ground_truth = df.select("run_accession", "is_cancer")
    df = df.drop("is_cancer", "cancer_type")

    # Build fallback provider chain
    # fallback_providers = [
    #     ExpandedSearchProvider(nlp_pipeline=nlp),
    #     NCBIProvider(email="nathanglen@ufl.edu"),
    # ]

    # Add LLM fallback if API key is available
    # gemini_key = os.environ.get("GEMINI_API_KEY")
    # if gemini_key:
    #     fallback_providers.append(GeminiProvider(api_key=gemini_key))
    #     print("LLM fallback enabled (Gemini)")
    # else:
    #     print("LLM fallback disabled (set GEMINI_API_KEY to enable)")

    # Add Ollama fallback
    # fallback_providers.append(OllamaProvider(model="llama3.2"))
    # print("LLM fallback enabled (Ollama)")
    fallback_providers = []
    predicted_df = classify_cancer_samples(
        df,
        fallback_providers=fallback_providers,
    )

    # Re-attach ground-truth is_cancer for validation
    predicted_df = predicted_df.join(ground_truth, on="run_accession", how="left")

    # confidence_category is now the final definitive classification
    # (no more uncertain categories)
    predicted_df = predicted_df.with_columns(
        pl.col("confidence_category").alias("final_classification")
    )

    # Display sample of results
    cols_to_keep = [
        "experiment_alias",
        "source_name",
        "tissue",
        "phenotype",
        "disease",
        "disease_state",
        "diagnosis",
    ]
    cols_to_keep += ["is_cancer", "final_classification"]
    cols_to_keep = [c for c in cols_to_keep if c in predicted_df.columns]
    print(predicted_df.select(cols_to_keep))

    # VALIDATING
    # Map final_classification to binary predicted label
    cancer_classes = ["confident_cancer", "confirmed_by_medspacy", "likely_cancer"]
    non_cancer_classes = ["likely_non_cancer", "confirmed_non_cancer"]

    predicted_df = predicted_df.with_columns(
        pl.when(pl.col("final_classification").is_in(cancer_classes))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("predicted_cancer")
    )

    # Ground truth: is_cancer >= 1 means cancer
    predicted_df = predicted_df.with_columns(
        pl.when(pl.col("is_cancer") >= 1)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("actual_cancer")
    )

    # Confusion matrix
    tp = predicted_df.filter(
        (pl.col("predicted_cancer") == 1) & (pl.col("actual_cancer") == 1)
    ).height
    fp = predicted_df.filter(
        (pl.col("predicted_cancer") == 1) & (pl.col("actual_cancer") == 0)
    ).height
    tn = predicted_df.filter(
        (pl.col("predicted_cancer") == 0) & (pl.col("actual_cancer") == 0)
    ).height
    fn = predicted_df.filter(
        (pl.col("predicted_cancer") == 0) & (pl.col("actual_cancer") == 1)
    ).height

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    print("\n=== Confusion Matrix ===")

    print(f"number of cancers {tp + fn}")
    print(f"number of non-cancers {tn + fp}")

    print(f"  TP (cancer → cancer):         {tp}")
    print(f"  FP (non-cancer → cancer):     {fp}")
    print(f"  TN (non-cancer → non-cancer): {tn}")
    print(f"  FN (cancer → non-cancer):     {fn}")
    print(f"\n  Overall Accuracy: {accuracy:.1f}%")
    print(f"  Precision (cancer): {precision:.1f}%")
    print(f"  Recall (cancer):    {recall:.1f}%")

    cancer_acc = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    non_cancer_acc = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0
    print(f"\n  Accuracy on cancer predictions:     {cancer_acc:.1f}% ({tp}/{tp + fp})")
    print(
        f"  Accuracy on non-cancer predictions: {non_cancer_acc:.1f}% ({tn}/{tn + fn})"
    )
    # --- Save outputs ---
    import os

    os.makedirs("outputs", exist_ok=True)

    # --- False Negatives: actually cancer but predicted non-cancer ---
    false_negatives = predicted_df.filter(
        (pl.col("predicted_cancer") == 0) & (pl.col("actual_cancer") == 1)
    )
    display_cols = [
        "run_accession",
        "title",
        "source_name",
        "tissue",
        "disease",
        "disease_state",
        "diagnosis",
        "regex_label",
        "med_label",
        "med_reason",
        "final_classification",
        "resolved_by",
        "fallback_reason",
        "is_cancer",
    ]
    display_cols = [c for c in display_cols if c in predicted_df.columns]

    print(f"\n=== FALSE NEGATIVES ({false_negatives.height}) ===")
    print("(Cancer samples misclassified as non-cancer)")
    if false_negatives.height > 0:
        print(false_negatives.select(display_cols))
        false_negatives.select(display_cols).write_csv("outputs/false_negatives.csv")

    # --- False Positives: actually non-cancer but predicted cancer ---
    false_positives = predicted_df.filter(
        (pl.col("predicted_cancer") == 1) & (pl.col("actual_cancer") == 0)
    )
    print(f"\n=== FALSE POSITIVES ({false_positives.height}) ===")
    print("(Non-cancer samples misclassified as cancer)")
    if false_positives.height > 0:
        print(false_positives.select(display_cols))
        false_positives.select(display_cols).write_csv("outputs/false_positives.csv")

    # Save full predictions
    predicted_df.select(display_cols).write_csv("outputs/all_predictions.csv")
    print(
        f"\nOutputs saved to outputs/false_negatives.csv, outputs/false_positives.csv, outputs/all_predictions.csv"
    )

    # Check for any remaining uncertain classifications
    uncertain = predicted_df.filter(
        pl.col("final_classification").str.contains("uncertain")
    )
    if uncertain.height > 0:
        print(f"\nWARNING: {uncertain.height} samples still uncertain")
    else:
        print(f"\nAll {predicted_df.height} samples classified definitively.")
