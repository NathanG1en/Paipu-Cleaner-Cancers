# %%
import pandas as pd
import spacy
import medspacy
from medspacy.ner import TargetRule

# --- Setup medspacy with TargetMatcher (run once) ---
nlp = medspacy.load()
target_matcher = nlp.get_pipe("medspacy_target_matcher")

# Define cancer-related terms to detect
cancer_rules = [
    TargetRule("cancer", "PROBLEM"),
    TargetRule("cancers", "PROBLEM"),
    TargetRule("tumor", "PROBLEM"),
    TargetRule("tumour", "PROBLEM"),
    TargetRule("tumors", "PROBLEM"),
    TargetRule("tumours", "PROBLEM"),
    TargetRule("malignant", "PROBLEM"),
    TargetRule("malignancy", "PROBLEM"),
    TargetRule("carcinoma", "PROBLEM"),
    TargetRule("adenocarcinoma", "PROBLEM"),
    TargetRule("sarcoma", "PROBLEM"),
    TargetRule("melanoma", "PROBLEM"),
    TargetRule("glioma", "PROBLEM"),
    TargetRule("glioblastoma", "PROBLEM"),
    TargetRule("leukemia", "PROBLEM"),
    TargetRule("leukaemia", "PROBLEM"),
    TargetRule("lymphoma", "PROBLEM"),
    TargetRule("myeloma", "PROBLEM"),
    TargetRule("metastatic", "PROBLEM"),
    TargetRule("metastasis", "PROBLEM"),
    TargetRule("metastases", "PROBLEM"),
    TargetRule("neoplasm", "PROBLEM"),
    TargetRule("neoplastic", "PROBLEM"),
]

target_matcher.add(cancer_rules)
print(f"✓ Loaded medspacy with {len(cancer_rules)} cancer target rules")


def analyze_text_with_medspacy(text: str) -> dict:
    """
    Analyze text using medspacy with TargetMatcher and context detection.
    Returns counts of affirmed vs negated cancer mentions.
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "affirmed_count": 0,
            "negated_count": 0,
            "uncertain_count": 0
        }

    doc = nlp(text.lower())

    affirmed = 0
    negated = 0
    uncertain = 0

    for ent in doc.ents:
        if ent.label_ == "PROBLEM":  # Our cancer entities
            if hasattr(ent._, "is_negated") and ent._.is_negated:
                negated += 1
            elif hasattr(ent._, "is_uncertain") and ent._.is_uncertain:
                uncertain += 1
            elif hasattr(ent._, "is_hypothetical") and ent._.is_hypothetical:
                uncertain += 1
            else:
                affirmed += 1

    return {
        "affirmed_count": affirmed,
        "negated_count": negated,
        "uncertain_count": uncertain
    }


def classify_cancer_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify samples as cancer / non-cancer / uncertain using medspacy.
    Returns the same DataFrame with added classification columns.
    """

    # Make a copy to avoid modifying the original
    df = df.copy()

    # --- Step 1: Setup ---
    PRIORITY_COLS = ["source_name", "tissue", "phenotype", "disease", "cell_type", "tumor_type"]
    PRIORITY_COLS = [c for c in PRIORITY_COLS if c in df.columns]

    # Onco-traps pattern (still useful for false positives)
    ONCO_TRAPS = r"(?:\boncophora\b|\boncorhynchus\b|\boncotic\b|\boncomodulin\b)"

    # Helper to normalize text columns
    def normalize_text(series):
        return (
            series.astype(str)
            .fillna("")
            .str.lower()
            .str.replace(r"[_/|]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # --- Step 2: Sample name detection using medspacy ---
    sample_name_col = "title" if "title" in df.columns else "biosample"

    # Extract texts for sample names
    sample_texts = normalize_text(df[sample_name_col]).tolist()

    # Analyze each sample name with medspacy
    sample_results = [analyze_text_with_medspacy(text) for text in sample_texts]

    df["cancer_in_sample_name"] = [r["affirmed_count"] > 0 for r in sample_results]
    df["negative_in_sample_name"] = [r["negated_count"] > 0 for r in sample_results]
    df["sample_name_affirmed_count"] = [r["affirmed_count"] for r in sample_results]
    df["sample_name_negated_count"] = [r["negated_count"] for r in sample_results]
    df["onco_trap_in_sample_name"] = normalize_text(df[sample_name_col]).str.contains(ONCO_TRAPS, regex=True)

    # --- Step 3: Check priority columns with medspacy ---
    for col in PRIORITY_COLS:
        col_texts = normalize_text(df[col]).tolist()
        col_results = [analyze_text_with_medspacy(text) for text in col_texts]

        df[f"cancer_in_{col}"] = [r["affirmed_count"] > 0 for r in col_results]
        df[f"negative_in_{col}"] = [r["negated_count"] > 0 for r in col_results]
        df[f"{col}_affirmed_count"] = [r["affirmed_count"] for r in col_results]
        df[f"{col}_negated_count"] = [r["negated_count"] for r in col_results]

    # --- Step 4: Count mentions across all priority columns ---
    cancer_mention_cols = [f"cancer_in_{c}" for c in PRIORITY_COLS]
    negative_mention_cols = [f"negative_in_{c}" for c in PRIORITY_COLS]

    # Count total affirmed cancer mentions (weighted by counts)
    affirmed_count_cols = ["sample_name_affirmed_count"] + [f"{c}_affirmed_count" for c in PRIORITY_COLS]
    negated_count_cols = ["sample_name_negated_count"] + [f"{c}_negated_count" for c in PRIORITY_COLS]

    df["n_cancer_mentions"] = df[[c for c in cancer_mention_cols if c in df.columns]].sum(axis=1)
    df["n_negative_mentions"] = df[[c for c in negative_mention_cols if c in df.columns]].sum(axis=1)
    df["total_affirmed_count"] = df[[c for c in affirmed_count_cols if c in df.columns]].sum(axis=1)
    df["total_negated_count"] = df[[c for c in negated_count_cols if c in df.columns]].sum(axis=1)

    # --- Step 5: Enhanced confidence category with negation awareness ---
    def assign_confidence(row):
        if row["onco_trap_in_sample_name"]:
            return "uncertain_onco_trap"

        # Strong negation signal - likely non-cancer
        if row["total_negated_count"] >= row["total_affirmed_count"] + 1:
            return "likely_non_cancer"

        # HIGH CONFIDENCE: Cancer in sample name AND in priority columns, no negations
        if (row["cancer_in_sample_name"] and
                row["n_cancer_mentions"] >= 1 and
                row["total_negated_count"] == 0):
            return "confident_cancer"

        # MEDIUM CONFIDENCE: Multiple cancer mentions, minimal negations
        if (((row["cancer_in_sample_name"] and row["total_negated_count"] == 0) or
             row["n_cancer_mentions"] >= 2) and
                row["total_negated_count"] <= row["total_affirmed_count"] // 2):
            return "likely_cancer"

        # LIKELY NON-CANCER: Negative indicators present
        if (row["negative_in_sample_name"] or
                row["n_negative_mentions"] >= 1 or
                row["total_negated_count"] > 0):
            return "likely_non_cancer"

        # UNCERTAIN: Weak signal
        if row["n_cancer_mentions"] == 1:
            return "uncertain_weak_signal"

        # UNCERTAIN: No cancer mentions at all
        return "uncertain_no_signal"

    df["confidence_category"] = df.apply(assign_confidence, axis=1)

    return df


print("✓ Classification function ready")

# %%
# Load the datasets using pandas
full_dataset = pd.read_csv(
    "data/combined_metadata_noncancer_removed.csv",
    dtype={"group": str},
    low_memory=False
)

df = pd.read_csv("data/train_test.csv")

# %%
# Select the columns you need from df
df = df[["experiment_accession", "bioproject", "biosample", "sample_accession", "run_accession", "is_cancer"]]

# %%
# Join the datasets
joined = full_dataset.merge(
    df,
    on=["experiment_accession", "bioproject", "biosample", "sample_accession", "run_accession"],
    how="inner"
)

print(f"✓ Joined dataset shape: {joined.shape}")
print(f"✓ Columns in joined dataset: {len(joined.columns)}")

# %%
# Apply the classification function
print("Starting classification...")
predicted_df = classify_cancer_samples(joined)
print(f"✓ Classification complete! Shape: {predicted_df.shape}")

# %%
# Find the position of "cancer_type" column if it exists
if "cancer_type" in predicted_df.columns:
    idx = predicted_df.columns.get_loc("cancer_type")
    # Select columns up to cancer_type + the two specific columns
    cols_to_keep = list(predicted_df.columns[:idx + 1]) + ["is_cancer", "confidence_category"]
    result_df = predicted_df[cols_to_keep]
else:
    # If cancer_type doesn't exist, just show key columns
    key_cols = ["experiment_accession", "biosample", "sample_accession", "is_cancer", "confidence_category"]
    result_df = predicted_df[[c for c in key_cols if c in predicted_df.columns]]

result_df

# %%
# Show summary statistics
print("\n=== Classification Summary ===")
print(predicted_df["confidence_category"].value_counts())

print("\n=== Comparison with True Labels ===")
if "is_cancer" in predicted_df.columns:
    comparison = pd.crosstab(
        predicted_df["is_cancer"],
        predicted_df["confidence_category"],
        margins=True
    )
    print(comparison)

#%%
# Sklearn metrics with confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np

if "is_cancer" in predicted_df.columns:
    # Map confidence categories to binary predictions
    # Adjust this mapping based on what you consider "cancer" vs "non-cancer"
    def map_to_binary(confidence_cat):
        cancer_categories = ["confident_cancer", "likely_cancer"]
        non_cancer_categories = ["likely_non_cancer"]
        
        if confidence_cat in cancer_categories:
            return 1  # Predict cancer
        elif confidence_cat in non_cancer_categories:
            return 0  # Predict non-cancer
        else:
            return None  # Uncertain - exclude from metrics
    
    # Create binary predictions
    predicted_df["predicted_cancer"] = predicted_df["confidence_category"].apply(map_to_binary)
    
    # Filter out uncertain predictions for cleaner metrics
    eval_df = predicted_df[predicted_df["predicted_cancer"].notna()].copy()
    
    if len(eval_df) > 0:
        y_true = eval_df["is_cancer"].astype(int)
        y_pred = eval_df["predicted_cancer"].astype(int)
        
        print("\n" + "="*60)
        print("SKLEARN CLASSIFICATION METRICS")
        print("="*60)
        
        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print("                 Predicted")
        print("                 Non-Cancer  Cancer")
        print(f"Actual Non-Cancer    {cm[0,0]:5d}     {cm[0,1]:5d}")
        print(f"Actual Cancer        {cm[1,0]:5d}     {cm[1,1]:5d}")
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        print("\n--- Breakdown ---")
        print(f"True Positives  (TP): {tp:5d}  (Correctly identified cancer)")
        print(f"True Negatives  (TN): {tn:5d}  (Correctly identified non-cancer)")
        print(f"False Positives (FP): {fp:5d}  (Incorrectly predicted cancer)")
        print(f"False Negatives (FN): {fn:5d}  (Missed cancer cases)")
        
        # Overall Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n--- Overall Accuracy ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification Report
        print("\n--- Detailed Classification Report ---")
        target_names = ["Non-Cancer (0)", "Cancer (1)"]
        print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
        
        # Additional metrics per class
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        print("\n--- Per-Class Metrics ---")
        print(f"\nNon-Cancer (Class 0):")
        print(f"  Precision: {precision[0]:.4f}  (When we predict non-cancer, we're right {precision[0]*100:.2f}% of the time)")
        print(f"  Recall:    {recall[0]:.4f}  (We catch {recall[0]*100:.2f}% of all non-cancer cases)")
        print(f"  F1-Score:  {f1[0]:.4f}")
        print(f"  Support:   {support[0]}")
        
        print(f"\nCancer (Class 1):")
        print(f"  Precision: {precision[1]:.4f}  (When we predict cancer, we're right {precision[1]*100:.2f}% of the time)")
        print(f"  Recall:    {recall[1]:.4f}  (We catch {recall[1]*100:.2f}% of all cancer cases)")
        print(f"  F1-Score:  {f1[1]:.4f}")
        print(f"  Support:   {support[1]}")
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f"\n--- Average Metrics ---")
        print(f"Macro Average    - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        print(f"Weighted Average - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}")
        
        # Show samples excluded due to uncertainty
        n_uncertain = len(predicted_df) - len(eval_df)
        print(f"\n--- Coverage ---")
        print(f"Samples evaluated: {len(eval_df)} / {len(predicted_df)}")
        print(f"Uncertain samples excluded: {n_uncertain}")
        
        if n_uncertain > 0:
            print("\nUncertain category breakdown:")
            uncertain_counts = predicted_df[predicted_df["predicted_cancer"].isna()]["confidence_category"].value_counts()
            print(uncertain_counts)
    
    else:
        print("\nNo samples with definitive predictions (all uncertain)")

#%%
# Optional: Visualize confusion matrix with seaborn/matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

if "is_cancer" in predicted_df.columns and len(eval_df) > 0:
    plt.figure(figsize=(8, 6))
    
    # Create confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Cancer', 'Cancer'],
                yticklabels=['Non-Cancer', 'Cancer'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix: Cancer Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Also show normalized version (percentages)
    plt.figure(figsize=(8, 6))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Non-Cancer', 'Cancer'],
                yticklabels=['Non-Cancer', 'Cancer'],
                cbar_kws={'label': 'Proportion'})
    
    plt.title('Normalized Confusion Matrix: Cancer Classification', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()