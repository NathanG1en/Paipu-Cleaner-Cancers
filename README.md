# Freya Cancer Classification Pipeline

Automated cancer/non-cancer classification of RNA-seq samples from public sequencing repositories (SRA, ENA, DDBJ) using metadata-based NLP. Combines regex pattern matching with [MedSpaCy](https://github.com/medspacy/medspacy) clinical NLP to classify samples based on text fields like `source_name`, `tissue`, `disease`, and `title`.

## How It Works

The pipeline uses a **two-stage classification** approach:

1. **Regex stage** — Fast pattern matching scans metadata columns for cancer-positive terms (e.g., `carcinoma`, `melanoma`, `DLBCL`) and negative-context terms (e.g., `normal`, `healthy`, `control`). Produces an initial confidence label.

2. **MedSpaCy NLP stage** — Clinical NLP with negation detection (e.g., "non-cancerous" → non-cancer, "no tumor" → non-cancer). Processes each column individually to track *where* cancer terms were found (sample-level vs. study-level).

3. **Resolution** — Combines both stages into a final binary classification. Sample-level signals (source_name, tissue) are weighted more heavily than study-level signals (title).

### Classification Labels

| Label | Meaning |
|---|---|
| `confident_cancer` | Strong regex signal + MedSpaCy agreement |
| `confirmed_by_medspacy` | Weak/no regex signal but MedSpaCy found cancer entities |
| `likely_cancer` | Probable cancer, some ambiguity |
| `confirmed_non_cancer` | No cancer signal or confirmed negative context |
| `likely_non_cancer` | Negative context detected (normal, control, healthy) |

## Project Structure

```
├── config.py                 # Central configuration: regex patterns, MedSpaCy rules, Enums
├── functions.py              # Backward-compatible facade (re-exports from modules below)
│
├── preprocessing.py          # Text cleaning, normalization, column analysis
├── regex_classifier.py       # Regex-based pattern matching classification
├── nlp_classifier.py         # MedSpaCy NLP classification + resolution logic
├── pipeline.py               # MedSpaCy pipeline singleton management
│
├── known_samples.py          # Validation against manually-labeled ground truth
├── full_data.py              # Full dataset classification pipeline
├── cancer_text_classifier_app.py  # Streamlit app interface
├── debug_classification.py   # Diagnostic tools for investigating misclassifications
│
├── data/
│   ├── manual_label_not_mouse.xlsx    # Ground truth (manually labeled, non-mouse)
│   └── combined_metadata_noncancer_removed.csv  # Full dataset
│
├── outputs/                  # Classification results (CSVs)
│   ├── false_negatives.csv
│   ├── false_positives.csv
│   └── all_predictions.csv
│
└── pyproject.toml            # Project dependencies (uv/pip)
```

## Quick Start

### Setup

```bash
# Clone and install dependencies (using uv)
git clone <repo-url>
cd Freya-Cancers-Clean
uv sync
```

### Validate Against Ground Truth

```bash
python known_samples.py
```

This runs classification on the manually-labeled dataset and outputs:
- Confusion matrix (TP, FP, TN, FN)
- Precision, recall, and accuracy
- False negatives and false positives with explanations
- CSVs in `outputs/`

### Classify a Full Dataset

```bash
python full_data.py
```

### Classify Programmatically

```python
import polars as pl
from functions import classify_cancer_samples, get_nlp

nlp = get_nlp()
df = pl.read_csv("your_metadata.csv")

results = classify_cancer_samples(df, nlp_pipeline=nlp)
print(results.select("run_accession", "confidence_category", "med_reason"))
```

## Current Accuracy

Validated against 410 manually-labeled non-mouse samples:

| Metric | Value |
|---|---|
| Overall Accuracy | 85.1% |
| Cancer Precision | 96.4% |
| Cancer Recall | 84.1% |
| Cancer Prediction Accuracy | 96.4% (270/280) |
| Non-Cancer Prediction Accuracy | 60.8% (79/130) |

> **Note:** Non-cancer accuracy is limited by ~25 samples with zero cancer-relevant metadata (titles like "Illumina HiSeq 4000 sequencing") and ~7 likely ground-truth labeling errors (matched-normal samples labeled as cancer).

## Dependencies

- Python ≥ 3.11.9
- [Polars](https://pola.rs/) — DataFrame processing
- [MedSpaCy](https://github.com/medspacy/medspacy) — Clinical NLP with negation detection
- [spaCy](https://spacy.io/) — NLP framework
- [BioPython](https://biopython.org/) — Bioinformatics utilities

## Adding New Cancer Patterns

Edit `config.py`:

```python
# Add to CANCER_RULE_DEFINITIONS list:
("your_cancer_term", "CANCER", ""),                    # Literal match
("", "CANCER", r"\byour_regex_pattern\b"),             # Regex match

# Add to RegexPatterns.cancer_positive if needed:
r"\byour_term\b|"
```

Run `python known_samples.py` to verify the impact on accuracy.
