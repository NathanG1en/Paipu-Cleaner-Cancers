# Freya Cancer Classification Pipeline

Automated cancer/non-cancer classification of RNA-seq samples from public sequencing repositories (SRA, ENA, DDBJ) using metadata-based NLP. Combines regex pattern matching with [MedSpaCy](https://github.com/medspacy/medspacy) clinical NLP to classify samples based on free-text metadata fields like `source_name`, `tissue`, `disease`, and `title`.

> **Part of the Paipu framework** — a pan-mammalian tumor data pipeline that standardizes and integrates large-scale RNA-seq data and associated SRA metadata into a unified dataset.

## How It Works

The pipeline uses a **multi-stage hybrid classification** approach:

1. **Regex Stage** — Fast pattern matching scans metadata columns for 300+ cancer-positive terms (e.g., `carcinoma`, `melanoma`, `DLBCL`) and negative-context terms (e.g., `normal`, `healthy`, `control`). Produces an initial confidence label.

2. **MedSpaCy NLP Stage** — Clinical NLP with negation detection. Understands linguistic context (e.g., "no evidence of tumor" → non-cancer, "non-cancerous" → non-cancer). Processes each column individually to track *where* cancer signals were found (sample-level vs. study-level).

3. **Resolution** — Combines both stages into a final classification. Sample-level signals (`source_name`, `tissue`) are weighted more heavily than study-level signals (`title`).

4. **Fallback Pipeline** *(optional)* — For unresolved samples, progressively escalates through: expanded column search → NCBI BioProject API enrichment → LLM classification (Gemini / Ollama).

5. **Metadata Enrichment** — Adds boolean flags (`is_cell_line`, `is_benign`) by scanning configurable column sets.

### Classification Labels

| Label | Final Mapping | Meaning |
|---|---|---|
| `confident_cancer` | CANCER | Strong regex signal + MedSpaCy agreement |
| `likely_cancer` | CANCER | Probable cancer, some ambiguity |
| `confirmed_by_medspacy` | CANCER | Weak/no regex signal but MedSpaCy found cancer entities |
| `confirmed_non_cancer` | NON_CANCER | No cancer signal or confirmed negative context |
| `likely_non_cancer` | NON_CANCER | Negative context detected (normal, control, healthy) |

## Project Structure

```
├── config.py                  # Central configuration: regex patterns, MedSpaCy rules, enums
├── functions.py               # Backward-compatible facade (re-exports from modules below)
│
├── preprocessing.py           # Text cleaning, normalization, column analysis
├── regex_classifier.py        # Regex-based pattern matching classification
├── nlp_classifier.py          # MedSpaCy NLP classification + resolution logic
├── pipeline.py                # MedSpaCy pipeline singleton management
├── text_column_processing.py  # Column tiering (priority/secondary/discovered)
├── metadata_enrichment.py     # Post-classification flags (is_cell_line, is_benign)
│
├── fallback.py                # Pluggable fallback pipeline + ExpandedSearchProvider
├── providers/
│   ├── __init__.py            # Re-exports all providers
│   ├── ncbi.py                # NCBI BioProject metadata enrichment provider
│   └── llm.py                 # LLM providers (GeminiProvider, OllamaProvider)
│
├── known_samples.py           # Validation against manually-labeled ground truth
├── full_data.py               # Full dataset classification (all species)
├── non_mouse_data.py          # Non-mouse-only classification
├── cancer_text_classifier_app.py  # Streamlit interactive app
│
├── data/
│   ├── manual_label_not_mouse.xlsx           # Ground truth (410 manually labeled samples)
│   └── combined_metadata_noncancer_removed.csv  # Full dataset (~120k samples)
│
├── outputs/                   # Classification results (CSVs)
│   ├── all_predictions.csv
│   ├── false_negatives.csv
│   ├── false_positives.csv
│   └── non_mouse_classified.csv
│
├── figures/                   # Poster visualizations (HTML + export scripts)
│
├── misc/                      # Legacy/exploratory scripts and notebooks
│
└── pyproject.toml             # Project dependencies (uv/pip)
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
# All species
python full_data.py

# Non-mouse only
python non_mouse_data.py
```

### Classify Programmatically

```python
import polars as pl
from functions import classify_cancer_samples, get_nlp

nlp = get_nlp()
df = pl.read_csv("your_metadata.csv")

results = classify_cancer_samples(df, nlp_pipeline=nlp)
print(results.select("run_accession", "confidence_category", "resolved_by", "med_reason"))
```

### Interactive Testing (Streamlit)

```bash
streamlit run cancer_text_classifier_app.py
```

## Current Accuracy

Validated against **410 manually-labeled non-mouse samples**:

| Metric | Value |
|---|---|
| Overall Accuracy | **90.7%** |
| Cancer Precision | **98.3%** |
| Cancer Recall | **89.8%** |
| N Samples | 410 |

## Dependencies

- Python ≥ 3.11.9
- [Polars](https://pola.rs/) — DataFrame processing
- [MedSpaCy](https://github.com/medspacy/medspacy) — Clinical NLP with negation detection
- [spaCy](https://spacy.io/) — NLP framework
- [BioPython](https://biopython.org/) — Bioinformatics utilities (NCBI provider)

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

## Further Documentation

- **[Quickstart Guide](docs/quickstart.md)** — Hands-on walkthrough of all features, from first run to contributing
- **[Architecture Guide](docs/architecture.md)** — Deep-dive into pipeline stages, data flow, and module responsibilities
- **[Developer Guide](docs/developer_guide.md)** — How to extend the pipeline: adding patterns, providers, columns, and flags
