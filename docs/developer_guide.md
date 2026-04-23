# Developer Guide

Practical how-to guide for extending, maintaining, and running the cancer classification pipeline.

## Table of Contents

- [Adding New Cancer Patterns](#adding-new-cancer-patterns)
- [Adding a Fallback Provider](#adding-a-fallback-provider)
- [Adding New Metadata Columns](#adding-new-metadata-columns)
- [Adding New Enrichment Flags](#adding-new-enrichment-flags)
- [Running Validation](#running-validation)
- [Running on New Datasets](#running-on-new-datasets)
- [Streamlit App](#streamlit-app)
- [Common Pitfalls](#common-pitfalls)
- [Environment Setup](#environment-setup)

---

## Adding New Cancer Patterns

### Step 1: Add a TargetRule

In `config.py`, find the `CANCER_RULE_DEFINITIONS` list and add your term:

```python
# Literal match (most common)
("your_cancer_term", "CANCER", ""),

# Regex match (for variant spellings)
("", "CANCER", r"\bchondrosarcoma[s]?\b"),
```

Each tuple is `(literal, category, regex_pattern)`. Use one or the other—if `literal` is non-empty, it's used for matching; if empty, `regex_pattern` is used instead.

### Step 2: (Optional) Add a Regex Pattern

If the term is common enough to warrant fast regex pre-screening, add it to `RegexPatterns.cancer_positive`:

```python
class RegexPatterns:
    cancer_positive: str = (
        # ... existing patterns ...
        r"\byour_term\b|"
    )
```

### Step 3: Validate

```bash
python known_samples.py
```

Check the confusion matrix. Adding a term should increase recall (fewer false negatives) without significantly hurting precision (more false positives).

### Adding Non-Cancer Patterns

Same process, but use `NON_CANCER_RULE_DEFINITIONS` and `RegexPatterns.cancer_negative`.

---

## Adding a Fallback Provider

### Step 1: Implement the Interface

Create a new file in `providers/` that extends `FallbackProvider`:

```python
# providers/my_provider.py
from fallback import FallbackProvider, FallbackResult
from config import ClassificationLabel as CL

class MyProvider(FallbackProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    def classify(self, sample: dict) -> FallbackResult:
        # sample is a dict of all column values for one row
        # Your classification logic here...

        return FallbackResult(
            label=CL.LIKELY_CANCER.value,  # or CL.CONFIRMED_NON_CANCER.value
            confidence=0.7,                 # 0.0 - 1.0; >= 0.5 = "resolved"
            reason="explanation for human review",
            provider_name=self.name,
        )

    # Optional: override for batched API calls
    def classify_batch(self, samples: list) -> list:
        return [self.classify(s) for s in samples]
```

### Step 2: Register in `providers/__init__.py`

```python
from providers.my_provider import MyProvider
__all__ = [..., "MyProvider"]
```

### Step 3: Use It

```python
from providers import MyProvider

results = classify_cancer_samples(
    df,
    fallback_providers=[ExpandedSearchProvider(), MyProvider()],
    use_fallback=True,
)
```

Providers are tried **in order**. The pipeline stops at the first provider that returns `confidence >= 0.5`.

---

## Adding New Metadata Columns

### Priority vs. Secondary Columns

In `config.py`, the `ClassifierConfig` (or `TextColumnConfig`) defines:

```python
priority_cols = ("title", "source_name", "tissue", "disease", "cell_type", ...)
secondary_cols = ("sample_name", "condition", "tumor", ...)
```

- **Priority columns** are always searched by both regex and MedSpaCy.
- **Secondary columns** are normalized but only used in expanded search or when explicitly requested.

To add a new column to the classification pipeline, add it to `priority_cols` in `config.py`.

### Impact

Adding a column means:
1. It gets a `_norm` normalized version during preprocessing.
2. `regex_classifier.py` will scan it for cancer/non-cancer patterns.
3. `nlp_classifier.py` will run MedSpaCy on its text.
4. `med_source_columns` will track if cancer was found there.

---

## Adding New Enrichment Flags

To add a new boolean flag (like `is_cell_line` or `is_benign`):

### Step 1: Define Pattern and Columns in `config.py`

```python
MY_FLAG_PATTERN = r"\byour_pattern\b"
MY_FLAG_SEARCH_COLS = ("disease", "source_name", "tissue")
```

### Step 2: Add to `enrich_metadata()` in `metadata_enrichment.py`

```python
def enrich_metadata(df, use_normalized=True):
    # ... existing flags ...

    df = _detect_flag(
        df,
        pattern=MY_FLAG_PATTERN,
        search_cols=MY_FLAG_SEARCH_COLS,
        flag_name="is_my_flag",
        use_normalized=use_normalized,
    )
    return df
```

### Step 3: Import in `config.py`

Add `MY_FLAG_PATTERN` and `MY_FLAG_SEARCH_COLS` exports so they're accessible from the central config.

---

## Running Validation

### Ground Truth Dataset

The file `data/manual_label_not_mouse.xlsx` contains 410 manually labeled non-mouse samples with an `is_cancer` column (1 = cancer, 0 = non-cancer).

### Running

```bash
python known_samples.py
```

### Understanding Output

```
=== Confusion Matrix ===
  TP (cancer → cancer):         264
  FP (non-cancer → cancer):       5
  TN (non-cancer → non-cancer):  81
  FN (cancer → non-cancer):      30

  Overall Accuracy: 90.7%
  Precision (cancer): 98.3%
  Recall (cancer):    89.8%
```

- **False negatives** (`outputs/false_negatives.csv`): Cancer samples the pipeline missed. Check `med_reason` and `regex_label` to understand why.
- **False positives** (`outputs/false_positives.csv`): Non-cancer samples incorrectly labeled as cancer.

### Diagnosing Failures

For each false negative/positive, look at:
1. `regex_label` — Did regex find anything?
2. `med_label` / `med_reason` — Did MedSpaCy find entities?
3. `resolved_by` — Which stage made the final call?
4. The raw text columns — Is there *any* cancer signal in the metadata?

---

## Running on New Datasets

### Input Format

Your CSV needs at minimum some of these columns (the more the better):

| Column | Importance | Description |
|---|---|---|
| `run_accession` | Required | Unique sample identifier |
| `title` | High | Study title (study-level signal) |
| `source_name` | High | Sample source description |
| `tissue` | High | Tissue type |
| `disease` | High | Disease annotation |
| `cell_type` | Medium | Cell type annotation |
| `phenotype` | Medium | Phenotype description |

The pipeline auto-discovers additional viable text columns, so extra columns are fine.

### Example Script

```python
import polars as pl
from functions import classify_cancer_samples, get_nlp
from text_column_processing import preprocess_dataframe

# Load
df = pl.read_csv("my_data.csv", infer_schema_length=0)

# Preprocess (creates _norm columns)
df, col_tiers = preprocess_dataframe(df)

# Classify
nlp = get_nlp()
results = classify_cancer_samples(df, nlp_pipeline=nlp, use_normalized=True)

# Map to binary
LABEL_MAP = {
    "confident_cancer": "CANCER",
    "likely_cancer": "CANCER",
    "confirmed_by_medspacy": "CANCER",
    "confirmed_non_cancer": "NON_CANCER",
    "likely_non_cancer": "NON_CANCER",
}
results = results.with_columns(
    pl.col("confidence_category").replace(LABEL_MAP).alias("final_label")
)

results.write_csv("classified_output.csv")
```

---

## Streamlit App

An interactive web UI for testing individual text inputs against the MedSpaCy pipeline.

### Running

```bash
streamlit run cancer_text_classifier_app.py
```

### Features

- Type any text and see real-time classification.
- View detected entities: cancer terms, negated terms, non-cancer terms.
- Sidebar shows pipeline stats (total rules, disease rules added).
- Example texts provided for quick testing.

---

## Common Pitfalls

### 1. Onco-Traps (False Positives from Species Names)

Some species contain cancer-related substrings:
- "Onchorhynchus" (salmon) contains "onc"
- "Capricornis" (goat-antelope) contains "cancer"

The regex stage has an `ONCO_TRAPS` pattern to catch these. If you see false positives from species names, add the term to the trap list in `config.py`.

### 2. Study-Level vs. Sample-Level Signals

A study titled "Breast Cancer RNA-seq" may contain both tumor and normal control samples. The pipeline handles this by:
- Giving higher weight to sample-level columns (`source_name`, `tissue`).
- In the resolution logic, cancer found *only* in study-level columns (`title`) does not override sample-level non-cancer evidence.

### 3. Negation Edge Cases

MedSpaCy handles most negation patterns, but some edge cases to watch for:
- **Double negation**: "not without cancer" — treated as affirmed.
- **Distant negation**: "no ... [many words] ... cancer" — the context window may miss this.
- **Implicit negation**: "cancer-free" — handled by regex negative patterns, not MedSpaCy.

### 4. Cell Lines vs. Patient Tissue

Cell lines derived from tumors (e.g., HeLa, MCF-7) ARE cancer samples. The pipeline correctly classifies these as cancer. The `is_cell_line` flag is added *after* classification to provide additional metadata, not to change the label.

### 5. Missing Metadata

~5-10% of samples have minimal metadata (e.g., title = "Illumina HiSeq 4000 sequencing"). These default to `confirmed_non_cancer` with `resolved_by = "default"`. The fallback pipeline (NCBI API, LLM) can help resolve these if enabled.

---

## Environment Setup

### Using uv (recommended)

```bash
uv sync
```

### Using pip

```bash
pip install polars medspacy spacy biopython lxml negspacy pyarrow
```

### Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Only for LLM fallback | Google Gemini API key |

### .env File

The project includes a `.env` file for sensitive configuration. Currently only used for the Gemini API key if the LLM fallback is enabled.

```bash
# .env
GEMINI_API_KEY=your-key-here
```
