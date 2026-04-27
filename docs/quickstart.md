# Quickstart Guide

A hands-on walkthrough of the Freya Cancer Classification Pipeline — from installation to running your first classification, exploring every feature, and contributing improvements.

> **Prerequisites**: Python ≥ 3.11.9, [uv](https://docs.astral.sh/uv/) (recommended) or pip.

---

## 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd Freya-Cancers-Clean

# Install dependencies (using uv — recommended)
uv sync

# Or, using pip:
pip install polars medspacy spacy biopython lxml negspacy pyarrow
```

### Verify the install

```bash
uv run python -c "from functions import classify_cancer_samples; print('✓ Ready')"
```

---

## 2. Your First Classification (Ground Truth Validation)

The fastest way to see the pipeline in action is to run it against the **410 manually-labeled samples**:

```bash
uv run python known_samples.py
```

**What happens:**
1. Loads `data/manual_label_not_mouse.xlsx` (ground truth labels) and joins against the full metadata in `data/combined_metadata_noncancer_removed.csv`.
2. Runs the full regex → MedSpaCy → resolution pipeline.
3. Prints a confusion matrix with precision, recall, and accuracy.
4. Saves detailed results to `outputs/`:

| Output file | Contents |
|---|---|
| `outputs/all_predictions.csv` | Every sample with predicted labels and explanations |
| `outputs/false_negatives.csv` | Cancer samples the pipeline missed |
| `outputs/false_positives.csv` | Non-cancer samples incorrectly flagged |

**Example output:**

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

---

## 3. Classify a Full Dataset

### All species

```bash
uv run python full_data.py
```

Processes `data/combined_metadata_noncancer_removed.csv` (~120k samples) through the complete pipeline including:
- Text preprocessing and normalization
- Auto-discovery of text columns
- Disease-specific MedSpaCy rule generation
- Post-classification metadata enrichment (`is_cell_line`, `is_benign` flags)

**Output:** `outputs/classified_samples.csv`

### Non-mouse species only

```bash
uv run python non_mouse_data.py
```

Same pipeline, but filters out *Mus musculus* samples first.

**Output:** `outputs/non_mouse_classified.csv`

---

## 4. Classify Your Own Data (Programmatic API)

```python
import polars as pl
from functions import classify_cancer_samples, get_nlp
from config import FINAL_LABEL_MAP

# Load your metadata CSV
df = pl.read_csv("my_metadata.csv", infer_schema_length=0)

# Initialize the MedSpaCy pipeline (singleton — created once, reused)
nlp = get_nlp()

# Classify
results = classify_cancer_samples(df, nlp_pipeline=nlp)

# Map confidence categories to binary CANCER / NON_CANCER
results = results.with_columns(
    pl.col("confidence_category").replace(FINAL_LABEL_MAP).alias("final_label")
)

# Inspect results
print(results.select(
    "run_accession",
    "final_label",
    "confidence_category",
    "resolved_by",
    "med_reason",
    "is_cell_line",
    "is_benign",
))

# Export
results.write_csv("my_results.csv")
```

### Required input columns

| Column | Importance | Description |
|---|---|---|
| `run_accession` | Required | Unique sample identifier |
| `title` | High | Study title |
| `source_name` | High | Sample source description |
| `tissue` | High | Tissue type |
| `disease` | High | Disease annotation |
| `cell_type` | Medium | Cell type annotation |

The pipeline auto-discovers additional text columns, so extra columns in your CSV are fine.

---

## 5. Interactive Testing (Streamlit App)

```bash
uv run streamlit run cancer_text_classifier_app.py
```

Opens a web UI where you can:
- Type any free text and see real-time MedSpaCy classification
- View detected entities (cancer terms, negated terms, non-cancer terms)
- Explore the rule set in the sidebar

**Example inputs to try:**
- `breast cancer tissue` → 🔴 CANCER
- `normal healthy sample` → 🟢 NOT_CANCER
- `no evidence of tumor` → 🟢 NOT_CANCER (negated)
- `metastatic carcinoma` → 🔴 CANCER
- `type 2 diabetes` → ⚪ NO_SIGNAL

---

## 6. Understanding the Pipeline Stages

Each sample passes through up to 5 stages. The key output columns tell you exactly what happened:

| Column | What it tells you |
|---|---|
| `regex_label` | What the regex stage decided |
| `regex_reason` | Which patterns matched (and in which columns) |
| `med_label` | What MedSpaCy NLP decided |
| `med_reason` | Entity counts, negation details |
| `med_source_columns` | Which metadata columns contained cancer entities |
| `confidence_category` | Final confidence label (see below) |
| `resolved_by` | Which stage(s) made the final call |
| `is_cell_line` | Whether the sample is from a cell line |
| `is_benign` | Whether the sample is flagged as benign |

### Confidence categories

| Category | Maps to | When |
|---|---|---|
| `confident_cancer` | CANCER | Regex + MedSpaCy both agree |
| `likely_cancer` | CANCER | Probable cancer, some ambiguity |
| `confirmed_by_medspacy` | CANCER | Regex uncertain, MedSpaCy found cancer |
| `confirmed_non_cancer` | NON_CANCER | No cancer signal or confirmed negative |
| `likely_non_cancer` | NON_CANCER | Negative context detected |

### Who resolved it?

| `resolved_by` value | Meaning |
|---|---|
| `regex` | Regex had a clear signal, MedSpaCy had none |
| `medspacy` | Regex was uncertain, MedSpaCy resolved it |
| `regex+medspacy` | Both stages contributed |
| `default` | No signal found — defaulted to non-cancer |
| `expanded_search` | Fallback: found terms in non-priority columns |
| `ncbi_api` | Fallback: NCBI BioProject metadata resolved it |
| `gemini_llm` / `ollama_llm` | Fallback: LLM classified the sample |

---

## 7. Enabling the Fallback Pipeline

For unresolved samples (no signal from regex or MedSpaCy), you can enable a progressive fallback chain:

```python
from functions import classify_cancer_samples, get_nlp
from fallback import ExpandedSearchProvider
from providers import NCBIProvider, GeminiProvider

nlp = get_nlp()

# Build the fallback provider chain (tried in order)
providers = [
    ExpandedSearchProvider(nlp_pipeline=nlp),      # Scan ALL text columns
    NCBIProvider(email="you@email.com"),            # NCBI BioProject API
    GeminiProvider(api_key="your-gemini-key"),      # Google Gemini LLM
]

results = classify_cancer_samples(
    df,
    nlp_pipeline=nlp,
    fallback_providers=providers,
    use_fallback=True,
)
```

Each provider is tried in order. The pipeline stops at the first provider that returns a confident result (confidence ≥ 0.5).

> **Environment variable**: Set `GEMINI_API_KEY` in your `.env` file to use the Gemini provider.

---

## 8. Verification & Quality Assurance

### Sampling for manual review

The `verification/` directory contains scripts to spot-check the metadata enrichment flags:

```bash
uv run python verification/sample_verification.py
```

This samples 50 random rows for each enrichment flag (`is_cell_line`, `is_benign`) and shows which metadata columns triggered the flag. Results are saved to:
- `verification/cell_line_verification_50.csv`
- `verification/benign_verification_50.csv`

### Diagnosing misclassifications

For each false negative or false positive in the validation output:
1. Check `regex_label` — did the regex stage find anything?
2. Check `med_label` / `med_reason` — did MedSpaCy detect entities?
3. Check `resolved_by` — which stage made the final decision?
4. Read the raw text columns — is there actually a cancer signal in the metadata?

---

## 9. Contributing: How to Extend the Pipeline

### Add a new cancer term

1. **Add to MedSpaCy rules** in `config.py`:

   ```python
   # In CANCER_RULE_DEFINITIONS:
   ("your_cancer_term", "CANCER", ""),                     # Literal match
   ("", "CANCER", r"\byour_regex_pattern\b"),              # Regex match
   ```

2. **(Optional) Add to regex fast-screening** in `config.py`:

   ```python
   class RegexPatterns:
       cancer_positive: str = (
           # ... existing patterns ...
           r"\byour_term\b|"
       )
   ```

3. **Validate** — run `uv run python known_samples.py` and check the confusion matrix. A good change increases recall without significantly hurting precision.

### Add a new metadata column to the pipeline

Add the column name to `priority_cols` in `ClassifierConfig` inside `config.py`. This automatically enables:
- Text normalization (`_norm` suffix column)
- Regex scanning
- MedSpaCy NLP processing
- Source column tracking in `med_source_columns`

### Add a new enrichment flag

1. Define the pattern and target columns in `config.py`:

   ```python
   MY_FLAG_PATTERN = r"\byour_pattern\b"
   MY_FLAG_SEARCH_COLS = ("disease", "source_name", "tissue")
   ```

2. Add the flag in `metadata_enrichment.py`:

   ```python
   df = _detect_flag(
       df,
       pattern=MY_FLAG_PATTERN,
       search_cols=MY_FLAG_SEARCH_COLS,
       flag_name="is_my_flag",
       use_normalized=use_normalized,
   )
   ```

### Add a new fallback provider

1. Create `providers/my_provider.py` implementing `FallbackProvider`:

   ```python
   from fallback import FallbackProvider, FallbackResult
   from config import ClassificationLabel as CL

   class MyProvider(FallbackProvider):
       @property
       def name(self) -> str:
           return "my_provider"

       def classify(self, sample: dict) -> FallbackResult:
           # sample is a dict of all column values for one row
           return FallbackResult(
               label=CL.LIKELY_CANCER.value,
               confidence=0.7,
               reason="your explanation",
               provider_name=self.name,
           )
   ```

2. Register in `providers/__init__.py`:

   ```python
   from providers.my_provider import MyProvider
   ```

3. Add to your provider chain when calling `classify_cancer_samples()`.

---

## 10. Project Structure at a Glance

```
├── config.py                  # Central config: regex patterns, MedSpaCy rules, enums
├── functions.py               # Orchestrator facade — main entry point
│
├── preprocessing.py           # Text cleaning & normalization
├── regex_classifier.py        # Regex-based pattern matching
├── nlp_classifier.py          # MedSpaCy NLP + resolution logic
├── pipeline.py                # MedSpaCy pipeline singleton
├── text_column_processing.py  # Column tiering (priority/secondary/discovered)
├── metadata_enrichment.py     # Post-classification flags (is_cell_line, is_benign)
│
├── fallback.py                # Fallback pipeline + ExpandedSearchProvider
├── providers/
│   ├── ncbi.py                # NCBI BioProject API provider
│   └── llm.py                 # Gemini & Ollama LLM providers
│
├── known_samples.py           # Validation against ground truth
├── full_data.py               # Full dataset classification
├── non_mouse_data.py          # Non-mouse classification
├── cancer_text_classifier_app.py  # Streamlit interactive app
│
├── data/                      # Input data (ground truth + full metadata)
├── outputs/                   # Classification results (CSVs)
├── verification/              # Enrichment flag spot-checking
├── figures/                   # Poster visualizations
├── docs/                      # Architecture & developer guides
└── pyproject.toml             # Dependencies
```

---

## Further Reading

- **[Architecture Guide](architecture.md)** — Deep-dive into pipeline stages, data flow, module responsibilities, and the MedSpaCy rules system
- **[Developer Guide](developer_guide.md)** — Detailed extension patterns, common pitfalls, and environment setup
