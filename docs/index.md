# Freya Cancer Classification Pipeline

Automated cancer/non-cancer classification of RNA-seq samples from public sequencing repositories (SRA, ENA, DDBJ) using metadata-based NLP. Part of the **Paipu framework** for pan-mammalian tumor data.

---

## What It Does

Classifies RNA-seq samples as **CANCER** or **NON_CANCER** by analyzing free-text metadata fields (`source_name`, `tissue`, `disease`, `title`, etc.) through a multi-stage hybrid pipeline:

1. **Regex** — Fast pattern matching against 300+ cancer terms
2. **MedSpaCy NLP** — Clinical NLP with negation detection (e.g., "no evidence of tumor" → non-cancer)
3. **Resolution** — Combines both stages, weighting sample-level signals over study-level
4. **Fallback** *(optional)* — Expanded search → NCBI API → LLM classification
5. **Enrichment** — Adds `is_cell_line` and `is_benign` metadata flags

## Current Accuracy

Validated against **410 manually-labeled non-mouse samples**:

| Metric | Value |
|---|---|
| Overall Accuracy | **90.7%** |
| Cancer Precision | **98.3%** |
| Cancer Recall | **89.8%** |

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Quickstart](quickstart.md)**

    Get up and running — install, classify, and explore every feature

-   :material-sitemap: **[Architecture](architecture.md)**

    Pipeline stages, data flow, module responsibilities, and MedSpaCy rules

-   :material-wrench: **[Developer Guide](developer_guide.md)**

    Extend the pipeline: add patterns, providers, columns, and flags

</div>

## Quick Example

```python
import polars as pl
from functions import classify_cancer_samples, get_nlp
from config import FINAL_LABEL_MAP

df = pl.read_csv("your_metadata.csv", infer_schema_length=0)
nlp = get_nlp()

results = classify_cancer_samples(df, nlp_pipeline=nlp)
results = results.with_columns(
    pl.col("confidence_category").replace(FINAL_LABEL_MAP).alias("final_label")
)
```
