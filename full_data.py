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
    get_nlp
)


if __name__ == "__main__":
    nlp = get_nlp()

    all_samples = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        schema_overrides={"group": pl.Utf8},  # <- dict of {col_name: dtype}
        infer_schema_length=0,  # scan entire file for other cols
    )

    unique_diseases = all_samples.select("disease").unique().to_series().to_list()

    generate_disease_rules(unique_diseases, nlp, existing_rules)

