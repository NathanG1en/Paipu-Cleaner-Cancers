"""Quick inspection script for is_cell_line and is_benign enrichment flags."""

import polars as pl
from metadata_enrichment import enrich_metadata

if __name__ == "__main__":
    df = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        infer_schema_length=10000,
        ignore_errors=True,
    )
    df = enrich_metadata(df, use_normalized=True)

    display_cols = [
        "run_accession",
        "source_name",
        "cell_line",
        "cell_type",
        "disease",
        "diagnosis",
        "tissue",
        "title",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    # --- Cell-line flagged samples ---
    cell_lines = df.filter(pl.col("is_cell_line"))
    print(f"=== is_cell_line = True  ({cell_lines.height} rows) ===")
    print(cell_lines.select(display_cols).head(30))

    # --- Benign flagged samples ---
    benign = df.filter(pl.col("is_benign"))
    print(f"\n=== is_benign = True  ({benign.height} rows) ===")
    print(benign.select(display_cols))

    # --- Summary ---
    print(f"\n=== Summary ===")
    print(f"  Total samples:    {df.height}")
    print(f"  is_cell_line:     {cell_lines.height}")
    print(f"  is_benign:        {benign.height}")

    # Save to CSV
    import os
    os.makedirs("outputs", exist_ok=True)
    cell_lines.select(display_cols + ["is_cell_line"]).write_csv("outputs/cell_line_flagged.csv")
    benign.select(display_cols + ["is_benign"]).write_csv("outputs/benign_flagged.csv")
    print("\nSaved to outputs/cell_line_flagged.csv and outputs/benign_flagged.csv")
