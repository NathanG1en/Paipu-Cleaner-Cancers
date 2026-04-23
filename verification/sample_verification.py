"""
Verification script: sample 50 random rows flagged as is_cell_line=True
and 50 random rows flagged as is_benign=True, then display a verification
table with columns:
    run accession | classification | reason(s)

The "reason(s)" column shows which metadata columns triggered the flag
and the matching text found in those columns.

Outputs are saved as CSVs and also printed as formatted tables.
"""

import re, sys, pathlib

import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VERIFICATION_DIR = pathlib.Path(__file__).resolve().parent

CELL_LINE_FLAGGED = OUTPUTS_DIR / "cell_line_flagged.csv"
BENIGN_FLAGGED = OUTPUTS_DIR / "benign_flagged.csv"

# ---------------------------------------------------------------------------
# Detection patterns (imported from config for consistency)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(PROJECT_ROOT))
from config import CELL_LINE_PATTERN, CELL_LINE_SEARCH_COLS, BENIGN_PATTERN, BENIGN_SEARCH_COLS, BENIGN_KEYWORDS

# Compile the patterns for Python re use
CELL_LINE_RE = re.compile(CELL_LINE_PATTERN, re.IGNORECASE)
BENIGN_RE = re.compile(BENIGN_PATTERN, re.IGNORECASE)


def _build_reason(row: dict, search_cols: tuple, pattern: re.Pattern, flag_type: str) -> str:
    """
    For a single row, check each search column for the regex pattern.
    Return a semicolon-separated string of  "column_name    matched_text"
    for every column that matched.
    """
    reasons = []
    for col in search_cols:
        val = row.get(col)
        if val is None or str(val).strip() in ("", "nan", "None"):
            continue
        text = str(val).strip()
        match = pattern.search(text)
        if match:
            reasons.append(f"{col}\t{text}")

    # For cell line: also check if the dedicated cell_line column has a real value
    # (the short-circuit logic in metadata_enrichment.py)
    if flag_type == "cell_line":
        cl_val = row.get("cell_line", "")
        if cl_val and str(cl_val).strip().lower() not in ("", "nan", "none"):
            cleaned = re.sub(r"[_/|\\;,]", " ", str(cl_val))
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            tokens = [t for t in cleaned.split() if t.lower() != "nan"]
            if tokens:
                # Avoid duplicate if regex already matched this column
                already = any(r.startswith("cell_line\t") for r in reasons)
                if not already:
                    reasons.append(f"cell_line\t{str(cl_val).strip()}")

    # For benign: also check BENIGN_KEYWORDS (specific benign tumor names)
    if flag_type == "benign":
        for col in search_cols:
            val = row.get(col)
            if val is None or str(val).strip() in ("", "nan", "None"):
                continue
            text_lower = str(val).strip().lower()
            for kw in BENIGN_KEYWORDS:
                if re.search(r"\b" + kw + r"\b", text_lower):
                    already = any(r.startswith(f"{col}\t") for r in reasons)
                    if not already:
                        reasons.append(f"{col}\t{str(val).strip()} [kw:{kw}]")
                    break

    if not reasons:
        # Fallback: show the title as context
        title = row.get("title", "")
        if title and str(title).strip() not in ("", "nan", "None"):
            reasons.append(f"title\t{str(title).strip()}")

    return "; ".join(reasons) if reasons else "(no column match found)"


def _make_verification_table(
    df: pl.DataFrame,
    search_cols: tuple,
    pattern: re.Pattern,
    flag_type: str,
    n_samples: int = 50,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Take n random samples from a pre-filtered (flagged=True) DataFrame
    and build the verification table.
    """
    print(f"  Total flagged rows: {df.height}")

    if df.height == 0:
        print("  WARNING: No rows found!")
        return pl.DataFrame({"run accession": [], "classification": [], "reason(s)": []})

    actual_n = min(n_samples, df.height)
    sample = df.sample(n=actual_n, seed=seed)

    # Build the reason column row by row
    rows_dicts = sample.to_dicts()
    results = []
    for row in rows_dicts:
        reason = _build_reason(row, search_cols, pattern, flag_type)
        results.append(
            {
                "run accession": row.get("run_accession", ""),
                "classification": 1,
                "reason(s)": reason,
            }
        )

    return pl.DataFrame(results)


def _print_table(table: pl.DataFrame, max_reason_width: int = 90):
    """Pretty-print the verification table."""
    if table.height == 0:
        print("  (empty table)")
        return

    print(f"  {'run accession':<18} {'classification':<16} reason(s)")
    print(f"  {'-'*18} {'-'*16} {'-'*max_reason_width}")
    for row in table.to_dicts():
        reason = row["reason(s)"]
        if len(reason) > max_reason_width:
            reason = reason[: max_reason_width - 3] + "..."
        print(f"  {row['run accession']:<18} {row['classification']:<16} {reason}")


def main():
    # ====================================================================
    # Cell Line verification
    # ====================================================================
    print("=" * 70)
    print("CELL LINE VERIFICATION (is_cell_line = True)")
    print("=" * 70)

    if CELL_LINE_FLAGGED.exists():
        df_cl = pl.read_csv(CELL_LINE_FLAGGED, infer_schema_length=0)
        print(f"  Loaded {df_cl.height} flagged cell-line rows from {CELL_LINE_FLAGGED.name}")
        table_cl = _make_verification_table(
            df_cl, CELL_LINE_SEARCH_COLS, CELL_LINE_RE, flag_type="cell_line"
        )
        out_path = VERIFICATION_DIR / "cell_line_verification_50.csv"
        table_cl.write_csv(out_path)
        print(f"  Saved to {out_path}\n")
        _print_table(table_cl)
    else:
        print(f"  File not found: {CELL_LINE_FLAGGED}")
    print()

    # ====================================================================
    # Benign verification
    # ====================================================================
    print("=" * 70)
    print("BENIGN VERIFICATION (is_benign = True)")
    print("=" * 70)

    if BENIGN_FLAGGED.exists():
        df_bn = pl.read_csv(BENIGN_FLAGGED, infer_schema_length=0)
        print(f"  Loaded {df_bn.height} flagged benign rows from {BENIGN_FLAGGED.name}")
        table_bn = _make_verification_table(
            df_bn, BENIGN_SEARCH_COLS, BENIGN_RE, flag_type="benign"
        )
        out_path = VERIFICATION_DIR / "benign_verification_50.csv"
        table_bn.write_csv(out_path)
        print(f"  Saved to {out_path}\n")
        _print_table(table_bn)
    else:
        print(f"  File not found: {BENIGN_FLAGGED}")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
