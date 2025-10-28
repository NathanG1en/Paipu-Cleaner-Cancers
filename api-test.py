# pip install polars biopython requests
import io, time, csv, re, requests, polars as pl
import xml.etree.ElementTree as ET
from functools import lru_cache
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from Bio import Entrez
import csv, io
import time

# -------------------------------
# Config
# -------------------------------

Entrez.email = "nathanglen@ufl.edu"   # required
# Entrez.api_key = "YOUR_KEY"          # optional
THROTTLE_S = 0.34                      # polite to NCBI

def sra_term_to_biosample_ids(term: str) -> list[str]:
    with Entrez.esearch(db="sra", term=term, retmax=10000) as h:
        rec = Entrez.read(h)
    sra_uids = rec.get("IdList", [])
    if not sra_uids:
        return []
    time.sleep(THROTTLE_S)

    # IMPORTANT
    with Entrez.elink(dbfrom="sra", db="biosample",
                      linkname="sra_biosample",
                      id=",".join(sra_uids)) as h:
        links = Entrez.read(h)

    biosample_uids = []
    for ls in links:
        for ldb in ls.get("LinkSetDb", []):
            if ldb.get("DbTo") == "biosample":
                biosample_uids.extend([lnk["Id"] for lnk in ldb.get("Link", [])])
    return sorted(set(biosample_uids))


def efetch_biosample_xml_batch(uids: list[str]) -> list[str]:
    """
    Fetch BioSample XML in batches and yield per-record XML strings.
    You can also keep your per-SAMN fetch if you prefer caching.
    """
    if not uids:
        return []
    xml_texts = []
    BATCH = 200
    for i in range(0, len(uids), BATCH):
        chunk = uids[i:i+BATCH]
        h = Entrez.efetch(db="biosample", id=",".join(chunk), rettype="xml", retmode="xml")
        xml = h.read()
        xml_texts.append(xml)
        time.sleep(THROTTLE_S)
    return xml_texts

def extract_biosample_accessions(xml_txt: str) -> list[str]:
    """
    Parse an efetch(biosample) XML blob and return SAMN accessions.
    """
    out = []
    root = ET.fromstring(xml_txt)
    for bs in root.findall(".//BioSample"):
        acc = bs.attrib.get("accession")
        if acc:
            out.append(acc)
    return out


# Shared session with retries/backoff
SESSION = requests.Session()
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5, backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"])
        )
    )
)

# -------------------------------
# Keywords (tune to your use case)
# -------------------------------



CANCER_RX = re.compile(
    r"\b(cancer|tumou?r|carcinoma|sarcoma|melanoma|glioma|glioblastoma|blastoma|"
    r"leuk(a)?emia|lymphoma|myeloma|myeloid|lymphoid|adenocarcinoma|neoplasm|"
    r"metastatic|metastasis|primary\s+tumou?r|carcinoid)\b", re.I)

# NOTE: if the goal is “sample is not tumor,” consider pruning infectious terms below.
NONCANCER_RX = re.compile(
    r"\b(normal|healthy|control|ctrl|adjacent[-\s]?normal|benign|"
    r"unaffected|non[-\s]?tumou?r|non[-\s]?cancer|naive\s+pbmc|"
    r"vaccine|vaccinated|infection|infected|virus|viral|bacteria|bacterial|"
    r"parasite|parasitic|mock[-\s]?treated|sham|wild[-\s]?type|wt|"
    r"placebo|vehicle|untreated)\b", re.I)

# Which BioSample attributes to harvest (names vary wildly, so cast a wide net)
ATTR_KEYS = {
    "disease","host_disease","diagnosis","tumor_type","disease_state","phenotype",
    "tissue","tissue_type","organ","organism","cell_type","cell_line","is_tumor",
    "biomaterial_type","source_name","sample_name","description","title"
}

# -------------------------------
# Helpers: BioSample fetch + parse
# -------------------------------

def fetch_runinfo_rows(term: str) -> list[dict]:
    """Get SRA RunInfo rows for an accession/project via Entrez."""
    handle = Entrez.esearch(db="sra", term=term)
    record = Entrez.read(handle)
    if not record["IdList"]:
        return []
    # fetch RunInfo CSV
    handle = Entrez.efetch(db="sra", id=",".join(record["IdList"]), rettype="runinfo", retmode="text")
    text = handle.read()
    if text.strip().lower().startswith("no items found"):
        return []
    return list(csv.DictReader(io.StringIO(text)))

@lru_cache(maxsize=20000)
def efetch_biosample_xml(samn: str) -> str:
    with Entrez.efetch(db="biosample", id=samn, rettype="xml", retmode="xml") as h:
        txt = h.read()
    time.sleep(THROTTLE_S)
    return txt


# def efetch_biosample_xml(samn: str) -> str:
#     """Cache BioSample XML by SAMN accession to avoid refetching."""
#     params = {"db": "biosample", "id": samn, "rettype": "xml", **NCBI_PARAMS}
#     r = SESSION.get(f"{EU_BASE}/efetch.fcgi", params=params, timeout=60)
#     r.raise_for_status()
#     time.sleep(THROTTLE_S)
#     return r.text

_key_norm = lambda s: re.sub(r'[^a-z0-9]+', '_', (s or '').lower()).strip('_')

ATTR_KEYS_NORM = {
    "disease","host_disease","diagnosis","tumor_type","disease_state","phenotype",
    "tissue","tissue_type","organ","organism","cell_type","cell_line","is_tumor",
    "biomaterial_type","source_name","sample_name","description","title"
}

def parse_biosample_attrs(xml_txt: str) -> dict:
    out = {}
    try:
        root = ET.fromstring(xml_txt)
    except ET.ParseError:
        return out

    bs = root.find(".//BioSample")
    if bs is None:
        return out

    out["biosample_accession"] = bs.attrib.get("accession")
    org = bs.find(".//Organism")
    if org is not None:
        out["organism"] = org.attrib.get("taxonomy_name")

    for a in bs.findall(".//Attributes/Attribute"):
        raw_k = (a.attrib.get("attribute_name") or "attribute").strip()
        k = _key_norm(raw_k)
        v = (a.text or "").strip()
        if not v:
            continue

        # keep curated superset; DO keep description/title-like
        if k in ATTR_KEYS_NORM or k in ("description", "title"):
            key = k
            i = 2
            while key in out and out[key] != v:
                key = f"{k}_{i}"; i += 1
            out[key] = v
            # optional: keep original name for debugging
            out.setdefault(f"{k}__orig", raw_k)

    if "description" not in out:
        desc = bs.findtext(".//Description")
        if desc:
            out["description"] = desc.strip()
    return out

def peek_attrs(acc: str, max_chars=120):
    uids = sra_term_to_biosample_ids(acc)
    blobs = efetch_biosample_xml_batch(uids)
    for blob in blobs:
        root = ET.fromstring(blob)
        for bs in root.findall(".//BioSample"):
            print("BioSample:", bs.attrib.get("accession"))
            for a in bs.findall(".//Attributes/Attribute"):
                rk = (a.attrib.get("attribute_name") or "").strip()
                print("  -", rk, "→", (a.text or "").strip()[:max_chars])

# Utility: fetch first value among base, base_2, base_3, ...
def first_with_prefix(d: dict, base: str):
    if base in d and d[base]:
        return d[base]
    i = 2
    while True:
        key = f"{base}_{i}"
        if key in d and d[key]:
            return d[key]
        if key not in d:
            break
        i += 1
    return None


# -------------------------------
# Classifier (rule-based)
# -------------------------------
def classify(attrs: dict) -> tuple[str, str]:
    """
    Return (label, reason) where label ∈ {'cancer','noncancerous','uncertain'}.
    Reason includes the *field* that matched and the keyword, e.g. 'diagnosis:cancer'.
    """
    # Iterate field-by-field so we can report the source key
    for k, v in attrs.items():
        if not isinstance(v, str):
            continue
        lk = k.lower()
        if lk not in ATTR_KEYS and lk not in ("description", "title"):
            continue
        txt = v.lower()

        m = CANCER_RX.search(txt)
        if m:
            return "cancer", f"{k}:{m.group(0)}"

        n = NONCANCER_RX.search(txt)
        if n:
            return "noncancerous", f"{k}:{n.group(0)}"

    return "uncertain", "no_keyword_hit"

# -------------------------------
# Main: process a Polars DataFrame
# -------------------------------
def enrich_with_biosample(df: pl.DataFrame, acc_col: str = "accession") -> pl.DataFrame:
    """
    Takes a Polars DF with a column of SRA accessions (SRX/ERX/SRR/etc).
    Returns a new DF with columns:
      accession | biosample | label | reason | organism | tissue | disease | cell_line | diagnosis | ...
    """
    # 1) unique accessions
    accessions = df.get_column(acc_col).unique().to_list()

    records = []
    for acc in accessions:
        # Resolve accession -> BioSample accessions (SAMN...) using Entrez
        try:
            biosample_uids = sra_term_to_biosample_ids(acc)  # list of numeric UIDs (may be empty)
            if not biosample_uids:
                records.append({
                    acc_col: acc, "biosample": None, "label": "uncertain",
                    "reason": "no_biosample_via_elink"
                })
                continue

            xml_blobs  = efetch_biosample_xml_batch(biosample_uids)  # list of XML strings
            biosamples = []
            for blob in xml_blobs:
                biosamples.extend(extract_biosample_accessions(blob))  # SAMN accessions
            biosamples = sorted(set(biosamples))
            if not biosamples:
                records.append({
                    acc_col: acc, "biosample": None, "label": "uncertain",
                    "reason": "no_biosample_accessions_in_efetch"
                })
                continue

        except Exception as e:
            records.append({
                acc_col: acc, "biosample": None, "label": "uncertain",
                "reason": f"elink_err:{type(e).__name__}"
            })
            continue

        # For each BioSample accession: fetch, parse, classify
        for samn in biosamples:
            try:
                xml = efetch_biosample_xml(samn)  # your cached per-SAMN fetcher
                attrs = parse_biosample_attrs(xml)
                label, reason = classify(attrs)
            except Exception as e:
                attrs = {}
                label, reason = "uncertain", f"biosample_err:{type(e).__name__}"

            records.append({
                acc_col: acc,
                "biosample": samn,
                "label": label,
                "reason": reason,
                "organism":   first_with_prefix(attrs, "organism"),
                "tissue":     first_with_prefix(attrs, "tissue")
                               or first_with_prefix(attrs, "tissue_type")
                               or first_with_prefix(attrs, "organ"),
                "disease":    first_with_prefix(attrs, "disease")
                               or first_with_prefix(attrs, "host_disease")
                               or first_with_prefix(attrs, "diagnosis"),
                "cell_line":  first_with_prefix(attrs, "cell_line"),
                "diagnosis":  first_with_prefix(attrs, "diagnosis"),
                "source_name":first_with_prefix(attrs, "source_name"),
                "description":first_with_prefix(attrs, "description"),
            })

    # Nothing resolved to any BioSample → return df with null meta columns
    if not records:
        return df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("biosample"),
            pl.lit(None, dtype=pl.Utf8).alias("label"),
            pl.lit(None, dtype=pl.Utf8).alias("reason"),
            pl.lit(None, dtype=pl.Utf8).alias("organism"),
            pl.lit(None, dtype=pl.Utf8).alias("tissue"),
            # pl.lit(None, dtype=pl.Utf8).alias("disease"),
            # pl.lit(None, dtype=pl.Utf8).alias("cell_line"),
            pl.lit(None, dtype=pl.Utf8).alias("diagnosis"),
            pl.lit(None, dtype=pl.Utf8).alias("source_name"),
            # pl.lit(None, dtype=pl.Utf8).alias("description"),
        )

    meta_df = pl.DataFrame(records)

    # 3) Prefer 'cancer' > 'noncancerous' > 'uncertain'
    pref = (
        pl.when(pl.col("label") == "cancer").then(2)
          .when(pl.col("label") == "noncancerous").then(1)
          .otherwise(0)
          .alias("score")
    )

    # Keep the highest-score row per accession (explicit group-by for clarity)
    best_meta = (
        meta_df
        .with_columns(pref)
        .sort([acc_col, "score"], descending=[False, True])
        .group_by(acc_col)
        .agg(pl.all().first())
        .drop("score")
    )

    # 4) join back to original df
    return df.join(best_meta, on=acc_col, how="left")


# -------------------------------
# Example
# -------------------------------
if __name__ == "__main__":
    Entrez.email = "nathanglen@ufl.edu"
    df = pl.DataFrame({"accession": ["SRX1166121", "SRX7636841"]})
    out = enrich_with_biosample(df, acc_col="accession")
    print(out)
