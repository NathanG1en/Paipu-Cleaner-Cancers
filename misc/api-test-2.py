import requests
from lxml import etree

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def fetch_sra_xml_from_run(run_accession: str):
    """Given SRR accession → return parsed XML record."""
    # Step 1: Search SRA db for internal UID
    esearch_url = f"{NCBI_BASE}/esearch.fcgi?db=sra&term={run_accession}"
    esearch_xml = etree.fromstring(requests.get(esearch_url).content)
    uid = esearch_xml.find(".//Id").text

    # Step 2: Fetch full SRA XML
    efetch_url = f"{NCBI_BASE}/efetch.fcgi?db=sra&id={uid}&rettype=xml"
    sra_xml = etree.fromstring(requests.get(efetch_url).content)
    return sra_xml


def extract_basic_metadata(xml_root):
    """Extract major metadata fields, handling attributes correctly."""
    def find_text(path):
        """XPath for element text only."""
        el = xml_root.find(path)
        return el.text if el is not None else None

    def xpath_first(path):
        """General XPath helper for attributes or nodes."""
        out = xml_root.xpath(path)
        return out[0] if out else None

    return {
        "study_title": find_text(".//STUDY_TITLE"),
        "study_accession": xpath_first("string(.//STUDY_REF/@accession)"),
        "bioproject": xpath_first("string(.//EXTERNAL_ID[@namespace='BioProject'])"),
        "biosample": xpath_first("string(.//EXTERNAL_ID[@namespace='BioSample'])"),
        "experiment_accession": xpath_first("string(.//EXPERIMENT/@accession)"),
        "sample_accession": xpath_first("string(.//SAMPLE/@accession)"),
        "run_accession": xpath_first("string(.//RUN/@accession)"),
    }


def main():
    run = "SRX24570384"  # test immediately

    print(f"Fetching metadata for {run} ...")
    xml_root = fetch_sra_xml_from_run(run)
    meta = extract_basic_metadata(xml_root)

    print("\n=== SRA METADATA ===")
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
