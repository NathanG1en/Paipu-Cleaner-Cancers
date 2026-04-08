"""
NCBI E-utilities provider for BioProject/BioSample metadata enrichment.

Fetches additional metadata from NCBI when local metadata is insufficient
for cancer classification. Uses Entrez API (requires email).

Features:
- Disk-based JSON cache (API calls only happen once per BioProject)
- Targeted XML field extraction (title + description only)
- Sample-level negative context check
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from fallback import FallbackProvider, FallbackResult
from config import CANCER_POS, CANCER_NEG, ClassificationLabel as CL


class NCBIProvider(FallbackProvider):
    """
    Fetches BioProject metadata from NCBI E-utilities
    and searches for cancer terms in the project title/description.

    Results are cached to disk so API calls only happen once per BioProject.

    Usage:
        provider = NCBIProvider(email="your@email.com")
        result = provider.classify(sample_dict)
    """

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        rate_limit: float = 0.34,
        cache_path: str = "data/ncbi_cache.json",
    ):
        self._email = email
        self._api_key = api_key
        self._rate_limit = rate_limit if not api_key else 0.1
        self._cache_path = Path(cache_path)
        self._cache: Dict[str, Optional[Dict[str, str]]] = self._load_cache()
        self._last_request: float = 0

    @property
    def name(self) -> str:
        return "ncbi_api"

    # =========================================================================
    # Disk Cache
    # =========================================================================

    def _load_cache(self) -> Dict[str, Optional[Dict[str, str]]]:
        """Load cached BioProject data from disk."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    # =========================================================================
    # API Fetching
    # =========================================================================

    def _throttle(self) -> None:
        """Respect NCBI rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)
        self._last_request = time.time()

    def _fetch_bioproject(self, bioproject_id: str) -> Optional[Dict[str, str]]:
        """
        Fetch BioProject metadata from NCBI.

        Returns dict with 'title' and 'description' keys,
        or None if fetch failed.
        """
        if bioproject_id in self._cache:
            return self._cache[bioproject_id]

        try:
            import urllib.request
            import urllib.parse

            self._throttle()

            # Step 1: Search for the BioProject UID
            params = urllib.parse.urlencode({
                "db": "bioproject",
                "term": bioproject_id,
                "retmode": "xml",
                "email": self._email,
                **({"api_key": self._api_key} if self._api_key else {}),
            })
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                search_xml = resp.read().decode("utf-8")

            root = ET.fromstring(search_xml)
            id_list = root.findall(".//Id")
            if not id_list:
                self._cache[bioproject_id] = None
                return None

            uid = id_list[0].text

            self._throttle()

            # Step 2: Fetch the full record
            params = urllib.parse.urlencode({
                "db": "bioproject",
                "id": uid,
                "retmode": "xml",
                "email": self._email,
                **({"api_key": self._api_key} if self._api_key else {}),
            })
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                detail_xml = resp.read().decode("utf-8")

            # Extract ONLY title and description — not the full XML dump
            detail_root = ET.fromstring(detail_xml)

            title = ""
            description = ""

            # BioProject title
            title_elem = detail_root.find(".//ProjectDescr/Title")
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

            # BioProject description
            desc_elem = detail_root.find(".//ProjectDescr/Description")
            if desc_elem is not None and desc_elem.text:
                description = desc_elem.text.strip()

            # Also check project name as fallback
            if not title:
                name_elem = detail_root.find(".//ProjectDescr/Name")
                if name_elem is not None and name_elem.text:
                    title = name_elem.text.strip()

            result = {"title": title, "description": description}
            self._cache[bioproject_id] = result
            return result

        except Exception:
            self._cache[bioproject_id] = None
            return None

    # =========================================================================
    # Classification
    # =========================================================================

    @staticmethod
    def _sample_has_negative_context(sample: Dict[str, Any]) -> bool:
        """
        Check if sample-level columns indicate this is a control/normal sample.

        If the sample itself says 'normal', 'control', 'healthy', we should
        NOT override that with study-level BioProject cancer context.
        """
        sample_cols = ("source_name", "tissue", "phenotype", "cell_type")

        for col in sample_cols:
            val = sample.get(col)
            if not val or not isinstance(val, str):
                continue
            val_lower = val.strip().lower()
            if val_lower in ("", "nan", "null", "none"):
                continue
            if re.search(CANCER_NEG, val_lower, re.IGNORECASE):
                return True
        return False

    def classify(self, sample: Dict[str, Any]) -> FallbackResult:
        """Fetch NCBI metadata and search for cancer terms."""
        bioproject = sample.get("bioproject") or sample.get("study_accession")

        if not bioproject or not isinstance(bioproject, str):
            return FallbackResult(
                label=CL.CONFIRMED_NON_CANCER.value,
                confidence=0.0,
                reason="no_bioproject_id",
                provider_name=self.name,
            )

        data = self._fetch_bioproject(bioproject.strip())

        if not data:
            return FallbackResult(
                label=CL.CONFIRMED_NON_CANCER.value,
                confidence=0.0,
                reason=f"ncbi_fetch_failed:{bioproject}",
                provider_name=self.name,
            )

        # Search title and description for cancer terms
        combined = f"{data.get('title', '')} {data.get('description', '')}".lower()
        cancer_match = re.search(CANCER_POS, combined, re.IGNORECASE)

        if not cancer_match:
            return FallbackResult(
                label=CL.CONFIRMED_NON_CANCER.value,
                confidence=0.0,
                reason=f"ncbi_bioproject:{bioproject} no_cancer_terms",
                provider_name=self.name,
            )

        # Cancer found in BioProject — but check sample-level negative context
        if self._sample_has_negative_context(sample):
            return FallbackResult(
                label=CL.CONFIRMED_NON_CANCER.value,
                confidence=0.0,
                reason=f"ncbi_bioproject:{bioproject} matched '{cancer_match.group()}' but sample has negative context",
                provider_name=self.name,
            )

        # Title match is stronger signal than description-only match
        title_match = re.search(CANCER_POS, data.get("title", "").lower(), re.IGNORECASE)
        if title_match:
            confidence = 0.7
            field = "title"
        else:
            confidence = 0.55
            field = "description"

        return FallbackResult(
            label=CL.LIKELY_CANCER.value,
            confidence=confidence,
            reason=f"ncbi_bioproject:{bioproject} {field}:'{cancer_match.group()}'",
            provider_name=self.name,
        )

    def classify_batch(self, samples: List[Dict[str, Any]]) -> List[FallbackResult]:
        """Classify batch with caching — only fetches each BioProject once."""
        # Pre-warm cache for unique BioProjects
        bioprojects = set()
        for s in samples:
            bp = s.get("bioproject") or s.get("study_accession")
            if bp and isinstance(bp, str):
                bioprojects.add(bp.strip())

        for bp in bioprojects:
            if bp not in self._cache:
                self._fetch_bioproject(bp)

        # Save cache to disk after fetching
        self._save_cache()

        return [self.classify(s) for s in samples]
