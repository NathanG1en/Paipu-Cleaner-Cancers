"""
Configuration settings for cancer sample classification.

This module centralizes all configuration constants and provides
a dataclass-based configuration system for the classification pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


@dataclass(frozen=True)
class ClassifierConfig:
    """
    Immutable configuration for the cancer classification pipeline.
    
    Attributes:
        priority_cols: Primary columns to search for cancer indicators.
        secondary_cols: Additional columns to check if present.
        exclude_patterns: Column name patterns to exclude from auto-discovery.
        min_avg_length: Minimum average string length for viable text columns.
        min_non_null_pct: Minimum non-null percentage for viable text columns.
        batch_size: Default batch size for MedSpaCy processing.
    """
    
    # Tier 1: Always process - biologically meaningful columns
    priority_cols: Tuple[str, ...] = (
        "source_name", "tissue", "phenotype", "disease",
        "cell_type", "tumor_type", "cancer_type"
    )
    
    # Tier 2: Process if present and populated - secondary metadata
    secondary_cols: Tuple[str, ...] = (
        "sample_name", "condition", "health_state", "tissue_type",
        "celltype", "model", "cell_types", "tissue_cell_type"
    )
    
    # Columns to never process (IDs, hashes, etc.)
    exclude_patterns: Tuple[str, ...] = (
        "_id", "accession", "uuid", "hash", "checksum", "md5", "sha",
        "url", "path", "file", "date", "time"
    )
    
    # Thresholds for auto-discovery of additional columns
    min_avg_length: float = 10.0
    min_non_null_pct: float = 0.01
    
    # Processing settings
    batch_size: int = 64


@dataclass(frozen=True)
class RegexPatterns:
    """
    Regex patterns for cancer detection.
    
    These patterns are used for the initial regex-based classification
    stage before MedSpaCy NLP processing.
    """
    
    # Positive cancer indicators
    cancer_positive: str = (
        r"(?:\bcancers?\b|\btumou?rs?\b|\bmalignan(?:t|cy)\b|\bcarcinomas?\b|"
        r"\bneoplasms?\b|\bmetasta(?:s|t)es?\b|\badenocarcinomas?\b|\bsarcomas?\b|"
        r"\bleukemi(?:a|as)\b|\blymphom(?:a|as)\b|\bglioblastomas?\b|\bmelanomas?\b|"
        r"\boncolog(?:y|ic|ical)\b)"
    )
    
    # Negative/control indicators
    cancer_negative: str = (
        r"(?:\bnormal\b|\bhealthy\b|\bctrl\b|\badjacent normal\b|"
        r"\bnon[-\s]?tumou?r(?:al)?\b|\bbenign\b|\bnon[-\s]?cancer(?:ous)?\b|"
        r"\bsham\b|\bunaffected\b)"
    )
    
    # False positive traps (species/protein names containing "onco")
    onco_traps: str = (
        r"(?:\boncophora\b|\boncorhynchus\b|\boncotic\b|\boncomodulin\b)"
    )


@dataclass(frozen=True)
class LabelMapping:
    """
    Label mappings for final classification output.
    """
    
    # Map confidence categories to final labels
    final_label_map: Dict[str, str] = field(default_factory=lambda: {
        "confident_cancer": "CANCER",
        "likely_cancer": "CANCER",
        "confirmed_by_medspacy": "CANCER",
        "confirmed_non_cancer": "NON_CANCER",
        "likely_non_cancer": "NON_CANCER",
        "uncertain_no_signal": "UNCERTAIN",
        "uncertain_weak_signal": "UNCERTAIN",
        "uncertain_medspacy": "UNCERTAIN",
    })


# =============================================================================
# Module-level constants (for backward compatibility)
# =============================================================================

# Default configuration instance
DEFAULT_CONFIG = ClassifierConfig()

# Expose commonly used values at module level for convenience
PRIORITY_COLS: Tuple[str, ...] = DEFAULT_CONFIG.priority_cols
SECONDARY_COLS: Tuple[str, ...] = DEFAULT_CONFIG.secondary_cols
EXCLUDE_PATTERNS: Tuple[str, ...] = DEFAULT_CONFIG.exclude_patterns

# Default regex patterns instance
DEFAULT_PATTERNS = RegexPatterns()

# Expose patterns at module level
CANCER_POS: str = DEFAULT_PATTERNS.cancer_positive
CANCER_NEG: str = DEFAULT_PATTERNS.cancer_negative
ONCO_TRAPS: str = DEFAULT_PATTERNS.onco_traps

# Default label mapping
DEFAULT_LABEL_MAP = LabelMapping()
FINAL_LABEL_MAP: Dict[str, str] = DEFAULT_LABEL_MAP.final_label_map


# =============================================================================
# Cancer detection keywords (used for rule generation)
# =============================================================================

CANCER_KEYWORDS: FrozenSet[str] = frozenset({
    "cancer", "tumor", "tumour", "carcinoma", "sarcoma", "lymphoma",
    "leukemia", "leukaemia", "melanoma", "glioma", "blastoma", "myeloma",
    "neoplasm", "malignant", "metastatic", "adenoma", "oncology"
})

SPECIFIC_CANCER_TYPES: FrozenSet[str] = frozenset({
    "carcinoma", "sarcoma", "lymphoma", "leukemia", "leukaemia",
    "melanoma", "glioma", "blastoma", "myeloma"
})
