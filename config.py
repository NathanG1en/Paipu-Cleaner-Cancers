"""
Configuration settings for cancer sample classification.

This module centralizes all configuration constants and provides
a dataclass-based configuration system for the classification pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Tuple
import medspacy
from medspacy.context import ConTextRule


@dataclass(frozen=True)
class ClassifierConfig:
    """
    Immutable configuration for the cancer classification pipeline.
    
    This is the single source of truth for column configuration,
    used by both text preprocessing and classification functions.
    
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
        "title", "source_name", "tissue", "phenotype", "disease",
        "cell_type", "tumor_type"
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


# Backward compatibility alias
TextColumnConfig = ClassifierConfig


@dataclass(frozen=True)
class RegexPatterns:
    """
    Regex patterns for cancer detection.
    
    These patterns are used for the initial regex-based classification
    stage before MedSpaCy NLP processing.
    """
    
    # Positive cancer indicators
    cancer_positive: str = (
        r"(?:\bcancers?\b|\bcancerous\b|\btumou?rs?\b|\bmalignan(?:t|cy)\b|\bcarcinomas?\b|"
        r"\bneoplasms?\b|\bneoplastic\b|\bneoplasia\b|"
        r"\bmetasta(?:s|t)(?:is|es)?\b|\bmetastatic\b|"
        r"\badenocarcinomas?\b|\bsarcomas?\b|"
        r"\bleuk[ae]mias?\b|\blymphomas?\b|\bglioblastomas?\b|\bmelanomas?\b|"
        r"\boncolog(?:y|ic|ical)\b|\boncogen(?:ic|e|es)\b|"
        r"\bhemangiosarcomas?\b|\bhaemangiosarcomas?\b|"
        r"\boligodendrogliomas?\b|\bmastocytomas?\b|\bfibrosarcomas?\b|"
        r"\bgliomas?\b|\bpheochromocytomas?\b|\bintratumou?ral\b|\bTILs?\b|"
        r"\bDLBCL\b|\bTNBC\b|\bHCC\b|\bNSCLC\b|\bSCLC\b|\bRCC\b|"
        r"\bAML\b|\bCLL\b|\bMCL\b|"
        r"\bCTVT\w*\b|\bwalker\s?256\b|"
        r"\badcarc\w*\b|\bmetadcarc\w*\b)"
    )
    
    # Negative/control indicators
    cancer_negative: str = (
        r"(?:\bnormal\b|\bhealthy\b|\bctrl\b|\badjacent normal\b|"
        r"\bnon[-\s]?tumou?r(?:al)?\b|\bbenign\b|\bnon[-\s]?cancer(?:ous)?\b|"
        r"\bsham\b|\bunaffected\b|"
        r"\btumou?r necrosis factor\b|\btumou?r microenvironment\b|"
        r"\bcontralateral normal\b|\bno\s+tumou?r\b)"
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
# MedSpaCy Target Rule Definitions (data only, no TargetRule import)
# =============================================================================
# These are stored as tuples of (literal, category, pattern) to avoid
# importing medspacy in the config module. The actual TargetRule objects
# are created in functions.py using get_default_target_rules().

CANCER_RULE_DEFINITIONS: List[Tuple[str, str, str]] = [
    # General cancer terms: (literal, category, regex_pattern)
    ("", "CANCER", r"\bcancers?\b"),
    ("", "CANCER", r"\btumou?rs?\b"),
    ("", "CANCER", r"\bmalignan(?:t|cy)\b"),
    ("", "CANCER", r"\bcarcinomas?\b"),
    ("", "CANCER", r"\bneoplasms?\b"),
    ("", "CANCER", r"\bmetasta(?:s|t)(?:is|es)?\b"),
    ("", "CANCER", r"\badenocarcinomas?\b"),
    ("", "CANCER", r"\bsarcomas?\b"),
    ("", "CANCER", r"\bleuk[ae]mias?\b"),
    ("", "CANCER", r"\blymphomas?\b"),
    ("", "CANCER", r"\bglioblastomas?\b"),
    ("", "CANCER", r"\bmelanomas?\b"),
    ("", "CANCER", r"\bmyelomas?\b"),
    ("", "CANCER", r"\bneuroblastomas?\b"),
    ("", "CANCER", r"\boncogen(?:ic|e|es)\b"),
    ("", "CANCER", r"\badenomas?\b"),
    ("", "CANCER", r"\bosteosarcoma?\b"),
    ("", "CANCER", r"\bmeningiomas?\b"),
    ("", "CANCER", r"\bgliomas?\b"),
    ("", "CANCER", r"\bpheochromocytomas?\b"),
    ("", "CANCER", r"\bintratumou?ral\b"),
    ("", "CANCER", r"\bTILs?\b"),
    # Specific cancer types (literal match only, no pattern)
    # Just the literals
    ("cancer", "CANCER",""),
    ("tumor", "CANCER",""),
    ("malignant", "CANCER",""),
    ("carcinoma", "CANCER",""),
    ("neoplasm", "CANCER",""),
    ("metastasis", "CANCER",""),
    ("adenocarcinoma", "CANCER",""),
    ("sarcoma", "CANCER",""),
    ("leukemia", "CANCER",""),
    ("lymphoma", "CANCER",""),
    ("glioblastoma", "CANCER",""),
    ("melanoma", "CANCER",""),
    ("myeloma", "CANCER",""),
    ("neuroblastoma", "CANCER",""),
    ("oncogenic", "CANCER",""),
    ("adenoma", "CANCER",""),
    ("osteosarcoma", "CANCER",""),
    ("meningioma", "CANCER",""),
    ("glioma", "CANCER", ""),
    ("pheochromocytoma", "CANCER", ""),
    ("intratumoral", "CANCER", ""),
    ("TIL", "CANCER", ""),
    ("TILs", "CANCER", ""),

    # Adjective/variant forms missed by base patterns
    ("", "CANCER", r"\bneoplastic\b"),
    ("", "CANCER", r"\bneoplasia\b"),
    ("", "CANCER", r"\bmetastatic\b"),
    ("", "CANCER", r"\btumourous\b"),
    ("neoplastic", "CANCER", ""),
    ("neoplasia", "CANCER", ""),
    ("metastatic", "CANCER", ""),

    # Cancer abbreviations
    ("", "CANCER", r"\bDLBCL\b"),          # Diffuse Large B-Cell Lymphoma
    # NOTE: \bFL\b removed - matches floxed allele notation (fl/fl) too broadly
    ("", "CANCER", r"\bTNBC\b"),           # Triple-Negative Breast Cancer
    ("", "CANCER", r"\bHCC\b"),            # Hepatocellular Carcinoma
    ("", "CANCER", r"\bNSCLC\b"),          # Non-Small Cell Lung Cancer
    ("", "CANCER", r"\bSCLC\b"),           # Small Cell Lung Cancer
    ("", "CANCER", r"\bRCC\b"),            # Renal Cell Carcinoma
    ("", "CANCER", r"\bAML\b"),            # Acute Myeloid Leukemia
    ("", "CANCER", r"\bCLL\b"),            # Chronic Lymphocytic Leukemia
    ("", "CANCER", r"\bALL\b"),            # Acute Lymphoblastic Leukemia
    ("", "CANCER", r"\bMCL\b"),            # Mantle Cell Lymphoma
    ("DLBCL", "CANCER", ""),
    ("TNBC", "CANCER", ""),
    ("HCC", "CANCER", ""),
    ("PDAC", "CANCER", ""),
    ("", "CANCER", r"\bPDAC\b"),            # Pancreatic Ductal Adenocarcinoma

    # Veterinary / comparative oncology cancer types
    ("", "CANCER", r"\bhemangiosarcomas?\b"),
    ("", "CANCER", r"\bhaemangiosarcomas?\b"),
    ("", "CANCER", r"\boligodendrogliomas?\b"),
    ("", "CANCER", r"\bmastocytomas?\b"),
    ("", "CANCER", r"\bfibrosarcomas?\b"),
    ("", "CANCER", r"\bchondrosarcomas?\b"),
    ("", "CANCER", r"\bliposarcomas?\b"),
    ("", "CANCER", r"\bleiomyosarcomas?\b"),
    ("", "CANCER", r"\brhabdomyosarcomas?\b"),
    ("", "CANCER", r"\bhistiocytic sarcomas?\b"),
    ("hemangiosarcoma", "CANCER", ""),
    ("haemangiosarcoma", "CANCER", ""),
    ("oligodendroglioma", "CANCER", ""),
    ("mastocytoma", "CANCER", ""),
    ("fibrosarcoma", "CANCER", ""),

    # Specific carcinoma subtypes
    ("", "CANCER", r"\burothelial carcinomas?\b"),
    ("", "CANCER", r"\bductal carcinomas?\b"),
    ("", "CANCER", r"\bmammary carcinomas?\b"),
    ("", "CANCER", r"\bprostate carcinomas?\b"),
    ("", "CANCER", r"\btransitional cell carcinomas?\b"),
    ("urothelial carcinoma", "CANCER", ""),
    ("ductal carcinoma", "CANCER", ""),
    ("mammary ductal carcinoma", "CANCER", ""),
    ("mammary carcinoma", "CANCER", ""),
    ("prostate carcinoma", "CANCER", ""),
    ("transitional cell carcinoma", "CANCER", ""),

    # Multi-word cancer descriptions
    ("hepatocellular carcinoma", "CANCER", ""),
    ("breast cancer", "CANCER", ""),
    ("lung cancer", "CANCER", ""),
    ("colon cancer", "CANCER", ""),
    ("prostate cancer", "CANCER", ""),
    ("pancreatic cancer", "CANCER", ""),
    ("ovarian cancer", "CANCER", ""),
    ("bladder cancer", "CANCER", ""),
    ("skin cancer", "CANCER", ""),
    ("brain cancer", "CANCER", ""),
    ("liver cancer", "CANCER", ""),
    ("kidney cancer", "CANCER", ""),
    ("renal cell carcinoma", "CANCER", ""),
    ("squamous cell carcinoma", "CANCER", ""),
    ("basal cell carcinoma", "CANCER", ""),
    ("non-small cell lung cancer", "CANCER", ""),
    ("small cell lung cancer", "CANCER", ""),
    ("triple negative breast cancer", "CANCER", ""),
    ("HER2 positive", "CANCER", ""),
    ("ER positive", "CANCER", ""),
    ("cutaneous melanoma", "CANCER", ""),
    ("oral melanoma", "CANCER", ""),
    ("soft tissue sarcoma", "CANCER", ""),
    ("mammary tumor", "CANCER", ""),
    ("mammary tumour", "CANCER", ""),
    ("b-cell lymphoma", "CANCER", ""),
    ("b cell lymphoma", "CANCER", ""),
    ("t-cell lymphoma", "CANCER", ""),
    ("t cell lymphoma", "CANCER", ""),

    # Additional patterns for non-mouse datasets
    ("", "CANCER", r"\bcancerous\b"),
    ("", "CANCER", r"\bCTVT\w*\b"),              # Canine Transmissible Venereal Tumor
    ("CTVT", "CANCER", ""),
    ("", "CANCER", r"\bwalker\s?256\b"),       # Walker 256 carcinosarcoma cell line
    ("Walker 256", "CANCER", ""),
    ("Walker256", "CANCER", ""),
    ("", "CANCER", r"\badcarc\w*\b"),             # Abbreviated adenocarcinoma
    ("", "CANCER", r"\bmetadcarc\w*\b"),          # Abbreviated metastatic adenocarcinoma
    ("cancerous", "CANCER", ""),
]

NON_CANCER_RULE_DEFINITIONS: List[Tuple[str, str, str]] = [
    # Non-cancer/control terms: (literal, category, regex_pattern)
    ("normal", "NON_CANCER", r"\bnormal\b"),
    ("healthy", "NON_CANCER", r"\bhealthy\b"),
    ("control", "NON_CANCER", r"\b(?:ctrl|control)\b"),
    ("benign", "NON_CANCER", r"\bbenign\b"),
    ("non-tumor", "NON_CANCER", r"\bnon[-\s]?tumou?r(?:al)?\b"),
    ("non-cancer", "NON_CANCER", r"\bnon[-\s]?cancer(?:ous)?\b"),
    ("sham", "NON_CANCER", r"\bsham\b"),
    ("unaffected", "NON_CANCER", r"\bunaffected\b"),
    ("wild type", "NON_CANCER", r"\bwild[-\s]?type\b"),
    ("WT", "NON_CANCER", r"\bWT\b"),
    # Literal match only
    ("adjacent normal", "NON_CANCER", ""),
    ("tumor-adjacent normal", "NON_CANCER", ""),
    ("matched normal", "NON_CANCER", ""),
    ("contralateral normal", "NON_CANCER", ""),
    # Non-cancer traps: terms with 'tumor' that don't indicate cancer
    ("tumor necrosis factor", "NON_CANCER", r"\btumou?r necrosis factor\b"),
    ("tumor microenvironment", "NON_CANCER", r"\btumou?r microenvironment\b"),
    # Onco-traps (false positives - species/proteins)
    ("oncorhynchus", "NON_CANCER", r"\boncorhynchus\b"),
    ("oncophora", "NON_CANCER", r"\boncophora\b"),
    ("oncotic", "NON_CANCER", r"\boncotic\b"),
    ("oncomodulin", "NON_CANCER", r"\boncomodulin\b"),
]


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
    "neoplasm", "neoplastic", "neoplasia", "malignant", "metastatic",
    "adenoma", "oncology", "hemangiosarcoma", "haemangiosarcoma",
    "oligodendroglioma", "mastocytoma", "fibrosarcoma",
    "chondrosarcoma", "liposarcoma", "leiomyosarcoma",
    "rhabdomyosarcoma", "glioma", "pheochromocytoma", "intratumoral",
})

SPECIFIC_CANCER_TYPES: FrozenSet[str] = frozenset({
    "carcinoma", "sarcoma", "lymphoma", "leukemia", "leukaemia",
    "melanoma", "glioma", "blastoma", "myeloma",
    "hemangiosarcoma", "haemangiosarcoma", "oligodendroglioma",
    "mastocytoma", "fibrosarcoma", "chondrosarcoma",
    "liposarcoma", "leiomyosarcoma", "rhabdomyosarcoma",
})

# =============================================================================
# MedSpaCy Context Rule Definitions (negation detection)
# =============================================================================
# These are stored as tuples of (literal, category, direction) to avoid
# importing medspacy in the config module. The actual ConTextRule objects
# are created in functions.py during pipeline initialization.

CONTEXT_RULE_DEFINITIONS: List[Tuple[str, str, str]] = [
    # Prefix-based negations: (literal, category, direction)
    ("non-", "NEGATED_EXISTENCE", "FORWARD"),
    ("non -", "NEGATED_EXISTENCE", "FORWARD"),  # Handles tokenization of "non-X"
    ("non", "NEGATED_EXISTENCE", "FORWARD"),    # Handles "non X" with space
    
    # Additional negation patterns
    ("no", "NEGATED_EXISTENCE", "FORWARD"),
    ("without", "NEGATED_EXISTENCE", "FORWARD"),
    ("absence of", "NEGATED_EXISTENCE", "FORWARD"),
    ("free of", "NEGATED_EXISTENCE", "FORWARD"),
    ("negative for", "NEGATED_EXISTENCE", "FORWARD"),
    ("no evidence of", "NEGATED_EXISTENCE", "FORWARD"),
    ("ruled out", "NEGATED_EXISTENCE", "BACKWARD"),
    ("denies", "NEGATED_EXISTENCE", "FORWARD"),
    ("denied", "NEGATED_EXISTENCE", "FORWARD"),
]
