"""
Fallback classification pipeline for samples with no cancer signal.

Provides a pluggable architecture for escalating classification through
progressively more expensive methods: expanded column search → API
enrichment → LLM classification.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import polars as pl

from config import (
    CANCER_POS,
    CANCER_NEG,
    ClassificationLabel as CL,
    MedSpaCyLabel as ML,
)

if TYPE_CHECKING:
    from spacy.language import Language


# =============================================================================
# Data Types
# =============================================================================


@dataclass
class FallbackResult:
    """Standardized result from any fallback provider."""

    label: str  # ClassificationLabel value
    confidence: float  # 0.0 - 1.0
    reason: str  # Human-readable explanation
    provider_name: str  # Which provider produced this result

    @property
    def is_cancer(self) -> bool:
        return self.label in (
            CL.CONFIDENT_CANCER.value,
            CL.LIKELY_CANCER.value,
            CL.CONFIRMED_BY_MEDSPACY.value,
        )

    @property
    def resolved(self) -> bool:
        """True if the provider made a confident determination."""
        return self.confidence >= 0.5


# =============================================================================
# Provider Protocol
# =============================================================================


class FallbackProvider(ABC):
    """
    Abstract base for any fallback classification provider.

    Implement `classify()` to create a new provider. The pipeline
    calls providers in order and stops at the first resolved result.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider name for metadata tracking."""
        ...

    @abstractmethod
    def classify(self, sample: Dict[str, Any]) -> FallbackResult:
        """
        Classify a single sample.

        Args:
            sample: Dict of all column values for one row.

        Returns:
            FallbackResult with label, confidence, reason, provider_name.
        """
        ...

    def classify_batch(self, samples: List[Dict[str, Any]]) -> List[FallbackResult]:
        """
        Classify multiple samples. Override for batched API calls.
        Default implementation calls classify() per sample.
        """
        return [self.classify(s) for s in samples]


# =============================================================================
# Built-in Provider: Expanded Column Search
# =============================================================================


class ExpandedSearchProvider(FallbackProvider):
    """
    Scans ALL text columns (not just priority) for cancer terms.

    This catches cases where cancer info is in columns like
    `library_construction_protocol`, `experiment_alias`, or
    auto-discovered metadata fields.
    """

    @property
    def name(self) -> str:
        return "expanded_search"

    def __init__(
        self,
        nlp_pipeline: Optional["Language"] = None,
        exclude_patterns: Tuple[str, ...] = (
            "accession",
            "uuid",
            "hash",
            "checksum",
            "md5",
            "sha",
            "url",
            "path",
            "date",
            "time",
            "run_",
            "experiment_",
            "study_",
            "sample_accession",
            "biosample",
            "bioproject",
            "library_",
            "protocol",
            "single-cell",
            "bulk",
        ),
    ):
        self._nlp = nlp_pipeline
        self._exclude = exclude_patterns

    def _get_nlp(self) -> "Language":
        if self._nlp is None:
            from pipeline import get_nlp

            self._nlp = get_nlp()
        return self._nlp

    def classify(self, sample: Dict[str, Any]) -> FallbackResult:
        """Search all text columns for cancer terms."""
        cancer_hits: List[str] = []
        non_cancer_hits: List[str] = []

        for col, val in sample.items():
            if not isinstance(val, str) or not val.strip():
                continue

            # Skip ID/accession columns
            col_lower = col.lower()
            if any(p in col_lower for p in self._exclude):
                continue

            # Skip columns already searched in priority stage
            if col in (
                "title",
                "source_name",
                "tissue",
                "disease",
                "cell_type",
                "tumor_type",
                "phenotype",
            ):
                continue

            text = val.strip().lower()

            # Regex check
            if re.search(CANCER_POS, text, re.IGNORECASE):
                cancer_hits.append(f"{col}: {val[:50]}")
            if re.search(CANCER_NEG, text, re.IGNORECASE):
                non_cancer_hits.append(f"{col}: {val[:50]}")

        # Also try NLP on concatenated non-priority text
        if not cancer_hits:
            extra_texts = []
            for col, val in sample.items():
                if not isinstance(val, str) or not val.strip():
                    continue
                col_lower = col.lower()
                if any(p in col_lower for p in self._exclude):
                    continue
                if col in (
                    "title",
                    "source_name",
                    "tissue",
                    "disease",
                    "cell_type",
                    "tumor_type",
                    "phenotype",
                ):
                    continue
                extra_texts.append(val.strip())

            if extra_texts:
                combined = " ".join(extra_texts).lower()
                nlp = self._get_nlp()
                doc = nlp(combined)
                for ent in doc.ents:
                    if ent.label_ == "CANCER" and not getattr(
                        ent._, "is_negated", False
                    ):
                        cancer_hits.append(f"nlp:{ent.text}")
                    elif ent.label_ == "NON_CANCER":
                        non_cancer_hits.append(f"nlp:{ent.text}")

        if cancer_hits and not non_cancer_hits:
            return FallbackResult(
                label=CL.LIKELY_CANCER.value,
                confidence=0.7,
                reason=f"expanded_search found: {', '.join(cancer_hits[:3])}",
                provider_name=self.name,
            )
        elif cancer_hits and non_cancer_hits:
            return FallbackResult(
                label=CL.LIKELY_CANCER.value,
                confidence=0.5,
                reason=f"mixed signals: cancer=[{', '.join(cancer_hits[:2])}] non_cancer=[{', '.join(non_cancer_hits[:2])}]",
                provider_name=self.name,
            )

        return FallbackResult(
            label=CL.CONFIRMED_NON_CANCER.value,
            confidence=0.0,
            reason="no_additional_terms_found",
            provider_name=self.name,
        )


# =============================================================================
# Fallback Pipeline Orchestrator
# =============================================================================


class FallbackPipeline:
    """
    Orchestrates fallback providers in order for unresolved samples.

    Usage:
        pipeline = FallbackPipeline([
            ExpandedSearchProvider(),
            NCBIProvider(email="..."),
            GeminiProvider(api_key="..."),
        ])
        df = pipeline.process(df)
    """

    def __init__(self, providers: List[FallbackProvider]):
        self.providers = providers

    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply fallback classification to no-signal samples.

        Only processes rows where:
        - confidence_category == "confirmed_non_cancer"
        - med_reason contains "no_relevant_terms"

        Adds/updates columns: confidence_category, resolved_by, fallback_reason
        """
        # Ensure resolved_by column exists
        if "resolved_by" not in df.columns:
            df = df.with_columns(
                pl.when(
                    pl.col("confidence_category").is_in(
                        [
                            CL.CONFIDENT_CANCER.value,
                            CL.LIKELY_CANCER.value,
                        ]
                    )
                )
                .then(pl.lit("regex"))
                .when(pl.col("confidence_category") == CL.CONFIRMED_BY_MEDSPACY.value)
                .then(pl.lit("medspacy"))
                .when(
                    pl.col("confidence_category").is_in(
                        [
                            CL.CONFIRMED_NON_CANCER.value,
                            CL.LIKELY_NON_CANCER.value,
                        ]
                    )
                )
                .then(pl.lit("regex+medspacy"))
                .otherwise(pl.lit("default"))
                .alias("resolved_by")
            )

        if "fallback_reason" not in df.columns:
            df = df.with_columns(pl.lit("").alias("fallback_reason"))

        # Find unresolved samples (no_relevant_terms)
        mask = (
            pl.col("confidence_category") == CL.CONFIRMED_NON_CANCER.value
        ) & pl.col("med_reason").str.contains("no_relevant_terms")

        unresolved_indices = (
            df.with_row_index("_idx").filter(mask).get_column("_idx").to_list()
        )

        if not unresolved_indices:
            return df

        rows = df.to_dicts()

        for provider in self.providers:
            still_unresolved = [
                i
                for i in unresolved_indices
                if rows[i].get("_resolved", False) is False
            ]

            if not still_unresolved:
                break

            samples = [rows[i] for i in still_unresolved]
            results = provider.classify_batch(samples)

            for idx, result in zip(still_unresolved, results):
                if result.resolved:
                    rows[idx]["confidence_category"] = result.label
                    rows[idx]["resolved_by"] = result.provider_name
                    rows[idx]["fallback_reason"] = result.reason
                    rows[idx]["_resolved"] = True

        # Rebuild DataFrame
        new_categories = [r["confidence_category"] for r in rows]
        new_resolved = [r.get("resolved_by", "default") for r in rows]
        new_reasons = [r.get("fallback_reason", "") for r in rows]

        df = df.with_columns(
            [
                pl.Series("confidence_category", new_categories),
                pl.Series("resolved_by", new_resolved),
                pl.Series("fallback_reason", new_reasons),
            ]
        )

        return df
