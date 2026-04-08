"""
MedSpaCy NLP pipeline management.

Provides singleton pattern for lazy initialization of the MedSpaCy pipeline,
including target rules and context rules for cancer/non-cancer detection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Tuple

from medspacy.ner import TargetRule

from config import (
    CANCER_KEYWORDS,
    SPECIFIC_CANCER_TYPES,
    CANCER_RULE_DEFINITIONS,
    NON_CANCER_RULE_DEFINITIONS,
    CONTEXT_RULE_DEFINITIONS,
)

if TYPE_CHECKING:
    from spacy.language import Language


# =============================================================================
# Target Rule Generation
# =============================================================================

@lru_cache(maxsize=1)
def get_default_target_rules() -> Tuple[Tuple[TargetRule, ...], Tuple[TargetRule, ...]]:
    """
    Get default TargetRules for cancer and non-cancer entity detection.
    Cached to avoid recreating rules on every call.

    Returns:
        Tuple of (cancer_rules, non_cancer_rules).
    """
    cancer_rules: List[TargetRule] = []
    for literal, category, pattern in CANCER_RULE_DEFINITIONS:
        if pattern:
            cancer_rules.append(TargetRule(literal, category, pattern=pattern))
        else:
            cancer_rules.append(TargetRule(literal, category))

    non_cancer_rules: List[TargetRule] = []
    for literal, category, pattern in NON_CANCER_RULE_DEFINITIONS:
        if pattern:
            non_cancer_rules.append(TargetRule(literal, category, pattern=pattern))
        else:
            non_cancer_rules.append(TargetRule(literal, category))

    return tuple(cancer_rules), tuple(non_cancer_rules)


# =============================================================================
# Pipeline Initialization
# =============================================================================

def initialize_medspacy_pipeline(
    cancer_rules: List[TargetRule],
    non_cancer_rules: List[TargetRule],
) -> "Language":
    """
    Initialize a MedSpaCy pipeline with target matcher and context detection.
    """
    import medspacy
    from medspacy.context import ConTextRule

    nlp = medspacy.load(enable=["medspacy_target_matcher", "medspacy_context"])

    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    target_matcher.add(cancer_rules + non_cancer_rules)

    context = nlp.get_pipe("medspacy_context")
    custom_negation_rules = [
        ConTextRule(literal, category, direction=direction)
        for literal, category, direction in CONTEXT_RULE_DEFINITIONS
    ]
    context.add(custom_negation_rules)

    return nlp


# =============================================================================
# Singleton Manager
# =============================================================================

class NLPPipelineManager:
    """
    Singleton manager for the MedSpaCy NLP pipeline.

    Usage:
        nlp = NLPPipelineManager.get_pipeline()
        NLPPipelineManager.add_rules([...])
        NLPPipelineManager.reset()
    """

    _instance: Optional["Language"] = None
    _additional_rules: List[TargetRule] = []

    @classmethod
    def get_pipeline(
        cls,
        additional_rules: Optional[List[TargetRule]] = None,
    ) -> "Language":
        """Get or create the singleton NLP pipeline."""
        if cls._instance is None:
            cancer_rules, non_cancer_rules = get_default_target_rules()
            cls._instance = initialize_medspacy_pipeline(cancer_rules, non_cancer_rules)

            if additional_rules:
                cls._additional_rules = additional_rules
                target_matcher = cls._instance.get_pipe("medspacy_target_matcher")
                target_matcher.add(additional_rules)

        return cls._instance

    @classmethod
    def add_rules(cls, rules: List[TargetRule]) -> None:
        """Add rules to the existing pipeline."""
        pipeline = cls.get_pipeline()
        target_matcher = pipeline.get_pipe("medspacy_target_matcher")
        target_matcher.add(rules)
        cls._additional_rules.extend(rules)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton pipeline for re-initialization."""
        cls._instance = None
        cls._additional_rules = []

    @classmethod
    def get_rule_count(cls) -> int:
        """Get the number of rules in the current pipeline."""
        if cls._instance is None:
            return 0
        target_matcher = cls._instance.get_pipe("medspacy_target_matcher")
        return len(target_matcher.rules)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_nlp(additional_rules: Optional[List[TargetRule]] = None) -> "Language":
    """
    Get the singleton MedSpaCy pipeline.
    This is the recommended entry point for getting the NLP pipeline.
    """
    return NLPPipelineManager.get_pipeline(additional_rules=additional_rules)


def reset_nlp() -> None:
    """Reset the singleton NLP pipeline."""
    NLPPipelineManager.reset()


# =============================================================================
# Disease Rule Generation
# =============================================================================

def generate_disease_rules(
    unique_diseases: List[Optional[str]],
    nlp: "Language",
    existing_rules,
) -> Tuple[List[TargetRule], List[str]]:
    """
    Auto-generate TargetRules from unique disease values in metadata.

    Returns:
        Tuple of (new_rules, skipped).
    """
    new_rules: List[TargetRule] = []
    skipped: List[str] = []

    existing_literals = {
        rule.literal.lower()
        for rule in existing_rules
        if hasattr(rule, "literal") and rule.literal
    }

    for disease in unique_diseases:
        if disease is None or not isinstance(disease, str):
            continue

        disease_clean = disease.strip().lower()

        if not disease_clean or disease_clean in ("nan", "none", "na", "n/a"):
            continue

        if disease_clean in existing_literals:
            skipped.append("{} (already exists)".format(disease))
            continue

        is_cancer_related = any(kw in disease_clean for kw in CANCER_KEYWORDS)

        if is_cancer_related:
            if any(kw in disease_clean for kw in SPECIFIC_CANCER_TYPES):
                label = "CANCER_TYPE"
            else:
                label = "CANCER"
            new_rules.append(TargetRule(disease.strip(), label))
        else:
            skipped.append("{} (not cancer-related)".format(disease))

    return new_rules, skipped
