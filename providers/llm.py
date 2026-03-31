"""
LLM-based fallback providers for cancer classification.

Provides pluggable LLM providers that classify samples by sending
all available metadata to an LLM and parsing the response.

Supported:
- GeminiProvider: Google Gemini API
- OllamaProvider: Local models via Ollama REST API
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from fallback import FallbackProvider, FallbackResult
from config import ClassificationLabel as CL


_SYSTEM_PROMPT = """You are a bioinformatics expert classifying RNA-seq samples as cancer or non-cancer.

Given the metadata for a sequencing sample, determine if this sample comes from a cancer/tumor study.

Respond with ONLY a JSON object:
{
  "is_cancer": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation"
}

Guidelines:
- Cell lines derived from tumors ARE cancer samples
- Matched normal/control samples from cancer studies are NOT cancer
- If sample has "control", "normal", "healthy", "sham" in sample-level fields, likely NOT cancer
- Study-level titles mentioning cancer don't mean every sample is cancer
- If there's truly no information, respond with confidence 0.0"""


def _build_prompt(sample: Dict[str, Any]) -> str:
    """Build a classification prompt from sample metadata."""
    # Filter to meaningful fields only
    skip = {"_resolved", "regex_label", "med_label", "med_reason",
            "med_source_columns", "confidence_category", "resolved_by",
            "fallback_reason", "predicted_cancer", "actual_cancer",
            "is_cancer", "cancer_type", "final_classification"}

    metadata = {}
    for k, v in sample.items():
        if k in skip or k.endswith("_norm"):
            continue
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "nan", "none", "null")):
            continue
        metadata[k] = str(v)[:200]  # Truncate long values

    prompt = "Classify this RNA-seq sample as cancer or non-cancer:\n\n"
    for k, v in metadata.items():
        prompt += f"  {k}: {v}\n"

    return prompt


def _parse_llm_response(text: str, provider_name: str) -> FallbackResult:
    """Parse LLM JSON response into FallbackResult."""
    try:
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if not json_match:
            return FallbackResult(
                label=CL.CONFIRMED_NON_CANCER.value,
                confidence=0.0,
                reason=f"llm_parse_error: no JSON in response",
                provider_name=provider_name,
            )

        data = json.loads(json_match.group())
        is_cancer = data.get("is_cancer", False)
        confidence = float(data.get("confidence", 0.0))
        reason = data.get("reason", "no_reason_given")

        if is_cancer and confidence >= 0.5:
            label = CL.LIKELY_CANCER.value
        else:
            label = CL.CONFIRMED_NON_CANCER.value
            confidence = max(0.0, 1.0 - confidence)  # Invert for non-cancer confidence

        return FallbackResult(
            label=label,
            confidence=confidence,
            reason=f"llm: {reason}",
            provider_name=provider_name,
        )

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        return FallbackResult(
            label=CL.CONFIRMED_NON_CANCER.value,
            confidence=0.0,
            reason=f"llm_parse_error: {e}",
            provider_name=provider_name,
        )


# =============================================================================
# Gemini Provider
# =============================================================================

class GeminiProvider(FallbackProvider):
    """
    Google Gemini API provider.

    Usage:
        provider = GeminiProvider(api_key="your-key")
        # or set GEMINI_API_KEY env var
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
    ):
        import os
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self._model = model

    @property
    def name(self) -> str:
        return "gemini_llm"

    def classify(self, sample: Dict[str, Any]) -> FallbackResult:
        """Classify a sample using Gemini with retry for rate limits."""
        import time
        import urllib.request
        import urllib.error

        prompt = _build_prompt(sample)

        payload = json.dumps({
            "contents": [{
                "parts": [{"text": _SYSTEM_PROMPT + "\n\n" + prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 200,
            }
        }).encode("utf-8")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )

        max_retries = 3
        for attempt in range(max_retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode("utf-8"))

                text = result["candidates"][0]["content"]["parts"][0]["text"]
                # Rate limit: pause between successful calls
                time.sleep(1.0)
                return _parse_llm_response(text, self.name)

            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries:
                    wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    print(f"  Gemini rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return FallbackResult(
                    label=CL.CONFIRMED_NON_CANCER.value,
                    confidence=0.0,
                    reason=f"gemini_error: HTTP {e.code}",
                    provider_name=self.name,
                )
            except Exception as e:
                return FallbackResult(
                    label=CL.CONFIRMED_NON_CANCER.value,
                    confidence=0.0,
                    reason=f"gemini_error: {e}",
                    provider_name=self.name,
                )

        return FallbackResult(
            label=CL.CONFIRMED_NON_CANCER.value,
            confidence=0.0,
            reason="gemini_error: max retries exceeded",
            provider_name=self.name,
        )


# =============================================================================
# Ollama Provider (Local LLM)
# =============================================================================

class OllamaProvider(FallbackProvider):
    """
    Ollama local LLM provider.

    Usage:
        provider = OllamaProvider(model="llama3.2")
        # Requires Ollama running locally: ollama serve
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama_llm"

    def classify(self, sample: Dict[str, Any]) -> FallbackResult:
        """Classify a sample using a local Ollama model."""
        import urllib.request

        prompt = _build_prompt(sample)

        payload = json.dumps({
            "model": self._model,
            "prompt": _SYSTEM_PROMPT + "\n\n" + prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }).encode("utf-8")

        url = f"{self._base_url}/api/generate"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            text = result.get("response", "")
            return _parse_llm_response(text, self.name)

        except Exception as e:
            return FallbackResult(
                label=CL.CONFIRMED_NON_CANCER.value,
                confidence=0.0,
                reason=f"ollama_error: {e}",
                provider_name=self.name,
            )
