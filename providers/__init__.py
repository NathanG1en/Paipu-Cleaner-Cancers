"""
Fallback classification providers.

Available providers:
- NCBIProvider: Fetches BioProject/BioSample metadata via NCBI E-utilities
- GeminiProvider: Google Gemini LLM classification
- OllamaProvider: Local LLM classification via Ollama
"""

from providers.ncbi import NCBIProvider
from providers.llm import GeminiProvider, OllamaProvider

__all__ = ["NCBIProvider", "GeminiProvider", "OllamaProvider"]
