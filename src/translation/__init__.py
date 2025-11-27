"""
Translation module for MAIRA-2.

Provides language translation capabilities using Meta's NLLB model.
Supports bidirectional translation between English and Spanish.
"""

from .nllb_translator import NLLBTranslator, TranslationConfig

__all__ = [
    "NLLBTranslator",
    "TranslationConfig",
]
