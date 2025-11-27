"""
Dataset Translation Module

This module provides utilities for translating evaluation datasets,
specifically handling the translation of radiology reports between
English and Spanish for cross-lingual evaluation.

Responsibilities:
- Translate reference reports from Spanish to English (for CheXbert metrics)
- Translate model predictions from English to Spanish (for output comparison)
- Cache translated data to avoid redundant translations
"""
import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
import json
import hashlib
import warnings

from tqdm import tqdm


@dataclass
class TranslatedDataset:
    """
    Container for translated evaluation data.
    
    Holds both original and translated versions of predictions and references.
    
    Attributes:
        predictions_en: English predictions (from model)
        predictions_es: Spanish predictions (translated)
        references_es: Spanish references (original)
        references_en: English references (translated for CheXbert)
        instance_ids: Sample identifiers
    """
    predictions_en: List[str] = field(default_factory=list)
    predictions_es: List[str] = field(default_factory=list)
    references_es: List[str] = field(default_factory=list)
    references_en: List[str] = field(default_factory=list)
    instance_ids: List[str] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.predictions_en)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "predictions_en": self.predictions_en,
            "predictions_es": self.predictions_es,
            "references_es": self.references_es,
            "references_en": self.references_en,
            "instance_ids": self.instance_ids,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranslatedDataset":
        """Create from dictionary."""
        return cls(
            predictions_en=data.get("predictions_en", []),
            predictions_es=data.get("predictions_es", []),
            references_es=data.get("references_es", []),
            references_en=data.get("references_en", []),
            instance_ids=data.get("instance_ids", []),
        )


class DatasetTranslator:
    """
    Handles translation of evaluation datasets for cross-lingual evaluation.
    
    This class manages the translation workflow for evaluation:
    1. Translates model predictions (EN → ES) for comparison with Spanish references
    2. Translates Spanish references (ES → EN) for English-only metrics (CheXbert)
    
    Includes caching to avoid redundant translations.
    
    Example:
        ```python
        from evaluation.dataset_translation import DatasetTranslator
        from translation import NLLBTranslator
        
        # Create translator
        translator = NLLBTranslator()
        dataset_translator = DatasetTranslator(translator)
        
        # Translate evaluation data
        translated = dataset_translator.translate_evaluation_data(
            predictions_en=["Normal heart.", "Clear lungs."],
            references_es=["Corazón normal.", "Pulmones claros."],
            instance_ids=["001", "002"],
        )
        
        # Access translated data
        print(translated.predictions_es)  # Spanish predictions
        print(translated.references_en)   # English references for CheXbert
        ```
    """
    
    def __init__(
        self,
        translator: Optional["NLLBTranslator"] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize the dataset translator.
        
        Args:
            translator: NLLBTranslator instance. Will create one if not provided.
            cache_dir: Directory for caching translations
            use_cache: Whether to use caching
        """
        self._translator = translator
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        if self.cache_dir and self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_translator(self):
        """Get or create the translator instance."""
        if self._translator is None:
            # Add src directory to path if needed for import
            src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            from translation import NLLBTranslator
            self._translator = NLLBTranslator()
        return self._translator
    
    def _compute_cache_key(self, texts: List[str], direction: str) -> str:
        """Compute a cache key for a list of texts."""
        content = f"{direction}:" + "\n".join(texts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _load_from_cache(self, cache_key: str) -> Optional[List[str]]:
        """Load cached translations if available."""
        if not self.use_cache or not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get("translations", [])
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, translations: List[str]):
        """Save translations to cache."""
        if not self.use_cache or not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"translations": translations}, f, ensure_ascii=False)
        except Exception as e:
            warnings.warn(f"Failed to save to cache: {e}")
    
    def translate_texts(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        show_progress: bool = True,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Translate a list of texts with optional caching.
        
        Args:
            texts: Texts to translate
            source_lang: Source language code
            target_lang: Target language code
            show_progress: Whether to show progress bar
            batch_size: Override batch size for translation
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        direction = f"{source_lang}_to_{target_lang}"
        
        # Check cache
        cache_key = self._compute_cache_key(texts, direction)
        cached = self._load_from_cache(cache_key)
        if cached is not None and len(cached) == len(texts):
            print(f"Loaded {len(cached)} translations from cache")
            return cached
        
        # Translate
        translator = self._get_translator()
        
        if batch_size:
            original_batch_size = translator.config.batch_size
            translator.config.batch_size = batch_size
        
        try:
            translations = translator.translate_batch(
                texts,
                source_lang,
                target_lang,
                show_progress=show_progress,
            )
        finally:
            if batch_size:
                translator.config.batch_size = original_batch_size
        
        # Save to cache
        self._save_to_cache(cache_key, translations)
        
        return translations
    
    def translate_evaluation_data(
        self,
        predictions_en: List[str],
        references_es: List[str],
        instance_ids: Optional[List[str]] = None,
        translate_predictions: bool = True,
        translate_references: bool = True,
        show_progress: bool = True,
    ) -> TranslatedDataset:
        """
        Translate evaluation data for cross-lingual evaluation.
        
        Performs two translations:
        1. predictions_en → predictions_es (for Spanish output comparison)
        2. references_es → references_en (for CheXbert metrics)
        
        Args:
            predictions_en: English predictions from the model
            references_es: Spanish reference reports from the dataset
            instance_ids: Sample identifiers
            translate_predictions: Whether to translate predictions to Spanish
            translate_references: Whether to translate references to English
            show_progress: Whether to show progress bars
            
        Returns:
            TranslatedDataset with all versions
        """
        if len(predictions_en) != len(references_es):
            raise ValueError(
                f"Number of predictions ({len(predictions_en)}) must match "
                f"number of references ({len(references_es)})"
            )
        
        result = TranslatedDataset(
            predictions_en=predictions_en,
            references_es=references_es,
            instance_ids=instance_ids or [str(i) for i in range(len(predictions_en))],
        )
        
        # Translate predictions EN → ES
        if translate_predictions:
            print("\nTranslating predictions (EN → ES)...")
            result.predictions_es = self.translate_texts(
                predictions_en,
                source_lang="en",
                target_lang="es",
                show_progress=show_progress,
            )
        else:
            result.predictions_es = [""] * len(predictions_en)
        
        # Translate references ES → EN
        if translate_references:
            print("\nTranslating references (ES → EN)...")
            result.references_en = self.translate_texts(
                references_es,
                source_lang="es",
                target_lang="en",
                show_progress=show_progress,
            )
        else:
            result.references_en = [""] * len(references_es)
        
        return result
    
    def save_translated_dataset(
        self,
        translated_data: TranslatedDataset,
        output_path: str,
    ):
        """
        Save translated dataset to a JSON file.
        
        Args:
            translated_data: TranslatedDataset to save
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"Saved translated dataset to: {output_path}")
    
    def load_translated_dataset(self, input_path: str) -> TranslatedDataset:
        """
        Load translated dataset from a JSON file.
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            TranslatedDataset loaded from file
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return TranslatedDataset.from_dict(data)


def translate_references_for_chexbert(
    references_es: List[str],
    translator=None,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Convenience function to translate Spanish references to English for CheXbert.
    
    Args:
        references_es: Spanish reference reports
        translator: Optional NLLBTranslator instance
        cache_dir: Optional cache directory
        
    Returns:
        English translations of the references
    """
    dataset_translator = DatasetTranslator(
        translator=translator,
        cache_dir=cache_dir,
    )
    return dataset_translator.translate_texts(
        references_es,
        source_lang="es",
        target_lang="en",
    )


def translate_predictions_to_spanish(
    predictions_en: List[str],
    translator=None,
    cache_dir: Optional[str] = None,
) -> List[str]:
    """
    Convenience function to translate English predictions to Spanish.
    
    Args:
        predictions_en: English predictions from the model
        translator: Optional NLLBTranslator instance
        cache_dir: Optional cache directory
        
    Returns:
        Spanish translations of the predictions
    """
    dataset_translator = DatasetTranslator(
        translator=translator,
        cache_dir=cache_dir,
    )
    return dataset_translator.translate_texts(
        predictions_en,
        source_lang="en",
        target_lang="es",
    )
