"""
NLLB-200 Translation Module

This module provides translation capabilities using Meta's NLLB-200-Distilled-600M model.
It is designed to be agnostic to the use case while providing optimized support for
medical/radiology text translation.

Usage:
    from translation import NLLBTranslator, TranslationConfig
    
    # Basic usage
    translator = NLLBTranslator()
    spanish_text = translator.translate("The lungs are clear.", "en", "es")
    
    # Batch translation
    english_texts = ["Heart is normal.", "No pneumonia."]
    spanish_texts = translator.translate_batch(english_texts, "en", "es")
    
    # Custom configuration
    config = TranslationConfig(
        model_name="facebook/nllb-200-distilled-600M",
        max_length=512,
        batch_size=16,
    )
    translator = NLLBTranslator(config)

Reference:
    NLLB Team et al., "No Language Left Behind: Scaling Human-Centered Machine Translation"
    https://huggingface.co/facebook/nllb-200-distilled-600M
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
import warnings

import torch

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# NLLB-200 language codes
# See: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANGUAGE_CODES = {
    # Common language codes
    "en": "eng_Latn",      # English
    "es": "spa_Latn",      # Spanish
    "pt": "por_Latn",      # Portuguese
    "fr": "fra_Latn",      # French
    "de": "deu_Latn",      # German
    "it": "ita_Latn",      # Italian
    "zh": "zho_Hans",      # Chinese (Simplified)
    "ja": "jpn_Jpan",      # Japanese
    "ko": "kor_Hang",      # Korean
    "ar": "arb_Arab",      # Arabic
    "ru": "rus_Cyrl",      # Russian
    "hi": "hin_Deva",      # Hindi
    # Full NLLB codes can also be used directly
}


@dataclass
class TranslationConfig:
    """
    Configuration for the NLLB translator.
    
    Attributes:
        model_name: HuggingFace model identifier
        max_length: Maximum sequence length for translation
        batch_size: Batch size for batch translation
        device: Device for inference ('cuda', 'cpu', or 'auto')
        torch_dtype: Torch dtype for model weights
        device_map: Device map for model distribution ('auto' for multi-GPU)
        num_beams: Number of beams for beam search
        early_stopping: Whether to stop when num_beams sentences are finished
    """
    model_name: str = "facebook/nllb-200-distilled-600M"
    max_length: int = 512
    batch_size: int = 16
    device: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    device_map: Optional[str] = None
    num_beams: int = 5
    early_stopping: bool = True


class NLLBTranslator:
    """
    NLLB-200 based translator supporting 200+ languages.
    
    This translator is optimized for efficiency and supports:
    - Single text translation
    - Batch translation for multiple texts
    - Bidirectional translation between any supported language pair
    - Memory-efficient loading with optional multi-GPU support
    
    The module is designed to be use-case agnostic but works well with
    medical/radiology texts which tend to have specialized vocabulary.
    
    Example:
        ```python
        translator = NLLBTranslator()
        
        # English to Spanish
        es_text = translator.translate("Normal heart size.", "en", "es")
        
        # Spanish to English  
        en_text = translator.translate("Corazón de tamaño normal.", "es", "en")
        
        # Batch translation
        texts = ["Clear lungs.", "No effusion."]
        translated = translator.translate_batch(texts, "en", "es")
        ```
    """
    
    def __init__(self, config: Optional[TranslationConfig] = None):
        """
        Initialize the NLLB translator.
        
        Args:
            config: Translation configuration. Uses defaults if not provided.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required for translation. "
                "Install with: pip install transformers"
            )
        
        self.config = config or TranslationConfig()
        self._model = None
        self._tokenizer = None
        self._loaded = False
    
    def _ensure_loaded(self):
        """Lazy load the model and tokenizer."""
        if self._loaded:
            return
        
        print(f"Loading NLLB translation model: {self.config.model_name}")
        
        # Determine device configuration
        if self.config.device_map:
            # Use device_map for multi-GPU distribution
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                device_map=self.config.device_map,
                torch_dtype=self.config.torch_dtype or torch.float16,
            )
            self._device = None  # device_map handles placement
        else:
            # Determine device
            if self.config.device:
                device = self.config.device
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            
            self._device = torch.device(device)
            
            # Load model
            dtype = self.config.torch_dtype
            if dtype is None and device == "cuda":
                dtype = torch.float16  # Use FP16 on GPU for efficiency
            
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
            )
            self._model = self._model.to(self._device)
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )
        
        self._model.eval()
        self._loaded = True
        print(f"NLLB model loaded successfully")
    
    def _get_nllb_code(self, lang: str) -> str:
        """
        Convert a language code to NLLB format.
        
        Args:
            lang: Short language code (e.g., 'en') or full NLLB code
            
        Returns:
            NLLB language code (e.g., 'eng_Latn')
        """
        # If already in NLLB format, return as-is
        if "_" in lang and len(lang) == 8:
            return lang
        
        # Look up in our mapping
        if lang.lower() in NLLB_LANGUAGE_CODES:
            return NLLB_LANGUAGE_CODES[lang.lower()]
        
        # Try common variations
        lang_lower = lang.lower()
        for key, value in NLLB_LANGUAGE_CODES.items():
            if lang_lower.startswith(key) or value.lower().startswith(lang_lower):
                return value
        
        raise ValueError(
            f"Unknown language code: {lang}. "
            f"Supported codes: {list(NLLB_LANGUAGE_CODES.keys())}"
        )
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> str:
        """
        Translate a single text from source to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en' or 'eng_Latn')
            target_lang: Target language code (e.g., 'es' or 'spa_Latn')
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Translated text
        """
        if not text or not text.strip():
            return ""
        
        results = self.translate_batch([text], source_lang, target_lang, **kwargs)
        return results[0] if results else ""
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        show_progress: bool = False,
        **kwargs
    ) -> List[str]:
        """
        Translate a batch of texts from source to target language.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            show_progress: Whether to show a progress bar
            **kwargs: Additional arguments passed to the model
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        self._ensure_loaded()
        
        # Get NLLB language codes
        src_code = self._get_nllb_code(source_lang)
        tgt_code = self._get_nllb_code(target_lang)
        
        # Set source language for tokenizer
        self._tokenizer.src_lang = src_code
        
        # Get forced BOS token ID for target language
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Process in batches
        all_translations = []
        batch_size = self.config.batch_size
        
        # Setup progress iterator
        if show_progress:
            try:
                from tqdm import tqdm
                batches = list(range(0, len(texts), batch_size))
                batch_iterator = tqdm(batches, desc="Translating")
            except ImportError:
                batch_iterator = range(0, len(texts), batch_size)
        else:
            batch_iterator = range(0, len(texts), batch_size)
        
        for start_idx in batch_iterator:
            batch_texts = texts[start_idx:start_idx + batch_size]
            
            # Handle empty texts
            non_empty_indices = [i for i, t in enumerate(batch_texts) if t and t.strip()]
            non_empty_texts = [batch_texts[i] for i in non_empty_indices]
            
            if not non_empty_texts:
                all_translations.extend([""] * len(batch_texts))
                continue
            
            # Tokenize
            inputs = self._tokenizer(
                non_empty_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            )
            
            # Move to device if not using device_map
            if self._device is not None:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=self.config.max_length,
                    num_beams=kwargs.get("num_beams", self.config.num_beams),
                    early_stopping=kwargs.get("early_stopping", self.config.early_stopping),
                )
            
            # Decode translations
            translations = self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )
            
            # Reconstruct full batch with empty strings
            batch_translations = [""] * len(batch_texts)
            for idx, trans in zip(non_empty_indices, translations):
                batch_translations[idx] = trans
            
            all_translations.extend(batch_translations)
        
        return all_translations
    
    def translate_en_to_es(self, text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Convenience method to translate from English to Spanish.
        
        Args:
            text: Text or list of texts to translate
            **kwargs: Additional arguments
            
        Returns:
            Translated text(s)
        """
        if isinstance(text, list):
            return self.translate_batch(text, "en", "es", **kwargs)
        return self.translate(text, "en", "es", **kwargs)
    
    def translate_es_to_en(self, text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Convenience method to translate from Spanish to English.
        
        Args:
            text: Text or list of texts to translate
            **kwargs: Additional arguments
            
        Returns:
            Translated text(s)
        """
        if isinstance(text, list):
            return self.translate_batch(text, "es", "en", **kwargs)
        return self.translate(text, "es", "en", **kwargs)
    
    @property
    def supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported language codes."""
        return NLLB_LANGUAGE_CODES.copy()
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded
    
    def unload(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
