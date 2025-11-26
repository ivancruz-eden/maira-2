"""
Natural Language Generation (NLG) metrics for report evaluation.

This module provides standard NLG metrics:
- BLEU (Bilingual Evaluation Understudy)
- METEOR (Metric for Evaluation of Translation with Explicit ORdering)
- ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
"""
from typing import List, Optional
import warnings

from .base import MetricCalculator, MetricResult

# Try to import nltk for BLEU and METEOR
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Try to import rouge_score for ROUGE-L
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def _ensure_nltk_data():
    """Download required NLTK data if not available."""
    if not NLTK_AVAILABLE:
        return
    
    required_packages = ['punkt', 'wordnet', 'punkt_tab']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass
        try:
            nltk.data.find(f'corpora/{package}')
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except:
                pass


class BLEUMetric(MetricCalculator):
    """
    BLEU (Bilingual Evaluation Understudy) metric.
    
    Measures n-gram precision between generated and reference text.
    Commonly used for machine translation and text generation evaluation.
    
    Args:
        n: Maximum n-gram order (1-4). Default is 4 for BLEU-4.
        smoothing: Whether to use smoothing for zero counts. Default True.
        
    Example:
        ```python
        bleu2 = BLEUMetric(n=2)
        bleu4 = BLEUMetric(n=4)
        
        result = bleu4.compute(predictions, references)
        print(f"BLEU-4: {result.mean:.4f}")
        ```
    """
    
    def __init__(self, n: int = 4, smoothing: bool = True):
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for BLEU computation. "
                "Install with: pip install nltk"
            )
        
        _ensure_nltk_data()
        
        self.n = n
        self.smoothing = smoothing
        self._smoothing_fn = SmoothingFunction().method1 if smoothing else None
        
        # Set weights based on n-gram order
        if n == 1:
            self._weights = (1.0, 0, 0, 0)
        elif n == 2:
            self._weights = (0.5, 0.5, 0, 0)
        elif n == 3:
            self._weights = (0.33, 0.33, 0.34, 0)
        else:  # n >= 4
            self._weights = (0.25, 0.25, 0.25, 0.25)
    
    @property
    def name(self) -> str:
        return f"BLEU-{self.n}"
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except Exception:
            # Fallback to simple tokenization
            return text.lower().split()
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute BLEU score for predictions vs references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult with BLEU scores
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
            
            try:
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=self._weights,
                    smoothing_function=self._smoothing_fn,
                )
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        
        return MetricResult.from_scores(
            name=self.name,
            scores=scores,
            metadata={"n": self.n, "smoothing": self.smoothing}
        )


class METEORMetric(MetricCalculator):
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) metric.
    
    Considers synonyms, stemming, and word order in addition to exact matches.
    Generally correlates better with human judgment than BLEU.
    
    Example:
        ```python
        meteor = METEORMetric()
        result = meteor.compute(predictions, references)
        print(f"METEOR: {result.mean:.4f}")
        ```
    """
    
    def __init__(self):
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for METEOR computation. "
                "Install with: pip install nltk"
            )
        
        _ensure_nltk_data()
    
    @property
    def name(self) -> str:
        return "METEOR"
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except Exception:
            return text.lower().split()
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute METEOR score for predictions vs references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult with METEOR scores
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)
            
            if not pred_tokens or not ref_tokens:
                scores.append(0.0)
                continue
            
            try:
                # METEOR expects tokenized strings
                score = meteor_score([ref_tokens], pred_tokens)
                scores.append(float(score))
            except Exception as e:
                warnings.warn(f"METEOR computation failed: {e}")
                scores.append(0.0)
        
        return MetricResult.from_scores(
            name=self.name,
            scores=scores,
        )


class ROUGELMetric(MetricCalculator):
    """
    ROUGE-L (Longest Common Subsequence) metric.
    
    Measures the longest common subsequence between generated and reference text.
    Good for evaluating fluency and content coverage.
    
    Args:
        use_stemmer: Whether to use Porter stemmer for token matching.
        
    Example:
        ```python
        rouge = ROUGELMetric()
        result = rouge.compute(predictions, references)
        print(f"ROUGE-L: {result.mean:.4f}")
        ```
    """
    
    def __init__(self, use_stemmer: bool = True):
        if not ROUGE_AVAILABLE:
            raise ImportError(
                "rouge_score is required for ROUGE-L computation. "
                "Install with: pip install rouge-score"
            )
        
        self.use_stemmer = use_stemmer
        self._scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)
    
    @property
    def name(self) -> str:
        return "ROUGE-L"
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute ROUGE-L score for predictions vs references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult with ROUGE-L F1 scores
        """
        scores = []
        
        for pred, ref in zip(predictions, references):
            if not pred.strip() or not ref.strip():
                scores.append(0.0)
                continue
            
            try:
                result = self._scorer.score(ref, pred)
                # Use F1 score (fmeasure)
                scores.append(float(result['rougeL'].fmeasure))
            except Exception as e:
                warnings.warn(f"ROUGE-L computation failed: {e}")
                scores.append(0.0)
        
        return MetricResult.from_scores(
            name=self.name,
            scores=scores,
            metadata={"use_stemmer": self.use_stemmer}
        )
