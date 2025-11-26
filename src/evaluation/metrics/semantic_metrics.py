"""
Semantic similarity metrics for report evaluation.

This module provides semantic-based metrics:
- BERTScore: Contextual embedding similarity using BERT models
"""
from typing import List, Optional, Tuple
import warnings

from .base import MetricCalculator, MetricResult

# Try to import bert_score
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False


class BERTScoreMetric(MetricCalculator):
    """
    BERTScore metric for semantic similarity evaluation.
    
    Uses contextual embeddings from BERT models to compute token-level
    similarity between generated and reference texts. Returns precision,
    recall, and F1 scores.
    
    Args:
        model_type: BERT model to use for embeddings. Default is
            "microsoft/deberta-xlarge-mnli" for best performance,
            or "distilbert-base-uncased" for faster computation.
        lang: Language code. Default is "en".
        device: Device to run on ('cuda', 'cpu', or None for auto).
        batch_size: Batch size for processing. Default is 32.
        return_all: If True, returns P, R, F1 as separate metrics.
        
    Example:
        ```python
        bertscore = BERTScoreMetric(return_all=True)
        results = bertscore.compute_all(predictions, references)
        # Returns list of MetricResult for F1, Precision, Recall
        
        # Or just F1:
        bertscore_f1 = BERTScoreMetric(return_all=False)
        result = bertscore_f1.compute(predictions, references)
        ```
    """
    
    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        lang: str = "en",
        device: Optional[str] = None,
        batch_size: int = 32,
        return_all: bool = True,
    ):
        if not BERTSCORE_AVAILABLE:
            raise ImportError(
                "bert_score is required for BERTScore computation. "
                "Install with: pip install bert-score"
            )
        
        self.model_type = model_type
        self.lang = lang
        self.device = device
        self.batch_size = batch_size
        self.return_all = return_all
    
    @property
    def name(self) -> str:
        return "BERTScore"
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute BERTScore F1 for predictions vs references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult with BERTScore F1 scores
        """
        results = self.compute_all(predictions, references)
        # Return F1 by default
        for r in results:
            if "F1" in r.name:
                return r
        return results[0]
    
    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
    ) -> List[MetricResult]:
        """
        Compute all BERTScore variants (Precision, Recall, F1).
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            List of MetricResult for [F1, Precision, Recall]
        """
        if not predictions or not references:
            return [
                MetricResult(name="BERTScore F1", mean=0.0, std=0.0),
                MetricResult(name="BERTScore Precision", mean=0.0, std=0.0),
                MetricResult(name="BERTScore Recall", mean=0.0, std=0.0),
            ]
        
        # Replace empty strings with placeholder to avoid errors
        predictions = [p if p.strip() else "." for p in predictions]
        references = [r if r.strip() else "." for r in references]
        
        try:
            P, R, F1 = bert_score_fn(
                predictions,
                references,
                model_type=self.model_type,
                lang=self.lang,
                device=self.device,
                batch_size=self.batch_size,
                verbose=False,
            )
            
            # Convert tensors to lists
            p_scores = P.tolist()
            r_scores = R.tolist()
            f1_scores = F1.tolist()
            
            metadata = {
                "model_type": self.model_type,
                "lang": self.lang,
            }
            
            return [
                MetricResult.from_scores("BERTScore F1", f1_scores, metadata),
                MetricResult.from_scores("BERTScore Precision", p_scores, metadata),
                MetricResult.from_scores("BERTScore Recall", r_scores, metadata),
            ]
            
        except Exception as e:
            warnings.warn(f"BERTScore computation failed: {e}")
            return [
                MetricResult(
                    name="BERTScore F1", mean=0.0, std=0.0,
                    metadata={"error": str(e)}
                ),
                MetricResult(
                    name="BERTScore Precision", mean=0.0, std=0.0,
                    metadata={"error": str(e)}
                ),
                MetricResult(
                    name="BERTScore Recall", mean=0.0, std=0.0,
                    metadata={"error": str(e)}
                ),
            ]
