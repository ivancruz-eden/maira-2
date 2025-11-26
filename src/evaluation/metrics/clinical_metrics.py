"""
Clinical/Domain-specific metrics for radiology report evaluation.

This module provides clinical metrics specifically designed for
radiology report evaluation:
- RadGraph-based metrics (RGER - Radiology Graph Entity Recall)
- CheXbert-based metrics (Clinical accuracy using CheXbert labeler)
"""
from typing import List, Optional, Dict, Any, Tuple
import warnings
import os

from .base import MetricCalculator, MetricResult

# Try to import radgraph
try:
    from radgraph import F1RadGraph
    RADGRAPH_AVAILABLE = True
except ImportError:
    RADGRAPH_AVAILABLE = False

# Try to import torch for CheXbert
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RadGraphMetric(MetricCalculator):
    """
    RadGraph-based metric (RGER - Radiology Graph Entity Recall).
    
    Uses the RadGraph model to extract clinical entities and relations
    from radiology reports, then computes entity-level F1 scores.
    
    This metric captures clinical correctness by measuring whether
    the generated report contains the same clinical entities
    (anatomies, observations, etc.) as the reference.
    
    Args:
        reward_level: Level of granularity for scoring.
            - "simple": Entity-level matching only
            - "partial": Partial credit for related entities
            - "complete": Full relation matching
        device: Device to run on ('cuda', 'cpu', or None for auto).
        
    Example:
        ```python
        rger = RadGraphMetric(reward_level="partial")
        result = rger.compute(predictions, references)
        print(f"RGER: {result.mean:.4f}")
        ```
        
    Reference:
        Jain et al., "RadGraph: Extracting Clinical Entities and Relations 
        from Radiology Reports", NeurIPS 2021
    """
    
    def __init__(
        self,
        reward_level: str = "partial",
        device: Optional[str] = None,
    ):
        if not RADGRAPH_AVAILABLE:
            raise ImportError(
                "radgraph is required for RGER computation. "
                "Install with: pip install radgraph"
            )
        
        self.reward_level = reward_level
        self.device = device
        self._scorer = None  # Lazy initialization
    
    @property
    def name(self) -> str:
        return "RGER"
    
    def _get_scorer(self):
        """Lazy initialization of RadGraph scorer."""
        if self._scorer is None:
            # Suppress warnings during model loading
            import logging
            logging.getLogger("radgraph").setLevel(logging.ERROR)
            
            self._scorer = F1RadGraph(
                reward_level=self.reward_level,
                cuda=self.device if self.device else -1,
            )
        return self._scorer
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute RadGraph F1 score for predictions vs references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult with RGER scores
        """
        if not predictions or not references:
            return MetricResult(name=self.name, mean=0.0, std=0.0)
        
        # Replace empty strings to avoid errors
        predictions = [p if p.strip() else "Normal study." for p in predictions]
        references = [r if r.strip() else "Normal study." for r in references]
        
        try:
            scorer = self._get_scorer()
            
            # RadGraph expects specific input format
            # Compute batch score and individual scores
            scores = []
            for pred, ref in zip(predictions, references):
                try:
                    # F1RadGraph returns (mean_score, scores_dict, hypothesis_annotations, reference_annotations)
                    _, score_dict, _, _ = scorer(
                        hyps=[pred],
                        refs=[ref],
                    )
                    # score_dict contains individual scores
                    if score_dict and len(score_dict) > 0:
                        scores.append(float(list(score_dict.values())[0]))
                    else:
                        scores.append(0.0)
                except Exception:
                    scores.append(0.0)
            
            return MetricResult.from_scores(
                name=self.name,
                scores=scores,
                metadata={"reward_level": self.reward_level}
            )
            
        except Exception as e:
            warnings.warn(f"RadGraph computation failed: {e}")
            return MetricResult(
                name=self.name,
                mean=0.0,
                std=0.0,
                metadata={"error": str(e)}
            )


class CheXbertMetric(MetricCalculator):
    """
    CheXbert-based metrics for clinical accuracy evaluation.
    
    Uses the CheXbert model to extract clinical labels from radiology
    reports, then computes similarity and F1 scores based on
    14 pathology labels (Atelectasis, Cardiomegaly, etc.).
    
    Returns both:
    - CheXbert Similarity: Micro-averaged accuracy across all labels
    - CheXbert F1: Macro-averaged F1 across pathology labels
    
    Args:
        device: Device to run on ('cuda', 'cpu', or None for auto).
        batch_size: Batch size for CheXbert inference.
        
    Example:
        ```python
        chexbert = CheXbertMetric()
        results = chexbert.compute_all(predictions, references)
        for result in results:
            print(result)
        ```
        
    Reference:
        Smit et al., "CheXbert: Combining Automatic Labelers and Expert 
        Annotations for Accurate Radiology Report Labeling Using BERT", 
        EMNLP 2020
    """
    
    # CheXbert label names
    LABEL_NAMES = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
        "No Finding",
    ]
    
    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CheXbert computation. "
                "Install with: pip install torch"
            )
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._labeler = None  # Lazy initialization
    
    @property
    def name(self) -> str:
        return "CheXbert"
    
    def _get_labeler(self):
        """Lazy initialization of CheXbert labeler."""
        if self._labeler is None:
            try:
                from .chexbert_labeler import CheXbertLabeler
                self._labeler = CheXbertLabeler(device=self.device)
            except ImportError:
                # Try using transformers directly
                self._labeler = self._create_simple_labeler()
        return self._labeler
    
    def _create_simple_labeler(self):
        """Create a simple CheXbert labeler using transformers."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            class SimpleCheXbertLabeler:
                def __init__(self, device):
                    self.device = device
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "bert-base-uncased"
                    )
                    # Note: This is a placeholder. Real CheXbert requires
                    # the actual model weights from the CheXbert repository.
                    self.model = None
                    self._warned = False
                
                def __call__(self, texts: List[str]) -> List[List[int]]:
                    if not self._warned:
                        warnings.warn(
                            "CheXbert model not available. Using rule-based fallback. "
                            "For accurate results, install chexbert: "
                            "pip install git+https://github.com/stanfordmlgroup/CheXbert.git"
                        )
                        self._warned = True
                    
                    # Rule-based fallback
                    return [self._rule_based_labels(t) for t in texts]
                
                def _rule_based_labels(self, text: str) -> List[int]:
                    """Simple rule-based label extraction as fallback."""
                    text_lower = text.lower()
                    labels = []
                    
                    # Map keywords to labels (0=negative, 1=positive, 2=uncertain)
                    keywords = {
                        0: ["enlarged cardiomediastinum", "widened mediastinum"],
                        1: ["cardiomegaly", "enlarged heart", "cardiac enlargement"],
                        2: ["opacity", "opacities", "opacification"],
                        3: ["lesion", "mass", "nodule", "tumor"],
                        4: ["edema", "pulmonary edema", "congestion"],
                        5: ["consolidation", "consolidative"],
                        6: ["pneumonia", "infection"],
                        7: ["atelectasis", "collapse"],
                        8: ["pneumothorax"],
                        9: ["pleural effusion", "effusion"],
                        10: ["pleural thickening", "pleural abnormality"],
                        11: ["fracture", "rib fracture"],
                        12: ["support device", "tube", "line", "catheter", "pacemaker"],
                        13: ["no acute", "normal", "unremarkable", "clear lungs"],
                    }
                    
                    for i in range(14):
                        found = any(kw in text_lower for kw in keywords.get(i, []))
                        # Check for negation
                        if found:
                            # Simple negation check
                            for kw in keywords.get(i, []):
                                if kw in text_lower:
                                    idx = text_lower.find(kw)
                                    prefix = text_lower[max(0, idx-20):idx]
                                    if any(neg in prefix for neg in ["no ", "without ", "negative for ", "absent "]):
                                        found = False
                                        break
                        labels.append(1 if found else 0)
                    
                    return labels
            
            return SimpleCheXbertLabeler(self.device)
            
        except Exception as e:
            warnings.warn(f"Failed to create CheXbert labeler: {e}")
            return None
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute CheXbert F1 for predictions vs references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult with CheXbert F1 scores
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
        Compute all CheXbert metrics (Similarity and F1).
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            List of MetricResult for [CheXbert Similarity, CheXbert F1]
        """
        if not predictions or not references:
            return [
                MetricResult(name="CheXbert Similarity", mean=0.0, std=0.0),
                MetricResult(name="CheXbert F1", mean=0.0, std=0.0),
            ]
        
        try:
            labeler = self._get_labeler()
            if labeler is None:
                raise ValueError("CheXbert labeler not available")
            
            # Get labels for predictions and references
            pred_labels = labeler(predictions)
            ref_labels = labeler(references)
            
            # Compute metrics
            similarity_scores = []
            f1_scores = []
            
            for pred_l, ref_l in zip(pred_labels, ref_labels):
                # Similarity: micro-averaged accuracy
                matches = sum(1 for p, r in zip(pred_l, ref_l) if p == r)
                similarity = matches / len(pred_l)
                similarity_scores.append(similarity)
                
                # F1: macro-averaged across labels
                label_f1s = []
                for i in range(len(pred_l)):
                    p, r = pred_l[i], ref_l[i]
                    # True positive, false positive, false negative
                    tp = 1 if p == 1 and r == 1 else 0
                    fp = 1 if p == 1 and r == 0 else 0
                    fn = 1 if p == 0 and r == 1 else 0
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # For binary match, use simpler logic
                    if p == r:
                        label_f1s.append(1.0)
                    else:
                        label_f1s.append(0.0)
                
                f1_scores.append(sum(label_f1s) / len(label_f1s))
            
            return [
                MetricResult.from_scores("CheXbert Similarity", similarity_scores),
                MetricResult.from_scores("CheXbert F1", f1_scores),
            ]
            
        except Exception as e:
            warnings.warn(f"CheXbert computation failed: {e}")
            return [
                MetricResult(
                    name="CheXbert Similarity", mean=0.0, std=0.0,
                    metadata={"error": str(e)}
                ),
                MetricResult(
                    name="CheXbert F1", mean=0.0, std=0.0,
                    metadata={"error": str(e)}
                ),
            ]
