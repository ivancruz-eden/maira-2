"""
Base classes for metric computation.

This module defines the abstract base classes and data structures
used by all metric implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class MetricResult:
    """
    Container for metric computation results.
    
    Attributes:
        name: Name of the metric
        mean: Average score across all samples
        std: Standard deviation of scores
        scores: Individual scores for each sample (optional)
        metadata: Additional information about the computation
    """
    name: str
    mean: float
    std: float
    scores: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"{self.name}: {self.mean:.4f} ± {self.std:.4f}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "scores": self.scores,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_scores(
        cls, 
        name: str, 
        scores: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> "MetricResult":
        """
        Create a MetricResult from a list of scores.
        
        Args:
            name: Name of the metric
            scores: List of individual scores
            metadata: Additional metadata
            
        Returns:
            MetricResult instance with computed mean and std
        """
        scores_array = np.array(scores)
        return cls(
            name=name,
            mean=float(np.mean(scores_array)),
            std=float(np.std(scores_array)),
            scores=scores,
            metadata=metadata or {},
        )


class MetricCalculator(ABC):
    """
    Abstract base class for metric calculators.
    
    All metric implementations should inherit from this class
    and implement the compute() method.
    
    Example:
        ```python
        class MyMetric(MetricCalculator):
            @property
            def name(self) -> str:
                return "my_metric"
            
            def compute(self, predictions, references) -> MetricResult:
                scores = [self._score(p, r) for p, r in zip(predictions, references)]
                return MetricResult.from_scores(self.name, scores)
        ```
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the metric."""
        pass
    
    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> MetricResult:
        """
        Compute the metric for a batch of predictions and references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            MetricResult containing scores and statistics
        """
        pass
    
    def compute_single(
        self,
        prediction: str,
        reference: str,
    ) -> float:
        """
        Compute the metric for a single prediction-reference pair.
        
        Args:
            prediction: Generated report
            reference: Ground truth report
            
        Returns:
            Score for the single pair
        """
        result = self.compute([prediction], [reference])
        return result.scores[0] if result.scores else result.mean


class CompositeMetricCalculator:
    """
    Calculator that combines multiple metrics.
    
    Example:
        ```python
        calculator = CompositeMetricCalculator([
            BLEUMetric(n=2),
            BLEUMetric(n=4),
            METEORMetric(),
            ROUGELMetric(),
        ])
        
        results = calculator.compute_all(predictions, references)
        for result in results:
            print(result)
        ```
    """
    
    def __init__(self, metrics: List[MetricCalculator]):
        """
        Initialize with a list of metric calculators.
        
        Args:
            metrics: List of MetricCalculator instances
        """
        self.metrics = metrics
    
    def compute_all(
        self,
        predictions: List[str],
        references: List[str],
    ) -> List[MetricResult]:
        """
        Compute all metrics.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            List of MetricResult for each metric
        """
        results = []
        for metric in self.metrics:
            try:
                result = metric.compute(predictions, references)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to compute {metric.name}: {e}")
                results.append(MetricResult(
                    name=metric.name,
                    mean=0.0,
                    std=0.0,
                    metadata={"error": str(e)}
                ))
        return results
    
    def compute_all_dict(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, MetricResult]:
        """
        Compute all metrics and return as dictionary.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            
        Returns:
            Dictionary mapping metric names to results
        """
        results = self.compute_all(predictions, references)
        return {r.name: r for r in results}
