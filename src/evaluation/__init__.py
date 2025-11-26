"""
MAIRA-2 Evaluation Framework

A modular evaluation framework for radiology report generation.
Provides reusable metrics and dataset loading capabilities.
"""

from .dataset import EvaluationDataset, DatasetItem, load_evaluation_dataset
from .metrics import (
    MetricResult,
    MetricCalculator,
    BLEUMetric,
    METEORMetric,
    ROUGELMetric,
    BERTScoreMetric,
    RadGraphMetric,
    CheXbertMetric,
)
from .runner import EvaluationRunner, EvaluationConfig

__all__ = [
    # Dataset
    "EvaluationDataset",
    "DatasetItem",
    "load_evaluation_dataset",
    # Metrics
    "MetricResult",
    "MetricCalculator",
    "BLEUMetric",
    "METEORMetric",
    "ROUGELMetric",
    "BERTScoreMetric",
    "RadGraphMetric",
    "CheXbertMetric",
    # Runner
    "EvaluationRunner",
    "EvaluationConfig",
]
