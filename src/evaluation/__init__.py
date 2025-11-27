"""
MAIRA-2 Evaluation Framework

A modular evaluation framework for radiology report generation.
Provides reusable metrics and dataset loading capabilities.

For cross-lingual evaluation (e.g., English predictions vs Spanish references):
- Enables translation of predictions to target language for comparison
- Translates references to English for CheXbert metrics
"""

from .dataset import EvaluationDataset, DatasetItem, load_evaluation_dataset
from .dataset_translation import (
    DatasetTranslator,
    TranslatedDataset,
    translate_references_for_chexbert,
    translate_predictions_to_spanish,
)
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
    # Dataset Translation
    "DatasetTranslator",
    "TranslatedDataset",
    "translate_references_for_chexbert",
    "translate_predictions_to_spanish",
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
