#!/usr/bin/env python
"""
Example: Using MAIRA-2 Evaluation Metrics Standalone

This script demonstrates how to use the evaluation metrics module
independently for evaluating any text generation task.

The metrics are reusable and can be applied to:
- Radiology report generation
- Medical text summarization
- Any text-to-text generation task
"""
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def example_basic_metrics():
    """Example: Using basic NLG metrics."""
    from evaluation.metrics import BLEUMetric, METEORMetric, ROUGELMetric
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic NLG Metrics")
    print("="*60)
    
    # Sample data
    predictions = [
        "The heart size is normal. The lungs are clear.",
        "There is cardiomegaly. Bilateral pleural effusions.",
        "No acute cardiopulmonary abnormality."
    ]
    references = [
        "Normal heart size. Lungs are clear without infiltrate.",
        "Cardiomegaly present. Bilateral pleural effusions noted.",
        "No acute cardiopulmonary process."
    ]
    
    # Compute BLEU-2 and BLEU-4
    bleu2 = BLEUMetric(n=2)
    result = bleu2.compute(predictions, references)
    print(f"\nBLEU-2: {result.mean:.4f} ± {result.std:.4f}")
    print(f"  Per-sample scores: {[f'{s:.4f}' for s in result.scores]}")
    
    bleu4 = BLEUMetric(n=4)
    result = bleu4.compute(predictions, references)
    print(f"\nBLEU-4: {result.mean:.4f} ± {result.std:.4f}")
    print(f"  Per-sample scores: {[f'{s:.4f}' for s in result.scores]}")
    
    # Compute METEOR
    meteor = METEORMetric()
    result = meteor.compute(predictions, references)
    print(f"\nMETEOR: {result.mean:.4f} ± {result.std:.4f}")
    print(f"  Per-sample scores: {[f'{s:.4f}' for s in result.scores]}")
    
    # Compute ROUGE-L
    rouge = ROUGELMetric()
    result = rouge.compute(predictions, references)
    print(f"\nROUGE-L: {result.mean:.4f} ± {result.std:.4f}")
    print(f"  Per-sample scores: {[f'{s:.4f}' for s in result.scores]}")


def example_bertscore():
    """Example: Using BERTScore for semantic similarity."""
    from evaluation.metrics import BERTScoreMetric
    
    print("\n" + "="*60)
    print("EXAMPLE 2: BERTScore (Semantic Similarity)")
    print("="*60)
    
    predictions = [
        "The cardiac silhouette is within normal limits.",
        "Pulmonary edema is present.",
    ]
    references = [
        "Normal heart size.",
        "There is pulmonary edema.",
    ]
    
    # BERTScore returns F1, Precision, and Recall
    bertscore = BERTScoreMetric(
        model_type="distilbert-base-uncased",  # Faster model for example
        return_all=True
    )
    
    print("\nComputing BERTScore (this may take a moment)...")
    results = bertscore.compute_all(predictions, references)
    
    for result in results:
        print(f"\n{result.name}: {result.mean:.4f} ± {result.std:.4f}")


def example_composite_metrics():
    """Example: Using composite metric calculator."""
    from evaluation.metrics import BLEUMetric, METEORMetric, ROUGELMetric
    from evaluation.metrics.base import CompositeMetricCalculator
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Composite Metric Calculator")
    print("="*60)
    
    predictions = ["Normal chest radiograph."]
    references = ["Chest X-ray shows no abnormalities."]
    
    # Create composite calculator
    calculator = CompositeMetricCalculator([
        BLEUMetric(n=2),
        BLEUMetric(n=4),
        METEORMetric(),
        ROUGELMetric(),
    ])
    
    # Compute all metrics at once
    results = calculator.compute_all(predictions, references)
    
    print("\nAll metrics computed:")
    for result in results:
        print(f"  {result.name}: {result.mean:.4f}")


def example_single_sample():
    """Example: Computing metrics for a single sample."""
    from evaluation.metrics import BLEUMetric, ROUGELMetric
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Single Sample Evaluation")
    print("="*60)
    
    prediction = "The heart is enlarged. Bilateral pleural effusions are present."
    reference = "Cardiomegaly. Bilateral pleural effusions."
    
    bleu4 = BLEUMetric(n=4)
    score = bleu4.compute_single(prediction, reference)
    print(f"\nBLEU-4 score: {score:.4f}")
    
    rouge = ROUGELMetric()
    score = rouge.compute_single(prediction, reference)
    print(f"ROUGE-L score: {score:.4f}")


def example_custom_metric():
    """Example: Creating a custom metric."""
    from evaluation.metrics.base import MetricCalculator, MetricResult
    from typing import List
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Metric Implementation")
    print("="*60)
    
    class WordOverlapMetric(MetricCalculator):
        """Simple word overlap ratio metric."""
        
        @property
        def name(self) -> str:
            return "Word Overlap"
        
        def compute(
            self,
            predictions: List[str],
            references: List[str],
        ) -> MetricResult:
            scores = []
            for pred, ref in zip(predictions, references):
                pred_words = set(pred.lower().split())
                ref_words = set(ref.lower().split())
                
                if not ref_words:
                    scores.append(0.0)
                else:
                    overlap = len(pred_words & ref_words)
                    score = overlap / len(ref_words)
                    scores.append(score)
            
            return MetricResult.from_scores(self.name, scores)
    
    # Use custom metric
    overlap_metric = WordOverlapMetric()
    
    predictions = ["The heart is normal. Lungs clear."]
    references = ["Normal heart. Clear lungs."]
    
    result = overlap_metric.compute(predictions, references)
    print(f"\nCustom Word Overlap: {result.mean:.4f}")


def main():
    print("="*60)
    print("MAIRA-2 EVALUATION METRICS - STANDALONE EXAMPLES")
    print("="*60)
    print("\nThis demonstrates how to use the metrics module independently.")
    print("These metrics can be reused for any text generation evaluation.")
    
    # Run examples
    example_basic_metrics()
    example_composite_metrics()
    example_single_sample()
    example_custom_metric()
    
    # BERTScore is slow, so optional
    try:
        example_bertscore()
    except ImportError:
        print("\nSkipping BERTScore example (bert-score not installed)")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
