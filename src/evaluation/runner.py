"""
Evaluation runner for MAIRA-2 radiology report generation.

This module provides the main evaluation pipeline that:
1. Loads the dataset
2. Generates reports using MAIRA-2
3. Computes evaluation metrics
4. Outputs results in various formats
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import warnings

from tqdm import tqdm

from .dataset import EvaluationDataset, DatasetItem
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
from .metrics.base import CompositeMetricCalculator


@dataclass
class EvaluationConfig:
    """
    Configuration for the evaluation pipeline.
    
    Attributes:
        output_dir: Directory to save evaluation results
        save_predictions: Whether to save individual predictions
        save_per_sample_scores: Whether to save per-sample metric scores
        use_bleu: Whether to compute BLEU metrics
        use_meteor: Whether to compute METEOR metric
        use_rouge: Whether to compute ROUGE-L metric
        use_bertscore: Whether to compute BERTScore metrics
        use_radgraph: Whether to compute RadGraph/RGER metric
        use_chexbert: Whether to compute CheXbert metrics
        bertscore_model: Model to use for BERTScore
        device: Device for model inference ('cuda', 'cpu', or None)
        batch_size: Batch size for metric computation
    """
    output_dir: str = "./evaluation_results"
    save_predictions: bool = True
    save_per_sample_scores: bool = True
    
    # Metric toggles
    use_bleu: bool = True
    use_meteor: bool = True
    use_rouge: bool = True
    use_bertscore: bool = True
    use_radgraph: bool = True
    use_chexbert: bool = True
    
    # Metric configurations
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"
    device: Optional[str] = None
    batch_size: int = 32


class EvaluationRunner:
    """
    Main evaluation runner for radiology report generation.
    
    This class orchestrates the full evaluation pipeline:
    - Loading dataset and model
    - Generating predictions
    - Computing metrics
    - Saving results
    
    Example:
        ```python
        from evaluation import EvaluationRunner, EvaluationConfig
        from evaluation.dataset import load_evaluation_dataset
        
        # Load dataset
        dataset = load_evaluation_dataset("data/evaluation/")
        
        # Create runner
        config = EvaluationConfig(output_dir="results/")
        runner = EvaluationRunner(config)
        
        # Option 1: Evaluate with MAIRA-2 model
        results = runner.evaluate_with_maira2(dataset)
        
        # Option 2: Evaluate with pre-generated predictions
        predictions = [...]  # Your generated reports
        references = dataset.get_all_references()
        results = runner.evaluate_predictions(predictions, references)
        ```
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration. Uses defaults if not provided.
        """
        self.config = config or EvaluationConfig()
        self.metrics = self._setup_metrics()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_metrics(self) -> List[MetricCalculator]:
        """Set up the metric calculators based on config."""
        metrics = []
        
        # NLG metrics
        if self.config.use_bleu:
            try:
                metrics.append(BLEUMetric(n=2))
                metrics.append(BLEUMetric(n=4))
            except ImportError as e:
                warnings.warn(f"BLEU metrics unavailable: {e}")
        
        if self.config.use_meteor:
            try:
                metrics.append(METEORMetric())
            except ImportError as e:
                warnings.warn(f"METEOR metric unavailable: {e}")
        
        if self.config.use_rouge:
            try:
                metrics.append(ROUGELMetric())
            except ImportError as e:
                warnings.warn(f"ROUGE-L metric unavailable: {e}")
        
        # Semantic metrics
        if self.config.use_bertscore:
            try:
                metrics.append(BERTScoreMetric(
                    model_type=self.config.bertscore_model,
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                ))
            except ImportError as e:
                warnings.warn(f"BERTScore metric unavailable: {e}")
        
        # Clinical metrics
        if self.config.use_radgraph:
            try:
                metrics.append(RadGraphMetric(device=self.config.device))
            except ImportError as e:
                warnings.warn(f"RadGraph metric unavailable: {e}")
        
        if self.config.use_chexbert:
            try:
                metrics.append(CheXbertMetric(
                    device=self.config.device,
                    batch_size=self.config.batch_size,
                ))
            except ImportError as e:
                warnings.warn(f"CheXbert metric unavailable: {e}")
        
        return metrics
    
    def evaluate_predictions(
        self,
        predictions: List[str],
        references: List[str],
        instance_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate pre-generated predictions against references.
        
        Args:
            predictions: List of generated reports
            references: List of ground truth reports
            instance_ids: Optional list of sample identifiers
            
        Returns:
            Dictionary containing all evaluation results
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must match "
                f"number of references ({len(references)})"
            )
        
        print(f"\n{'='*60}")
        print("EVALUATING PREDICTIONS")
        print(f"{'='*60}")
        print(f"Samples: {len(predictions)}")
        print(f"Metrics: {len(self.metrics)}")
        
        # Compute all metrics
        all_results = []
        
        for metric in self.metrics:
            print(f"\nComputing {metric.name}...")
            
            try:
                if hasattr(metric, 'compute_all'):
                    # Metrics that return multiple results (BERTScore, CheXbert)
                    results = metric.compute_all(predictions, references)
                    all_results.extend(results)
                    for r in results:
                        print(f"  {r}")
                else:
                    result = metric.compute(predictions, references)
                    all_results.append(result)
                    print(f"  {result}")
            except Exception as e:
                warnings.warn(f"Failed to compute {metric.name}: {e}")
        
        # Build results dictionary
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(predictions),
            "metrics": {},
            "summary": [],
        }
        
        # Add metric results
        for result in all_results:
            results_dict["metrics"][result.name] = {
                "mean": result.mean,
                "std": result.std,
            }
            
            results_dict["summary"].append({
                "metric": result.name,
                "mean": result.mean,
                "std": result.std,
            })
            
            if self.config.save_per_sample_scores and result.scores:
                results_dict["metrics"][result.name]["scores"] = result.scores
        
        # Save predictions if requested
        if self.config.save_predictions:
            results_dict["predictions"] = []
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                item = {
                    "prediction": pred,
                    "reference": ref,
                }
                if instance_ids:
                    item["instance_id"] = instance_ids[i]
                
                # Add per-sample scores
                if self.config.save_per_sample_scores:
                    item["scores"] = {}
                    for result in all_results:
                        if result.scores and i < len(result.scores):
                            item["scores"][result.name] = result.scores[i]
                
                results_dict["predictions"].append(item)
        
        # Save results to file
        output_path = Path(self.config.output_dir) / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary table
        self._print_summary_table(all_results)
        
        return results_dict
    
    def evaluate_with_maira2(
        self,
        dataset: EvaluationDataset,
        model=None,
        processor=None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate using MAIRA-2 model to generate predictions.
        
        Args:
            dataset: EvaluationDataset to evaluate
            model: Pre-loaded MAIRA-2 model (will load if not provided)
            processor: Pre-loaded MAIRA-2 processor (will load if not provided)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing all evaluation results
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        # Load model if not provided
        if model is None or processor is None:
            print("\nLoading MAIRA-2 model...")
            
            if torch.cuda.is_available():
                model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/maira-2",
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                device = torch.device("cuda:0")
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/maira-2",
                    trust_remote_code=True,
                )
                device = torch.device("cpu")
                model = model.to(device)
            
            processor = AutoProcessor.from_pretrained(
                "microsoft/maira-2",
                trust_remote_code=True,
            )
            model = model.eval()
            print("Model loaded successfully")
        else:
            device = next(model.parameters()).device
        
        # Generate predictions
        print(f"\nGenerating predictions for {len(dataset)} samples...")
        predictions = []
        references = []
        instance_ids = []
        
        for i, item in enumerate(tqdm(dataset, desc="Generating")):
            if progress_callback:
                progress_callback(i + 1, len(dataset))
            
            try:
                # Load image
                image = item.load_image()
                
                # Prepare inputs for non-grounded reporting
                processed_inputs = processor.format_and_preprocess_reporting_input(
                    current_frontal=image,
                    current_lateral=None,
                    prior_frontal=None,
                    indication=item.indication or "Not provided.",
                    technique=item.technique or "PA view of the chest.",
                    comparison=item.comparison or "None.",
                    prior_report=None,
                    return_tensors="pt",
                    get_grounding=False,
                )
                processed_inputs = processed_inputs.to(device)
                
                # Generate report
                with torch.no_grad():
                    output = model.generate(
                        **processed_inputs,
                        max_new_tokens=300,
                        use_cache=True,
                    )
                
                # Decode output
                prompt_length = processed_inputs["input_ids"].shape[-1]
                decoded_text = processor.decode(
                    output[0][prompt_length:],
                    skip_special_tokens=True,
                ).lstrip()
                
                prediction = processor.convert_output_to_plaintext_or_grounded_sequence(
                    decoded_text
                )
                
                if isinstance(prediction, list):
                    prediction = " ".join(str(p) for p in prediction)
                
                predictions.append(str(prediction))
                references.append(item.reference_report)
                instance_ids.append(item.instance_id)
                
            except Exception as e:
                warnings.warn(f"Failed to process {item.instance_id}: {e}")
                predictions.append("")
                references.append(item.reference_report)
                instance_ids.append(item.instance_id)
        
        # Now evaluate the predictions
        return self.evaluate_predictions(
            predictions=predictions,
            references=references,
            instance_ids=instance_ids,
        )
    
    def _print_summary_table(self, results: List[MetricResult]):
        """Print a formatted summary table of results."""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Metric':<25} {'Mean':>12} {'Std':>12}")
        print(f"{'-'*60}")
        
        for result in results:
            print(f"{result.name:<25} {result.mean:>12.4f} {result.std:>12.4f}")
        
        print(f"{'='*60}")


def create_default_metrics() -> List[MetricCalculator]:
    """
    Create a default set of metric calculators.
    
    Returns:
        List of metric calculators for standard evaluation.
    """
    metrics = []
    
    # Try to add each metric, warn if unavailable
    metric_classes = [
        (BLEUMetric, {"n": 2}),
        (BLEUMetric, {"n": 4}),
        (METEORMetric, {}),
        (ROUGELMetric, {}),
        (BERTScoreMetric, {}),
        (RadGraphMetric, {}),
        (CheXbertMetric, {}),
    ]
    
    for metric_class, kwargs in metric_classes:
        try:
            metrics.append(metric_class(**kwargs))
        except ImportError as e:
            warnings.warn(f"Metric {metric_class.__name__} unavailable: {e}")
    
    return metrics
