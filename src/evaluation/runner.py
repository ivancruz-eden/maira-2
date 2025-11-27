"""
Evaluation runner for MAIRA-2 radiology report generation.

This module provides the main evaluation pipeline that:
1. Loads the dataset
2. Generates reports using MAIRA-2
3. Handles translation for cross-lingual evaluation (EN ↔ ES)
4. Computes evaluation metrics
5. Outputs results in various formats

For Spanish evaluation datasets:
- Model predictions (EN) are translated to Spanish for comparison
- Spanish references are translated to English for CheXbert metrics
"""
import json
import sys
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import warnings

from tqdm import tqdm

from .dataset import EvaluationDataset, DatasetItem
from .dataset_translation import DatasetTranslator, TranslatedDataset
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
        
        # Translation settings
        source_language: Language of model predictions ('en')
        target_language: Language of reference reports ('es' for Spanish datasets)
        enable_translation: Whether to enable translation for cross-lingual evaluation
        translation_cache_dir: Directory to cache translations
        save_translations: Whether to save translated texts
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
    
    # Translation settings
    source_language: str = "en"  # Model output language
    target_language: str = "es"  # Reference report language
    enable_translation: bool = True  # Enable cross-lingual translation
    translation_cache_dir: Optional[str] = None  # Cache dir for translations
    save_translations: bool = True  # Save translated texts to results


class EvaluationRunner:
    """
    Main evaluation runner for radiology report generation.
    
    This class orchestrates the full evaluation pipeline:
    - Loading dataset and model
    - Generating predictions
    - Translating for cross-lingual evaluation (EN ↔ ES)
    - Computing metrics
    - Saving results
    
    For Spanish evaluation datasets:
    - Model predictions (EN) are compared directly with translated EN references
    - CheXbert metrics use translated English references
    - Optionally, predictions are also translated to Spanish
    
    Example:
        ```python
        from evaluation import EvaluationRunner, EvaluationConfig
        from evaluation.dataset import load_evaluation_dataset
        
        # Load dataset (Spanish references)
        dataset = load_evaluation_dataset("data/evaluation/")
        
        # Create runner with translation enabled
        config = EvaluationConfig(
            output_dir="results/",
            source_language="en",      # Model outputs in English
            target_language="es",      # References are in Spanish
            enable_translation=True,   # Translate for cross-lingual eval
        )
        runner = EvaluationRunner(config)
        
        # Option 1: Evaluate with MAIRA-2 model
        results = runner.evaluate_with_maira2(dataset)
        
        # Option 2: Evaluate with pre-generated predictions
        predictions = [...]  # Your generated reports (in English)
        references = dataset.get_all_references()  # Spanish references
        results = runner.evaluate_predictions(predictions, references)
        ```
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        translator=None,
    ):
        """
        Initialize the evaluation runner.
        
        Args:
            config: Evaluation configuration. Uses defaults if not provided.
            translator: NLLBTranslator instance for cross-lingual evaluation.
                       Will be created lazily if translation is enabled.
        """
        self.config = config or EvaluationConfig()
        self._translator = translator
        self._dataset_translator = None
        self.metrics = self._setup_metrics()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _get_dataset_translator(self) -> DatasetTranslator:
        """Get or create the dataset translator."""
        if self._dataset_translator is None:
            cache_dir = self.config.translation_cache_dir
            if cache_dir is None:
                cache_dir = str(Path(self.config.output_dir) / "translation_cache")
            
            self._dataset_translator = DatasetTranslator(
                translator=self._translator,
                cache_dir=cache_dir,
                use_cache=True,
            )
        return self._dataset_translator
    
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
        references_language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate pre-generated predictions against references.
        
        Handles cross-lingual evaluation when predictions (English) and 
        references (Spanish) are in different languages:
        - Translates predictions EN → ES for Spanish comparison metrics
        - Translates references ES → EN for CheXbert metrics (English-only)
        
        Args:
            predictions: List of generated reports (typically in English)
            references: List of ground truth reports (may be in Spanish)
            instance_ids: Optional list of sample identifiers
            references_language: Language of references (overrides config.target_language)
            
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
        
        # Determine languages
        pred_lang = self.config.source_language
        ref_lang = references_language or self.config.target_language
        cross_lingual = (pred_lang != ref_lang) and self.config.enable_translation
        
        if cross_lingual:
            print(f"\nCross-lingual evaluation:")
            print(f"  Predictions language: {pred_lang.upper()}")
            print(f"  References language: {ref_lang.upper()}")
        
        # Initialize data containers
        translated_data = None
        predictions_for_metrics = predictions
        references_for_metrics = references
        references_for_chexbert = references
        
        # Handle translation if needed
        if cross_lingual:
            print(f"\n{'='*60}")
            print("TRANSLATION STEP")
            print(f"{'='*60}")
            
            dataset_translator = self._get_dataset_translator()
            
            # Translate predictions to target language (EN → ES) for output
            print("\n1. Translating predictions (EN → ES)...")
            predictions_translated = dataset_translator.translate_texts(
                predictions,
                source_lang=pred_lang,
                target_lang=ref_lang,
                show_progress=True,
            )
            
            # Translate references to source language (ES → EN) for CheXbert
            print("\n2. Translating references (ES → EN) for CheXbert metrics...")
            references_for_chexbert = dataset_translator.translate_texts(
                references,
                source_lang=ref_lang,
                target_lang=pred_lang,
                show_progress=True,
            )
            
            # Store translated data
            translated_data = TranslatedDataset(
                predictions_en=predictions,
                predictions_es=predictions_translated,
                references_es=references,
                references_en=references_for_chexbert,
                instance_ids=instance_ids or [str(i) for i in range(len(predictions))],
            )
            
            print(f"\nTranslation complete!")
        
        print(f"\n{'='*60}")
        print("COMPUTING METRICS")
        print(f"{'='*60}")
        
        # Compute all metrics
        all_results = []
        
        for metric in self.metrics:
            print(f"\nComputing {metric.name}...")
            
            try:
                # Determine which data to use for this metric
                # CheXbert and RadGraph need English text - use translated references
                if isinstance(metric, (CheXbertMetric, RadGraphMetric)):
                    metric_predictions = predictions  # Already in English
                    metric_references = references_for_chexbert  # English translations
                    if cross_lingual:
                        print(f"  (Using English-translated references for {metric.name})")
                else:
                    # Other metrics use original data
                    metric_predictions = predictions_for_metrics
                    metric_references = references_for_metrics
                
                if hasattr(metric, 'compute_all'):
                    # Metrics that return multiple results (BERTScore, CheXbert)
                    results = metric.compute_all(metric_predictions, metric_references)
                    all_results.extend(results)
                    for r in results:
                        print(f"  {r}")
                else:
                    result = metric.compute(metric_predictions, metric_references)
                    all_results.append(result)
                    print(f"  {result}")
            except Exception as e:
                warnings.warn(f"Failed to compute {metric.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Build results dictionary
        results_dict = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(predictions),
            "languages": {
                "predictions": pred_lang,
                "references": ref_lang,
                "cross_lingual_evaluation": cross_lingual,
            },
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
                item = {}
                
                # Add instance ID first if available
                if instance_ids:
                    item["instance_id"] = instance_ids[i]
                
                # Add predictions with translations
                item["prediction"] = {
                    "original": pred,
                    "language": pred_lang,
                }
                if translated_data:
                    item["prediction"]["translated"] = translated_data.predictions_es[i]
                    item["prediction"]["translated_language"] = ref_lang
                
                # Add references with translations
                item["reference"] = {
                    "original": ref,
                    "language": ref_lang,
                }
                if translated_data:
                    item["reference"]["translated"] = translated_data.references_en[i]
                    item["reference"]["translated_language"] = pred_lang
                
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
