#!/usr/bin/env python
"""
Example: Cross-Lingual Evaluation with Translation

This example demonstrates how to use the MAIRA-2 evaluation framework
with Spanish reference reports. It shows:

1. How to translate model predictions (EN) to Spanish for comparison
2. How to translate Spanish references to English for CheXbert metrics
3. How to configure and run the evaluation pipeline

The example can be run standalone to test the translation integration.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_direct_translation():
    """
    Example 1: Direct translation using NLLBTranslator
    
    Use this when you need to translate texts outside of the evaluation pipeline.
    """
    print("=" * 70)
    print("Example 1: Direct Translation with NLLBTranslator")
    print("=" * 70)
    
    from translation import NLLBTranslator
    
    # Create translator
    translator = NLLBTranslator()
    
    # Translate single text
    en_text = "The heart is normal in size. The lungs are clear."
    es_text = translator.translate(en_text, "en", "es")
    print(f"\nEN: {en_text}")
    print(f"ES: {es_text}")
    
    # Translate batch
    en_batch = [
        "No acute cardiopulmonary abnormality.",
        "Mild cardiomegaly is present.",
        "Small bilateral pleural effusions.",
    ]
    es_batch = translator.translate_batch(en_batch, "en", "es")
    
    print("\nBatch translation EN → ES:")
    for en, es in zip(en_batch, es_batch):
        print(f"  EN: {en}")
        print(f"  ES: {es}")


def example_dataset_translation():
    """
    Example 2: Dataset translation for evaluation
    
    Use this when preparing data for cross-lingual evaluation.
    """
    print("\n" + "=" * 70)
    print("Example 2: Dataset Translation for Evaluation")
    print("=" * 70)
    
    from evaluation import DatasetTranslator
    
    # Create dataset translator
    translator = DatasetTranslator(
        cache_dir="./translation_cache",
        use_cache=True,
    )
    
    # Simulated evaluation data
    predictions_en = [
        "Normal heart size. Clear lungs.",
        "Cardiomegaly present. No effusion.",
    ]
    references_es = [
        "Tamaño cardíaco normal. Pulmones claros.",
        "Cardiomegalia presente. Sin derrame.",
    ]
    
    # Translate for evaluation
    translated = translator.translate_evaluation_data(
        predictions_en=predictions_en,
        references_es=references_es,
        instance_ids=["001", "002"],
    )
    
    # Show results
    print("\nTranslation results:")
    for i in range(len(translated)):
        print(f"\n  Sample {translated.instance_ids[i]}:")
        print(f"    Prediction EN: {translated.predictions_en[i]}")
        print(f"    Prediction ES: {translated.predictions_es[i]}")
        print(f"    Reference ES:  {translated.references_es[i]}")
        print(f"    Reference EN:  {translated.references_en[i]}")


def example_full_evaluation():
    """
    Example 3: Full evaluation pipeline with translation
    
    This shows how to configure the evaluation runner for
    cross-lingual evaluation.
    """
    print("\n" + "=" * 70)
    print("Example 3: Evaluation Configuration for Cross-Lingual Evaluation")
    print("=" * 70)
    
    from evaluation import EvaluationRunner, EvaluationConfig
    
    # Configuration for Spanish evaluation dataset
    config = EvaluationConfig(
        output_dir="./evaluation_results",
        
        # Translation settings
        source_language="en",           # Model outputs English
        target_language="es",           # References are in Spanish
        enable_translation=True,        # Enable cross-lingual translation
        translation_cache_dir=None,     # Auto-create in output_dir
        save_translations=True,         # Save translated texts
        
        # Metric toggles
        use_bleu=True,
        use_meteor=True,
        use_rouge=True,
        use_bertscore=True,
        use_radgraph=False,             # Disable if not installed
        use_chexbert=True,              # Will use translated EN references
        
        # Other settings
        save_predictions=True,
        save_per_sample_scores=True,
    )
    
    print("\nEvaluation configuration:")
    print(f"  Source language (predictions): {config.source_language}")
    print(f"  Target language (references):  {config.target_language}")
    print(f"  Translation enabled:           {config.enable_translation}")
    print(f"  Save translations:             {config.save_translations}")
    
    print("\nMetrics configuration:")
    print(f"  BLEU:      {'✓' if config.use_bleu else '✗'}")
    print(f"  METEOR:    {'✓' if config.use_meteor else '✗'}")
    print(f"  ROUGE-L:   {'✓' if config.use_rouge else '✗'}")
    print(f"  BERTScore: {'✓' if config.use_bertscore else '✗'}")
    print(f"  RadGraph:  {'✓' if config.use_radgraph else '✗'}")
    print(f"  CheXbert:  {'✓' if config.use_chexbert else '✗'} (uses translated EN refs)")
    
    # Show how to run evaluation
    print("\nTo run evaluation with this configuration:")
    print("""
    runner = EvaluationRunner(config)
    
    # Option A: With MAIRA-2 model
    results = runner.evaluate_with_maira2(dataset)
    
    # Option B: With pre-generated predictions
    results = runner.evaluate_predictions(
        predictions=predictions_en,    # English predictions
        references=references_es,      # Spanish references
    )
    """)


def example_convenience_functions():
    """
    Example 4: Convenience functions for common translation tasks
    """
    print("\n" + "=" * 70)
    print("Example 4: Convenience Functions")
    print("=" * 70)
    
    from evaluation import (
        translate_references_for_chexbert,
        translate_predictions_to_spanish,
    )
    
    print("\nConvenience functions available:")
    print("""
    # Translate Spanish references to English for CheXbert
    references_en = translate_references_for_chexbert(
        references_es=spanish_references,
        cache_dir="./cache",
    )
    
    # Translate English predictions to Spanish
    predictions_es = translate_predictions_to_spanish(
        predictions_en=english_predictions,
        cache_dir="./cache",
    )
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Examples of cross-lingual evaluation with translation"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific example (1-4). Runs all if not specified."
    )
    parser.add_argument(
        "--skip-translation",
        action="store_true",
        help="Skip examples that require model download"
    )
    
    args = parser.parse_args()
    
    if args.skip_translation:
        print("Skipping examples that require translation model download")
        example_full_evaluation()
        example_convenience_functions()
    elif args.example:
        if args.example == 1:
            example_direct_translation()
        elif args.example == 2:
            example_dataset_translation()
        elif args.example == 3:
            example_full_evaluation()
        elif args.example == 4:
            example_convenience_functions()
    else:
        example_direct_translation()
        example_dataset_translation()
        example_full_evaluation()
        example_convenience_functions()
    
    print("\n" + "=" * 70)
    print("✓ Examples completed!")
    print("=" * 70)
