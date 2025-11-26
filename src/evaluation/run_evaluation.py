#!/usr/bin/env python
"""
MAIRA-2 Evaluation Script

This script evaluates the MAIRA-2 non-grounded report generation
against the evaluation dataset.

Usage:
    # Evaluate using MAIRA-2 model
    python run_evaluation.py --data-dir ../data/first_labeling_studies_june_2025/
    
    # Evaluate with custom output directory
    python run_evaluation.py --data-dir ../data/first_labeling_studies_june_2025/ --output-dir ./results/
    
    # Skip certain metrics
    python run_evaluation.py --data-dir ../data/first_labeling_studies_june_2025/ --no-radgraph
"""
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    EvaluationRunner,
    EvaluationConfig,
    load_evaluation_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate MAIRA-2 non-grounded report generation"
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../data/first_labeling_studies_june_2025/",
        help="Path to the evaluation data directory"
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="images",
        help="Subdirectory containing images (default: images)"
    )
    parser.add_argument(
        "--csv-filename",
        type=str,
        default="merged_june_andres_labeling.csv",
        help="Name of the CSV file with metadata"
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png", "dcm"],
        help="Image format to use"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Don't save individual predictions"
    )
    parser.add_argument(
        "--no-save-scores",
        action="store_true",
        help="Don't save per-sample scores"
    )
    
    # Metric toggles
    parser.add_argument(
        "--no-bleu",
        action="store_true",
        help="Skip BLEU metrics"
    )
    parser.add_argument(
        "--no-meteor",
        action="store_true",
        help="Skip METEOR metric"
    )
    parser.add_argument(
        "--no-rouge",
        action="store_true",
        help="Skip ROUGE-L metric"
    )
    parser.add_argument(
        "--no-bertscore",
        action="store_true",
        help="Skip BERTScore metrics"
    )
    parser.add_argument(
        "--no-radgraph",
        action="store_true",
        help="Skip RadGraph/RGER metric"
    )
    parser.add_argument(
        "--no-chexbert",
        action="store_true",
        help="Skip CheXbert metrics"
    )
    
    # Model arguments
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (cuda, cpu, or auto)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for metric computation"
    )
    
    # Limit samples (for testing)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("MAIRA-2 EVALUATION - Non-Grounded Report Generation")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    try:
        dataset = load_evaluation_dataset(
            data_dir=args.data_dir,
            csv_filename=args.csv_filename,
            images_subdir=args.images_subdir,
            image_format=args.image_format,
        )
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(dataset):
        print(f"Limiting to {args.max_samples} samples")
        dataset._items = dataset._items[:args.max_samples]
    
    # Create evaluation config
    config = EvaluationConfig(
        output_dir=args.output_dir,
        save_predictions=not args.no_save_predictions,
        save_per_sample_scores=not args.no_save_scores,
        use_bleu=not args.no_bleu,
        use_meteor=not args.no_meteor,
        use_rouge=not args.no_rouge,
        use_bertscore=not args.no_bertscore,
        use_radgraph=not args.no_radgraph,
        use_chexbert=not args.no_chexbert,
        device=args.device,
        batch_size=args.batch_size,
    )
    
    # Create runner
    runner = EvaluationRunner(config)
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    try:
        results = runner.evaluate_with_maira2(dataset)
        
        print(f"\nEvaluation complete!")
        print(f"Results saved to: {args.output_dir}/evaluation_results.json")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
