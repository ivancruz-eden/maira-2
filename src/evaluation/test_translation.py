#!/usr/bin/env python
"""
Test script for the translation module.

This script tests the NLLB translation functionality with sample
radiology report text to ensure the translation module works correctly.

Usage:
    python test_translation.py
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_basic_translation():
    """Test basic translation functionality."""
    print("=" * 60)
    print("Testing Translation Module")
    print("=" * 60)
    
    from translation import NLLBTranslator, TranslationConfig
    
    # Create translator
    print("\n1. Creating NLLB translator...")
    config = TranslationConfig(
        model_name="facebook/nllb-200-distilled-600M",
        batch_size=4,
    )
    translator = NLLBTranslator(config)
    
    # Test EN -> ES translation
    print("\n2. Testing English to Spanish translation...")
    english_texts = [
        "The heart is normal in size.",
        "The lungs are clear without infiltrates.",
        "No pleural effusion is seen.",
        "There is mild cardiomegaly.",
    ]
    
    spanish_translations = translator.translate_batch(english_texts, "en", "es")
    
    print("\nEnglish → Spanish:")
    for en, es in zip(english_texts, spanish_translations):
        print(f"  EN: {en}")
        print(f"  ES: {es}")
        print()
    
    # Test ES -> EN translation
    print("\n3. Testing Spanish to English translation...")
    spanish_texts = [
        "El corazón es de tamaño normal.",
        "Los pulmones están limpios sin infiltrados.",
        "No se observa derrame pleural.",
        "Hay una leve cardiomegalia.",
    ]
    
    english_translations = translator.translate_batch(spanish_texts, "es", "en")
    
    print("\nSpanish → English:")
    for es, en in zip(spanish_texts, english_translations):
        print(f"  ES: {es}")
        print(f"  EN: {en}")
        print()
    
    # Test convenience methods
    print("\n4. Testing convenience methods...")
    single_en = translator.translate_en_to_es("Normal chest X-ray.")
    single_es = translator.translate_es_to_en("Radiografía de tórax normal.")
    
    print(f"  EN → ES: 'Normal chest X-ray.' → '{single_en}'")
    print(f"  ES → EN: 'Radiografía de tórax normal.' → '{single_es}'")
    
    print("\n" + "=" * 60)
    print("Translation tests completed successfully!")
    print("=" * 60)


def test_dataset_translator():
    """Test the dataset translator functionality."""
    print("\n" + "=" * 60)
    print("Testing Dataset Translator")
    print("=" * 60)
    
    from evaluation.dataset_translation import DatasetTranslator, TranslatedDataset
    
    # Create dataset translator
    print("\n1. Creating dataset translator...")
    translator = DatasetTranslator(
        cache_dir="./test_translation_cache",
        use_cache=True,
    )
    
    # Test evaluation data translation
    print("\n2. Translating evaluation data...")
    predictions_en = [
        "The heart is normal in size. Lungs are clear.",
        "Mild cardiomegaly is present. No pleural effusion.",
    ]
    references_es = [
        "El corazón es de tamaño normal. Los pulmones están claros.",
        "Cardiomegalia leve presente. Sin derrame pleural.",
    ]
    instance_ids = ["sample_001", "sample_002"]
    
    translated = translator.translate_evaluation_data(
        predictions_en=predictions_en,
        references_es=references_es,
        instance_ids=instance_ids,
    )
    
    print("\n3. Translation results:")
    for i in range(len(translated)):
        print(f"\n  Sample {translated.instance_ids[i]}:")
        print(f"    Prediction (EN): {translated.predictions_en[i]}")
        print(f"    Prediction (ES): {translated.predictions_es[i]}")
        print(f"    Reference (ES):  {translated.references_es[i]}")
        print(f"    Reference (EN):  {translated.references_en[i]}")
    
    # Test serialization
    print("\n4. Testing serialization...")
    output_path = "./test_translated_dataset.json"
    translator.save_translated_dataset(translated, output_path)
    
    loaded = translator.load_translated_dataset(output_path)
    print(f"  Saved and loaded {len(loaded)} samples")
    
    # Clean up
    import os
    os.remove(output_path)
    
    print("\n" + "=" * 60)
    print("Dataset translator tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test translation module")
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="Skip basic translation tests"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset translator tests"
    )
    
    args = parser.parse_args()
    
    if not args.skip_basic:
        test_basic_translation()
    
    if not args.skip_dataset:
        test_dataset_translator()
    
    print("\n✓ All tests passed!")
