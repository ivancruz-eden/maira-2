"""
MAIRA-2 Model Testing - All Use Cases

This script runs all three MAIRA-2 capabilities sequentially:
1. Non-grounded findings generation
2. Grounded report generation
3. Phrase grounding
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import get_sample_data, save_image_with_boxes


def test_non_grounded_generation(model, processor, sample_data, device):
    """Test non-grounded findings generation."""
    print("\n" + "=" * 80)
    print("TEST 1: NON-GROUNDED FINDINGS GENERATION")
    print("=" * 80)
    
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample_data["frontal"],
        current_lateral=sample_data["lateral"],
        prior_frontal=None,
        indication=sample_data["indication"],
        technique=sample_data["technique"],
        comparison=sample_data["comparison"],
        prior_report=None,
        return_tensors="pt",
        get_grounding=False,
    )
    processed_inputs = processed_inputs.to(device)
    
    print("Generating report...")
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=300,
            use_cache=True,
        )
    
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(
        output_decoding[0][prompt_length:], 
        skip_special_tokens=True
    ).lstrip()
    
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    
    print("\nFINDINGS:")
    print("-" * 80)
    print(prediction)
    print("-" * 80)
    
    return prediction


def test_grounded_generation(model, processor, sample_data, device):
    """Test grounded report generation."""
    print("\n" + "=" * 80)
    print("TEST 2: GROUNDED REPORT GENERATION")
    print("=" * 80)
    
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample_data["frontal"],
        current_lateral=sample_data["lateral"],
        prior_frontal=None,
        indication=sample_data["indication"],
        technique=sample_data["technique"],
        comparison=sample_data["comparison"],
        prior_report=None,
        return_tensors="pt",
        get_grounding=True,
    )
    processed_inputs = processed_inputs.to(device)
    
    print("Generating grounded report...")
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=450,
            use_cache=True,
        )
    
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(
        output_decoding[0][prompt_length:], 
        skip_special_tokens=True
    )
    
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    
    print("\nGROUNDED FINDINGS:")
    print("-" * 80)
    for item in prediction:
        if isinstance(item, tuple):
            text, boxes = item
            print(f"• {text}")
            if boxes:
                print(f"  └─ Boxes: {boxes}")
        else:
            print(item)
    print("-" * 80)
    
    return prediction


def test_phrase_grounding(model, processor, sample_data, device):
    """Test phrase grounding."""
    print("\n" + "=" * 80)
    print("TEST 3: PHRASE GROUNDING")
    print("=" * 80)
    
    phrase = sample_data["phrase"]
    print(f"Target phrase: '{phrase}'")
    
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=sample_data["frontal"],
        phrase=phrase,
        return_tensors="pt",
    )
    processed_inputs = processed_inputs.to(device)
    
    print("Performing phrase grounding...")
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=150,
            use_cache=True,
        )
    
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(
        output_decoding[0][prompt_length:], 
        skip_special_tokens=True
    )
    
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    
    print("\nRESULT:")
    print("-" * 80)
    if isinstance(prediction, tuple):
        text, boxes = prediction
        print(f"Text: {text}")
        if boxes:
            print(f"Boxes: {boxes}")
    else:
        print(prediction)
    print("-" * 80)
    
    return prediction


def main():
    print("=" * 80)
    print("MAIRA-2 COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("\nThis script will test all three MAIRA-2 capabilities:")
    print("  1. Non-grounded findings generation")
    print("  2. Grounded report generation")
    print("  3. Phrase grounding")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    print("\nLoading MAIRA-2 model and processor...")
    print("(This may take a few minutes on first run)")
    
    # Check available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Use device_map="auto" to distribute model across all GPUs
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/maira-2", 
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
        )
        device = torch.device("cuda:0")  # Primary device for inputs
        print("✓ Model distributed across GPUs using device_map='auto'")
    else:
        print("No GPU detected, using CPU (this will be very slow)")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/maira-2", 
            trust_remote_code=True
        )
        device = torch.device("cpu")
        model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/maira-2", 
        trust_remote_code=True
    )
    
    model = model.eval()
    print("✓ Model loaded successfully")
    
    # Get sample data
    print("\n" + "=" * 80)
    print("LOADING SAMPLE DATA")
    print("=" * 80)
    sample_data = get_sample_data()
    
    # Run all tests
    results = {}
    
    try:
        results['non_grounded'] = test_non_grounded_generation(
            model, processor, sample_data, device
        )
    except Exception as e:
        print(f"\n✗ Test 1 failed: {e}")
    
    try:
        results['grounded'] = test_grounded_generation(
            model, processor, sample_data, device
        )
    except Exception as e:
        print(f"\n✗ Test 2 failed: {e}")
    
    try:
        results['phrase_grounding'] = test_phrase_grounding(
            model, processor, sample_data, device
        )
    except Exception as e:
        print(f"\n✗ Test 3 failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\nCompleted {len(results)}/3 tests successfully")
    print("\nResults saved to individual output files:")
    print("  - output_non_grounded.txt")
    print("  - output_grounded.txt")
    print("  - output_phrase_grounding.txt")
    print("\n✓ All tests completed!\n")


if __name__ == "__main__":
    main()
