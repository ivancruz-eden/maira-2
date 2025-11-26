"""
MAIRA-2 Model Testing - Findings Generation (Non-Grounded)

This script demonstrates basic radiology report generation from chest X-rays
without spatial grounding (bounding boxes).
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import get_sample_data


def main():
    print("=" * 80)
    print("MAIRA-2: Findings Generation (Non-Grounded)")
    print("=" * 80)
    
    # Initialize model and processor
    print("\n1. Loading MAIRA-2 model and processor...")
    print("   (This may take a few minutes on first run)")
    
    # Check available GPUs and load model
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"   Detected {num_gpus} GPU(s)")
        
        # Use device_map="auto" to distribute model across all GPUs
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/maira-2", 
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
        )
        device = torch.device("cuda:0")  # Primary device for inputs
        print("   ✓ Model distributed across GPUs")
    else:
        print("   No GPU detected, using CPU (this will be very slow)")
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
    print("   ✓ Model loaded successfully\n")
    
    # Get sample data
    print("2. Loading sample chest X-ray data...")
    sample_data = get_sample_data()
    print()
    
    # Prepare inputs for non-grounded reporting
    print("3. Preprocessing inputs for non-grounded report generation...")
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample_data["frontal"],
        current_lateral=sample_data["lateral"],
        prior_frontal=None,  # No prior study in this example
        indication=sample_data["indication"],
        technique=sample_data["technique"],
        comparison=sample_data["comparison"],
        prior_report=None,  # No prior report
        return_tensors="pt",
        get_grounding=False,  # Non-grounded reporting
    )
    processed_inputs = processed_inputs.to(device)
    print("   ✓ Inputs prepared\n")
    
    # Generate report
    print("4. Generating radiology findings...")
    print("   (This may take 30-60 seconds)")
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=300,  # For non-grounded reporting
            use_cache=True,
        )
    
    # Decode and parse output
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(
        output_decoding[0][prompt_length:], 
        skip_special_tokens=True
    )
    decoded_text = decoded_text.lstrip()
    
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    
    # Display results
    print("\n" + "=" * 80)
    print("GENERATED FINDINGS:")
    print("=" * 80)
    print(f"\n{prediction}\n")
    print("=" * 80)
    
    # Save to file
    with open("output_non_grounded.txt", "w") as f:
        f.write("MAIRA-2 Generated Findings (Non-Grounded)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Indication: {sample_data['indication']}\n")
        f.write(f"Technique: {sample_data['technique']}\n")
        f.write(f"Comparison: {sample_data['comparison']}\n\n")
        f.write("FINDINGS:\n")
        f.write(prediction)
    
    print("\n✓ Results saved to: output_non_grounded.txt\n")


if __name__ == "__main__":
    main()
