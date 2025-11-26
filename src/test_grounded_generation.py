"""
MAIRA-2 Model Testing - Grounded Report Generation

This script demonstrates radiology report generation with spatial grounding,
where findings are accompanied by bounding box coordinates.
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import get_sample_data, save_image_with_boxes


def main():
    print("=" * 80)
    print("MAIRA-2: Grounded Report Generation")
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
    
    # Prepare inputs for grounded reporting
    print("3. Preprocessing inputs for grounded report generation...")
    processed_inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=sample_data["frontal"],
        current_lateral=sample_data["lateral"],
        prior_frontal=None,
        indication=sample_data["indication"],
        technique=sample_data["technique"],
        comparison=sample_data["comparison"],
        prior_report=None,
        return_tensors="pt",
        get_grounding=True,  # Enable grounding
    )
    processed_inputs = processed_inputs.to(device)
    print("   ✓ Inputs prepared\n")
    
    # Generate grounded report
    print("4. Generating grounded radiology findings...")
    print("   (This may take 30-60 seconds)")
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=450,  # Increased for grounded reporting
            use_cache=True,
        )
    
    # Decode and parse output
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(
        output_decoding[0][prompt_length:], 
        skip_special_tokens=True
    )
    
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    
    # Display results
    print("\n" + "=" * 80)
    print("GENERATED GROUNDED FINDINGS:")
    print("=" * 80)
    
    for item in prediction:
        if isinstance(item, tuple):
            text, boxes = item
            if boxes:
                print(f"\n• {text}")
                print(f"  Boxes: {boxes}")
            else:
                print(f"\n• {text}")
                print(f"  (No spatial annotation)")
        else:
            print(f"\n{item}")
    
    print("\n" + "=" * 80)
    
    # Save to file
    with open("output_grounded.txt", "w") as f:
        f.write("MAIRA-2 Generated Findings (Grounded)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Indication: {sample_data['indication']}\n")
        f.write(f"Technique: {sample_data['technique']}\n")
        f.write(f"Comparison: {sample_data['comparison']}\n\n")
        f.write("FINDINGS:\n\n")
        
        for item in prediction:
            if isinstance(item, tuple):
                text, boxes = item
                f.write(f"• {text}\n")
                if boxes:
                    f.write(f"  Boxes: {boxes}\n")
                else:
                    f.write(f"  (No spatial annotation)\n")
            else:
                f.write(f"{item}\n")
    
    print("\n✓ Results saved to: output_grounded.txt")
    
    # Extract and visualize boxes
    print("\n5. Extracting bounding boxes for visualization...")
    all_boxes = []
    for item in prediction:
        if isinstance(item, tuple):
            text, boxes = item
            if boxes:
                all_boxes.extend(boxes)
    
    if all_boxes:
        print(f"   Found {len(all_boxes)} bounding box(es)")
        save_image_with_boxes(
            sample_data["frontal"], 
            all_boxes, 
            "output_grounded_visualization.png"
        )
        print("   Note: Boxes are relative to the cropped image MAIRA-2 sees.")
        print("   Use processor.adjust_box_for_original_image_size() for original coordinates.\n")
    else:
        print("   No bounding boxes found in the generated report.\n")


if __name__ == "__main__":
    main()
