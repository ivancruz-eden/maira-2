"""
MAIRA-2 Model Testing - Phrase Grounding

This script demonstrates phrase grounding, where the model localizes
a specific finding (phrase) in the chest X-ray image.
"""
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from utils import get_sample_data, save_image_with_boxes


def main():
    print("=" * 80)
    print("MAIRA-2: Phrase Grounding")
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
    phrase = sample_data["phrase"]
    print(f"   Target phrase to ground: '{phrase}'\n")
    
    # Prepare inputs for phrase grounding
    print("3. Preprocessing inputs for phrase grounding...")
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=sample_data["frontal"],
        phrase=phrase,
        return_tensors="pt",
    )
    processed_inputs = processed_inputs.to(device)
    print("   ✓ Inputs prepared\n")
    
    # Generate phrase grounding
    print("4. Performing phrase grounding...")
    print("   (This may take 10-30 seconds)")
    with torch.no_grad():
        output_decoding = model.generate(
            **processed_inputs,
            max_new_tokens=150,
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
    print("PHRASE GROUNDING RESULT:")
    print("=" * 80)
    print(f"\nInput phrase: '{phrase}'")
    print(f"\nPrediction: {prediction}")
    
    if isinstance(prediction, tuple):
        text, boxes = prediction
        if boxes:
            print(f"\nLocalized finding: {text}")
            print(f"Bounding box(es): {boxes}")
            print("\nBox format: (x_topleft, y_topleft, x_bottomright, y_bottomright)")
            print("Coordinates are relative to the cropped image.")
        else:
            print("\nNo bounding boxes detected for this phrase.")
    
    print("\n" + "=" * 80)
    
    # Save to file
    with open("output_phrase_grounding.txt", "w") as f:
        f.write("MAIRA-2 Phrase Grounding Result\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input phrase: '{phrase}'\n\n")
        f.write(f"Prediction: {prediction}\n")
    
    print("\n✓ Results saved to: output_phrase_grounding.txt")
    
    # Visualize bounding boxes
    if isinstance(prediction, tuple):
        text, boxes = prediction
        if boxes:
            print("\n5. Creating visualization with bounding boxes...")
            save_image_with_boxes(
                sample_data["frontal"], 
                boxes, 
                "output_phrase_grounding_visualization.png"
            )
            print("   Note: Boxes are relative to the cropped image MAIRA-2 sees.")
            print("   Use processor.adjust_box_for_original_image_size() for original coordinates.\n")


if __name__ == "__main__":
    main()
