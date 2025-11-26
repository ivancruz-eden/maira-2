"""
Utility functions for MAIRA-2 testing.
"""
import requests
from PIL import Image
from typing import Dict


def get_sample_data() -> Dict[str, Image.Image | str]:
    """
    Download chest X-rays from IU-Xray dataset (CC license).
    These images were not used in MAIRA-2 training.
    
    Returns:
        Dictionary containing sample frontal/lateral images and metadata
    """
    frontal_image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-1001.png"
    lateral_image_url = "https://openi.nlm.nih.gov/imgs/512/145/145/CXR145_IM-0290-2001.png"

    def download_and_open(url: str) -> Image.Image:
        response = requests.get(url, headers={"User-Agent": "MAIRA-2"}, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)

    print("Downloading sample chest X-ray images...")
    frontal_image = download_and_open(frontal_image_url)
    lateral_image = download_and_open(lateral_image_url)
    print(f"✓ Frontal image: {frontal_image.size}")
    print(f"✓ Lateral image: {lateral_image.size}")

    sample_data = {
        "frontal": frontal_image,
        "lateral": lateral_image,
        "indication": "Dyspnea.",
        "comparison": "None.",
        "technique": "PA and lateral views of the chest.",
        "phrase": "Pleural effusion."  # For phrase grounding (this patient has pleural effusion)
    }
    return sample_data


def save_image_with_boxes(image: Image.Image, boxes: list, output_path: str):
    """
    Save an image with bounding boxes drawn on it.
    
    Args:
        image: PIL Image
        boxes: List of box coordinates [(x1, y1, x2, y2), ...]
        output_path: Path to save the annotated image
    """
    from PIL import ImageDraw
    
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # Convert relative coordinates to absolute if needed
        if all(0 <= coord <= 1 for coord in box):
            w, h = image.size
            x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    
    img_copy.save(output_path)
    print(f"✓ Saved annotated image to {output_path}")
