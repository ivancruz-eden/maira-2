# MAIRA-2 Testing Repository - AI Agent Instructions

## Project Overview
This is a testing environment for Microsoft's MAIRA-2 multimodal transformer model that generates radiology reports from chest X-rays. The model supports three capabilities: non-grounded findings generation, grounded reports with bounding boxes, and phrase grounding.

## Architecture & Key Components

### Model Loading Pattern (Critical)
**Always use multi-GPU distribution with FP16** - the 7B parameter model requires ~14GB memory:

```python
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/maira-2",
    trust_remote_code=True,
    device_map="auto",  # Auto-distributes across GPUs
    torch_dtype=torch.float16,  # Essential for memory efficiency
)
device = torch.device("cuda:0")  # Primary device for inputs only
# Never call model.to(device) - device_map handles placement
```

**Why**: The model must be sharded across multiple GPUs (4x NVIDIA L4 with 22GB each). Using `device_map="auto"` + FP16 distributes ~3.5GB per GPU. Manual `.to(device)` causes OOM errors.

### Three Test Patterns

1. **Non-grounded reporting**: `processor.format_and_preprocess_reporting_input(..., get_grounding=False)` + `max_new_tokens=300`
2. **Grounded reporting**: Same but `get_grounding=True` + `max_new_tokens=450` (needs extra tokens for bounding boxes)
3. **Phrase grounding**: `processor.format_and_preprocess_phrase_grounding_input(...)` + `max_new_tokens=150`

See `src/test_all.py` for canonical implementations of all three patterns.

## Critical Workflows

### Environment Setup
```bash
uv venv --python 3.10           # Must use Python 3.10
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

**HuggingFace Authentication Required**:
```bash
huggingface-cli login  # Model requires accepted license agreement
```

### GPU Check Before Running Tests
```bash
cd src && python check_gpu.py  # Verifies 4 GPUs detected, shows memory
```

### Running Tests
```bash
cd src
python test_all.py                    # Runs all three capabilities
python test_findings_generation.py    # Non-grounded only
python test_grounded_generation.py    # With bounding boxes
python test_phrase_grounding.py       # Phrase localization
```

## Project-Specific Conventions

### Output Files Structure
- Test scripts auto-generate `output_*.txt` files in `src/` directory
- Grounded tests also create `output_*_visualization.png` with drawn bounding boxes
- All outputs ignored in `.gitignore` (line 68-71)

### Bounding Box Coordinates
**Critical**: Boxes are relative to cropped image MAIRA-2 sees, NOT original image:
```python
# Box format: (x_topleft, y_topleft, x_bottomright, y_bottomright)
# Values are 0-1 relative to cropped dimensions
# Use processor.adjust_box_for_original_image_size() to convert
```

### Sample Data Pattern
`src/utils.py:get_sample_data()` downloads IU-Xray public dataset images. To use custom images:
```python
sample_data = {
    "frontal": Image.open("path/to/frontal.png"),
    "lateral": Image.open("path/to/lateral.png"),  # Optional, can be None
    "indication": "Clinical indication text",
    "comparison": "None." if no prior else "Comparison text",
    "technique": "PA and lateral views of the chest.",
    "phrase": "Finding to localize"  # For phrase grounding only
}
```

## Dependencies & Version Constraints

- `transformers>=4.48.0,<4.52` - Tested with 4.51.3, **do not upgrade to 4.52+**
- `accelerate>=0.20.0` - Required for multi-GPU device_map
- `torch>=2.0.0` with CUDA support - CPU mode works but is 30x slower

## Common Pitfalls

1. **OOM Errors**: If model loading fails, ensure `device_map="auto"` and `torch_dtype=torch.float16` are set. Never use `model.to(device)` with device_map.

2. **Missing HuggingFace Token**: Model download fails without `huggingface-cli login` and accepting terms at https://huggingface.co/microsoft/maira-2

3. **Wrong max_new_tokens**: Non-grounded uses 300, grounded uses 450, phrase grounding uses 150. Using wrong values truncates output or wastes compute.

4. **Processor Methods**: Use `format_and_preprocess_reporting_input` for reports, `format_and_preprocess_phrase_grounding_input` for phrase grounding - they are NOT interchangeable.

## File Organization

```
src/
├── utils.py                    # get_sample_data() + box visualization helpers
├── test_findings_generation.py # Pattern: non-grounded reporting
├── test_grounded_generation.py # Pattern: grounded with bboxes
├── test_phrase_grounding.py    # Pattern: phrase localization
├── test_all.py                 # Runs all three sequentially
└── check_gpu.py                # GPU detection and memory check utility
```

## Research Use Only
Model is **not for clinical use**. All test scripts include this constraint. Do not suggest or implement clinical deployment features.
