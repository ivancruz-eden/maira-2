# MAIRA-2 Testing Repository

A comprehensive testing environment for Microsoft's MAIRA-2 model - a multimodal transformer designed for grounded radiology report generation from chest X-rays.

## About MAIRA-2

**MAIRA-2** is a state-of-the-art model that can:

1. **Generate radiology findings** from chest X-rays (with optional lateral views, prior studies, and clinical context)
2. **Produce grounded reports** where findings are accompanied by bounding box annotations
3. **Perform phrase grounding** to localize specific findings in chest X-ray images

### Model Architecture

- **Image Encoder**: RAD-DINO-MAIRA-2 (frozen)
- **Language Model**: Vicuna-7b-v1.5 (fully fine-tuned)
- **Projection Layer**: Trained from scratch
- **Parameters**: ~7B
- **License**: Microsoft Research License Agreement (MSRLA)

### Important Notes

⚠️ **Research Use Only**: MAIRA-2 is intended for research purposes only and should NOT be used in clinical decision-making or any clinical use.

🔐 **Model Access**: You need to accept the model's terms and conditions on HuggingFace before downloading:
   - Visit: https://huggingface.co/microsoft/maira-2
   - Log in to your HuggingFace account
   - Accept the disclaimer to access the model

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM (32GB+ recommended)
- ~15GB disk space for model weights
- HuggingFace account with accepted MAIRA-2 license

## Setup Instructions

### 1. Install UV (Python Package Manager)

If you don't have `uv` installed:

```bash
# On Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone or Navigate to This Repository

```bash
cd /path/to/maira-2
```

### 3. Create Python Virtual Environment

Using `uv`, create and activate a virtual environment:

```bash
# Create virtual environment with Python 3.10
uv venv --python 3.10

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
# Install all required packages
uv pip install -r requirements.txt
```

This will install:
- `pillow` - Image processing
- `protobuf` - Protocol buffers
- `sentencepiece` - Text tokenization
- `torch` - PyTorch deep learning framework
- `transformers` (v4.48-4.51) - HuggingFace transformers library
- `requests` - HTTP library for downloading sample data
- `accelerate` - Multi-GPU and distributed training support

### 5. Authenticate with HuggingFace

The model requires authentication to download. Log in using the HuggingFace CLI:

```bash
# Install HuggingFace CLI if needed
uv pip install huggingface-hub

# Login to HuggingFace
huggingface-cli login
```

Enter your HuggingFace token when prompted. You can create a token at: https://huggingface.co/settings/tokens

### 6. Verify CUDA/GPU Setup (Optional but Recommended)

Check if PyTorch can detect your GPU:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

If CUDA is not available and you have a compatible GPU, you may need to install the CUDA-enabled PyTorch:

```bash
# For CUDA 11.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Model Weights and Data

### Automatic Downloads

The test scripts automatically handle:

✅ **Model Weights**: Downloaded automatically from HuggingFace on first run (~15GB)
- Cached in `~/.cache/huggingface/hub/models--microsoft--maira-2/`
- Subsequent runs will use cached weights

✅ **Sample Data**: Test scripts download example chest X-rays from IU X-ray dataset
- Public domain medical images (CC license)
- Not used in MAIRA-2 training
- Downloaded on-demand during test execution

### Manual Model Download (Optional)

If you prefer to download the model manually:

```bash
huggingface-cli download microsoft/maira-2 --local-dir ./models/maira-2
```

Then modify test scripts to load from local path:
```python
model = AutoModelForCausalLM.from_pretrained(
    "./models/maira-2",  # Local path
    trust_remote_code=True
)
```

## Usage

### Multi-GPU Support

The test scripts automatically detect and utilize multiple GPUs if available. The model will be distributed across all available GPUs using `device_map="auto"` and FP16 precision to minimize memory usage.

**Benefits:**
- Distributes 7B parameter model across multiple GPUs
- Reduces per-GPU memory requirements
- Faster inference than CPU
- Automatic load balancing

To verify GPU detection and memory:
```bash
cd src
python check_gpu.py
```

### Test Scripts

The repository includes four test scripts in the `src/` directory:

#### 1. Test Non-Grounded Findings Generation

Generates a radiology findings report without spatial annotations:

```bash
cd src
python test_findings_generation.py
```

**Output**: `output_non_grounded.txt`

#### 2. Test Grounded Report Generation

Generates findings with bounding box annotations:

```bash
cd src
python test_grounded_generation.py
```

**Output**: 
- `output_grounded.txt` - Report with box coordinates
- `output_grounded_visualization.png` - Image with boxes drawn

#### 3. Test Phrase Grounding

Localizes a specific finding (e.g., "Pleural effusion") in the image:

```bash
cd src
python test_phrase_grounding.py
```

**Output**:
- `output_phrase_grounding.txt` - Grounding result
- `output_phrase_grounding_visualization.png` - Annotated image

#### 4. Run All Tests

Execute all three capabilities sequentially:

```bash
cd src
python test_all.py
```

This comprehensive test runs all use cases and saves all outputs.

### Using Your Own Images

To test with your own chest X-ray images, modify the `get_sample_data()` function in `src/utils.py`:

```python
def get_sample_data() -> Dict[str, Image.Image | str]:
    from PIL import Image
    
    # Load your own images
    frontal_image = Image.open("/path/to/frontal_xray.png")
    lateral_image = Image.open("/path/to/lateral_xray.png")  # Optional
    
    sample_data = {
        "frontal": frontal_image,
        "lateral": lateral_image,  # or None if not available
        "indication": "Your clinical indication here",
        "comparison": "Comparison text or 'None.'",
        "technique": "PA and lateral views of the chest.",
        "phrase": "Your phrase to ground"  # For phrase grounding
    }
    return sample_data
```

## Expected Performance

### Inference Time (approximate)

| Hardware | Non-Grounded | Grounded | Phrase Grounding |
|----------|-------------|----------|------------------|
| NVIDIA A100 (40GB) | ~10-15s | ~20-30s | ~5-10s |
| NVIDIA RTX 3090 (24GB) | ~20-30s | ~40-60s | ~10-15s |
| CPU (32GB RAM) | ~5-10min | ~10-15min | ~2-5min |

### Memory Requirements

- **GPU**: Minimum 12GB VRAM (16GB+ recommended)
- **CPU**: Minimum 16GB RAM (32GB+ recommended)

## Understanding the Output

### Non-Grounded Report

Plain text describing findings:

```
There is a large right pleural effusion with associated right basilar 
atelectasis. The left lung is clear. No pneumothorax is identified...
```

### Grounded Report

Findings with bounding box coordinates:

```python
[
    ('There is a large right pleural effusion.', [(0.055, 0.275, 0.445, 0.665)]),
    ('The left lung is clear.', None),
    ('No pneumothorax is identified.', None),
    ...
]
```

**Bounding Box Format**: `(x_topleft, y_topleft, x_bottomright, y_bottomright)`

⚠️ **Important**: Coordinates are relative to the **cropped image** that MAIRA-2 processes, not the original image. Use `processor.adjust_box_for_original_image_size()` to convert to original image coordinates.

### Phrase Grounding

```python
('Pleural effusion.', [(0.025, 0.345, 0.425, 0.575)])
```

The model repeats the phrase and provides bounding box(es) localizing the finding.

## Troubleshooting

### Issue: "You need to agree to share your contact information"

**Solution**: Visit https://huggingface.co/microsoft/maira-2, log in, and accept the model's terms.

### Issue: CUDA out of memory

**Solutions**:
1. Close other GPU applications
2. Reduce batch size (scripts use batch_size=1 by default)
3. Use CPU mode (slower but works):
   ```python
   device = torch.device("cpu")
   ```

### Issue: Model download is slow or fails

**Solutions**:
1. Check your internet connection
2. Use HuggingFace mirrors if available
3. Download manually and use local path (see "Manual Model Download" above)

### Issue: Import errors with transformers

**Solution**: Ensure you have the correct transformers version:
```bash
uv pip install "transformers>=4.48.0,<4.52"
```

MAIRA-2 has been tested with transformers v4.51.3.

### Issue: Sample images fail to download

**Solution**: The test scripts download from openi.nlm.nih.gov. If blocked:
1. Check firewall/proxy settings
2. Download images manually from IU X-ray dataset
3. Use your own chest X-ray images (see "Using Your Own Images")

## Project Structure

```
maira-2/
├── src/
│   ├── utils.py                       # Utility functions
│   ├── test_findings_generation.py    # Test non-grounded reporting
│   ├── test_grounded_generation.py    # Test grounded reporting
│   ├── test_phrase_grounding.py       # Test phrase grounding
│   └── test_all.py                    # Comprehensive test suite
├── .gitignore                         # Git ignore rules
├── .python-version                    # Python version specification
├── pyproject.toml                     # Project configuration (uv)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Citation

If you use MAIRA-2 in your research, please cite:

```bibtex
@article{Bannur2024MAIRA2GR,
  title={MAIRA-2: Grounded Radiology Report Generation},
  author={Shruthi Bannur and Kenza Bouzid and Daniel C. Castro and Anton Schwaighofer 
          and Anja Thieme and Sam Bond-Taylor and Maximilian Ilse and 
          Fernando P\'{e}rez-Garc\'{i}a and Valentina Salvatelli and Harshita Sharma 
          and Felix Meissen and Mercy Prasanna Ranjit and Shaury Srivastav and 
          Julia Gong and Noel C. F. Codella and Fabian Falck and Ozan Oktay and 
          Matthew P. Lungren and Maria T. A. Wetscherek and Javier Alvarez-Valle and 
          Stephanie L. Hyland},
  journal={arXiv},
  year={2024},
  volume={abs/2406.04449},
  url={https://arxiv.org/abs/2406.04449}
}
```

## Resources

- **Paper**: [MAIRA-2: Grounded Radiology Report Generation](https://arxiv.org/abs/2406.04449)
- **Model Card**: https://huggingface.co/microsoft/maira-2
- **Image Encoder**: https://huggingface.co/microsoft/rad-dino-maira-2
- **Base LLM**: https://huggingface.co/lmsys/vicuna-7b-v1.5

## License

This repository code is provided as-is for research purposes. The MAIRA-2 model is governed by the [Microsoft Research License Agreement](https://huggingface.co/microsoft/maira-2/blob/main/LICENSE).

## Support

For issues with:
- **This repository**: Open an issue in this repo
- **MAIRA-2 model**: Contact Microsoft Research:
  - Stephanie Hyland (stephanie.hyland@microsoft.com)
  - Shruthi Bannur (shruthi.bannur@microsoft.com)
- **HuggingFace/Transformers**: Visit HuggingFace support forums

---

**Last Updated**: November 2024  
**Model Version**: MAIRA-2  
**Tested with**: transformers v4.51.3, Python 3.10
