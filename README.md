# hugging-face

A collection of Hugging Face inference examples using both the Inference API and local model execution.

## Installation

This installation assumes you are working on Windows. I'm using Windows so that I can run models locally against my 3080ti.
Yes, I know. I could use Linux but unfortunately NVIDIA drivers suck and so here I am.

### Prerequisites
- Python 3.14+ (managed via `uv`)
- NVIDIA GPU with CUDA support (tested on RTX 3080 Ti)
- NVIDIA drivers with CUDA 12.6+ support

### Setup

1. Create a virtual environment:
```bash
uv venv
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Verification

Verify GPU is detected:
```bash
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Usage

### Inference API Example
Uses Hugging Face's hosted inference API:
```bash
uv run python inference.py
```

### Local Model Example
Runs models locally on your GPU:
```bash
uv run python script.py
```

## Files

- `inference.py` - Example using Hugging Face Inference API
- `script.py` - Example running models locally with transformers pipeline
- `requirements.txt` - Project dependencies

## Notes

- The project uses `uv` for Python package management
- PyTorch is configured with CUDA 12.6 for GPU acceleration
- Local models are cached in `~/.cache/huggingface/`
