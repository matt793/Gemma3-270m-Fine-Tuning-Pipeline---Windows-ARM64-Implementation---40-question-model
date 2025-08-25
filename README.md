# Gemma3 270m Fine-Tuning Pipeline - Windows ARM64 Implementation

A complete pipeline for fine-tuning small language models for code assistance using **LoRA** on Windows ARM64 (Copilot+ PC) with 16GB RAM.

**✅ Main Output**: `artifacts/gemma3-270m-it-code-ft.gguf`

## Table of Contents

- [Features](#features)
- [Quick Start - Working Commands](#quick-start---working-commands)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Analysis](#2-data-analysis)
  - [3. Training](#3-training)
  - [4. Model Conversion to GGUF](#4-model-conversion-to-gguf)
  - [5. Using Your Model](#5-using-your-model)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [Hardware Requirements](#hardware-requirements)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Features

- ✅ **Local-only training** - No cloud dependencies
- ✅ **Windows ARM64 compatible** - Works without bitsandbytes/quantization
- ✅ **CPU training** - Fine-tunes Gemma 3 270M on a 16GB RAM machine
- ✅ **Enhanced dataset** - 40 high-quality code samples across multiple languages
- ✅ **Complete pipeline** - Data prep, training, evaluation, inference
- ✅ **GGUF conversion** - Compatible with LM Studio and llama.cpp
- ✅ **Deterministic** - Reproducible results with pinned versions

## Quick Start - Working Commands

### 1. Environment Setup

```powershell
# Setup Python environment (requires Python 3.11+)
.\env\setup.ps1 -DirectML

# Fix PowerShell execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### 2. Data Analysis

```powershell
# Analyze dataset and create train/eval splits
python data/analyze_dataset.py
```

### 3. Training

```powershell
# Training Gemma 3 270M
python train/train_sft.py
```

### 4. Model Conversion to GGUF

```powershell
# Merge LoRA into base model
python scripts/merge_lora_and_export_hf.py --base google/gemma-3-270m-it --peft artifacts/ft-gemma3-270m-it-code-lora --out artifacts/merged-gemma3-270m-it-code

# Install GGUF conversion dependencies
pip install mistral-common

# Clone llama.cpp tools
git clone https://github.com/ggerganov/llama.cpp

# Convert to GGUF (outputs to artifacts directory)
python llama.cpp/convert_hf_to_gguf.py artifacts/merged-gemma3-270m-it-code --outfile artifacts/gemma3-270m-it-code-ft.gguf
```

### 5. Using Your Model

**In LM Studio:**
- **Load Model**: `artifacts/gemma3-270m-it-code-ft.gguf`

**Alternative Testing:**
```powershell
python inference/generate.py --model artifacts/merged-gemma3-270m-it-code --interactive
```

## Project Structure

```
Gemma3_270m_Fine-Tune/
├── env/                    # Environment setup
│   ├── requirements.txt    # Python dependencies
│   └── setup.ps1          # Automated setup script
├── data/                   # Dataset and analysis
│   ├── gemma3_code_tutor_enhanced.jsonl  # 40 training samples
│   ├── README_DATASET.md   # Dataset documentation
│   └── analyze_dataset.py  # Data analysis script
├── train/                  # Training pipeline
│   ├── config.yaml        # Training configuration
│   └── train_sft.py       # SFT training script
├── eval/                   # Evaluation
│   └── eval_small.py      # Evaluation script
├── inference/              # Inference scripts
│   └── generate.py        # Interactive & batch inference
├── scripts/                # Utilities
│   ├── merge_lora_and_export_hf.py # Model merging
├── artifacts/              # Output models and adapters
└── README.md              # This file
```

## Dataset

The enhanced dataset contains **40 carefully curated samples** covering:

- **Python** (17 samples): Algorithms, async, pandas, testing, security
- **JavaScript/TypeScript** (8 samples): Modern patterns, React hooks, error handling
- **PowerShell** (5 samples): System administration, networking, reliability
- **SQL** (4 samples): Window functions, CTEs, performance optimization
- **Git** (2 samples): Version control workflows
- **Power Fx** (3 samples): Power Platform formulas
- **Multi-language** (1 sample): Code review

## Training Configuration

**Optimized for 16GB RAM:**
- **LoRA without quantization** - bitsandbytes unavailable on Windows ARM64
- **Gradient accumulation** - Effective batch size of 8 with micro-batch size of 1
- **Gradient checkpointing** - Trades computation for memory
- **LoRA rank 16** - Good balance of efficiency and expressiveness

**Key Parameters:**
- Learning rate: 1e-4 with cosine scheduling
- Epochs: 3 (15 steps total)
- Sequence length: 2048 tokens
- Mixed precision: fp16 (CPU fallback)

## Hardware Requirements

**Minimum:**
- Windows 10/11 on ARM64 (Copilot+ PC recommended)
- 16GB RAM
- 10GB free disk space
- Python 3.11+

**Recommended:**
- Qualcomm Snapdragon X series processor
- DirectML-compatible GPU
- 32GB RAM (for faster training)
- SSD storage

## Technical Details

### Memory Optimization

The pipeline uses several techniques to fit training in 16GB RAM:

1. **LoRA adaptation** - Only trains 1.39% of parameters
2. **Gradient checkpointing** - Trades computation for memory
3. **Mixed precision** - Use fp16 where possible on CPU

### Chat Template

The model uses a custom chat template optimized for code assistance:

```
<|system|>
You are a concise, reliable coding assistant...
<|end|>
<|user|>
Python: write is_prime function
<|end|>
<|assistant|>
**Explanation (brief):** Trial division up to sqrt(n)...
<|end|>
```

## Contributing

1. **Replace base model** with a code-capable model (StarCoder, CodeT5, etc.)
2. **Expand dataset** to 500+ diverse code samples
3. **Test thoroughly** before claiming functionality
4. **Use appropriate evaluation** specific to code generation

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with:

- Google's Gemma model license
- Your organization's AI usage policies
- Any applicable data protection regulations

---

**Version**: 2.0
**Last Updated**: 2025-08-24
**Tested On**: Windows 11 ARM64, Python 3.11+, 16GB RAM
**Model Status**: ✅ **SUCCESS** - Fine-tuned Gemma 3 270M