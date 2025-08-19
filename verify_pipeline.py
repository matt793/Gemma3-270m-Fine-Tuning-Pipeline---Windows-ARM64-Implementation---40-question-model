#!/usr/bin/env python
"""
Verification script for Gemma3 270m fine-tuning pipeline.
Checks all components and dependencies before training.

Usage:
    python verify_pipeline.py [--full]
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import importlib.util

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} - Need 3.11+"

def check_file_structure() -> Tuple[bool, List[str]]:
    """Verify project file structure."""
    required_files = [
        "env/requirements.txt",
        "env/setup.ps1", 
        "data/gemma3_code_tutor_enhanced.jsonl",
        "data/README_DATASET.md",
        "data/analyze_dataset.py",
        "train/config.yaml",
        "train/train_sft.py",
        "eval/eval_small.py",
        "inference/generate.py",
        "scripts/convert_lora_to_gguf.ps1",
        "scripts/merge_lora_and_export_hf.py",
        "README.md"
    ]
    
    required_dirs = [
        "env", "data", "train", "eval", "inference", "scripts", "artifacts"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    success = len(missing_files) == 0 and len(missing_dirs) == 0
    messages = []
    
    if missing_dirs:
        messages.append(f"Missing directories: {', '.join(missing_dirs)}")
    if missing_files:
        messages.append(f"Missing files: {', '.join(missing_files)}")
    if success:
        messages.append(f"All {len(required_files)} files and {len(required_dirs)} directories found ✓")
    
    return success, messages

def check_package_imports() -> Tuple[bool, List[str]]:
    """Check if required packages can be imported."""
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("datasets", "Datasets"),
        ("safetensors", "SafeTensors"),
        ("accelerate", "Accelerate"),
        ("jsonlines", "JSONLines"),
        ("yaml", "PyYAML"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn"),
        ("tqdm", "TQDM"),
        ("colorama", "Colorama")
    ]
    
    results = []
    all_success = True
    
    for module_name, display_name in packages:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                results.append(f"{display_name} ✓")
            else:
                results.append(f"{display_name} ✗ (not found)")
                all_success = False
        except Exception as e:
            results.append(f"{display_name} ✗ ({str(e)[:50]})")
            all_success = False
    
    return all_success, results

def check_dataset_integrity() -> Tuple[bool, Dict]:
    """Verify dataset format and content."""
    dataset_path = "data/gemma3_code_tutor_enhanced.jsonl"
    
    if not Path(dataset_path).exists():
        return False, {"error": "Dataset file not found"}
    
    try:
        import jsonlines
        
        samples = []
        with jsonlines.open(dataset_path) as reader:
            for obj in reader:
                samples.append(obj)
        
        # Basic validation
        total_samples = len(samples)
        valid_samples = 0
        languages = set()
        difficulties = set()
        
        for sample in samples:
            if all(key in sample for key in ['messages', 'tags', 'difficulty']):
                valid_samples += 1
                languages.update(sample['tags'])
                difficulties.add(sample['difficulty'])
        
        return True, {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "unique_tags": len(languages),
            "difficulties": list(difficulties),
            "success": valid_samples == total_samples
        }
    
    except Exception as e:
        return False, {"error": str(e)}

def check_device_availability() -> Tuple[bool, List[str]]:
    """Check available compute devices."""
    results = []
    has_acceleration = False
    
    try:
        import torch
        results.append(f"PyTorch version: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            results.append(f"CUDA available: {device_name} ✓")
            has_acceleration = True
        else:
            results.append("CUDA: Not available")
        
        # Check DirectML
        try:
            import torch_directml
            if torch_directml.is_available():
                device = torch_directml.device()
                results.append(f"DirectML available: {device} ✓")
                has_acceleration = True
            else:
                results.append("DirectML: Not available")
        except ImportError:
            results.append("DirectML: Not installed")
        
        # Check MPS (Apple)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            results.append("MPS available ✓")
            has_acceleration = True
        else:
            results.append("MPS: Not available")
        
        if not has_acceleration:
            results.append("⚠️ No GPU acceleration available - will use CPU")
        else:
            results.append("✓ GPU acceleration available")
    
    except ImportError:
        results.append("❌ PyTorch not available")
        return False, results
    
    return True, results

def check_base_model_availability() -> Tuple[bool, str]:
    """Check if base model is available."""
    model_id = "gemma-3-270m-it"
    
    # Check local cache first
    cache_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface" / "transformers"
    ]
    
    for cache_path in cache_paths:
        if cache_path.exists():
            for model_dir in cache_path.iterdir():
                if model_id in model_dir.name:
                    return True, f"Base model found in cache: {model_dir} ✓"
    
    # Check if available online (without downloading)
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        return True, f"Base model '{model_id}' available online ✓"
    except Exception as e:
        return False, f"Base model '{model_id}' not accessible: {str(e)[:100]}"

def run_quick_training_test() -> Tuple[bool, str]:
    """Run a quick training dry run."""
    try:
        result = subprocess.run([
            sys.executable, "train/train_sft.py", "--dry-run"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return True, "Training dry-run completed successfully ✓"
        else:
            return False, f"Training dry-run failed: {result.stderr[:200]}"
    
    except subprocess.TimeoutExpired:
        return False, "Training dry-run timed out"
    except Exception as e:
        return False, f"Training test failed: {str(e)[:100]}"

def print_section(title: str, success: bool, messages: List[str]):
    """Print a verification section."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"\n{title}: {status}")
    print("-" * (len(title) + len(status) + 2))
    
    for message in messages:
        print(f"  {message}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify Gemma3 pipeline")
    parser.add_argument('--full', action='store_true', help='Run full verification including training test')
    args = parser.parse_args()
    
    print("=== Gemma3 270m Fine-tuning Pipeline Verification ===")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check Python version
    success, message = check_python_version()
    print_section("Python Version", success, [message])
    
    # Check file structure
    success, messages = check_file_structure()
    print_section("File Structure", success, messages)
    
    # Check package imports
    success, messages = check_package_imports()
    print_section("Package Dependencies", success, messages)
    
    # Check dataset
    success, info = check_dataset_integrity()
    if success:
        messages = [
            f"Total samples: {info['total_samples']}",
            f"Valid samples: {info['valid_samples']}",
            f"Unique tags: {info['unique_tags']}",
            f"Difficulties: {', '.join(info['difficulties'])}"
        ]
    else:
        messages = [f"Error: {info.get('error', 'Unknown error')}"]
    print_section("Dataset Integrity", success, messages)
    
    # Check device availability
    success, messages = check_device_availability()
    print_section("Device Availability", success, messages)
    
    # Check base model
    success, message = check_base_model_availability()
    print_section("Base Model", success, [message])
    
    # Full verification
    if args.full:
        print("\n🔍 Running full verification (this may take a minute)...")
        success, message = run_quick_training_test()
        print_section("Training Pipeline", success, [message])
    
    print(f"\n{'='*60}")
    print("Verification complete!")
    
    if args.full:
        print("\nNext steps:")
        print("1. Run: python data/analyze_dataset.py")
        print("2. Run: python train/train_sft.py")
        print("3. Run: python eval/eval_small.py --model artifacts/ft-gemma3-270m-it-code-lora --data data/eval_split.jsonl")
        print("4. Run: python inference/generate.py --model artifacts/ft-gemma3-270m-it-code-lora --interactive")
    else:
        print("\nFor full verification, run: python verify_pipeline.py --full")

if __name__ == "__main__":
    main()
