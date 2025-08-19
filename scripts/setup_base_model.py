#!/usr/bin/env python
"""
Helper script to setup the base model for fine-tuning.
Handles downloading or locating the HuggingFace format model needed for training.

Usage:
    python scripts/setup_base_model.py [--download] [--check-only]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def check_hf_model_availability(model_id: str) -> tuple[bool, str]:
    """Check if HuggingFace model is available locally or online."""
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Try to load config (doesn't download model weights)
        try:
            config = AutoConfig.from_pretrained(model_id, local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            return True, f"✅ Model '{model_id}' found locally in HuggingFace cache"
        except:
            # Not in local cache, try online
            try:
                config = AutoConfig.from_pretrained(model_id)
                return True, f"✅ Model '{model_id}' available online for download"
            except Exception as e:
                return False, f"❌ Model '{model_id}' not accessible: {str(e)}"
    
    except ImportError:
        return False, "❌ transformers not installed. Run: pip install transformers"

def download_hf_model(model_id: str) -> bool:
    """Download HuggingFace model for training."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logging.info(f"Downloading model: {model_id}")
        logging.info("This may take several minutes depending on your connection...")
        
        # Download tokenizer first (smaller)
        logging.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Download model
        logging.info("Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        logging.info("✅ Model downloaded successfully!")
        logging.info(f"Model cached in: {Path.home() / '.cache' / 'huggingface'}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to download model: {e}")
        return False

def check_gguf_files():
    """Check for GGUF files in current directory."""
    gguf_files = list(Path(".").glob("*.gguf"))
    
    if gguf_files:
        logging.info("📁 GGUF files found in current directory:")
        for file in gguf_files:
            logging.info(f"  - {file.name}")
        
        logging.info("\n💡 Note: GGUF files are for inference only.")
        logging.info("Training requires HuggingFace format models.")
    else:
        logging.info("No GGUF files found in current directory")

def suggest_alternatives():
    """Suggest alternative approaches."""
    logging.info("\n🔧 Alternative approaches:")
    logging.info("1. Use a different Gemma model that's available in HF format:")
    logging.info("   - google/gemma-2b-it")
    logging.info("   - google/gemma-7b-it")
    logging.info("   - microsoft/DialoGPT-medium (smaller alternative)")
    
    logging.info("\n2. Convert GGUF to HuggingFace format (advanced):")
    logging.info("   - Use llama.cpp tools to convert back to HF format")
    logging.info("   - This requires additional tools and may be complex")
    
    logging.info("\n3. Train on a similar small model:")
    logging.info("   - microsoft/DialoGPT-small (117M parameters)")
    logging.info("   - distilgpt2 (82M parameters)")

def main():
    parser = argparse.ArgumentParser(description="Setup base model for fine-tuning")
    parser.add_argument('--download', action='store_true', 
                       help='Download the model if available online')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check availability without downloading')
    parser.add_argument('--model-id', default='gemma-3-270m-it',
                       help='Model ID to check/download')
    args = parser.parse_args()
    
    setup_logging()
    
    logging.info("=== Gemma3 Base Model Setup ===")
    logging.info(f"Target model: {args.model_id}")
    
    # Check GGUF files
    check_gguf_files()
    
    # Check HuggingFace model availability
    available, message = check_hf_model_availability(args.model_id)
    logging.info(f"\n{message}")
    
    if not available:
        logging.warning(f"\n⚠️ Model '{args.model_id}' not available in HuggingFace format")
        suggest_alternatives()
        
        # Update config to use an alternative model
        logging.info("\n🔧 Updating config.yaml to use DialoGPT-medium as fallback...")
        try:
            import yaml
            config_path = Path("train/config.yaml")
            
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                
                config['model']['base_model_id'] = 'microsoft/DialoGPT-medium'
                
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                logging.info("✅ Updated config to use microsoft/DialoGPT-medium")
                logging.info("This is a 355M parameter conversational model that should work well")
            else:
                logging.warning("Config file not found")
                
        except Exception as e:
            logging.error(f"Failed to update config: {e}")
        
        return
    
    if args.check_only:
        logging.info("Check complete. Use --download to download the model.")
        return
    
    if args.download:
        if "available online" in message:
            success = download_hf_model(args.model_id)
            if success:
                logging.info("\n✅ Model setup complete! You can now run training.")
                logging.info("Next step: python train/train_sft.py --dry-run")
        else:
            logging.info("Model already available locally. No download needed.")
    else:
        if "available online" in message:
            logging.info("\nUse --download to download the model for training.")
        else:
            logging.info("\n✅ Model ready for training!")

if __name__ == "__main__":
    main()
