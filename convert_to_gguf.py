#!/usr/bin/env python
"""
Simple GGUF conversion script for the fine-tuned model.
Merges LoRA into base model and creates a merged model for GGUF conversion.

Usage:
    python convert_to_gguf.py
"""

import os
import sys
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def merge_lora_model():
    """Merge LoRA adapter back into base model."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        logging.error(f"Required packages not available: {e}")
        sys.exit(1)
    
    logging.info("=== GGUF Conversion: Merge LoRA + Base Model ===")
    
    # Paths
    peft_path = "artifacts/ft-gemma3-270m-it-code-lora"
    base_model_id = "distilgpt2"
    output_path = "fine-tuned-distilgpt2-code"
    
    if not Path(peft_path).exists():
        logging.error(f"PEFT model not found at: {peft_path}")
        logging.error("Run training first: python train/train_sft.py")
        sys.exit(1)
    
    logging.info(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map=None  # CPU only
    )
    
    logging.info(f"Loading LoRA adapter: {peft_path}")
    model = PeftModel.from_pretrained(base_model, peft_path)
    
    logging.info("Merging LoRA into base model...")
    merged_model = model.merge_and_unload()
    
    logging.info(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.save_pretrained(output_path)
    
    logging.info("✅ Merged model saved successfully!")
    logging.info(f"📁 Location: {Path(output_path).absolute()}")
    
    logging.info("\n🔧 Next steps for GGUF conversion:")
    logging.info("1. Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp")
    logging.info("2. Run conversion:")
    logging.info(f"   python llama.cpp/convert-hf-to-gguf.py {output_path} --outfile fine-tuned-distilgpt2-code.gguf")
    
    return output_path

def main():
    setup_logging()
    
    try:
        merged_path = merge_lora_model()
        logging.info("\n🎉 Conversion preparation complete!")
        logging.info(f"Merged model ready at: {merged_path}")
        
    except Exception as e:
        logging.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
