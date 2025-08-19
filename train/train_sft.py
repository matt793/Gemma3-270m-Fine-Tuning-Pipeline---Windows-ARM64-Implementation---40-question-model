#!/usr/bin/env python
"""
Gemma3 270m LoRA Fine-tuning Script
Optimized for Windows ARM64 with 16GB RAM using QLoRA

Usage:
    python train/train_sft.py [--config train/config.yaml] [--dry-run]
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import torch
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging(log_level: str = "info") -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def detect_device() -> str:
    """Detect best available device (DirectML > CUDA > MPS > CPU)."""
    try:
        # Check for DirectML (Windows ARM64)
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            logging.info(f"Using DirectML device: {device}")
            return "dml"
    except ImportError:
        pass
    
    # Check for CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logging.info(f"Using CUDA device: {device_name}")
        return "cuda"
    
    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logging.info("Using MPS device")
        return "mps"
    
    # Fallback to CPU
    logging.info("Using CPU device")
    return "cpu"

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_device_and_precision(config: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Setup device-specific configuration."""
    training_config = config['training'].copy()
    
    # Auto-detect mixed precision capability
    if device in ["cuda", "dml"]:
        # Try bfloat16 first, then float16
        try:
            if torch.cuda.is_bf16_supported() if device == "cuda" else True:
                training_config['bf16'] = True
                training_config['fp16'] = False
                logging.info("Using bfloat16 mixed precision")
            else:
                training_config['bf16'] = False
                training_config['fp16'] = True
                logging.info("Using float16 mixed precision")
        except:
            training_config['bf16'] = False
            training_config['fp16'] = True
            logging.info("Using float16 mixed precision (fallback)")
    else:
        # CPU training - no mixed precision
        training_config['bf16'] = False
        training_config['fp16'] = False
        logging.info("Using float32 (CPU mode)")
    
    return training_config

def prepare_model_and_tokenizer(config: Dict[str, Any], device: str):
    """Load and prepare model with LoRA configuration."""
    try:
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer
        )
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logging.error(f"Required packages not available: {e}")
        logging.error("Run: pip install transformers peft trl datasets accelerate")
        sys.exit(1)
    
    model_id = config['model']['base_model_id']
    logging.info(f"Loading base model: {model_id}")
    
    # Setup quantization config for memory efficiency (Windows ARM64 compatible)
    quantization_config = None
    use_quantization = config['quantization']['load_in_4bit']
    
    if use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config['quantization']['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type']
            )
            logging.info("Using 4-bit quantization (QLoRA)")
        except (ImportError, Exception) as e:
            logging.warning(f"BitsAndBytes not available on this platform: {e}")
            logging.warning("Disabling quantization - will use more memory but should work")
            quantization_config = None
            use_quantization = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=config['model']['trust_remote_code'],
        use_fast=True
    )
    
    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("Set pad_token to eos_token")
    
    # Load model
    model_kwargs = {
        'trust_remote_code': config['model']['trust_remote_code'],
        'torch_dtype': 'auto',
        'device_map': 'auto' if device != 'cpu' else None,
    }
    
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Setup LoRA
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        target_modules=config['lora']['target_modules'],
        lora_dropout=config['lora']['lora_dropout'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def load_dataset(config: Dict[str, Any]):
    """Load and prepare training dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("datasets package not available. Run: pip install datasets")
        sys.exit(1)
    
    train_file = config['data']['train_file']
    eval_file = config['data'].get('eval_file')
    
    logging.info(f"Loading training data from: {train_file}")
    
    # Load dataset files
    data_files = {'train': train_file}
    if eval_file and Path(eval_file).exists():
        data_files['validation'] = eval_file
        logging.info(f"Loading validation data from: {eval_file}")
    
    dataset = load_dataset('json', data_files=data_files)
    
    # Create validation split if not provided
    if 'validation' not in dataset:
        split_ratio = config['data'].get('validation_split', 0.1)
        split_dataset = dataset['train'].train_test_split(test_size=split_ratio, seed=42)
        dataset['train'] = split_dataset['train']
        dataset['validation'] = split_dataset['test']
        logging.info(f"Created validation split ({split_ratio:.1%})")
    
    logging.info(f"Dataset sizes - Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")
    return dataset

def format_chat_sample(sample: Dict[str, Any]) -> str:
    """Format a single sample using chat template."""
    messages = sample['messages']
    formatted = ""
    
    for message in messages:
        role = message['role']
        content = message['content']
        
        if role == 'system':
            formatted += f"<|system|>\n{content}\n<|end|>\n"
        elif role == 'user':
            formatted += f"<|user|>\n{content}\n<|end|>\n"
        elif role == 'assistant':
            formatted += f"<|assistant|>\n{content}\n<|end|>\n"
    
    return formatted

def prepare_dataset_for_training(dataset, tokenizer, max_length: int = 2048):
    """Prepare dataset with chat formatting."""
    def formatting_prompts_func(examples):
        texts = []
        for i in range(len(examples['messages'])):
            sample = {'messages': examples['messages'][i]}
            text = format_chat_sample(sample)
            texts.append(text)
        return {'text': texts}
    
    # Apply formatting
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Gemma3 270m LoRA Fine-tuning")
    parser.add_argument('--config', default='train/config.yaml', help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true', help='Run setup without training')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config['system']['log_level'])
    
    logging.info("=== Gemma3 270m LoRA Fine-tuning ===")
    logging.info(f"Config: {args.config}")
    logging.info(f"Torch version: {torch.__version__}")
    
    # Detect device
    device = detect_device()
    
    # Update training config for device
    training_config = setup_device_and_precision(config, device)
    
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(config, device)
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Prepare dataset for training
    formatted_dataset = prepare_dataset_for_training(
        dataset, tokenizer, config['training']['max_seq_length']
    )
    
    if args.dry_run:
        logging.info("=== DRY RUN COMPLETE ===")
        logging.info("Model and dataset loaded successfully")
        logging.info("Ready for training!")
        return
    
    # Setup training
    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        logging.error("TRL not available. Run: pip install trl")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(training_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=training_config['overwrite_output_dir'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        adam_beta1=training_config['adam_beta1'],
        adam_beta2=training_config['adam_beta2'],
        adam_epsilon=training_config['adam_epsilon'],
        max_grad_norm=training_config['max_grad_norm'],
        num_train_epochs=training_config['num_train_epochs'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        eval_strategy=training_config['evaluation_strategy'],
        eval_steps=training_config['eval_steps'],
        save_strategy=training_config['save_strategy'],
        save_steps=training_config['save_steps'],
        save_total_limit=training_config['save_total_limit'],
        gradient_checkpointing=training_config['gradient_checkpointing'],
        fp16=training_config['fp16'],
        bf16=training_config['bf16'],
        remove_unused_columns=training_config['remove_unused_columns'],
        seed=training_config['seed'],
        data_seed=training_config['data_seed'],
        report_to=config['system']['report_to'],
        logging_dir=str(output_dir / 'logs'),
        disable_tqdm=config['system']['disable_tqdm']
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset['train'],
        eval_dataset=formatted_dataset['validation']
    )
    
    # Start training
    logging.info("=== Starting Training ===")
    try:
        trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        logging.info("=== Training Complete ===")
        logging.info(f"Model saved to: {output_dir}")
        
        # Save training config for reference
        with open(output_dir / 'training_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
