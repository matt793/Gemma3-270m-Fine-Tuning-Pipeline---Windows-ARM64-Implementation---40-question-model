#!/usr/bin/env python
"""
Inference script for fine-tuned Gemma3 270m code assistant.
Supports interactive chat and batch processing.

Usage:
    python inference/generate.py --model artifacts/ft-gemma3-270m-it-code-lora --prompt "Python: write is_prime function"
    python inference/generate.py --model artifacts/ft-gemma3-270m-it-code-lora --interactive
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load fine-tuned model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        logging.error(f"Required packages not available: {e}")
        logging.error("Run: pip install transformers peft")
        sys.exit(1)
    
    logging.info(f"Loading model from: {model_path}")
    
    # Check if this is a merged model (direct HF format) or PEFT adapter
    model_path_obj = Path(model_path)
    
    # If it's a merged model directory, load directly
    if (model_path_obj / "config.json").exists() and not (model_path_obj / "adapter_config.json").exists():
        logging.info("Loading merged model directly")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model directly
        device_map = "auto" if device == "auto" else None
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        model.eval()
        
        # Move to specific device if requested
        if device not in ["auto", "cpu"] and not device_map:
            model = model.to(device)
        
        logging.info(f"Merged model loaded successfully on device: {model.device}")
        return model, tokenizer
    
    # Otherwise, it's a PEFT adapter - load with base model
    else:
        logging.info("Loading PEFT adapter with base model")
        
        # Extract base model from config if available
        config_path = model_path_obj / "training_config.yaml"
        base_model_id = "distilgpt2"  # Default fallback
        
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
                base_model_id = config['model']['base_model_id']
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        device_map = "auto" if device == "auto" else None
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        # Move to specific device if requested
        if device not in ["auto", "cpu"] and not device_map:
            model = model.to(device)
        
        logging.info(f"PEFT model loaded successfully on device: {model.device}")
        return model, tokenizer

def format_prompt(user_prompt: str, system_prompt: Optional[str] = None) -> str:
    """Format user prompt with system context."""
    if system_prompt is None:
        system_prompt = "You are a concise, reliable coding assistant. Respond with correct, production-ready code and a brief explanation. Do not include chain-of-thought. Prefer small, composable functions, type hints when helpful, input validation, edge cases, and unit tests when asked."
    
    formatted = f"<|system|>\n{system_prompt}\n<|end|>\n"
    formatted += f"<|user|>\n{user_prompt}\n<|end|>\n"
    formatted += "<|assistant|>\n"
    
    return formatted

def generate_response(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1
) -> str:
    """Generate response from model."""
    
    # Tokenize input
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=1536
    )
    
    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up response
    response = response.replace('<|end|>', '').strip()
    
    # Stop at end token or double newlines (common end pattern)
    if '<|end|>' in response:
        response = response.split('<|end|>')[0]
    
    return response

def interactive_chat(model, tokenizer, args):
    """Run interactive chat session."""
    print("=== Gemma3 Code Assistant - Interactive Mode ===")
    print("Type your coding questions. Use 'quit', 'exit', or Ctrl+C to stop.")
    print("Commands:")
    print("  /temp <float>   - Set temperature (default: 0.1)")
    print("  /tokens <int>   - Set max new tokens (default: 512)")
    print("  /system <text>  - Set custom system prompt")
    print("  /reset          - Reset to default system prompt")
    print("  /help           - Show this help")
    print("-" * 60)
    
    # Default settings
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    system_prompt = None
    
    conversation_history = []
    
    try:
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split(' ', 1)
                    command = parts[0].lower()
                    
                    if command == '/help':
                        print("Commands:")
                        print("  /temp <float>   - Set temperature")
                        print("  /tokens <int>   - Set max new tokens")
                        print("  /system <text>  - Set custom system prompt")
                        print("  /reset          - Reset to default system prompt")
                        print("  /help           - Show this help")
                        continue
                    
                    elif command == '/temp':
                        if len(parts) > 1:
                            try:
                                temperature = float(parts[1])
                                print(f"Temperature set to {temperature}")
                            except ValueError:
                                print("Invalid temperature value")
                        else:
                            print(f"Current temperature: {temperature}")
                        continue
                    
                    elif command == '/tokens':
                        if len(parts) > 1:
                            try:
                                max_new_tokens = int(parts[1])
                                print(f"Max new tokens set to {max_new_tokens}")
                            except ValueError:
                                print("Invalid token count")
                        else:
                            print(f"Current max tokens: {max_new_tokens}")
                        continue
                    
                    elif command == '/system':
                        if len(parts) > 1:
                            system_prompt = parts[1]
                            print("Custom system prompt set")
                        else:
                            print(f"Current system prompt: {system_prompt or 'default'}")
                        continue
                    
                    elif command == '/reset':
                        system_prompt = None
                        print("System prompt reset to default")
                        continue
                    
                    else:
                        print(f"Unknown command: {command}")
                        continue
                
                # Generate response
                prompt = format_prompt(user_input, system_prompt)
                
                print("\nGenerating response...")
                response = generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty
                )
                
                print(f"\nAssistant:\n{response}")
                
                # Store conversation
                conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")

def single_prompt_mode(model, tokenizer, args):
    """Process single prompt and exit."""
    prompt = format_prompt(args.prompt, args.system_prompt)
    
    print("Generating response...")
    response = generate_response(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty
    )
    
    print(f"\nPrompt: {args.prompt}")
    print(f"\nResponse:\n{response}")

def main():
    parser = argparse.ArgumentParser(description="Gemma3 Code Assistant Inference")
    parser.add_argument('--model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--prompt', help='Single prompt to process')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--system-prompt', help='Custom system prompt')
    
    # Generation parameters
    parser.add_argument('--max-new-tokens', type=int, default=512, help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty')
    
    # Device settings
    parser.add_argument('--device', default='auto', help='Device to use (auto, cpu, cuda, etc.)')
    
    args = parser.parse_args()
    
    if not args.prompt and not args.interactive:
        parser.error("Must specify either --prompt or --interactive")
    
    setup_logging()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    # Run appropriate mode
    if args.interactive:
        interactive_chat(model, tokenizer, args)
    else:
        single_prompt_mode(model, tokenizer, args)

if __name__ == "__main__":
    main()
