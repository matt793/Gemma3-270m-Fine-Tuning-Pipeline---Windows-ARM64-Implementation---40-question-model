#!/usr/bin/env python
"""
Evaluation script for fine-tuned Gemma3 270m code assistant.
Tests on held-out evaluation set and runs lightweight code validation.

Usage:
    python eval/eval_small.py --model artifacts/ft-gemma3-270m-it-code-lora --data data/eval_split.jsonl
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def load_model_and_tokenizer(model_path: str):
    """Load fine-tuned model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        import torch
    except ImportError as e:
        logging.error(f"Required packages not available: {e}")
        sys.exit(1)
    
    logging.info(f"Loading model from: {model_path}")
    
    # Extract base model from config
    config_path = Path(model_path) / "training_config.yaml"
    base_model_id = "gemma-3-270m-it"  # Default fallback
    
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
            base_model_id = config['model']['base_model_id']
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load PEFT model
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    logging.info("Model loaded successfully")
    return model, tokenizer

def load_eval_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset."""
    import jsonlines
    
    samples = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            samples.append(obj)
    
    logging.info(f"Loaded {len(samples)} evaluation samples")
    return samples

def format_prompt(messages: List[Dict[str, str]]) -> str:
    """Format messages into prompt format."""
    formatted = ""
    for message in messages[:-1]:  # Exclude assistant message
        role = message['role']
        content = message['content']
        
        if role == 'system':
            formatted += f"<|system|>\n{content}\n<|end|>\n"
        elif role == 'user':
            formatted += f"<|user|>\n{content}\n<|end|>\n"
    
    formatted += "<|assistant|>\n"
    return formatted

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate response from model."""
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            attention_mask=inputs.attention_mask.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up response - remove end token if present
    response = response.replace('<|end|>', '').strip()
    return response

def extract_code_from_response(response: str, language: str) -> Optional[str]:
    """Extract code block from model response."""
    # Look for code blocks with language specification
    pattern = rf'```{language}\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return matches[0].strip()
    
    # Fallback - look for any code block
    pattern = r'```\s*\n(.*?)\n```'
    matches = re.findall(pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    
    return None

def test_python_code(code: str) -> Tuple[bool, str]:
    """Test Python code execution."""
    if not code:
        return False, "No code found"
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Try to run the code
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Clean up
        os.unlink(temp_file)
        
        if result.returncode == 0:
            return True, "Code executed successfully"
        else:
            return False, f"Runtime error: {result.stderr[:200]}"
    
    except subprocess.TimeoutExpired:
        return False, "Code execution timeout"
    except Exception as e:
        return False, f"Test error: {str(e)[:200]}"

def validate_javascript_syntax(code: str) -> Tuple[bool, str]:
    """Basic JavaScript syntax validation."""
    if not code:
        return False, "No code found"
    
    # Basic syntax checks
    if code.count('{') != code.count('}'):
        return False, "Mismatched braces"
    
    if code.count('(') != code.count(')'):
        return False, "Mismatched parentheses"
    
    # Look for common patterns
    if 'function' in code or '=>' in code or 'const' in code or 'let' in code:
        return True, "Valid JavaScript patterns found"
    
    return False, "No recognizable JavaScript patterns"

def validate_sql_syntax(code: str) -> Tuple[bool, str]:
    """Basic SQL syntax validation."""
    if not code:
        return False, "No code found"
    
    sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH']
    code_upper = code.upper()
    
    if any(keyword in code_upper for keyword in sql_keywords):
        return True, "Valid SQL keywords found"
    
    return False, "No recognizable SQL patterns"

def validate_powershell_syntax(code: str) -> Tuple[bool, str]:
    """Basic PowerShell syntax validation."""
    if not code:
        return False, "No code found"
    
    ps_patterns = ['param(', 'function ', '$', 'Get-', 'Set-', 'New-', 'Write-Host']
    
    if any(pattern in code for pattern in ps_patterns):
        return True, "Valid PowerShell patterns found"
    
    return False, "No recognizable PowerShell patterns"

def evaluate_code_quality(generated: str, expected: str) -> float:
    """Evaluate code quality with simple heuristics."""
    score = 0.0
    
    # Check if code block is present
    if '```' in generated:
        score += 0.3
    
    # Check if explanation is present
    if any(word in generated.lower() for word in ['explanation', 'brief', 'use', 'implement']):
        score += 0.2
    
    # Check for key elements from expected response
    expected_words = set(expected.lower().split())
    generated_words = set(generated.lower().split())
    
    # Simple word overlap
    overlap = len(expected_words & generated_words)
    total_unique = len(expected_words | generated_words)
    
    if total_unique > 0:
        overlap_score = overlap / total_unique
        score += 0.5 * overlap_score
    
    return min(score, 1.0)

def run_evaluation(model, tokenizer, eval_samples: List[Dict], max_samples: Optional[int] = None):
    """Run evaluation on samples."""
    if max_samples:
        eval_samples = eval_samples[:max_samples]
    
    results = {
        'total_samples': len(eval_samples),
        'by_language': {},
        'by_difficulty': {},
        'code_execution': {'passed': 0, 'failed': 0},
        'quality_scores': [],
        'details': []
    }
    
    for i, sample in enumerate(eval_samples):
        logging.info(f"Evaluating sample {i+1}/{len(eval_samples)}")
        
        messages = sample['messages']
        tags = sample.get('tags', [])
        difficulty = sample.get('difficulty', 'unknown')
        
        # Determine primary language
        language = 'unknown'
        for tag in tags:
            if tag in ['python', 'javascript', 'typescript', 'sql', 'powershell', 'powerfx']:
                language = tag
                break
        
        # Generate response
        prompt = format_prompt(messages)
        generated = generate_response(model, tokenizer, prompt)
        expected = messages[-1]['content']  # Assistant's expected response
        
        # Evaluate quality
        quality_score = evaluate_code_quality(generated, expected)
        results['quality_scores'].append(quality_score)
        
        # Code execution test (Python only for now)
        execution_passed = False
        execution_msg = "Not tested"
        
        if language == 'python':
            code = extract_code_from_response(generated, 'python')
            if code:
                execution_passed, execution_msg = test_python_code(code)
                if execution_passed:
                    results['code_execution']['passed'] += 1
                else:
                    results['code_execution']['failed'] += 1
        elif language == 'javascript':
            code = extract_code_from_response(generated, 'javascript')
            if code:
                execution_passed, execution_msg = validate_javascript_syntax(code)
        elif language == 'sql':
            code = extract_code_from_response(generated, 'sql')
            if code:
                execution_passed, execution_msg = validate_sql_syntax(code)
        elif language == 'powershell':
            code = extract_code_from_response(generated, 'powershell')
            if code:
                execution_passed, execution_msg = validate_powershell_syntax(code)
        
        # Update counters
        if language not in results['by_language']:
            results['by_language'][language] = {'count': 0, 'quality_sum': 0}
        results['by_language'][language]['count'] += 1
        results['by_language'][language]['quality_sum'] += quality_score
        
        if difficulty not in results['by_difficulty']:
            results['by_difficulty'][difficulty] = {'count': 0, 'quality_sum': 0}
        results['by_difficulty'][difficulty]['count'] += 1
        results['by_difficulty'][difficulty]['quality_sum'] += quality_score
        
        # Store detailed results
        results['details'].append({
            'sample_id': i,
            'language': language,
            'difficulty': difficulty,
            'quality_score': quality_score,
            'execution_passed': execution_passed,
            'execution_msg': execution_msg,
            'prompt_length': len(prompt),
            'response_length': len(generated)
        })
    
    return results

def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Overall metrics
    avg_quality = sum(results['quality_scores']) / len(results['quality_scores']) * 100
    print(f"Total Samples: {results['total_samples']}")
    print(f"Average Quality Score: {avg_quality:.1f}%")
    
    # Code execution results
    total_exec = results['code_execution']['passed'] + results['code_execution']['failed']
    if total_exec > 0:
        exec_rate = results['code_execution']['passed'] / total_exec * 100
        print(f"Code Execution Pass Rate: {exec_rate:.1f}% ({results['code_execution']['passed']}/{total_exec})")
    
    # By language
    print(f"\nBy Language:")
    for lang, stats in results['by_language'].items():
        avg_score = stats['quality_sum'] / stats['count'] * 100
        print(f"  {lang:12}: {stats['count']:2d} samples, {avg_score:.1f}% avg quality")
    
    # By difficulty
    print(f"\nBy Difficulty:")
    for diff, stats in results['by_difficulty'].items():
        avg_score = stats['quality_sum'] / stats['count'] * 100
        print(f"  {diff:8}: {stats['count']:2d} samples, {avg_score:.1f}% avg quality")
    
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Gemma3 270m")
    parser.add_argument('--model', required=True, help='Path to fine-tuned model')
    parser.add_argument('--data', required=True, help='Path to evaluation dataset')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to evaluate')
    parser.add_argument('--save-results', help='Save detailed results to JSON file')
    args = parser.parse_args()
    
    setup_logging()
    
    # Load model and data
    model, tokenizer = load_model_and_tokenizer(args.model)
    eval_samples = load_eval_dataset(args.data)
    
    # Run evaluation
    results = run_evaluation(model, tokenizer, eval_samples, args.max_samples)
    
    # Print results
    print_results(results)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to: {args.save_results}")

if __name__ == "__main__":
    main()
