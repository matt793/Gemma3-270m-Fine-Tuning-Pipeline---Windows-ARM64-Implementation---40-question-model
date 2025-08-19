#!/usr/bin/env python
"""
Merge a PEFT LoRA adapter into a base HF model and save a merged HF checkpoint.

Usage:
  python merge_lora_and_export_hf.py --base <HF_BASE_DIR_OR_ID> --peft <PEFT_DIR> --out <OUT_DIR>

Notes:
- Works well for small models like gemma-3-270m-it on CPU. For larger models, consider 8-bit/4-bit loading.
- After merging, you can run llama.cpp's convert-hf-to-gguf.py to produce a GGUF file.
"""

import argparse, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="HF base model dir or model id (e.g., gemma-3-270m-it)")
    ap.add_argument("--peft", required=True, help="Path to PEFT LoRA adapter dir (contains adapter_model.safetensors)")
    ap.add_argument("--out",  required=True, help="Output HF dir for merged model")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as e:
        print("ERROR: Required packages not found. Install:", file=sys.stderr)
        print("  pip install 'transformers>=4.42' 'peft>=0.11' 'safetensors' 'accelerate' ", file=sys.stderr)
        raise

    # Prefer float16 to reduce memory when possible; fall back to float32 on CPU if needed
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    print(f"Loading PEFT adapter: {args.peft}")
    peft_model = PeftModel.from_pretrained(base, args.peft)

    print("Merging LoRA into base (this may take a minute)...")
    merged = peft_model.merge_and_unload()  # applies LoRA weights into the base and drops adapters

    print(f"Saving merged model to: {args.out}")
    merged.save_pretrained(args.out)
    print("Saving tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tok.save_pretrained(args.out)

    print("Done. You can now convert HF->GGUF using llama.cpp's convert-hf-to-gguf.py.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
