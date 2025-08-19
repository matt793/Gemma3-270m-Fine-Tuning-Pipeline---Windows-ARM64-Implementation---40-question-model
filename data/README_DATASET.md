# Gemma3 Code Tutor (mini) — Instructional Dataset
Date: 2025-08-19

This small, curated dataset targets **coding assistance behaviors** suitable for fine-tuning a ~**270M** parameter model via **LoRA/QLoRA** on a local Copilot+ PC (16 GB RAM).

## Design Principles
- **No chain-of-thought**: responses include code and a **brief** explanation only.
- **Production-leaning**: emphasizes correctness, small functions, tests, and input validation.
- **Multi-language**: Python, JS/TS, PowerShell, Power Fx, SQL, Git/CLI.
- **Task variety**: algorithms, bug fixes, refactors, tests, FIM (fill-in-the-middle), code review, security, and simple DevOps tasks.

## Format
Chat-style JSONL, one object per line:
```json
{
  "messages": [{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
  "tags": ["python","algorithms"],
  "difficulty": "easy"
}
```
Use with Hugging Face `trl` SFT fine-tuning or any chat-instruction pipeline.

## Suggested LoRA/QLoRA Settings (Transformers + PEFT + TRL)
- base model: `gemma-3-270m` (or your local ID)
- sequence length: 2048
- optimizer: AdamW (betas 0.9/0.95), weight decay 0.01
- learning rate: 1e-4 to 2e-4 (small model)
- lr scheduler: cosine with 100-500 warmup steps
- epochs: 3–5 (monitor eval; stop early on plateau)
- batch size: 8–16 **effective** (use gradient accumulation; QLoRA 4-bit to fit memory)
- LoRA: r=16, alpha=32, dropout=0.05; target modules per Gemma attention/MLP
- bf16 if supported; else fp16; on Windows ARM use CPU/DirectML fallback
- eval every ~200–500 steps using a held-out split of this dataset plus a few unseen tasks

## Windows ARM (Copilot+) Notes
- Prefer Python 3.11+
- If GPU acceleration is available: `pip install torch-directml` and set device to `dml`
- Otherwise use CPU with small batch sizes and gradient accumulation
- Consider using WSL2 only if your setup supports it on ARM

## Fine-tuning Skeleton (TRL SFTTrainer)
```python
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model

model_id = "gemma-3-270m"  # replace with your local checkpoint
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

ds = load_dataset("json", data_files={"train":"gemma3_code_tutor.jsonl"})

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = AutoModelForCausalLM.from_pretrained(model_id)
model = get_peft_model(model, lora)

args = TrainingArguments(
    output_dir="ft-gemma3-270m-code",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=False, fp16=True
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    dataset_text_field=None, # using chat format; wrap with a template or collator as needed
    tokenizer=tok,
    max_seq_length=2048,
)

trainer.train()
model.save_pretrained("ft-gemma3-270m-code-lora")
```

## Licensing & Safety
- Content is original and intentionally compact. Review for org policies before release.
- Avoid training on private secrets or licensed code without permission.

—
