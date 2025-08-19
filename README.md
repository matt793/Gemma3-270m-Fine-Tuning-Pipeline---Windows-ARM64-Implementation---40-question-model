# Gemma3 270m Fine-Tuning Pipeline - Windows ARM64 Implementation - 40 question model

A complete pipeline for fine-tuning small language models for code assistance using **LoRA** on Windows ARM64 (Copilot+ PC) with 16GB RAM. 

**IMPORTANT**: Due to Windows ARM64 limitations, this implementation uses **DistilGPT-2** (82M parameters) instead of Gemma-3-270m-IT.

**🎯 Main Output**: `fine-tuned-40-CodeTrainedQuestions.gguf` (165.5MB) - Ready for LM Studio

## Table of Contents

- [Features](#features)
- [Quick Start - ACTUAL WORKING COMMANDS](#quick-start---actual-working-commands)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Model Setup (CRITICAL STEP)](#2-model-setup-critical-step)
  - [3. Data Analysis](#3-data-analysis)
  - [4. Training (FAST - 5-10 minutes)](#4-training-fast---5-10-minutes)
  - [5. Model Conversion to GGUF](#5-model-conversion-to-gguf)
  - [6. Using Your Model](#6-using-your-model)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Training Configuration](#training-configuration)
- [Hardware Requirements](#hardware-requirements)
- [Usage Examples](#usage-examples)
- [Real Implementation Experience & Lessons Learned](#real-implementation-experience--lessons-learned)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Contributing](#contributing)  
- [License](#license)
- [Training Prompts - Test Your Model](#training-prompts---test-your-model)

## Features

- ✅ **Local-only training** - No cloud dependencies
- ✅ **Windows ARM64 compatible** - Works without bitsandbytes/quantization
- ✅ **Fast CPU training** - DistilGPT-2 trains in 5-10 minutes
- ✅ **Enhanced dataset** - 40 high-quality code samples across multiple languages
- ✅ **Complete pipeline** - Data prep, training, evaluation, inference
- ✅ **GGUF conversion** - Compatible with LM Studio and llama.cpp
- ✅ **Deterministic** - Reproducible results with pinned versions

## Quick Start - ACTUAL WORKING COMMANDS

### 1. Environment Setup

```powershell
# Setup Python environment (requires Python 3.11+)
.\env\setup.ps1 -DirectML

# Fix PowerShell execution policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### 2. Model Setup (CRITICAL STEP)

```powershell
# Check model availability - Gemma-3-270m-it not available in HF format
python scripts/setup_base_model.py --check-only

# Download working model (DistilGPT-2)
python scripts/setup_base_model.py --download --model-id distilgpt2
```

### 3. Data Analysis

```powershell
# Analyze dataset and create train/eval splits
python data/analyze_dataset.py
```

### 4. Training (FAST - 5-10 minutes)

```powershell
# Training completes quickly with DistilGPT-2
python train/train_sft.py
```

### 5. Model Conversion to GGUF

```powershell
# Merge LoRA into base model
python convert_to_gguf.py

# Install GGUF conversion dependencies
pip install mistral-common

# Clone llama.cpp tools
git clone https://github.com/ggerganov/llama.cpp

# Convert to GGUF (outputs to root folder)
python llama.cpp/convert_hf_to_gguf.py fine-tuned-distilgpt2-code --outfile fine-tuned-40-CodeTrainedQuestions.gguf
```

### 6. Using Your Model

**🎯 MAIN DELIVERABLE - In LM Studio:**
- **Load Model**: `fine-tuned-40-CodeTrainedQuestions.gguf` (165.5MB)
- **Specialized for**: The 40 specific code training question types
- **Languages**: Python, JavaScript/TypeScript, SQL, PowerShell, Power Fx, Git
- **Performance**: Fast inference, optimized for the training dataset

**Alternative - Direct Python Inference:**
```powershell
python inference/generate.py --model fine-tuned-distilgpt2-code --interactive
```

**Alternative - PEFT Adapter Format:**
```powershell
python inference/generate.py --model artifacts/ft-gemma3-270m-it-code-lora --interactive
```

## Project Structure

```
Gemma3_270m_Fine-Tune/
├── env/                    # Environment setup
│   ├── requirements.txt    # Python dependencies
│   └── setup.ps1          # Automated setup script
├── data/                   # Dataset and analysis
│   ├── gemma3_code_tutor_enhanced.jsonl  # 37 training samples
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
│   ├── convert_lora_to_gguf.ps1    # GGUF conversion
│   └── merge_lora_and_export_hf.py # Model merging
├── artifacts/              # Output models and adapters
└── README.md              # This file
```

## Dataset

The enhanced dataset contains **37 carefully curated samples** covering:

- **Python** (13 samples): Algorithms, async, pandas, testing, security
- **JavaScript/TypeScript** (8 samples): Modern patterns, React hooks, error handling
- **PowerShell** (5 samples): System administration, networking, reliability
- **SQL** (4 samples): Window functions, CTEs, performance optimization
- **Git** (2 samples): Version control workflows
- **Power Fx** (3 samples): Power Platform formulas
- **Multi-language** (2 samples): Code review, general patterns

**Difficulty Distribution:**
- Easy: 40% (15 samples)
- Medium: 54% (20 samples)  
- Hard: 5% (2 samples)

## Training Configuration

**Optimized for 16GB RAM:**
- **QLoRA 4-bit quantization** - Reduces memory usage by ~75%
- **Gradient accumulation** - Effective batch size of 8 with micro-batch size of 1
- **Gradient checkpointing** - Trades computation for memory
- **LoRA rank 16** - Good balance of efficiency and expressiveness

**Key Parameters:**
- Learning rate: 1e-4 with cosine scheduling
- Epochs: 3 (typically 200-300 steps total)
- Sequence length: 2048 tokens
- Mixed precision: bf16 (GPU) or fp16 (fallback)

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

## Usage Examples

### Training with Custom Config

```powershell
python train/train_sft.py --config train/config.yaml
```

### Evaluation with Detailed Results

```powershell
python eval/eval_small.py \
    --model artifacts/ft-gemma3-270m-it-code-lora \
    --data data/eval_split.jsonl \
    --save-results eval_results.json \
    --max-samples 10
```

### Interactive Inference

```powershell
python inference/generate.py \
    --model artifacts/ft-gemma3-270m-it-code-lora \
    --interactive \
    --temperature 0.2 \
    --max-new-tokens 512
```

### GGUF Usage in LM Studio

1. Convert LoRA to GGUF adapter (see conversion script above)
2. Open LM Studio
3. Load base model: `C:\Users\<user>\.lmstudio\models\lmstudio-community\gemma-3-270m-it-GGUF`
4. Apply LoRA adapter: `artifacts\lora-gemma3-270m-it.gguf`

## Real Implementation Experience & Lessons Learned

### ❌ **Issues Encountered on Windows ARM64**

**1. BitsAndBytes Incompatibility**
```
PackageNotFoundError: No package metadata was found for bitsandbytes
```
- **Root Cause**: bitsandbytes doesn't support Windows ARM64
- **Solution**: Automatic detection and graceful fallback to non-quantized training
- **Impact**: Uses more memory (~8GB vs 2GB) but works reliably

**2. Gemma-3-270m-IT Not Available**
```
❌ Model 'gemma-3-270m-it' not accessible: not a valid model identifier
```
- **Root Cause**: Model not published in HuggingFace format
- **Solution**: Auto-switch to DistilGPT-2 (82M params)
- **Impact**: Actually faster training and better CPU compatibility

**3. Large Model CPU Training Issues**
- **Problem**: DialoGPT-medium (355M) extremely slow on CPU (hours, no progress)
- **Solution**: Switch to DistilGPT-2 (82M) for fast CPU training
- **Result**: Training completed in 5.5 minutes!

**4. API Compatibility Issues**
```
TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'
TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'
```
- **Solution**: Updated API calls for current transformers/TRL versions
- **Fixed**: `evaluation_strategy` → `eval_strategy`, removed unsupported parameters

**5. PowerShell Execution Policy**
```
File cannot be loaded. The file is not digitally signed.
```
- **Solution**: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force`

### ✅ **What Actually Works**

**Successful Configuration:**
- **Model**: DistilGPT-2 (82M parameters)
- **Training**: LoRA without quantization (1.42% trainable params)
- **Platform**: Windows ARM64 CPU with gradient checkpointing
- **Duration**: 5-10 minutes training time
- **Output**: 165.5MB GGUF model ready for LM Studio

**Proven Commands:**
```powershell
# These commands are tested and work:
.\env\setup.ps1 -DirectML
python scripts/setup_base_model.py --download --model-id distilgpt2
python data/analyze_dataset.py
python train/train_sft.py
python convert_to_gguf.py
pip install mistral-common
git clone https://github.com/ggerganov/llama.cpp
python llama.cpp/convert_hf_to_gguf.py fine-tuned-distilgpt2-code --outfile fine-tuned-40-CodeTrainedQuestions.gguf
```

### 🎯 **Model Limitations (Important)**

**⚠️ Overfitting Warning**: The model with 40 training samples will primarily answer within the scope of those specific questions. For production use:

1. **Expand Dataset**: Need 500+ diverse samples for generalization
2. **More Epochs**: With larger dataset, train for more epochs
3. **Evaluation**: Current model best for proof-of-concept and learning the pipeline

**Current Model Best For:**
- Demonstrating the fine-tuning pipeline
- Learning LoRA techniques on ARM64
- Testing GGUF conversion workflow

## Technical Details

### Memory Optimization

The pipeline uses several techniques to fit training in 16GB RAM:

1. **QLoRA 4-bit quantization** - Base model uses ~2GB instead of 8GB
2. **LoRA adaptation** - Only trains 0.1% of parameters
3. **Gradient checkpointing** - Recompute activations to save memory
4. **Mixed precision** - Use 16-bit floats where possible

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

### Evaluation Metrics

- **Quality Score**: Heuristic based on code blocks, explanations, and keyword overlap
- **Code Execution**: Python code tested for syntax and runtime errors
- **Syntax Validation**: Basic pattern matching for JavaScript, SQL, PowerShell

## Limitations

- **Small dataset**: 37 samples may limit generalization
- **Limited languages**: Focus on Python, JS, SQL, PowerShell
- **No RAG**: Model doesn't access external documentation
- **Inference speed**: CPU inference is slower than GPU

## Contributing

To add more training samples:

1. Follow the format in `data/gemma3_code_tutor_enhanced.jsonl`
2. Include brief explanations (1-3 sentences)
3. Test with various difficulty levels
4. Run `data/analyze_dataset.py` to verify data quality

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with:

- Google's Gemma model license
- Your organization's AI usage policies
- Any applicable data protection regulations

## Citation

If you use this pipeline in your research or projects, please cite:

```
Gemma3 270m Fine-Tuning Pipeline for Code Assistance
Windows ARM64 Local Training Implementation
2025
```

## Training Prompts - Test Your Model

The `fine-tuned-40-CodeTrainedQuestions.gguf` model is specifically trained on these 40 prompts. Use these exact prompts to test the model's capabilities:

| Prompt | Difficulty | Tags | Expected Answer |
|--------|------------|------|----------------|
| Python: Write a function `is_prime(n: int) -> bool` that returns True if n is prime. Handle n<=1. Add 3 quick doctests. | Easy | algorithms, doctest | Trial division up to sqrt(n) with 6k±1 optimization, includes doctests |
| Python: Implement `two_sum(nums, target)` returning indices i,j (i<j) whose values sum to target or (-1,-1) if none. | Easy | hashmap | Single pass with hash map storing value→index |
| Python: Given a nested list like [1,[2,3],[4,[5]]], implement `flatten(iterable)` that yields items depth-first. | Medium | generators | Recursive generator using `collections.abc.Iterable`, excludes str/bytes |
| Python: Bugfix. The function should return a stable unique list preserving first occurrence order. | Easy | bugfix | Use seen set + list append instead of `set()` to preserve order |
| Python: Write `roman_to_int(s: str) -> int` for I,V,X,L,C,D,M up to 3999. Brief explanation only. | Medium | parsing | Sum values, subtract when smaller numeral precedes larger |
| Python: Add pytest tests for the function `is_anagram(a,b)` that returns True if two strings are anagrams ignoring case and spaces. Write 6 tests covering edge cases. | Easy | testing, pytest | 6 pytest functions covering case/space insensitivity, unicode, empty strings |
| JavaScript: Implement a `debounce(fn, delay)` returning a function that delays invocation until inactivity. | Easy | patterns | Reset timer on each call, invoke on trailing edge with clearTimeout |
| TypeScript: Implement a typed `EventEmitter<T extends Record<string, any[]>>`. Methods: on, off, emit. | Medium | events | Map event→Set<listener> with generic type inference per event key |
| PowerShell: Write `Get-TopCpuProcess -Top 5` that lists the top N processes by CPU with Name, Id, CPU. | Easy | admin | `Get-Process | Sort-Object CPU -Descending | Select-Object -First $Top` |
| PowerShell: Bugfix. This should restart a service with retries (make it robust: stop if running, start, retry up to 3 times with 2s delay). | Medium | reliability | Try/catch loop with service status verification and retry logic |
| Power Fx: Show the user photo for the current user in an Image control. Keep it simple. | Easy | powerapps | `Office365Users.UserPhotoV2(Office365Users.MyProfileV2().id)` |
| Power Fx: Filter a People directory gallery by a TextInput 'txtSearch'. If blank, show top 999. Otherwise search by query. | Easy | office365users, filter | Conditional `Office365Users.SearchUser` with blank check |
| SQL: Given table `orders(order_id, customer_id, amount, created_at)`, return each customer's latest order row. | Medium | window-functions | ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_at DESC) |
| Python: Code review: The function below parses CSV lines but fails on quoted commas. Suggest a safer approach and provide a corrected Python implementation using the stdlib. | Easy | code-review | Use `csv.reader(StringIO(line))` instead of `line.split(',')` |
| Python: TDD: Write pytest tests first for a new function `slugify(text: str) -> str` that lowercases, replaces non-alphanumerics with '-', squeezes repeats, and strips leading/trailing '-'. Provide 6 tests. | Medium | tdd, tests | 6 test functions covering spaces, punctuation, unicode, repeats, edges |
| Python: FIM: Complete the missing middle to implement a Python median. | Easy | fim | Check parity: odd→middle element, even→average of two middle |
| JavaScript: JavaScript bugfix: The `once` helper should call the function at most once. Fix it and add a small test snippet. | Easy | bugfix, testing | Cache result, only call when not called before + test assertions |
| SQL: A slow query filters `WHERE email LIKE '%@example.com'`. Suggest an index strategy and rewrite (PostgreSQL). | Medium | postgres, indexing | CREATE INDEX on `lower(email)`, use `lower(email) LIKE '%@example.com'` |
| Git: Provide commands to create a new branch `feature/lora-finetune`, commit all changes with message 'WIP: LoRA config', and push to origin. | Easy | cli | `git checkout -b`, `git add -A`, `git commit -m`, `git push -u origin` |
| Python: Implement `safe_join(base: str, *paths: str) -> str` that prevents path traversal. Raise ValueError on escape. | Medium | security | Use `Path.resolve()` and verify final path is within base directory |
| Power Fx: In a Gallery of SharePoint list 'Tickets', show only items where Status = 'Open' and the text box 'txtAssignee' appears in the 'AssignedTo.DisplayName'. | Easy | filter | `Filter(Tickets, Status = "Open" && txtAssignee.Text in AssignedTo.DisplayName)` |
| JavaScript: Implement a `throttle(fn, interval)` that invokes at most once per interval (trailing edge only). | Medium | patterns | Track timestamp and timeout, gate calls with interval checking |
| Python: Implement async `fetch_urls(urls: List[str]) -> List[str]` that fetches URLs concurrently with aiohttp. Handle errors gracefully. | Medium | async, aiohttp | `aiohttp.ClientSession` with `asyncio.gather`, return error strings for failures |
| Python: Write a context manager `timer()` that prints elapsed time when exiting. Use it to time a sleep(2). | Easy | context-manager | `@contextmanager` with `time.perf_counter()` in try/finally |
| Python: Implement `validate_email(email: str) -> bool` using regex. Allow common formats but reject obviously invalid ones. | Easy | regex, validation | Basic regex pattern `^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$` |
| Python: Create a `retry(max_attempts=3, delay=1)` decorator that retries on exceptions. Show usage on a flaky function. | Medium | decorator, error-handling | Decorator with loop, sleep on failure, re-raise final exception |
| Python: Implement a simple LRU cache `LRUCache(capacity)` with get/put methods using OrderedDict. | Medium | data-structures, lru | `OrderedDict` with `move_to_end()` and `popitem(last=False)` |
| Python: Write a pandas function `analyze_sales(df)` that returns total sales by category and month. Assume columns: 'date', 'category', 'amount'. | Medium | pandas, data-analysis | Convert date, extract month period, groupby + pivot with fillna(0) |
| TypeScript: Create a `useLocalStorage<T>(key, initialValue)` React hook that syncs state with localStorage. | Medium | react, hooks | `useState` + `useEffect` with JSON serialization and error handling |
| JavaScript: Implement Promise-based `sleep(ms)` function and use it in an async function that retries with exponential backoff. | Medium | async, patterns | `setTimeout` Promise + retry loop with `delay *= 2` |
| PowerShell: Write `Backup-Files -Source C:\\Data -Destination D:\\Backup -ExcludePatterns @('*.tmp', 'logs')` with progress and error handling. | Medium | backup, robocopy | `robocopy` with exclude args, progress tracking, and exit code handling |
| SQL: Write a query to find the second highest salary from `employees(id, name, salary, department_id)` table. | Easy | ranking, window-functions | `DENSE_RANK() OVER (ORDER BY salary DESC)` with `WHERE rank = 2` |
| Python: Implement `binary_search(arr: List[int], target: int) -> int` returning index or -1. Include a simple test. | Easy | algorithms, binary-search | Classic left/right pointers with `mid = (left + right) // 2` |
| JavaScript: Create a `deepClone(obj)` function that handles nested objects, arrays, dates, and circular references. | Hard | deep-clone, algorithms | `WeakMap` for visited objects, handle Date/Array/Object types recursively |
| Python: Write `hash_password(password: str) -> str` and `verify_password(password: str, hashed: str) -> bool` using bcrypt. | Easy | security, bcrypt | `bcrypt.gensalt()` + `bcrypt.hashpw()` and `bcrypt.checkpw()` |
| PowerShell: Create `Test-PortConnectivity -ComputerName server01 -Port 443 -Timeout 5` that tests TCP connectivity. | Medium | networking, connectivity | `Test-NetConnection` or `TcpClient.BeginConnect` fallback |
| SQL: Create a CTE to find customers who made orders in consecutive months. Table: `orders(customer_id, order_date)`. | Hard | cte, window-functions | CTE with `DATE_TRUNC` + `LAG()` to check `prev_month + INTERVAL '1 month'` |
| Git: Show commands to squash the last 3 commits into one with a new message 'feat: implement user authentication'. | Easy | rebase, squash | `git rebase -i HEAD~3` or `git reset --soft HEAD~3` + new commit |
| TypeScript: Create a generic `Result<T, E>` type for error handling without exceptions. Include helper functions. | Medium | error-handling, generics | Discriminated union with `Ok/Err` helpers and `isOk/isErr` type guards |
| Power Fx: Create a formula for a Gallery that shows items from SharePoint list 'Tasks' filtered by current user and sorted by priority (High, Medium, Low). | Medium | sharepoint, sorting | `SortByColumns(Filter(Tasks, AssignedTo.Email = User().Email), "Priority", Switch(...))` |

### Usage Tips for Testing

**Best Results**: Use prompts exactly as listed above - the model is specifically trained on these formats.

**Example Test Commands:**
```powershell
# Load your model in LM Studio with: fine-tuned-40-CodeTrainedQuestions.gguf
# Then try these prompts:

"Python: Write a function `is_prime(n: int) -> bool` that returns True if n is prime. Handle n<=1. Add 3 quick doctests."

"JavaScript: Implement a `debounce(fn, delay)` returning a function that delays invocation until inactivity."

"SQL: Given table `orders(order_id, customer_id, amount, created_at)`, return each customer's latest order row."
```

---

**Version**: 1.0  
**Last Updated**: 2025-08-19  
**Tested On**: Windows 11 ARM64, Python 3.11+, 16GB RAM  
**Model Output**: `fine-tuned-40-CodeTrainedQuestions.gguf` (165.5MB)
