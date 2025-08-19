#!/usr/bin/env python
"""
Analyze the enhanced Gemma3 code tutor dataset.
Shows distribution by tags, difficulty, and creates train/eval splits.
"""

import json
import jsonlines
from collections import Counter, defaultdict
from pathlib import Path
import random

def analyze_dataset(dataset_path: str):
    """Analyze dataset composition and create splits."""
    
    samples = []
    with jsonlines.open(dataset_path, 'r') as reader:
        for obj in reader:
            samples.append(obj)
    
    print(f"=== Dataset Analysis ===")
    print(f"Total samples: {len(samples)}")
    print()
    
    # Analyze tags
    tag_counts = Counter()
    difficulty_counts = Counter()
    
    for sample in samples:
        tags = sample.get('tags', [])
        difficulty = sample.get('difficulty', 'unknown')
        
        for tag in tags:
            tag_counts[tag] += 1
        difficulty_counts[difficulty] += 1
    
    print("=== Tag Distribution ===")
    for tag, count in tag_counts.most_common():
        print(f"{tag:20} : {count:3d}")
    print()
    
    print("=== Difficulty Distribution ===")
    for difficulty, count in difficulty_counts.most_common():
        print(f"{difficulty:10} : {count:3d}")
    print()
    
    # Language analysis
    language_mapping = {
        'python': ['python', 'pytest', 'pandas', 'async', 'aiohttp', 'bcrypt'],
        'javascript': ['javascript', 'patterns'],
        'typescript': ['typescript', 'events', 'react', 'hooks', 'generics'],
        'powershell': ['powershell', 'admin', 'reliability', 'backup', 'robocopy', 'networking'],
        'sql': ['sql', 'window-functions', 'postgres', 'indexing', 'ranking', 'cte'],
        'powerfx': ['powerfx', 'powerapps', 'office365users', 'filter', 'sharepoint'],
        'git': ['git', 'cli', 'rebase', 'squash']
    }
    
    language_counts = defaultdict(int)
    for sample in samples:
        tags = sample.get('tags', [])
        for lang, lang_tags in language_mapping.items():
            if any(tag in lang_tags for tag in tags):
                language_counts[lang] += 1
                break
    
    print("=== Language Distribution ===")
    for lang, count in sorted(language_counts.items()):
        print(f"{lang:12} : {count:3d}")
    print()
    
    # Create deterministic train/eval split (90/10)
    random.seed(42)  # Deterministic split
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    split_point = int(len(shuffled) * 0.9)
    train_samples = shuffled[:split_point]
    eval_samples = shuffled[split_point:]
    
    print(f"=== Data Split (seed=42) ===")
    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples:  {len(eval_samples)}")
    
    # Save splits
    base_path = Path(dataset_path).parent
    
    with jsonlines.open(base_path / 'train_split.jsonl', 'w') as writer:
        for sample in train_samples:
            writer.write(sample)
    
    with jsonlines.open(base_path / 'eval_split.jsonl', 'w') as writer:
        for sample in eval_samples:
            writer.write(sample)
    
    print(f"\n✅ Splits saved:")
    print(f"  - {base_path / 'train_split.jsonl'}")
    print(f"  - {base_path / 'eval_split.jsonl'}")
    
    return {
        'total': len(samples),
        'train': len(train_samples),
        'eval': len(eval_samples),
        'tags': dict(tag_counts),
        'difficulties': dict(difficulty_counts),
        'languages': dict(language_counts)
    }

if __name__ == "__main__":
    stats = analyze_dataset("data/gemma3_code_tutor_enhanced.jsonl")
