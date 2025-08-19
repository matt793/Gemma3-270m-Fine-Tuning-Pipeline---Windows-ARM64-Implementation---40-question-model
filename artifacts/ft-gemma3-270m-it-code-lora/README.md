---
base_model: distilgpt2
library_name: peft
model_name: ft-gemma3-270m-it-code-lora
tags:
- base_model:adapter:distilgpt2
- lora
- sft
- transformers
- trl
licence: license
pipeline_tag: text-generation
---

# Model Card for ft-gemma3-270m-it-code-lora

This model is a fine-tuned version of [distilgpt2](https://huggingface.co/distilgpt2).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.17.0
- TRL: 0.21.0
- Transformers: 4.55.2
- Pytorch: 2.8.0
- Datasets: 4.0.0
- Tokenizers: 0.21.4

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```