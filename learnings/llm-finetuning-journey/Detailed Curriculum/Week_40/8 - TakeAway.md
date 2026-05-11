# Week 40 TakeAway — Phase 4 Gate

Phase 4 is complete when your 7B model is on the Hub, your harness runs, and your checklist passes.

---

## Phase 4 Deliverable Summary

| Deliverable | Location | Target metric |
|---|---|---|
| Full SFT run | GitHub / W&B | val loss < 2.5 |
| LoRA from scratch | GitHub | merge() produces coherent output |
| LoRA via peft | GitHub | rank sweep r∈{8,16,32,64} documented |
| QLoRA 7B fine-tune | GitHub / W&B | training completes without OOM |
| Unsloth QLoRA | GitHub | 2x+ speed vs. vanilla QLoRA |
| 15K domain dataset | HuggingFace Datasets | train 14500 / val 500 splits |
| Eval harness | GitHub | exec success %, exec correct % on 100 examples |
| v1 adapter on Hub | HuggingFace Hub | exec correctness ≥ 60% |

---

## Key Code Patterns

**Push adapter + tokenizer to Hub:**
```python
model.push_to_hub("your-handle/postgres-sqlcoder-7b-v1")
tokenizer.push_to_hub("your-handle/postgres-sqlcoder-7b-v1")
```

**Load adapter from Hub (fresh session):**
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B", quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "your-handle/postgres-sqlcoder-7b-v1")
```

**Push dataset to Hub:**
```python
from datasets import DatasetDict, Dataset
import json

ds = DatasetDict({
    "train": Dataset.from_list([json.loads(l) for l in open("train_15k.jsonl")]),
    "validation": Dataset.from_list([json.loads(l) for l in open("val_500.jsonl")]),
})
ds.push_to_hub("your-handle/postgres-text2sql-15k")
```

---

## Decision Rules

- If exec correctness < 60%: do NOT proceed to Phase 5 — re-run Week 38 QLoRA with more epochs or better data before entering GRPO training.
- If any of the 5 minimum gate items fail: fix them now; Phase 5 assumes all are working.
- If exec correctness > exact match by more than 15 points: your model is generating valid alternate SQL — this is healthy; use exec correctness as your headline number.
- If held-out test set size is under 100: expand it before Phase 5 — GRPO reward signal calibration requires a reliable held-out benchmark.

---

## Numbers to Remember (Phase 4 Summary)

- LoRA default: rank=16, alpha=32, all 7 modules (q/k/v/o/gate/up/down)
- QLoRA: NF4 + double_quant + BF16 compute + paged_adamw_8bit + LR=2e-4
- Unsloth: `use_gradient_checkpointing="unsloth"` (NOT `gradient_checkpointing=True`)
- DoRA: `use_dora=True` in LoraConfig; RSLoRA: `use_rslora=True`
- Dataset: 14,500 train / 500 val / 100 held-out (never in training)
- Expected exec correctness baseline for v1 model: 60–80%
- Phase 5 target (after GRPO): 80–88%

---

## Red Flags

- Adapter loads but generates empty SQL: the prompt format used for inference differs from training — check your `### SQL:` stop token.
- Hub model card has no prompt format section: any downstream user (including Phase 5 you) will load the model and get garbage because they use the wrong template.
- `held_out_test.json` pushed to HuggingFace: test set contamination — remove it immediately and retrain or acknowledge the contamination in your model card.
- Phase 5 planned without a working eval harness: GRPO requires the harness as its reward function; starting Phase 5 without it means you cannot train.
