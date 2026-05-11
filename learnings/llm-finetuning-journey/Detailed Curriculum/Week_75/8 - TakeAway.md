# Week 75 TakeAway — Iteration: Different Base Models

**This week in 15 words:** A controlled four-model comparison tells you which base model best fits your pipeline and data.

---

## Key Code Pattern — Chat Template Verification

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
sample = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Translate to SQL: count orders by region"},
     {"role": "assistant", "content": "SELECT region, COUNT(*) FROM orders GROUP BY region;"}],
    tokenize=False
)
print(sample)  # Inspect boundaries manually before training
```

---

## Key Code Pattern — Response Template for Loss Masking

```python
# Must match EXACTLY what apply_chat_template produces for each model
RESPONSE_TEMPLATES = {
    "qwen":   "<|im_start|>assistant\n",
    "llama":  "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "gemma":  "<start_of_turn>model\n",
    "deepseek": "<|Assistant|>",
}
```

---

## Decision Rules

If validation loss diverges for one model but not others → check chat template alignment first, not learning rate.

If mean EM improvement >= 2 pp but std dev > mean improvement → collect more seeds before switching.

If a model wins on complex queries but loses on domain-specific functions → data coverage problem, not architecture; fix with targeted augmentation.

If switching base models → verify the entire pipeline (CPT → SFT → DPO → GRPO), not SFT alone.

If two models are within 1 pp EM → keep the current model; switching cost exceeds expected gain.

---

## Numbers to Remember

- Switching threshold: ≥2 pp EM improvement to justify pipeline migration
- Minimum seeds for a reliable comparison: 3 (ideally 5)
- SFT steps for comparison run: 1000 (not full scale, but enough to separate models)
- Gemma 2 9B sliding window: 4096 tokens per layer (check against your schema length)
- DeepSeek-R1-Distill-Qwen-7B reasoning CoT training: can degrade with >3000 SFT steps on direct format

---

## Red Flags

Loss diverges for exactly one model but not others → template error, not LR.

All models converge to the same EM within 0.5% → your benchmark may be too easy to discriminate architectures.

Winner model has high variance (std dev > 2 pp across seeds) → unstable training; tune before committing.

SFT-stage winner loses advantage after DPO → the base model's preference-learning capacity is the bottleneck, not SQL accuracy.

One model is consistently 5–10x slower per step → check that LoRA target modules are correct; training full layers by accident.
