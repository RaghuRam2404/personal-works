# Week 34 TakeAway — Unsloth Speed Unlock

**One-liner:** Unsloth = fused LoRA kernels + custom GC + RoPE optimization → 2–5x faster, 40–60% less VRAM. Same model quality.

---

## Minimal Unsloth Setup

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B",
    max_seq_length=512,
    dtype=None,        # auto BF16 on Ampere
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # NOT True
    random_state=42,
)
# Then use SFTTrainer exactly as in Week 33 — no other changes needed
```

---

## What Unsloth Optimizes

| Optimization | Benefit |
|---|---|
| Fused LoRA forward/backward | 1.5–2x kernel speedup |
| Custom RoPE kernel | Reduces embedding recompute |
| Custom gradient checkpointing | 40–60% VRAM reduction vs. PyTorch GC |
| Flash Attention 2 integration | O(n) memory for attention |

---

## Vanilla vs Unsloth (Typical A100)

| | Vanilla QLoRA | Unsloth |
|---|---|---|
| Steps/sec | 1.0 | 2.5 |
| Peak VRAM | 24 GB | 14 GB |
| Training time | 45 min | 18 min |
| Final quality | Baseline | Equal |

---

## Numbers to Remember

- Unsloth speedup claim: 2–5x (empirically 2–3x on A100 for most setups)
- VRAM reduction: 40–60% vs. vanilla HF QLoRA
- `lora_dropout=0` → maximum speed; use `0.05` if overfitting is a concern
- Colab Pro A100 rate: ~$1.20/hour (estimate); 15K example run ≈ $0.30–0.50

---

## Decision Rules

- Use Unsloth for ALL single-GPU training from this week forward
- Do NOT set `gradient_checkpointing=True` in SFTConfig when using Unsloth
- On Mac → use vanilla peft + SFTTrainer for debugging; Unsloth = CUDA only
- If model not supported by Unsloth → fall back to Week 33 vanilla QLoRA setup

---

## Red Flags

- `gradient_checkpointing=True` in SFTConfig with Unsloth → conflict, remove it
- Steps/sec < 1.5 with Unsloth on A100 → Unsloth installation issue; reinstall
- VRAM > 30GB with Unsloth → not using Unsloth's GC; check `use_gradient_checkpointing="unsloth"`
- Quality significantly different from vanilla → check dtype consistency
