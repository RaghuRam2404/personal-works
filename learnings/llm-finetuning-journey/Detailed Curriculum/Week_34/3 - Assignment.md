# Week 34 Assignment — Unsloth Speed Comparison

## Setup Checklist

- [ ] Colab Pro with A100 runtime
- [ ] Install Unsloth: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` (or use their pinned pip release)
- [ ] Vanilla QLoRA script from Week 33 saved and ready
- [ ] Same 5K training examples and 100-example test set from previous weeks
- [ ] W&B project `week-34-unsloth` created

---

## Task 1 — Establish Vanilla QLoRA Baseline

**Goal:** Get clean baseline metrics from your Week 33 setup before modifying anything.

**Requirements:**
- Run your Week 33 training script for exactly 200 steps (not full training — just enough to measure throughput)
- Record in `week34_comparison.md`:
  - Steps per second (from W&B or SFTTrainer progress bar)
  - Peak VRAM usage (`torch.cuda.max_memory_allocated() / 1e9`)
  - Loss at step 200
  - GPU type (`!nvidia-smi | head -5`)
- Set `max_steps=200` in SFTConfig to stop after exactly 200 steps

**Deliverable:** Baseline metrics recorded in `week34_comparison.md`. W&B run URL.

---

## Task 2 — Rewrite Training Script with Unsloth

**Goal:** Convert the vanilla QLoRA script to use Unsloth's model loading and LoRA setup.

**Requirements:**

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

- All other settings (SFTConfig, dataset) remain identical to Task 1
- Run for exactly 200 steps with `max_steps=200`
- Record same metrics: steps/sec, peak VRAM, loss at step 200

**Deliverable:** `train_unsloth_7b.py` committed. Unsloth run metrics in `week34_comparison.md`.

---

## Task 3 — Full Training Run with Unsloth

**Goal:** Run the complete 2-epoch training with Unsloth and confirm the speedup holds.

**Requirements:**
- Remove `max_steps=200` — run full 2 epochs on 5K examples
- W&B run name: `qwen-coder-7b-unsloth-sql-5k`
- Record: total training time, final train loss, final eval loss
- Compare to Week 33's training time (from your results file)

**Deliverable:** Full run metrics added to `week34_comparison.md`. GitHub commit: `week-34-unsloth`.

---

## Task 4 — Comparison Report

**Goal:** Document your empirical findings clearly.

**Requirements:**
- Create a comparison table in `week34_comparison.md`:

| Metric | Vanilla QLoRA (Week 33) | Unsloth (Week 34) | Speedup |
|---|---|---|---|
| Steps/second | | | Nx |
| Peak VRAM (GB) | | | ↓ X% |
| Training time (min) | | | Nx |
| Final eval loss | | | (should be equal) |

- Write a 2-paragraph analysis:
  1. What speedup did you observe? Is it within the claimed 2–5x range?
  2. Was there any quality difference in the final model? What does this tell you about Unsloth's kernel correctness?

**Deliverable:** Complete comparison table and analysis in `week34_comparison.md`.

---

## Task 5 — Read Unsloth Source (Optional but Recommended)

**Goal:** Understand one key optimization at the source code level.

**Requirements:**
- Navigate to [Unsloth GitHub](https://github.com/unslothai/unsloth)
- Find the file that implements the custom RoPE kernel or the fused LoRA backward pass
- In `week34_notes.md`, write 5–10 bullet points explaining what you found: what does the kernel do, what HuggingFace standard implementation does it replace, and what is the memory/compute benefit?

**Deliverable:** `week34_notes.md` committed.

---

## Stretch Goals

- Try Unsloth with `lora_dropout=0.05` vs `lora_dropout=0` — measure the speed difference
- Run Unsloth on a larger batch size (that would OOM on vanilla QLoRA) to demonstrate the VRAM reduction benefit
- Read the [Unsloth blog post on 80% memory reduction](https://unsloth.ai/blog/mistral-benchmark) and verify whether the claims match your A100 results
