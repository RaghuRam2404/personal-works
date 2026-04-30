# Week 36 Assignment — DoRA, RSLoRA, LoftQ Experiments

## Setup Checklist

- [ ] Colab Pro A100
- [ ] peft >= 0.9.0 (check: `pip show peft | grep Version` — DoRA requires peft 0.9+)
- [ ] Same 5K training / 200 eval / 100 held-out test split from previous weeks
- [ ] W&B project `week-36-lora-variants` created
- [ ] Unsloth installed (DoRA available via `use_dora=True` in `get_peft_model`)

---

## Task 1 — DoRA vs. Standard LoRA Comparison

**Goal:** Empirically determine whether DoRA improves SQL fine-tuning quality.

**Requirements:**
- Run two training runs with identical settings except the adapter type:
  - **Run A:** Standard LoRA — `use_dora=False` (your Week 34 setup)
  - **Run B:** DoRA — `use_dora=True`
- Both runs: rank 16, alpha 32, all 7 linear layers, 2 epochs on 5K SQL examples, LR=2e-4
- Log both to W&B project `week-36-lora-variants` with run names `lora-baseline` and `dora-baseline`
- For each run, record: trainable params, final train loss, final eval loss, training time (min)
- After training: evaluate both models on your 100-example held-out test set (exact match %)

**Deliverable:** `week36_results.md` with comparison table and held-out eval results. GitHub commit: `week-36-dora`.

**Hints:**
- `use_dora=True` adds a small magnitude vector per adapted layer; total trainable params increase by ~d_out per layer — negligible relative to A and B
- If Unsloth doesn't support your peft version's DoRA, fall back to vanilla peft (slower but works)

---

## Task 2 — RSLoRA at High Rank

**Goal:** Verify that RSLoRA improves training stability at high ranks.

**Requirements:**
- Run two training runs at rank 64:
  - **Run C:** Standard LoRA, rank 64, alpha 128 (standard scaling = 2.0)
  - **Run D:** RSLoRA, rank 64, alpha 32 (`use_rslora=True`, scaling = 32/sqrt(64) = 4.0)
- All other settings identical; 5K examples, 2 epochs, LR=2e-4
- Record: final eval loss, training stability (did loss spike or plateau early?)
- Compare to your rank 16 baseline from Run A

**Deliverable:** RSLoRA comparison in `week36_results.md`.

**Hints:**
- peft configuration: `LoraConfig(r=64, lora_alpha=32, use_rslora=True, ...)`
- RSLoRA's effective scaling at rank 64, alpha 32: `32 / sqrt(64) = 4.0` — significantly higher than standard LoRA's `128/64 = 2.0`
- You may need to reduce LR slightly for RSLoRA at high alpha/sqrt(r) values

---

## Task 3 — LoftQ Investigation (Conceptual + Optional Implementation)

**Goal:** Understand LoftQ's initialization approach.

**Requirements:**
- Write `week36_loftq_analysis.md` (300–500 words) answering:
  - What specific initialization problem does LoftQ solve that standard QLoRA (B=0) doesn't?
  - Walk through the SVD step: if the quantization error matrix (W_fp16 - W_nf4) has rank 200 for a 4096×4096 weight matrix, what does truncating to rank 16 mean for the initial adapter?
  - Under what conditions would you use LoftQ in your PostgreSQL fine-tuning project? (Think: when would quantization error be large enough to matter?)

- **Optional implementation:** Apply LoftQ initialization to your QLoRA setup:
  ```python
  from peft import LoftQConfig
  loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=1)
  lora_config = LoraConfig(r=16, init_lora_weights="loftq", loftq_config=loftq_config, ...)
  ```
  Compare: starting loss with LoftQ initialization vs. standard B=0 initialization.

**Deliverable:** `week36_loftq_analysis.md` committed.

---

## Task 4 — Variant Selection for Week 38

**Goal:** Make a justified decision about which adapter type to use in the Week 38 sprint.

**Requirements:**
- Based on Tasks 1–3, write a recommendation in `week36_results.md`:
  - Which adapter variant (standard LoRA, DoRA, or RSLoRA) will you use for Week 38 and why?
  - What rank will you use and why?
  - What did the experiments reveal about your specific SQL task's sensitivity to adapter design?

**Deliverable:** Recommendation section in `week36_results.md`.

---

## Stretch Goals

- Read the DoRA paper's Section 5 (experiments) and identify which tasks showed the largest improvement over LoRA; compare to your SQL task characteristics
- Try `use_rslora=True` at rank 16 and compare to standard LoRA at rank 16 — RSLoRA at low ranks should have minimal benefit
- Implement a minimal DoRA module from scratch (mirroring your Week 30 LoraLinear), adding the magnitude vector and decomposed forward pass
