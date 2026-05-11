# Week 32 Assignment — Quantization Benchmarking

## Setup Checklist

- [ ] Colab Pro (T4 or A100 — T4 is sufficient)
- [ ] `pip install bitsandbytes transformers auto-gptq autoawq`
- [ ] Note: bitsandbytes requires a CUDA GPU — cannot run on Mac MPS
- [ ] Model: `Qwen/Qwen2.5-Coder-1.5B` (fast to load and quantize for benchmarking)
- [ ] Calibration dataset for GPTQ: any 100–500 SQL examples from your dataset

---

## Task 1 — Build Your Held-Out SQL Test Set

**Goal:** Create the evaluation dataset you will use for all remaining Phase 4 runs.

**Requirements:**
- Assemble 100 (question, schema, expected_SQL) triples
- Sources: a mix of Spider dev set, sql-create-context, or hand-crafted examples that test PostgreSQL-specific features (JOINs, subqueries, GROUP BY with HAVING, window functions)
- Save as `held_out_test.json` with keys `"question"`, `"schema"`, `"expected_sql"`
- This is the test set you will use through Week 39's execution-based eval
- **Do not use these examples during any training run.** They are for evaluation only.

**Deliverable:** `held_out_test.json` committed to GitHub. Committed separately from training data.

> **Note:** The curriculum explicitly says you must build this test set yourself in Week 32 — it will not be provided.

---

## Task 2 — Quantize with bitsandbytes 4-bit

**Goal:** Load Qwen2.5-Coder-1.5B in 4-bit NF4 and measure the impact.

**Requirements:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B",
    quantization_config=bnb_config,
    device_map="auto",
)
```

- Measure and record: (a) GPU memory used after loading (use `torch.cuda.memory_allocated()`), (b) time to generate 50 tokens for a fixed SQL prompt, (c) perplexity on a fixed 500-token text sample
- Compare to BF16 baseline (same model without quantization)

**Deliverable:** `week32_results.md` with: memory (BF16 vs 4-bit), inference speed (tokens/sec), perplexity or generation quality comparison.

---

## Task 3 — Quantize with bitsandbytes INT8

**Goal:** Compare INT8 (LLM.int8()) to 4-bit NF4.

**Requirements:**
- Load the same model with `load_in_8bit=True`
- Measure: GPU memory, inference speed, generation quality
- Add to `week32_results.md` comparison table

---

## Task 4 — GPTQ Quantization (optional but strongly recommended)

**Goal:** Experience post-training quantization with calibration data.

**Requirements:**
- Use `auto-gptq` to quantize Qwen2.5-Coder-1.5B to 4-bit using 100 calibration examples from your SQL dataset
- Measure: quantization time, resulting model size on disk, inference speed
- If GPTQ is too slow on T4 (>30 min), record this as your finding and note the calibration dataset size needed

**Deliverable:** Section in `week32_results.md`.

---

## Task 5 — Comparison Table

**Goal:** Produce a clear summary of all quantization methods.

**Requirements:**
- Create a table in `week32_results.md`:

| Method | Bits | GPU Memory | Inference Speed | Perplexity / Quality Notes |
|---|---|---|---|---|
| BF16 (baseline) | 16 | X GB | X tok/s | Reference |
| bitsandbytes INT8 | 8 | X GB | X tok/s | ... |
| bitsandbytes NF4 | 4 | X GB | X tok/s | ... |
| GPTQ-4bit | 4 | X GB | X tok/s | ... |

- Write a 1-paragraph recommendation: for the QLoRA training you will do in Week 33 (on a 7B model), which format should you use for the base model and why?

**Deliverable:** Complete table and recommendation in `week32_results.md`. GitHub commit: `week-32-quantization`.

---

## Task 6 — Write the NF4 vs. INT4 Explanation

**Goal:** Force yourself to articulate the key difference in your own words.

**Requirements:**
- Write `week32_nf4_vs_int4.md` (200–300 words) explaining:
  - Why uniformly spaced integers (INT4) are suboptimal for normally distributed weights
  - How NF4 uses quantile-based non-uniform values
  - What "information theory optimal" means in this context
  - Include a mini-diagram or table showing the 16 INT4 levels vs. the 16 NF4 levels

**Deliverable:** `week32_nf4_vs_int4.md` committed.

---

## Stretch Goals

- Try AWQ via `autoawq`: `from awq import AutoAWQForCausalLM`; compare with GPTQ and NF4
- Measure the effect of group size in GPTQ (group_size=128 vs 64 vs 32) on model quality and memory
- Look at the actual weight distributions of Qwen2.5-Coder-1.5B: `plt.hist(model.model.layers[0].self_attn.q_proj.weight.flatten().float().numpy())` — does it look Gaussian? Log your finding.
