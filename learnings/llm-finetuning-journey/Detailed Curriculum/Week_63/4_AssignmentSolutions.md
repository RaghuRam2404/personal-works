# Week 63 Assignment Solutions

## Task 1 — Expected GGUF File Sizes (Approximate)

| Quantization | Size (7B model) | Notes |
|-------------|----------------|-------|
| bf16 | 14.2 GB | Baseline |
| Q8_0 | 7.2 GB | ~50% of bf16 |
| Q6_K | 5.5 GB | ~38% |
| Q5_K_M | 4.8 GB | ~34% |
| Q4_K_M | 4.4 GB | ~31% (recommended sweet spot) |
| Q4_K_S | 3.9 GB | ~27% |
| Q3_K_M | 3.3 GB | ~23% |
| Q2_K | 2.7 GB | ~19% (significant accuracy loss) |

## Task 2 — GPTQ Calibration Data Preparation

```python
from datasets import load_dataset

# Use 128 examples from your v3 training set as calibration
ds = load_dataset("json", data_files="v3_train_final.jsonl")["train"].shuffle(seed=42).select(range(128))

calibration_data = []
for ex in ds:
    text = tokenizer.apply_chat_template(ex["messages"], tokenize=False)
    calibration_data.append(tokenizer(text, return_tensors="pt").input_ids[0])
```

Important: GPTQ calibration data should come from your DOMAIN, not a generic dataset. Using WikiText-2 (common default) for a SQL-specialized model will produce suboptimal quantization because the weight Hessians are computed on wrong-distribution data.

## Task 4 — Typical Results (Reference Values)

| Variant | Exec Acc | Tok/s (Mac M2) | Tok/s (A100) |
|---------|----------|----------------|--------------|
| bf16 (A100) | 65% | N/A | 85 |
| Q4_K_M (Mac M2) | 63% | 18–25 | N/A |
| GPTQ INT4 (A100) | 64% | N/A | 110 |
| AWQ INT4 (A100) | 64.5% | N/A | 130 |

---

## Common Gotchas

- **GGUF export from Unsloth requires the merged model (not LoRA).** Run `model.save_pretrained_merged()` first, then export to GGUF.
- **GPTQ quantization takes 45–90 minutes for a 7B model** on A100. Run it overnight; it's a one-time cost.
- **AWQ requires specific GPU support.** GEMM kernel requires sm_80+ (A100, RTX 3090+). On older GPUs, fall back to GEMV.
- **Calibration data tokenization must match training.** If your training used ChatML format (system + user + assistant), the calibration data must also be in ChatML format.
- **llama.cpp build on Mac.** Use: `LLAMA_METAL=1 make` to enable Metal GPU acceleration for faster inference on Apple Silicon.

---

## How to Verify You Did It Right

1. All GGUF files exist and sizes match expected ranges (±10%)
2. Q4_K_M on Mac achieves ≥ 15 tok/s (below this, check Metal is enabled)
3. Execution accuracy drop from bf16 to Q4_K_M is < 5 percentage points
4. `quantization_comparison_study.md` has all 5 variants with all 4 metrics
5. If Q2_K accuracy drops > 15pp vs bf16, document it as the "unusable" threshold in your study
