# Week 32 Assignment Solutions

## Task 2 — bitsandbytes 4-bit: Key Snippets

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import time

# BF16 baseline
model_bf16 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.bfloat16, device_map="auto"
)
mem_bf16 = torch.cuda.memory_allocated() / 1e9
print(f"BF16 memory: {mem_bf16:.2f} GB")

# 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
del model_bf16; torch.cuda.empty_cache()
model_4bit = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B",
    quantization_config=bnb_config, device_map="auto"
)
mem_4bit = torch.cuda.memory_allocated() / 1e9
print(f"NF4 memory: {mem_4bit:.2f} GB")

# Speed benchmark
prompt = tokenizer("SELECT", return_tensors="pt").input_ids.cuda()
start = time.time()
for _ in range(10):
    model_4bit.generate(prompt, max_new_tokens=50, do_sample=False)
speed = 500 / (time.time() - start)
print(f"Speed: {speed:.1f} tok/s")
```

---

## Expected Results (Qwen2.5-Coder-1.5B on T4)

| Method | GPU Memory | Inference Speed | Notes |
|---|---|---|---|
| BF16 | ~3.2 GB | 40–60 tok/s | Reference quality |
| INT8 (LLM.int8) | ~1.8 GB | 25–40 tok/s | Slight speed decrease due to mixed-precision |
| NF4 (bitsandbytes) | ~1.0 GB | 35–55 tok/s | Near-BF16 quality, best memory savings |
| GPTQ-4bit | ~1.0 GB | 45–65 tok/s | Slightly faster inference than NF4; longer setup |

For a 7B model (Week 33), multiply by ~4.7:
- BF16: ~14 GB
- NF4: ~4.5 GB (fits on 16GB GPU — this is QLoRA's key benefit)

---

## Task 6 — NF4 vs. INT4: Reference Explanation

**INT4 levels** (symmetric): -8, -7, -6, ..., -1, 0, 1, ..., 7 — uniformly spaced.

**NF4 levels** (computed from standard normal quantiles): approximately:
`[-1.0, -0.694, -0.512, -0.373, -0.256, -0.150, -0.052, 0.0, 0.052, 0.150, 0.256, 0.373, 0.512, 0.694, 1.0, ...]` (normalized to [-1, 1]).

LLM weights are approximately normally distributed (bell-shaped, most weights near zero). INT4's uniform spacing wastes quantization levels in the tails (values like ±7 are rarely needed) and has poor resolution near zero (where most weights are). NF4 places more levels near zero (where weights are dense) and fewer in the tails. This minimizes expected quantization error for normal distributions — "information-theoretically optimal" for this distribution.

---

## Common Gotchas

- **`bitsandbytes` not finding CUDA**: Ensure you are on a CUDA GPU (not CPU/MPS). Run `python -c "import bitsandbytes; print(bitsandbytes.__version__)"` to verify installation.
- **Memory not freed between model loads**: Always `del model` and `torch.cuda.empty_cache()` before loading the next model format.
- **GPTQ quantization OOM**: If GPTQ runs OOM during Hessian computation, reduce `dataset_size` in the calibration set or use a smaller `group_size`.
- **Perplexity measurement requires text data**: For a quick quality proxy, generate SQL for 10 held-out examples and manually count correct responses rather than computing perplexity.

---

## How to Verify You Did It Right

- NF4 model uses ~3–4x less GPU memory than BF16
- INT8 model uses ~1.5–2x less GPU memory than BF16
- Your `held_out_test.json` has exactly 100 examples with all three keys
- Week32 comparison table has all 3–4 methods filled in
- `week32_nf4_vs_int4.md` correctly explains that NF4 is non-uniform and INT4 is uniform
