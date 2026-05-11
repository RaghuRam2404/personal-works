# Week 32 TakeAway — Quantization Fundamentals

**One-liner:** NF4 = normally-distributed-optimal 4-bit; QLoRA training uses frozen NF4 base + trainable LoRA; deployment uses GPTQ/AWQ.

---

## Format Comparison

| Format | Bits | 7B Memory | Training? | Quality |
|---|---|---|---|---|
| FP32 | 32 | 28 GB | Baseline | Reference |
| BF16 | 16 | 14 GB | Standard | Near-lossless |
| FP16 | 16 | 14 GB | Risk overflow | Near-lossless |
| INT8 (LLM.int8) | 8 | 7 GB | No | Small loss |
| NF4 (bnb) | 4 | ~4.5 GB | Yes (via QLoRA) | Small loss |
| GPTQ-4bit | 4 | ~3.5 GB | No | Small loss |
| AWQ-4bit | 4 | ~3.5 GB | No | Small loss |

---

## bitsandbytes 4-bit Load

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 > INT4 for LLM weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # compute in bf16
    bnb_4bit_use_double_quant=True,      # saves ~0.37 bits/param extra
)
model = AutoModelForCausalLM.from_pretrained("model-id", quantization_config=bnb_config)
```

---

## Key Distinctions

- **BF16 vs FP16:** BF16 = wide exponent (no overflow) + low mantissa. Prefer BF16 for training.
- **INT4 vs NF4:** INT4 = uniform levels. NF4 = quantile-based, optimal for normal distribution of weights.
- **GPTQ vs AWQ:** Both are 4-bit PTQ. GPTQ uses Hessian (slower, more accurate). AWQ uses activation scaling (faster, competitive quality).
- **Weight-only vs activation quantization:** LLM.int8() handles activation outliers with mixed-precision.

---

## Numbers to Remember

- NF4 has 16 levels; INT4 has 16 levels — same count, different placement
- Double quantization saves ~0.37 bits/param extra
- GPTQ calibration: 100–512 examples, takes 5–30 minutes
- AWQ calibration: fast, < 5 minutes typically
- Outlier threshold in LLM.int8(): 6.0 (activation values above this go to FP16 path)

---

## Decision Rules

- Training 7B on 16GB GPU → NF4 (QLoRA)
- Inference on 16GB GPU, max quality → BF16 (just fits)
- Inference, want batching + speed → GPTQ-4bit or AWQ-4bit
- Memory constraint above quality → pick lowest bit that passes quality threshold

---

## Red Flags

- FP16 training NaN on step 1 → gradient overflow; switch to BF16
- NF4 model generates garbage → quantization not applied; check `model.config.quantization_config`
- GPTQ OOM during calibration → reduce group size or calibration dataset size
