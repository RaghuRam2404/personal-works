# Week 63 Assignment — Quantization Method Comparison Study

## Setup Checklist

- [ ] `llama.cpp` cloned and built: `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make`
- [ ] `auto-gptq` installed: `pip install auto-gptq optimum`
- [ ] `autoawq` installed: `pip install autoawq`
- [ ] A reference 7B model (use `Qwen/Qwen2.5-Coder-7B-Instruct` — not your final model, which you quantize in Week 64)
- [ ] 100 BIRD-SQL dev examples for comparison eval

---

## Task 1 — Create GGUF Quantizations

**Goal:** Generate and test multiple GGUF quantization levels.

**Requirements:**
Using Unsloth's GGUF export:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=False,
)

# Save to multiple GGUF quantization levels
for quant in ["q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_k_s", "q3_k_m", "q2_k"]:
    model.save_pretrained_gguf(
        f"gguf_{quant}",
        tokenizer,
        quantization_method=quant,
    )
```

Record file sizes for each quantization level. Create a table:

| Quant | File size | % of bf16 |
|-------|-----------|-----------|
| bf16 (reference) | 14.2 GB | 100% |
| Q8_0 | — | — |
| Q6_K | — | — |
| Q5_K_M | — | — |
| Q4_K_M | — | — |
| Q4_K_S | — | — |
| Q3_K_M | — | — |
| Q2_K | — | — |

**Deliverable:** All GGUF files created; file size table committed to `gguf_sizes.md`.

---

## Task 2 — GPTQ Quantization

**Goal:** Create INT4 and INT8 GPTQ variants of the reference model.

**Requirements:**
```python
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,       # True is slower but sometimes more accurate
)

model = AutoGPTQForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    quantize_config,
)

# Calibration data: 128 examples from your training set
examples = [...]  # list of tokenized inputs
model.quantize(examples)
model.save_quantized("qwen25coder7b_gptq_int4", use_safetensors=True)
```

Record:
- Quantization time
- Output model size
- Memory required during quantization

**Deliverable:** GPTQ INT4 model saved locally (or pushed to HuggingFace).

---

## Task 3 — AWQ Quantization

**Goal:** Create INT4 AWQ variant of the reference model.

**Requirements:**
```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",  # or "GEMV" for slower CPU-like inference
}

model.quantize(tokenizer, quant_config=quant_config,
               calib_data="<your calibration dataset>")
model.save_quantized("qwen25coder7b_awq_int4")
```

**Deliverable:** AWQ INT4 model saved.

---

## Task 4 — Accuracy and Speed Comparison

**Goal:** Empirically compare all variants on 100 BIRD-SQL examples.

**Requirements:**
Run your eval harness on:
1. bf16 reference (baseline, GPU required)
2. Q4_K_M GGUF (via llama.cpp or Ollama, on your Mac)
3. Q4_K_S GGUF (via llama.cpp)
4. GPTQ INT4 (via auto-gptq on GPU)
5. AWQ INT4 (via autoawq on GPU)

Measure:
- Execution accuracy on 100 BIRD-SQL examples
- Tokens per second (throughput)
- First token latency (time-to-first-token)

Fill in comparison table in `quantization_comparison_study.md`:

| Variant | Size | Exec Acc | Tok/s | TTFT |
|---------|------|----------|-------|------|
| bf16 (GPU) | 14.2GB | X% | X | Xs |
| Q4_K_M (Mac CPU) | 4.5GB | X% | X | Xs |
| Q4_K_S (Mac CPU) | 3.9GB | X% | X | Xs |
| GPTQ INT4 (GPU) | 4.0GB | X% | X | Xs |
| AWQ INT4 (GPU) | 3.8GB | X% | X | Xs |

**Deliverable:** `quantization_comparison_study.md` with complete table committed.

---

## Stretch Goals

- Test Q2_K (the most aggressive GGUF quantization) — at what accuracy level does it fail so badly that it is unusable?
- Profile the accuracy degradation by query difficulty (Simple vs Challenging): is quantization accuracy loss uniform across difficulty levels?
- Compare AWQ GEMM vs GEMV mode: GEMM is faster for batch inference, GEMV for single-token generation; measure both
