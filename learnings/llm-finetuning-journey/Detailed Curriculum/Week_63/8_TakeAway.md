# Week 63 TakeAway — Quantization Formats Comparison

**One-liner:** Q4_K_M for Mac/CPU deployment; AWQ INT4 for GPU cloud serving; GPTQ INT4 as alternative GPU format.

---

## Format Selection Guide

| Need | Format | Library |
|------|--------|---------|
| Run on Mac, no GPU | Q4_K_M GGUF | Ollama or llama.cpp |
| High-throughput GPU API | AWQ INT4 | autoawq + vLLM |
| GPU, maximum ecosystem | GPTQ INT4 | auto-gptq |
| Maximum quality | Q8_0 GGUF or bf16 | — |
| Minimum size, acceptable quality | Q4_K_S or Q3_K_M | llama.cpp |

## Quantization Commands

```bash
# GGUF via Unsloth (Python)
model.save_pretrained_gguf("output", tokenizer, quantization_method="q4_k_m")

# Build llama.cpp for Metal (Mac)
git clone https://github.com/ggerganov/llama.cpp
LLAMA_METAL=1 make -j$(nproc)

# Run inference with llama.cpp
./llama-cli -m model-q4_k_m.gguf -p "SELECT" -n 200
```

## Expected Accuracy vs Size (7B model)

| Format | Size | Acc vs bf16 |
|--------|------|------------|
| bf16 | 14.2 GB | 100% |
| Q8_0 | 7.2 GB | ~99% |
| Q5_K_M | 4.8 GB | ~97% |
| Q4_K_M | 4.4 GB | ~96% (sweet spot) |
| Q2_K | 2.7 GB | ~85% (too lossy) |

---

## Decision Rules

- Use domain calibration data for GPTQ/AWQ (not WikiText-2)
- Q4_K_M is the default recommendation: best quality-to-size ratio
- Below Q3_K_M: accuracy degradation typically exceeds 10pp — avoid for production
- For TimescaleDB accuracy: Q5_K_M or add post-quantization LoRA adaptation
- Always measure accuracy empirically on your task before claiming quality

---

## Numbers to Remember

- Mac M2 memory bandwidth: ~68–100 GB/s
- A100 HBM bandwidth: 2,000 GB/s
- Q4_K_M throughput on Mac M2: 15–25 tok/s
- AWQ INT4 throughput on A100: 120–150 tok/s
- Typical accuracy drop Q4_K_M vs bf16: 2–5 pp overall

---

## Red Flags

- GGUF export from LoRA model (not merged): export will fail or produce wrong weights
- Using WikiText-2 for calibration of a domain-specialized model: suboptimal quantization
- Q4_K_M accuracy drop > 8pp: unusual — may indicate your model has activation outliers; try AWQ instead
- llama.cpp not built with Metal flag: CPU-only, expect 3–8 tok/s instead of 15–25
