# Week 64 TakeAway — Quantize and Ship Your Final Model

Merge first, quantize second, benchmark third, then push — in that exact order, every time.

## Key Code Patterns

```bash
# GGUF pipeline (CPU-friendly)
python convert_hf_to_gguf.py ./model --outtype f16 --outfile model-f16.gguf
./build/bin/llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

```python
# AWQ with domain calibration
model.quantize(tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
    calib_data=sql_instruction_strings[:128])

# GPTQ with numerical stability
GPTQConfig(bits=4, group_size=128, damp_percent=0.1, desc_act=False, dataset=calib_texts)
```

```python
# Push GGUF (single file)
from huggingface_hub import upload_file
upload_file(path_or_fileobj="model-Q4_K_M.gguf",
            path_in_repo="model-Q4_K_M.gguf",
            repo_id="<handle>/model-Q4_K_M-GGUF")
```

## Decision Rules

- If target is CPU-only laptop: Q4_K_M GGUF
- If target is cloud GPU with throughput requirement > 70 tok/s: AWQ INT4 with GEMM kernels
- If calibration NaN appears in GPTQ: raise `damp_percent` to 0.1, then 0.2, then 0.5
- If AWQ accuracy is worse than expected: switch calibration data from generic text to domain examples
- If you have the F16 GGUF, re-quantize to Q5_K_M or Q8_0 for free — keep it

## Numbers to Remember

| Format | Disk | VRAM | Tok/s (A10G) |
|--------|------|------|-------------|
| Q4_K_M GGUF | ~4.5 GB | ~5 GB RAM | ~38 (CPU) |
| AWQ INT4 | ~4.2 GB | ~5.0 GB | ~85 |
| GPTQ INT4 | ~4.5 GB | ~5.2 GB | ~72 |

- Calibration set: 128–512 examples is sufficient; more does not help beyond 512
- `group_size=128` is the standard; smaller (64) improves accuracy 0.2–0.5 pp at 2x overhead
- GPTQ `damp_percent=0.1` is safe for all 7B models tested

## Red Flags

- Quantized model outputs all-EOS or repeated tokens: adapter was not merged before quantization
- GPTQ perplexity is NaN: `damp_percent` too small — increase it
- AWQ accuracy 3+ pp below BF16: calibration data is out-of-domain
- Hub push completes but model card is empty: write `README.md` before pushing, not after
- GGUF loads but generates garbage SQL: verify `convert_hf_to_gguf.py` used the correct chat template
