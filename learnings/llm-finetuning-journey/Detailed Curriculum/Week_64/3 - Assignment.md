# Week 64 Assignment — Quantize and Publish Your Final Model

## Setup Checklist

- [ ] `postgres-sqlcoder-7b-final` merged BF16 checkpoint available locally (14–15 GB)
- [ ] llama.cpp cloned and built: `cmake -B build && cmake --build build -j8 --config Release`
- [ ] `autoawq` installed: `pip install autoawq`
- [ ] `auto-gptq` installed: `pip install auto-gptq optimum`
- [ ] Hugging Face account with `huggingface-cli login` completed
- [ ] Calibration dataset prepared: 512 SQL instruction strings from your v3 training set
- [ ] 50 GB free disk space (F16 GGUF + three quantized variants)

## Task 1: Produce Q4_K_M GGUF

**Goal:** Create a deployable GGUF file that runs on CPU with 5 GB RAM.

**Requirements:**
- [ ] Run `convert_hf_to_gguf.py` to produce `postgres-sqlcoder-7b-f16.gguf`
- [ ] Run `llama-quantize` to produce `postgres-sqlcoder-7b-Q4_K_M.gguf`
- [ ] Verify the output file size is between 4.2–4.8 GB
- [ ] Run a quick smoke test: `llama-cli -m postgres-sqlcoder-7b-Q4_K_M.gguf -p "SELECT all orders from" -n 50` and confirm coherent SQL output
- [ ] Measure perplexity: run `llama-perplexity -m postgres-sqlcoder-7b-Q4_K_M.gguf -f held_out_sql.txt`
- [ ] Record: perplexity value, file size in GB, peak RAM usage

**Deliverable:** `postgres-sqlcoder-7b-Q4_K_M.gguf` file + `quant_results.md` with measurements

## Task 2: Produce AWQ INT4

**Goal:** Create an AWQ-quantized model using domain-specific calibration data.

**Requirements:**
- [ ] Prepare `calib_data.json`: 512 SQL instructions from your v3 training set (use `json.dumps([ex["instruction"] for ex in ...])`)
- [ ] Run AutoAWQ quantization with `q_group_size=128`, `w_bit=4`, `zero_point=True`
- [ ] Verify the quantized directory contains `config.json` and `quantize_config.json`
- [ ] Run accuracy check on your 200-example benchmark:
  ```bash
  python eval_sql.py \
      --model_path ./postgres-sqlcoder-7b-awq-int4 \
      --benchmark data/custom_200.json \
      --output results/awq_results.json
  ```
- [ ] Measure throughput: average tok/s over 10 prompts of length ~50 tokens
- [ ] Confirm accuracy degradation is less than 3 percentage points vs BF16 baseline

**Deliverable:** `./postgres-sqlcoder-7b-awq-int4/` directory + measurements added to `quant_results.md`

**Hints:** If AWQ raises `RuntimeError: CUDA out of memory`, reduce `q_group_size` to 64 or pass `low_cpu_mem_usage=True` to `from_pretrained`.

## Task 3: Produce GPTQ INT4

**Goal:** Create a GPTQ-quantized model using HuggingFace's GPTQConfig.

**Requirements:**
- [ ] Run GPTQ quantization with `bits=4`, `group_size=128`, `damp_percent=0.1`, `desc_act=False`
- [ ] Use the same 512-example SQL calibration set as Task 2
- [ ] Confirm quantization completes without NaN loss (if you see NaN, increase `damp_percent` to 0.2)
- [ ] Run the same 200-example accuracy benchmark
- [ ] Measure throughput and peak VRAM with `torch.cuda.max_memory_allocated()`
- [ ] Record results in `quant_results.md`

**Deliverable:** `./postgres-sqlcoder-7b-gptq-int4/` directory + complete `quant_results.md`

## Task 4: Build Comparison Table and Push to Hub

**Goal:** Synthesize your measurements and publish all three variants publicly.

**Requirements:**
- [ ] Complete `quant_results.md` with a markdown table comparing all four variants (BF16 + 3 quantized) across: disk size, VRAM/RAM, perplexity, accuracy, tok/s
- [ ] Write a model card (`README.md`) for each Hub repo with: base model, training lineage, quantization method, eval results table, load snippet
- [ ] Push GGUF to `<your-handle>/postgres-sqlcoder-7b-Q4_K_M-GGUF` using `huggingface_hub.upload_file()`
- [ ] Push AWQ to `<your-handle>/postgres-sqlcoder-7b-awq-int4` using `model.push_to_hub()`
- [ ] Push GPTQ to `<your-handle>/postgres-sqlcoder-7b-gptq-int4` using `model.push_to_hub()`
- [ ] Verify all three repos are public and have model cards visible on the Hub

**Deliverable:** Three public HuggingFace Hub repos + links documented in `quant_results.md`

## Stretch Goals

- Quantize to Q5_K_M GGUF and compare perplexity/accuracy with Q4_K_M to quantify the quality step
- Try AWQ with generic calibration data (e.g., WikiText-103 sample) and compare accuracy to domain-calibrated AWQ — empirically demonstrate the calibration data effect
- Run the GPTQ model with `exllamav2` kernels and report throughput improvement vs standard AutoGPTQ
- Set up an Ollama Modelfile for your GGUF and run a structured generation test end-to-end (preview of Week 65)
