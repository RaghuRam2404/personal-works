# Week 64 — Quantization Part 2: Quantize Your Final Model

## Learning Objectives

By the end of this week, you will be able to:

- Export `postgres-sqlcoder-7b-final` to three quantized formats: Q4_K_M GGUF, AWQ INT4, and GPTQ INT4
- Run perplexity evaluation on each variant using llama.cpp's `perplexity` binary and a held-out SQL corpus
- Measure end-to-end inference throughput (tokens/second) and memory footprint for each format
- Diagnose accuracy degradation by running your 200-example custom benchmark on each quantized model
- Push all three variants to Hugging Face Hub under a consistent naming convention

## Concepts

### The Quantization Execution Pipeline

Last week you compared GGUF, GPTQ, and AWQ theoretically. This week you execute the full pipeline on `postgres-sqlcoder-7b-final`, the model you trained over the previous 12 weeks. There is a specific order of operations that matters.

You start from your merged BF16 model — not a LoRA checkpoint. Every quantization tool expects a standard HuggingFace model directory with `config.json`, `tokenizer.json`, and the full weight files (either a single `model.safetensors` or sharded files). If you still have an unmerged adapter, run `model.save_pretrained()` with `merge_adapter=True` first.

The three pipelines are independent and can run in parallel if you have multiple machines. On a single machine, run them sequentially: GGUF first (fastest, CPU-only), then AWQ (needs GPU calibration, ~15 minutes), then GPTQ (heaviest calibration, ~30–60 minutes for a 7B model).

### Producing the Q4_K_M GGUF

The GGUF pipeline has two steps: convert then quantize.

```bash
# Step 1: Convert HF model to fp16 GGUF
python llama.cpp/convert_hf_to_gguf.py \
    ./postgres-sqlcoder-7b-final \
    --outtype f16 \
    --outfile postgres-sqlcoder-7b-f16.gguf

# Step 2: Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize \
    postgres-sqlcoder-7b-f16.gguf \
    postgres-sqlcoder-7b-Q4_K_M.gguf \
    Q4_K_M
```

The intermediate F16 GGUF will be ~14 GB; keep it around so you can re-quantize to other levels (Q5_K_M, Q8_0) without re-converting. Q4_K_M targets roughly 4.8 bits per weight effective after the mixed-precision tensor assignments — the model lands near 4.5 GB on disk.

### Producing the AWQ INT4

AWQ quantization uses your SQL calibration corpus to identify salient channels before quantizing. The key parameter is `group_size=128`, which balances quality vs overhead. With Qwen2.5 architecture, use `zero_point=True` (asymmetric quantization) because the feed-forward activations are not zero-centered.

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "./postgres-sqlcoder-7b-final"
quant_path = "./postgres-sqlcoder-7b-awq-int4"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(
    model_path, device_map="auto", safetensors=True
)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

# Use your SQL training examples as calibration data
calib_data = [ex["instruction"] for ex in calibration_examples[:128]]

model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
model.save_quantized(quant_path, safetensors=True)
tokenizer.save_pretrained(quant_path)
```

Using domain-specific calibration data (SQL instructions from your training set) rather than generic text improves AWQ quality for SQL tasks. This is one of the non-obvious production tricks.

### Producing the GPTQ INT4

GPTQ is the most calibration-intensive format. It processes each layer's weight matrix using an approximate Hessian computed from the calibration activations. The `damp_percent` parameter controls numerical stability — set it to 0.1 for 7B models.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

tokenizer = AutoTokenizer.from_pretrained("./postgres-sqlcoder-7b-final")

# Load calibration dataset
calib_texts = load_sql_calibration_texts(n=512)  # your SQL examples

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset=calib_texts,
    tokenizer=tokenizer,
    damp_percent=0.1,
    desc_act=False,  # False is faster; True is slightly better quality
)

model = AutoModelForCausalLM.from_pretrained(
    "./postgres-sqlcoder-7b-final",
    quantization_config=gptq_config,
    device_map="auto",
    torch_dtype="auto"
)

model.save_pretrained("./postgres-sqlcoder-7b-gptq-int4")
tokenizer.save_pretrained("./postgres-sqlcoder-7b-gptq-int4")
```

### Evaluation Across Formats

You must evaluate three things for each quantized variant:

1. Perplexity on a held-out SQL corpus: Use llama.cpp's `perplexity` binary for GGUF. For AWQ/GPTQ, use the HuggingFace `lm_eval` harness with `perplexity` task on your held-out set.

2. Accuracy on your custom 200-example benchmark: Run the same inference script from Week 61, substituting each quantized model. Record exact-match SQL accuracy.

3. Throughput and memory: For GGUF, report tokens/sec from `llama-cli` with `--n-predict 200`. For AWQ/GPTQ, report from a Python loop with `time.perf_counter()` and `torch.cuda.memory_allocated()`.

Assemble results in a table:

| Format | Disk (GB) | Peak VRAM (GB) | PPL (SQL) | Accuracy (200ex) | Tok/s |
|--------|-----------|----------------|-----------|------------------|-------|
| BF16 (baseline) | 14.5 | 16.2 | 8.4 | 0.831 | 45 |
| Q4_K_M GGUF | 4.4 | 5.1* | ~9.2 | ~0.81 | 38** |
| AWQ INT4 | 4.2 | 5.0 | ~9.0 | ~0.82 | 85 |
| GPTQ INT4 | 4.5 | 5.2 | ~9.1 | ~0.81 | 72 |

(*CPU RAM for llama.cpp; **single-threaded on Apple Silicon M2)

### Pushing to Hugging Face Hub

Use a consistent naming convention. The Hub expects each variant as a separate repository:

```
<your-handle>/postgres-sqlcoder-7b-Q4_K_M-GGUF
<your-handle>/postgres-sqlcoder-7b-awq-int4
<your-handle>/postgres-sqlcoder-7b-gptq-int4
```

For the GGUF, use `huggingface_hub.upload_file()` directly (it is a single binary, not a model directory). For AWQ and GPTQ, use `model.push_to_hub()`.

Each repo needs a `README.md` (model card) that documents: base model, training pipeline, quantization method, evaluation results, and how to load it.

## Connections

This week completes the quantization arc (Weeks 63–64). Weeks 65–66 consume these artifacts directly — your `postgres-sqlcoder-7b-Q4_K_M.gguf` is the file you load into Ollama in Week 65, and your AWQ variant is what powers the cloud vLLM endpoint in Week 66.

The evaluation table you build this week is Table 3 in your technical report (Week 67–70).

## Common Misconceptions / Pitfalls

Merging the adapter before quantizing is non-negotiable. Quantizing a LoRA checkpoint directly (without merging) will produce a model that loads but generates garbage because the adapter weights are not incorporated into the base weight matrices.

GPTQ with `desc_act=True` (activation ordering) is significantly slower to quantize and marginally better on perplexity but often worse on throughput due to non-contiguous memory access patterns. For deployment, start with `desc_act=False`.

The calibration dataset for AWQ should be semantically similar to your inference distribution. Using WikiText-103 (the common default) for a SQL-specialized model will produce worse salient-channel detection than using your own SQL examples.

Do not compare tok/s between GGUF on CPU and AWQ/GPTQ on GPU — they are different hardware. Keep hardware fixed when comparing quantization methods.

## Time Allocation (6–8 hours)

- 0.5h: Merge adapter, verify BF16 model directory is complete
- 1.5h: GGUF conversion and Q4_K_M quantization + perplexity check
- 1.5h: AWQ quantization with SQL calibration data
- 1.5h: GPTQ quantization
- 1.0h: Evaluation harness across all three variants, build comparison table
- 0.5h: Write model cards, push all three to Hub
