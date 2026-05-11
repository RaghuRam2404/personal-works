# Week 64 Assignment Solutions

## Task 1: Q4_K_M GGUF — Key Snippets

The conversion script requires the model directory path, not any individual file. A common failure is passing the safetensors file directly.

```bash
# Full working command sequence
cd llama.cpp

python convert_hf_to_gguf.py \
    /path/to/postgres-sqlcoder-7b-final \
    --outtype f16 \
    --outfile /data/postgres-sqlcoder-7b-f16.gguf

./build/bin/llama-quantize \
    /data/postgres-sqlcoder-7b-f16.gguf \
    /data/postgres-sqlcoder-7b-Q4_K_M.gguf \
    Q4_K_M

# Perplexity evaluation
./build/bin/llama-perplexity \
    -m /data/postgres-sqlcoder-7b-Q4_K_M.gguf \
    -f /data/held_out_sql.txt \
    --ctx-size 512
```

Expected F16 GGUF size: 14.1–14.5 GB. Expected Q4_K_M size: 4.3–4.5 GB. Expected perplexity: 8.8–10.5 (your SQL corpus, not WikiText).

## Task 2: AWQ INT4 — Key Snippets

The trickiest part is preparing calibration data in the correct format — a list of strings, not tokenized tensors.

```python
import json
from datasets import load_from_disk

# Load your v3 training set, take first 512 examples
ds = load_from_disk("./data/sqlcoder_v3_train")
calib_data = [ex["instruction"] for ex in ds.select(range(512))]

# AWQ expects a list of strings — no tokenization needed
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./postgres-sqlcoder-7b-final")
model = AutoAWQForCausalLM.from_pretrained(
    "./postgres-sqlcoder-7b-final",
    device_map="cuda:0",
    safetensors=True,
    torch_dtype="auto"
)

model.quantize(
    tokenizer,
    quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"},
    calib_data=calib_data
)
model.save_quantized("./postgres-sqlcoder-7b-awq-int4", safetensors=True)
tokenizer.save_pretrained("./postgres-sqlcoder-7b-awq-int4")
```

## Task 3: GPTQ INT4 — Key Snippets

The most common failure is running out of VRAM mid-quantization. Offload to CPU if needed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

calib_texts = [ex["instruction"] for ex in train_ds.select(range(512))]

gptq_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset=calib_texts,
    tokenizer=AutoTokenizer.from_pretrained("./postgres-sqlcoder-7b-final"),
    damp_percent=0.1,
    desc_act=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "./postgres-sqlcoder-7b-final",
    quantization_config=gptq_config,
    device_map="auto",   # offloads to CPU if GPU too small
    torch_dtype="auto",
)
model.save_pretrained("./postgres-sqlcoder-7b-gptq-int4")
```

If you see `NaN` in perplexity: increase `damp_percent` to 0.2 or even 0.5.

## Task 4: Pushing to Hub

```python
from huggingface_hub import HfApi, upload_file

api = HfApi()

# GGUF: single file upload
api.create_repo("postgres-sqlcoder-7b-Q4_K_M-GGUF", exist_ok=True)
upload_file(
    path_or_fileobj="/data/postgres-sqlcoder-7b-Q4_K_M.gguf",
    path_in_repo="postgres-sqlcoder-7b-Q4_K_M.gguf",
    repo_id="<your-handle>/postgres-sqlcoder-7b-Q4_K_M-GGUF",
)

# AWQ and GPTQ: use model.push_to_hub()
awq_model.push_to_hub("<your-handle>/postgres-sqlcoder-7b-awq-int4")
```

## Common Gotchas

- The `convert_hf_to_gguf.py` script sometimes fails on Qwen2.5 with a tokenizer error. If so, pass `--vocab-type bpe` explicitly.
- AWQ `version="GEMM"` is required for Ampere+ GPUs. Use `"GEMV"` for inference-only on older cards.
- GPTQ will hang if your calibration strings are longer than the model's context length. Truncate to `max_length=512` before passing.
- `model.push_to_hub()` defaults to private repos. Pass `private=False` to make them public immediately.
- After GPTQ quantization, the saved model will load with `from_pretrained(..., device_map="auto")` and auto-detect the GPTQ weights via `quantization_config` in `config.json` — you do not need to pass `GPTQConfig` again at load time.

## How to Verify You Did It Right

- GGUF: `llama-cli -m postgres-sqlcoder-7b-Q4_K_M.gguf -p "SELECT * FROM orders WHERE" -n 30` should output valid-looking SQL continuation in under 5 seconds on CPU.
- AWQ: Load with `AutoModelForCausalLM.from_pretrained("./postgres-sqlcoder-7b-awq-int4", device_map="auto")` — no extra args. If it loads without error and `model.config.quantization_config` exists, quantization was saved correctly.
- GPTQ: `model.config.quantization_config.bits == 4` should be True after loading.
- Hub: Visit `https://huggingface.co/<your-handle>/postgres-sqlcoder-7b-Q4_K_M-GGUF` in a browser; the GGUF file should be listed under Files.
