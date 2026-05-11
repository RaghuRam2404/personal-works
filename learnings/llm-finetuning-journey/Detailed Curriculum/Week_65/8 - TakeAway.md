# Week 65 TakeAway — Local Deployment with Ollama and llama.cpp

Your model runs locally at 40–55 tok/s on Apple Silicon — fast enough for interactive SQL generation from any terminal.

## Key Code Patterns

```bash
# Build llama.cpp with Metal (do this once)
cmake -B build -DLLAMA_METAL=ON && cmake --build build -j8

# Run inference on GPU
./build/bin/llama-cli -m model-Q4_K_M.gguf -p "SELECT" -n 100 --n-gpu-layers 35
```

```dockerfile
# Minimal working Modelfile
FROM /path/to/model-Q4_K_M.gguf
PARAMETER num_ctx 4096
PARAMETER temperature 0.1
PARAMETER stop "<|im_end|>"
SYSTEM "You are a PostgreSQL SQL expert. Output only SQL. No explanation."
```

```python
# Ollama REST call (non-streaming)
import httpx
r = httpx.post("http://localhost:11434/v1/chat/completions",
               json={"model": "postgres-sqlcoder",
                     "messages": [{"role": "user", "content": prompt}],
                     "temperature": 0.1})
sql = r.json()["choices"][0]["message"]["content"].strip()
```

## Decision Rules

- If Metal is not activating: delete build/, reconfigure with `--fresh`, rebuild
- If model generates prose after SQL: add `<|im_end|>` to stop list in Modelfile
- If TTFT > 1s: truncate schema to only relevant tables
- If accuracy drops vs benchmark: verify prompt template matches SFT training format exactly
- If deploying to non-Python users: use PyInstaller binary or plain curl + jq wrapper

## Numbers to Remember

- Apple Silicon M2 Pro, Q4_K_M, 35 GPU layers: 38–55 tok/s
- Q4_K_M GGUF memory: ~4.5 GB VRAM/RAM + ~0.5 GB KV cache at ctx=4096
- TTFT scales roughly linearly with prompt tokens; 900-token prompt ≈ 2s on M2
- Safe temperature for deterministic SQL: 0.01 (not 0.0)
- Ollama default port: 11434; OpenAI-compatible endpoint: `/v1/chat/completions`

## Red Flags

- Inference speed same as CPU: Metal build flag was not applied — rebuild from scratch
- Model outputs explanation after SQL: missing `<|im_end|>` stop token in Modelfile
- Accuracy 10–15 pp below benchmark: prompt template does not match SFT format
- `ollama run` works but REST API fails: `ollama serve` is not running as a background service
- GGUF generates garbled output: chat template in GGUF does not match what `convert_hf_to_gguf.py` expected — re-convert
