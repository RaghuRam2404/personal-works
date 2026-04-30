# Week 65 Glossary — Local Deployment

**Ollama**: A local model server that wraps llama.cpp inference behind a Docker-style model registry and OpenAI-compatible REST API.

**Modelfile**: Ollama's configuration file (analogous to a Dockerfile) that specifies the GGUF path, inference parameters, stop tokens, and system prompt for a named model.

**Metal (Apple GPU backend)**: Apple's GPU compute framework; llama.cpp uses Metal shaders to run matrix multiplications on the Apple Silicon GPU, yielding 3–5x speedup over CPU inference.

**`--n-gpu-layers`**: llama.cpp flag that controls how many transformer layers are offloaded to the GPU; set to 35 or higher to put all layers (including non-transformer tensors) on GPU for a 7B model.

**TTFT (Time to First Token)**: Latency from prompt submission to the first output token; dominated by the prefill (prompt processing) step and scales linearly with prompt length.

**Prefill**: The forward pass over all input tokens to compute the KV cache; happens once per request and determines TTFT.

**Decode**: The autoregressive generation loop that produces one token at a time after prefill; determines total generation time and tokens-per-second.

**OpenAI-compatible API**: A REST interface that accepts the same JSON schema as OpenAI's `/v1/chat/completions` endpoint; Ollama exposes this at `http://localhost:11434/v1`.

**Stop token**: A token or string that signals the inference engine to halt generation; for Qwen2.5 chat models the primary stop token is `<|im_end|>`.

**GBNF grammar**: llama.cpp's grammar format for constraining model output to a specific syntax (e.g., valid SQL); prevents generation of characters outside the grammar at the sampling level.
