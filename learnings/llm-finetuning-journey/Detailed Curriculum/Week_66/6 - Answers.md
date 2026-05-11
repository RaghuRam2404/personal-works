# Week 66 Answers

## Q1

**Answer: B**

**Why correct:** vLLM's `--gpu-memory-utilization 0.90` allocates 90% of total VRAM for the combined model weights + KV cache. Total available: 0.90 × 24 GB = 21.6 GB. Weights occupy ~5 GB. The KV cache page pool receives the remainder: approximately 21.6 − 5 = 16.6 GB. The exact value is slightly less due to CUDA overhead, but 14.6 GB is the closest answer to the real allocation. This large KV cache pool is what allows vLLM to handle hundreds of concurrent in-flight requests without evicting pages.

**Why others are wrong:**
- A: The KV cache is additional to the model weights, not equal to them.
- C: 21.6 GB would be the KV cache pool if weights were zero-sized.
- D: vLLM never uses 100% VRAM; CUDA requires headroom for kernel buffers.

---

## Q2

**Answer: B**

**Why correct:** If `asyncio.gather` is called with coroutines that contain a blocking (synchronous) HTTP call — for example, using the standard `requests` library inside an async function — the event loop cannot run multiple coroutines concurrently. All requests execute one at a time, and vLLM receives them sequentially, so continuous batching has nothing to batch. The fix is to use `httpx.AsyncClient` or `aiohttp` for async HTTP calls inside the coroutines.

**Why others are wrong:**
- A: AWQ models in vLLM fully support batched inference.
- C: Continuous batching is the default in vLLM; no extra flag is needed.
- D: A10G fully supports paged attention.

---

## Q3

**Answer: C**

**Why correct:** The input question is natural language — blocking it with a regex is both fragile (the model might paraphrase the question in many ways) and not the right layer of defense. The correct first-line defense is at the output: parse the generated SQL using `sqlparse` (or `pglast` for a full PostgreSQL AST) and reject any statement whose top-level node type is not `SELECT`. This is deterministic, reliable, and ensures that even if the model generates `DROP TABLE orders;`, the API returns an error rather than executing it.

**Why others are wrong:**
- A: LLMs absolutely can and do generate DDL statements, especially when the question contains SQL-like phrasing.
- B: `clean_sql()` only strips markdown formatting; it does not parse SQL structure or detect DDL.
- D: Input regex is fragile — it would block many legitimate questions containing the word "drop" (e.g., "how many orders were dropped this week?").

---

## Q4

**Answer: A**

**Why correct:** The dominant cost of cold-start for a 4+ GB model is the network download from HuggingFace Hub, which can take 2–5 minutes on a standard RunPod instance. Caching the model to RunPod's persistent network volume (using `--download-dir /workspace/models`) means subsequent pod restarts skip the download entirely and cold-start drops to 30–60 seconds (just GPU loading time).

**Why others are wrong:**
- B: Higher `gpu-memory-utilization` affects KV cache allocation, not model loading speed.
- C: vLLM does not support GGUF format; it requires safetensors/PyTorch format.
- D: Disabling paged attention reduces throughput with no cold-start benefit.

---

## Q5

**Model answer:** Standard KV cache management pre-allocates a fixed contiguous block of GPU memory for each request based on the maximum sequence length. If a request only uses 200 tokens out of a 4096-token allocation, the other 3896 tokens worth of memory is wasted. When many concurrent requests exist, this fragmentation means the GPU VRAM fills up long before it is actually needed, capping concurrency. Paged attention manages the KV cache as fixed-size pages (analogous to OS virtual memory pages), allocating pages on demand as a sequence grows. This eliminates internal fragmentation entirely — a 200-token sequence uses exactly 200-tokens' worth of pages. The result is that vLLM can maintain 3–10x more concurrent in-flight sequences in the same VRAM, directly increasing throughput when requests are bursty.

---

## Q6

**Model answer:** Capacity arithmetic: at 200 tok/s aggregate throughput and 100 tokens average output, the instance can complete 200 / 100 = 2 queries per second = 120 queries per minute. The peak load is 50 requests per minute — well below the 120 qpm capacity. A single A10G is sufficient with substantial headroom (~2.4x). At 200 tok/s and 50 RPM average, the GPU utilization is approximately 50 × 100 / (200 × 60) ≈ 42%. You could handle nearly 2.4x the current peak before needing to scale.

---

## Q7

**Model answer:** First, Lambda has a maximum execution timeout of 15 minutes, but more critically it has a cold-start problem that is catastrophic for LLMs. Loading a 4–5 GB quantized model from S3 into Lambda's ephemeral environment takes 2–5 minutes, making cold-start latency unacceptable for interactive use. Second, Lambda functions do not retain GPU state between invocations — each invocation would need to reload the model, eliminating any benefit of KV cache warming or continuous batching. Lambda's design assumes stateless, short-lived functions; LLM inference servers are fundamentally stateful (maintaining KV cache across concurrent requests) and long-running. Lambda's GPU layer also has very limited VRAM compared to dedicated GPU instances.

---

## Q8 — Deep Scenario

**Model answer:** Hypothesis 1: `time_bucket` is a TimescaleDB-specific function with unusual tokenization — it may be tokenized as two or three tokens (`time`, `_bucket`) rather than one, and the AWQ salient-channel selection (calibrated on generic SQL training examples) may have under-protected the channels most important for recognizing and generating this specific function signature. Test: compute the per-token generation probabilities of `time_bucket` in both BF16 and AWQ models across 20 examples containing `time_bucket`; if AWQ assigns significantly lower probability to `time_bucket` vs alternatives, the salient-channel hypothesis is confirmed.

Hypothesis 2: The AWQ calibration data (512 examples from your v3 training set) contained few or no `time_bucket` examples, so the time-series-specific channels were not identified as salient and were aggressively quantized. Test: count `time_bucket` occurrences in the 512 calibration examples; if fewer than 5 examples contain it, this hypothesis holds.

Remediation without retraining: Re-run AWQ quantization with a calibration set enriched with TimescaleDB-specific examples (at least 50 `time_bucket` queries). This does not require any gradient updates — it is a post-training quantization pass that takes 15–20 minutes. Compare accuracy on the `time_bucket` subset before and after. If the gap closes to within 2 pp of BF16, ship the re-calibrated AWQ model.
