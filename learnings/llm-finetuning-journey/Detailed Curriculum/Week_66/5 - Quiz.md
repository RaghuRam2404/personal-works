# Week 66 Quiz — Cloud Deployment with vLLM

## Multiple Choice

**Q1.** You launch vLLM with `--gpu-memory-utilization 0.90` on an A10G (24 GB). Your AWQ model uses 5 GB of VRAM. Approximately how many GB does vLLM allocate to the KV cache page pool?

A. 5 GB (only as much as the model weights)
B. 14.6 GB (0.90 × 24 − 5)
C. 21.6 GB (0.90 × 24)
D. 24 GB (all available VRAM)

**Q2.** You observe that concurrent throughput (aggregate tok/s at N=8) is only 1.1x the single-request baseline, not the 4–5x you expected from continuous batching. What is the most likely cause?

A. Your AWQ model does not support batched inference
B. Your requests are being sent sequentially rather than concurrently — the asyncio gather is blocked by a synchronous call
C. vLLM requires `--enable-continuous-batching` flag to activate the feature
D. A10G does not support paged attention; you need an A100

**Q3.** A caller sends a POST request to your FastAPI `/sql` endpoint with a question containing a SQL injection attempt in the question string: `"DROP TABLE orders; --"`. Your FastAPI wrapper calls vLLM which generates `DROP TABLE orders;`. What is your first line of defense?

A. The LLM will never generate DDL statements; this is impossible
B. Your `clean_sql()` function strips the statement down to valid SQL
C. Parse the generated SQL with sqlparse and reject any statement that is not a SELECT
D. Block the input using a regex on the question string before calling the model

**Q4.** You want to minimize cold-start latency when the first request arrives after the vLLM server starts. The model is hosted on HuggingFace Hub. What is the most effective approach?

A. Use `--download-dir` to cache the model on the RunPod pod's persistent storage volume, so subsequent pod starts skip the download
B. Increase `--gpu-memory-utilization` to 0.99 to speed up initialization
C. Use the GGUF format with vLLM instead of AWQ
D. Disable vLLM's paged attention to reduce initialization time

## Short Answer

**Q5.** Explain the difference between paged attention and standard KV cache management, and explain why paged attention enables higher concurrent throughput.

**Q6.** Your FastAPI SQL API is called by a BI dashboard that sends 50 requests per minute at peak. Your single A10G instance handles 200 tok/s at N=8 concurrency, and average SQL output is 100 tokens. Show your arithmetic for whether one A10G is sufficient.

**Q7.** A team member proposes deploying the model to AWS Lambda with a GPU layer. Describe two fundamental reasons why Lambda is a poor fit for LLM inference serving.

## Deep Scenario

**Q8.** Three months after deployment, your cloud SQL API receives a complaint: accuracy on TimescaleDB-specific `time_bucket` queries has dropped from 74% to 61% since you started serving the AWQ quantized model. Your BF16 reference model still achieves 74% on the same queries.

Write a root-cause investigation plan (200–250 words) that: (a) proposes two specific hypotheses for why AWQ performs worse specifically on `time_bucket`, (b) designs a targeted evaluation to test each hypothesis, and (c) proposes a remediation path that does not require retraining the full model.
