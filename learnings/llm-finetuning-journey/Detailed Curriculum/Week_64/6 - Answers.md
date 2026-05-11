# Week 64 Answers

## Q1

**Answer: B**

**Why correct:** GPTQ inverts the layer-wise Hessian to compute the optimal quantization direction. When `damp_percent` is too small, the Hessian is near-singular and the inversion produces extreme values that overflow to NaN. Adding a small diagonal damping term (0.1–0.2 of the mean diagonal) regularizes the inversion without meaningfully degrading accuracy.

**Why others are wrong:**
- A: Calibration dataset size affects the quality of Hessian estimation but not NaN instability — you would get slightly noisier Hessians, not NaN.
- C: GPTQ works on BF16 models natively; the conversion to FP32 is handled internally.
- D: `desc_act` controls whether weights are reordered by activation magnitude; it is independent of numerical stability.

---

## Q2

**Answer: B**

**Why correct:** AWQ identifies salient channels by measuring which input channels produce the largest activation magnitudes across the calibration set. If the calibration data is Wikipedia text, the language model activates a different set of channels than when processing SQL queries (which heavily use specific token patterns for table names, JOIN keywords, aggregations). Protecting the wrong salient channels leaves the SQL-critical channels under-protected, causing higher quantization error on SQL tasks.

**Why others are wrong:**
- A: Calibration data does not affect perplexity computation; it affects which weights are protected from aggressive rounding.
- C: AWQ quantization quality is directly determined by calibration — this is its entire design premise.
- D: The tokenizer is fixed regardless of calibration data and does not interact with the AWQ channel-selection step.

---

## Q3

**Answer: C**

**Why correct:** Q4_K_M GGUF with llama.cpp is the only format that runs efficiently on CPU with no GPU. It is designed for exactly this scenario — the entire llama.cpp project exists to enable capable inference on consumer hardware. 8 GB RAM comfortably fits a Q4_K_M 7B model (~4.5 GB) with room for OS and KV cache.

**Why others are wrong:**
- A and B: AWQ and GPTQ require CUDA GPU at inference time for their optimized kernels.
- D: BF16 with 8-bit bitsandbytes still requires a CUDA GPU; bitsandbytes does not support CPU inference.

---

## Q4

**Answer: B**

**Why correct:** When you run llama.cpp in its default configuration, it performs inference on CPU. The GGUF weights are loaded into RAM and dequantized to F32 on CPU before each matrix multiply. AWQ uses GEMM kernels that run the dequantize-and-multiply step natively on GPU tensor cores, which are orders of magnitude faster for this pattern. The 85 vs 38 tok/s difference is a hardware comparison, not a quality comparison.

**Why others are wrong:**
- A: Q4_K_M is actually slightly less compressed per bit than AWQ GEMM due to mixed-precision overheads, not more.
- C: AWQ stores weights as INT4 integers, not FP16.
- D: GGUF does support batched inference; it is the CPU vs GPU hardware difference that dominates.

---

## Q5

**Model answer:** A LoRA adapter checkpoint stores only the low-rank delta matrices (A and B) for selected layers — it does not contain the full merged weight matrices. Quantization tools (GPTQ, AWQ, llama.cpp conversion) operate on complete weight tensors W = W_base + BA. If you pass an unmerged adapter, the quantization tool sees only W_base and ignores the adapter, producing a quantized model that has effectively undone all your fine-tuning. The model will load and run, but it will behave like the quantized base Qwen2.5 model rather than your SQL-specialized model. You must call `model.merge_and_unload()` (for PEFT) or `model.save_pretrained()` with Unsloth's merge option before any quantization pipeline.

---

## Q6

**Model answer:** Two hypotheses: First, GPTQ may be using a less accurate Hessian estimate because the 512-example calibration set is insufficient for a 7B model's 32 layers — the Hessian estimation for later layers degrades as calibration activations become less diverse. Experiment: double the calibration set to 1,024 examples and re-run GPTQ; if perplexity improves toward AWQ levels, calibration size was the bottleneck. Second, AWQ's salient-channel protection specifically preserves the most information-critical weights, whereas GPTQ treats all channels equally in its Hessian calculation — on SQL data with concentrated activation patterns (a few channels dominate for SQL keywords), AWQ's approach naturally fits better. Experiment: run both methods on a generic text benchmark (WikiText-103); if the GPTQ gap disappears on generic text but remains on SQL, the domain concentration hypothesis is confirmed.

---

## Q7

**Model answer:** The five essential pieces of information for reproduction are: (1) the exact base model and revision hash (e.g., `Qwen/Qwen2.5-Coder-7B-Instruct` at commit `abc123`), (2) the quantization method, library version, and all non-default hyperparameters used (`AutoAWQ 0.2.x`, `q_group_size=128`, `w_bit=4`, `zero_point=True`), (3) the calibration dataset source and size (e.g., "512 examples from `<your-handle>/sqlcoder-v3-train`, first split"), (4) the evaluation benchmark and metric definition (your custom 200-example set with exact-match SQL accuracy, link to benchmark file), and (5) the hardware and software environment (GPU model, CUDA version, library versions). Without all five, another researcher cannot reproduce your numbers.

---

## Q8 — Deep Scenario

**Model answer:** Recommendation: AWQ INT4.

AWQ satisfies all three requirements: it runs on an A10G with 5.0 GB VRAM (well within the 8 GB limit, leaving 3 GB for KV cache growth under concurrent requests), achieves 85 tok/s (exceeding the 80 tok/s target), and shows only 0.3 percentage point accuracy degradation (within the 2 pp tolerance).

GPTQ is eliminated because it falls below the 80 tok/s threshold at 72 tok/s — a real-time SQL assistant at 72 tok/s will feel sluggish for queries that require 200+ token outputs. Q4_K_M GGUF is eliminated because 38 tok/s on CPU is far below target and there is no path to GPU GEMM with llama.cpp in standard configuration.

The one remaining risk with AWQ is that the 0.3 pp accuracy gap was measured on your 200-example benchmark, which may not cover the full distribution of production queries. TimescaleDB-specific queries (time-series aggregations, `time_bucket` syntax) were likely underrepresented in your benchmark, and these may see higher quantization error due to rare token patterns.

Monitoring strategy: log every SQL query and generated output in production. Daily, run 50 sampled query-output pairs through the BF16 reference model and compute semantic equivalence using your LLM-judge from Week 55. If the agreement rate drops below 90%, trigger a re-evaluation of the full 200-example benchmark and flag for potential re-quantization with a larger calibration set.
