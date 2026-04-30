# Week 32 Quiz — Quantization Fundamentals

## Multiple Choice

**Q1.** Why does BF16 cause fewer training instabilities than FP16, even though BF16 has lower mantissa precision?

A. BF16 uses fewer bits so gradient computations are faster and less likely to accumulate errors  
B. BF16 has the same exponent range as FP32 (8 bits), preventing overflow that causes FP16 NaN values  
C. BF16 is always computed in higher precision internally by modern GPUs  
D. BF16 rounds to the nearest even number, which reduces bias in weight updates

---

**Q2.** LLM.int8() uses mixed-precision matrix multiplication. When computing a large matrix multiply for a transformer layer, what specifically triggers the switch from INT8 to FP16?

A. When the layer index is greater than 12 (deep layers use higher precision)  
B. When the absolute value of any input activation exceeds a threshold (default: 6.0) — these outlier dimensions are computed in FP16  
C. When the weight matrix has rank greater than the INT8 threshold  
D. When batch size is greater than 1 (batched inference uses FP16 for stability)

---

**Q3.** You have 16GB VRAM. Which configurations can you use to load Qwen2.5-Coder-7B (7B parameters)?

A. BF16 only (requires 14GB)  
B. BF16 (14GB) and INT8 (7GB) only  
C. BF16 (14GB), INT8 (7GB), and NF4 4-bit (~3.5GB) — all fit  
D. Only NF4 4-bit fits; BF16 requires >16GB

---

**Q4.** NF4 is described as "information-theoretically optimal" for LLM weights. What property of LLM weights makes this true?

A. LLM weights are always sparse (many exact zeros), so NF4 can represent zeros perfectly  
B. LLM weights are approximately normally distributed, and NF4 places quantization levels at the quantiles of the standard normal distribution  
C. LLM weights are bounded between -1 and 1, which matches NF4's range  
D. LLM weights have low rank, and NF4 exploits this structure

---

**Q5.** GPTQ quantizes weights to 4-bit with calibration data, adjusting remaining weights to compensate for quantization error. AWQ also achieves high-quality 4-bit quantization. What is the primary practical advantage of AWQ over GPTQ?

A. AWQ uses fewer bits (3-bit) than GPTQ (4-bit), so models are smaller  
B. AWQ does not require a calibration dataset  
C. AWQ's quantization is faster to apply and does not require computing the inverse Hessian matrix  
D. AWQ supports activation quantization while GPTQ only quantizes weights

---

## Short Answer

**Q6.** A colleague says: "We should use INT8 quantization for our 7B inference server because it cuts memory in half." You agree on the memory saving but raise a concern about throughput. What is the concern, and under what GPU architecture does it become significant?

---

**Q7.** Explain double quantization (from the QLoRA paper) in 2–3 sentences. What is quantized, what is the memory saving per parameter, and what is the potential quality trade-off?

---

**Q8.** You quantize Qwen2.5-Coder-7B to 4-bit NF4 and run it on 50 held-out SQL examples. The base model (BF16) scores 40% exact match; the NF4 model scores 37% exact match. Is this quality drop acceptable for your use case? Give a concrete decision framework for when to accept vs. reject a quantization approach.

---

## Scenario

**Q9.** Your team is deploying a SQL text-to-SQL inference API. You have: one A100 (40GB VRAM), 500 queries/day volume, latency SLA of <2 seconds per query, and quality requirement of >85% execution correctness on your held-out test set. 

Your current BF16 7B model achieves 88% execution correctness but only serves 1 query at a time (cannot batch). List the quantization options you would evaluate, explain the experiment you would run to choose, and give your recommendation.
