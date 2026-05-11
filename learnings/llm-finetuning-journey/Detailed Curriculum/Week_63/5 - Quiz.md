# Week 63 Quiz — Quantization Comparison

## Multiple Choice

**Q1.** For a 7B model generating single tokens auto-regressively (batch size = 1), which hardware resource is the primary bottleneck?

A) GPU FLOPS — the matrix multiplications in attention and FFN layers.
B) Memory bandwidth — loading 7B model weights for each token generation step.
C) PCIe bandwidth — data transfer between CPU and GPU.
D) Cache memory — the KV cache for long contexts.

---

**Q2.** You quantize your model to Q4_K_M GGUF and run it on your Mac. You get 18 tok/s. The bf16 model on A100 achieves 85 tok/s. Which statement best explains the speed difference?

A) The A100 has more CUDA cores and therefore computes attention faster.
B) The A100 has higher memory bandwidth (900 GB/s vs Mac's 68 GB/s) which is the primary bottleneck for inference.
C) GGUF quantization is slower than native PyTorch because it requires dequantization at each step.
D) The Q4_K_M model is actually slower than bf16 due to the overhead of dequantization.

---

**Q3.** AWQ uses "salient channel protection" to improve quantization quality. Which weights does it protect and why?

A) The first and last layers — these are most important for input and output quality.
B) Attention query and key projections — these produce the largest gradients during training.
C) Weights corresponding to channels with consistently large activation magnitudes — because quantizing these causes disproportionate output error.
D) The biases of all linear layers — biases are small and easily corrupted by quantization.

---

**Q4.** You use WikiText-2 (a general English corpus) as calibration data for GPTQ quantization of your SQL-specialized model. Why might this produce worse SQL accuracy than using your domain calibration data?

A) WikiText-2 is copyrighted and its use in calibration is legally problematic.
B) The Hessian approximation in GPTQ is computed on calibration data; if this data doesn't reflect the model's actual use distribution, the quantization minimizes reconstruction error for the wrong distribution.
C) WikiText-2 contains SQL queries that confuse the GPTQ algorithm.
D) GPTQ only works correctly with code-specific calibration data.

---

## Short Answer

**Q5.** Your Q4_K_M model achieves 63% execution accuracy on BIRD-SQL while the bf16 model achieves 65%. Is this 2pp drop acceptable? How does your answer change depending on the deployment scenario?

---

**Q6.** A user asks: "I have 8GB of RAM on my laptop and no GPU. Can I run your 7B model?" Which quantization format should you recommend and why? What is the practical user experience (speed, latency)?

---

## Deep Scenario

**Q7.** Your quantized Q4_K_M model (63% on BIRD-SQL) drops to 52% on your TimescaleDB-specific subset, while the bf16 model scores 70% on the same subset. This is an 18pp drop (much larger than the 2pp drop on BIRD-SQL overall). Diagnose why quantization disproportionately affects TimescaleDB accuracy and propose a targeted fix that does not require switching to a less compressed format.
