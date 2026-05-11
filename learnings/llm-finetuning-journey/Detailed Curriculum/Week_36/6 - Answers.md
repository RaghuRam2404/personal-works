# Week 36 Quiz Answers

## Q1 — Answer: C

**Answer:** C. Both magnitude (via learned scalar) and direction (via BA).

**Why:** DoRA's core insight is that full fine-tuning adapts both the magnitude and direction of weight vectors, while standard LoRA adapts primarily direction. DoRA explicitly decomposes W into W = m × (V/||V||) and then: (1) learns a scalar magnitude vector m (one scalar per row/output dimension, initialized to the norms of pretrained weight rows), and (2) learns a low-rank direction update BA applied to the normalized V/||V||. Both components are trainable, enabling DoRA to more closely approximate the behavior of full fine-tuning.

---

## Q2 — Answer: B

**Answer:** B. 16 / sqrt(64) = 2.0

**Why:** RSLoRA replaces the standard `alpha / r` scaling with `alpha / sqrt(r)`. With alpha=16 and r=64: scaling = 16 / sqrt(64) = 16 / 8 = 2.0. For comparison, standard LoRA with alpha=16 and r=64 would give 16/64 = 0.25 — the LoRA contribution would be heavily attenuated, making high-rank LoRA ineffective. RSLoRA's 2.0 scaling keeps the adapter contribution meaningful regardless of rank.

---

## Q3 — Answer: B

**Answer:** B. Chat template format mismatch — the 4.5 represents task difficulty.

**Why:** At step 0 with B=0, the QLoRA model behaves identically to the quantized NF4 base model (since the adapter contributes nothing). The NF4 base loss should be close to 2.4 (similar to BF16 base's 2.1, with small quantization error). If the loss is 4.5, it means the model is being evaluated on output tokens that are very unfamiliar — specifically, the SQL answers formatted with the ChatML template are not the natural output distribution of the base model. This is expected and represents the training task difficulty, not an initialization problem. Loss will drop as the model learns the SFT format.

---

## Q4 — Answer: B

**Answer:** B. Truncated SVD of the quantization error matrix.

**Why:** LoftQ computes E = W_fp16 - W_nf4 (the quantization error). Then applies SVD to E: E = U Σ V^T. Truncating to the top-r singular values gives the best rank-r approximation: E_r = U_r Σ_r V_r^T. Setting B_init = U_r × sqrt(Σ_r) and A_init = sqrt(Σ_r) × V_r^T gives B_init × A_init = E_r ≈ E. The scaling factor (alpha/r) adjusts the magnitude. This is the most principled approach to compensating for quantization error at initialization.

---

## Q5 — Answer: C

**Answer:** C. DoRA at rank 16.

**Why:** The DoRA paper's experiments consistently show that DoRA outperforms standard LoRA on supervised fine-tuning tasks (commonsense reasoning, visual instruction tuning, natural language tasks) at the same rank. The overhead is ~5–10% in compute and minimal in memory. For a PostgreSQL SQL task (which is a supervised fine-tuning task), DoRA is the first variant to try. LoftQ is not prioritized because Qwen2.5-Coder-7B with NF4 quantization shows small quantization error; RSLoRA at rank 64 adds significant parameter count and memory overhead for uncertain gain on 5K examples.

---

## Q6 — Short Answer

With alpha fixed and rank increasing:
- Standard LoRA: scaling = alpha / r. As r doubles (8→16→32→64), scaling halves. At r=64 with alpha=16: scaling = 0.25 — the adapter contributes only 25% of the magnitude it contributes at r=4 with the same alpha. This means very high ranks make the LoRA contribution negligible.
- RSLoRA: scaling = alpha / sqrt(r). As r doubles, scaling decreases only by sqrt(2) ≈ 1.41. At r=64 with alpha=16: scaling = 2.0. The contribution stays substantial even at large ranks.

RSLoRA is "rank-stabilized" because the scaling does not collapse to near-zero at high ranks, allowing the optimizer to use the full expressiveness of high-rank adapters.

---

## Q7 — Short Answer

Factors to consider before recommending DoRA as default:

1. **Statistical significance:** A 3% exact match difference (48% vs. 45%) on 100 examples corresponds to 3 additional correct answers. Confidence interval for this difference: ±4–5% at 95% confidence — the difference may not be statistically significant. Run the comparison on 500+ held-out examples or 3 independent seeds.

2. **Training overhead:** DoRA adds ~5–10% compute overhead. For the Week 38 15K sprint at 3 steps/second, this adds ~1–2 minutes — negligible. But for Phase 5–6 runs with 100K+ examples, this cost accumulates.

3. **Generalization vs. this dataset:** DoRA's improvement on your SQL dataset may be specific to the schema diversity or question types. Before defaulting, test on at least 2 different SQL subsets or task types.

Recommendation: use DoRA in Week 38 (low cost, likely better), but verify with more examples before declaring it the permanent default.

---

## Q8 — Short Answer

The misunderstanding is about what "starting from the pretrained baseline" means in QLoRA. In standard QLoRA (B=0), the model starts from the pretrained NF4 baseline — but that baseline already has quantization error: the NF4 model is not identical to the BF16 base. The "pretrained baseline" in QLoRA is already a lossy approximation.

LoftQ's initialization compensates for this: after initializing B_init and A_init, the combined model (W_nf4 + B_init × A_init × scaling) more closely approximates the original FP16 weights than W_nf4 alone. The model now starts from a better approximation of the true pretrained baseline, not a degraded one.

The training stability concern is valid if the LoftQ initialization is incorrect (e.g., if the magnitude of B_init × A_init is very large). But by construction, B_init × A_init is the rank-r approximation of the quantization error, which is small relative to the full weight values — it does not destabilize training.

---

## Q9 — Scenario Answer

Given results: DoRA rank 16 (loss 1.15, ~43M trainable, +5% time), standard LoRA rank 16 (loss 1.21, ~42M), RSLoRA rank 64 (loss 1.13, ~160M, ~4× memory for adapters).

**Recommendation: DoRA at rank 16 for Week 38.**

Reasoning:
- RSLoRA rank 64 achieves the best eval loss (1.13) but at a significant cost: 160M trainable parameters increase optimizer state memory by ~640MB and training time by ~50%. With 15K examples on A100, this is manageable, but the benefit over DoRA (1.13 vs. 1.15 = 0.02 difference) is marginal and may not hold on larger test sets.
- DoRA rank 16 achieves 1.15 eval loss with minimal overhead (+5% time, negligible VRAM). On 15K examples (3× more data than the 5K sweep), DoRA may show larger gains over standard LoRA because the richer dataset amplifies the benefit of magnitude+direction decomposition.
- Standard LoRA rank 16 is the safe baseline but underperforms both alternatives.

Use DoRA rank 16 for Week 38. If execution correctness on the held-out test after Week 38 is unsatisfactory, retry with RSLoRA rank 32 as the next candidate.
