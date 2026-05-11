# Week 17 — Scaling Laws

## Learning Objectives

By the end of this week, you will be able to:

- Explain the Kaplan (2020) scaling law findings and their limitations
- Apply the Chinchilla (2022) compute-optimal formula to choose model size and token count for a given compute budget
- Distinguish between compute-optimal training and inference-optimal training
- Calculate the approximate FLOP cost of a training run given model parameters and tokens
- Identify why over-parameterized, under-trained models were common before Chinchilla

---

## Concepts

### What Are Scaling Laws?

Scaling laws describe how model performance (measured as loss on a held-out set) changes predictably as you increase model parameters (N), training tokens (D), or compute (C). If you can predict loss from N and D, you can plan experiments without running them.

The foundational insight is that loss follows a power law in all three quantities. Before Chinchilla, practitioners had an empirical rule-of-thumb: bigger is better. After Chinchilla, the field understood that "bigger" needs to be paired with "more data" in a precise ratio.

### Kaplan et al. 2020 — The Original Scaling Laws

The Kaplan paper ([Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)) identified the following:

**Power-law relationships:**

```
L(N) ~ N^{-0.076}     (fixed data, vary params)
L(D) ~ D^{-0.095}     (fixed params, vary data)
L(C) ~ C^{-0.050}     (optimal allocation)
```

Where L is cross-entropy loss, N is non-embedding parameter count, D is dataset token count, and C is compute in FLOPs.

The key finding: for a fixed compute budget, Kaplan recommended allocating most of the budget to parameters, training for fewer tokens. This led to models like GPT-3 (175B params, trained on ~300B tokens) — massively over-parameterized by Chinchilla standards.

**Critical limitation of Kaplan:** the paper held batch size constant and did not fully optimize over (N, D) jointly. It underestimated the value of more data.

### Chinchilla (Hoffmann et al. 2022) — The Compute-Optimal Revision

The Chinchilla paper ([Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)) ran ~400 models to find the true optimal frontier. Their key result: for compute-optimal training, **you should scale tokens and parameters equally**.

The rule of thumb: train on approximately **20 tokens per parameter**.

```
N_opt ∝ C^{0.50}
D_opt ∝ C^{0.50}
```

This means:
- If you have C FLOPs, split roughly equally between params and tokens
- Chinchilla-70B (trained on 1.4T tokens) outperforms Gopher-280B (trained on 300B tokens) despite being 4x smaller

**The practical formula for compute cost:**

A useful approximation for transformer training FLOP count is:

```
C ≈ 6 × N × D
```

Where the factor 6 accounts for forward pass (2ND), backward pass (4ND), and the standard memory bandwidth. This is a rough approximation that holds well for large models where embeddings are a small fraction of parameters.

**Deriving optimal N and D from a compute budget:**

Given compute budget C (FLOPs):

```
N_opt ≈ (C / 6)^{0.5} × (D_ratio)^{-0.5}
D_opt ≈ 20 × N_opt
```

Or more concretely using Chinchilla's fitted constants:

```
N_opt = a × C^{0.50}
D_opt = b × C^{0.50}
```

where a ≈ 0.083 and b ≈ 1.96 (from Table A3 of the Chinchilla paper, approach 3).

**Worked example — your $50 compute budget:**

A100 80GB on RunPod costs roughly $1.50/hr. With $50:

```
Hours = 50 / 1.5 ≈ 33 hrs
FLOP/s (A100, FP16, ~40% MFU) ≈ 0.4 × 312e12 ≈ 1.25e14 FLOP/s
C = 1.25e14 × 33 × 3600 ≈ 1.5e19 FLOPs
```

Applying Chinchilla:
```
N_opt ≈ 0.083 × (1.5e19)^{0.5} ≈ 0.083 × 1.22e9.5 ≈ 320M params
D_opt ≈ 20 × 320M ≈ 6.4B tokens
```

So with $50 you should train roughly a 300M-parameter model on 6B tokens — not a 7B model on 1B tokens, and not a 1B model on 100M tokens.

### Compute-Optimal vs. Inference-Optimal

Chinchilla optimizes for loss per FLOP — i.e., given a fixed compute budget, what training run minimizes final loss? But this is not always the right objective.

If you will run the model for inference millions of times, the total cost is:

```
Total cost = Training cost + N_inference × Inference cost per token
```

A smaller model trained on more tokens is cheaper to run at inference time. This is why Llama 3 8B was trained on 15T tokens (far beyond Chinchilla-optimal) — Meta's inference compute dwarfs their training compute.

**The takeaway:** Chinchilla is the baseline. Inference-heavy use cases justify over-training smaller models.

### Scaling Law Applicability to Fine-Tuning

Scaling laws were derived for pretraining. They do not directly apply to fine-tuning — a 50M-parameter model fine-tuned on domain data does not follow the same power laws. However, understanding scaling laws tells you:

- Why the base model quality matters so much (larger base → better fine-tuning ceiling)
- How to estimate the compute cost of continued pretraining or domain-adaptive pretraining
- What "compute budget" means when your advisor asks you to justify your Phase 6 model choice

---

## Connections

**Prior weeks:** Week 1–16 (DL fundamentals, transformers, attention). You now apply that knowledge to understand why bigger transformers are better up to a point.

**Later weeks:** Weeks 20–22 will have you train a 50M model. Your Chinchilla calculation will justify that size. Week 24 (reading SOTA recipes) will show you how Llama 3 and Qwen2.5 made deliberate decisions to train smaller models on more tokens.

---

## Common Misconceptions

- **"Bigger model always wins."** Only if you have proportionally more data. A 7B model trained on 10B tokens will underperform a 1B model trained on 20B tokens in many situations.
- **"Chinchilla says train on exactly 20 tokens per parameter."** The 20× rule is a heuristic from one approach in the paper. The actual optimal ratio depends on the compute budget and the power-law constants, which vary by architecture and data.
- **"I should apply Chinchilla to my fine-tuning run."** Chinchilla applies to pretraining on a large general corpus. Fine-tuning dynamics are different — you are not in the power-law regime with 5K examples.
- **"6N FLOPs is exact."** It is an approximation. Attention layers scale as O(N_seq^2) and have additional costs. For long-context training it underestimates.

---

## Time Allocation (6–8 hrs)

- 2h: Read Kaplan 2020 (Sections 1–4 are enough; skim the rest)
- 2h: Read Chinchilla 2022 (focus on Sections 2–4 and Table A3)
- 1h: Watch Yannic Kilcher's Chinchilla walkthrough
- 1.5h: Write your deliverable — the compute budget writeup applying Chinchilla to your $50 Phase 6 budget
- 0.5h: Sanity-check your math against published model cards (GPT-3, Llama-1, Chinchilla)
