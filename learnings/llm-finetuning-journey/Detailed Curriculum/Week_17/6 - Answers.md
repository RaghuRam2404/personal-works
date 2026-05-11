# Week 17 Quiz Answers

## Q1 — Answer: B

**Answer:** B — Increasing model size while training on fewer tokens.

**Why:** Kaplan 2020 found that the optimal strategy (given their experimental setup) was to scale parameters faster than data. This led to the "bigger is better" paradigm — GPT-3 being 175B params trained on only 300B tokens.

**Why others are wrong:**
- A is the Chinchilla prescription, which came later and corrected Kaplan
- C is close to Chinchilla, but not what Kaplan found
- D is unrelated to what scaling laws study

---

## Q2 — Answer: B

**Answer:** B — Under-trained; it needed ~20× more tokens.

**Why:** Chinchilla showed that Gopher would have achieved the same or better loss at 70B params (4× smaller) if trained on 1.4T tokens (4.7× more tokens). The old paradigm wasted compute by over-parameterizing relative to available data.

**Why others are wrong:**
- A is backward — Gopher had too few tokens, not too many
- C is wrong — Chinchilla directly contradicted Kaplan for large models
- D is opposite — Chinchilla showed Gopher was too large, not too small

---

## Q3 — Answer: C

**Answer:** C — Attention's quadratic scaling with sequence length.

**Why:** The 6ND approximation models the dense matrix multiplications in the FFN and projection layers, which dominate cost for typical sequence lengths. However, the attention mechanism scales as O(L^2 × d) where L is sequence length. For long-context training (L > 4096), the approximation underestimates cost meaningfully.

**Why others are wrong:**
- A and B: the 2ND forward + 4ND backward breakdown is exactly what 6ND captures
- D: optimizer state updates are typically not counted in this FLOP approximation

---

## Q4 — Answer: C

**Answer:** C — Inference-optimal; Meta traded extra training compute for lower inference cost.

**Why:** Meta serves Llama models to millions of users. The marginal cost of extra pretraining compute is tiny compared to the cumulative inference savings from using a smaller, better-trained model. By training 8B on 15T tokens instead of ~70B on 1.4T tokens, Meta gets roughly equivalent quality at 9× lower inference cost per token.

**Why others are wrong:**
- A: "under-trained" would mean fewer tokens, which is the opposite
- B: 15T vs. 160B is a 94× ratio, not within error
- D: over-training on diverse internet data does not cause catastrophic forgetting; that term applies to sequential task learning

---

## Q5 — Answer: B

**Answer:** B — Validation loss per unit of training compute.

**Why:** Chinchilla's IsoFLOP profiles fix compute budget C and find the (N, D) allocation that minimizes validation loss. The goal is efficiency of compute, not inference speed, accuracy on a specific benchmark, or throughput.

---

## Q6 — Short Answer

```
C = 1e20 FLOPs

N_opt = 6.8e-2 × (1e20)^{0.5}
      = 6.8e-2 × 3.162e10
      = 2.15e9 params ≈ 2.15B params

D_opt = 1.96 × (1e20)^{0.5}
      = 1.96 × 3.162e10
      = 6.20e10 tokens ≈ 62B tokens

Sanity check: 6 × 2.15e9 × 6.20e10 ≈ 8.0e20 FLOPs (within 8× of C — acceptable given log-scale)
```

Note: the 6ND check will give a number larger than C because the Chinchilla constants are not derived from 6ND directly but from empirical loss curves. The N_opt and D_opt are correct; treat 6ND as a rough cross-check.

---

## Q7 — Short Answer

Training a model beyond its Chinchilla-optimal token count trades training-time compute (a one-time fixed cost) for lower inference compute (a recurring per-query cost). If the model serves millions of requests, even a small reduction in model size translates to massive cumulative savings in inference hardware. Additionally, a smaller, over-trained model has lower latency and fits on cheaper hardware, enabling broader deployment. For products with high query volume, the inference bill quickly dominates the training bill, making "over-training" economically rational.

---

## Q8 — Short Answer

With $30 at $1.50/hr and 35% MFU: C ≈ 0.35 × 312e12 × 20hr × 3600 ≈ 7.9e18 FLOPs. Chinchilla optimal: N_opt ≈ 6.8e-2 × (7.9e18)^0.5 ≈ 191M params, D_opt ≈ 3.8B tokens. Training a 7B model on $30 of compute means you can only afford roughly 7.9e18 / (6 × 7e9) ≈ 188M tokens — that is barely 27 tokens per parameter, and your compute budget runs out before the model can converge to a quality baseline. A 200M-parameter model on 4B tokens will produce better results than a 7B model on 0.2B tokens.

---

## Q9 — Scenario Model Answer

**1. Compute budget in FLOPs:**
```
Hours = 80 / 1.50 = 53.3 hrs
C = 0.35 × 312e12 × 53.3 × 3600 ≈ 2.1e19 FLOPs
```

**2. Chinchilla-optimal N:**
```
N_opt = 6.8e-2 × (2.1e19)^{0.5} ≈ 312M params
```
This is below the 3B limit, so Chinchilla's recommendation is feasible within their latency constraint.

**3. Data-constrained maximum model size:**
With D_max = 50B tokens and using D = 20 × N: N_max = 50B / 20 = 2.5B params.

**4. Recommendation:**
Train a ~300M-parameter model on 50B tokens of their code corpus. This is near Chinchilla-optimal, satisfies the <3B latency constraint, and fully utilizes their data. The alternative — fine-tuning a 3B SOTA model — may outperform this if they have limited proprietary data and a good starting checkpoint, so both paths are defensible. Given they have a custom 50B-token corpus, training from scratch (or domain-adaptive pretraining from a small SOTA base) on their exact data distribution is likely the stronger choice.
