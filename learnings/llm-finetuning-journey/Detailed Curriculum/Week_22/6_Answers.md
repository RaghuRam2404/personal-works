# Week 22 Quiz Answers

## Q1 — Answer: C

**Answer:** C — 33.1.

**Why:** Perplexity = exp(cross_entropy_loss) = exp(3.5) = 33.1. This is the standard formula. Loss is measured in nats (natural log base), so exp() reverses the log.

**Why others are wrong:**
- A (3.5): confusing the loss with the perplexity directly
- B (16.4): this would be exp(2.8), not exp(3.5)
- D (101.2): this would require a loss of ~4.6

---

## Q2 — Answer: B

**Answer:** B — Average the log-likelihoods (losses), then exponentiate.

**Why:** The correct definition of perplexity over a sequence is exp(-(1/N) × Σ log P(x_i | x_<i)), where the sum is over all tokens. This is equivalent to averaging per-token cross-entropy (which is what averaging per-batch losses gives you, assuming equal batch sizes). Your colleague's approach `mean(exp(loss_i))` computes the arithmetic mean of perplexities, which is dominated by high-perplexity batches and gives a biased estimate. Averaging in log space (then exponentiating) computes the geometric mean of perplexities, which is the correct definition.

**Why others are wrong:**
- A: the colleague's approach is wrong; geometric mean is correct, but it is computed by exponentiation of the average log, not by averaging individual exponents
- C: they give different results; Jensen's inequality says E[exp(X)] ≥ exp(E[X]) for convex exp
- D: this is computationally impractical and not required by the definition

---

## Q3 — Answer: C

**Answer:** C — Repetitive, deterministic output converging to greedy decoding.

**Why:** Temperature T < 1 sharpens the softmax distribution by dividing logits by T before softmax. As T→0, the distribution becomes a point mass at the argmax (greedy decoding). With T=0.1, nearly all probability mass is on the top 1–2 tokens, causing the model to repeatedly generate the same high-probability continuations — which are often common phrases, resulting in loops.

**Why others are wrong:**
- A: low temperature produces less diversity, not more
- B: generation speed depends on sequence length, not temperature
- D: perplexity is a property of the model-data pair, not the sampling procedure

---

## Q4 — Answer: B

**Answer:** B — The model assigns lower probability (is more surprised) by SQL tokens than by English tokens.

**Why:** Perplexity measures how surprised the model is on average. A perplexity of 58 on SQL means the model is behaving as if it is choosing among ~58 equally likely options at each SQL token, compared to ~32 options for English. Since the model was trained predominantly on English web text, it learned the statistical patterns of English far better than SQL — exactly what a higher perplexity indicates.

**Why others are wrong:**
- A: higher perplexity means WORSE performance, not better
- C: SQL is not inherently harder to model; SQL models achieve very low perplexity on SQL data
- D: data contamination would show up as anomalously LOW perplexity, not high

---

## Q5 — Answer: B

**Answer:** B — Selects from the 50 highest-probability tokens, setting others to -infinity before softmax.

**Why:** Top-k sampling at each step: sort logits in descending order, keep the top k values, set all others to -infinity, then apply softmax and sample. This prevents the model from ever generating very-low-probability tokens that are unpredictable and often garbage.

**Why others are wrong:**
- A: top-k is computed per position, not globally from training data
- C: sampling still introduces randomness among the top-k — it does not always pick top-1
- D: top-k works with multinomial sampling independently of beam search

---

## Q6 — Short Answer (PPL=35, expected ~24.5)

1. **Evaluation data distribution mismatch:** Your val.bin may be from a different FineWeb-Edu shard than what you trained on. If the val set has different topic distributions (more technical text, different style), perplexity will be higher. Check: compute perplexity on a small slice of train.bin — if train perplexity is much lower (~20), the model has domain-specialized to the train distribution.

2. **Checkpoint is not the best (loaded wrong step):** Your training monitor showed val loss 3.2 at the end, but you may have saved the "latest" checkpoint rather than the "best" checkpoint. If training overfit slightly in the final steps, the best checkpoint at an earlier step would give lower perplexity. Check: does your best_val_loss from the W&B run match 3.2 or was it lower at an intermediate step?

3. **Evaluation batch size or stride error:** If your eval function uses overlapping windows or a different stride than expected, the effective token count changes and the loss estimate is biased. Check: manually compute perplexity on a 100-token snippet by hand and compare to your eval.py output.

---

## Q7 — Short Answer

Perplexity is defined over tokens, not characters or words. If two tokenizers segment the same text differently, they produce different numbers of tokens, and each token predicts a different set of possible continuations. A 50K-vocabulary tokenizer tends to produce fewer tokens per sentence (it merges more aggressively), so each token carries more information. A model using a 50K tokenizer must be more confident at each step (lower loss per token) to achieve the same word-level perplexity as a model using a 32K tokenizer. To compare models with different tokenizers fairly, you must normalize by computing bits-per-character (BPC) or bits-per-word (BPW) instead of perplexity.

---

## Q8 — Short Answer

Increase temperature (e.g., from 0.7 to 1.0–1.2) and increase top_k (e.g., from 20 to 100 or 200). The repetitive sentence patterns indicate that the model is concentrating probability mass on a small set of high-frequency continuations. Higher temperature flattens the distribution, giving more weight to less common but valid continuations. Higher top_k expands the candidate set. If repetition persists, add a repetition penalty: before sampling, reduce the logit of any token that appeared in the last 20 generated tokens by a factor of 1.3.

---

## Q9 — Scenario Model Answer

**1. What accounts for the 1.78× perplexity gap:**
GPT-2-small has 2.1× more parameters (117M vs. 56M) AND was trained on 20× more tokens (40B vs. 2B). More parameters give the model more capacity to store statistical patterns; more tokens give better calibration of those patterns. The combination of 2× parameters and 20× tokens easily explains the 1.78× perplexity improvement. Additionally, GPT-2 used a 50K vocabulary (slightly higher resolution) and was trained with a more carefully tuned pipeline.

**2. Why not fine-tune your 50M model:**
(a) Capacity: 50M parameters is insufficient to memorize and generalize complex SQL semantics. A 5K-example fine-tuning dataset adds maybe 50M "useful bits" of SQL knowledge — crammed into a model that barely knows SQL from pretraining. The model will catastrophically forget its general language ability while only partially learning SQL. (b) Starting knowledge: Qwen2.5-Coder-7B has already been pretrained on trillions of code tokens including SQL. Fine-tuning it with 5K domain-specific examples amplifies strong existing knowledge. Fine-tuning your 50M model is teaching SQL to someone who has never seen a database; fine-tuning Qwen2.5-Coder is teaching PostgreSQL specifics to an expert SQL developer.

**3. SQL generation diagnosis:**
This is likely a model capability issue, not a generation settings issue. Your model was trained primarily on general English web text and has seen very little SQL. However, you should test with lower temperature (0.3) and smaller top_k (10) to see if the model can produce valid SQL when forced to be more conservative. If temperature=0.3 also produces invalid SQL, the model simply does not know SQL syntax well enough — this is expected for a 50M model trained on web text.

**4. What to do differently:**
Include more code data (5–10% of training tokens from GitHub code), specifically SQL files. Train on at least 1B unique tokens (not repeated epochs of 100M). Use a larger model (125M params is feasible in the same compute budget with fewer tokens). Consider using the GPT-2 tokenizer to enable comparison with published benchmarks.
