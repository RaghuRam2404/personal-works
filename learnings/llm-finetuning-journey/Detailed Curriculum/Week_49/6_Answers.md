# Week 49 Quiz Answers

## Q1. Answer: C

**Answer:** C — KTO is designed for unpaired good/bad annotation datasets.

**Why:** KTO's core innovation is that it does not require (prompt, chosen, rejected) triples — it works with (prompt, completion, label) pairs where labels are simply "desirable" or "undesirable." Your dataset of "good SQL" and "bad SQL" for different prompts fits this format exactly. DPO would require you to pair good and bad examples for the same prompt, which you cannot do here. GRPO requires online generation, not a static dataset. ORPO requires paired preferences, not unpaired.

---

## Q2. Answer: B

**Answer:** B — Without a reference model, there is no KL regularization anchor.

**Why:** DPO and GRPO both include a reference model that the KL penalty is computed against. This anchor prevents the model from drifting too far from the pretrained distribution. ORPO's odds ratio penalty is a local (per-step) regularizer but does not anchor to the pretrained weights in the same way. As a result, ORPO-trained models can lose general language quality more rapidly than DPO-trained models, especially when the preference signal is noisy.

---

## Q3. Answer: B

**Answer:** B — Length normalization prevents systematic preference for shorter or longer completions.

**Why:** In DPO, the log-probability of a completion is summed over all tokens. A long completion naturally has a lower (more negative) log-probability than a short one, even if per-token probabilities are similar. This causes DPO to implicitly prefer shorter responses. SimPO divides by |y|, making the reward a per-token average — now long correct SQL and short correct SQL are compared fairly. For SQL, where complex queries are necessarily longer, this matters.

---

## Q4. Answer: B

**Answer:** B — DPO is offline and cannot adapt to new SQL patterns; GRPO's online rollouts provide fresh rewards at every step.

**Why:** This is the fundamental DPO limitation for dynamic domains. Your v3 model at step 1000 is a different distribution than v2 — it generates different SQL patterns. DPO trained on Week 44's preference pairs was optimal for v1's output distribution. As the model improves, the preference data becomes stale, and DPO's optimization increasingly targets patterns the model no longer generates. GRPO generates fresh completions at every step and evaluates them against the Postgres database — it always has current, relevant signal.

**Why others are wrong:**
- A: DPO requires 2 forward passes (training model + ref model); GRPO also requires 2 (training model + ref model for KL). Memory costs are similar.
- C: DPO loss going negative is a signal for mislabeled data, not a permanent failure; GRPO loss can also be negative.
- D: DPO does not require human labelers — your Week 44 dataset used execution-based labeling.

---

## Q5. New Alignment Method — Inferred Tradeoffs

Based on the pattern in KTO, ORPO, and SimPO:

- "No reference model" → borrowed from ORPO and SimPO. The tradeoff: weaker regularization, risk of model drift from pretrained quality. Without an anchor, the model may lose general language fluency.
- "No reward model" → standard since DPO (2023). All of KTO, ORPO, SimPO do this. The tradeoff: requires pre-labeled preference or annotation data; cannot adapt to online rewards.
- "Works with unpaired data" → borrowed from KTO. The tradeoff: less statistical efficiency per training example; removing the pairing discards the contrastive signal between chosen and rejected for the same prompt.

Combined, this method is likely a variant of KTO or SimPO applied without a reference model — similar to ORPO but with unpaired data. The likely tradeoffs: (1) fast and memory-efficient (1× model), (2) works with any labeled dataset regardless of pairing, but (3) prone to distribution drift and cannot use online execution-based rewards. Best used when memory is very constrained and the annotation data is unpaired.

---

## Q6. Data Efficiency: DPO vs. KTO

With 1000 paired preference examples:

**DPO:** Each training step uses N pairs. Each pair contributes 2 forward passes (chosen + rejected through training model) + 2 more (through ref model) = 4 forward passes per pair. With a batch of 8 pairs: 32 forward passes per step.

**KTO:** Each step samples from the desirable and undesirable pools separately. If you have 700 desirable and 300 undesirable examples, each step uses desirable_batch + undesirable_batch. With a batch of 8, you use 8 examples (not 16 like DPO). However, each example only contributes 1 training signal (desirable or not), not a contrastive signal.

**Efficiency:** DPO is more information-efficient per example because the contrastive signal (chosen vs. rejected for the same prompt) teaches the model both what to prefer AND what to avoid in one step. KTO sees each example separately, requiring more steps to learn the same discrimination. For the same number of examples, DPO converges faster when data is paired. KTO wins when data is not paired and DPO cannot be applied at all.

---

## Q7. Production Data — v4 Training Decision

**Data:** 100K positive (successful SQL, different prompts) + 30K negative (corrected SQL, different prompts). Unpaired.

**Decision: KTO.**

The data is unpaired (positive and negative examples are for different prompts), which eliminates DPO, ORPO, and SimPO (all require paired chosen/rejected for the same prompt). GRPO requires generating fresh rollouts — it cannot directly use the static production data. PPO similarly requires online generation + a reward model.

**KTO** accepts exactly this format: a flat list of (prompt, completion, desirable_label) triples. Your 100K successful SQL become desirable examples. Your 30K corrected SQL become undesirable examples (the original wrong SQL before correction). KTO trains the model to increase the probability of desirable completions and decrease the probability of undesirable ones.

**Rejected methods and why:**
- **DPO:** Cannot be used without pairing a positive and negative example for the same prompt. You cannot pair a "user approved SQL" from Monday with a "user corrected SQL" from Tuesday — they answer different questions.
- **GRPO:** Requires generating new completions at training time. Your production SQL is a static dataset — GRPO would ignore it unless you reformulate the prompts and use the production data as a reward calibration tool, not training data.
- **PPO:** Requires a reward model trained separately. You could train a reward model on your production data (SQL that was approved = positive, SQL that was corrected = negative), then use PPO. This is a valid approach but adds a full training stage. KTO achieves a similar result more directly.

**Additional consideration:** Run GRPO (fresh execution rewards) in parallel on new prompts to keep the model improving on current SQL patterns. KTO handles the historical production signal; GRPO handles the current online signal. Combine them in a two-stage pipeline: KTO on production data first, then GRPO for online tuning.
