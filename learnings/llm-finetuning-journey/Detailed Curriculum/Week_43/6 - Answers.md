# Week 43 Quiz Answers

## Q1. Answer: B

**Answer:** B — Z(x) cancels when computing r(y_w) − r(y_l) because Z depends only on x.

**Why:** The reward reparameterization gives r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x). When you compute the preference difference r(y_w) − r(y_l), the β·log Z(x) term is identical for both completions (Z depends only on x, the prompt) and cancels exactly. This is a key structural property that makes DPO tractable.

**Why others are wrong:**
- A: Z(x) is not approximated; it exactly cancels.
- C: β is a separate hyperparameter; Z is not absorbed into it.
- D: Z is not replaced — it cancels algebraically.

---

## Q2. Answer: D

**Answer:** D — The model is collapsing toward the reference model and not learning the preference signal.

**Why:** In healthy DPO training, `rewards/chosen` (log π_θ(y_w) − log π_ref(y_w)) should increase and `rewards/rejected` (log π_θ(y_l) − log π_ref(y_l)) should decrease (become more negative). If both increase together, the training model is simply increasing the probability of all outputs relative to the reference, which means it is drifting away from the reference uniformly — not actually learning to prefer chosen over rejected. Check that the reference model weights are truly frozen.

**Why others are wrong:**
- A: Both increasing together is a training failure, not success.
- B: If the reference model were being updated, the log-ratio would stay near zero, not increase for both.
- C: High β would keep both log-ratios near zero; it would not cause both to increase.

---

## Q3. Answer: B

**Answer:** B — The model must stay close to π_ref; large updates would defeat the KL constraint.

**Why:** DPO's derivation assumes that π_θ converges to π*, which is defined relative to the fixed π_ref. If large gradient steps move π_θ very far from π_ref quickly, the implicit KL constraint in the loss is violated (the loss no longer accurately represents the original RL objective). Additionally, DPO operates on the difference in log-probabilities, which is sensitive to small changes in the numerics.

---

## Q4. Answer: C

**Answer:** C — The rejected responses often contain refusals; DPO reduces their probability along with other rejected patterns.

**Why:** DPO directly decreases the log probability of rejected completions. If the preference dataset's "rejected" examples include many refusals (e.g., "I cannot answer that" as a rejected response when a good answer exists), the model will learn to suppress refusal patterns broadly. This can make the model more compliant but can also cause it to generate refusals in unrelated contexts. This is a known failure mode documented in DPO follow-up work.

Fix: audit the rejection labels in your preference dataset. If refusals appear as "rejected" in benign contexts, consider filtering them out or using a different dataset split.

---

## Q5. Answer: B

**Answer:** B — Teaching a model to generate code that passes automated unit tests.

**Why:** DPO requires a fixed dataset of preference pairs. For code that passes unit tests, you can evaluate correctness programmatically at training time without needing labeled pairs. GRPO (Week 46) handles this much better: it generates multiple candidates, executes them, and uses the execution result as the reward — no preference labeling required. DPO's offline data assumption is a poor fit when verifiable rewards are available, because fresh execution feedback is more informative than stale labeled pairs.

---

## Q6. DPO Loss — Key Components

DPO loss:
```
L_DPO = -log σ(β · (log_ratio_w - log_ratio_l))
where log_ratio_y = log π_θ(y|x) - log π_ref(y|x)
```

`logps_chosen`: the sum of log-probabilities of tokens in y_w under the training model π_θ. Computed by a forward pass of the training model on the chosen completion.

`logps_rejected`: the sum of log-probabilities of tokens in y_l under the training model π_θ. Computed by a forward pass of the training model on the rejected completion.

The reference model (frozen) computes `ref_logps_chosen` and `ref_logps_rejected` in the same way. The log-ratio is `logps_chosen - ref_logps_chosen` for the chosen side, and similarly for rejected.

TRL computes all four quantities and then computes: `reward_margin = β · (log_ratio_w - log_ratio_l)`, then applies `-log σ(reward_margin)` as the loss.

---

## Q7. Bradley-Terry Assumptions and SQL Violations

**Assumption 1: Transitivity of preferences.** If y_w is preferred over y_l and y_l is preferred over y_m, then y_w is preferred over y_m. SQL violation: Preference might depend on query context. A query that "executes correctly on the test DB" might be preferred in isolation but lead to incorrect results on the production schema, making preferences non-transitive across contexts.

**Assumption 2: Preferences are determined solely by a scalar reward (no context-dependence between pairs).** SQL violation: Two SQL queries might both execute correctly but return different row counts — neither is definitively preferred without knowing the ground-truth expected output. If the labeling was done without a reference output, the "preferred" label is ambiguous, violating the assumption of a clear underlying scalar quality.

---

## Q8. DPO vs PPO Comparison

| Dimension | DPO | PPO |
|---|---|---|
| Data requirements | Fixed offline dataset of preference pairs | Online rollouts generated during training |
| Compute per step | 2 forward passes (train + ref) | 4 forward passes (actor + critic + RM + ref) + sampling |
| Verifiable rewards | Poor — must pre-label preferences | Good — reward computed fresh from verifier at each step |

For verifiable rewards (SQL execution), PPO or GRPO wins because you can compute exact rewards at training time without pre-labeling. For style/tone alignment with human preferences, DPO wins on compute and simplicity.

---

## Q9. DPO Failing on Hard Queries — Diagnosis and Interventions

**Diagnosis:** The preference dataset likely has very few hard (3+ join) examples. DPO is an offline method — it can only learn from the distribution of the dataset. If hard queries represent 5% of the dataset, the model's improvement on them is negligible. Additionally, reward margin 0.3 suggests the chosen/rejected pairs for hard queries are very close in quality (the model cannot yet distinguish them), giving the loss little signal.

**Intervention 1 (dataset design):** Oversample hard queries in the preference dataset. Generate 5× more preference pairs for complex queries specifically. Alternatively, use a stronger teacher model (GPT-4o) to generate the "chosen" SQL for hard queries, ensuring the chosen is actually better — if both chosen and rejected fail to execute, the preference label is meaningless.

**Intervention 2 (DPO configuration):** Reduce β from its current value. A high β keeps the model close to the SFT reference, which already struggles on hard queries. A lower β (e.g., 0.05 instead of 0.1) allows larger divergence from π_ref, giving the model more freedom to shift toward correct complex SQL patterns. Alternatively, consider a curriculum: train DPO on easy examples first to push reward_margin up, then fine-tune on hard examples only in a second DPO pass.
