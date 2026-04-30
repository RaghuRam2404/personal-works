# Week 51 Quiz Answers

## Q1. Answer: B

**Answer:** B — Stop iterating; pick the best checkpoint.

**Why:** Three consecutive experiments with <1pp improvement each is the textbook diminishing returns signal. You have reached the ceiling of what targeted GRPO iteration can achieve with the current dataset size. Continuing is wasting compute that should be saved for Phase 6's larger-scale training. A 6th experiment would likely produce another 0.5pp or less — not worth $6–10 on RunPod.

**Why others are wrong:**
- A: Switching metrics is fine conceptually, but the diminishing returns rule applies per-metric too. If semantic accuracy is also plateauing, the same logic applies.
- C: More steps does not overcome a data distribution ceiling. You need more diverse training data, not more optimization on the same data.
- D: Switching algorithms is a major change that should be hypothesis-driven. You have not identified GRPO as the problem — the problem is data coverage.

---

## Q2. Answer: B

**Answer:** B — v3-iter1 — semantic accuracy reflects actual correctness.

**Why:** The Phase 5 Gate requires v3 to beat v1 on execution accuracy, which both models do. Given that both pass the Gate criterion, semantic accuracy is the deciding factor for practical usefulness. A model that executes 85% of queries but only 58% return correct results is less useful than a model that executes 84% and 61% are correct. The 1pp execution accuracy advantage of v3-iter2 is within measurement noise on 200 queries (±2pp at 95% confidence). The 3pp semantic accuracy advantage of v3-iter1 is more meaningful.

Note: option D is a distractor — the Gate does not specify a minimum semantic accuracy threshold.

---

## Q3. Answer: B

**Answer:** B — Acknowledge the limitation and test on 50 additional held-out queries.

**Why:** Using the same eval set for both iteration and final evaluation creates an implicit selection bias — experiments that improved the 200-query eval set were kept, and those that hurt it were discarded. This can cause the final model to be better on those specific 200 queries than on a fresh distribution. The correct response is: acknowledge the limitation in the report (scientific honesty) and add a blind validation set (50 fresh queries not used during iteration) to verify the generalization claim.

**Why others are wrong:**
- A: Discarding all results is drastic and unnecessary — just add the supplementary blind test.
- C: A 5th experiment does not resolve the concern; it is the number of eval accesses, not experiments, that matters.
- D: Greedy decoding does not prevent eval set overfit through hyperparameter selection.

---

## Q4. KL Divergence and Phase 6 Implications

Higher KL (5.1 nats vs 3.2 nats) means v3-iter2 has drifted further from the SFT reference model than v3. For Phase 6 training, this is a minor concern: Phase 6 runs SFT → DPO → GRPO from scratch on the larger 50K dataset. The starting point for Phase 6's GRPO will be a fresh SFT model trained on the larger dataset — the v3-iter2 weights are not directly reused as the initialization. Therefore, the KL of v3-iter2 does not carry into Phase 6. However, if Phase 6 uses v3-iter2 as the starting checkpoint (to save compute), the higher KL means the model is further from the SFT prior, which may require increasing β in Phase 6's GRPO to re-anchor.

---

## Q5. Phase 6 Dataset Priority

The 60% single-table win rate suggests your domain SQL training has been effective for simple queries — your model is already competitive with GPT-4o on easy SQL. The 20% 3+ JOIN win rate is the critical gap. Phase 6 dataset construction should prioritize complex query examples with the following breakdown:
- 40%+ of all 50K examples should involve 2+ JOINs
- 20%+ should involve CTEs or window functions
- 10%+ should involve TimescaleDB-specific functions (time_bucket, hyperfunctions)

Simple (single-table) examples can be kept at 30% — you already handle them well, and a small maintenance dose prevents catastrophic forgetting during Phase 6 SFT.

---

## Q6. Phase 5 Retrospective

SFT in Phase 4 established the domain knowledge (schema adherence, SQL syntax for our PostgreSQL/TimescaleDB schema) but left a 22% syntax error rate and poor complex query handling. DPO in Week 45 was the most efficient improvement: using execution-labeled preference pairs, it dropped syntax errors to 7% and improved easy/medium query accuracy by 11pp in a few hours of training with no online generation. GRPO in Weeks 47–48 was harder to get right: the first run improved execution accuracy on complex queries by only 1pp because the training prompt set was too simple — the most important lesson of Phase 5 was that the data distribution of the GRPO training set must match the difficulty distribution of queries where you want improvement. What did not work as expected: DPO's semantic accuracy did not improve (it improved execution but not correctness), which we traced to labeling quality in Week 44 — chosen examples often executed but returned wrong rows. If starting Phase 5 over, we would build the Week 44 preference dataset with reference SQL for every prompt from the beginning, ensuring the labeling captures semantic correctness, not just execution success.
