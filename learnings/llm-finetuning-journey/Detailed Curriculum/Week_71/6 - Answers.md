# Week 71 Answers

## Q1

**Answer: B**

**Why correct:** RLHF trains a separate reward model on human preference labels (Bradley-Terry model), then uses this learned reward model to score policy outputs. The learned reward model can be wrong, over-optimized, and expensive to train. RLVR replaces the learned reward model with a verifiable objective: code that executes correctly, math that equals the ground truth, SQL that returns the right result set. The reward is deterministic and requires no human labels beyond the task definition. Your GRPO training uses RLVR: the executable-SQL reward is a verifiable function.

**Why others are wrong:**
- A: This reverses the correct definitions.
- C: Tulu 3 specifically applies RLVR to general instruction following, math, and code — not only math.
- D: RLVR still requires task definitions and some form of ground truth; it eliminates the learned reward model, not all preference signal.

---

## Q2

**Answer: C**

**Why correct:** SmolLM2's core claim is that data quality is the binding constraint at sub-2B parameter scale, not parameter count. The comparison models (Phi-3-mini-3.8B, Gemma-2B) were trained on either proprietary or lower-quality open data. SmolLM2's FineWeb-Edu filtering selects high-educational-quality web text using a classifier, producing a dataset that is more information-dense per token. This means SmolLM2 learns more per gradient step, allowing a 1.7B model to match a 3B model trained on noisier data.

**Why others are wrong:**
- A: Architecture depth/width differences are not the primary explanation cited.
- B: Flash attention affects speed, not learned capability.
- D: Context window affects what the model can process, not the fundamental learning efficiency.

---

## Q3

**Answer: B**

**Why correct:** OLMo 2's mid-training phase adds domain-specific, high-quality data (StackExchange, code) on top of the broad pretraining corpus — exactly analogous to your CPT stage, which adds PostgreSQL-specific text on top of Qwen2.5-Coder's existing code pretraining. Both strategies use the same mechanism: a second pretraining pass with domain-upweighted data to shift the model's knowledge distribution toward the target domain before fine-tuning.

**Why others are wrong:**
- A: GRPO uses reinforcement signals; OLMo 2's mid-training uses supervised prediction on curated text.
- C: DPO uses preference pairs; OLMo 2's mid-training does not.
- D: LLM-as-judge filtering is a quality gate, not a training stage.

---

## Q4

**Answer: B**

**Why correct:** Intermediate checkpoints are the only way to study capability emergence — the non-linear development of specific skills during training. With only the final model, you can measure what capabilities exist but not when they emerged or which training stage caused them. OLMo 2 found, for example, that certain reasoning capabilities emerge sharply at specific training fractions — a finding only possible by evaluating intermediate checkpoints. This enables attribution of capabilities to training phases, which is essential for understanding what pretraining data contributes vs what fine-tuning contributes.

**Why others are wrong:**
- A: Model size selection is a benefit of releasing multiple model sizes, not intermediate checkpoints of the same model.
- C: Intermediate checkpoints are typically the same or larger than final checkpoints.
- D: There is no legal requirement for multiple checkpoint versions.

---

## Q5

**Model answer:** Tulu 3's finding strongly validates your Week 59 choice. You generated 5K curated preference pairs using execution verification (chosen = correct SQL, rejected = wrong SQL), rather than generating 50K pairs with weaker quality criteria. Tulu 3's ablation shows that pairs with high signal (large chosen-rejected margin) outperform high-quantity low-signal pairs. Your 5K verified pairs (with ground-truth execution-based labeling) are high-signal by construction — every chosen example is provably correct and every rejected example is provably wrong. If anything, Tulu 3 suggests you could have gotten similar DPO quality with even fewer examples if the signal was consistent. The finding does not challenge your choices; it provides theoretical backing for why 5K carefully verified pairs gave you 2.3 pp improvement over SFT-only.

---

## Q6

**Model answer:** What is missing: your intermediate training checkpoints (CPT-only, SFT-only, DPO-only) and the raw training data (your 25K SFT examples and 5K DPO pairs). OLMo 2 releases all intermediate checkpoints and the full training corpus. The scientific benefit of releasing your intermediate checkpoints: other researchers could study when SQL-specific capabilities emerge in your pipeline (after CPT? after SFT? after GRPO?), enabling them to determine the minimum training required to achieve a given capability level. For example, if the time_bucket accuracy only appears after GRPO, that tells future researchers to prioritize GRPO over DPO for domain-specific syntax acquisition.

---

## Q7

**Model answer:** A SQL model's inference speed depends on two factors: GPU FLOPs (fixed for a given model size) and effective sequence length. Effective sequence length is the number of tokens in the prompt + output. If the tokenizer splits SQL keywords like `time_bucket` into 3 tokens (`time`, `_`, `bucket`) instead of 1, then every SQL query with `time_bucket` is proportionally longer in token space. A 150-token SQL query in "token-efficient SQL tokenization" might be 220 tokens in a poor tokenizer — increasing both prefill time (TTFT) and decode time proportionally. In a production system handling thousands of queries per day, this 47% token overhead compounds into significantly higher latency and GPU cost. A model with SQL-aware tokenization — like Qwen2.5-Coder which was trained heavily on code — tends to have better single-token coverage of SQL keywords than models trained primarily on natural language.

---

## Q8 — Deep Scenario

**Model answer:** Your colleague is describing model collapse (also called "self-play collapse" or "mode collapse in on-policy generation") — the phenomenon where a model trained on its own outputs progressively amplifies its own biases and errors, narrowing the output distribution and degrading quality. The theoretical mechanism: if the model makes systematic errors (e.g., always using INNER JOIN instead of LEFT JOIN for null-preserving joins), these errors appear in the generated training data, reinforcing the bias.

Two specific safeguards: First, execute-and-verify filtering — only add completions that pass database execution against a truth query. This is an objective filter that cannot amplify systematic errors that are wrong; the database is the arbiter. Second, diversity control — compute embedding similarity between new examples and existing training examples; reject any new example with cosine similarity > 0.95 to an existing training example to prevent mode collapse into a narrow distribution.

Tulu 3 demonstrates safe on-policy generation by using exactly the first safeguard: all RLVR training signals come from verified correctness, not from the model's own confidence. When the reward is external verification (database execution), the model cannot fool itself.
