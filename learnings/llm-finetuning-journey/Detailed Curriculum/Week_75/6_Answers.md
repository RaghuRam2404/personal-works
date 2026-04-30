# Week 75 Answers — Iteration: Different Base Models

---

## Q1. Answer: C

**Why C is correct:** Gemma 2's chat template uses specific control tokens (`<start_of_turn>`, `<end_of_turn>`) that differ from most other models. If you copy a generic formatting function and it applies the wrong template, the model receives malformed instruction boundaries. Crucially, if the `response_template` passed to `DataCollatorForCompletionOnlyLM` does not match Gemma's actual boundary tokens, loss is computed over the prompt tokens as well as the response. This dramatically increases loss magnitude and causes apparent divergence — the model is trying to memorize the prompt, not just the completion.

**Why A is wrong:** Tokenizer vocabulary differences affect token count per sample, but the effect on effective batch size is modest (10–20% typically) and would not cause divergence — it would shift loss values slightly.

**Why B is wrong:** GQA vs. full MHA does affect memory and speed but does not inherently require a different learning rate. Many GQA models train stably at standard LoRA LRs (1e-4 to 3e-4).

**Why D is wrong:** Embedding norm differences are real between architectures, but LoRA initialization is near-zero by default (the B matrix is zero at init), so the interaction with embedding norm does not cause divergence at step 300.

---

## Q2. Answer: B

**Why B is correct:** DeepSeek-R1-Distill models were trained with long chain-of-thought supervision. Their internal representations encode richer compositional structure than models trained purely on next-token prediction over code. Even when you fine-tune for direct SQL generation (no CoT in target), those intermediate representations carry over and help on complex queries. However, if you apply too many SFT steps on direct-format data, you can overwrite the reasoning-aligned weight structure — a form of catastrophic forgetting specific to instruction-format mismatch.

**Why A is wrong:** Reasoning ability transfers across tasks because it reflects better compositional feature learning, not task-specific reasoning traces.

**Why C is wrong:** Behavior is not permanently "baked in" — SFT can and does shift output format, especially with sufficient steps. The model will learn to produce direct SQL if trained that way.

**Why D is wrong:** Distillation refers to the training signal (teacher outputs), not token count. DeepSeek-R1-Distill-Qwen-7B has seen substantial pretraining data. Its lower baseline performance (if any) on code is domain-specific, not a token-count artifact.

---

## Q3. Answer: D

**Why D is correct:** Tokenizer vocabulary size is an architectural property of each model that you cannot and should not control — it is fixed by the base model's pretraining. Controlling it would mean changing the architecture, which defeats the purpose of a base model comparison. The goal is to hold training procedure constant while varying only the base model.

**Why A, B, C are wrong:** These are all legitimate control variables. Number of gradient steps ensures equal training compute. LoRA rank, alpha, and target modules ensure equivalent parameter efficiency (note: module names must be remapped per architecture, e.g., `q_proj` exists in Llama but `query_key_value` in some models). The evaluation benchmark must be held constant — evaluating each model on a different benchmark would make the comparison meaningless.

---

## Q4. Answer: B

**Why B is correct:** A 2.2 pp improvement at the SFT-only stage is promising but not sufficient to commit to switching the entire pipeline. The full training recipe includes CPT (100M tokens), DPO, and GRPO — stages that interact with the base model's architecture and may amplify or diminish the SFT-stage advantage. DeepSeek's reasoning-distillation origin may interact differently with DPO preference optimization. You must verify end-to-end before switching.

**Why A is wrong:** Running double steps on one model only while keeping others at standard steps breaks experimental control and does not answer the right question.

**Why C is wrong:** Spider 1.0 measures general SQL ability, not TimescaleDB-specific ability. Since your deployment target is TimescaleDB, Custom-200 is the more relevant benchmark. The concern about benchmark bias is not applicable here.

**Why D is wrong:** Quantization degradation is relevant for deployment decisions, not for deciding whether to switch the training pipeline. It is a separate concern evaluated after the base model decision is made.

---

## Q5. Answer: B

**Why B is correct:** `apply_chat_template` is the canonical method that encodes the model's expected instruction format — including special tokens, role markers, and separator tokens. The critical verification steps are: (1) confirm that SQL keywords (`SELECT`, `GROUP BY`, etc.) tokenize cleanly and are not fragmented into unusual sub-tokens, and (2) confirm that the response boundary (used by the data collator to mask loss on the prompt) correctly identifies where completions begin. Both checks are architecture-specific and must be re-done for every new base model.

**Why A is wrong:** Checking for `<unk>` is useful but insufficient. The critical issue is boundary alignment, not just vocabulary coverage.

**Why C is wrong:** Llama does not require fixed-length inputs. Dynamic padding is standard practice and preferred for efficiency.

**Why D is wrong:** Adding domain tokens requires embedding matrix expansion and reinitialization, which is expensive and unnecessary. Existing tokens handle SQL keywords adequately.

---

## Q6 — Short Answer

Code-specialized models (Qwen2.5-Coder, DeepSeek-Coder) have been pretrained or fine-tuned on large code corpora with emphasis on syntax correctness, identifier handling, and programming language structure. Reasoning-distilled models (DeepSeek-R1-Distill series) have been trained to produce intermediate reasoning steps via chain-of-thought distillation from a teacher model, resulting in richer compositional representations.

For NL→SQL, you would prefer a code-specialized model when your dataset is small to medium (under 10K examples), because domain-specific pretraining reduces the sample complexity needed to reach good SQL syntax accuracy. You would also prefer code-specialized models for TimescaleDB-specific functions that are rare or absent from reasoning corpora. You would prefer a reasoning-distilled model when your benchmark queries are structurally complex (nested subqueries, multi-step CTEs, complex aggregations), your dataset is sufficiently large to exploit the richer representations, and you can tolerate some risk of overwriting reasoning ability during SFT.

---

## Q7 — Short Answer

First, Gemma 2's sliding-window attention (SWA) restricts each token's attention to a local window. For TimescaleDB queries with long schema contexts (many table definitions, column lists), tokens at the beginning of the prompt may fall outside the attention window of tokens generating the SQL. This could cause the model to drop schema information, leading to hallucinated column names. You would need to evaluate whether schema context exceeds the effective window size and possibly restructure prompts to front-load the most relevant tables. Second, Llama 3.1 8B's RoPE with 128K context support means it handles long-context inputs natively, making it architecturally better suited for verbose schema prompts. In your experiment design, you should measure whether evaluation accuracy degrades as schema length increases for Gemma 2 but not for Llama — a length-stratified evaluation, not just aggregate EM. This would reveal an architectural limitation rather than a training data limitation.

---

## Q8 — Short Answer

The high standard deviation for DeepSeek (±2.8%) relative to Qwen2.5-Coder (±0.6%) tells you that DeepSeek's performance is more sensitive to the random seed, meaning its training dynamics are less stable at this learning rate and step count with your current hyperparameters. While the mean advantage (2.2 pp) clears your 2 pp threshold, the wide confidence interval means the true improvement could be as small as 0.6 pp or as large as 3.8 pp — the lower bound is marginal. This affects your switching decision by requiring you to either: (a) increase the number of evaluation seeds to narrow the confidence interval, (b) tune the learning rate for DeepSeek to reduce variance before committing, or (c) re-run the comparison with a longer SFT schedule where DeepSeek's advantage (if real) should compound. You should not switch on a single noisy result when the variance exceeds the effect size.

---

## Q9 — Short Answer

The argument for per-model hyperparameter search: different architectures have different loss landscapes, and a fixed LR that is optimal for Qwen may be too high for Gemma (causing divergence) or too low for Llama (causing underfitting). Comparing models at their respective optima is a fairer test of ceiling performance. The argument for fixed hyperparameters: searching hyperparameters per model conflates architecture quality with hyperparameter tuning effort and introduces experimenter degrees of freedom (multiple comparisons, search budget differences). A fixed-hyperparameter comparison isolates the base model variable cleanly.

For this week's experiment, the fixed hyperparameter approach is preferred, because the goal is to identify the best starting point for your existing pipeline — not to find the theoretical maximum for each architecture. You want the result to be reproducible and to answer a specific question: "Given my current training recipe, which base model wins?" A secondary tuning sweep on the winning model is the right next step, not a full grid search on all four simultaneously.

---

## Q10 — Deep Scenario

**Model answer:**

DeepSeek's advantage on complex JOIN + GROUP BY + HAVING queries aligns with its reasoning-distilled training: chain-of-thought supervision teaches the model to decompose multi-step problems, and this compositional structure transfers to SQL queries that require combining multiple clauses in a constrained order. Qwen2.5-Coder's advantage on TimescaleDB-specific functions (time_bucket, first, last) likely reflects pretraining corpus composition — Qwen was trained on a broader set of code repositories including database-adjacent libraries, whereas DeepSeek's distillation corpus may have under-indexed on niche time-series SQL dialects.

To test this hypothesis, you would run a corpus analysis: check whether the Qwen2.5-Coder pretraining data (which Alibaba has described at a high level) contains TimescaleDB or similar time-series SQL, and measure token-level generation confidence on time_bucket calls for both models. A targeted ablation would be: fine-tune DeepSeek on 500 TimescaleDB-specific examples only and re-evaluate — if the gap closes, it is a data coverage problem, not an architectural one.

If shipping today: choose Qwen2.5-Coder-7B, because TimescaleDB-specific accuracy is your production requirement and Qwen's advantage there is the most business-critical dimension. The 4% EM gain on complex JOINs from DeepSeek is valuable but secondary. If you have two more weeks: fine-tune DeepSeek on a TimescaleDB-augmented dataset, then re-evaluate. If DeepSeek closes the gap on time-series while retaining its JOIN advantage, switch the full pipeline to DeepSeek as the base model.
