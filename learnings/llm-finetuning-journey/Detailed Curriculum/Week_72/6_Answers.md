# Week 72 Answers

## Q1

**Answer: D**

**Why correct:** In MoE models, all expert FFN parameters exist in memory (that is what 16B total means). LoRA adapters are typically applied to all expert FFN layers — not just the routed active ones — because at training time, different examples route to different experts, and all experts need gradient updates across the training batch. The gradient for each expert is computed only from tokens that were routed to that expert. So: LoRA adapter matrices exist for all experts (D is correct that adapters are applied to all experts) but in any single forward pass, only the active experts compute gradients (D is correct that only activated experts receive gradient updates in that step).

**Why others are wrong:**
- A: 16B is the total parameter count; not all parameters are in the LoRA adapter target — only specified layers.
- B: 2.4B active parameters per forward pass does not mean only 2.4B parameters get adapters.
- C: Routing decisions vary per training example but the adapter matrices are fixed structures applied to all experts.

---

## Q2

**Answer: B**

**Why correct:** The fundamental limitation of training a 7B model with GRPO directly is the quality of the exploration: a 7B model generating K=8 samples on hard reasoning problems produces mostly wrong or incoherent answers, providing weak training signal. The 671B model, after GRPO, produces coherent multi-step reasoning traces even for very hard problems. Distilling these high-quality traces into the 7B model via SFT is more efficient than asking the 7B model to discover these reasoning patterns through its own RL exploration. This is the standard knowledge distillation argument applied to reasoning traces.

**Why others are wrong:**
- A: GRPO works at 7B scale — your own GRPO training proves this.
- C: GRPO at 7B is compute-intensive but not prohibitive (you ran it in Week 60).
- D: Context length is not the limiting factor for K-sample generation.

---

## Q3

**Answer: B**

**Why correct:** DPO is an optimization process that shifts the model's policy away from rejected distributions and toward preferred ones. The first RLHF established a preference for helpful, conversational responses; your domain DPO shifts this toward SQL-only, non-explanatory responses for SQL prompts. The two DPO stages are not in conflict — they operate on different prompt distributions (general questions vs SQL questions). The result is a model that is helpful and conversational on general questions (RLHF baseline) and SQL-precise on SQL questions (domain DPO). This is a practical advantage of prompt-conditional DPO.

**Why others are wrong:**
- A: DPO does not completely overwrite previous fine-tuning; it shifts the policy direction from the previous checkpoint.
- C: Two DPO runs do not typically cause destructive interference if the prompt distributions differ.
- D: Qwen's general RLHF did not include TimescaleDB patterns — domain DPO was necessary and effective.

---

## Q4

**Answer: B**

**Why correct:** The KV cache stores key and value vectors for every token in the context. At 4096-token context, standard attention requires ~0.8 GB of KV cache per layer for a 7B model. MLA's latent compression reduces this to ~0.1 GB by caching the compressed latent representation and reconstructing K, V on the fly. This has two practical benefits: you can run a 4096-token context in the same VRAM that previously required 2048 tokens, or you can batch more sequences simultaneously in the same VRAM. This is directly why DeepSeek-V3 can serve very long context without proportional VRAM cost increase.

**Why others are wrong:**
- A: Attention computation FLOPs are determined by sequence length, not KV storage format.
- C: The latent compression is not a denoising operation; it is a learned projection that trades slight accuracy for significant memory savings.
- D: Batching still has memory overhead; MLA reduces KV cache overhead, not all memory overhead.

---

## Q5

**Model answer:** Cold start in DeepSeek-R1 refers to a small-scale supervised fine-tuning step before GRPO, using a curated set of problems with explicit multi-step reasoning traces as demonstrations. It is necessary because GRPO exploration from a base model (without any reasoning demonstration) produces incoherent outputs — the model does not know that it should generate intermediate reasoning steps before the final answer. The base model's natural completion style does not include structured chain-of-thought. Cold start seeds the model with the behavioral template ("think step by step before answering") that GRPO can then optimize. Without cold start, GRPO reward signals are too sparse to bootstrap the reasoning behavior — the model almost never generates a correct long reasoning trace by chance.

---

## Q6

**Model answer:** Two reasons to prefer R1-Distill-Qwen-7B: (1) Same Qwen2.5 architecture as your existing pipeline — full Unsloth support, same tokenizer, same quantization workflow; you can run your existing training scripts with minimal changes. (2) The R1 distillation provides chain-of-thought reasoning capabilities that may help on complex multi-table SQL and subquery problems that require multi-step reasoning, which pure code-tuned models handle less consistently. Two reasons to prefer Llama 3.1 8B: (1) Llama 3.1 is a broader-base model with stronger commonsense reasoning and schema understanding from a more diverse pretraining set; (2) R1-Distill was fine-tuned from Qwen2.5-Math (not Coder), which may mean weaker initial SQL syntax coverage compared to Llama 3.1's code training.

---

## Q7

**Model answer:** HumanEval measures Python code generation — generating syntactically correct Python that passes unit tests. SQL generation shares several properties: both require generating formally correct structured output, handling complex logic (loops/joins, conditionals/WHERE), and respecting type constraints. Strong HumanEval performance indicates the model can generate correct, executable structured output, which is a necessary condition for SQL quality. However, HumanEval is an imperfect predictor because: SQL requires schema awareness (knowing which columns exist in a specific table), while HumanEval problems are self-contained. SQL also requires understanding database semantics (NULL handling, JOIN nullability, aggregate function ordering) that Python does not have direct analogues for. A model that excels at Python algorithms may still fail on TimescaleDB-specific SQL patterns.

---

## Q8 — Deep Scenario

**Model answer:** Two benefits of chain-of-thought for SQL: First, explicit reasoning about schema structure (identifying relevant tables, join keys, filter columns) before writing SQL reduces schema hallucination — the model commits to a schema interpretation before generating syntax. Second, step-by-step reasoning enables more accurate handling of complex aggregations (time_bucket + GROUP BY + HAVING) where the model must correctly order SQL clauses; the reasoning trace enforces logical ordering.

Two SQL-specific challenges: First, SQL reasoning traces are harder to verify than math — "I should use LEFT JOIN because null customers need to be preserved" is semantically correct reasoning, but verifying this reasoning automatically requires natural language understanding, not just execution. Second, SQL reasoning can produce correct SQL through incorrect reasoning (e.g., reasoning about the wrong table but accidentally generating the right query), making the reasoning-trace signal noisier than math chain-of-thought.

Training data approach without a 671B teacher: Use GPT-4o (via API) to generate SQL chain-of-thought traces for 500 of your training examples. Verify that the final SQL in each trace executes correctly. Add these verified traces to your SFT dataset as a "reasoning format" variant. This gives you 500 high-quality SQL reasoning examples at API cost rather than training a 671B model.
