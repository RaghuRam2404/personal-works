# Week 24 Quiz Answers

## Q1 — Answer: B

**Answer:** B — Mixture of Experts with a subset of experts activated per token.

**Why:** In DeepSeek-V3, the standard FFN in each transformer block is replaced with 256 "expert" FFNs. A router network selects 2–4 experts for each token. Only selected experts compute the FFN output; the other 252 experts are idle. This gives the model 671B total parameters (much more storage for knowledge) while requiring only 37B parameters to be active (and thus in GPU memory bandwidth / compute) per forward pass.

**Why others are wrong:**
- A: sparse attention is a different technique; DeepSeek-V3 uses MLA for attention (not sparse)
- C: quantization reduces precision per weight, not the number of active parameters in this sense
- D: embedding layers in 671B MoE models are not the dominant parameter count

---

## Q2 — Answer: B

**Answer:** B — Better at code generation, potentially weaker on factual trivia.

**Why:** Qwen2.5-Coder's code-first training data means it has seen much more SQL, Python, Java, and code documentation than Llama 3-8B at a per-token level. However, Llama 3-8B's 15T tokens includes more diverse factual text (Wikipedia, books, news) — Llama 3 would likely score higher on MMLU subtopics like history or medicine. For your SQL use case, Qwen2.5-Coder's code focus is an advantage.

**Why others are wrong:**
- A: Qwen2.5-Coder has strong general language understanding; its base is Qwen2.5 which is trained on diverse text before code specialization
- C: they are both 7–8B scale models; parameter count alone does not determine performance
- D: fine-tuning on domain data is always possible and is the intended use

---

## Q3 — Answer: B

**Answer:** B — KV cache memory during long-context inference.

**Why:** In standard multi-head attention, every layer stores Key and Value matrices for every token in the context (the KV cache). For a 128K context window, this cache can be 10–20GB per model layer, making long-context inference impractical. MLA compresses the KV cache by projecting Keys and Values into a low-dimensional latent space before storage, reducing the cache size by 5–10× without significant quality loss.

**Why others are wrong:**
- A: batch size limitations are addressed by MoE compute efficiency, not attention design
- C: gradient vanishing is addressed by architecture design (residual connections, LayerNorm), not attention type
- D: tokenization is model-agnostic; MLA is purely about attention computation

---

## Q4 — Answer: B

**Answer:** B — Trains the model to infill a missing middle section given prefix and suffix.

**Why:** Fill-in-the-middle (FIM) training randomly selects a span within a document, places the prefix before and the suffix after a special separator, and trains the model to predict the missing middle. This teaches the model bidirectional context awareness despite being a causal (left-to-right) model. It dramatically improves code completion tasks where the IDE provides both the code before and after the cursor.

**Why others are wrong:**
- A: that is standard next-token prediction
- C: FIM is a training objective, not a data augmentation
- D: FIM is applied during pretraining in both DeepSeek-Coder and Qwen2.5-Coder

---

## Q5 — Answer: B

**Answer:** B — Larger embedding/LM head matrices; more information per token (fewer tokens per word).

**Why:** With 151,936 vocabulary entries, the embedding matrix is 151,936 × d_model — for d_model=4096 at 7B scale, this is ~2.5GB in FP16, significantly larger than Llama 3's 128K × d_model ≈ 2.1GB. More importantly, a larger vocabulary means the tokenizer creates fewer, longer tokens per word, so each token carries more semantic content and the model needs fewer forward passes for the same text length.

**Why others are wrong:**
- A: larger vocabulary actually requires more examples to see each token frequently enough; it does not speed convergence
- C: the vocabulary size does affect logit computation time, but this is a very minor overhead compared to the attention and FFN computation
- D: QLoRA works regardless of vocabulary size; you can freeze the embedding and LM head during LoRA fine-tuning

---

## Q6 — Short Answer

Grouped Query Attention (GQA) uses fewer Key and Value heads than Query heads. Where standard Multi-Head Attention (MHA) has H Query, K, and V heads, GQA groups the H Query heads into G groups, with each group sharing one K head and one V head. During inference, only G sets of K/V activations need to be cached (instead of H), reducing KV cache size by H/G×. Since the model still has H Query heads, the attention computation remains rich in query diversity; only the keys and values (which are used for retrieval, not projection) are reduced. Empirically, GQA with G=8 (as in Llama 3-8B, which has 32 Q heads and 8 KV heads) loses very little perplexity compared to full MHA.

---

## Q7 — Short Answer

1. **Inference cost dominates at scale.** Meta serves Llama models to billions of users. A smaller, more thoroughly trained 8B model costs less per inference request than a 70B Chinchilla-optimal model. By training 8B on 15T tokens instead of using a 70B model on 1.4T tokens (which might have similar benchmark performance), Meta reduces inference hardware costs by ~8×. The extra training compute is a one-time cost; inference savings recur for every request.

2. **Emergent capabilities and knowledge density.** More tokens means the model has encountered more facts, more reasoning chains, and more diverse writing styles. The model's representations become denser and more generalizable. At the 8B parameter scale, increasing training tokens from 160B (Chinchilla-optimal) to 15T provides substantial gains on benchmarks like MMLU and HellaSwag that would not have been predicted by the Chinchilla power law alone (Chinchilla was fitted at smaller scales).

---

## Q8 — Short Answer

For MySQL, I would still choose Qwen2.5-Coder-7B as the base — it has seen MySQL SQL from GitHub alongside PostgreSQL. For fine-tuning data, I would additionally include:
- MySQL-specific syntax that differs from PostgreSQL: `AUTO_INCREMENT` vs. `SERIAL`, `SHOW TABLES` vs. `\dt`, MySQL's `GROUP_CONCAT` vs. PostgreSQL's `STRING_AGG`, MySQL's stored procedures syntax, MySQL's `INSERT IGNORE` vs. PostgreSQL's `ON CONFLICT DO NOTHING`
- MySQL 8.0 window functions and CTE syntax (which differ subtly from PostgreSQL)
- MySQL-specific system tables (`information_schema` usage for MySQL vs. PostgreSQL catalog queries)

I would NOT need to teach TimescaleDB hypertables, `time_bucket()`, continuous aggregates, or PostgreSQL JSONB operators.

---

## Q9 — Scenario Model Answer

**1. Existing SQL knowledge in Qwen2.5-Coder-7B:**
- Standard ANSI SQL: SELECT, FROM, WHERE, JOIN (INNER, LEFT, RIGHT, FULL), GROUP BY, HAVING, ORDER BY, LIMIT
- Subqueries and CTEs (WITH clause)
- Window functions: ROW_NUMBER(), RANK(), LAG(), LEAD(), PARTITION BY
- Aggregate functions: COUNT, SUM, AVG, MAX, MIN, STDDEV
- Data types: VARCHAR, INTEGER, DECIMAL, DATE, TIMESTAMP
- Index usage hints and EXPLAIN concept (from Stack Overflow content)

**2. Knowledge gaps (PostgreSQL/TimescaleDB specific):**
- TimescaleDB-specific functions: `time_bucket()`, `time_bucket_gapfill()`, continuous aggregates syntax, hypertable creation
- PostgreSQL-specific: `ON CONFLICT DO UPDATE` (UPSERT), `RETURNING`, JSONB operators (`->`, `->>`), `LATERAL` joins, `generate_series()`, PostgreSQL extension functions (`pg_stat_statements`, `pg_partman`)
- Your specific database schema (table names, column names, business logic)
- These are absent from training because TimescaleDB documentation and internal schemas are rarely in GitHub code files at scale

**3. Fine-tuning budget:**
$150 / $1.50/hr = 100 hours of A100 time. For a 7B model fine-tuning with QLoRA (r=64):
- Memory: ~12GB for 4-bit quantized 7B + LoRA adapters — fits on A100 40GB
- Training speed: ~5,000–10,000 tokens/sec for 7B with QLoRA
- With 5,000 training examples at 512 avg tokens: 2.5M tokens per epoch
- 100 hours × 5,000 tok/sec × 3600 = 1.8B tokens processable — allows ~720 epochs on 5K examples (massively over-sufficient; 3–5 epochs is enough)
- Use QLoRA (r=64, alpha=128) for this budget — full SFT of 7B requires ~140GB GPU memory (not feasible without FSDP on 2+ A100s), LoRA with bf16 is possible but tight on one A100

**4. First 3 dataset examples:**
```
Q: "List all orders placed in the last 7 days with their total value"
A: SELECT order_id, created_at, SUM(quantity * unit_price) AS total_value
   FROM orders
   WHERE created_at >= NOW() - INTERVAL '7 days'
   GROUP BY order_id, created_at
   ORDER BY created_at DESC;

Q: "Show hourly count of events per user using TimescaleDB"
A: SELECT time_bucket('1 hour', created_at) AS hour,
          user_id,
          COUNT(*) AS event_count
   FROM events
   GROUP BY hour, user_id
   ORDER BY hour, user_id;

Q: "Find users who have not placed any orders in the last 30 days"
A: SELECT u.user_id, u.email
   FROM users u
   WHERE NOT EXISTS (
     SELECT 1 FROM orders o
     WHERE o.user_id = u.user_id
       AND o.created_at >= NOW() - INTERVAL '30 days'
   );
```
