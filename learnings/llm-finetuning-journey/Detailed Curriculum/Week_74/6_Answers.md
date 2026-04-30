# Week 74 Answers

## Q1

**Answer: B**

**Why correct:** RoPE encodes position by rotating query and key vectors at angle θ_i × m where θ_i is the per-dimension frequency and m is the position index. During training at 4096 tokens, the model only sees m ∈ {0, ..., 4095}. At m=5000, the rotation angle θ × 5000 is larger than any angle seen during training. The attention mechanism, which depends on the dot product Q_m · K_n = f(m-n, {θ}), encounters out-of-distribution values for large m and n. The attention degrades because the model has not learned to interpret these rotation magnitudes correctly.

**Why others are wrong:**
- A: Memory limitation is a real concern but not the primary failure mode of the position encoding itself.
- C: RoPE does not wrap around — it uses continuously increasing rotation angles.
- D: Flash Attention supports any sequence length given sufficient memory; it does not have a hard context limit.

---

## Q2

**Answer: C**

**Why correct:** RoPE uses a range of frequency components. High-frequency components (fast-rotating) create rapidly changing angles between consecutive positions — this is what allows the model to distinguish token at position 5 from token at position 6. If you interpolate (compress) these high-frequency dimensions, nearby positions get similar rotation angles, and the model loses the ability to distinguish short-range relationships. This degrades the quality of attention between nearby tokens — critical for SQL, where token-adjacent relationships (parentheses matching, column lists) matter greatly. YaRN preserves these high-frequency components and only interpolates the low-frequency dimensions that handle long-range position encoding.

**Why others are wrong:**
- A: High-frequency dimensions participate in attention computation just like all other dimensions.
- B: High-frequency dimensions encode short-range (not long-range) relationships.
- D: Frequency dimensions are not layer-specific; they are properties of the position encoding applied to all layers.

---

## Q3

**Answer: B**

**Why correct:** LongRoPE's evolutionary search evaluates N_population × N_generations candidate rescaling vectors. Each evaluation requires a short fine-tuning run (even a few hundred steps) to measure the candidate's perplexity. With population size 64 and 100–200 generations, this amounts to 6,400–12,800 fine-tuning runs — each costing minutes to hours. The total compute is 50–200 GPU-hours, significantly more than YaRN's single 400-step fine-tuning run. This is the fundamental computational cost LongRoPE pays for its per-dimension optimization advantage.

**Why others are wrong:**
- A: LongRoPE's advantage is the search procedure, not more fine-tuning steps per run.
- C: LongRoPE is applied to the same model sizes as YaRN.
- D: The evolutionary search does not store history in GPU memory — it runs sequentially.

---

## Q4

**Answer: B**

**Why correct:** Schema compression works by filtering the full schema to only the tables and columns most relevant to the user's question. This filtering step can fail in two ways: (1) it misidentifies which tables are relevant (e.g., the question requires a join between a main table and a lookup table, but only the main table is kept), or (2) an edge case query requires a table that looks irrelevant based on the question keywords but is actually required for a constraint. When these failures occur, the SQL model receives an incomplete schema and generates SQL that references non-existent tables or omits required joins. This failure mode is particularly dangerous because it is silent — the model generates valid-looking SQL that is logically wrong.

**Why others are wrong:**
- A: Schema compression with a semantic similarity model is very fast (< 100ms).
- C: Schema compression can use a small embedding model (e.g., all-MiniLM-L6, 80M params) rather than a large SQL model.
- D: TimescaleDB hypertables appear as normal tables in the schema definition; compression works identically.

---

## Q5

**Model answer:** Position interpolation maps position m → m × (old_context / new_context) so that all positions fit within the trained range. For example, extending from 4096 to 8192 maps position 8192 → 4096, staying in-distribution. The problem position interpolation solves: out-of-distribution rotation angles. The problem it creates: high-frequency dimensions are also scaled down, so nearby positions (m=5 and m=6) now have similar rotation angles after scaling, degrading short-range attention resolution. NTK-aware scaling addresses this by applying different scaling factors to different frequency dimensions: high-frequency dimensions are not interpolated (preserving short-range distinction) while low-frequency dimensions are interpolated (extending long-range range). The NTK intuition is that the model's learning is governed by a kernel whose effective bandwidth must be preserved at different frequency scales.

---

## Q6

**Model answer:** Hypothesis 1: Without fine-tuning at the extended context length, the model's attention patterns have not adapted to the new range. The attention mechanisms were trained to produce specific patterns for positions 0–4096; at 5000–6000 tokens, the scaled rotation angles are novel but not catastrophically wrong, yet the attention learned for the original range does not correctly "route" information from the far-schema tokens to the final SQL generation position. The model can generate coherent text because the language modeling is mostly local, but SQL that requires referencing a table defined at token 5000 may fail because the attention to that position is sub-optimal.

Hypothesis 2: The 12 pp accuracy drop may be specifically due to schema tokens near the beginning of the prompt (positions 4096+) being de-emphasized by the model. Transformers with RoPE have a mild recency bias — they attend more strongly to recent tokens. At 6000-token inputs, the schema tokens at positions 0–2000 are now very far from the SQL generation position, and the scaled-but-untuned RoPE may not provide sufficient long-range attention signal to those early schema positions.

---

## Q7

**Model answer:** Standard attention computes the full N×N attention matrix for a sequence of N tokens, requiring O(N²) memory to store the intermediate attention weights. At N=8192, this is 8192² ≈ 67M float16 values ≈ 134 MB per attention head, per layer. For 32 layers × 32 heads = 1024 attention heads, the total is 134 MB × 1024 ≈ 137 GB — completely infeasible on a single GPU. Flash Attention 2 uses a tiled computation that processes the attention matrix in blocks that fit in on-chip SRAM, never materializing the full N×N matrix in HBM (GPU RAM). Memory complexity drops from O(N²) to O(N), making 8192-token sequences feasible on a 40 GB A100.

---

## Q8 — Deep Scenario

**Model answer:** For a 7,200-token schema with 45 tables, I recommend a combined approach: schema compression first, context extension as a fallback.

Primary approach: schema compression. Use a small embedding model (all-MiniLM-L6-v2) to embed each table's description and columns, then retrieve the top-5 most relevant tables for each query. This reduces 45 tables to 5, dropping the schema from 7,200 tokens to approximately 800 tokens — well within the current 4096-token context. Implementation: 2 hours of development, no GPU cost, real-time latency < 50ms. Estimated accuracy on queries requiring only the top-5 tables: >90% of queries fall into this category.

Context extension fallback: for queries that require more than 5 tables (complex analytical queries with many joins), apply YaRN fine-tuning to extend to 8192 tokens. Cost: 0.5 GPU-hours. This covers queries requiring up to 12 tables.

Residual risk: queries that genuinely require more than 12 tables and cannot be decomposed into sub-queries. These are rare in practice (< 2% of SQL queries across the customer's reported workload) but represent the hard floor of this architecture. For those queries, add a human-review step or use a closed-source model with 128K context as a fallback.
