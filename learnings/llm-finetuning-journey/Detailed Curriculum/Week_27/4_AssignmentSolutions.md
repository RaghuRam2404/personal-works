# Week 27 Assignment Solutions

## Task 1 — What Correct Evidence Looks Like

**Criterion 1 — Parameter count output:**
```
56.7 M params
```
If you get a number outside 50–65M: your model config does not match the Week 20 specification. Recheck `n_layers=8, d_model=768, n_heads=12, vocab_size=32000`.

**Criterion 2 — Perplexity formula (from memory):**

Correct 3-sentence answer:
"Perplexity is exp(mean cross-entropy loss), where the mean is taken over all evaluation tokens. We average the log-likelihoods (CE losses) before exponentiating because perplexity is defined as the geometric mean of inverse probabilities — averaging in log space is the correct way to compute a geometric mean. If we averaged the per-token perplexities directly (mean of exp(per-token loss)), we would get an arithmetic mean of perplexities, which is dominated by high-perplexity outliers and is not the standard definition."

If you cannot write this without notes: re-read Week 22 Curriculum until you can.

**Criterion 3 — Dataset output:**
```
Train: 4000, Val: 1000
SQL validity rate on val split: 97.8%
```
If `len(train_split) < 4000`: your v1 dataset is incomplete. Acceptable CONDITIONAL if ≥ 3,500 train examples.

**Criterion 4 — Qwen2.5-Coder-7B from memory (model answer):**

"Qwen2.5-Coder-7B has 7.6 billion parameters in a dense transformer architecture with grouped query attention (8 KV heads, 28 Q heads), SwiGLU FFN activation, and rotary positional embeddings (RoPE). It has a 151,936-token vocabulary using tiktoken-style BPE and supports up to 128K context length. It was trained on 5.5 trillion tokens with a code-first data distribution heavily weighted toward Python, Java, SQL, C++, and related languages. It is better than Llama 3-8B for SQL fine-tuning because: (1) it already understands SQL syntax deeply from pretraining, so fine-tuning teaches PostgreSQL idioms rather than SQL from scratch; (2) its code-focused training data gives it schema understanding (CREATE TABLE → query relationship) that general web text does not; and (3) it has been explicitly designed for code completion tasks including SQL, with evaluation benchmarks showing strong performance on Spider and HumanEval."

---

## Remediation Guidance

**If Criterion 1 FAIL (no trained model):**
- Train for 500M tokens (5,000 steps at 65K tokens/step) on Colab Free T4 or Pro GPU
- This takes 2–3 hours and produces a model with val loss ~5.5–6.5 (worse than target, but meets the "trained from scratch" criterion)
- A higher val loss is acceptable; the pipeline experience is the goal

**If Criterion 3 FAIL (dataset < 3,500 examples):**
- Week 1 of Phase 4: complete self-instruct generation to reach 4,000+ examples
- In the meantime, proceed with Phase 4 Week 28 (reading week) while generating data in parallel
- Do not start Week 29 (SFT) without completing the dataset

**If Criterion 4 FAIL (cannot explain architecture without notes):**
- Spend 2 hours studying the Qwen2.5-Coder-7B HuggingFace model card
- Write out the architecture description 3 times from memory
- This is not optional — you will be using this model for the next 50 weeks

---

## Common Phase 3 Gate Patterns

**Strong student:** All 4 criteria PASS, perplexity in 20–30 range, dataset > 4,500 examples. Ready for Phase 4 immediately.

**Typical student:** Criteria 1 and 2 PASS (trained model, working eval), Criterion 3 CONDITIONAL (3,500 examples), Criterion 4 PASS (read the papers). Proceed with CONDITIONAL PASS.

**Struggling student:** Training run did not complete (Colab issues), dataset at 2,000 examples, weak on paper reading. Spend 2 more weeks: 1 on retraining (shorter run), 1 on completing the dataset. Do not rush into Phase 4.
