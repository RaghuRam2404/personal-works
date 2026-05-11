# Week 16 Assignment — Phase 2 Gate

## Overview

This assignment is the Phase 2 Gate. Complete it in order. If you fail a section, stop and go back to the relevant week before proceeding.

**Rules:**
- The gate project (Task 3) must be completed without any external reference. No browser, no GitHub, no notes. Only your memory and PyTorch documentation (you may look up function signatures — not implementations).
- Tasks 1 and 2 are self-assessments. Be honest. No one is grading you; failing to flag your gaps now means failing harder later.

---

## Task 1 — Written Self-Assessment (30 min)

Without notes or references, write answers to each of the following in `self_assessment.md`. Aim for 3–5 sentences each.

1. **Bahdanau attention:** What problem did it solve? Write the score function formula. What is the context vector?

2. **Scaled dot-product attention:** Write the formula. Why divide by `sqrt(d_k)`? What is the consequence of using `d_k` instead?

3. **Multi-head attention:** How many parameter matrices are there? What is the relationship between d_model, num_heads, and d_k? Why multiple heads?

4. **Causal mask:** What is it, how is it implemented in practice, and what happens during training if you forget it?

5. **RMSNorm:** Write the formula. What does it drop compared to LayerNorm? Why is that OK?

6. **SwiGLU:** Write the formula. How many projections? What is the correct intermediate dimension?

7. **RoPE:** Explain the key property that makes it a "relative" position encoding. What is theta for LLaMA 3?

8. **GQA:** What is `num_key_value_heads`? How does `repeat_kv` work? Compute KV cache savings for n_kv=8 vs. n_heads=32.

9. **KV cache:** What does it store? Why is no causal mask needed in single-token inference mode? Why does GQA directly reduce KV cache memory?

10. **Top-p sampling:** Write the algorithm in pseudocode. Why is it better than top-k for variable-confidence distributions?

11. **GPT-2 reproduction:** What is gradient accumulation and why do you divide loss by `grad_accum_steps`? What does `bfloat16` offer over `float16`?

12. **LLaMA 1 vs. 3:** List 5 concrete differences (architecture, data, training, tokenizer).

**Scoring:** Go back and check your answers against Weeks 9–15 materials. Mark each question Pass/Fail. Record in `self_assessment.md`.

**Deliverable:** `self_assessment.md` with answers and self-scored Pass/Fail.

---

## Task 2 — Whiteboard Checks (Optional but Strongly Recommended)

Perform these with a timer and no notes. If you have a colleague, ask them to time you. If alone, record yourself.

- [ ] Draw the full seq2seq+Bahdanau attention architecture. Label all tensors. Time limit: 5 min.
- [ ] Write `scaled_dot_product_attention` and `MultiHeadAttention` on paper. Time limit: 10 min.
- [ ] Write a decoder-only `Block` (Pre-LN, causal MHA, SwiGLU FFN) on paper. Time limit: 10 min.
- [ ] Explain RoPE to an imaginary colleague (rotation, relative position property, theta). Time limit: 5 min.

If you can't complete any of these in time: go back to the relevant week.

**Deliverable:** Check each box you completed successfully.

---

## Task 3 — Gate Project: From-Scratch Modern Transformer (3 Hours, No Reference)

**Implement a complete decoder-only modern transformer and train it on SQL.**

**Spec:**
- File: `gate_model.py` — model definition only
- File: `gate_train.py` — training loop
- Config: `n_layer=4, n_head=8, n_kv_heads=2, n_embd=256, block_size=256, rope_base=10000`
- Required components: RMSNorm, RoPE (LLaMA convention), SwiGLU (8/3x dim), GQA with repeat_kv, KV cache in generate()
- Optimizer: AdamW, lr=3e-4
- LR schedule: cosine decay from 3e-4 to 3e-5, warmup 200 steps
- Data: Spider SQL training queries (character-level), or your SQL corpus from Week 11
- Train: 3000 steps, batch_size=32
- Logging: print loss every 100 steps. Print val loss every 500 steps.
- Generation: after training, generate 10 SQL snippets starting from "SELECT" using top-p=0.9, temp=0.5
- Save outputs to `gate_samples.txt`

**Acceptance criteria:**
- No NaN during training
- Val loss < 1.5 at step 3000
- Generated SQL contains SELECT/FROM/WHERE/JOIN in syntactically plausible positions
- Every required component is present (RMSNorm, SwiGLU, RoPE, GQA, KV cache in generate)

**What counts as failure:**
- Using LayerNorm instead of RMSNorm
- Using learned absolute position embeddings instead of RoPE
- Missing GQA (all heads have KV)
- KV cache not implemented in generate()
- Loss diverges or is NaN

**After completing:** Review your code against `model_v2.py` from Week 12. Note every difference. Write 5 bullet points in `gate_notes.md` describing what you would do differently if you had another hour.

**Deliverable:** `gate_model.py`, `gate_train.py`, `gate_samples.txt`, `gate_notes.md` committed to `week-16-phase2-gate`.

---

## Task 4 — Go/No-Go Decision

After Tasks 1–3, make the decision:

Fill in `gate_decision.md`:

```markdown
# Phase 2 Gate Decision

**Date:** YYYY-MM-DD

**Self-assessment score:** X/12 questions passed

**Gate project result:** PASS / FAIL
- Val loss achieved: X.XX (target: < 1.5)
- Generated SQL looks syntactically valid: YES / NO
- All required components present: YES / NO
- NaN during training: YES / NO

**Decision:** PROCEED TO PHASE 3 / REPEAT WEEKS [N, M]

**Weak areas to revisit (even if proceeding):**
- [List any questions you got wrong or concepts that felt shaky]
```

Be honest. The curriculum requires honesty about your gaps.

**Deliverable:** `gate_decision.md` committed.
