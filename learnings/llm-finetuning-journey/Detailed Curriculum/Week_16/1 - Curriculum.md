# Week 16 — Phase 2 Gate

## Learning Objectives

By the end of this week, you will be able to:

- Self-certify that you have mastered all Phase 2 concepts with concrete evidence (code, W&B runs, reproduced results)
- Implement a complete decoder-only transformer — RMSNorm, RoPE, SwiGLU, GQA, KV cache, top-p sampling — from memory in under 3 hours
- Explain every component of the Phase 2 architecture stack (Bahdanau attention, scaled dot-product attention, MHA, Pre-LN, causal masking) without reference materials
- Identify your specific weak areas and plan targeted remediation before starting Phase 3
- Demonstrate that your GPT-2 124M reproduction meets the val loss threshold (≤ 3.27)

---

## Concepts

Week 16 is not a learning week — it is a gate. No new concepts are introduced. Instead, this section recaps what you must be able to explain and implement from memory. If any of the following descriptions feel unfamiliar, that is your signal to return to the corresponding week before attempting the gate project.

### What the Gate Tests

| Topic (source week) | What you must be able to do |
|---|---|
| Bahdanau attention (W9) | Draw seq2seq+attention; implement alignment score from scratch |
| Scaled dot-product attention (W10) | Derive `1/sqrt(d_k)` scaling; implement with causal mask |
| Multi-head attention (W10) | Implement Q/K/V projections, head split, attention, concat, output projection — under 15 min |
| Modern arch components (W12) | RMSNorm, SwiGLU, RoPE (cos/sin on paired Q/K dims), GQA with `repeat_kv` |
| KV cache (W13) | Implement cache; explain why no causal mask is needed once active |
| Sampling (W13) | Temperature, top-k, top-p — composable, from scratch |
| GPT-2 124M reproduction (W15) | `train_gpt2.py` reaches val loss ≤ 3.27 with gradient accumulation, bf16, cosine schedule |
| LLaMA reading (W14) | Explain LLaMA 1→ 2 → 3 differences (data scale, GQA, RoPE theta, tokenizer); navigate `modeling_llama.py` in <30 min |

If any row feels unfamiliar, that is your signal to return to the corresponding week before attempting the gate project.

---

## Gate Checklist — What You Must Be Able to Do

### From Memory (No Reference Allowed)

- [ ] Draw the Bahdanau seq2seq+attention architecture on a whiteboard. Label every component, every arrow, every tensor shape.
- [ ] Write the scaled dot-product attention formula on paper. Derive why `1/sqrt(d_k)` is needed.
- [ ] Implement `MultiHeadAttention` in PyTorch, from scratch, in < 15 minutes. No imports except `torch` and `torch.nn`.
- [ ] Implement a decoder-only transformer `Block` (Pre-LN, causal self-attention, SwiGLU FFN). No imports except `torch` and `torch.nn`.
- [ ] Implement `RMSNorm`, `SwiGLU`, and `apply_rotary_emb` from memory.
- [ ] Explain GQA and write the `repeat_kv` logic from memory.
- [ ] Implement `sample_next_token` with temperature, top-k, and top-p from memory.

### In Code (Reference Allowed — Your Own Code Only)

- [ ] Your GPT-2 124M reproduction (`train_gpt2.py`) is committed and produces val loss ≤ 3.27.
- [ ] Your W&B run `week-15-gpt2-repro` shows a smooth loss curve with cosine decay.
- [ ] Your `model_v2.py` (Week 12) contains working RMSNorm, SwiGLU, RoPE, and GQA.
- [ ] Your `seq2seq.py` (Week 9) produces an anti-diagonal attention heatmap on the string reversal task.

### Explanations (Oral or Written)

- [ ] Explain RoPE on a whiteboard using the rotation matrix concept. Show why `q_m · k_n` depends only on `m-n`.
- [ ] Read `modeling_llama.py` and explain every block in < 30 minutes (with the file open).
- [ ] Explain the difference between LLaMA 1, 2, and 3 in 5 minutes without notes.
- [ ] Explain why beam search is rarely used in modern LLM inference.
- [ ] Explain gradient accumulation, mixed precision, and gradient clipping without notes.

---

## Gate Project — The Capstone Implementation

**Project:** Without looking at any reference (no notes, no GitHub, no browser), implement a decoder-only transformer with the following specs:

- Architecture: RMSNorm, RoPE, SwiGLU, GQA, KV cache, top-p sampling
- Config: `n_layer=4, n_head=8, n_kv_heads=2, n_embd=256, block_size=256, rope_base=10000`
- Train on the Spider SQL training set queries (character-level)
- Train for 3000 steps, batch_size=32, LR=3e-4 with cosine decay
- Generate plausible SQL from the prompt `"SELECT"`

**You have 3 hours. No internet. No reference.**

If you cannot do this in 3 hours, you are not ready for Phase 3. Go back.

**Acceptance criteria:**
- Model trains without NaN or divergence
- Generated SQL contains recognizable keywords (SELECT, FROM, WHERE, JOIN) in syntactically plausible positions
- Val loss < 1.5 on SQL corpus
- Every component (RMSNorm, RoPE, SwiGLU, GQA, KV cache) is present and correct

**Commit:** `week-16-phase2-gate` with `gate_model.py`, `gate_train.py`, `gate_samples.txt`.

---

## How to Self-Assess

After completing the gate project (or attempting it), score yourself:

| Area | Pass Criteria |
|---|---|
| Bahdanau attention | Can explain alignment mechanism, implement additive score function |
| Scaled dot-product attention | Can derive formula, implement correctly, explain sqrt(d_k) |
| Multi-head attention | Can implement from memory with correct shapes |
| Decoder-only transformer | Can implement complete Block from memory |
| RMSNorm | Can implement and explain why no beta/mean subtraction |
| SwiGLU | Can implement with correct intermediate dim (8/3×d) |
| RoPE | Can implement, explain relative position property |
| GQA | Can implement repeat_kv, compute KV cache savings |
| KV cache | Can implement, explain why no causal mask needed |
| Sampling | Can implement temperature + top-k + top-p |
| GPT-2 repro | Val loss ≤ 3.27, W&B run committed |
| LLaMA reading | Can explain LLaMA 1/2/3 differences without notes |

If you have 3 or more areas where you cannot pass: **stop Phase 2. Re-do failing weeks.**

If you have 1–2 weak areas: re-do those specific weeks, then come back.

If you pass all areas: move to Phase 3.

---

## Connections

This week closes everything in Phase 2 (Weeks 9–15). Week 9 introduced encoder-decoder attention; Week 10 built the full Transformer; Weeks 11–13 built the decoder-only GPT family with modern components (nanoGPT, RMSNorm, RoPE, GQA, KV cache, sampling); Week 14 mapped production LLaMA code to those components; Week 15 reproduced GPT-2 124M. The gate is the integration point where all of those threads must be simultaneously available from memory.

Phase 3 (starting Week 17) assumes you can implement attention and a full decoder transformer from memory, read production transformer code fluently, and run training loops with mixed precision, gradient accumulation, and LR scheduling. Phase 3 immediately introduces pretraining at scale — scaling laws, data pipeline engineering, and 50M-parameter training runs on FineWeb-Edu. If you proceed without meeting the gate criteria, the Phase 3 and Phase 4 work will expose these gaps when you least want to discover them (debugging a 7B fine-tuning run at step 2000).

---

## Common Misconceptions / Pitfalls

- **Passing the gate by reading code instead of writing it.** If your gate project involves looking at your Week 12 implementation while coding, you have not passed. The test is whether you can reconstruct the architecture from understanding, not whether you can copy it from memory. Close all files.
- **Skipping the timed 3-hour implementation.** The time constraint is not arbitrary — Phase 3 and 4 require you to rapidly prototype model changes and debug code under pressure. If you cannot implement a small transformer in 3 hours with no reference, you will be slow and error-prone during the more complex work ahead.
- **Treating weak areas as acceptable.** "I understand GQA conceptually but can't implement `repeat_kv` quickly" is a gap. You will need to implement or debug GQA in production LLaMA code during Phase 4. Fix it now.
- **Confusing Pre-LN and Post-LN implementations.** nanoGPT uses Pre-LN; older references and some tutorial code use Post-LN. If your gate implementation diverges and training is unstable, check which LN placement you used.
- **Skipping the val loss verification on GPT-2 repro.** Val loss ≤ 3.27 is a concrete, testable criterion. Running the training and never checking the final validation loss means you do not know whether your implementation is actually correct at scale.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Work through Gate Checklist; for each unchecked item, review the corresponding TakeAway | 1 hr |
| Timed gate project: implement decoder-only transformer from memory (no reference) | 3 hrs |
| Score yourself using the self-assessment table; identify any failing areas | 0.5 hrs |
| Remediation for any failing areas (re-do the specific week's assignment) | 1–2 hrs |
| Write a brief gate reflection: what was hard, what was easy, what you would revisit | 0.5 hrs |
| Commit gate artifacts and reflection | 0.5 hrs |
