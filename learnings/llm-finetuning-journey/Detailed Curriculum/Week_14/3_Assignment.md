# Week 14 Assignment — Annotating the LLaMA Papers and Codebase

This is a reading and annotation week. There is no new model training. The deliverables are written notes and annotations that demonstrate you read the papers carefully.

## Setup Checklist

- [ ] Print (or open side-by-side) the three LLaMA papers
- [ ] Download `modeling_llama.py` from HuggingFace transformers (or open in browser)
- [ ] A notebook or `journal.md` file to write in
- [ ] Your Week 12 `model_v2.py` for comparison

---

## Task 1 — Read LLaMA 1 Paper and Answer These Questions

**Read:** [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

Answer each question in `journal.md` with at least 3 sentences each:

1. What training data composition did LLaMA 1 use? What is the rationale for including GitHub code data in a general language model?
2. The paper shows LLaMA 13B outperforms GPT-3 175B on most benchmarks. Explain the key insight (from Chinchilla) that makes this possible.
3. LLaMA uses RMSNorm before each sublayer (Pre-RMSNorm), not Post-LN as in the original Transformer. What does the paper say about this choice?
4. What is the exact formula for the SwiGLU intermediate dimension used in LLaMA? Why is it different from the standard SwiGLU (8/3 × d_model)?
5. Does LLaMA 1 use weight tying between embeddings and LM head? What does the paper say?

**Deliverable:** Answers in `journal.md`.

---

## Task 2 — Read LLaMA 2 and LLaMA 3 Papers

**Read:** [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — focus on Section 2 (Pretraining) and Section 5 (Safety)

**Read:** [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — read Sections 1 (Introduction), 3 (Pre-Training), 5 (Results)

Answer in `journal.md`:

1. What changed between LLaMA 1 and LLaMA 2 in terms of architecture? (Be specific: context length, GQA, training tokens.)
2. LLaMA 3 uses a 128k-token vocabulary compared to LLaMA 2's 32k. What is the effect on tokenization efficiency, and why does that matter for training cost?
3. LLaMA 3's RoPE theta is 500,000 vs. LLaMA 1/2's 10,000. What is the effect of a larger theta on the position encoding, and why does it help with long contexts?
4. What is `num_key_value_heads=8` in LLaMA-3 8B? Compute the exact KV cache memory savings compared to full MHA with 32 KV heads, for a 4096-token context in FP16.

**Deliverable:** Answers in `journal.md`.

---

## Task 3 — Annotate `modeling_llama.py`

**Instructions:**
- Download the current `modeling_llama.py` from [HuggingFace transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- Print it (it's ~1000 lines) or annotate digitally
- For every major class and function, write a 1–3 sentence annotation explaining:
  - What it does
  - How it maps to your Week 12 implementation
  - Any differences you notice

**Required annotations (at minimum):**
- `LlamaRMSNorm` — compare to your `RMSNorm` class
- `LlamaRotaryEmbedding` — compare to your `precompute_rope_freqs`
- `rotate_half` / `apply_rotary_pos_emb` — compare to your `apply_rotary_emb`
- `LlamaMLP` — compare to your `SwiGLUMLP`
- `repeat_kv` function — explain what it does and when it's called
- `LlamaAttention.forward` — walk through the Q/K/V projection, RoPE, KV cache, GQA repeat, attention
- `LlamaDecoderLayer.forward` — identify Pre-RMSNorm pattern
- `LlamaForCausalLM.forward` — identify where the LM head is applied and whether weights are tied

**Deliverable:** A scanned/photographed PDF of your annotated printout, OR a Markdown file `llama_annotations.md` with your line-by-line notes. Commit to `week-14-llama-reading`.

---

## Task 4 — Comparison Table

Create a table in `journal.md` with the following columns:
`Component | LLaMA 1 | LLaMA 2 7B | LLaMA 3 8B | Your Week 12 nanoGPT`

Rows: Architecture, Context length, Vocab size, Tokenizer, Position encoding, RoPE theta, Normalization, FFN type, Attention type, n_kv_heads, Training tokens, Weight tying.

Fill in every cell. Leave blanks only for things not specified in the papers.

**Deliverable:** Table in `journal.md`.

---

## Acceptance Criteria

You can answer from memory:
- What is `num_key_value_heads` and what does it control?
- Why does LLaMA 3 use `rope_theta=500000`?
- What is `repeat_kv` and when is it called?
- How does LLaMA's weight tying differ from GPT-2's?
