# Week 8 — Assignment (Phase 1 Gate Capstone)

## Setup Checklist

- [ ] All prior week commits exist in your `llm-finetuning-journey` GitHub repo (Weeks 1–7).
- [ ] HuggingFace account has at least 1 uploaded artifact (Spider tokenized dataset from Week 7).
- [ ] Colab Pro or Free T4 available for training (if using Option A with nanoGPT-size model).
- [ ] W&B project `week-08-capstone` created.
- [ ] Choose your option: Option A (char-level transformer) or Option B (distilgpt2 fine-tuning).

---

## Task 1 — The Capstone Project

### Option A: Char-Level Transformer on Spider SQL

**Goal:** Train a nanoGPT-style character-level transformer on Spider SQL queries and generate SQL samples.

**Requirements:**
- Create a folder `week_08/option_a/` with:
  - `model.py` — nanoGPT architecture. You may read [karpathy/nanoGPT/model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for reference, but type every line yourself. Do NOT copy-paste.
  - `data.py` — your own data loader. Reads `week_04/sql_queries.txt`, builds char vocabulary, yields `(X, Y)` batches for next-char prediction.
  - `train.py` — your own training loop using: AdamW, warmup + cosine schedule (from your Week 5 implementation), gradient clipping (`max_norm=1.0`), optional AMP. Logs to W&B project `week-08-capstone`.
  - `generate.py` — generates SQL samples from a saved checkpoint.
  - `README.md` — described below.

- Training configuration (recommended):
  - `n_layer=4, n_head=4, n_embd=128` — a ~1M param model that trains in ~30 min on Colab T4.
  - `block_size=128` — 128 character context.
  - `batch_size=64`.
  - Train for at least 5000 steps.
  - Target: train loss < 1.2 nats.

- Generate 10 SQL samples after training (temperature=0.8) and save to `week_08/option_a/sql_samples.txt`.

**Deliverable:** All files in `week_08/option_a/`. Commit message: `week-08-capstone-option-a`.

### Option B: distilgpt2 Fine-Tuning on Spider SQL

**Goal:** Fine-tune distilgpt2 on Spider SQL using the HuggingFace stack and your own training loop.

**Requirements:**
- Create folder `week_08/option_b/` with:
  - `prepare_data.py` — loads Spider, formats as `"### Question: {q}\n### SQL: {sql}"`, tokenizes with distilgpt2 tokenizer, saves to disk.
  - `finetune.py` — fine-tuning loop. Requirements:
    - Use `AutoModelForCausalLM.from_pretrained("distilgpt2")`.
    - Use a raw PyTorch `DataLoader` (not `Trainer`).
    - Apply the modern training recipe: AdamW (lr=5e-5), warmup 50 steps, cosine decay, gradient clipping (max_norm=1.0).
    - `labels = input_ids.clone()`, mask padding with -100.
    - Train for 200 steps (fast — distilgpt2 fine-tuning is quick).
    - Log to W&B project `week-08-capstone`: train loss per step.
  - `generate.py` — generates SQL given a question prompt.
  - `README.md` — described below.

- Generate 5 SQL completions for each of these prompts:
  ```
  "### Question: Find all customers who placed more than 3 orders.\n### SQL:"
  "### Question: List the top 5 products by revenue.\n### SQL:"
  ```
- Save generated outputs to `week_08/option_b/sql_samples.txt`.

**Deliverable:** All files in `week_08/option_b/`. Commit message: `week-08-capstone-option-b`.

---

## Task 2 — README

**Goal:** Demonstrate technical communication.

**Requirements:**
The README must contain these sections (minimum 500 words total):

1. **Project summary** (2–3 sentences): what you built, what dataset, what model.
2. **Dataset**: where the data came from, how many examples, how you preprocessed it, token length statistics.
3. **Model**: architecture description (number of layers, heads, embedding dim, parameter count). For Option A: include the parameter count (`sum(p.numel() for p in model.parameters())`). For Option B: distilgpt2's 82M params.
4. **Training**: optimizer, LR schedule (with values), batch size, number of steps, compute (Mac/Colab).
5. **Results**: W&B run link. Final train loss. Plot screenshot or link. For Option A: sample SQL output (show 3 examples). For Option B: completion examples from the 2 prompts above.
6. **What worked**: 2–3 things that helped (e.g., "gradient clipping stabilized training significantly").
7. **What surprised you**: 2–3 genuine surprises from the training process.
8. **What I still don't understand**: honest list of 3–5 things in the codebase you used but don't fully grasp yet. For Option A: parts of the transformer architecture. For Option B: HuggingFace internals.
9. **Phase Gate self-assessment**: answer YES/NO to each gate criterion.

---

## Task 3 — The Timed Test (Non-Negotiable)

**Goal:** Prove you can write the training loop from memory.

**Instructions:**
1. Set a 10-minute timer.
2. Open a new Python file: `week_08/timed_loop.py`.
3. Close all tabs — browser, editor, notes. No references.
4. Write a complete PyTorch training loop for a simple classifier on random data. Must include:
   - Model definition (any architecture, at least 2 layers).
   - `AdamW` optimizer with weight decay.
   - A learning rate schedule (any type — cosine, step, linear).
   - `autocast()` context manager (even if on CPU, import and wrap).
   - `clip_grad_norm_()` with `max_norm=1.0`.
   - `zero_grad()` in the correct position.
   - `model.eval()` + `torch.no_grad()` block for validation.
   - Loss logging every 10 steps.
5. Time yourself. If you finish in under 10 minutes: note "PASS." If over 10 minutes: note "PARTIAL — re-do Week 1."
6. Run the file. It must not crash. If it crashes: fix it, but the crash counts against you.

**Deliverable:** `week_08/timed_loop.py` with a comment at the top: `# Completed in X minutes. PASS / PARTIAL`.

---

## Phase Gate Checklist (Self-Assessment)

Complete this in `week_08/phase_gate.md`:

```markdown
# Phase 1 Gate Self-Assessment

Date: [date]

## Gate Criteria

- [ ] Training loop from memory in <10 min: [PASS / FAIL]
- [ ] Loss curve diagnosis: [PASS / FAIL — describe how you tested this]
- [ ] Backprop by hand (2-layer network): [PASS / FAIL — attach your handwritten notes photo]
- [ ] GitHub commits for all weeks 1–7: [PASS / FAIL]
- [ ] HuggingFace artifact uploaded: [URL or FAIL]

## Knowledge Check (answer without notes)

1. LSTM gating equations: [your answer]
2. Conv output shape formula: [your answer]
3. AdamW vs. Adam for weight decay: [your answer]
4. Why BPE starts from bytes: [your answer]
5. The -100 label convention: [your answer]

## Decision

[ ] All 5 gate criteria met, all 5 knowledge checks answered. ADVANCING TO PHASE 2.
[ ] Criteria X, Y failed. Repeating weeks [N, M] before advancing.
```

**Deliverable:** `week_08/phase_gate.md` completed honestly.
