# Week 28 Quiz Answers

## Q1 — Answer: C

**Answer:** C. SFT on 8K pairs directly.

**Why:** `Qwen2.5-Coder-7B` was pretrained on large-scale code corpora including SQL. The domain vocabulary is already present. Continued pretraining on 50GB of PostgreSQL documentation would consume most of your $50 budget and provide marginal benefit since the model already understands SQL syntax and PostgreSQL semantics. The 8K labeled pairs give you direct supervised signal for the exact task (schema-conditioned SQL generation). Start with SFT; only add continued pretraining if you find the model systematically fails on PostgreSQL-specific constructs (e.g., `WITH RECURSIVE`, `generate_series`, TimescaleDB functions) after SFT.

**Why others are wrong:**
- A: Continued pretraining alone with no task-specific SFT produces no task-aligned behavior — the model would still complete documents, not answer text-to-SQL queries.
- B: Not wrong in principle, but a waste of compute given the model already knows the domain. Save your budget for more SFT data or longer training.
- D: DPO requires (prompt, chosen, rejected) triples — your labeled pairs are not in that format and DPO cannot replace SFT as a first stage.

---

## Q2 — Answer: C

**Answer:** C. Only the response tokens.

**Why:** In SFT, the training signal comes from predicting the output (response) tokens given the input (prompt). The prompt tokens are passed through the model in the forward pass (they attend to each other and are used as context) but their positions are masked out of the loss computation. This is called "input masking" or "label masking." If you included the prompt in the loss, you would be penalizing the model for not predicting the prompt — which is both unnecessary and potentially harmful.

**Why others are wrong:**
- A: Including all tokens conflates the task with language modeling on prompt text.
- B: Computing loss only on prompts would teach the model to predict questions, not answers.
- D: Random sampling is never used in standard SFT.

---

## Q3 — Answer: C

**Answer:** C. Incorrect — SFT shapes output format and activates existing knowledge.

**Why:** Fine-tuning updates weights via gradient descent on the supervised loss, but the gradient signal for new facts is extremely weak unless those facts appeared many times in pretraining. The model's factual associations are encoded in its weights during pretraining over hundreds of billions of tokens. SFT on thousands of examples cannot reliably overwrite or add factual associations — it changes which knowledge the model retrieves given certain input formats, not what knowledge it has. Studies show that models frequently hallucinate when fine-tuned on facts absent from pretraining.

**Why others are wrong:**
- A: While all weights technically update, the effective change is controlled by the loss signal and learning rate. New factual knowledge requires the signal to appear thousands of times.
- B: Partially accurate but understates the problem — even vocabulary shift is limited.
- D: Incorrect — SFT does change output style and formatting behavior significantly.

---

## Q4 — Answer: B

**Answer:** B. Weight changes during fine-tuning lie in a low-dimensional subspace.

**Why:** Hu et al. (LoRA paper, 2021) empirically demonstrated that when you fine-tune large models, the matrix delta W = W_finetuned - W_pretrained has low intrinsic rank — meaning its information content can be captured by a rank-r matrix where r << min(d_in, d_out). This is what makes LoRA possible: instead of storing full delta W, you store two small matrices A and B where delta W ≈ BA. The hypothesis explains why fine-tuning generalizes despite overfitting the parameter count to the task.

---

## Q5 — Answer: C

**Answer:** C. GRPO.

**Why:** GRPO (Group Relative Policy Optimization) is designed for verifiable scalar rewards. For SQL, the reward is objective: run the generated SQL against a Postgres database, compare output rows to expected rows. This binary/scalar signal is exactly what GRPO optimizes. DPO requires human (or model) preference rankings between pairs, which is harder to scale for SQL. SFT trains on correct examples but cannot explicitly penalize incorrect SQL that "looks right" syntactically.

---

## Q6 — Short Answer

The team can skip continued pretraining if: the base model already performs reasonably on their domain vocabulary — specifically, if it correctly tokenizes support ticket terminology, product names, and internal jargon without splitting them into many unusual subwords, and if a preliminary eval shows the model understands the domain's semantic content.

They should add continued pretraining first if: the 500GB ticket corpus contains heavy proprietary jargon, product names, or domain abbreviations that the base model consistently misunderstands or hallucinates around. A quick test: run the base model on 10 representative tickets and check coherence.

---

## Q7 — Short Answer

Pretraining on 300B tokens teaches the model language, reasoning patterns, and world knowledge. SFT on 13K examples does not teach new knowledge — it teaches the model to activate and format its existing knowledge in response to instruction-style prompts. The data efficiency is high because the "work" of understanding language was done in pretraining. SFT is behavioral calibration, not knowledge injection. This is why the ratio of fine-tuning to pretraining data can be 1:10,000,000 and still produce dramatic behavioral shifts.

---

## Q8 — Short Answer

**Risk:** If the SFT loss is computed too aggressively (high learning rate, many epochs, small dataset relative to model size), gradient updates overwrite pretrained weight subspaces that encode general-purpose capabilities, not just the target task.

**Symptom:** Your text-to-SQL model, after fine-tuning, refuses or fails to answer general Python or SQL syntax questions it answered correctly before — even unrelated to the specific training examples. It might also generate repetitive or degenerate outputs on out-of-distribution prompts.

**Mitigation:** Use LoRA (Weeks 30–31) to constrain updates to a low-rank subspace, reducing the surface area of pretrained weights that can be overwritten. With full SFT, lower the learning rate (1e-5 instead of 2e-4) and use early stopping based on held-out validation loss.

---

## Q9 — Scenario Answer

**Diagnosis, ranked by likelihood:**

1. **Overfitting due to too many epochs.** 3 epochs on 5K examples is often too many for a 7B model. The model has memorized training examples rather than generalizing. Fix: use 1–2 epochs, add held-out validation set, stop when val loss starts rising.

2. **Learning rate too high, causing catastrophic forgetting.** The model's pretrained general capabilities are being overwritten. Fix: reduce LR from any value above 1e-4 to 2e-5 or lower for full SFT. Or switch to LoRA, which is much more resistant to forgetting.

3. **No input masking.** If the prompt tokens are included in the loss, the model is being trained to predict both questions and answers, degrading generation quality. Fix: verify your `SFTTrainer` is masking input tokens (this is the default, but check `dataset_text_field` vs manual collation).

4. **Insufficient data diversity.** 5K examples may not cover enough variation in schema structures, question phrasings, or SQL patterns. Fix: add data augmentation or synthetic generation to reach 15K examples before the next training run.

5. **Training without regularization.** No dropout, no weight decay, no LoRA constraint. Fix: add `weight_decay=0.01` in AdamW, or switch to LoRA (Week 30+).
