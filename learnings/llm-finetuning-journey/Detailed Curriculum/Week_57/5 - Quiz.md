# Week 57 Quiz — Continued Pretraining

## Multiple Choice

**Q1.** Which training objective is used for continued pretraining?

A) Masked language modeling (BERT-style: predict masked tokens).
B) Causal language modeling (GPT-style: predict next token given all previous tokens).
C) Instruction following (SFT): predict the response to an instruction prompt.
D) Contrastive learning: distinguish real documents from corrupted ones.

---

**Q2.** You run CPT for 3 epochs on your 100M-token corpus. After epoch 2, your general-text perplexity has increased by 1.2 bits. What does this indicate and what should you do?

A) The model is learning domain knowledge successfully — continue training.
B) Catastrophic forgetting is occurring — you should have stopped at epoch 1. Revert to the epoch-1 checkpoint.
C) The learning rate is too low — increase it and continue from the current checkpoint.
D) Your corpus has data contamination — deduplicate and restart.

---

**Q3.** You are packing a 100M-token corpus into sequences of length 2048. Without EOS tokens between documents, what problem arises during training?

A) The model runs out of VRAM because cross-document attention is computed.
B) The model learns to predict tokens from the next document as valid continuations of the current document.
C) Gradient accumulation is broken because gradients are mixed across documents.
D) The tokenizer generates incorrect IDs for cross-document boundary positions.

---

**Q4.** CPT improves your TimescaleDB eval performance by 4 percentage points. SFT on the v3 dataset without CPT improves it by 12 percentage points. What does this tell you about the CPT step?

A) CPT is not worth the cost — SFT dominates, so skip CPT.
B) CPT provides complementary gains; with CPT as initialization, SFT likely achieves > 12 pp improvement total.
C) CPT overfits the domain, making SFT less effective.
D) SFT should be run before CPT for maximum performance.

---

## Short Answer

**Q5.** Your 100M-token corpus is composed of: 60% Stack Overflow posts, 15% PostgreSQL docs, 15% GitHub SQL files, 10% TimescaleDB content. After CPT, you find the model has become very good at answering StackOverflow-style "how do I do X" questions but hasn't improved on complex time-series SQL. What is the likely cause and how would you fix it for a future CPT run?

---

**Q6.** Explain the difference between packing efficiency and data utilization in the context of CPT corpus preparation. Why does high packing efficiency matter for training speed on H100?

---

**Q7.** You want to verify that CPT has actually increased the model's domain knowledge without forgetting general knowledge. Design a before/after evaluation with 3 specific metrics and explain what each measures.

---

## Deep Scenario

**Q8.** You have finished CPT and are about to start SFT in Week 58. You compare three initialization options:
- Option A: Start SFT from the original Qwen2.5-Coder-7B base model
- Option B: Start SFT from your CPT checkpoint  
- Option C: Start SFT from your Phase 5 GRPO checkpoint (which already has good SQL capability)

Design an experiment to determine which initialization is best. What metrics would you measure, what training configurations would you use, and how would you make the final decision?
