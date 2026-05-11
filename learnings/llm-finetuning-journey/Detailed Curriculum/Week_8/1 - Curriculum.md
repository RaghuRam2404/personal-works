# Week 8 — Phase 1 Gate: Capstone Mini-Project

## Learning Objectives

By the end of this week, you will be able to:

- Synthesize all Phase 1 skills into a single end-to-end project with clean, documented code.
- Train either a char-level transformer (nanoGPT-style) or fine-tune distilgpt2 on Spider SQL data using the full modern training recipe.
- Write a project README that clearly explains dataset, model, training setup, results, and lessons learned.
- Self-assess honestly against the Phase Gate criteria and identify your weakest areas.
- Articulate what PyTorch fluency means and demonstrate it by writing a training loop from memory.

---

## Concepts

This week is integration, not new material. You are proving that Weeks 1–7 are internalized.

### What Phase 1 Was About

You entered Phase 1 having coded 1–2 layer NNs in plain Python. You needed to build the "muscle memory" that lets you:
1. Write a training loop from scratch in under 10 minutes.
2. Read a loss curve and diagnose the problem.
3. Know what initialization to use, what optimizer, what schedule.
4. Understand tokenization well enough to not make data pipeline errors.
5. Use the HuggingFace Hub without mystery.

These skills are prerequisites for Phase 2 (building transformers) and non-negotiable for Phase 4 (fine-tuning).

### The Two Capstone Options

**Option A (Recommended): Char-level transformer on Spider SQL**

Use the [nanoGPT](https://github.com/karpathy/nanoGPT) architecture (you will understand what it means in Phase 2 — for now, copy the architecture and write your own data loader + training loop). Train on the Spider SQL query corpus. Generate SQL samples.

Why this is recommended:
- Forces you to read code you don't fully understand yet (the transformer architecture) — a preview of Phase 2.
- The SQL domain connection is direct.
- You will revisit this codebase in Weeks 9–11.

**Option B: distilgpt2 fine-tuning on Spider**

Use HuggingFace `Trainer` (or your own loop) to fine-tune distilgpt2 for 100–200 steps on Spider's SQL queries. The goal is to feel the full HuggingFace fine-tuning workflow before you apply it to a 7B model in Phase 4.

### What Makes a Good Capstone

A passing capstone has:
1. **Clean code** — well-commented, no dead code, organized into files.
2. **A working training loop** — uses AdamW, warmup + cosine decay, gradient clipping.
3. **W&B logging** — train loss, val loss per step.
4. **Generated samples** — at least 10 SQL samples from the trained model.
5. **A README** — dataset, model, training config, results table, what worked, what surprised you.

A failing capstone is one that copies code from nanoGPT or HuggingFace tutorials without understanding it, has no W&B logs, or produces no generated samples.

### Phase Gate Criteria

You must be able to confirm ALL of these before advancing to Phase 2:

- [ ] **Training loop from memory:** Set a 10-minute timer. Close all tabs. Write a complete PyTorch training loop (model, optimizer, loss, backward, step) with no references. Include: `zero_grad`, `autocast`, `clip_grad_norm_`, `scheduler.step()`. If you cannot do this, repeat Week 1.
- [ ] **Loss curve diagnosis:** Given any loss curve (W&B screenshot), you can identify: overfitting, underfitting, LR too high, LR too low. If you cannot do this, repeat Week 5.
- [ ] **Backprop explanation:** Explain backpropagation through a 2-layer network on paper, by hand. No code. Include: forward pass, loss computation, chain rule at each layer, gradient accumulation. If you cannot do this, repeat Week 1.
- [ ] **GitHub commits:** Every week (1–7) has at least one commit with working code. No exceptions.
- [ ] **HuggingFace artifact:** At least one dataset pushed to your HuggingFace account. Verify the URL is accessible.

### Self-Assessment Protocol

Before advancing, answer these questions honestly:

1. Can you write the LSTM gating equations from memory? (Week 4)
2. Can you compute a conv layer output shape in your head? (Week 3)
3. Can you explain why AdamW differs from Adam for weight decay? (Week 5)
4. Can you explain why BPE starts from bytes rather than characters? (Week 6)
5. What is the `-100` convention in HuggingFace labels and why does it exist? (Week 7)

If you cannot answer more than 2 of these without looking at notes, revisit the relevant weeks before advancing.

### On "Not Fully Understanding" nanoGPT

Option A requires you to use transformer code you don't yet understand. This is intentional. The nanoGPT architecture contains multi-head self-attention, layer norm, positional embeddings, and residual connections — all concepts Phase 2 will teach formally.

Your task is not to understand the transformer. Your task is to:
1. Read the code enough to trace the data flow from input tokens to output logits.
2. Write your own data loader and training loop around it.
3. Train it and generate samples.
4. Note 5 things you do not understand — these are Phase 2 previews.

Curiosity, not completeness, is the right mindset for this week.

---

## Connections

**Builds on:** Everything from Weeks 1–7. This is a synthesis week.

**Unlocks:** Phase 2 — if you pass the gate, you have the foundation to learn transformers from scratch. If you don't, the cost of advancing while weak is compounding — every Phase 2 exercise assumes Phase 1 fluency.

---

## Common Misconceptions and Pitfalls

- **"I'll just use the nanoGPT training loop as-is."** You must write your own. The exercise value is in the writing, not the running.
- **"A README is optional documentation."** Wrong. The README is part of the deliverable and demonstrates your ability to communicate technical work — a skill as important as the code.
- **"I'll just do Option B because it's easier."** Option B is fine. But if you choose it, make sure you understand every line of the `Trainer` call or replace it with your own loop.
- **"I passed the gate if the code runs."** The gate is about understanding, not execution. You pass if you can explain the code you wrote.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Choose Option A or B, plan your approach | 30 min |
| Set up project structure and data pipeline | 1 h |
| Implement model (copy nanoGPT arch for A; configure Trainer for B) | 1 h |
| Write training loop with full modern recipe | 1.5 h |
| Train model, monitor W&B, generate samples | 1.5 h |
| Write README + journal self-assessment | 1 h |
| Self-test: write training loop from memory (timed) | 30 min |
