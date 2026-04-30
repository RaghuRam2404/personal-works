# Week 17 Assignment — Scaling Laws in Practice

## Setup Checklist

- [ ] Papers downloaded and skimmed: Kaplan 2020, Chinchilla 2022
- [ ] Calculator or Python notebook ready (no GPU needed this week)
- [ ] GitHub repo with a `week-17-scaling-laws/` directory

---

## Task 1 — Reproduce the Chinchilla Formula

**Goal:** Derive N_opt and D_opt from first principles for several compute budgets, then verify against known models.

**Requirements:**
- Write a Python script `chinchilla_calculator.py` that takes a compute budget in FLOPs and returns recommended N_opt and D_opt
- Use the Chinchilla constants from the paper (Approach 3, Table A3): a = 6.8e-2, b = 1.96, exponent = 0.50
- Also compute the approximate A100 dollar cost given a user-specified dollar_per_hour and MFU
- Print a table for C = [1e18, 1e19, 1e20, 1e21, 1e22] FLOPs

**Deliverable:** `week-17-scaling-laws/chinchilla_calculator.py` committed to GitHub.

**Sample output your script should produce:**

```
C (FLOPs)    N_opt (M)   D_opt (B)   C = 6ND check
1.00e+18     11.3M       0.23B       1.54e+18 ✓
1.00e+19     35.7M       0.71B       1.52e+19 ✓
1.00e+20    113.1M       2.26B       1.53e+20 ✓
1.00e+21    357.8M       7.16B       1.53e+21 ✓
1.00e+22   1132.0M      22.64B       1.53e+22 ✓
```

**Hints:**
- The "6ND check" column verifies your output is self-consistent: `C_check = 6 × N_opt × D_opt` should be within 2× of your input C
- MFU (Model FLOP Utilization) is typically 30–45% for well-tuned single-GPU training. Use 0.35 as a default

---

## Task 2 — Analyze Historic Models Against Chinchilla Frontier

**Goal:** Determine which pre-Chinchilla models are compute-optimal, compute-inefficient, and over-trained.

**Requirements:**
- Create a Markdown table in `scaling_analysis.md` with the following models: GPT-3 (175B, 300B tokens), LLaMA-1-7B (7B, 1T tokens), Chinchilla-70B (70B, 1.4T tokens), Llama-3-8B (8B, 15T tokens)
- For each model, compute the Chinchilla-optimal token count given the actual parameter count (D_chinchilla = 20 × N)
- Label each model as: "under-trained" (actual D << D_chinchilla), "near-optimal", or "over-trained for inference" (actual D >> D_chinchilla)
- Add a 2-sentence commentary on each

**Deliverable:** `week-17-scaling-laws/scaling_analysis.md`

**Acceptance criteria:** Your table is factually correct and your commentaries demonstrate understanding of why over-training can be rational (inference cost).

---

## Task 3 — Your $50 Phase 6 Compute Budget Writeup

**Goal:** Apply Chinchilla to your actual project constraint.

**Requirements:**
- Assume your Phase 6 budget is $50 of GPU compute (A100 at $1.50/hr on RunPod)
- Calculate total FLOPs available at 35% MFU
- Apply Chinchilla to find N_opt and D_opt
- Then argue whether you should follow Chinchilla or train a smaller model on more tokens (inference-optimal reasoning)
- Minimum 400 words, written in first person, in `phase6_compute_plan.md`

**You must answer these specific questions in your writeup:**
1. What model size does Chinchilla recommend for $50?
2. Is that model size practical to fine-tune on a single A100 (memory-wise)?
3. Qwen2.5-Coder-7B was trained on 5.5T tokens — is it Chinchilla-optimal? Why might Alibaba have made that choice?
4. For your PostgreSQL/TimescaleDB text-to-SQL goal, would you fine-tune a Chinchilla-optimal model trained from scratch, or take a pre-trained SOTA model as your starting point? Why?

**Deliverable:** `week-17-scaling-laws/phase6_compute_plan.md`. GitHub commit `week-17-scaling-laws`.

---

## Stretch Goals

- Plot the Chinchilla Pareto frontier (L vs C) for your architecture class and mark where GPT-3, Chinchilla-70B, and Llama-3-8B sit
- Read [Scaling Laws for Fine-Tuning](https://arxiv.org/abs/2206.07660) (Zhao et al.) and note how fine-tuning scaling differs from pretraining scaling
- Compute the FLOP cost of a single forward pass through GPT-2-small (117M params, 1024 sequence length). Verify with `torch.profiler`
