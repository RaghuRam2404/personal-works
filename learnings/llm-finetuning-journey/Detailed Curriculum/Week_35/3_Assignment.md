# Week 35 Assignment — Hyperparameter Sweep for SFT/LoRA

## Setup Checklist

- [ ] Colab Pro (A100 or T4)
- [ ] 1K training examples and 200 eval examples (subset of your domain dataset)
- [ ] W&B project `week-35-hp-sweep` created
- [ ] Raschka's article read: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- [ ] Unsloth installed (from Week 34)

---

## Task 1 — Design the Sweep

**Goal:** Define a principled hyperparameter sweep before running it.

**Requirements:**
- In `week35_sweep_design.md`, document:
  - Which hyperparameters you will sweep and why (prioritized by expected impact)
  - The values you will test for each hyperparameter
  - The metric you will minimize (eval/loss)
  - The dataset size used for the sweep (1K train, 200 eval) and why this is sufficient for relative comparisons
  - Expected number of runs and approximate total compute time
  - What you will keep fixed and why (target_modules, optimizer, scheduler type)

**Deliverable:** `week35_sweep_design.md` committed.

---

## Task 2 — Run W&B Sweep

**Goal:** Execute a systematic hyperparameter search.

**Requirements:**
- Sweep over: LR ∈ {1e-5, 5e-5, 2e-4}, rank ∈ {16, 32}, alpha ∈ {rank, 2×rank}
- All other settings fixed: Unsloth, Qwen2.5-Coder-7B (or 1.5B for speed), 2 epochs, packing=True, cosine scheduler, warmup_ratio=0.1, batch 4, grad_accum 4
- Each run logs to W&B project `week-35-hp-sweep` with a descriptive run name (e.g., `lr-2e4_r16_a32`)
- Must run at least 8 of the 12 combinations

**W&B sweep config:**
```python
sweep_config = {
    "method": "grid",
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 5e-5, 2e-4]},
        "lora_rank": {"values": [16, 32]},
        "alpha_mult": {"values": [1, 2]},  # alpha = rank × alpha_mult
    }
}
```

**Deliverable:** W&B sweep URL in `week35_results.md`. At least 8 runs visible.

**Hints:**
- If A100 is not available, use 1.5B model instead of 7B to make each run shorter
- On T4, each 1K-example run at rank 16 should take ~3–5 minutes

---

## Task 3 — Analyze Results

**Goal:** Extract actionable conclusions from the sweep.

**Requirements:**
- Screenshot or export the W&B parallel coordinates plot to `week35_sweep_plot.png`
- In `week35_results.md`, write a table showing all runs with: LR, rank, alpha, final eval loss
- Answer these questions in `week35_results.md`:
  1. Which LR gave the best eval loss? Was the winner consistent across different ranks?
  2. Did higher rank (32) outperform lower rank (16) on the 1K dataset?
  3. What happened to eval loss at LR=1e-5? What does this tell you?
  4. What was the worst-performing configuration, and why?

**Deliverable:** `week35_results.md` with analysis, `week35_sweep_plot.png` committed.

---

## Task 4 — Articulate Each Hyperparameter's Effect

**Goal:** Verify you understand what each hyperparameter does, not just which values win.

**Requirements:**
- In `week35_hp_explanations.md`, write 3–5 sentences each for:
  - **Learning rate:** What does it control in gradient descent? Why is 2e-4 typically good for LoRA but bad for full SFT?
  - **LoRA rank:** What does rank control? Why is rank not always "more is better"?
  - **Effective batch size:** How does gradient accumulation create a larger effective batch? What is the trade-off?
  - **Epochs:** Why is fewer epochs often better than more for small datasets?
  - **LR warmup:** What problem does warmup solve? What is a good warmup_ratio for 2-epoch training?

**Deliverable:** `week35_hp_explanations.md` committed.

---

## Task 5 — Final Configuration Recommendation

**Goal:** Choose hyperparameters for the Week 38 15K sprint.

**Requirements:**
- Based on your sweep results, write a recommendation in `week35_results.md` for Week 38:
  - Exact LR value
  - LoRA rank and alpha
  - Number of epochs (considering you will have 15K examples instead of 1K)
  - Batch size and gradient accumulation
  - Any other changes from your current default settings
- Justify each choice in 1–2 sentences

**Deliverable:** "Week 38 Recommended Hyperparameters" section in `week35_results.md`.

---

## Stretch Goals

- Run a sweep with `packing=False` vs `packing=True` — measure throughput difference and quality difference
- Add `neftune_noise_alpha` to your sweep: try values 0, 5, 15 — does NEFTune help SQL quality?
- Implement early stopping: add `load_best_model_at_end=True` and `early_stopping_patience=3` to `SFTConfig`
