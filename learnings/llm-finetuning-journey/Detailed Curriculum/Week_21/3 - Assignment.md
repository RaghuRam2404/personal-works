# Week 21 Assignment — Run the 50M Pretraining Job

## Pre-Launch Checklist

- [ ] `train.bin` contains at least 500M tokens (1B preferred)
- [ ] 200-step sanity check passes: initial loss ~10.4, loss after 200 steps < 7.0
- [ ] W&B project `week-21-50m-pretrain` created and verified (run 1 step, check W&B receives the data)
- [ ] Checkpoint directory exists and write permissions verified
- [ ] Accelerate config set to bf16 mixed precision on single GPU
- [ ] `torch.compile(model)` is applied if using PyTorch 2.0+ (adds ~5 min compile time, then faster)
- [ ] `train.py` has `--resume` argument that loads the latest checkpoint in `checkpoints/`

---

## Task 1 — Launch the Full Training Run

**Goal:** Train the 56M GPT model for ~2B tokens, logging everything to W&B.

**Requirements:**

Run with these exact hyperparameters (justifications in Curriculum.md):

```bash
python train.py \
  --batch_size 16 \
  --gradient_accumulation 4 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_steps 200 \
  --max_steps 30500 \
  --weight_decay 0.1 \
  --grad_clip 1.0 \
  --mixed_precision bf16 \
  --log_interval 10 \
  --eval_interval 500 \
  --save_interval 2000 \
  --wandb_project week-21-50m-pretrain \
  --data_path data/train.bin \
  --val_path data/val.bin
```

**Required W&B metrics (log all of these):**
- `train/loss` (every 10 steps)
- `train/lr` (every 10 steps)
- `train/grad_norm` (every 10 steps)
- `train/tokens_per_sec` (every 10 steps)
- `train/tokens_seen` (every 10 steps, cumulative)
- `val/loss` (every 500 steps)

**Deliverable:** A W&B run URL where all metrics are visible publicly (set run visibility to public). Paste this URL into `week-21-training-log.md`.

**Acceptance criteria:**
- Training completes at least 15,000 steps (targeting 30,500)
- Final val loss < 4.0 (good run: < 3.5)
- No NaN losses in the run
- All 6 metrics are visible in W&B

---

## Task 2 — Monitor and Document Training

**Goal:** Develop the habit of reading training curves critically.

**Requirements:**

Create `week-21-training-log.md` with:

1. **Hourly snapshots** (at hours 1, 2, 4, 8, 11): for each, record:
   - Current step and tokens seen
   - Train loss and val loss (if eval ran)
   - Tokens/sec throughput
   - Any anomalies observed

2. **Loss spike report** (if any spikes occurred): step number, magnitude of spike, recovery behavior, your hypothesis for cause

3. **Final statistics** at run completion:
   - Total steps completed
   - Total tokens seen
   - Best val loss achieved (step number)
   - Total wall-clock time
   - Average tokens/sec for the full run

---

## Task 3 — Checkpoint and HuggingFace Upload

**Goal:** Preserve your trained model in a reusable format.

**Requirements:**
- Convert your PyTorch checkpoint to HuggingFace format
- Push to `<your-handle>/fineweb-50m-pretrain` on HuggingFace Hub
- Include a `README.md` (model card) documenting: architecture, training data, hyperparameters, val perplexity
- The model should be loadable with `AutoModelForCausalLM.from_pretrained()`

For conversion, either:
- Implement `save_pretrained()` using `GPT2Config` (your arch is GPT-2-like)
- Or save your checkpoint and document the loading procedure in the model card

**Deliverable:** HuggingFace Hub URL in `week-21-training-log.md`.

GitHub commit: `week-21-50m-pretrain`

---

## Task 4 — Debugging Journal

**Goal:** Document whatever went wrong and how you fixed it.

**Requirements:**

Write a minimum 200-word section in `week-21-training-log.md` titled "What Went Wrong and How I Fixed It." Even if the run was smooth, document:
- Any Colab disconnections and how you resumed
- Any warnings in the training output
- Configuration decisions you changed during setup (and why)

If something genuinely broke: describe the error, your hypothesis, what you tried, and what fixed it.

---

## Stretch Goals

- Run `torch.compile(model, mode="reduce-overhead")` and measure the throughput improvement vs. eager mode
- Implement learning rate finder (run for 100 steps with exponentially increasing LR, plot loss vs. LR)
- Add a text generation callback: every 2000 steps, generate 5 random completions of "The history of" and log to W&B as `train/generated_text`
