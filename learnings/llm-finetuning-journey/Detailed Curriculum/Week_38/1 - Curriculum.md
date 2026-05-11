# Week 38 — Domain-Tuning Sprint Week 2: QLoRA Fine-Tune on 15K

## Learning Objectives

By the end of this week, you will be able to:

- Execute a full production-quality QLoRA fine-tuning run of Qwen2.5-Coder-7B on 15K SQL examples
- Apply best-practice hyperparameters from your Week 35 sweep and adapter choice from Week 36
- Monitor training with W&B and make real-time decisions if issues arise
- Compare fine-tuned model quality against the base model using your held-out test set
- Push the production adapter to HuggingFace Hub with a proper model card

---

## Concepts

### 1. This Is the Main Event

Weeks 29–37 were practice runs and preparation. Week 38 is where everything comes together:

- **Infrastructure:** Unsloth on Colab Pro A100 (Week 34)
- **Adapter:** DoRA or LoRA at rank 16, alpha 32 (Week 36 decision)
- **Dataset:** 15K curated PostgreSQL SQL examples (Week 37)
- **Hyperparameters:** LR=2e-4, cosine scheduler, warmup=0.05, packing=True, max_seq_length=512 (Week 35)
- **Evaluation:** 100-example held-out test set (Week 32)

You are building `postgres-sqlcoder-7b-v1`.

### 2. Training Duration and Cost Estimate

With Unsloth on Colab Pro A100:
- 15K examples, packing ratio ~3 → ~5K packed sequences
- At effective batch size 16 (4 × 4): ~312 steps per epoch
- 2 epochs: ~625 optimizer steps
- At 3 steps/second: ~208 seconds ≈ 3.5 minutes of pure training
- With overhead (data loading, eval, checkpointing): ~15–25 minutes total

Cost: ~$0.30–0.50 at $1.20/hr A100 rate. Your budget allows multiple iterations.

### 3. Monitoring During Training

Watch these W&B metrics in real time:

**Healthy training signs:**
- Train loss decreases monotonically (or with small oscillations) for the first epoch
- Eval loss decreases until it plateaus near the best val loss from your 5K run
- Steps/second stable at 2–4 on A100 (with Unsloth)
- GPU memory stable (no memory leak — not growing step by step)

**Warning signs:**
- Train loss spike to 5+ at any point → gradient explosion (add `max_grad_norm=1.0`)
- Eval loss rising by step 300 → learning too fast, overfitting on the 15K dataset at 2 epochs? unlikely, but possible — check
- Steps/second dropping → memory pressure, reduce batch size
- Loss plateau at 1.5+ → dataset quality issue or LR too low

### 4. Mid-Training Decisions

If training is going wrong:
1. If loss spike: stop, add `max_grad_norm=1.0` to SFTConfig, restart
2. If eval loss rises early: reduce epochs to 1, or enable early stopping
3. If OOM: reduce `per_device_train_batch_size` from 4 to 2

If training is going right:
- Let it run to completion — don't restart unless clearly broken

### 5. Post-Training: Quality Evaluation

After training, run evaluation on your 100-example held-out test set. Compare to:

1. **Qwen2.5-Coder-7B base (no fine-tuning):** Run the same prompts through the base model. Expected: 5–15% exact match, 60–80% valid SQL.
2. **Your Week 33 model (5K training):** Expected: 30–50% exact match.
3. **Your Week 38 model (15K training):** Target: >50% exact match, significant improvement over Week 33.

Qualitative evaluation: for 5–10 examples where the model produces incorrect SQL, read the output and categorize the error:
- Wrong table referenced
- Wrong column referenced
- Wrong JOIN type
- Wrong aggregation
- Correct SQL but slightly different from expected (which would be counted correct on execution eval)

### 6. The Model Card

A production model needs a thorough model card. At minimum:

```markdown
# postgres-sqlcoder-7b-v1

## Model Description
Qwen2.5-Coder-7B fine-tuned on 15K PostgreSQL text-to-SQL examples using QLoRA (rank 16, DoRA).

## Training Details
- Base model: Qwen/Qwen2.5-Coder-7B
- Fine-tuning method: QLoRA (NF4 + DoRA, rank 16, alpha 32)
- Training data: 15K PostgreSQL SQL pairs (sql-create-context + synthetic + TimescaleDB)
- Hardware: Colab Pro A100 40GB
- Training time: ~20 minutes, 2 epochs
- Framework: Unsloth + TRL SFTTrainer

## Performance
- Held-out exact match: X% (100 examples)
- Base model exact match: Y%
- Valid SQL %: Z%

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# Load base model and adapter...
```

## Limitations
...
```

### 7. Publishing and Sharing

Push adapter only (not the full merged model) to save HuggingFace storage:

```python
model.push_to_hub("<your-handle>/postgres-sqlcoder-7b-v1")
tokenizer.push_to_hub("<your-handle>/postgres-sqlcoder-7b-v1")
```

Tag the model with `text2sql`, `postgresql`, `sql`, `qwen2.5`, `lora` on HuggingFace for discoverability.

---

## Connections

**Builds on:** Literally every week of Phase 4. This is the culmination of Weeks 29–37.

**Needed for:** Week 39 (you will evaluate this model with execution-based eval). Phase 5 starts from this model checkpoint.

---

## Common Misconceptions / Pitfalls

- **"More epochs = better on 15K."** Not necessarily. 2 epochs is the default; only add a third if eval loss is still significantly improving at epoch 2.
- **"I should train until loss reaches 0."** Never. Monitor eval loss. Train loss below 0.5 with eval loss above 1.0 = overfitting.
- **"I need to wait for perfect results before pushing."** Push after training completes. Week 39 will diagnose remaining failure modes.
- **"The exact match % is the final metric."** No — exact match is conservative. Week 39's execution-based eval will give a more accurate picture.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Verify dataset from Week 37 is correctly formatted | 30m |
| Set up and run the training script on A100 | 1h |
| Monitor training in real time via W&B | 1h |
| Run held-out evaluation (base vs. fine-tuned comparison) | 1.5h |
| Write model card and push to HuggingFace | 1h |
| Commit everything to GitHub with commit: `week-38-qlora-15k` | 30m |
| Write `week38_results.md` with findings | 1h |
