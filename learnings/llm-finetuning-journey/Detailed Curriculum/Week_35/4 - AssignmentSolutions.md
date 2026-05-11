# Week 35 Assignment Solutions

## Task 2 — W&B Sweep: Key Snippet

```python
import wandb
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

def train_sweep():
    run = wandb.init()
    config = wandb.config
    rank = config.lora_rank
    alpha = rank * config.alpha_mult
    lr = config.learning_rate
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B", max_seq_length=512,
        dtype=None, load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=rank, lora_alpha=alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        args=SFTConfig(
            output_dir=f"./sweep-r{rank}-lr{lr:.0e}",
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            packing=True, max_seq_length=512,
            logging_steps=5,
            eval_steps=50,
            evaluation_strategy="steps",
            report_to="wandb",
            run_name=f"lr-{lr:.0e}_r{rank}_a{alpha}",
            dataset_text_field="text",
        ),
        train_dataset=train_ds, eval_dataset=eval_ds,
    )
    trainer.train()
    wandb.finish()

sweep_config = {
    "method": "grid",
    "metric": {"name": "eval/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 5e-5, 2e-4]},
        "lora_rank": {"values": [16, 32]},
        "alpha_mult": {"values": [1, 2]},
    }
}
sweep_id = wandb.sweep(sweep_config, project="week-35-hp-sweep")
wandb.agent(sweep_id, train_sweep, count=12)
```

---

## Expected Sweep Results (typical for 1K SQL examples, 2 epochs)

| LR | Rank | Alpha | Eval Loss |
|---|---|---|---|
| 1e-5 | 16 | 16 | 1.8–2.0 (too slow) |
| 5e-5 | 16 | 32 | 1.4–1.6 |
| 2e-4 | 16 | 32 | 1.1–1.4 (usually best) |
| 2e-4 | 32 | 64 | 1.1–1.5 (may overfit on 1K) |
| 1e-5 | 32 | 32 | 1.7–2.0 |
| 2e-4 | 16 | 16 | 1.1–1.5 |

General finding: LR=2e-4 consistently wins. Rank 16 usually matches rank 32 on 1K data. Alpha=2×rank (scaling=2) marginally better than alpha=rank in most settings.

---

## Task 5 — Recommended Config for Week 38

Based on typical sweep outcomes on 1K SQL subset:

```
LR: 2e-4
Rank: 16
Alpha: 32 (= 2 × rank)
Epochs: 2 (with early stopping)
Batch size: 4 (per device)
Gradient accumulation: 4 (effective batch 16)
Warmup ratio: 0.05 (5% of steps)
Scheduler: cosine
Packing: True
Max seq length: 512
```

Rationale for Week 38 changes vs. current: with 15K examples (3× more data), overfitting risk is lower — rank 16 is still appropriate. The same LR applies but you may wish to try 1e-4 as well. Run with early stopping enabled.

---

## Common Gotchas

- **Sweep produces runs with very different step counts**: If packing ratio varies between configurations, some runs have fewer steps per epoch. Normalize comparison by eval loss at the same wall-clock time, not step count.
- **Alpha sweep interfering with LR sweep**: alpha/rank scaling changes the effective learning rate. If you sweep both LR and alpha, always report `effective_lr = lr × (alpha/rank)` to make comparisons meaningful.
- **Eval set too small for sweep**: With 200 eval examples, eval loss has ±0.05–0.10 noise. Run each config twice and average if you have compute budget.

---

## How to Verify You Did It Right

- W&B sweep shows at least 8 runs with distinct hyperparameter combinations
- The run with LR=1e-5 has noticeably higher eval loss than LR=2e-4 runs (validates LR importance)
- Your Week 38 recommendation includes all 8 hyperparameters listed above with justification
- `week35_hp_explanations.md` has 3–5 sentences per hyperparameter, not just one line
